"""
Dual Glove Receiver Module
Listens for Wi-Fi UDP (and Bluetooth/Serial) telemetry from both Left and Right ESP32 gloves,
time-synchronizes readings, applies calibration scaling, and generates 28-dimensional feature vectors.
"""
import socket
import json
import time
import threading
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GloveReceiver")

class DualGloveReceiver:
    def __init__(self, port: int = config.DEFAULT_UDP_PORT):
        self.port = port
        self.running = False
        self.socket: Optional[socket.socket] = None
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        # Calibration data (min/max raw ADC per finger for normalization: 0=flat, 1=bent)
        self.calibration = self.load_default_calibration()

        # Telemetry State per hand
        self.left_state: Dict[str, Any] = self._empty_hand_state("LEFT")
        self.right_state: Dict[str, Any] = self._empty_hand_state("RIGHT")

        # Packet counters for FPS calculation
        self.left_packet_count = 0
        self.right_packet_count = 0
        self.left_fps = 0.0
        self.right_fps = 0.0
        self.last_fps_calc = time.time()

        # Simulation Mode (for testing without physical hardware)
        self.simulation_mode = False
        self.sim_thread: Optional[threading.Thread] = None

        # Track Ping RTT per hand
        self.ping_results = {
            "LEFT": {"success": False, "rtt_ms": 0.0, "last_ping": 0.0, "status": "Idle"},
            "RIGHT": {"success": False, "rtt_ms": 0.0, "last_ping": 0.0, "status": "Idle"}
        }

    def _empty_hand_state(self, hand: str) -> Dict[str, Any]:
        return {
            "hand": hand,
            "connected": False,
            "last_seen": 0.0,
            "seq": 0,
            "flex_raw": [2800, 2800, 2800, 2800, 2800],
            "flex_norm": [0.0, 0.0, 0.0, 0.0, 0.0],  # Neutral/flat rest position (0.0), NOT 0.5 (C handshape)
            "accel": [0.0, 0.0, 9.81],
            "gyro": [0.0, 0.0, 0.0],
            "mag": [0.0, 0.0, 0.0],
            "ip_addr": None
        }

    def load_default_calibration(self) -> Dict[str, Any]:
        """Load saved calibration or fallback to defaults."""
        if config.CALIBRATION_JSON_PATH.exists():
            try:
                with open(config.CALIBRATION_JSON_PATH, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load calibration file: {e}")

        # Default: ADC raw ~2800 flat, ~1200 bent
        return {
            "LEFT": {
                "flex_flat": [2800, 2800, 2800, 2800, 2800],
                "flex_bent": [1200, 1200, 1200, 1200, 1200]
            },
            "RIGHT": {
                "flex_flat": [2800, 2800, 2800, 2800, 2800],
                "flex_bent": [1200, 1200, 1200, 1200, 1200]
            }
        }

    def save_calibration(self, new_cal: Dict[str, Any]):
        """Persist calibration parameters."""
        self.calibration = new_cal
        try:
            with open(config.CALIBRATION_JSON_PATH, "w") as f:
                json.dump(new_cal, f, indent=2)
            logger.info("Saved calibration profile.")
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")

    def normalize_flex(self, hand: str, raw_flex: list) -> list:
        """Map raw ADC counts to [0.0 (flat), 1.0 (fully bent)]."""
        cal = self.calibration.get(hand, {})
        flat = cal.get("flex_flat", [2800] * 5)
        bent = cal.get("flex_bent", [1200] * 5)

        norm = []
        for i in range(5):
            f_val = flat[i]
            b_val = bent[i]
            diff = f_val - b_val
            if abs(diff) < 50:
                diff = 1000  # Avoid division by zero
            # In our voltage divider, higher resistance (bent) = lower voltage = lower ADC
            ratio = (f_val - raw_flex[i]) / float(diff)
            norm.append(float(np.clip(ratio, 0.0, 1.0)))
        return norm

    def start(self):
        """Start UDP listener socket thread."""
        if self.running:
            return

        self.running = True
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((config.DEFAULT_UDP_HOST, self.port))
            self.socket.settimeout(0.5)
            logger.info(f"UDP Receiver listening on 0.0.0.0:{self.port}")
        except Exception as e:
            logger.error(f"Failed to bind UDP socket on port {self.port}: {e}")
            self.running = False
            return

        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop receiver thread and release socket."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
        self.stop_simulation()
        logger.info("UDP Receiver stopped.")

    def _listen_loop(self):
        while self.running:
            try:
                data, addr = self.socket.recvfrom(config.UDP_BUFFER_SIZE)
                text = data.decode("utf-8", errors="ignore").strip()
                self._process_raw_packet(text, addr[0])
            except socket.timeout:
                pass
            except Exception as e:
                if self.running:
                    logger.debug(f"Packet read error: {e}")

            self._update_connection_status()

    def _process_raw_packet(self, text: str, ip_addr: str):
        # Handle cases where multiple JSON objects arrive concatenated
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line.startswith("{") or not line.endswith("}"):
                continue
            try:
                packet = json.loads(line)
                hand = packet.get("hand", "").upper()
                if hand not in ["LEFT", "RIGHT"]:
                    continue

                # Check for PONG ping reply
                if packet.get("pong") or packet.get("cmd") == "PONG":
                    with self.lock:
                        sent_ts = packet.get("echo_ts", 0.0)
                        rtt = (time.time() - sent_ts) * 1000.0 if sent_ts > 0 else 5.0
                        self.ping_results[hand]["success"] = True
                        self.ping_results[hand]["rtt_ms"] = max(1.0, rtt)
                        self.ping_results[hand]["last_ping"] = time.time()
                        self.ping_results[hand]["status"] = f"Connected ({rtt:.1f}ms)"
                    continue

                raw_flex = packet.get("flex", [2800]*5)
                accel = packet.get("accel", [0, 0, 0])
                gyro = packet.get("gyro", [0, 0, 0])
                mag = packet.get("mag", [0, 0, 0])
                seq = packet.get("seq", 0)

                # Normalize flex values
                norm_flex = self.normalize_flex(hand, raw_flex)

                with self.lock:
                    target_state = self.left_state if hand == "LEFT" else self.right_state
                    target_state["connected"] = True
                    target_state["last_seen"] = time.time()
                    target_state["seq"] = seq
                    target_state["flex_raw"] = raw_flex
                    target_state["flex_norm"] = norm_flex
                    target_state["accel"] = accel
                    target_state["gyro"] = gyro
                    target_state["mag"] = mag
                    target_state["ip_addr"] = ip_addr

                    if hand == "LEFT":
                        self.left_packet_count += 1
                    else:
                        self.right_packet_count += 1

            except Exception as e:
                logger.debug(f"JSON decode failure: {e}")

    def send_udp_cmd(self, ip: str, port: int, payload: dict):
        """Send a JSON command packet to a glove IP."""
        if not ip:
            return
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            msg = json.dumps(payload).encode("utf-8") + b"\n"
            sock.sendto(msg, (ip, port))
            sock.close()
        except Exception as e:
            logger.error(f"Failed to send UDP command to {ip}:{port}: {e}")

    def ping_glove(self, hand: str, ip: str, port: int = config.DEFAULT_UDP_PORT) -> Tuple[bool, str]:
        """Send a Ping command to the glove and check network reachability."""
        hand = hand.upper()
        if not ip or not ip.strip():
            return False, "IP address is empty"

        ip_clean = ip.strip()
        now = time.time()
        with self.lock:
            self.ping_results[hand]["status"] = "Pinging..."

        payload = {"cmd": "PING", "hand": hand, "ts": now}
        self.send_udp_cmd(ip_clean, port, payload)

        # Wait briefly for response
        time.sleep(0.15)

        with self.lock:
            # Check if recently received packets from this IP or got a PONG
            target_state = self.left_state if hand == "LEFT" else self.right_state
            if target_state["connected"] and target_state.get("ip_addr") == ip_clean:
                self.ping_results[hand]["success"] = True
                self.ping_results[hand]["status"] = f"Online ({self.left_fps if hand=='LEFT' else self.right_fps:.0f} Hz)"
                return True, self.ping_results[hand]["status"]
            elif self.ping_results[hand]["success"] and (time.time() - self.ping_results[hand]["last_ping"] < 2.0):
                return True, f"Online ({self.ping_results[hand]['rtt_ms']:.1f}ms)"
            else:
                self.ping_results[hand]["success"] = False
                self.ping_results[hand]["status"] = "No Response"
                return False, "No Response"

    def blink_glove_led(self, hand: str, ip: str, port: int = config.DEFAULT_UDP_PORT):
        """Send a BLINK command to trigger physical ESP32 onboard LED flashing."""
        if not ip or not ip.strip():
            return
        payload = {"cmd": "BLINK", "hand": hand.upper()}
        self.send_udp_cmd(ip.strip(), port, payload)
        logger.info(f"Sent BLINK LED command to {hand} glove at {ip}:{port}")


    def _update_connection_status(self):
        now = time.time()
        # FPS Calculation
        dt = now - self.last_fps_calc
        if dt >= 1.0:
            with self.lock:
                self.left_fps = self.left_packet_count / dt
                self.right_fps = self.right_packet_count / dt
                self.left_packet_count = 0
                self.right_packet_count = 0
                self.last_fps_calc = now

                # Timeout disconnection check
                if now - self.left_state["last_seen"] > config.TELEMETRY_TIMEOUT_SEC:
                    self.left_state["connected"] = False
                if now - self.right_state["last_seen"] > config.TELEMETRY_TIMEOUT_SEC:
                    self.right_state["connected"] = False

    def get_latest_states(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return thread-safe snapshot of both hand telemetry states."""
        with self.lock:
            return dict(self.left_state), dict(self.right_state)

    def get_synchronized_feature_vector(self) -> np.ndarray:
        """
        Produce synchronized 28-dimensional feature vector.
        Features: [L_Flex(5), L_Accel(3), L_Gyro(3), L_Mag(3), R_Flex(5), R_Accel(3), R_Gyro(3), R_Mag(3)]
        """
        with self.lock:
            l = self.left_state
            r = self.right_state

            vec = [
                *l["flex_norm"],
                l["accel"][0] / 16384.0, l["accel"][1] / 16384.0, l["accel"][2] / 16384.0,
                l["gyro"][0] / 131.0,   l["gyro"][1] / 131.0,   l["gyro"][2] / 131.0,
                l["mag"][0] / 100.0,    l["mag"][1] / 100.0,    l["mag"][2] / 100.0,

                *r["flex_norm"],
                r["accel"][0] / 16384.0, r["accel"][1] / 16384.0, r["accel"][2] / 16384.0,
                r["gyro"][0] / 131.0,   r["gyro"][1] / 131.0,   r["gyro"][2] / 131.0,
                r["mag"][0] / 100.0,    r["mag"][1] / 100.0,    r["mag"][2] / 100.0
            ]
            return np.array(vec, dtype=np.float32)

    # ==========================================
    # Simulation Support (for quick testing/demo)
    # ==========================================
    def start_simulation(self):
        """Simulate active gloves with synthetic waving/bending signals."""
        if self.simulation_mode:
            return
        self.simulation_mode = True
        self.sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self.sim_thread.start()
        logger.info("Hardware Simulation mode started.")

    def stop_simulation(self):
        self.simulation_mode = False
        if self.sim_thread:
            self.sim_thread.join(timeout=0.5)

    def _sim_loop(self):
        t = 0.0
        while self.simulation_mode and self.running:
            time.sleep(0.02) # 50Hz
            t += 0.02
            # Synthesize oscillating finger and IMU signals
            l_flex_norm = [
                0.2 + 0.3 * np.sin(t * 1.5 + 0),
                0.5 + 0.4 * np.sin(t * 2.0 + 1),
                0.4 + 0.4 * np.sin(t * 2.1 + 2),
                0.3 + 0.3 * np.sin(t * 1.8 + 3),
                0.2 + 0.2 * np.sin(t * 1.9 + 4)
            ]
            r_flex_norm = [
                0.1 + 0.4 * np.cos(t * 1.7 + 0),
                0.6 + 0.3 * np.cos(t * 2.2 + 1),
                0.5 + 0.4 * np.cos(t * 2.0 + 2),
                0.2 + 0.3 * np.cos(t * 1.6 + 3),
                0.1 + 0.2 * np.cos(t * 1.5 + 4)
            ]

            now = time.time()
            with self.lock:
                self.left_state["connected"] = True
                self.left_state["last_seen"] = now
                self.left_state["flex_norm"] = [float(np.clip(v, 0, 1)) for v in l_flex_norm]
                self.left_state["flex_raw"] = [int(2800 - v * 1600) for v in self.left_state["flex_norm"]]
                self.left_state["accel"] = [int(2000 * np.sin(t)), int(1000 * np.cos(t)), 16000]
                self.left_state["gyro"] = [int(500 * np.sin(t * 3)), int(400 * np.cos(t * 2)), 0]
                self.left_state["mag"] = [120, -80, 240]
                self.left_state["ip_addr"] = "127.0.0.1 (SIM)"

                self.right_state["connected"] = True
                self.right_state["last_seen"] = now
                self.right_state["flex_norm"] = [float(np.clip(v, 0, 1)) for v in r_flex_norm]
                self.right_state["flex_raw"] = [int(2800 - v * 1600) for v in self.right_state["flex_norm"]]
                self.right_state["accel"] = [int(-1500 * np.cos(t)), int(1800 * np.sin(t)), 16200]
                self.right_state["gyro"] = [int(300 * np.cos(t * 2)), int(-500 * np.sin(t * 2)), 0]
                self.right_state["mag"] = [-90, 140, 210]
                self.right_state["ip_addr"] = "127.0.0.1 (SIM)"

                self.left_packet_count += 1
                self.right_packet_count += 1
