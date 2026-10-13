"""
ASL to Nigerian Languages Translator - Desktop GUI Application
Integrates dual-glove wireless telemetry, live dataset collection, model training,
real-time gesture inference, N-Atlas translation (Yoruba, Igbo, Hausa), and multilingual TTS.
"""
import os
import sys
import time
import threading
import logging
from typing import Dict, Any, Optional

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

import config
from receiver import DualGloveReceiver
from dataset_recorder import DatasetRecorder
from train_model import train_classifier
from gesture_recognizer import GestureRecognizer
from sentence_builder import SentenceBuilder
from translator import NAtlasTranslator
from tts_engine import TTSEngine
from hand_visualizer import HandVisualizerWidget

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ASLTranslatorApp")

class ASLTranslatorApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("N-Atlas: ASL Glove Translator to Nigerian Languages")
        self.geometry("1100x800")
        self.minsize(980, 720)

        # Style configuration (Modern Dark Palette)
        self.configure(bg="#121820")
        self._init_styles()

        # Core Backend Modules
        self.receiver = DualGloveReceiver()
        self.recorder = DatasetRecorder(self.receiver)
        self.recognizer = GestureRecognizer()
        self.builder = SentenceBuilder()
        self.translator = NAtlasTranslator()
        self.tts = TTSEngine()

        # UI State Variables
        self.target_lang_var = tk.StringVar(value="Yoruba")
        self.neural_mode_var = tk.BooleanVar(value=False)
        self.auto_speak_var = tk.BooleanVar(value=False)
        self.sim_mode_var = tk.BooleanVar(value=False)
        self.live_recognition_active = tk.BooleanVar(value=True)
        self.dataset_hand_mode_var = tk.StringVar(value="right")  # "right", "left", "both"
        self.train_hand_mode_var = tk.StringVar(value="auto")    # "auto", "right", "left", "dual"
        self.auto_clear_enabled_var = tk.BooleanVar(value=True)
        self.auto_clear_delay_var = tk.StringVar(value="6s")
        self.last_sentence_activity_time = time.time()

        # Connection Mode Variables (WIFI vs SERIAL)
        self.left_conn_mode_var = tk.StringVar(value="WIFI")
        self.right_conn_mode_var = tk.StringVar(value="WIFI")
        self.left_serial_port_var = tk.StringVar(value="")
        self.right_serial_port_var = tk.StringVar(value="")

        # Build UI Layout
        self._build_header()
        self._build_connection_bar()
        self._build_tabs()

        # Start Receiver & Background GUI Refresh Loop
        self.receiver.start()
        self.after(100, self._telemetry_refresh_loop)

        # Handle clean exit
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _init_styles(self):
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Dark theme palette
        bg_dark = "#121820"
        card_bg = "#1a2332"
        accent_blue = "#2563eb"
        text_light = "#f8fafc"
        text_dim = "#94a3b8"

        self.style.configure(".", background=bg_dark, foreground=text_light, font=("Segoe UI", 10))
        self.style.configure("TNotebook", background=bg_dark, borderwidth=0)
        self.style.configure("TNotebook.Tab", background="#1e293b", foreground=text_dim, padding=[16, 8], font=("Segoe UI", 10, "bold"))
        self.style.map("TNotebook.Tab", background=[("selected", accent_blue)], foreground=[("selected", text_light)])
        self.style.configure("TFrame", background=bg_dark)
        self.style.configure("Card.TFrame", background=card_bg, relief="flat")
        self.style.configure("TLabel", background=bg_dark, foreground=text_light)
        self.style.configure("Card.TLabel", background=card_bg, foreground=text_light)
        self.style.configure("TButton", padding=[10, 6], background="#1e293b", foreground=text_light, borderwidth=1)
        self.style.map("TButton", background=[("active", "#334155")])

        # Treeview styling: Crisp black text on clean white background so dataset entries are clearly visible
        self.style.configure(
            "Treeview",
            background="#ffffff",
            foreground="#000000",
            fieldbackground="#ffffff",
            rowheight=24,
            font=("Segoe UI", 10)
        )
        self.style.configure(
            "Treeview.Heading",
            background="#e2e8f0",
            foreground="#000000",
            font=("Segoe UI", 9, "bold")
        )
        self.style.map(
            "Treeview",
            background=[("selected", "#0284c7")],
            foreground=[("selected", "#ffffff")]
        )

    def _build_header(self):
        header_frame = tk.Frame(self, bg="#0f172a", height=70, padx=20, pady=10)
        header_frame.pack(fill=tk.X)

        title_label = tk.Label(
            header_frame,
            text="N-ATLAS : DUAL-GLOVE ASL TRANSLATOR",
            font=("Segoe UI", 14, "bold"),
            fg="#38bdf8",
            bg="#0f172a"
        )
        title_label.pack(side=tk.LEFT)

        subtitle_label = tk.Label(
            header_frame,
            text=" (ASL -> English -> Yorùbá / Igbo / Hausa)",
            font=("Segoe UI", 10),
            fg="#94a3b8",
            bg="#0f172a"
        )
        subtitle_label.pack(side=tk.LEFT, padx=5)

        # Hardware Status indicators
        status_box = tk.Frame(header_frame, bg="#0f172a")
        status_box.pack(side=tk.RIGHT)

        # Simulation Toggle
        sim_chk = tk.Checkbutton(
            status_box,
            text="Simulation Mode",
            variable=self.sim_mode_var,
            command=self._toggle_simulation,
            bg="#0f172a",
            fg="#cbd5e1",
            selectcolor="#1e293b",
            activebackground="#0f172a",
            activeforeground="#38bdf8",
            font=("Segoe UI", 9)
        )
        sim_chk.pack(side=tk.LEFT, padx=12)

        # Left Hand Pill
        self.left_indicator = tk.Label(
            status_box,
            text="● LEFT GLOVE: OFF",
            font=("Segoe UI", 9, "bold"),
            fg="#ef4444",
            bg="#1e293b",
            padx=10,
            pady=4,
            relief="flat"
        )
        self.left_indicator.pack(side=tk.LEFT, padx=6)

        # Right Hand Pill
        self.right_indicator = tk.Label(
            status_box,
            text="● RIGHT GLOVE: OFF",
            font=("Segoe UI", 9, "bold"),
            fg="#ef4444",
            bg="#1e293b",
            padx=10,
            pady=4,
            relief="flat"
        )
        self.right_indicator.pack(side=tk.LEFT, padx=6)

        # Active Model / Hand Mode Indicator
        self.lbl_model_indicator = tk.Label(
            status_box,
            text="● MODEL: LOADING...",
            font=("Segoe UI", 9, "bold"),
            fg="#fbbf24",
            bg="#1e293b",
            padx=10,
            pady=4,
            relief="flat"
        )
        self.lbl_model_indicator.pack(side=tk.LEFT, padx=6)

    def _build_connection_bar(self):
        conn_frame = tk.Frame(self, bg="#161f2e", padx=14, pady=8, bd=1, relief="ridge")
        conn_frame.pack(fill=tk.X, padx=15, pady=(8, 0))

        # Title & Info Button
        title_box = tk.Frame(conn_frame, bg="#161f2e")
        title_box.pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(
            title_box,
            text="HARDWARE LINK:",
            font=("Segoe UI", 9, "bold"),
            fg="#fbbf24",
            bg="#161f2e"
        ).pack(side=tk.LEFT)
        tk.Button(
            title_box,
            text="ℹ️ Protocol Guide",
            font=("Segoe UI", 8),
            bg="#334155",
            fg="#38bdf8",
            padx=4,
            pady=0,
            command=self._show_connection_help,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=4)

        # --- LEFT GLOVE SECTION ---
        l_box = tk.Frame(conn_frame, bg="#161f2e")
        l_box.pack(side=tk.LEFT, padx=6)

        # Inbuilt LED for Left Glove
        self.led_left = tk.Label(
            l_box,
            text="● LED: OFF",
            font=("Segoe UI", 8, "bold"),
            fg="#64748b",
            bg="#0f172a",
            padx=6,
            pady=3,
            relief="groove"
        )
        self.led_left.pack(side=tk.LEFT, padx=3)

        # Mode Toggle Button (Wi-Fi vs USB Serial)
        self.btn_left_mode = tk.Button(
            l_box,
            text="📶 Wi-Fi",
            font=("Segoe UI", 8, "bold"),
            bg="#1e293b",
            fg="#38bdf8",
            padx=6,
            pady=1,
            command=self._toggle_left_conn_mode,
            cursor="hand2"
        )
        self.btn_left_mode.pack(side=tk.LEFT, padx=2)

        # Wi-Fi input container for Left
        self.frame_left_wifi = tk.Frame(l_box, bg="#161f2e")
        tk.Label(self.frame_left_wifi, text="IP:", font=("Segoe UI", 9, "bold"), fg="#38bdf8", bg="#161f2e").pack(side=tk.LEFT, padx=(2, 2))
        self.entry_left_ip = tk.Entry(self.frame_left_wifi, width=13, font=("Segoe UI", 9), bg="#0f172a", fg="#f8fafc", insertbackground="white")
        self.entry_left_ip.insert(0, "192.168.205.81")
        self.entry_left_ip.pack(side=tk.LEFT, padx=2)
        self.entry_left_ip.bind("<Return>", lambda e: self._connect_left_glove())
        self.frame_left_wifi.pack(side=tk.LEFT, padx=2)

        # Serial input container for Left (hidden by default)
        self.frame_left_serial = tk.Frame(l_box, bg="#161f2e")
        tk.Label(self.frame_left_serial, text="Port:", font=("Segoe UI", 9, "bold"), fg="#38bdf8", bg="#161f2e").pack(side=tk.LEFT, padx=(2, 2))
        self.combo_left_serial = ttk.Combobox(
            self.frame_left_serial,
            textvariable=self.left_serial_port_var,
            width=14,
            state="readonly",
            font=("Segoe UI", 8)
        )
        self.combo_left_serial.pack(side=tk.LEFT, padx=2)
        tk.Button(
            self.frame_left_serial,
            text="🔄",
            font=("Segoe UI", 8),
            bg="#1e293b",
            fg="#38bdf8",
            padx=3,
            pady=0,
            command=self._refresh_serial_ports,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=1)

        self.btn_left_connect = tk.Button(
            l_box,
            text="🔗 Connect",
            font=("Segoe UI", 8, "bold"),
            bg="#0284c7",
            fg="white",
            padx=7,
            pady=1,
            command=self._connect_left_glove
        )
        self.btn_left_connect.pack(side=tk.LEFT, padx=3)

        tk.Button(
            l_box,
            text="⚡ Ping",
            font=("Segoe UI", 8, "bold"),
            bg="#2563eb",
            fg="white",
            padx=6,
            pady=1,
            command=self._ping_left_glove
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            l_box,
            text="💡 Flash LED",
            font=("Segoe UI", 8),
            bg="#334155",
            fg="white",
            padx=5,
            pady=1,
            command=self._blink_left_glove
        ).pack(side=tk.LEFT, padx=2)

        self.lbl_left_conn_status = tk.Label(
            l_box,
            text="Standby",
            font=("Segoe UI", 8),
            fg="#94a3b8",
            bg="#161f2e"
        )
        self.lbl_left_conn_status.pack(side=tk.LEFT, padx=4)

        # Separator
        tk.Label(conn_frame, text="|", font=("Segoe UI", 12), fg="#334155", bg="#161f2e").pack(side=tk.LEFT, padx=6)

        # --- RIGHT GLOVE SECTION ---
        r_box = tk.Frame(conn_frame, bg="#161f2e")
        r_box.pack(side=tk.LEFT, padx=6)

        # Inbuilt LED for Right Glove
        self.led_right = tk.Label(
            r_box,
            text="● LED: OFF",
            font=("Segoe UI", 8, "bold"),
            fg="#64748b",
            bg="#0f172a",
            padx=6,
            pady=3,
            relief="groove"
        )
        self.led_right.pack(side=tk.LEFT, padx=3)

        # Mode Toggle Button (Wi-Fi vs USB Serial)
        self.btn_right_mode = tk.Button(
            r_box,
            text="📶 Wi-Fi",
            font=("Segoe UI", 8, "bold"),
            bg="#1e293b",
            fg="#10b981",
            padx=6,
            pady=1,
            command=self._toggle_right_conn_mode,
            cursor="hand2"
        )
        self.btn_right_mode.pack(side=tk.LEFT, padx=2)

        # Wi-Fi input container for Right
        self.frame_right_wifi = tk.Frame(r_box, bg="#161f2e")
        tk.Label(self.frame_right_wifi, text="IP:", font=("Segoe UI", 9, "bold"), fg="#10b981", bg="#161f2e").pack(side=tk.LEFT, padx=(2, 2))
        self.entry_right_ip = tk.Entry(self.frame_right_wifi, width=13, font=("Segoe UI", 9), bg="#0f172a", fg="#f8fafc", insertbackground="white")
        self.entry_right_ip.insert(0, "192.168.205.82")
        self.entry_right_ip.pack(side=tk.LEFT, padx=2)
        self.entry_right_ip.bind("<Return>", lambda e: self._connect_right_glove())
        self.frame_right_wifi.pack(side=tk.LEFT, padx=2)

        # Serial input container for Right (hidden by default)
        self.frame_right_serial = tk.Frame(r_box, bg="#161f2e")
        tk.Label(self.frame_right_serial, text="Port:", font=("Segoe UI", 9, "bold"), fg="#10b981", bg="#161f2e").pack(side=tk.LEFT, padx=(2, 2))
        self.combo_right_serial = ttk.Combobox(
            self.frame_right_serial,
            textvariable=self.right_serial_port_var,
            width=14,
            state="readonly",
            font=("Segoe UI", 8)
        )
        self.combo_right_serial.pack(side=tk.LEFT, padx=2)
        tk.Button(
            self.frame_right_serial,
            text="🔄",
            font=("Segoe UI", 8),
            bg="#1e293b",
            fg="#10b981",
            padx=3,
            pady=0,
            command=self._refresh_serial_ports,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=1)

        self.btn_right_connect = tk.Button(
            r_box,
            text="🔗 Connect",
            font=("Segoe UI", 8, "bold"),
            bg="#059669",
            fg="white",
            padx=7,
            pady=1,
            command=self._connect_right_glove
        )
        self.btn_right_connect.pack(side=tk.LEFT, padx=3)

        tk.Button(
            r_box,
            text="⚡ Ping",
            font=("Segoe UI", 8, "bold"),
            bg="#059669",
            fg="white",
            padx=6,
            pady=1,
            command=self._ping_right_glove
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            r_box,
            text="💡 Flash LED",
            font=("Segoe UI", 8),
            bg="#334155",
            fg="white",
            padx=5,
            pady=1,
            command=self._blink_right_glove
        ).pack(side=tk.LEFT, padx=2)

        self.lbl_right_conn_status = tk.Label(
            r_box,
            text="Standby",
            font=("Segoe UI", 8),
            fg="#94a3b8",
            bg="#161f2e"
        )
        self.lbl_right_conn_status.pack(side=tk.LEFT, padx=4)

        # Initial serial ports discovery
        self._refresh_serial_ports()

        # Connection Guidance Sub-Banner
        hint_frame = tk.Frame(self, bg="#0f172a", padx=16, pady=5, bd=1, relief="flat")
        hint_frame.pack(fill=tk.X, padx=15, pady=(2, 0))
        self.lbl_conn_hint = tk.Label(
            hint_frame,
            text="💡 Dual Link: Choose [📶 Wi-Fi] (enter ESP32 IP) or [🔌 USB Serial] (select COM port), then click '🔗 Connect'. Zero host PC IP setup needed!",
            font=("Segoe UI", 9, "bold"),
            fg="#38bdf8",
            bg="#0f172a"
        )
        self.lbl_conn_hint.pack(side=tk.LEFT)

    def _refresh_serial_ports(self):
        """Scans system for available COM / Serial ports and populates dropdowns."""
        ports = self.receiver.list_available_serial_ports()
        port_items = []
        for p in ports:
            desc = p.get("description", "")
            port_name = p.get("port", "")
            if desc and desc != port_name:
                port_items.append(f"{port_name} ({desc[:24]})")
            else:
                port_items.append(port_name)

        if not port_items:
            port_items = ["(No COM Ports)"]

        if hasattr(self, "combo_left_serial"):
            self.combo_left_serial["values"] = port_items
            if not self.left_serial_port_var.get() or self.left_serial_port_var.get() not in port_items:
                self.left_serial_port_var.set(port_items[0])

        if hasattr(self, "combo_right_serial"):
            self.combo_right_serial["values"] = port_items
            if not self.right_serial_port_var.get() or self.right_serial_port_var.get() not in port_items:
                self.right_serial_port_var.set(port_items[0])

    def _toggle_left_conn_mode(self):
        """Switch between Wi-Fi and USB Serial connection mode for Left Glove."""
        if self.receiver.left_state["connected"]:
            messagebox.showinfo("Disconnect First", "Please disconnect Left Glove before switching connection mode.")
            return

        cur = self.left_conn_mode_var.get()
        if cur == "WIFI":
            self.left_conn_mode_var.set("SERIAL")
            self.btn_left_mode.config(text="🔌 USB Serial", fg="#fbbf24")
            self.frame_left_wifi.pack_forget()
            self.frame_left_serial.pack(side=tk.LEFT, padx=2, before=self.btn_left_connect)
            self._refresh_serial_ports()
        else:
            self.left_conn_mode_var.set("WIFI")
            self.btn_left_mode.config(text="📶 Wi-Fi", fg="#38bdf8")
            self.frame_left_serial.pack_forget()
            self.frame_left_wifi.pack(side=tk.LEFT, padx=2, before=self.btn_left_connect)

    def _toggle_right_conn_mode(self):
        """Switch between Wi-Fi and USB Serial connection mode for Right Glove."""
        if self.receiver.right_state["connected"]:
            messagebox.showinfo("Disconnect First", "Please disconnect Right Glove before switching connection mode.")
            return

        cur = self.right_conn_mode_var.get()
        if cur == "WIFI":
            self.right_conn_mode_var.set("SERIAL")
            self.btn_right_mode.config(text="🔌 USB Serial", fg="#fbbf24")
            self.frame_right_wifi.pack_forget()
            self.frame_right_serial.pack(side=tk.LEFT, padx=2, before=self.btn_right_connect)
            self._refresh_serial_ports()
        else:
            self.right_conn_mode_var.set("WIFI")
            self.btn_right_mode.config(text="📶 Wi-Fi", fg="#10b981")
            self.frame_right_serial.pack_forget()
            self.frame_right_wifi.pack(side=tk.LEFT, padx=2, before=self.btn_right_connect)

    def _show_connection_help(self):
        """Displays explanation modal about zero-config ESP32 dual connection architecture."""
        help_win = tk.Toplevel(self)
        help_win.title("How ESP32 Glove Connection Works (Dual Transport Link)")
        help_win.geometry("620x530")
        help_win.configure(bg="#0f172a")
        help_win.resizable(False, False)
        help_win.transient(self)
        help_win.grab_set()

        card = tk.Frame(help_win, bg="#1a2332", padx=20, pady=16, bd=2, relief="groove")
        card.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        tk.Label(
            card,
            text="📡 DUAL TRANSPORT LINK PROTOCOL (WI-FI & USB SERIAL)",
            font=("Segoe UI", 11, "bold"),
            fg="#fbbf24",
            bg="#1a2332"
        ).pack(pady=(0, 8))

        info_text = (
            "OPTION 1: WIRELESS WI-FI UDP (Primary)\n"
            "   • No Host PC IP setup needed! Gloves wait silently in STANDBY on boot.\n"
            "   • Check glove IP on boot (e.g. 192.168.205.81 or 192.168.205.82).\n"
            "   • Ensure glove mode is set to '📶 Wi-Fi', enter IP, and click '🔗 Connect'.\n"
            "   • ESP32 acknowledges handshake, turns LED solid ON, and streams 50 Hz JSON.\n\n"
            "OPTION 2: WIRED USB SERIAL / COM PORT (Secondary / Fallback)\n"
            "   • Plug ESP32 glove into laptop via micro-USB / Type-C data cable.\n"
            "   • Click the mode button to toggle to '🔌 USB Serial'.\n"
            "   • Select the COM port from dropdown (e.g. COM10 - CP210x USB Bridge).\n"
            "   • Click '🔗 Connect' — Python handshakes at 115200 baud and streams 50 Hz JSON.\n"
            "   • Completely immune to Wi-Fi drops, router client isolation, or hotspot issues.\n\n"
            "⚠️ IMPORTANT SERIAL MONITOR NOTE:\n"
            "   • On Windows, a COM port can only be opened by ONE application at a time.\n"
            "   • Always close the Arduino IDE Serial Monitor before connecting in this GUI!"
        )

        msg_box = tk.Label(
            card,
            text=info_text,
            font=("Segoe UI", 9),
            fg="#f8fafc",
            bg="#1a2332",
            justify=tk.LEFT,
            wraplength=570
        )
        msg_box.pack(fill=tk.BOTH, expand=True, pady=4)

        tk.Button(
            card,
            text="Got it!",
            font=("Segoe UI", 9, "bold"),
            bg="#0284c7",
            fg="white",
            padx=16,
            pady=4,
            command=help_win.destroy
        ).pack(pady=(8, 0))

    def _show_connect_orientation_prompt(self, hand: str, target_desc: str, on_proceed):
        """
        Display instructional modal prompting user to hold hand vertically:
        - Fingertips pointing straight upward at 90°
        - Palm facing the PC monitor (dorsal side with sensor facing user)
        Ensures accurate zeroing and reference orientation when glove connects.
        """
        dialog = tk.Toplevel(self)
        dialog.title(f"{hand} Glove Orientation Setup")
        dialog.geometry("520x430")
        dialog.configure(bg="#0f172a")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        # Center dialog over main app window
        try:
            x = self.winfo_x() + (self.winfo_width() // 2) - 260
            y = self.winfo_y() + (self.winfo_height() // 2) - 215
            dialog.geometry(f"+{x}+{y}")
        except Exception:
            pass

        card = tk.Frame(dialog, bg="#1a2332", padx=20, pady=16, bd=2, relief="groove")
        card.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        # Header Badge
        header_color = "#38bdf8" if hand == "LEFT" else "#10b981"
        tk.Label(
            card,
            text=f"🖐 {hand} GLOVE ORIENTATION SETUP",
            font=("Segoe UI", 13, "bold"),
            fg=header_color,
            bg="#1a2332"
        ).pack(pady=(0, 4))

        tk.Label(
            card,
            text=f"Position your hand before connecting to {target_desc}:",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#1a2332"
        ).pack(pady=(0, 10))

        # Instructions Frame
        inst_box = tk.Frame(card, bg="#0f172a", padx=14, pady=10, bd=1, relief="ridge")
        inst_box.pack(fill=tk.X, pady=(0, 10))

        steps = [
            ("1. Hand Posture:", "Raise hand up vertically in front of your chest."),
            ("2. Fingertips (90°):", "Point fingertips straight UPWARD toward the ceiling."),
            ("3. Palm Facing PC:", "Face palm towards PC monitor (back of hand faces YOU)."),
            ("4. Hold Steady:", "Keep hand still for 2 sec while connection zeroes IMU.")
        ]

        for title, detail in steps:
            row = tk.Frame(inst_box, bg="#0f172a")
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=title, font=("Segoe UI", 9, "bold"), fg="#fbbf24", bg="#0f172a", width=18, anchor=tk.W).pack(side=tk.LEFT)
            tk.Label(row, text=detail, font=("Segoe UI", 9), fg="#f8fafc", bg="#0f172a", anchor=tk.W).pack(side=tk.LEFT)

        # Visual ASCII Cue Box
        art_box = tk.Frame(card, bg="#162235", padx=10, pady=6, bd=1, relief="sunken")
        art_box.pack(fill=tk.X, pady=(0, 10))

        ascii_art = (
            "           ▲ [90° Vertical Fingertips]\n"
            "           │ \n"
            "     [ 🖐  PALM FACING PC MONITOR  ]\n"
            "     [   Dorsal Sensor Faces You  ]\n"
            "           │ \n"
            "           ▼ [Wrist & Forearm]"
        )
        tk.Label(
            art_box,
            text=ascii_art,
            font=("Consolas", 8, "bold"),
            fg="#38bdf8" if hand == "LEFT" else "#10b981",
            bg="#162235",
            justify=tk.CENTER
        ).pack()

        # Action Buttons
        btn_row = tk.Frame(card, bg="#1a2332")
        btn_row.pack(fill=tk.X, pady=(4, 0))

        def _on_confirm():
            dialog.destroy()
            on_proceed()

        def _on_cancel():
            dialog.destroy()

        btn_confirm = tk.Button(
            btn_row,
            text="✓ Ready - Connect & Calibrate",
            font=("Segoe UI", 10, "bold"),
            bg="#0284c7" if hand == "LEFT" else "#059669",
            fg="white",
            padx=14,
            pady=6,
            command=_on_confirm
        )
        btn_confirm.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 6))

        btn_cancel = tk.Button(
            btn_row,
            text="✕ Cancel",
            font=("Segoe UI", 10),
            bg="#334155",
            fg="white",
            padx=10,
            pady=6,
            command=_on_cancel
        )
        btn_cancel.pack(side=tk.RIGHT, padx=(6, 0))

    def _connect_left_glove(self):
        if self.receiver.left_state["connected"]:
            self.receiver.disconnect_glove("LEFT")
            self.btn_left_connect.config(text="🔗 Connect", bg="#0284c7")
            self.lbl_left_conn_status.config(text="Standby", fg="#94a3b8")
            return

        mode = self.left_conn_mode_var.get()
        if mode == "SERIAL":
            raw_port = self.left_serial_port_var.get().strip()
            if not raw_port or "(No COM" in raw_port:
                messagebox.showwarning(
                    "No Port Selected",
                    "Please select an available USB Serial / COM port for the Left Glove.\n\n"
                    "If your ESP32 is plugged in, click the 🔄 refresh button."
                )
                return
            port = raw_port.split(" ")[0].strip() if " " in raw_port else raw_port

            def _do_serial_connect():
                self.btn_left_connect.config(text="Connecting...", bg="#d97706", state=tk.DISABLED)
                self.lbl_left_conn_status.config(text=f"Connecting to {port}...", fg="#f59e0b")

                def _worker():
                    success, msg = self.receiver.connect_serial_glove("LEFT", port)
                    def _update():
                        self.btn_left_connect.config(state=tk.NORMAL)
                        if success:
                            self.btn_left_connect.config(text="✕ Disconnect", bg="#b91c1c")
                            self.lbl_left_conn_status.config(text=f"{msg} (90° Tared)", fg="#10b981")
                            self.receiver.tare_orientation_vertical("LEFT")
                            if hasattr(self, "hand_visualizer"):
                                self.hand_visualizer.canvas.orientation_calc.zero_to_vertical()
                        elif msg.startswith("HAND_MISMATCH:"):
                            detected_hand = msg.split(":")[1].strip()
                            self.btn_left_connect.config(text="🔗 Connect", bg="#0284c7")
                            self.lbl_left_conn_status.config(text=f"Detected: {detected_hand}", fg="#f59e0b")
                            do_switch = messagebox.askyesno(
                                "Hand Mismatch Detected",
                                f"Port {port} responded, but it is transmitting as the {detected_hand} GLOVE!\n\n"
                                f"You clicked Connect under 'Left Glove'.\n\n"
                                f"Would you like to connect {port} as your {detected_hand} Glove now?"
                            )
                            if do_switch and detected_hand == "RIGHT":
                                if self.right_conn_mode_var.get() != "SERIAL":
                                    self._toggle_right_conn_mode()
                                self.right_serial_port_var.set(raw_port)
                                self._connect_right_glove()
                        else:
                            self.btn_left_connect.config(text="🔗 Connect", bg="#0284c7")
                            self.lbl_left_conn_status.config(text=f"Failed: {msg}", fg="#ef4444")
                            messagebox.showerror(
                                "Serial Connection Failed",
                                f"Could not connect to Left Glove on {port}.\n\n"
                                f"Details: {msg}\n\n"
                                "Troubleshooting Checklist:\n"
                                "1. Close Arduino Serial Monitor (Windows locks COM ports to one app).\n"
                                "2. Ensure the USB cable supports data and is firmly plugged in.\n"
                                "3. If this board is your Right Glove, connect it under the 'Right Glove' panel."
                            )
                    self.after(0, _update)

                threading.Thread(target=_worker, daemon=True).start()

            self._show_connect_orientation_prompt("LEFT", f"USB Serial ({port})", _do_serial_connect)

        else: # Wi-Fi Mode
            ip = self.entry_left_ip.get().strip()
            if not ip:
                messagebox.showwarning(
                    "Missing IP",
                    "Please enter the Left Glove ESP32 IP address.\n\n"
                    "Look at the Arduino Serial Monitor when the ESP32 boots up to find its IP (e.g. 192.168.205.81).\n"
                    "Note: Your ESP32 does NOT need your PC IP — entering the glove IP here links them automatically."
                )
                return

            def _do_wifi_connect():
                self.btn_left_connect.config(text="Connecting...", bg="#d97706", state=tk.DISABLED)
                self.lbl_left_conn_status.config(text=f"Connecting to {ip}...", fg="#f59e0b")

                def _worker():
                    success, msg = self.receiver.connect_glove("LEFT", ip)
                    def _update():
                        self.btn_left_connect.config(state=tk.NORMAL)
                        if success:
                            self.btn_left_connect.config(text="✕ Disconnect", bg="#b91c1c")
                            self.lbl_left_conn_status.config(text=f"{msg} (90° Tared)", fg="#10b981")
                            self.receiver.tare_orientation_vertical("LEFT")
                            if hasattr(self, "hand_visualizer"):
                                self.hand_visualizer.canvas.orientation_calc.zero_to_vertical()
                        else:
                            self.btn_left_connect.config(text="🔗 Connect", bg="#0284c7")
                            self.lbl_left_conn_status.config(text=f"Failed: {msg}", fg="#ef4444")
                            messagebox.showerror(
                                "Connection Failed",
                                f"Could not connect to Left Glove at {ip}:5005.\n\n"
                                "Troubleshooting Checklist:\n"
                                "1. Confirm you uploaded the updated sketch (esp32_glove_left.ino) to the ESP32.\n"
                                "2. Check Arduino Serial Monitor — it should show:\n"
                                f"   'GLOVE IP: {ip} | STATUS: STANDBY'.\n"
                                "3. Confirm your PC and ESP32 are both connected to Wi-Fi 'KINGDOM'.\n"
                                "4. The ESP32 does NOT look for your PC IP — clicking Connect sends the link from your PC directly to the glove."
                            )
                    self.after(0, _update)

                threading.Thread(target=_worker, daemon=True).start()

            self._show_connect_orientation_prompt("LEFT", f"Wi-Fi ({ip})", _do_wifi_connect)

    def _connect_right_glove(self):
        if self.receiver.right_state["connected"]:
            self.receiver.disconnect_glove("RIGHT")
            self.btn_right_connect.config(text="🔗 Connect", bg="#059669")
            self.lbl_right_conn_status.config(text="Standby", fg="#94a3b8")
            return

        mode = self.right_conn_mode_var.get()
        if mode == "SERIAL":
            raw_port = self.right_serial_port_var.get().strip()
            if not raw_port or "(No COM" in raw_port:
                messagebox.showwarning(
                    "No Port Selected",
                    "Please select an available USB Serial / COM port for the Right Glove.\n\n"
                    "If your ESP32 is plugged in, click the 🔄 refresh button."
                )
                return
            port = raw_port.split(" ")[0].strip() if " " in raw_port else raw_port

            def _do_serial_connect():
                self.btn_right_connect.config(text="Connecting...", bg="#d97706", state=tk.DISABLED)
                self.lbl_right_conn_status.config(text=f"Connecting to {port}...", fg="#f59e0b")

                def _worker():
                    success, msg = self.receiver.connect_serial_glove("RIGHT", port)
                    def _update():
                        self.btn_right_connect.config(state=tk.NORMAL)
                        if success:
                            self.btn_right_connect.config(text="✕ Disconnect", bg="#b91c1c")
                            self.lbl_right_conn_status.config(text=f"{msg} (90° Tared)", fg="#10b981")
                            self.receiver.tare_orientation_vertical("RIGHT")
                            if hasattr(self, "hand_visualizer"):
                                self.hand_visualizer.canvas.orientation_calc.zero_to_vertical()
                        elif msg.startswith("HAND_MISMATCH:"):
                            detected_hand = msg.split(":")[1].strip()
                            self.btn_right_connect.config(text="🔗 Connect", bg="#059669")
                            self.lbl_right_conn_status.config(text=f"Detected: {detected_hand}", fg="#f59e0b")
                            do_switch = messagebox.askyesno(
                                "Hand Mismatch Detected",
                                f"Port {port} responded, but it is transmitting as the {detected_hand} GLOVE!\n\n"
                                f"You clicked Connect under 'Right Glove'.\n\n"
                                f"Would you like to connect {port} as your {detected_hand} Glove now?"
                            )
                            if do_switch and detected_hand == "LEFT":
                                if self.left_conn_mode_var.get() != "SERIAL":
                                    self._toggle_left_conn_mode()
                                self.left_serial_port_var.set(raw_port)
                                self._connect_left_glove()
                        else:
                            self.btn_right_connect.config(text="🔗 Connect", bg="#059669")
                            self.lbl_right_conn_status.config(text=f"Failed: {msg}", fg="#ef4444")
                            messagebox.showerror(
                                "Serial Connection Failed",
                                f"Could not connect to Right Glove on {port}.\n\n"
                                f"Details: {msg}\n\n"
                                "Troubleshooting Checklist:\n"
                                "1. Close Arduino Serial Monitor (Windows locks COM ports to one app).\n"
                                "2. Ensure the USB cable supports data and is firmly plugged in.\n"
                                "3. If this board is your Left Glove, connect it under the 'Left Glove' panel."
                            )
                    self.after(0, _update)

                threading.Thread(target=_worker, daemon=True).start()

            self._show_connect_orientation_prompt("RIGHT", f"USB Serial ({port})", _do_serial_connect)

        else: # Wi-Fi Mode
            ip = self.entry_right_ip.get().strip()
            if not ip:
                messagebox.showwarning(
                    "Missing IP",
                    "Please enter the Right Glove ESP32 IP address.\n\n"
                    "Look at the Arduino Serial Monitor when the ESP32 boots up to find its IP (e.g. 192.168.205.82).\n"
                    "Note: Your ESP32 does NOT need your PC IP — entering the glove IP here links them automatically."
                )
                return

            def _do_wifi_connect():
                self.btn_right_connect.config(text="Connecting...", bg="#d97706", state=tk.DISABLED)
                self.lbl_right_conn_status.config(text=f"Connecting to {ip}...", fg="#f59e0b")

                def _worker():
                    success, msg = self.receiver.connect_glove("RIGHT", ip)
                    def _update():
                        self.btn_right_connect.config(state=tk.NORMAL)
                        if success:
                            self.btn_right_connect.config(text="✕ Disconnect", bg="#b91c1c")
                            self.lbl_right_conn_status.config(text=f"{msg} (90° Tared)", fg="#10b981")
                            self.receiver.tare_orientation_vertical("RIGHT")
                            if hasattr(self, "hand_visualizer"):
                                self.hand_visualizer.canvas.orientation_calc.zero_to_vertical()
                        else:
                            self.btn_right_connect.config(text="🔗 Connect", bg="#059669")
                            self.lbl_right_conn_status.config(text=f"Failed: {msg}", fg="#ef4444")
                            messagebox.showerror(
                                "Connection Failed",
                                f"Could not connect to Right Glove at {ip}:5005.\n\n"
                                "Troubleshooting Checklist:\n"
                                "1. Confirm you uploaded the updated sketch (esp32_glove_right.ino) to the ESP32.\n"
                                "2. Check Arduino Serial Monitor — it should show:\n"
                                f"   'GLOVE IP: {ip} | STATUS: STANDBY'.\n"
                                "3. Confirm your PC and ESP32 are both connected to Wi-Fi 'KINGDOM'.\n"
                                "4. The ESP32 does NOT look for your PC IP — clicking Connect sends the link from your PC directly to the glove."
                            )
                    self.after(0, _update)

                threading.Thread(target=_worker, daemon=True).start()

            self._show_connect_orientation_prompt("RIGHT", f"Wi-Fi ({ip})", _do_wifi_connect)

    def _ping_left_glove(self):
        mode = self.left_conn_mode_var.get()
        if mode == "SERIAL":
            raw_port = self.left_serial_port_var.get().strip()
            target = raw_port.split(" ")[0].strip() if " " in raw_port else raw_port
        else:
            target = self.entry_left_ip.get().strip()

        if not target or "(No COM" in target:
            self.lbl_left_conn_status.config(text="No target configured", fg="#ef4444")
            return

        self.lbl_left_conn_status.config(text="Pinging...", fg="#f59e0b")
        def _worker():
            success, msg = self.receiver.ping_glove("LEFT", target)
            color = "#10b981" if success else "#ef4444"
            self.after(0, lambda: self.lbl_left_conn_status.config(text=msg, fg=color))
        threading.Thread(target=_worker, daemon=True).start()

    def _blink_left_glove(self):
        mode = self.left_conn_mode_var.get()
        if mode == "SERIAL":
            raw_port = self.left_serial_port_var.get().strip()
            target = raw_port.split(" ")[0].strip() if " " in raw_port else raw_port
        else:
            target = self.entry_left_ip.get().strip()
        self.receiver.blink_glove_led("LEFT", target)

    def _ping_right_glove(self):
        mode = self.right_conn_mode_var.get()
        if mode == "SERIAL":
            raw_port = self.right_serial_port_var.get().strip()
            target = raw_port.split(" ")[0].strip() if " " in raw_port else raw_port
        else:
            target = self.entry_right_ip.get().strip()

        if not target or "(No COM" in target:
            self.lbl_right_conn_status.config(text="No target configured", fg="#ef4444")
            return

        self.lbl_right_conn_status.config(text="Pinging...", fg="#f59e0b")
        def _worker():
            success, msg = self.receiver.ping_glove("RIGHT", target)
            color = "#10b981" if success else "#ef4444"
            self.after(0, lambda: self.lbl_right_conn_status.config(text=msg, fg=color))
        threading.Thread(target=_worker, daemon=True).start()

    def _blink_right_glove(self):
        mode = self.right_conn_mode_var.get()
        if mode == "SERIAL":
            raw_port = self.right_serial_port_var.get().strip()
            target = raw_port.split(" ")[0].strip() if " " in raw_port else raw_port
        else:
            target = self.entry_right_ip.get().strip()
        self.receiver.blink_glove_led("RIGHT", target)

    def _build_tabs(self):

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Tab 1: Live Sign Recognition & Multilingual Translation
        self.tab_translate = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_translate, text="  Live ASL Translation & TTS  ")
        self._build_translate_tab(self.tab_translate)

        # Tab 2: Glove Telemetry & Hardware Calibration
        self.tab_telemetry = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_telemetry, text="  Sensor Monitor & Calibration  ")
        self._build_telemetry_tab(self.tab_telemetry)

        # Tab 3: Dataset Studio
        self.tab_dataset = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_dataset, text="  Dataset Studio  ")
        self._build_dataset_tab(self.tab_dataset)

        # Tab 4: Model Training
        self.tab_train = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_train, text="  Model Training & Evaluation  ")
        self._build_train_tab(self.tab_train)

    # =========================================================================
    # TAB 1: Live Translation & TTS
    # =========================================================================
    def _build_translate_tab(self, parent):
        grid = tk.Frame(parent, bg="#121820")
        grid.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top Section: Real-time detected sign and confidence
        top_card = tk.Frame(grid, bg="#1a2332", bd=1, relief="ridge", padx=15, pady=12)
        top_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(top_card, text="LIVE ASL SIGN DETECTOR", font=("Segoe UI", 9, "bold"), fg="#94a3b8", bg="#1a2332").pack(anchor=tk.W)

        det_row = tk.Frame(top_card, bg="#1a2332")
        det_row.pack(fill=tk.X, pady=5)

        self.lbl_current_sign = tk.Label(det_row, text="WAITING FOR GESTURE...", font=("Segoe UI", 26, "bold"), fg="#38bdf8", bg="#1a2332")
        self.lbl_current_sign.pack(side=tk.LEFT)

        self.lbl_confidence = tk.Label(det_row, text="Confidence: 0.0%", font=("Segoe UI", 12), fg="#10b981", bg="#1a2332")
        self.lbl_confidence.pack(side=tk.RIGHT, padx=10)

        self.prob_bar = ttk.Progressbar(top_card, orient=tk.HORIZONTAL, length=400, mode="determinate")
        self.prob_bar.pack(fill=tk.X, pady=4)

        # Kinematics & Gesture Classification Bar (Static vs. Dynamic Workshop)
        kinematics_row = tk.Frame(top_card, bg="#1a2332")
        kinematics_row.pack(fill=tk.X, pady=(2, 6))

        self.lbl_gesture_type = tk.Label(
            kinematics_row,
            text="TYPE: STATIC POSE ✋",
            font=("Segoe UI", 9, "bold"),
            fg="#38bdf8",
            bg="#0f172a",
            padx=8,
            pady=3,
            relief="groove"
        )
        self.lbl_gesture_type.pack(side=tk.LEFT, padx=(0, 6))

        self.lbl_motion_state = tk.Label(
            kinematics_row,
            text="KINEMATICS: STEADY (0.0°/s)",
            font=("Segoe UI", 9),
            fg="#10b981",
            bg="#0f172a",
            padx=8,
            pady=3,
            relief="groove"
        )
        self.lbl_motion_state.pack(side=tk.LEFT, padx=6)

        self.lbl_latch_status = tk.Label(
            kinematics_row,
            text="LATCH: READY FOR SIGN",
            font=("Segoe UI", 9, "bold"),
            fg="#94a3b8",
            bg="#0f172a",
            padx=8,
            pady=3,
            relief="groove"
        )
        self.lbl_latch_status.pack(side=tk.LEFT, padx=6)

        # Quick Calibration Tare buttons right on Tab 1
        tk.Button(
            kinematics_row,
            text="🎯 Tare Flat (0°)",
            bg="#0284c7",
            fg="white",
            font=("Segoe UI", 8, "bold"),
            padx=6,
            pady=1,
            command=lambda: self._cal_set_flat(None)
        ).pack(side=tk.RIGHT, padx=3)

        tk.Button(
            kinematics_row,
            text="✊ Tare Fist (90°)",
            bg="#e11d48",
            fg="white",
            font=("Segoe UI", 8, "bold"),
            padx=6,
            pady=1,
            command=lambda: self._cal_set_bent(None)
        ).pack(side=tk.RIGHT, padx=3)

        # Active Model Info Badge
        self.lbl_model_status_badge = tk.Label(
            top_card,
            text="Active Model: Detecting...",
            font=("Segoe UI", 8),
            fg="#94a3b8",
            bg="#1a2332"
        )
        self.lbl_model_status_badge.pack(anchor=tk.W, pady=(2, 0))

        # Middle Section: English Sentence Construction
        mid_card = tk.Frame(grid, bg="#1a2332", bd=1, relief="ridge", padx=15, pady=12)
        mid_card.pack(fill=tk.X, pady=5)

        # Header with Auto-Clear Controls
        mid_hdr = tk.Frame(mid_card, bg="#1a2332")
        mid_hdr.pack(fill=tk.X, pady=(0, 4))

        tk.Label(mid_hdr, text="CONSTRUCTED ENGLISH SENTENCE", font=("Segoe UI", 9, "bold"), fg="#94a3b8", bg="#1a2332").pack(side=tk.LEFT)

        # Auto-Clear Controls on the right side of header
        auto_clear_frame = tk.Frame(mid_hdr, bg="#1a2332")
        auto_clear_frame.pack(side=tk.RIGHT)

        self.lbl_auto_clear_timer = tk.Label(
            auto_clear_frame,
            text="",
            font=("Segoe UI", 8, "bold"),
            fg="#fbbf24",
            bg="#1a2332"
        )
        self.lbl_auto_clear_timer.pack(side=tk.RIGHT, padx=(6, 0))

        # Auto-Clear delay selector dropdown
        self.cmb_auto_clear_delay = ttk.Combobox(
            auto_clear_frame,
            textvariable=self.auto_clear_delay_var,
            values=["3s", "5s", "6s", "8s", "10s", "15s", "30s"],
            width=5,
            state="readonly",
            font=("Segoe UI", 8)
        )
        self.cmb_auto_clear_delay.pack(side=tk.RIGHT, padx=(4, 6))

        # Auto-Clear checkbox toggle
        self.chk_auto_clear = tk.Checkbutton(
            auto_clear_frame,
            text="Auto-Clear when idle:",
            variable=self.auto_clear_enabled_var,
            bg="#1a2332",
            fg="#cbd5e1",
            selectcolor="#0f172a",
            activebackground="#1a2332",
            activeforeground="#38bdf8",
            font=("Segoe UI", 9)
        )
        self.chk_auto_clear.pack(side=tk.RIGHT)

        self.txt_english_sentence = tk.Text(mid_card, height=3, bg="#0f172a", fg="#f8fafc", font=("Segoe UI", 13), relief="flat", wrap=tk.WORD, padx=8, pady=8)
        self.txt_english_sentence.pack(fill=tk.X, pady=8)
        self.txt_english_sentence.bind("<Key>", lambda e: self._on_user_typing_event())

        btn_row = tk.Frame(mid_card, bg="#1a2332")
        btn_row.pack(fill=tk.X)

        tk.Button(btn_row, text="⌫ Backspace", bg="#334155", fg="white", font=("Segoe UI", 9), command=self._action_backspace).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_row, text="✕ Clear Sentence", bg="#475569", fg="white", font=("Segoe UI", 9), command=self._action_clear_sentence).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_row, text="🔊 Speak English", bg="#2563eb", fg="white", font=("Segoe UI", 9, "bold"), command=self._speak_english).pack(side=tk.LEFT, padx=8)
        tk.Button(btn_row, text="✓ Translate Now", bg="#059669", fg="white", font=("Segoe UI", 9, "bold"), command=self._trigger_translation).pack(side=tk.RIGHT, padx=3)

        # Bottom Section: N-Atlas Nigerian Languages Translation Card
        bot_card = tk.Frame(grid, bg="#1e293b", bd=2, relief="groove", padx=18, pady=14)
        bot_card.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        lang_control_row = tk.Frame(bot_card, bg="#1e293b")
        lang_control_row.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            lang_control_row,
            text="N-ATLAS NIGERIAN TRANSLATION",
            font=("Segoe UI", 11, "bold"),
            fg="#fbbf24",
            bg="#1e293b"
        ).pack(side=tk.LEFT)

        tk.Button(
            lang_control_row,
            text="ℹ️ N-Atlas Model Details",
            font=("Segoe UI", 8, "bold"),
            bg="#334155",
            fg="#38bdf8",
            padx=6,
            pady=1,
            command=self._show_natlas_model_info
        ).pack(side=tk.LEFT, padx=(10, 4))

        tk.Button(
            lang_control_row,
            text="☁️ Online API Token",
            font=("Segoe UI", 8, "bold"),
            bg="#0369a1",
            fg="white",
            padx=6,
            pady=1,
            command=self._prompt_set_hf_token
        ).pack(side=tk.LEFT, padx=4)

        # Neural Engine Toggle Checkbox
        chk_neural = tk.Checkbutton(
            lang_control_row,
            text="Use Deep Neural Transformer",
            variable=self.neural_mode_var,
            command=self._toggle_neural_model,
            bg="#1e293b",
            fg="#cbd5e1",
            selectcolor="#0f172a",
            activebackground="#1e293b",
            activeforeground="#38bdf8",
            font=("Segoe UI", 9)
        )
        chk_neural.pack(side=tk.LEFT, padx=6)

        # Language Radio Buttons
        lang_pill_box = tk.Frame(lang_control_row, bg="#1e293b")
        lang_pill_box.pack(side=tk.RIGHT)

        for lang in ["Yoruba", "Igbo", "Hausa"]:
            r = tk.Radiobutton(
                lang_pill_box,
                text=lang,
                variable=self.target_lang_var,
                value=lang,
                command=self._trigger_translation,
                bg="#1e293b",
                fg="#f8fafc",
                selectcolor="#0f172a",
                activebackground="#1e293b",
                activeforeground="#38bdf8",
                font=("Segoe UI", 10, "bold")
            )
            r.pack(side=tk.LEFT, padx=8)

        # Translation Output Box
        self.lbl_translation_output = tk.Label(
            bot_card,
            text="[Translation will appear here]",
            font=("Segoe UI", 18, "bold"),
            fg="#f8fafc",
            bg="#1e293b",
            wraplength=800,
            justify=tk.LEFT
        )
        self.lbl_translation_output.pack(anchor=tk.W, pady=10)

        # Phonetics Guide
        self.lbl_phonetics = tk.Label(
            bot_card,
            text="Pronunciation: -",
            font=("Segoe UI", 11, "italic"),
            fg="#94a3b8",
            bg="#1e293b"
        )
        self.lbl_phonetics.pack(anchor=tk.W, pady=(0, 10))

        # Bottom Actions & Engine Status
        bot_action_row = tk.Frame(bot_card, bg="#1e293b")
        bot_action_row.pack(fill=tk.X, side=tk.BOTTOM)

        self.btn_speak_nigerian = tk.Button(
            bot_action_row,
            text="🔊 Speak Translated Version (TTS)",
            font=("Segoe UI", 11, "bold"),
            bg="#d97706",
            fg="white",
            padx=16,
            pady=6,
            command=self._speak_nigerian
        )
        self.btn_speak_nigerian.pack(side=tk.LEFT)

        self.lbl_engine_info = tk.Label(
            bot_action_row,
            text="Engine: N-Atlas Lexicon | Diacritics: Active",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#1e293b"
        )
        self.lbl_engine_info.pack(side=tk.RIGHT, padx=10)

    # =========================================================================
    # =========================================================================
    # TAB 2: Telemetry & Calibration with 3D Skeleton Visualizer
    # =========================================================================
    def _build_telemetry_tab(self, parent):
        container = tk.Frame(parent, bg="#121820")
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Split: Left side = 3D Hand Skeleton Visualizer, Right side = Telemetry Sidebar
        left_pane = tk.Frame(container, bg="#121820")
        left_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        right_pane = tk.Frame(container, bg="#121820", width=420)
        right_pane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(8, 0))
        right_pane.pack_propagate(False)

        # 1. 3D Hand Skeleton Visualizer with MPU-9250 Orientation Tracking (+Y pointing away from fingers)
        self.hand_visualizer = HandVisualizerWidget(
            left_pane,
            receiver=self.receiver,
            on_cal_flat=self._cal_set_flat,
            on_cal_bent=self._cal_set_bent,
            on_cal_reset=self._cal_reset
        )
        self.hand_visualizer.pack(fill=tk.BOTH, expand=True)

        # 2. Right Pane: Detailed Telemetry Bars & Diagnostics
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        self.l_bars = []
        self.l_bar_labels = []
        self.r_bars = []
        self.r_bar_labels = []

        # Left Hand Panel
        self.l_hand_frame = tk.LabelFrame(
            right_pane,
            text=" Left Glove Telemetry (ADC & IMU) ",
            font=("Segoe UI", 9, "bold"),
            fg="#38bdf8",
            bg="#1a2332",
            padx=10,
            pady=8
        )
        self.l_hand_frame.pack(fill=tk.X, pady=(0, 6))

        for fname in finger_names:
            row = tk.Frame(self.l_hand_frame, bg="#1a2332")
            row.pack(fill=tk.X, pady=2)
            lbl = tk.Label(row, text=f"{fname}: 0%", width=16, anchor=tk.W, font=("Segoe UI", 8), bg="#1a2332", fg="#f8fafc")
            lbl.pack(side=tk.LEFT)
            bar = ttk.Progressbar(row, orient=tk.HORIZONTAL, length=140, mode="determinate")
            bar.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            self.l_bars.append(bar)
            self.l_bar_labels.append(lbl)

        self.l_imu_label = tk.Label(self.l_hand_frame, text="IMU: Ax:0 Ay:0 Az:0 | Gx:0 Gy:0 Gz:0", font=("Consolas", 8), fg="#94a3b8", bg="#1a2332", justify=tk.LEFT)
        self.l_imu_label.pack(fill=tk.X, pady=(4, 0))

        self.l_orient_label = tk.Label(self.l_hand_frame, text="Orientation: Pitch: 0.0° | Roll: 0.0° | Yaw: 0.0°", font=("Consolas", 8), fg="#94a3b8", bg="#1a2332", justify=tk.LEFT)
        self.l_orient_label.pack(fill=tk.X, pady=(2, 0))

        self.l_palm_badge = tk.Label(self.l_hand_frame, text="PALM: PALM DOWN (Flat Rest)", font=("Segoe UI", 9, "bold"), fg="#38bdf8", bg="#0f172a", padx=6, pady=3, relief="groove")
        self.l_palm_badge.pack(fill=tk.X, pady=(3, 0))

        # Right Hand Panel
        self.r_hand_frame = tk.LabelFrame(
            right_pane,
            text=" Right Glove Telemetry (ADC & IMU) ",
            font=("Segoe UI", 9, "bold"),
            fg="#10b981",
            bg="#1a2332",
            padx=10,
            pady=8
        )
        self.r_hand_frame.pack(fill=tk.X, pady=6)

        for fname in finger_names:
            row = tk.Frame(self.r_hand_frame, bg="#1a2332")
            row.pack(fill=tk.X, pady=2)
            lbl = tk.Label(row, text=f"{fname}: 0%", width=16, anchor=tk.W, font=("Segoe UI", 8), bg="#1a2332", fg="#f8fafc")
            lbl.pack(side=tk.LEFT)
            bar = ttk.Progressbar(row, orient=tk.HORIZONTAL, length=140, mode="determinate")
            bar.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            self.r_bars.append(bar)
            self.r_bar_labels.append(lbl)

        self.r_imu_label = tk.Label(self.r_hand_frame, text="IMU: Ax:0 Ay:0 Az:0 | Gx:0 Gy:0 Gz:0", font=("Consolas", 8), fg="#94a3b8", bg="#1a2332", justify=tk.LEFT)
        self.r_imu_label.pack(fill=tk.X, pady=(4, 0))

        self.r_orient_label = tk.Label(self.r_hand_frame, text="Orientation: Pitch: 0.0° | Roll: 0.0° | Yaw: 0.0°", font=("Consolas", 8), fg="#94a3b8", bg="#1a2332", justify=tk.LEFT)
        self.r_orient_label.pack(fill=tk.X, pady=(2, 0))

        self.r_palm_badge = tk.Label(self.r_hand_frame, text="PALM: PALM DOWN (Flat Rest)", font=("Segoe UI", 9, "bold"), fg="#10b981", bg="#0f172a", padx=6, pady=3, relief="groove")
        self.r_palm_badge.pack(fill=tk.X, pady=(3, 0))

        # Calibration Controls Panel in Sidebar
        cal_panel = tk.LabelFrame(right_pane, text=" Dual Glove ADC Calibration ", font=("Segoe UI", 9, "bold"), fg="#fbbf24", bg="#1e293b", padx=10, pady=8)
        cal_panel.pack(fill=tk.X, pady=(6, 0))

        cal_btns = tk.Frame(cal_panel, bg="#1e293b")
        tk.Button(cal_btns, text="Right Flat", bg="#0284c7", fg="white", font=("Segoe UI", 8, "bold"), command=lambda: self._cal_set_flat("RIGHT")).pack(side=tk.LEFT, padx=2)
        tk.Button(cal_btns, text="Right Fist", bg="#e11d48", fg="white", font=("Segoe UI", 8, "bold"), command=lambda: self._cal_set_bent("RIGHT")).pack(side=tk.LEFT, padx=2)
        tk.Button(cal_btns, text="Both Flat", bg="#0369a1", fg="white", font=("Segoe UI", 8), command=lambda: self._cal_set_flat(None)).pack(side=tk.LEFT, padx=2)
        tk.Button(cal_btns, text="Both Fist", bg="#be123c", fg="white", font=("Segoe UI", 8), command=lambda: self._cal_set_bent(None)).pack(side=tk.LEFT, padx=2)
        tk.Button(cal_btns, text="Reset", bg="#475569", fg="white", font=("Segoe UI", 8), command=lambda: self._cal_reset(None)).pack(side=tk.LEFT, padx=2)

        self.lbl_cal_status = tk.Label(cal_panel, text="Status: Ready", fg="#94a3b8", bg="#1e293b", font=("Segoe UI", 8))
        self.lbl_cal_status.pack(fill=tk.X, pady=(4, 0))

    # =========================================================================
    # TAB 3: Dataset Studio
    # =========================================================================
    def _build_dataset_tab(self, parent):
        container = tk.Frame(parent, bg="#121820")
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        top_controls = tk.Frame(container, bg="#1a2332", padx=15, pady=12, bd=1, relief="ridge")
        top_controls.pack(fill=tk.X, pady=(0, 10))

        # Row 0: Target Gesture & Hand Mode Selection
        row0 = tk.Frame(top_controls, bg="#1a2332")
        row0.pack(fill=tk.X, pady=(0, 6))

        tk.Label(row0, text="Target ASL Gesture:", font=("Segoe UI", 10, "bold"), bg="#1a2332", fg="#f8fafc").pack(side=tk.LEFT)
        self.cmb_gestures = ttk.Combobox(row0, values=config.DEFAULT_GESTURES, width=16, font=("Segoe UI", 10))
        self.cmb_gestures.set("A")
        self.cmb_gestures.pack(side=tk.LEFT, padx=(6, 4))
        self.cmb_gestures.bind("<<ComboboxSelected>>", self._on_gesture_combo_change)

        self.lbl_gesture_type = tk.Label(row0, text="[STATIC POSE]", font=("Segoe UI", 8, "bold"), fg="#38bdf8", bg="#032b43", padx=8, pady=3, relief="groove")
        self.lbl_gesture_type.pack(side=tk.LEFT, padx=(4, 15))

        tk.Label(row0, text="Hand Mode:", font=("Segoe UI", 9, "bold"), bg="#1a2332", fg="#fbbf24").pack(side=tk.LEFT, padx=(10, 6))
        tk.Radiobutton(row0, text="🖐 Right Hand (14F)", variable=self.dataset_hand_mode_var, value="right", bg="#1a2332", fg="#f8fafc", selectcolor="#0f172a", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=4)
        tk.Radiobutton(row0, text="🖐 Left Hand (14F)", variable=self.dataset_hand_mode_var, value="left", bg="#1a2332", fg="#f8fafc", selectcolor="#0f172a", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=4)
        tk.Radiobutton(row0, text="👐 Dual Gloves (28F)", variable=self.dataset_hand_mode_var, value="both", bg="#1a2332", fg="#f8fafc", selectcolor="#0f172a", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=4)

        # Row 1: Dual Recording Controls (Static 50+ Burst vs Dynamic Motion Stream)
        row1 = tk.Frame(top_controls, bg="#1a2332")
        row1.pack(fill=tk.X, pady=(4, 6))

        # Static Pose 50+ Burst Controls
        static_box = tk.LabelFrame(row1, text=" Static Pose Capture (Spatial Invariance) ", font=("Segoe UI", 8, "bold"), fg="#38bdf8", bg="#0f172a", padx=8, pady=6)
        static_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        self.btn_burst_snapshot = tk.Button(
            static_box,
            text="📸 Burst Capture (50+ Snapshots)",
            bg="#0284c7",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=4,
            command=self._start_static_burst_recording
        )
        self.btn_burst_snapshot.pack(side=tk.LEFT, padx=3)

        tk.Label(static_box, text="Count:", font=("Segoe UI", 9), bg="#0f172a", fg="#94a3b8").pack(side=tk.LEFT, padx=(6, 2))
        self.spn_burst_count = ttk.Spinbox(static_box, from_=10, to=200, increment=10, width=5)
        self.spn_burst_count.set("50")
        self.spn_burst_count.pack(side=tk.LEFT, padx=3)

        self.btn_snapshot = tk.Button(
            static_box,
            text="+ 1 Frame",
            bg="#334155",
            fg="white",
            font=("Segoe UI", 8),
            padx=6,
            pady=3,
            command=self._record_single_snapshot
        )
        self.btn_snapshot.pack(side=tk.LEFT, padx=(6, 2))

        # Dynamic Motion Stream Controls
        dynamic_box = tk.LabelFrame(row1, text=" Dynamic Gesture Capture (Trajectory) ", font=("Segoe UI", 8, "bold"), fg="#fbbf24", bg="#0f172a", padx=8, pady=6)
        dynamic_box.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0))

        self.btn_record = tk.Button(
            dynamic_box,
            text="⏺ Record Stream",
            bg="#e11d48",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=4,
            command=self._start_recording_session
        )
        self.btn_record.pack(side=tk.LEFT, padx=3)

        tk.Label(dynamic_box, text="Duration (s):", font=("Segoe UI", 9), bg="#0f172a", fg="#94a3b8").pack(side=tk.LEFT, padx=(6, 2))
        self.spn_duration = ttk.Spinbox(dynamic_box, from_=1.0, to=10.0, increment=0.5, width=5)
        self.spn_duration.set("3.0")
        self.spn_duration.pack(side=tk.LEFT, padx=3)

        # Row 2: Live Spatial Guidance Banner
        self.lbl_capture_guidance = tk.Label(
            top_controls,
            text="💡 Static Pose Tip: Hold 'A' handshape and click '📸 Burst Capture (50+)'. Slowly move hand in 3D space during capture for position invariance!",
            font=("Segoe UI", 9, "italic"),
            fg="#38bdf8",
            bg="#0f172a",
            padx=10,
            pady=5,
            relief="groove",
            wraplength=850,
            justify=tk.LEFT
        )
        self.lbl_capture_guidance.pack(fill=tk.X, pady=(4, 6))

        # Progress bar
        self.rec_prog_bar = ttk.Progressbar(top_controls, orient=tk.HORIZONTAL, mode="determinate")
        self.rec_prog_bar.pack(fill=tk.X, pady=(2, 4))

        self.lbl_rec_status = tk.Label(top_controls, text="Status: Ready to capture", font=("Segoe UI", 9), bg="#1a2332", fg="#94a3b8")
        self.lbl_rec_status.pack(anchor=tk.W)

        # Dataset Table & Stats
        bot_frame = tk.Frame(container, bg="#121820")
        bot_frame.pack(fill=tk.BOTH, expand=True)

        list_frame = tk.LabelFrame(bot_frame, text=" Dataset Summary (Class Distribution) ", font=("Segoe UI", 10, "bold"), fg="#38bdf8", bg="#1a2332", padx=10, pady=10)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.tree_dataset = ttk.Treeview(list_frame, columns=("Gesture", "Samples"), show="headings", height=12)
        self.tree_dataset.heading("Gesture", text="Gesture Class")
        self.tree_dataset.heading("Samples", text="Sample Count")
        self.tree_dataset.column("Gesture", width=180)
        self.tree_dataset.column("Samples", width=100)
        self.tree_dataset.tag_configure("black_text", foreground="#000000", background="#ffffff")
        self.tree_dataset.pack(fill=tk.BOTH, expand=True)

        manage_col = tk.Frame(bot_frame, bg="#121820", width=220)
        manage_col.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        tk.Button(manage_col, text="🔄 Refresh Table", bg="#1e293b", fg="white", font=("Segoe UI", 9), width=22, command=self._refresh_dataset_table).pack(pady=4)
        tk.Button(manage_col, text="⚡ Generate Bootstrap Data", bg="#0284c7", fg="white", font=("Segoe UI", 9, "bold"), width=22, command=self._generate_bootstrap).pack(pady=4)
        tk.Button(manage_col, text="🗑 Delete Selected Gesture", bg="#b91c1c", fg="white", font=("Segoe UI", 9), width=22, command=self._delete_selected_gesture).pack(pady=4)
        tk.Button(manage_col, text="⚠ Wipe Entire Dataset", bg="#7f1d1d", fg="white", font=("Segoe UI", 9), width=22, command=self._wipe_dataset).pack(pady=4)

        # Dataset Metadata Info Badge
        self.lbl_dataset_mode_info = tk.Label(
            manage_col,
            text="Dataset: Loading...",
            font=("Segoe UI", 8),
            fg="#38bdf8",
            bg="#121820",
            wraplength=200,
            justify=tk.LEFT
        )
        self.lbl_dataset_mode_info.pack(pady=(10, 4))

        self._refresh_dataset_table()

    # =========================================================================
    # TAB 4: Model Training
    # =========================================================================
    def _build_train_tab(self, parent):
        container = tk.Frame(parent, bg="#121820")
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        card = tk.Frame(container, bg="#1a2332", padx=20, pady=16, bd=1, relief="ridge")
        card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(card, text="CLASSIFIER CONFIGURATION", font=("Segoe UI", 11, "bold"), fg="#38bdf8", bg="#1a2332").pack(anchor=tk.W, pady=(0, 10))

        opt_row = tk.Frame(card, bg="#1a2332")
        opt_row.pack(fill=tk.X, pady=4)

        tk.Label(opt_row, text="Algorithm:", font=("Segoe UI", 10), bg="#1a2332", fg="#f8fafc").pack(side=tk.LEFT)
        self.model_type_var = tk.StringVar(value="rf")
        tk.Radiobutton(opt_row, text="Random Forest", variable=self.model_type_var, value="rf", bg="#1a2332", fg="white", selectcolor="#0f172a").pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(opt_row, text="MLP Neural Net", variable=self.model_type_var, value="mlp", bg="#1a2332", fg="white", selectcolor="#0f172a").pack(side=tk.LEFT, padx=10)

        # Hand Mode Training Selector
        train_mode_row = tk.Frame(card, bg="#1a2332")
        train_mode_row.pack(fill=tk.X, pady=6)

        tk.Label(train_mode_row, text="Hand Mode:", font=("Segoe UI", 10, "bold"), bg="#1a2332", fg="#38bdf8").pack(side=tk.LEFT)
        tk.Radiobutton(train_mode_row, text="Auto-Detect from Data", variable=self.train_hand_mode_var, value="auto", bg="#1a2332", fg="white", selectcolor="#0f172a").pack(side=tk.LEFT, padx=6)
        tk.Radiobutton(train_mode_row, text="Right Hand Only (14 Feat)", variable=self.train_hand_mode_var, value="right", bg="#1a2332", fg="white", selectcolor="#0f172a").pack(side=tk.LEFT, padx=6)
        tk.Radiobutton(train_mode_row, text="Left Hand Only (14 Feat)", variable=self.train_hand_mode_var, value="left", bg="#1a2332", fg="white", selectcolor="#0f172a").pack(side=tk.LEFT, padx=6)
        tk.Radiobutton(train_mode_row, text="Dual Gloves (28 Feat)", variable=self.train_hand_mode_var, value="dual", bg="#1a2332", fg="white", selectcolor="#0f172a").pack(side=tk.LEFT, padx=6)

        # Spatial Invariance Checkbox Row
        spatial_row = tk.Frame(card, bg="#1a2332")
        spatial_row.pack(fill=tk.X, pady=(6, 4))

        self.spatial_invariance_var = tk.BooleanVar(value=True)
        self.chk_spatial = tk.Checkbutton(
            spatial_row,
            text=" 🌐 Spatial Translation & Position Invariance (Augment 3D Position, Motion & Tilt)",
            variable=self.spatial_invariance_var,
            bg="#1a2332",
            fg="#38bdf8",
            selectcolor="#0f172a",
            font=("Segoe UI", 9, "bold")
        )
        self.chk_spatial.pack(side=tk.LEFT)

        tk.Label(
            card,
            text="ℹ Augments training dataset with simulated 3D spatial translations, gravity tilts (±15°), and magnetic field variations so hand poses are detected anywhere in space.",
            font=("Segoe UI", 8, "italic"),
            fg="#94a3b8",
            bg="#1a2332",
            wraplength=850,
            justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(0, 8))

        self.btn_train = tk.Button(
            card,
            text="🚀 Train ASL Recognition Model",
            bg="#2563eb",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            padx=16,
            pady=8,
            command=self._start_training
        )
        self.btn_train.pack(anchor=tk.W, pady=(4, 8))

        # Metrics Display Box
        metrics_box = tk.LabelFrame(container, text=" Training Evaluation Results ", font=("Segoe UI", 10, "bold"), fg="#10b981", bg="#1a2332", padx=15, pady=12)
        metrics_box.pack(fill=tk.BOTH, expand=True)

        self.lbl_train_acc = tk.Label(metrics_box, text="Test Accuracy: Not Trained Yet", font=("Segoe UI", 14, "bold"), fg="#10b981", bg="#1a2332")
        self.lbl_train_acc.pack(anchor=tk.W, pady=4)

        self.txt_train_report = tk.Text(metrics_box, bg="#0f172a", fg="#f8fafc", font=("Consolas", 10), relief="flat")
        self.txt_train_report.pack(fill=tk.BOTH, expand=True, pady=6)


    # =========================================================================
    # Background Refresh & Inference Loop
    # =========================================================================
    def _telemetry_refresh_loop(self):
        try:
            l_state, r_state = self.receiver.get_latest_states()

            # Update Header Badges & Connection Bar Inbuilt LEDs
            if l_state["connected"]:
                l_mode = self.receiver.connection_modes.get("LEFT", "WIFI")
                l_tag = "USB" if l_mode == "SERIAL" else "Wi-Fi"
                self.left_indicator.config(text=f"● LEFT GLOVE [{l_tag}]: {self.receiver.left_fps:.0f} Hz", fg="#10b981")
                self.led_left.config(text="● LED: ON", fg="#38bdf8", bg="#032b43")
                self.lbl_left_conn_status.config(text=f"Active [{l_tag}] ({self.receiver.left_fps:.0f} Hz)", fg="#10b981")
                if hasattr(self, "btn_left_connect"):
                    self.btn_left_connect.config(text="✕ Disconnect", bg="#b91c1c")
            else:
                self.left_indicator.config(text="● LEFT GLOVE: OFF", fg="#ef4444")
                self.led_left.config(text="● LED: OFF", fg="#64748b", bg="#0f172a")
                if hasattr(self, "btn_left_connect") and str(self.btn_left_connect["state"]) != "disabled":
                    self.btn_left_connect.config(text="🔗 Connect", bg="#0284c7")
                curr_l = self.lbl_left_conn_status.cget("text")
                if not any(k in curr_l for k in ["Pinging", "Connecting", "Failed", "Standby"]):
                    self.lbl_left_conn_status.config(text="Standby", fg="#94a3b8")

            if r_state["connected"]:
                r_mode = self.receiver.connection_modes.get("RIGHT", "WIFI")
                r_tag = "USB" if r_mode == "SERIAL" else "Wi-Fi"
                self.right_indicator.config(text=f"● RIGHT GLOVE [{r_tag}]: {self.receiver.right_fps:.0f} Hz", fg="#10b981")
                self.led_right.config(text="● LED: ON", fg="#10b981", bg="#064e3b")
                self.lbl_right_conn_status.config(text=f"Active [{r_tag}] ({self.receiver.right_fps:.0f} Hz)", fg="#10b981")
                if hasattr(self, "btn_right_connect"):
                    self.btn_right_connect.config(text="✕ Disconnect", bg="#b91c1c")
            else:
                self.right_indicator.config(text="● RIGHT GLOVE: OFF", fg="#ef4444")
                self.led_right.config(text="● LED: OFF", fg="#64748b", bg="#0f172a")
                if hasattr(self, "btn_right_connect") and str(self.btn_right_connect["state"]) != "disabled":
                    self.btn_right_connect.config(text="🔗 Connect", bg="#059669")
                curr_r = self.lbl_right_conn_status.cget("text")
                if not any(k in curr_r for k in ["Pinging", "Connecting", "Failed", "Standby"]):
                    self.lbl_right_conn_status.config(text="Standby", fg="#94a3b8")

            # Update Telemetry Bars
            finger_abbr = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            for i in range(5):
                l_val = int(l_state["flex_norm"][i] * 100)
                l_raw = l_state["flex_raw"][i]
                self.l_bars[i]["value"] = l_val
                self.l_bar_labels[i].config(text=f"{finger_abbr[i]}: {l_val}% ({l_raw})")

                r_val = int(r_state["flex_norm"][i] * 100)
                r_raw = r_state["flex_raw"][i]
                self.r_bars[i]["value"] = r_val
                self.r_bar_labels[i].config(text=f"{finger_abbr[i]}: {r_val}% ({r_raw})")

            lax, lay, laz = l_state["accel"]
            lgx, lgy, lgz = l_state["gyro"]
            self.l_imu_label.config(text=f"Ax:{lax} Ay:{lay} Az:{laz} | Gx:{lgx} Gy:{lgy} Gz:{lgz}")

            rax, ray, raz = r_state["accel"]
            rgx, rgy, rgz = r_state["gyro"]
            self.r_imu_label.config(text=f"Ax:{rax} Ay:{ray} Az:{raz} | Gx:{rgx} Gy:{rgy} Gz:{rgz}")

            # Update live orientation text & semantic palm direction labels
            lp = l_state.get("pitch_deg", 0.0)
            lr = l_state.get("roll_deg", 0.0)
            ly = l_state.get("yaw_deg", 0.0)
            l_palm = l_state.get("palm_orientation", "UNKNOWN")
            self.l_orient_label.config(text=f"Orientation: Pitch:{lp:+.1f}° | Roll:{lr:+.1f}° | Yaw:{ly:+.1f}°")
            self.l_palm_badge.config(text=f"PALM: {l_palm}")

            rp = r_state.get("pitch_deg", 0.0)
            rr = r_state.get("roll_deg", 0.0)
            ry = r_state.get("yaw_deg", 0.0)
            r_palm = r_state.get("palm_orientation", "UNKNOWN")
            self.r_orient_label.config(text=f"Orientation: Pitch:{rp:+.1f}° | Roll:{rr:+.1f}° | Yaw:{ry:+.1f}°")
            self.r_palm_badge.config(text=f"PALM: {r_palm}")

            # Update Model Badge in Header & Tab 1
            if self.recognizer.is_loaded:
                m_mode_str = self.recognizer.hand_mode.upper()
                n_f = self.recognizer.expected_features
                n_c = len(self.recognizer.label_encoder.classes_) if self.recognizer.label_encoder else 0
                self.lbl_model_indicator.config(text=f"● MODEL: {m_mode_str} ({n_f}F)", fg="#10b981")
                if hasattr(self, "lbl_model_status_badge"):
                    self.lbl_model_status_badge.config(
                        text=f"Active Model: {m_mode_str} ({n_f} Feat) | {n_c} Classes Trained",
                        fg="#38bdf8"
                    )
            else:
                self.lbl_model_indicator.config(text="● MODEL: UNTRAINED", fg="#ef4444")
                if hasattr(self, "lbl_model_status_badge"):
                    self.lbl_model_status_badge.config(text="Active Model: Not Loaded (Train model in Tab 4)", fg="#ef4444")

            # Update 3D Hand Skeleton Visualizer (MPU-9250 orientation + flex kinematics)
            if hasattr(self, "hand_visualizer"):
                self.hand_visualizer.update_frame(l_state, r_state)

            # Real-Time Gesture Inference (Gated: checks glove connectivity according to active model hand)
            model_hand = self.recognizer.hand_mode if self.recognizer.is_loaded else "right"
            if model_hand == "right":
                is_glove_active = r_state["connected"] or self.receiver.simulation_mode
                wait_text = "WAITING FOR RIGHT GLOVE"
            elif model_hand == "left":
                is_glove_active = l_state["connected"] or self.receiver.simulation_mode
                wait_text = "WAITING FOR LEFT GLOVE"
            else:
                is_glove_active = (l_state["connected"] or r_state["connected"]) or self.receiver.simulation_mode
                wait_text = "WAITING FOR GLOVES (DISCONNECTED)"

            if self.live_recognition_active.get() and self.recognizer.is_loaded and is_glove_active:
                vec = self.receiver.get_synchronized_feature_vector(hand_mode=model_hand)
                motion_meta = self.receiver.get_motion_metrics(hand_mode=model_hand)
                label, conf, is_new_event, sign_meta = self.recognizer.process_live_stream(vec, motion_meta)

                # Update live kinematics & classification badges
                if hasattr(self, "lbl_gesture_type"):
                    g_type = sign_meta.get("gesture_type", "NONE")
                    if g_type != "NONE":
                        self.lbl_gesture_type.config(
                            text=f"TYPE: {g_type}",
                            fg="#c084fc" if sign_meta.get("is_dynamic") else "#38bdf8"
                        )
                    else:
                        self.lbl_gesture_type.config(text="TYPE: WAITING ✋", fg="#64748b")

                if hasattr(self, "lbl_motion_state"):
                    vel = sign_meta.get("velocity_dps", 0.0)
                    m_state = sign_meta.get("motion_state", "STEADY")
                    is_mov = "MOTION" in m_state
                    self.lbl_motion_state.config(
                        text=f"KINEMATICS: {m_state} ({vel:.1f}°/s)",
                        fg="#f59e0b" if is_mov else "#10b981"
                    )

                if hasattr(self, "lbl_latch_status"):
                    if sign_meta.get("is_latched"):
                        self.lbl_latch_status.config(text="LATCH: COMMITTED (NO REPEAT)", fg="#fbbf24")
                    else:
                        self.lbl_latch_status.config(text="LATCH: READY FOR NEW SIGN", fg="#10b981")

                if label and label not in ["IDLE", "REST"]:
                    self.lbl_current_sign.config(text=f"SIGN: {label}")
                    self.lbl_confidence.config(text=f"Confidence: {conf*100:.1f}%")
                    self.prob_bar["value"] = int(conf * 100)
                elif label in ["IDLE", "REST"]:
                    self.lbl_current_sign.config(text="RESTING (IDLE)")
                    self.lbl_confidence.config(text=f"Confidence: {conf*100:.1f}%")
                    self.prob_bar["value"] = int(conf * 100)
                else:
                    self.lbl_current_sign.config(text="LISTENING...")
                    self.lbl_confidence.config(text="Confidence: 0.0%")
                    self.prob_bar["value"] = 0

                if is_new_event and label and label not in ["IDLE", "REST"]:
                    self._on_gesture_event(label)

                is_user_actively_signing = bool((label and label not in ["IDLE", "REST"]) or is_mov)
                self._check_auto_clear_timer(is_user_actively_signing)
            else:
                # Gloves are not streaming or recognition paused
                if not is_glove_active:
                    self.lbl_current_sign.config(text=wait_text)
                else:
                    self.lbl_current_sign.config(text="PAUSED")
                self.lbl_confidence.config(text="Confidence: 0.0%")
                self.prob_bar["value"] = 0
                if hasattr(self, "lbl_latch_status"):
                    self.lbl_latch_status.config(text="LATCH: IDLE", fg="#64748b")
                if hasattr(self, "lbl_motion_state"):
                    self.lbl_motion_state.config(text="KINEMATICS: IDLE", fg="#64748b")
                if hasattr(self, "lbl_gesture_type"):
                    self.lbl_gesture_type.config(text="TYPE: IDLE", fg="#64748b")
                self.recognizer.reset_state()
                self._check_auto_clear_timer(is_user_actively_signing=False)


        except Exception as e:
            logger.debug(f"Refresh loop exception: {e}")

        self.after(50, self._telemetry_refresh_loop) # ~20Hz UI refresh

    def _on_gesture_event(self, sign: str):
        """Called when a stable gesture is confirmed."""
        sentence = self.builder.add_token(sign)
        self.txt_english_sentence.delete("1.0", tk.END)
        self.txt_english_sentence.insert("1.0", sentence)
        self.last_sentence_activity_time = time.time()
        self._trigger_translation()

    # =========================================================================
    # Translation & TTS Handlers
    # =========================================================================
    def _trigger_translation(self):
        self.last_sentence_activity_time = time.time()
        eng_text = self.txt_english_sentence.get("1.0", tk.END).strip()
        if not eng_text:
            return

        target_lang = self.target_lang_var.get()
        result = self.translator.translate(eng_text, target_lang)

        trans_text = result["translation"]
        phonetic = result["phonetic"]
        engine = result["engine"]

        self.lbl_translation_output.config(text=trans_text)
        self.lbl_phonetics.config(text=f"Pronunciation: {phonetic}")
        self.lbl_engine_info.config(text=f"Engine: {engine} | Target: {target_lang}")

        if self.auto_speak_var.get():
            self._speak_nigerian()

    def _toggle_neural_model(self):
        """Toggle between N-Atlas Authentic Lexicon and Deep Neural Transformer."""
        if self.neural_mode_var.get():
            if not self.translator.neural_ready:
                self.lbl_engine_info.config(text="Engine: Loading N-Atlas Neural Model in background...", fg="#f59e0b")
                self.translator.load_neural_model_async()
                def _check_ready():
                    if self.translator.neural_ready:
                        self.lbl_engine_info.config(text="Engine: N-Atlas Neural Transformer (Active)", fg="#10b981")
                        self._trigger_translation()
                    elif self.translator.neural_loading:
                        self.after(1000, _check_ready)
                    else:
                        self.lbl_engine_info.config(text="Engine: N-Atlas Lexicon (Neural load failed/offline)", fg="#94a3b8")
                self.after(1000, _check_ready)
            else:
                self.lbl_engine_info.config(text="Engine: N-Atlas Neural Transformer (Active)", fg="#10b981")
                self._trigger_translation()
        else:
            self.lbl_engine_info.config(text="Engine: N-Atlas Authentic Lexicon (Active)", fg="#94a3b8")
            self._trigger_translation()

    def _show_natlas_model_info(self):
        """Displays modal explaining how N-Atlas is included and its architecture."""
        info_win = tk.Toplevel(self)
        info_win.title("N-Atlas Multilingual Model Details & Architecture")
        info_win.geometry("640x520")
        info_win.configure(bg="#0f172a")
        info_win.resizable(False, False)
        info_win.transient(self)
        info_win.grab_set()

        card = tk.Frame(info_win, bg="#1a2332", padx=20, pady=16, bd=2, relief="groove")
        card.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        tk.Label(
            card,
            text="🇳🇬 N-ATLAS TRANSLATION MODEL INTEGRATION",
            font=("Segoe UI", 13, "bold"),
            fg="#fbbf24",
            bg="#1a2332"
        ).pack(pady=(0, 6))

        details = self.translator.get_model_details()
        status_line = f"Active Engine: {details['active_engine']} | Neural Checkpoint: {details['neural_checkpoint']}"
        tk.Label(card, text=status_line, font=("Segoe UI", 9, "bold"), fg="#10b981", bg="#1a2332").pack(pady=(0, 8))

        doc_text = (
            "1. WHAT IS NCAIR1/N-ATLaS?\n"
            "   N-ATLaS is Nigeria's flagship Sovereign Multilingual AI Model released by the\n"
            "   National Centre for AI and Robotics (NCAIR) in collaboration with Awarri AI.\n"
            "   • Hugging Face Repository: https://huggingface.co/NCAIR1/N-ATLaS\n"
            "   • Base Architecture: Meta Llama-3-8B Causal LM (8 Billion Parameters)\n"
            "   • Pre-training & SFT: 392M+ tokens across English, Hausa, Igbo, and Yoruba\n"
            "   • Context Length: 8,092 tokens | Vocabulary Size: 128,256 tokens\n\n"
            "2. HOW IT IS INTEGRATED IN THIS CODEBASE (software/translator.py):\n"
            "   The project provides a resilient Multi-Tier Nigerian translation pipeline:\n"
            "   • TIER 1 - Ultra-Fast Offline Tonal Lexicon:\n"
            "     Curated high-accuracy bilingual repository with authentic tonal diacritics.\n"
            "     Runs instantly at 0ms latency with zero GPU or network requirements.\n"
            "   • TIER 2 - N-ATLaS 8B Hugging Face Cloud Inference API:\n"
            "     Instant serverless GPU inference via huggingface_hub using your free HF token.\n"
            "     Allows full 8B LLM translation without downloading 16 GB of weights locally!\n"
            "   • TIER 3 - Local Offline Model Weights (data/models/natlas/):\n"
            "     Can be downloaded using 'python software/download_natlas_model.py'.\n\n"
            "3. TONAL DIACRITIC PRESERVATION:\n"
            "   Yoruba and Igbo are tonal languages where accents fundamentally alter meaning:\n"
            "   (e.g., in Yoruba: 'ọwọ́' = hand, 'owó' = money, 'ọ̀wọ̀' = respect).\n"
            "   N-ATLaS preserves authentic high (´), low (`), and subdots (ẹ, ọ, ṣ, ụ, ị, ṅ)!\n\n"
            "4. SPEECH SYNTHESIS (TTS):\n"
            "   The translation output feeds directly into the Indigenous TTS engine\n"
            "   to speak authentic audio for Yoruba, Igbo, and Hausa listeners."
        )

        txt_box = tk.Text(card, bg="#0f172a", fg="#f8fafc", font=("Consolas", 9), relief="sunken", padx=10, pady=10, wrap=tk.WORD)
        txt_box.insert("1.0", doc_text)
        txt_box.config(state=tk.DISABLED)
        txt_box.pack(fill=tk.BOTH, expand=True, pady=4)

        tk.Button(
            card,
            text="Close",
            font=("Segoe UI", 10, "bold"),
            bg="#0284c7",
            fg="white",
            padx=16,
            pady=4,
            command=info_win.destroy
        ).pack(pady=(8, 0))

    def _prompt_set_hf_token(self):
        """Displays dialog allowing user to configure their Hugging Face API token for online N-ATLaS inference."""
        token_win = tk.Toplevel(self)
        token_win.title("Hugging Face Online API Token Setup")
        token_win.geometry("580x440")
        token_win.configure(bg="#0f172a")
        token_win.resizable(False, False)
        token_win.transient(self)
        token_win.grab_set()

        card = tk.Frame(token_win, bg="#1a2332", padx=20, pady=16, bd=2, relief="groove")
        card.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        tk.Label(
            card,
            text="☁️ N-ATLaS ONLINE CLOUD INFERENCE SETUP",
            font=("Segoe UI", 12, "bold"),
            fg="#38bdf8",
            bg="#1a2332"
        ).pack(pady=(0, 6))

        tk.Label(
            card,
            text="Run N-ATLaS 8B on cloud GPUs without downloading 16 GB of weights!",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#1a2332"
        ).pack(pady=(0, 10))

        # Guidance Box
        guide_box = tk.Frame(card, bg="#0f172a", padx=12, pady=10, bd=1, relief="ridge")
        guide_box.pack(fill=tk.X, pady=(0, 10))

        steps = (
            "1. Visit https://huggingface.co/NCAIR1/N-ATLaS while logged in.\n"
            "   Click 'Acknowledge' to accept the model license (required once).\n\n"
            "2. Generate a FREE user access token at:\n"
            "   https://huggingface.co/settings/tokens (Role: 'Read').\n\n"
            "3. Paste your token (starts with 'hf_...') into the box below:"
        )
        tk.Label(guide_box, text=steps, font=("Segoe UI", 8, "bold"), fg="#f8fafc", bg="#0f172a", justify=tk.LEFT).pack(anchor=tk.W)

        # Existing Token Check
        token_file = config.DATA_DIR / "hf_token.txt"
        existing_token = ""
        if token_file.exists():
            try:
                existing_token = token_file.read_text(encoding="utf-8").strip()
            except Exception:
                pass

        lbl_status = tk.Label(
            card,
            text=f"Current Status: {'✓ Active (' + existing_token[:6] + '...)' if existing_token else '✕ No Token Configured'}",
            font=("Segoe UI", 9, "bold"),
            fg="#10b981" if existing_token else "#f59e0b",
            bg="#1a2332"
        )
        lbl_status.pack(anchor=tk.W, pady=(0, 6))

        entry_frame = tk.Frame(card, bg="#1a2332")
        entry_frame.pack(fill=tk.X, pady=(0, 12))

        tk.Label(entry_frame, text="HF Token:", font=("Segoe UI", 9, "bold"), fg="#cbd5e1", bg="#1a2332").pack(side=tk.LEFT, padx=(0, 6))
        txt_token = tk.Entry(entry_frame, font=("Segoe UI", 10), bg="#0f172a", fg="#f8fafc", insertbackground="white", width=36)
        if existing_token:
            txt_token.insert(0, existing_token)
        txt_token.pack(side=tk.LEFT, fill=tk.X, expand=True)

        def _save_and_test():
            tok = txt_token.get().strip()
            if not tok:
                messagebox.showwarning("Empty Token", "Please paste your Hugging Face token first.", parent=token_win)
                return

            lbl_status.config(text="Connecting to Hugging Face Cloud...", fg="#38bdf8")
            token_win.update_idletasks()

            success = self.translator.set_hf_token(tok)
            if success:
                lbl_status.config(text=f"✓ Token Activated ({tok[:6]}...)", fg="#10b981")
                self.neural_mode_var.set(True)
                self.lbl_engine_info.config(text="Engine: N-ATLaS Cloud API (Active ☁️)", fg="#10b981")
                messagebox.showinfo(
                    "Token Saved & Cloud API Ready",
                    f"Successfully connected to {config.NATLAS_MODEL_NAME}!\n\n"
                    "Online cloud translation is now active. When you construct sentences, "
                    "N-ATLaS will generate authentic translations on cloud GPUs without local downloads.",
                    parent=token_win
                )
                self._trigger_translation()
                token_win.destroy()
            else:
                lbl_status.config(text="✕ Connection Failed", fg="#ef4444")
                messagebox.showerror(
                    "Connection Failed",
                    "Could not initialize Hugging Face InferenceClient with that token.\n\n"
                    "Please verify:\n"
                    "1. You accepted the license on https://huggingface.co/NCAIR1/N-ATLaS\n"
                    "2. The token has 'Read' permission on Hugging Face.",
                    parent=token_win
                )

        btn_row = tk.Frame(card, bg="#1a2332")
        btn_row.pack(fill=tk.X, pady=(6, 0))

        tk.Button(
            btn_row,
            text="💾 Save & Activate Online Engine",
            font=("Segoe UI", 9, "bold"),
            bg="#0284c7",
            fg="white",
            padx=12,
            pady=4,
            command=_save_and_test
        ).pack(side=tk.LEFT)

        tk.Button(
            btn_row,
            text="Cancel",
            font=("Segoe UI", 9),
            bg="#334155",
            fg="white",
            padx=12,
            pady=4,
            command=token_win.destroy
        ).pack(side=tk.RIGHT)

    def _speak_english(self):
        text = self.txt_english_sentence.get("1.0", tk.END).strip()
        if text:
            self.last_sentence_activity_time = time.time()
            self.tts.speak(text, "English")

    def _speak_nigerian(self):
        text = self.lbl_translation_output.cget("text")
        if text and text != "[Translation will appear here]":
            self.last_sentence_activity_time = time.time()
            self.tts.speak(text, self.target_lang_var.get())

    def _action_backspace(self):
        self.last_sentence_activity_time = time.time()
        new_text = self.builder.backspace()
        self.txt_english_sentence.delete("1.0", tk.END)
        self.txt_english_sentence.insert("1.0", new_text)
        self._trigger_translation()

    def _action_clear_sentence(self):
        self.builder.clear()
        self.txt_english_sentence.delete("1.0", tk.END)
        self.lbl_translation_output.config(text="[Translation will appear here]")
        self.lbl_phonetics.config(text="Pronunciation: -")
        self.recognizer.reset_state()
        self.last_sentence_activity_time = time.time()
        if hasattr(self, "lbl_auto_clear_timer"):
            self.lbl_auto_clear_timer.config(text="")

    def _on_user_typing_event(self):
        """Called when user types directly into the English sentence box."""
        self.last_sentence_activity_time = time.time()
        if hasattr(self, "lbl_auto_clear_timer"):
            curr = self.lbl_auto_clear_timer.cget("text")
            if "Auto-cleared" in curr:
                self.lbl_auto_clear_timer.config(text="")

    def _auto_clear_sentence(self):
        """Automatically clears sentence and translations after idle timeout."""
        sentence_text = self.txt_english_sentence.get("1.0", tk.END).strip()
        if not sentence_text:
            return
        self.builder.clear()
        self.txt_english_sentence.delete("1.0", tk.END)
        self.lbl_translation_output.config(text="[Translation will appear here]")
        self.lbl_phonetics.config(text="Pronunciation: -")
        self.recognizer.reset_state()
        self.last_sentence_activity_time = time.time()
        if hasattr(self, "lbl_auto_clear_timer"):
            self.lbl_auto_clear_timer.config(text="✓ Auto-cleared (Ready for new sign)", fg="#10b981")
            self.after(2500, self._clear_auto_clear_badge)

    def _clear_auto_clear_badge(self):
        if hasattr(self, "lbl_auto_clear_timer"):
            curr = self.lbl_auto_clear_timer.cget("text")
            if "Auto-cleared" in curr:
                self.lbl_auto_clear_timer.config(text="")

    def _check_auto_clear_timer(self, is_user_actively_signing: bool):
        """Evaluates auto-clear timeout condition and updates UI countdown badge."""
        if not hasattr(self, "txt_english_sentence"):
            return
        sentence_text = self.txt_english_sentence.get("1.0", tk.END).strip()
        if not sentence_text:
            if hasattr(self, "lbl_auto_clear_timer"):
                curr = self.lbl_auto_clear_timer.cget("text")
                if "Auto-clear in" in curr or "Signing" in curr:
                    self.lbl_auto_clear_timer.config(text="")
            return

        if not self.auto_clear_enabled_var.get():
            if hasattr(self, "lbl_auto_clear_timer"):
                curr = self.lbl_auto_clear_timer.cget("text")
                if "Auto-clear in" in curr or "Signing" in curr:
                    self.lbl_auto_clear_timer.config(text="")
            return

        # If user is actively moving hands or holding a sign, reset idle clock
        if is_user_actively_signing:
            self.last_sentence_activity_time = time.time()
            if hasattr(self, "lbl_auto_clear_timer"):
                self.lbl_auto_clear_timer.config(text="● Signing...", fg="#38bdf8")
            return

        delay_str = self.auto_clear_delay_var.get().replace("s", "").strip()
        try:
            delay_sec = float(delay_str)
        except ValueError:
            delay_sec = 6.0

        idle_time = time.time() - self.last_sentence_activity_time
        remaining = delay_sec - idle_time

        if remaining <= 0:
            self._auto_clear_sentence()
        elif remaining <= 3.0:
            if hasattr(self, "lbl_auto_clear_timer"):
                self.lbl_auto_clear_timer.config(text=f"⏳ Auto-clear in {remaining:.1f}s", fg="#fbbf24")
        else:
            if hasattr(self, "lbl_auto_clear_timer"):
                self.lbl_auto_clear_timer.config(text=f"⏳ Auto-clear in {int(remaining)}s", fg="#94a3b8")

    # =========================================================================
    # Calibration Handlers
    # =========================================================================
    def _cal_set_flat(self, hand=None):
        l_s, r_s = self.receiver.get_latest_states()
        cal = self.receiver.calibration
        if hand == "LEFT":
            cal["LEFT"]["flex_flat"] = list(l_s["flex_raw"])
            msg = "Left Glove Flat (0°) Calibrated!"
        elif hand == "RIGHT":
            cal["RIGHT"]["flex_flat"] = list(r_s["flex_raw"])
            msg = "Right Glove Flat (0°) Calibrated!"
        else:
            cal["LEFT"]["flex_flat"] = list(l_s["flex_raw"])
            cal["RIGHT"]["flex_flat"] = list(r_s["flex_raw"])
            msg = "Both Gloves Flat (0°) Calibrated!"

        self.receiver.save_calibration(cal)
        self.lbl_cal_status.config(text=f"Status: {msg}", fg="#10b981")
        if hasattr(self, "hand_visualizer"):
            self.hand_visualizer.lbl_cal_status.config(text=f"Status: {msg}", fg="#10b981")
        return msg

    def _cal_set_bent(self, hand=None):
        l_s, r_s = self.receiver.get_latest_states()
        cal = self.receiver.calibration
        if hand == "LEFT":
            cal["LEFT"]["flex_bent"] = list(l_s["flex_raw"])
            msg = "Left Glove Closed Fist (90°) Calibrated!"
        elif hand == "RIGHT":
            cal["RIGHT"]["flex_bent"] = list(r_s["flex_raw"])
            msg = "Right Glove Closed Fist (90°) Calibrated!"
        else:
            cal["LEFT"]["flex_bent"] = list(l_s["flex_raw"])
            cal["RIGHT"]["flex_bent"] = list(r_s["flex_raw"])
            msg = "Both Gloves Closed Fist (90°) Calibrated!"

        self.receiver.save_calibration(cal)
        self.lbl_cal_status.config(text=f"Status: {msg}", fg="#10b981")
        if hasattr(self, "hand_visualizer"):
            self.hand_visualizer.lbl_cal_status.config(text=f"Status: {msg}", fg="#10b981")
        return msg

    def _cal_reset(self, hand=None):
        cal = self.receiver.load_default_calibration()
        self.receiver.save_calibration(cal)
        msg = "Calibration Defaults Restored"
        self.lbl_cal_status.config(text=f"Status: {msg}", fg="#94a3b8")
        if hasattr(self, "hand_visualizer"):
            self.hand_visualizer.lbl_cal_status.config(text=f"Status: {msg}", fg="#94a3b8")
        return msg

    # =========================================================================
    # Dataset & Training Handlers
    # =========================================================================
    def _toggle_simulation(self):
        if self.sim_mode_var.get():
            self.receiver.start_simulation()
        else:
            self.receiver.stop_simulation()

    def _on_gesture_combo_change(self, event=None):
        gest = self.cmb_gestures.get().strip().upper()
        if gest in getattr(config, "STATIC_GESTURES", []):
            self.lbl_gesture_type.config(text="[STATIC POSE]", fg="#38bdf8", bg="#032b43")
            if hasattr(self, "lbl_capture_guidance"):
                self.lbl_capture_guidance.config(
                    text=f"💡 Static Pose Guidance: Hold '{gest}' shape and click '📸 Burst Capture (50+)'. Slowly move hand in 3D space (up/down/left/right) during capture for position invariance!",
                    fg="#38bdf8"
                )
        elif gest in getattr(config, "DYNAMIC_GESTURES", []):
            self.lbl_gesture_type.config(text="[DYNAMIC MOTION]", fg="#fbbf24", bg="#2a1f05")
            if hasattr(self, "lbl_capture_guidance"):
                self.lbl_capture_guidance.config(
                    text=f"💡 Dynamic Motion Guidance: Click '⏺ Record Stream' and execute the full '{gest}' motion trajectory smoothly over 3 seconds.",
                    fg="#fbbf24"
                )
        else:
            self.lbl_gesture_type.config(text="[CUSTOM SIGN]", fg="#94a3b8", bg="#1e293b")
            if hasattr(self, "lbl_capture_guidance"):
                self.lbl_capture_guidance.config(
                    text=f"💡 Custom Sign: Choose '📸 Burst Capture (50+)' for static hand poses or '⏺ Record Stream' for gestures with hand movement.",
                    fg="#94a3b8"
                )

    def _start_static_burst_recording(self):
        gesture = self.cmb_gestures.get().strip().upper()
        if not gesture:
            messagebox.showwarning("Warning", "Select or type a gesture label first.")
            return

        try:
            count = int(self.spn_burst_count.get())
        except ValueError:
            count = 50
        count = max(count, 10)

        hand_mode = self.dataset_hand_mode_var.get()
        l_s, r_s = self.receiver.get_latest_states()
        if hand_mode == "right" and not (r_s["connected"] or self.receiver.simulation_mode):
            messagebox.showwarning("Glove Disconnected", "Right glove is not connected. Connect the glove or toggle Simulation Mode in Tab 2.")
            return
        elif hand_mode == "left" and not (l_s["connected"] or self.receiver.simulation_mode):
            messagebox.showwarning("Glove Disconnected", "Left glove is not connected. Connect the glove or toggle Simulation Mode in Tab 2.")
            return

        dur = max(2.5, count * 0.05)

        self.btn_burst_snapshot.config(state=tk.DISABLED)
        self.btn_record.config(state=tk.DISABLED)
        self.btn_snapshot.config(state=tk.DISABLED)
        self.lbl_rec_status.config(text=f"📸 Hold '{gesture}' pose! Move hand slowly in space... (0/{count})", fg="#38bdf8")
        self.rec_prog_bar["value"] = 0

        def _prog(progress, current_count, total_count):
            def _update():
                self.rec_prog_bar["value"] = int(progress * 100)
                self.lbl_rec_status.config(
                    text=f"📸 Recording '{gesture}' spatial variation: {current_count}/{total_count} ({progress*100:.0f}%)",
                    fg="#38bdf8"
                )
            self.after(0, _update)

        def _done(total):
            def _update():
                self.btn_burst_snapshot.config(state=tk.NORMAL)
                self.btn_record.config(state=tk.NORMAL)
                self.btn_snapshot.config(state=tk.NORMAL)
                self.lbl_rec_status.config(
                    text=f"✅ Done! Saved {total} spatial snapshots for '{gesture}' ({hand_mode.upper()} mode).",
                    fg="#10b981"
                )
                self._refresh_dataset_table()
            self.after(0, _update)

        self.recorder.record_static_burst(
            label=gesture,
            count=count,
            duration_sec=dur,
            hand_mode=hand_mode,
            progress_cb=_prog,
            done_cb=_done
        )

    def _start_recording_session(self):
        gesture = self.cmb_gestures.get().strip().upper()
        if not gesture:
            messagebox.showwarning("Warning", "Select or type a gesture label first.")
            return

        try:
            dur = float(self.spn_duration.get())
        except ValueError:
            dur = 3.0

        hand_mode = self.dataset_hand_mode_var.get()
        self.btn_record.config(state=tk.DISABLED)
        self.btn_burst_snapshot.config(state=tk.DISABLED)
        self.btn_snapshot.config(state=tk.DISABLED)
        self.lbl_rec_status.config(text=f"Recording '{gesture}' ({hand_mode.upper()} mode) in progress...", fg="#f59e0b")

        def _prog(progress, count):
            self.rec_prog_bar["value"] = int(progress * 100)

        def _done(total):
            self.btn_record.config(state=tk.NORMAL)
            self.btn_burst_snapshot.config(state=tk.NORMAL)
            self.btn_snapshot.config(state=tk.NORMAL)
            self.lbl_rec_status.config(text=f"Finished! Saved {total} samples for '{gesture}' ({hand_mode.upper()}).", fg="#10b981")
            self._refresh_dataset_table()

        self.recorder.record_gesture_session(gesture, duration_sec=dur, progress_cb=_prog, done_cb=_done, hand_mode=hand_mode)

    def _record_single_snapshot(self):
        gesture = self.cmb_gestures.get().strip().upper()
        if not gesture:
            return
        hand_mode = self.dataset_hand_mode_var.get()
        vec = self.receiver.get_synchronized_feature_vector(hand_mode=hand_mode)
        self.recorder.append_single_frame(gesture, vec, hand_mode=hand_mode)
        self._refresh_dataset_table()
        self.lbl_rec_status.config(text=f"Snapshot saved for '{gesture}' ({hand_mode.upper()} mode)", fg="#10b981")

    def _refresh_dataset_table(self):
        for item in self.tree_dataset.get_children():
            self.tree_dataset.delete(item)
        summary = self.recorder.get_dataset_summary()
        self.tree_dataset.tag_configure("black_text", foreground="#000000", background="#ffffff")
        for gest, count in sorted(summary.items()):
            self.tree_dataset.insert("", tk.END, values=(gest, count), tags=("black_text",))

        # Update dataset info badge
        if hasattr(self, "lbl_dataset_mode_info"):
            info = self.recorder.get_dataset_info()
            if info["exists"] and info["samples"] > 0:
                self.lbl_dataset_mode_info.config(
                    text=f"Dataset: {info['mode']}\nSamples: {info['samples']} | Classes: {len(info['classes'])}",
                    fg="#10b981"
                )
            else:
                self.lbl_dataset_mode_info.config(text="Dataset: Empty (Record or Bootstrap)", fg="#94a3b8")

    def _generate_bootstrap(self):
        hand_mode = self.dataset_hand_mode_var.get()
        self.recorder.generate_synthetic_bootstrap_dataset(hand_mode=hand_mode)
        self._refresh_dataset_table()
        messagebox.showinfo("Success", f"Synthetic dataset generated for {hand_mode.upper()} mode across all common ASL signs!")

    def _delete_selected_gesture(self):
        sel = self.tree_dataset.selection()
        if not sel:
            return
        item = self.tree_dataset.item(sel[0])
        gest = item["values"][0]
        if messagebox.askyesno("Confirm", f"Delete all samples for gesture '{gest}'?"):
            self.recorder.delete_gesture_samples(gest)
            self._refresh_dataset_table()

    def _wipe_dataset(self):
        if messagebox.askyesno("Confirm", "Are you sure you want to delete the ENTIRE dataset?"):
            self.recorder.clear_dataset()
            self._refresh_dataset_table()

    def _start_training(self):
        self.btn_train.config(state=tk.DISABLED, text="Training in progress...")
        chosen_mode = self.train_hand_mode_var.get()
        use_spatial = self.spatial_invariance_var.get()

        def _worker():
            success, metrics = train_classifier(
                model_type=self.model_type_var.get(),
                hand_mode=chosen_mode,
                spatial_augmentation=use_spatial
            )

            def _update_ui():
                self.btn_train.config(state=tk.NORMAL, text="🚀 Train ASL Recognition Model")
                if success:
                    acc = metrics["accuracy"] * 100
                    h_mode = metrics.get("hand_mode", "dual").upper()
                    n_feats = metrics.get("num_features", 28)
                    self.lbl_train_acc.config(text=f"Test Accuracy: {acc:.2f}% | Mode: {h_mode} ({n_feats} Feat)", fg="#10b981")
                    self.txt_train_report.delete("1.0", tk.END)
                    report_str = f"Model Algorithm: {metrics['model_type'].upper()}\n"
                    report_str += f"Hand Mode: {h_mode} ({n_feats} Features)\n"
                    report_str += f"Spatial Invariance: {'ENABLED (Augmented across 3D space)' if metrics.get('spatial_augmentation') else 'DISABLED'}\n"
                    orig_cnt = metrics.get('original_train_samples', metrics['train_samples'])
                    report_str += f"Training Samples: {metrics['train_samples']} (expanded from {orig_cnt} raw snapshots) | Test Samples: {metrics['test_samples']}\n\n"
                    report_str += "Classes (" + str(metrics['num_classes']) + "): " + ", ".join(metrics["classes"]) + "\n"
                    self.txt_train_report.insert("1.0", report_str)

                    # Reload recognizer model
                    self.recognizer.load_model()
                    messagebox.showinfo("Trained!", f"Model successfully trained in {h_mode} mode with {acc:.1f}% accuracy and spatial invariance!")
                else:
                    self.lbl_train_acc.config(text="Training Failed", fg="#ef4444")
                    messagebox.showerror("Error", metrics.get("error", "Unknown error"))

            self.after(0, _update_ui)

        threading.Thread(target=_worker, daemon=True).start()

    def on_close(self):
        self.receiver.stop()
        self.destroy()

if __name__ == "__main__":
    app = ASLTranslatorApp()
    app.mainloop()
