"""
Dataset Recorder Module
Captures, labels, and manages dual-glove sensory dataset samples for ASL recognition model training.
"""
import time
import threading
import logging
from typing import Dict, Optional, Callable
import pandas as pd
import numpy as np
import config
from receiver import DualGloveReceiver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetRecorder")

class DatasetRecorder:
    def __init__(self, receiver: DualGloveReceiver):
        self.receiver = receiver
        self.is_recording = False
        self.recording_thread: Optional[threading.Thread] = None

    def append_single_frame(self, label: str, feature_vector: np.ndarray) -> bool:
        """Append an instantaneous 28-D vector to the dataset CSV."""
        if len(feature_vector) != len(config.FEATURE_COLUMNS):
            logger.error(f"Vector size mismatch: expected {len(config.FEATURE_COLUMNS)}, got {len(feature_vector)}")
            return False

        row = {
            "timestamp": time.time(),
            "label": label.strip().upper(),
            **{col: val for col, val in zip(config.FEATURE_COLUMNS, feature_vector)}
        }
        df = pd.DataFrame([row])
        header = not config.DATASET_CSV_PATH.exists()
        df.to_csv(config.DATASET_CSV_PATH, mode="a", header=header, index=False)
        return True

    def record_gesture_session(
        self,
        label: str,
        duration_sec: float = 3.0,
        sample_hz: float = 50.0,
        progress_cb: Optional[Callable[[float, int], None]] = None,
        done_cb: Optional[Callable[[int], None]] = None
    ):
        """
        Record a stream of frames for a specific ASL sign over a set duration.
        Executes on a background thread.
        """
        if self.is_recording:
            logger.warning("Recording already in progress.")
            return

        def _worker():
            self.is_recording = True
            label_clean = label.strip().upper()
            interval = 1.0 / sample_hz
            total_samples = int(duration_sec * sample_hz)
            collected_rows = []

            start_t = time.time()
            for idx in range(total_samples):
                if not self.is_recording:
                    break
                t_frame_start = time.time()
                vec = self.receiver.get_synchronized_feature_vector()
                row = {
                    "timestamp": time.time(),
                    "label": label_clean,
                    **{col: val for col, val in zip(config.FEATURE_COLUMNS, vec)}
                }
                collected_rows.append(row)

                progress = (idx + 1) / total_samples
                if progress_cb:
                    progress_cb(progress, idx + 1)

                elapsed = time.time() - t_frame_start
                sleep_t = interval - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

            if collected_rows:
                df = pd.DataFrame(collected_rows)
                header = not config.DATASET_CSV_PATH.exists()
                df.to_csv(config.DATASET_CSV_PATH, mode="a", header=header, index=False)
                logger.info(f"Saved {len(collected_rows)} frames for gesture '{label_clean}'")

            self.is_recording = False
            if done_cb:
                done_cb(len(collected_rows))

        self.recording_thread = threading.Thread(target=_worker, daemon=True)
        self.recording_thread.start()

    def cancel_recording(self):
        """Interrupt active recording session."""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=0.5)

    def get_dataset_summary(self) -> Dict[str, int]:
        """Return sample counts per gesture label."""
        if not config.DATASET_CSV_PATH.exists():
            return {}
        try:
            df = pd.read_csv(config.DATASET_CSV_PATH)
            if "label" not in df.columns:
                return {}
            return df["label"].value_counts().to_dict()
        except Exception as e:
            logger.error(f"Failed to read dataset: {e}")
            return {}

    def clear_dataset(self):
        """Wipe dataset file."""
        if config.DATASET_CSV_PATH.exists():
            config.DATASET_CSV_PATH.unlink()
            logger.info("Cleared dataset CSV.")

    def delete_gesture_samples(self, label: str) -> int:
        """Remove all entries belonging to a given label."""
        if not config.DATASET_CSV_PATH.exists():
            return 0
        df = pd.read_csv(config.DATASET_CSV_PATH)
        initial_len = len(df)
        df = df[df["label"] != label.strip().upper()]
        df.to_csv(config.DATASET_CSV_PATH, index=False)
        deleted = initial_len - len(df)
        logger.info(f"Deleted {deleted} samples of '{label}'")
        return deleted

    def generate_synthetic_bootstrap_dataset(self, samples_per_class: int = 60):
        """
        Generate realistic synthetic data for common ASL signs to allow
        immediate testing of the ML and Translation pipeline before physical gloves are connected.
        """
        logger.info("Generating synthetic bootstrap dataset for ASL signs...")
        # Define finger deflection signatures for select ASL signs:
        # [Thumb, Index, Middle, Ring, Pinky] (0.0 = straight, 1.0 = bent)
        hand_signatures = {
            "IDLE": ([0.05, 0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05, 0.05]),
            "A": ([0.2, 0.9, 0.9, 0.9, 0.9], [0.2, 0.9, 0.9, 0.9, 0.9]),
            "B": ([0.9, 0.1, 0.1, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1, 0.1]),
            "C": ([0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5]),
            "D": ([0.8, 0.1, 0.8, 0.8, 0.8], [0.8, 0.1, 0.8, 0.8, 0.8]),
            "L": ([0.1, 0.1, 0.9, 0.9, 0.9], [0.1, 0.1, 0.9, 0.9, 0.9]),
            "Y": ([0.1, 0.9, 0.9, 0.9, 0.1], [0.1, 0.9, 0.9, 0.9, 0.1]),
            "HELLO": ([0.2, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1]),
            "THANK_YOU": ([0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1]),
            "PLEASE": ([0.2, 0.1, 0.1, 0.1, 0.1], [0.8, 0.8, 0.8, 0.8, 0.8]),
            "YES": ([0.9, 0.9, 0.9, 0.9, 0.9], [0.8, 0.8, 0.8, 0.8, 0.8]),
            "NO": ([0.1, 0.7, 0.7, 0.9, 0.9], [0.8, 0.8, 0.8, 0.8, 0.8]),
            "HELP": ([0.1, 0.8, 0.8, 0.8, 0.8], [0.1, 0.1, 0.1, 0.1, 0.1]),
            "WATER": ([0.8, 0.1, 0.1, 0.1, 0.8], [0.8, 0.8, 0.8, 0.8, 0.8]),
            "FOOD": ([0.6, 0.6, 0.6, 0.6, 0.6], [0.8, 0.8, 0.8, 0.8, 0.8]),
            "I_LOVE_YOU": ([0.1, 0.1, 0.9, 0.9, 0.1], [0.1, 0.1, 0.9, 0.9, 0.1]),
            "GOOD_MORNING": ([0.2, 0.2, 0.2, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1, 0.1]),
            "HOW_ARE_YOU": ([0.3, 0.2, 0.2, 0.3, 0.3], [0.3, 0.2, 0.2, 0.3, 0.3]),
            "DOCTOR": ([0.8, 0.2, 0.2, 0.8, 0.8], [0.1, 0.1, 0.1, 0.1, 0.1]),
            "FAMILY": ([0.7, 0.7, 0.1, 0.1, 0.1], [0.7, 0.7, 0.1, 0.1, 0.1]),
            "GOODBYE": ([0.2, 0.1, 0.1, 0.1, 0.1], [0.2, 0.1, 0.1, 0.1, 0.1])
        }

        all_rows = []
        for gesture, (l_flex_base, r_flex_base) in hand_signatures.items():
            for _ in range(samples_per_class):
                # Add Gaussian sensor noise
                l_flex = np.clip(np.array(l_flex_base) + np.random.normal(0, 0.04, 5), 0, 1)
                r_flex = np.clip(np.array(r_flex_base) + np.random.normal(0, 0.04, 5), 0, 1)

                l_accel = np.random.normal(0, 0.1, 3) + [0, 0, 0.98]
                r_accel = np.random.normal(0, 0.1, 3) + [0, 0, 0.98]
                l_gyro = np.random.normal(0, 0.05, 3)
                r_gyro = np.random.normal(0, 0.05, 3)
                l_mag = np.random.normal(0, 0.2, 3) + [0.5, -0.3, 0.8]
                r_mag = np.random.normal(0, 0.2, 3) + [-0.4, 0.6, 0.7]

                vec = np.concatenate([
                    l_flex, l_accel, l_gyro, l_mag,
                    r_flex, r_accel, r_gyro, r_mag
                ])

                row = {
                    "timestamp": time.time(),
                    "label": gesture,
                    **{col: val for col, val in zip(config.FEATURE_COLUMNS, vec)}
                }
                all_rows.append(row)

        df = pd.DataFrame(all_rows)
        df.to_csv(config.DATASET_CSV_PATH, index=False)
        logger.info(f"Generated {len(all_rows)} synthetic samples across {len(hand_signatures)} classes.")
