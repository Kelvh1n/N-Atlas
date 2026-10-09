"""
Real-Time Gesture Recognizer
Performs live classification on synchronized dual-glove feature streams with confidence
thresholding, consecutive frame stability gating, and debouncing.
"""
import time
import logging
from typing import Optional, Tuple, Dict
import numpy as np
import joblib
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GestureRecognizer")

class GestureRecognizer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.is_loaded = False

        # Stability and debounce state
        self.candidate_label: Optional[str] = None
        self.candidate_count: int = 0
        self.last_confirmed_gesture: Optional[str] = None
        self.last_trigger_time: float = 0.0

        self.load_model()

    def load_model(self) -> bool:
        """Load trained model, scaler, and label encoder from disk."""
        if not (config.MODEL_PATH.exists() and config.SCALER_PATH.exists() and config.LABEL_ENCODER_PATH.exists()):
            self.is_loaded = False
            return False

        try:
            self.model = joblib.load(config.MODEL_PATH)
            self.scaler = joblib.load(config.SCALER_PATH)
            self.label_encoder = joblib.load(config.LABEL_ENCODER_PATH)
            self.is_loaded = True
            logger.info(f"GestureRecognizer model loaded. Classes ({len(self.label_encoder.classes_)}): {list(self.label_encoder.classes_)}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {e}")
            self.is_loaded = False
            return False

    def predict_frame(self, feature_vector: np.ndarray) -> Tuple[Optional[str], float, Dict[str, float]]:
        """
        Predict single frame.
        Returns:
            (best_label, confidence, top_probabilities_dict)
        """
        if not self.is_loaded:
            return None, 0.0, {}

        try:
            vec_reshaped = feature_vector.reshape(1, -1)
            vec_scaled = self.scaler.transform(vec_reshaped)

            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(vec_scaled)[0]
                best_idx = int(np.argmax(probs))
                confidence = float(probs[best_idx])
                best_label = self.label_encoder.inverse_transform([best_idx])[0]

                # Extract top 3 candidates for diagnostics
                top_indices = np.argsort(probs)[::-1][:3]
                top_dict = {
                    self.label_encoder.inverse_transform([idx])[0]: float(probs[idx])
                    for idx in top_indices
                }
                return best_label, confidence, top_dict
            else:
                pred = self.model.predict(vec_scaled)[0]
                label = self.label_encoder.inverse_transform([pred])[0]
                return label, 1.0, {label: 1.0}

        except Exception as e:
            logger.debug(f"Inference error: {e}")
            return None, 0.0, {}

    def process_live_stream(self, feature_vector: np.ndarray) -> Tuple[Optional[str], float, bool]:
        """
        Process incoming live telemetry frame.
        
        Returns:
            (current_instant_label, current_confidence, is_new_event_confirmed)
        """
        label, conf, _ = self.predict_frame(feature_vector)
        if label is None or conf < config.CONFIDENCE_THRESHOLD:
            self.candidate_label = None
            self.candidate_count = 0
            return None, 0.0, False

        # Debounce and stability logic
        now = time.time()
        is_new_event = False

        if label == self.candidate_label:
            self.candidate_count += 1
        else:
            self.candidate_label = label
            self.candidate_count = 1

        # Check if held for required stability frames
        if self.candidate_count >= config.STABILITY_FRAMES:
            # Check cooldown to prevent duplicate triggers
            if (label != self.last_confirmed_gesture) or (now - self.last_trigger_time > config.DEBOUNCE_COOLDOWN_SEC):
                self.last_confirmed_gesture = label
                self.last_trigger_time = now
                is_new_event = True
                self.candidate_count = 0  # Reset counter after successful trigger

        return label, conf, is_new_event

    def reset_state(self):
        """Reset temporal state filters."""
        self.candidate_label = None
        self.candidate_count = 0
        self.last_confirmed_gesture = None
        self.last_trigger_time = 0.0
