"""
Model Training Module
Trains and validates machine learning classifiers (Random Forest / MLP) on dual-glove ASL features.
Saves model weights, feature scalers, and label encoders for real-time inference.
"""
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ModelTrainer")

def train_classifier(
    model_type: str = "rf",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[bool, Dict[str, Any]]:
    """
    Train ASL classifier using data in config.DATASET_CSV_PATH.
    
    Args:
        model_type: 'rf' for Random Forest, 'mlp' for Multi-Layer Perceptron.
        test_size: Fraction of samples reserved for testing (0.1 - 0.3).
        random_state: Random seed for reproducibility.
        
    Returns:
        (success_bool, metrics_dict)
    """
    if not config.DATASET_CSV_PATH.exists():
        msg = f"Dataset file does not exist at {config.DATASET_CSV_PATH}"
        logger.error(msg)
        return False, {"error": msg}

    try:
        df = pd.read_csv(config.DATASET_CSV_PATH)
        if len(df) < 10:
            msg = f"Dataset too small ({len(df)} samples). Record more samples before training."
            logger.error(msg)
            return False, {"error": msg}

        # Check required columns
        missing_cols = [c for c in config.FEATURE_COLUMNS if c not in df.columns]
        if missing_cols:
            msg = f"Dataset missing feature columns: {missing_cols}"
            logger.error(msg)
            return False, {"error": msg}

        X = df[config.FEATURE_COLUMNS].values.astype(np.float32)
        y_raw = df["label"].values.astype(str)

        # Encode categorical gesture labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        classes = list(label_encoder.classes_)

        if len(classes) < 2:
            msg = "Need at least 2 distinct gesture classes to train a classifier."
            logger.error(msg)
            return False, {"error": msg}

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize model
        if model_type.lower() == "mlp":
            model = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                max_iter=300,
                random_state=random_state,
                early_stopping=True
            )
        else: # Default: Random Forest
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=16,
                n_jobs=-1,
                random_state=random_state
            )

        logger.info(f"Training {model.__class__.__name__} on {len(X_train)} samples across {len(classes)} classes...")
        model.fit(X_train_scaled, y_train)

        # Evaluate performance
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=classes, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        logger.info(f"Training completed successfully! Test Accuracy: {accuracy * 100:.2f}%")

        # Save artifacts
        joblib.dump(model, config.MODEL_PATH)
        joblib.dump(scaler, config.SCALER_PATH)
        joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)

        metrics = {
            "accuracy": float(accuracy),
            "num_classes": len(classes),
            "classes": classes,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "report": report,
            "confusion_matrix": cm,
            "model_type": model_type
        }
        return True, metrics

    except Exception as e:
        logger.exception("Error during model training:")
        return False, {"error": str(e)}

if __name__ == "__main__":
    from dataset_recorder import DatasetRecorder
    from receiver import DualGloveReceiver
    # If no data exists, generate bootstrap dataset
    if not config.DATASET_CSV_PATH.exists():
        rec = DatasetRecorder(DualGloveReceiver())
        rec.generate_synthetic_bootstrap_dataset()
    success, metrics = train_classifier()
    print("Result:", "Success" if success else "Failed")
    if success:
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("Classes:", metrics['classes'])
