"""
LPG Anomaly & Leak Detection — Inference Helper
Quick utility to score real-time sensor readings for leaks or sudden anomalies.

Example:
    from ml.inference import predict_reading
    result = predict_reading(
        rate_g_per_min=1.8,
        duration_min=140,
        hour=2,
        weight_kg=14.2
    )
    print(result) # {"is_anomaly": True, "anomaly_type": "continuous_gas_leak", ...}
"""

import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "models" / "lpg_leak_detector.pkl"

_model = None

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_lpg_anomaly_model.py first.")
        _model = joblib.load(str(MODEL_PATH))
    return _model


def predict_reading(rate_g_per_min: float, duration_min: float, hour: int, weight_kg: float, noise_score: float = 0.02) -> dict:
    model = get_model()
    is_inactive = 1.0 if hour in [23, 0, 1, 2, 3, 4, 5, 6] else 0.0

    features = np.array([[
        rate_g_per_min,
        duration_min,
        is_inactive,
        float(hour),
        noise_score,
        weight_kg
    ]])

    pred = model.predict(features)[0] # 1 = normal, -1 = anomaly
    score = float(model.decision_function(features)[0]) # negative = anomalous

    is_anomaly = (pred == -1)
    anomaly_type = "normal"

    if is_anomaly:
        if rate_g_per_min > 50.0:
            anomaly_type = "sudden_disconnect_or_theft"
        elif duration_min >= 60 and rate_g_per_min > 0.5:
            anomaly_type = "continuous_gas_leak"
        elif rate_g_per_min > 10.0:
            anomaly_type = "burner_left_unlit"
        else:
            anomaly_type = "unusual_consumption"

    return {
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type,
        "anomaly_score": round(score, 4),
        "status": "DANGER" if is_anomaly else "NORMAL",
        "action_required": anomaly_type in ["continuous_gas_leak", "burner_left_unlit", "sudden_disconnect_or_theft"]
    }


if __name__ == "__main__":
    # Test normal cooking
    print("Normal Cooking Test (Daytime, 30 min):", predict_reading(3.5, 30, 12, 18.5))

    # Test slow nighttime gas leak
    print("Night Gas Leak Test (2 AM, 120 min):", predict_reading(1.5, 120, 2, 14.0))

    # Test sudden cylinder removal
    print("Sudden Removal Test:", predict_reading(350.0, 2, 15, 0.5))
