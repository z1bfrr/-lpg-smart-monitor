"""
LPG Gas Leak & Consumption Anomaly Detection Model
Trains an IsolationForest model on LPG cylinder telemetry to distinguish
between normal cooking consumption and hazardous continuous gas leaks.

Run:
    python ml/train_lpg_anomaly_model.py
"""

import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "lpg_leak_detector.pkl"
META_PATH = MODEL_DIR / "model_meta.json"

FEATURE_NAMES = [
    "consumption_rate_g_per_min",   # Weight drop rate (grams/minute)
    "duration_continuous_drop_min", # Consecutive minutes of declining weight
    "is_inactive_hours",            # 1 if night/inactive (23:00 - 06:00), else 0
    "hour_of_day",                  # Hour of the day (0 - 23)
    "noise_stability_score",        # Reading consistency (std dev of readings)
    "current_weight_kg",            # Current total cylinder weight
]

rng = np.random.RandomState(42)

def generate_normal_data(n=800):
    """
    Normal patterns:
    - Inactive / Storing: 0 rate, flat weight, any hour
    - Active Cooking: 1.5 - 6.0 g/min drop for 10-45 mins, during cooking hours
    """
    n_idle = int(n * 0.6)
    n_cook = n - n_idle

    # 1. Idle cylinder (stove off)
    idle_hours = rng.randint(0, 24, n_idle)
    idle_inactive = np.isin(idle_hours, [23, 0, 1, 2, 3, 4, 5, 6]).astype(float)
    idle_rates = rng.uniform(-0.1, 0.1, n_idle) # negligible sensor drift
    idle_duration = rng.uniform(0, 5, n_idle)
    idle_noise = rng.uniform(0.01, 0.05, n_idle)
    idle_weights = rng.uniform(8.0, 29.5, n_idle)

    idle_samples = np.column_stack([
        idle_rates, idle_duration, idle_inactive, idle_hours, idle_noise, idle_weights
    ])

    # 2. Normal cooking (morning, noon, evening)
    cooking_hours = rng.choice([7, 8, 9, 12, 13, 19, 20, 21], n_cook)
    cooking_inactive = np.zeros(n_cook)
    cooking_rates = rng.uniform(2.0, 7.5, n_cook) # normal burner flow (grams/min)
    cooking_duration = rng.uniform(10.0, 45.0, n_cook) # typical cooking session duration
    cooking_noise = rng.uniform(0.02, 0.08, n_cook)
    cooking_weights = rng.uniform(8.0, 29.5, n_cook)

    cooking_samples = np.column_stack([
        cooking_rates, cooking_duration, cooking_inactive, cooking_hours, cooking_noise, cooking_weights
    ])

    return np.vstack([idle_samples, cooking_samples])


def generate_anomaly_data(n=300):
    """
    Anomalous patterns:
    1. Slow continuous gas leak (silent hazard) - low rate but unending duration during off hours
    2. Severe burner left on unlit (high rate prolonged)
    3. Sudden disconnect / theft / sensor detachment (massive abrupt plunge)
    """
    n_leak = int(n * 0.5)
    n_unlit = int(n * 0.3)
    n_disconnect = n - n_leak - n_unlit

    # 1. Slow gas leak: 0.8 - 2.5 g/min continuously for hours, especially at night
    leak_hours = rng.choice([23, 0, 1, 2, 3, 4, 5, 14, 15], n_leak)
    leak_inactive = np.isin(leak_hours, [23, 0, 1, 2, 3, 4, 5]).astype(float)
    leak_rates = rng.uniform(0.8, 3.0, n_leak)
    leak_duration = rng.uniform(90.0, 360.0, n_leak) # 1.5 to 6 continuous hours!
    leak_noise = rng.uniform(0.01, 0.04, n_leak)     # very smooth, steady decline
    leak_weights = rng.uniform(6.0, 25.0, n_leak)

    leak_samples = np.column_stack([
        leak_rates, leak_duration, leak_inactive, leak_hours, leak_noise, leak_weights
    ])

    # 2. Severe leak or unlit burner: high rate for > 70 mins
    unlit_hours = rng.randint(0, 24, n_unlit)
    unlit_inactive = np.isin(unlit_hours, [23, 0, 1, 2, 3, 4, 5]).astype(float)
    unlit_rates = rng.uniform(7.0, 15.0, n_unlit)
    unlit_duration = rng.uniform(75.0, 240.0, n_unlit)
    unlit_noise = rng.uniform(0.03, 0.09, n_unlit)
    unlit_weights = rng.uniform(6.0, 25.0, n_unlit)

    unlit_samples = np.column_stack([
        unlit_rates, unlit_duration, unlit_inactive, unlit_hours, unlit_noise, unlit_weights
    ])

    # 3. Sudden disconnect / theft / sensor fault: massive drop rate
    disc_hours = rng.randint(0, 24, n_disconnect)
    disc_inactive = np.isin(disc_hours, [23, 0, 1, 2, 3, 4, 5]).astype(float)
    disc_rates = rng.uniform(100.0, 1000.0, n_disconnect) # multi-kg drop in minutes
    disc_duration = rng.uniform(1.0, 5.0, n_disconnect)
    disc_noise = rng.uniform(0.20, 0.80, n_disconnect)
    disc_weights = rng.uniform(0.0, 5.0, n_disconnect)

    disc_samples = np.column_stack([
        disc_rates, disc_duration, disc_inactive, disc_hours, disc_noise, disc_weights
    ])

    return np.vstack([leak_samples, unlit_samples, disc_samples])


def main():
    print("=" * 60)
    print("  TRAINING LPG SMART MONITOR ANOMALY DETECTION MODEL")
    print("=" * 60)

    X_normal = generate_normal_data(800)
    X_anomaly = generate_anomaly_data(300)
    X_train = X_normal # Isolation Forest is fitted primarily on normal telemetry

    print(f"Dataset generated:")
    print(f"  • Normal telemetry samples:     {len(X_normal)}")
    print(f"  • Anomaly / Leak test samples:  {len(X_anomaly)}")
    print(f"  • Feature dimensions:           {len(FEATURE_NAMES)}")

    model = IsolationForest(
        n_estimators=150,
        max_samples="auto",
        contamination=0.15,
        random_state=42,
        n_jobs=1
    )

    print("\nFitting IsolationForest...")
    model.fit(X_train)

    # Evaluation
    pred_normal = model.predict(X_normal)
    pred_anomaly = model.predict(X_anomaly)

    normal_acc = np.mean(pred_normal == 1) * 100
    anomaly_recall = np.mean(pred_anomaly == -1) * 100

    print(f"\nTraining Results:")
    print(f"  • Normal Data Correct:  {normal_acc:.1f}%")
    print(f"  • Leak/Anomaly Recall:  {anomaly_recall:.1f}%")

    # Save artifact
    joblib.dump(model, str(MODEL_PATH))
    print(f"\n[SUCCESS] Model saved to: {MODEL_PATH}")

    # Save metadata & lightweight client rules
    meta = {
        "model_name": "LPG-LeakGuard-IF",
        "version": "1.0.0",
        "algorithm": "IsolationForest",
        "n_estimators": 150,
        "features": FEATURE_NAMES,
        "metrics": {
            "normal_accuracy": round(normal_acc, 2),
            "anomaly_recall": round(anomaly_recall, 2)
        },
        "rule_thresholds": {
            "critical_continuous_duration_min": 60,
            "inactive_leak_rate_min_g_per_min": 0.8,
            "sudden_drop_threshold_g_per_min": 50.0
        }
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[SUCCESS] Metadata saved to: {META_PATH}")


if __name__ == "__main__":
    main()
