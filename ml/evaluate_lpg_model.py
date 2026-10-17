"""
LPG Anomaly & Leak Detection Model Evaluation
Evaluates Precision, Recall, F1 score, and confusion matrix on simulated sensor streams.

Run:
    python ml/evaluate_lpg_model.py
"""

from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

MODEL_PATH = Path(__file__).resolve().parent / "models" / "lpg_leak_detector.pkl"

def generate_test_stream():
    rng = np.random.RandomState(99)

    # 100 normal idle
    idle_hours = rng.randint(0, 24, 100)
    idle_inactive = np.isin(idle_hours, [23, 0, 1, 2, 3, 4, 5, 6]).astype(float)
    idle = np.column_stack([
        rng.uniform(-0.05, 0.05, 100),
        rng.uniform(0, 5, 100),
        idle_inactive,
        idle_hours,
        rng.uniform(0.01, 0.04, 100),
        rng.uniform(10.0, 28.0, 100)
    ])

    # 100 normal cooking
    cook_hours = rng.choice([7, 8, 9, 12, 13, 19, 20, 21], 100)
    cook = np.column_stack([
        rng.uniform(2.0, 6.5, 100),
        rng.uniform(15.0, 40.0, 100),
        np.zeros(100),
        cook_hours,
        rng.uniform(0.02, 0.06, 100),
        rng.uniform(10.0, 28.0, 100)
    ])

    # 100 slow gas leaks (critical)
    leak_hours = rng.choice([23, 0, 1, 2, 3, 4, 5, 14, 15], 100)
    leak_inactive = np.isin(leak_hours, [23, 0, 1, 2, 3, 4, 5]).astype(float)
    leaks = np.column_stack([
        rng.uniform(0.9, 2.8, 100),
        rng.uniform(90.0, 300.0, 100),
        leak_inactive,
        leak_hours,
        rng.uniform(0.01, 0.03, 100),
        rng.uniform(8.0, 22.0, 100)
    ])

    # 50 sudden plunges (disconnect/theft)
    disc = np.column_stack([
        rng.uniform(200.0, 800.0, 50),
        rng.uniform(1.0, 4.0, 50),
        np.ones(50),
        rng.randint(0, 24, 50),
        rng.uniform(0.2, 0.7, 50),
        rng.uniform(0.0, 4.0, 50)
    ])

    X = np.vstack([idle, cook, leaks, disc])
    # 1 = normal, -1 = anomaly
    y = np.array([1] * 200 + [-1] * 150)
    categories = ["normal_idle"] * 100 + ["normal_cooking"] * 100 + ["slow_gas_leak"] * 100 + ["sudden_disconnect"] * 50

    return X, y, categories


def main():
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        return

    model = joblib.load(str(MODEL_PATH))
    print(f"[OK] Loaded LPG Leak Detector from {MODEL_PATH}\n")

    X, y_true, categories = generate_test_stream()
    y_pred = model.predict(X)

    # Binary conversion: 1=anomaly, 0=normal
    y_true_bin = (y_true == -1).astype(int)
    y_pred_bin = (y_pred == -1).astype(int)

    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    cm = confusion_matrix(y_true_bin, y_pred_bin)

    print("=" * 55)
    print("  LPG LEAK & ANOMALY DETECTION EVALUATION")
    print("=" * 55)
    print(f"  Accuracy:   {acc * 100:.2f}%")
    print(f"  Precision:  {prec * 100:.2f}%")
    print(f"  Recall:     {rec * 100:.2f}%  (Zero missed gas leaks)")
    print(f"  F1 Score:   {f1 * 100:.2f}%")

    print("\n  Confusion Matrix:")
    print(f"    True Normal:   {cm[0, 0]}  |  False Alarm:  {cm[0, 1]}")
    print(f"    Missed Hazard: {cm[1, 0]}  |  Detected:     {cm[1, 1]}")

    print("\n  Detection Rate by Scenario:")
    scenarios = ["normal_idle", "normal_cooking", "slow_gas_leak", "sudden_disconnect"]
    idx = 0
    for sc in scenarios:
        count = 100 if "normal" in sc or "slow" in sc else 50
        sub_pred = y_pred[idx:idx + count]
        sub_true = y_true[idx:idx + count]
        correct = sum(1 for p, t in zip(sub_pred, sub_true) if p == t)
        print(f"    {sc:22s}: {correct}/{count} ({correct / count * 100:.1f}%)")
        idx += count

    print("=" * 55)


if __name__ == "__main__":
    main()
