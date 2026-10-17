# 🧠 LPG Smart Monitor — Machine Learning Anomaly Pipeline

An **Isolation Forest** unsupervised anomaly detection engine built for LPG gas cylinder telemetry. It actively differentiates normal domestic cooking patterns from hazardous gas leaks and cylinder detachment.

---

## 🎯 What It Solves

Standard threshold alarms only trigger when the cylinder is **almost empty (< 5kg)**.  
This ML engine detects:
1. 🚨 **Slow Gas Leaks (Unattended Danger):** Persistent continuous drop (0.8–3 g/min) for > 60 minutes, especially during inactive hours (23:00–06:00).
2. ⚠️ **Burner Left Unlit:** Extended high-rate flow (> 10 g/min for > 70 mins).
3. 🛑 **Theft / Sensor Disconnect:** Abrupt drop (> 50 g/min plunge to tare or zero).

---

## 📊 Evaluation Performance

Tested across 350 simulated realistic telemetry streams:

| Metric | Score | Detail |
| :--- | :--- | :--- |
| **Accuracy** | **96.00%** | Overall correct classifications |
| **Recall** | **100.00%** | **Zero missed gas leaks** (100% of leaks detected) |
| **Precision** | **91.46%** | Low false alarm rate on normal cooking |
| **F1 Score** | **95.54%** | Harmonic balance between safety and precision |

### Detection by Scenario:
* 🟢 **Normal Inactive / Idle:** 93% recognized normal
* 🍳 **Normal Cooking (Morning/Night):** 93% recognized normal
* 🚨 **Slow Nighttime Gas Leak:** **100% caught**
* ⚡ **Sudden Disconnect / Removal:** **100% caught**

---

## 📂 Directory Structure

```
ml/
├── models/
│   ├── lpg_leak_detector.pkl   # Serialized IsolationForest binary
│   └── model_meta.json         # Feature list, versions, and metrics
├── train_lpg_anomaly_model.py  # Generates training telemetry and fits model
├── evaluate_lpg_model.py       # Validates precision/recall and confusion matrix
├── inference.py                # Standalone helper for real-time scoring
└── README.md                   # This documentation
```

---

## 🚀 How to Run

### 1. Train the Model
```bash
python ml/train_lpg_anomaly_model.py
```

### 2. Run Evaluations
```bash
python ml/evaluate_lpg_model.py
```

### 3. Run Quick Test Inference
```bash
python ml/inference.py
```
