# ⚖️ IntelliCase: AI-Powered Case Management & Predictive Analytics

IntelliCase is a full-stack AI solution designed to predict case outcomes in real-time. It bridges the gap between **Data Science (Python)** and **Enterprise Software (Node.js)** by deploying a machine learning model via the ONNX runtime.

---

## 🚀 Overview

This project implements a dual-layer AI strategy:
1.  **Predictive Layer:** A Decision Tree model that predicts if a case will be "Resolved" or "Escalated" based on staff workload, experience, and case priority.
2.  **Infrastructure:** A high-performance Node.js API that serves predictions with sub-millisecond latency.



---

## 🛠️ Technical Stack

* **Language:** Python 3.x (Modeling), Node.js (API)
* **Machine Learning:** Scikit-Learn
* **Interoperability:** ONNX (Open Neural Network Exchange)
* **Server Framework:** Express.js
* **Runtime:** ONNX Runtime for Node.js

---

## 📊 Model Performance

The model was trained on a synthetic dataset representing complex relationships between employee burnout (workload) and case complexity (priority).

### Evaluation Metrics:
| Metric | Score | Analysis |
| :--- | :--- | :--- |
| **Recall** | **88%** | High sensitivity; captures nearly all successful resolutions. |
| **Precision** | **52%** | The model is "optimistic," successfully identifying all wins but flagging some risks as potential wins. |
| **F1-Score** | **0.65** | A robust baseline for high-variance case data. |



---

## ⚙️ Architectural Decisions

### 1. The ONNX Bridge
I utilized **ONNX** to decouple the model from Python. This allows the Node.js server to run the model natively without needing a Python environment in production, significantly reducing the server's resource footprint.

### 2. Disabling ZipMap
A critical optimization was setting `zipmap: False` during the model conversion. This ensures the output is a standard **Tensor (Float32Array)**, preventing "Non-tensor type" errors in the Node.js environment and simplifying data parsing.

### 3. Real-Time Inference
By utilizing `onnxruntime-node`, the API handles inference within the Node.js event loop, ensuring that predictions are served as fast as any standard JSON response.

---

## 🚀 Getting Started

### Prerequisites
* Node.js (v18 or higher)
* The `case_predictor.onnx` file (generated via the Python training script)

