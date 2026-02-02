# ⚖️ IntelliCase: AI-Powered Case Management & Predictive Analytics

**IntelliCase** is a full-stack AI solution that predicts whether a case will be **Resolved** or **Not Resolved** (Pending/Escalated). It bridges **Python-based model training** with **Node.js deployment** using **ONNX Runtime**, enabling enterprise-ready, real-time predictions.

## Overview

IntelliCase implements a dual-layer architecture:

1. **Predictive Layer (Python)**

   * Decision Tree model trained on synthetic case, client, and staff data
   * Inputs: `age`, `risk_score`, `years_exp`, `workload`, `priority`
   * Output: `is_resolved` (1 = Resolved, 0 = Not Resolved)

2. **Deployment Layer (Node.js)**

   * ONNX model loaded in Node.js via `onnxruntime-node`
   * Receives JSON input, runs inference, returns predictions with optional confidence probabilities
   * Serves predictions via `/predict` API endpoint

## Technical Stack

| Layer            | Technology                                       |
| ---------------- | ------------------------------------------------ |
| Model Training   | Python 3.x, Scikit-Learn                         |
| Model Conversion | ONNX (zipmap disabled for Node.js compatibility) |
| API Server       | Node.js, Express.js                              |
| Runtime          | ONNX Runtime for Node.js                         |
| Deployment       | Docker-ready (optional)                          |

## Model Training and Conversion

* Data: synthetic datasets for `clients`, `staff`, and `cases`
* Features: `age`, `risk_score`, `years_exp`, `workload`, `priority`
* Target: `is_resolved` (1 = Resolved, 0 = Not Resolved)
* Trained Decision Tree Classifier (max_depth=3)
* Converted to ONNX using `skl2onnx` with `zipmap=False`
* Output tensor is Float32Array for JavaScript compatibility

## API Usage (Node.js)

### Predict Endpoint

**POST `/predict`**

**Request Body:**

```json
{
  "age": 35,
  "risk_score": 0.7,
  "years_exp": 5,
  "workload": 10,
  "priority": 2
}
```

**Response:**

```json
{
  "success": true,
  "prediction": {
    "is_resolved": 1,
    "confidence": [0.65, 0.35]
  }
}
```

### Inference Code Example

```javascript
const ort = require('onnxruntime-node');

async function runModel(inputData) {
    const session = await ort.InferenceSession.create('./case_predictor.onnx');
    const inputValues = Float32Array.from(inputData);
    const tensor = new ort.Tensor('float32', inputValues, [1, 5]);

    const feeds = { float_input: tensor };
    const results = await session.run(feeds);

    const labelTensor = results.output_label || results.label;
    const probTensor = results.output_probability || results.probabilities;

    return {
        is_resolved: Number(labelTensor.data[0]),
        confidence: probTensor ? Array.from(probTensor.data) : 'Not available'
    };
}
```

## Getting Started

### Prerequisites

* Node.js v18+
* Python 3.x (for retraining, optional)
* `case_predictor.onnx` model file


## Future Enhancements

* Expand to multi-class predictions (Resolved, Escalated, Pending)
* Add authentication and multi-user tracking
* Real-time dashboard for case predictions and analytics
* Incorporate historical data for improved accuracy
* Full Docker deployment for production environments

## Project Significance

**IntelliCase** demonstrates **Python-to-JavaScript AI model deployment**, bridging data science and enterprise software. It enables real-time case management, predictive analytics, and confidence-based decision support in a lightweight, production-ready setup.
