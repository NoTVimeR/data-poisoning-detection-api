# 🛡️ AI Data Poisoning Detection System

## 📌 Overview

This project presents a system for detecting and preventing data poisoning attacks in machine learning models.
The system is implemented as a FastAPI-based web service that analyzes incoming data and identifies suspicious samples before they are used in ML pipelines.

---

## 🎯 Project Goal

To develop a modular detection system that improves the reliability and security of AI models by filtering potentially poisoned data.

---

## ⚙️ Key Features

* Multi-method anomaly detection:

  * Z-score (statistical detection)
  * Isolation Forest (machine learning-based)
  * Local Outlier Factor (density-based)
  * KNN Label Consistency (for label flipping attacks)
* Hybrid detection approach
* REST API (FastAPI)
* Data cleaning functionality
* Real-time analysis via HTTP requests

---

## 🧠 Detection Methods

| Method                | Purpose                         |
| --------------------- | ------------------------------- |
| Z-score               | Detects obvious outliers        |
| Isolation Forest      | Detects subtle anomalies        |
| LOF                   | Detects density-based anomalies |
| KNN Label Consistency | Detects label flipping attacks  |
| Hybrid                | Combines multiple methods       |

---

## 🔬 Experiments

We evaluated the system on different types of attacks:

* Simple Outlier Attack
* Complex Subtle Poisoning
* Label Flipping Attack

### Key Findings:

* Z-score performs best for simple outliers
* Isolation Forest is effective for subtle attacks
* KNN method detects label manipulation
* No single method is sufficient for all attack types

---

## 🌐 API Endpoints

### `POST /detect`

Detects anomalies in input data.

### `POST /clean`

Removes suspicious samples and returns cleaned data.

### `POST /analyze`

Performs full multi-method analysis.

---

## 🚀 Deployment

The system is deployed as a cloud-based service using Render:

👉 **Live API:**
https://YOUR-URL.onrender.com

👉 **Swagger UI:**
https://YOUR-URL.onrender.com/docs

---

## 🔌 Example Usage

```python
import requests

data = {
    "values": [45, 50, 120]
}

response = requests.post(
    "https://YOUR-URL.onrender.com/analyze",
    json=data
)

print(response.json())
```

---

## 🏗️ System Architecture

The system acts as a middleware between data and machine learning models:

```
Data → API → Detection → Clean Data → ML Model
```

---

## 📊 Technologies Used

* Python
* FastAPI
* Scikit-learn
* NumPy / Pandas
* Matplotlib

---

## 🔮 Future Work

* Integration with real-world datasets (MNIST, CIFAR)
* Real-time monitoring dashboard
* Advanced ensemble detection methods
* Cloud scaling and optimization

---

## 📚 Authors

Adal & Ainur
Astana IT University
