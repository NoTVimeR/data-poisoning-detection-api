# AI Data Poisoning Detection System v2.1

FastAPI-based prototype for detecting data poisoning attacks before data is used in machine learning pipelines.

## Practical Improvements

- Real-world validation with Breast Cancer Wisconsin, Wine, and Digits datasets.
- Simple outlier, complex subtle poisoning, and label flipping attack simulation.
- Improved complex subtle poisoning scenario using feature shift and correlation drift.
- API v2.2 demo endpoints for end-to-end and multi-dataset validation.
- Minimal deep learning poisoning experiment using the Scikit-learn Digits dataset and an MLP classifier.
- Architecture diagram for the diploma text and presentation.

## Detection Methods

| Method | Purpose |
| --- | --- |
| Z-score | Detects statistically obvious outliers |
| Isolation Forest | Detects anomaly patterns using ensemble-based isolation |
| Local Outlier Factor | Detects local density anomalies |
| KNN Label Consistency | Detects label flipping through neighborhood label mismatch |
| Hybrid | Combines detector outputs through voting |

## API Endpoints

- `GET /health` - service status.
- GET /methods - supported detection methods.
- GET /datasets - supported built-in real-world datasets.
- `POST /analyze` - multi-method numeric anomaly analysis.
- `POST /clean` - removes suspicious samples and returns cleaned data.
- `POST /analyze/labels` - label flipping detection with KNN consistency.
- `GET /demo/end-to-end` - real dataset -> poisoning -> API detection -> cleaned data.
- `GET /demo/real-dataset-results` - quantitative results for real-world dataset scenarios.

## Run Locally

```powershell
cd C:\pycharm\Diploma
.\.venv\Scripts\activate
uvicorn app.main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Run Experiments

```powershell
cd C:\pycharm\Diploma
.\.venv\Scripts\python.exe experiments\run_real_dataset_experiment.py
.\.venv\Scripts\python.exe experiments\run_threshold_tuning.py
.\.venv\Scripts\python.exe experiments\run_deep_learning_poisoning.py
```

Generated result files:

- `real_dataset_results.csv`
- `complex_subtle_threshold_tuning.csv`
- `deep_learning_poisoning_results.csv`

## Architecture

See:

- `docs/architecture.md`

Short flow:

```text
Data Sources -> Poisoning Simulation -> FastAPI Detection Service
-> Preprocessing -> Detection Layer -> Hybrid Decision Engine
-> Suspicious Indices / Cleaned Data -> Downstream ML Pipeline
```

## Diploma Positioning

The project can be described as an API-based security layer for machine learning pipelines. It detects suspicious samples using statistical, machine learning-based, and neighborhood consistency methods, evaluates attacks using quantitative metrics, and returns cleaned data for downstream model training.

