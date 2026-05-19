from fastapi import FastAPI, HTTPException
from app.schemas import (
    AnalyzeRequest,
    CleanRequest,
    LabelAnalyzeRequest,
    HealthResponse,
    AnalyzeResponse,
    CleanResponse,
    LabelAnalyzeResponse,
    DemoResponse,
)
from app.detector import DetectorService
from app.attacks import (
    inject_complex_subtle_poisoning,
    inject_label_flipping,
    inject_simple_outliers,
    load_real_breast_cancer_dataset,
)
from app.evaluation import evaluate_flags

app = FastAPI(
    title="AI Data Poisoning Detection System v2",
    description="Multi-method API for detecting and preventing data poisoning attacks in ML pipelines",
    version="2.1.0",
)

detector = DetectorService()


@app.get("/", tags=["General"])
def root():
    return {
        "message": "AI Data Poisoning Detection System v2 is running",
        "docs": "/docs",
        "end_to_end_demo": "/demo/end-to-end",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    return {
        "status": "ok",
        "service": "AI Data Poisoning Detection System v2"
    }


@app.get("/methods", tags=["General"])
def methods():
    return {
        "methods": [
            {
                "name": "z_score",
                "purpose": "Detects obvious outliers using statistical deviation"
            },
            {
                "name": "isolation_forest",
                "purpose": "Detects subtle anomalies using an ensemble-based ML method"
            },
            {
                "name": "lof",
                "purpose": "Detects local density anomalies"
            },
            {
                "name": "hybrid",
                "purpose": "Combines multiple detectors using majority voting"
            },
            {
                "name": "knn_label_consistency",
                "purpose": "Detects label flipping by checking local label consistency"
            }
        ]
    }


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Numeric Analysis"])
def analyze(request: AnalyzeRequest):
    try:
        return detector.analyze_numeric(
            values=request.values,
            methods=request.methods,
            z_threshold=request.z_threshold,
            contamination=request.contamination,
            n_estimators=request.n_estimators,
            lof_neighbors=request.lof_neighbors,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/clean", response_model=CleanResponse, tags=["Numeric Analysis"])
def clean(request: CleanRequest):
    try:
        return detector.clean_numeric(
            values=request.values,
            methods=request.methods,
            z_threshold=request.z_threshold,
            contamination=request.contamination,
            n_estimators=request.n_estimators,
            lof_neighbors=request.lof_neighbors,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/analyze/labels", response_model=LabelAnalyzeResponse, tags=["Label Analysis"])
def analyze_labels(request: LabelAnalyzeRequest):
    try:
        return detector.analyze_labels(
            values=request.values,
            labels=request.labels,
            k_neighbors=request.k_neighbors,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/demo/end-to-end", response_model=DemoResponse, tags=["Demo"])
def demo_end_to_end():
    x, _, _ = load_real_breast_cancer_dataset()
    scenario = inject_complex_subtle_poisoning(x, count=60)
    analysis = detector.analyze_numeric(
        values=scenario.values,
        methods=["hybrid"],
        contamination=0.10,
    )
    predicted = [1 if item["final_flag"] else 0 for item in analysis["results"]]
    metrics = evaluate_flags(scenario.poisoning_labels, predicted)
    return {
        "dataset": "Breast Cancer Wisconsin",
        "scenario": scenario.description,
        "rows_before_cleaning": len(scenario.values),
        "rows_after_cleaning": len(analysis["cleaned_data"]),
        "metrics": metrics,
        "suspicious_indices_preview": analysis["suspicious_indices"][:20],
    }


@app.get("/demo/real-dataset-results", tags=["Demo"])
def demo_real_dataset_results():
    x, y, _ = load_real_breast_cancer_dataset()
    scenarios = [
        inject_simple_outliers(x, count=50),
        inject_complex_subtle_poisoning(x, count=60),
        inject_label_flipping(x, y, flip_fraction=0.12),
    ]
    rows = []
    for scenario in scenarios:
        if "Label" in scenario.description:
            label_analysis = detector.analyze_labels(values=scenario.values, labels=scenario.labels)
            predicted = [0] * len(scenario.values)
            for index in label_analysis["suspicious_indices"]:
                predicted[index] = 1
            rows.append({
                "scenario": scenario.description,
                "method": "knn_label_consistency",
                **evaluate_flags(scenario.poisoning_labels, predicted),
                "detected_samples": len(label_analysis["suspicious_indices"]),
            })
            continue

        for method in ["z_score", "isolation_forest", "lof", "hybrid"]:
            analysis = detector.analyze_numeric(values=scenario.values, methods=[method], contamination=0.10)
            predicted = [1 if item["final_flag"] else 0 for item in analysis["results"]]
            rows.append({
                "scenario": scenario.description,
                "method": method,
                **evaluate_flags(scenario.poisoning_labels, predicted),
                "detected_samples": len(analysis["suspicious_indices"]),
            })
    return {"dataset": "Breast Cancer Wisconsin", "results": rows}
