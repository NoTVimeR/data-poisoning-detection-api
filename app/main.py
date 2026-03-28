from fastapi import FastAPI, HTTPException
from app.schemas import (
    AnalyzeRequest,
    CleanRequest,
    LabelAnalyzeRequest,
    HealthResponse,
    AnalyzeResponse,
    CleanResponse,
    LabelAnalyzeResponse,
)
from app.detector import DetectorService

app = FastAPI(
    title="AI Data Poisoning Detection System v2",
    description="Multi-method API for detecting and preventing data poisoning attacks in ML pipelines",
    version="2.0.0",
)

detector = DetectorService()


@app.get("/", tags=["General"])
def root():
    return {
        "message": "AI Data Poisoning Detection System v2 is running",
        "docs": "/docs"
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