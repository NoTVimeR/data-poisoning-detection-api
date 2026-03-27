from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.models.detector import PoisoningDetector

app = FastAPI(
    title="AI Data Poisoning Detection System",
    description="Simple prototype for pre-defense demo",
)

detector = PoisoningDetector()

class DataInput(BaseModel):
    values: List[float]

@app.post("/detect")
def detect(data: DataInput):
    detector.fit(data.values)

    results = detector.detect(data.values)

    total = len(results)
    anomalies = sum(1 for r in results if r["is_suspicious"])

    return {
        "results": results,
        "summary": {
            "total_samples": total,
            "anomalies_detected": anomalies,
            "anomaly_ratio": anomalies / total if total > 0 else 0
        }
    }

@app.post("/clean")
def clean(data: DataInput):
    results = detector.detect(data.values)

    clean_data = [r["value"] for r in results if not r["is_suspicious"]]

    return {
        "clean_data": clean_data,
        "removed_samples": len(data.values) - len(clean_data)
    }