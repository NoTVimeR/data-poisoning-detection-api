from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.attacks import inject_complex_subtle_poisoning, load_real_breast_cancer_dataset
from app.detector import DetectorService
from app.evaluation import evaluate_flags


def main() -> None:
    detector = DetectorService()
    x, _, _ = load_real_breast_cancer_dataset()
    scenario = inject_complex_subtle_poisoning(x, count=60)
    rows = []

    for contamination in [0.05, 0.08, 0.10, 0.15]:
        for method in ["isolation_forest", "lof", "hybrid"]:
            analysis = detector.analyze_numeric(values=scenario.values, methods=[method], contamination=contamination)
            predicted = [1 if item["final_flag"] else 0 for item in analysis["results"]]
            rows.append({
                "scenario": scenario.description,
                "method": method,
                "contamination": contamination,
                **evaluate_flags(scenario.poisoning_labels, predicted),
                "detected_samples": len(analysis["suspicious_indices"]),
            })

    results = pd.DataFrame(rows)
    print(results.to_string(index=False))
    results.to_csv("complex_subtle_threshold_tuning.csv", index=False)
    print("\nSaved as complex_subtle_threshold_tuning.csv")


if __name__ == "__main__":
    main()
