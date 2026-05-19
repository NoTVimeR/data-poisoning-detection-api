from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.attacks import (
    inject_complex_subtle_poisoning,
    inject_label_flipping,
    inject_simple_outliers,
    load_real_breast_cancer_dataset,
)
from app.detector import DetectorService
from app.evaluation import evaluate_flags


def main() -> None:
    detector = DetectorService()
    x, y, _ = load_real_breast_cancer_dataset()
    scenarios = [
        inject_simple_outliers(x, count=50),
        inject_complex_subtle_poisoning(x, count=60),
        inject_label_flipping(x, y, flip_fraction=0.12),
    ]

    rows = []
    for scenario in scenarios:
        if "Label" in scenario.description:
            analysis = detector.analyze_labels(values=scenario.values, labels=scenario.labels)
            predicted = [0] * len(scenario.values)
            for index in analysis["suspicious_indices"]:
                predicted[index] = 1
            rows.append({
                "dataset": "Breast Cancer Wisconsin",
                "scenario": scenario.description,
                "method": "knn_label_consistency",
                **evaluate_flags(scenario.poisoning_labels, predicted),
                "detected_samples": len(analysis["suspicious_indices"]),
            })
            continue

        for method in ["z_score", "isolation_forest", "lof", "hybrid"]:
            analysis = detector.analyze_numeric(values=scenario.values, methods=[method], contamination=0.10)
            predicted = [1 if item["final_flag"] else 0 for item in analysis["results"]]
            rows.append({
                "dataset": "Breast Cancer Wisconsin",
                "scenario": scenario.description,
                "method": method,
                **evaluate_flags(scenario.poisoning_labels, predicted),
                "detected_samples": len(analysis["suspicious_indices"]),
            })

    results = pd.DataFrame(rows)
    print(results.to_string(index=False))
    results.to_csv("real_dataset_results.csv", index=False)
    print("\nSaved as real_dataset_results.csv")


if __name__ == "__main__":
    main()
