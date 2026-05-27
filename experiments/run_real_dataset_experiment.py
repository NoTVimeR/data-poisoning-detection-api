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
    load_real_dataset_catalog,
)
from app.detector import DetectorService
from app.evaluation import evaluate_flags


def evaluate_scenario(detector: DetectorService, dataset_name: str, scenario) -> list[dict[str, object]]:
    rows = []
    if "Label" in scenario.description:
        analysis = detector.analyze_labels(values=scenario.values, labels=scenario.labels)
        predicted = [0] * len(scenario.values)
        for index in analysis["suspicious_indices"]:
            predicted[index] = 1
        rows.append({
            "dataset": dataset_name,
            "scenario": scenario.description,
            "method": "knn_label_consistency",
            **evaluate_flags(scenario.poisoning_labels, predicted),
            "detected_samples": len(analysis["suspicious_indices"]),
        })
        return rows

    for method in ["z_score", "isolation_forest", "lof", "hybrid"]:
        analysis = detector.analyze_numeric(values=scenario.values, methods=[method], contamination=0.10)
        predicted = [1 if item["final_flag"] else 0 for item in analysis["results"]]
        rows.append({
            "dataset": dataset_name,
            "scenario": scenario.description,
            "method": method,
            **evaluate_flags(scenario.poisoning_labels, predicted),
            "detected_samples": len(analysis["suspicious_indices"]),
        })
    return rows


def main() -> None:
    detector = DetectorService()
    rows = []

    for dataset_name, (x, y, _) in load_real_dataset_catalog().items():
        poison_count = max(10, min(60, int(len(x) * 0.10)))
        scenarios = [
            inject_simple_outliers(x, count=poison_count),
            inject_complex_subtle_poisoning(x, count=poison_count),
            inject_label_flipping(x, y, flip_fraction=0.12),
        ]
        for scenario in scenarios:
            rows.extend(evaluate_scenario(detector, dataset_name, scenario))

    results = pd.DataFrame(rows)
    print(results.to_string(index=False))
    results.to_csv("real_dataset_results.csv", index=False)
    print("\nSaved as real_dataset_results.csv")


if __name__ == "__main__":
    main()
