from __future__ import annotations

from typing import Dict, List

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_flags(ground_truth: List[int], predicted: List[int]) -> Dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(ground_truth, predicted)), 4),
        "precision": round(float(precision_score(ground_truth, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(ground_truth, predicted, zero_division=0)), 4),
        "f1_score": round(float(f1_score(ground_truth, predicted, zero_division=0)), 4),
    }
