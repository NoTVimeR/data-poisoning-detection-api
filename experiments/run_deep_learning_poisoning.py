from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def poison_labels(y: np.ndarray, fraction: float = 0.15, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    poisoned = y.copy()
    count = int(len(y) * fraction)
    indices = rng.choice(len(y), size=count, replace=False)
    poisoned[indices] = (poisoned[indices] + 1) % 10
    return poisoned, indices


def train_mlp(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=80, random_state=42)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def main() -> None:
    digits = load_digits()
    x = StandardScaler().fit_transform(digits.data)
    y = digits.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=42)

    clean_predictions = train_mlp(x_train, y_train, x_test)
    poisoned_y_train, flipped_indices = poison_labels(y_train, fraction=0.15)
    poisoned_predictions = train_mlp(x_train, poisoned_y_train, x_test)

    results = pd.DataFrame([
        {
            "scenario": "clean training",
            "poisoned_training_labels": 0,
            "accuracy": round(float(accuracy_score(y_test, clean_predictions)), 4),
            "macro_f1": round(float(f1_score(y_test, clean_predictions, average="macro")), 4),
        },
        {
            "scenario": "deep learning label flipping",
            "poisoned_training_labels": len(flipped_indices),
            "accuracy": round(float(accuracy_score(y_test, poisoned_predictions)), 4),
            "macro_f1": round(float(f1_score(y_test, poisoned_predictions, average="macro")), 4),
        },
    ])
    print(results.to_string(index=False))
    results.to_csv("deep_learning_poisoning_results.csv", index=False)
    print("\nSaved as deep_learning_poisoning_results.csv")


if __name__ == "__main__":
    main()
