from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetBundle:
    values: List[List[float]]
    labels: List[int]
    poisoning_labels: List[int]
    description: str


def load_real_breast_cancer_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    dataset = load_breast_cancer()
    return dataset.data.astype(float), dataset.target.astype(int), list(dataset.feature_names)


def make_synthetic_dataset(samples: int = 200, features: int = 6, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    x, y = make_classification(
        n_samples=samples,
        n_features=features,
        n_informative=max(2, min(features, features - 1)),
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.8,
        random_state=random_state,
    )
    return x.astype(float), y.astype(int)


def inject_simple_outliers(
    x: np.ndarray,
    count: int = 50,
    magnitude: float = 6.0,
    random_state: int = 42,
) -> DatasetBundle:
    rng = np.random.default_rng(random_state)
    clean = np.asarray(x, dtype=float)
    center = clean.mean(axis=0)
    scale = clean.std(axis=0) + 1e-8
    poisoned = center + rng.normal(loc=magnitude, scale=1.0, size=(count, clean.shape[1])) * scale
    features = np.vstack([clean, poisoned])
    poisoning_labels = np.r_[np.zeros(clean.shape[0], dtype=int), np.ones(count, dtype=int)]
    return DatasetBundle(
        values=features.round(6).tolist(),
        labels=[0] * len(features),
        poisoning_labels=poisoning_labels.astype(int).tolist(),
        description="Simple outlier attack on real-world data",
    )


def inject_complex_subtle_poisoning(
    x: np.ndarray,
    count: int = 60,
    shift: float = 0.45,
    correlation_strength: float = 0.85,
    random_state: int = 42,
) -> DatasetBundle:
    rng = np.random.default_rng(random_state)
    clean = StandardScaler().fit_transform(np.asarray(x, dtype=float))
    count = min(count, max(5, clean.shape[0] // 4))
    base_indices = rng.choice(clean.shape[0], size=count, replace=False)
    poisoned = clean[base_indices].copy()

    signal = poisoned[:, 0] * correlation_strength + rng.normal(0, 0.08, size=count)
    poisoned[:, 0] = poisoned[:, 0] + shift
    if poisoned.shape[1] > 1:
        poisoned[:, 1] = signal
    if poisoned.shape[1] > 3:
        poisoned[:, 2:4] = poisoned[:, 2:4] + rng.normal(0.25, 0.12, size=(count, 2))

    features = np.vstack([clean, poisoned])
    poisoning_labels = np.r_[np.zeros(clean.shape[0], dtype=int), np.ones(count, dtype=int)]
    return DatasetBundle(
        values=features.round(6).tolist(),
        labels=[0] * len(features),
        poisoning_labels=poisoning_labels.astype(int).tolist(),
        description="Complex subtle poisoning with correlation drift",
    )


def inject_label_flipping(
    x: np.ndarray,
    y: np.ndarray,
    flip_fraction: float = 0.12,
    random_state: int = 42,
) -> DatasetBundle:
    rng = np.random.default_rng(random_state)
    features = np.asarray(x, dtype=float)
    labels = np.asarray(y, dtype=int).copy()
    count = max(1, int(len(labels) * flip_fraction))
    flip_indices = rng.choice(len(labels), size=count, replace=False)
    classes = np.unique(labels)
    if len(classes) == 2:
        labels[flip_indices] = 1 - labels[flip_indices]
    else:
        for index in flip_indices:
            choices = classes[classes != labels[index]]
            labels[index] = int(rng.choice(choices))

    poisoning_labels = np.zeros(len(labels), dtype=int)
    poisoning_labels[flip_indices] = 1
    return DatasetBundle(
        values=features.round(6).tolist(),
        labels=labels.astype(int).tolist(),
        poisoning_labels=poisoning_labels.astype(int).tolist(),
        description="Label flipping attack on real-world data",
    )
