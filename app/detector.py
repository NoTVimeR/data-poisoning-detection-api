from __future__ import annotations

from typing import List, Dict, Any
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors


class DetectorService:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _to_array(values: List[List[float]]) -> np.ndarray:
        arr = np.array(values, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Input values must be a 2D array-like structure.")
        if arr.shape[0] == 0:
            raise ValueError("Input values must not be empty.")
        return arr

    @staticmethod
    def _scale(X: np.ndarray) -> np.ndarray:
        scaler = StandardScaler()
        return scaler.fit_transform(X)

    @staticmethod
    def z_score_detect(X_scaled: np.ndarray, threshold: float = 2.0) -> np.ndarray:
        # Use max absolute z-score across features
        z_scores = np.max(np.abs(X_scaled), axis=1)
        return (z_scores > threshold).astype(int)

    @staticmethod
    def isolation_forest_detect(
        X_scaled: np.ndarray,
        contamination: float = 0.08,
        n_estimators: int = 300
    ) -> np.ndarray:
        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )
        preds = model.fit_predict(X_scaled)
        return np.where(preds == -1, 1, 0)

    @staticmethod
    def lof_detect(
        X_scaled: np.ndarray,
        contamination: float = 0.08,
        n_neighbors: int = 20
    ) -> np.ndarray:
        # LOF requires n_neighbors < n_samples
        n_neighbors = max(2, min(n_neighbors, len(X_scaled) - 1))
        model = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=n_neighbors
        )
        preds = model.fit_predict(X_scaled)
        return np.where(preds == -1, 1, 0)

    @staticmethod
    def knn_label_consistency(
        X_scaled: np.ndarray,
        labels: List[int],
        k_neighbors: int = 5
    ) -> np.ndarray:
        y = np.array(labels, dtype=int)
        if len(y) != len(X_scaled):
            raise ValueError("labels length must match number of samples")

        k_neighbors = max(2, min(k_neighbors, len(X_scaled) - 1))
        nn = NearestNeighbors(n_neighbors=k_neighbors + 1)
        nn.fit(X_scaled)

        _, indices = nn.kneighbors(X_scaled)

        suspicious = []
        for i in range(len(X_scaled)):
            neighbor_indices = indices[i][1:]  # exclude self
            neighbor_labels = y[neighbor_indices]
            majority_label = np.bincount(neighbor_labels).argmax()
            suspicious.append(1 if y[i] != majority_label else 0)

        return np.array(suspicious)

    def analyze_numeric(
        self,
        values: List[List[float]],
        methods: List[str] | None = None,
        z_threshold: float = 2.0,
        contamination: float = 0.08,
        n_estimators: int = 300,
        lof_neighbors: int = 20
    ) -> Dict[str, Any]:
        X = self._to_array(values)
        X_scaled = self._scale(X)

        if methods is None or len(methods) == 0:
            methods = ["z_score", "isolation_forest", "lof", "hybrid"]

        z_flags = np.zeros(len(X), dtype=int)
        if_flags = np.zeros(len(X), dtype=int)
        lof_flags = np.zeros(len(X), dtype=int)

        if "z_score" in methods or "hybrid" in methods:
            z_flags = self.z_score_detect(X_scaled, threshold=z_threshold)

        if "isolation_forest" in methods or "hybrid" in methods:
            if_flags = self.isolation_forest_detect(
                X_scaled,
                contamination=contamination,
                n_estimators=n_estimators
            )

        if "lof" in methods or "hybrid" in methods:
            lof_flags = self.lof_detect(
                X_scaled,
                contamination=contamination,
                n_neighbors=lof_neighbors
            )

        if "hybrid" in methods:
            # Majority vote among available detectors
            stacked = np.vstack([z_flags, if_flags, lof_flags])
            final_flags = (np.sum(stacked, axis=0) >= 2).astype(int)
            confidence = np.sum(stacked, axis=0) / 3.0
        else:
            # If no hybrid, combine any enabled detectors using OR
            enabled = []
            if "z_score" in methods:
                enabled.append(z_flags)
            if "isolation_forest" in methods:
                enabled.append(if_flags)
            if "lof" in methods:
                enabled.append(lof_flags)

            if enabled:
                stacked = np.vstack(enabled)
                final_flags = (np.sum(stacked, axis=0) >= 1).astype(int)
                confidence = np.sum(stacked, axis=0) / len(enabled)
            else:
                final_flags = np.zeros(len(X), dtype=int)
                confidence = np.zeros(len(X), dtype=float)

        results = []
        suspicious_indices = []

        for i in range(len(X)):
            item = {
                "index": i,
                "value": X[i].tolist(),
                "z_score_flag": bool(z_flags[i]),
                "isolation_forest_flag": bool(if_flags[i]),
                "lof_flag": bool(lof_flags[i]),
                "final_flag": bool(final_flags[i]),
                "confidence": round(float(confidence[i]), 2),
            }
            results.append(item)
            if final_flags[i] == 1:
                suspicious_indices.append(i)

        cleaned_data = [X[i].tolist() for i in range(len(X)) if final_flags[i] == 0]

        summary = {
            "total_samples": int(len(X)),
            "suspicious_count": int(np.sum(final_flags)),
            "anomaly_ratio": round(float(np.sum(final_flags) / len(X)), 4),
            "enabled_methods": methods,
        }

        return {
            "summary": summary,
            "results": results,
            "suspicious_indices": suspicious_indices,
            "cleaned_data": cleaned_data,
        }

    def clean_numeric(
        self,
        values: List[List[float]],
        methods: List[str] | None = None,
        z_threshold: float = 2.0,
        contamination: float = 0.08,
        n_estimators: int = 300,
        lof_neighbors: int = 20
    ) -> Dict[str, Any]:
        analysis = self.analyze_numeric(
            values=values,
            methods=methods,
            z_threshold=z_threshold,
            contamination=contamination,
            n_estimators=n_estimators,
            lof_neighbors=lof_neighbors,
        )

        return {
            "total_samples": analysis["summary"]["total_samples"],
            "removed_samples": analysis["summary"]["suspicious_count"],
            "cleaned_data": analysis["cleaned_data"],
            "suspicious_indices": analysis["suspicious_indices"],
        }

    def analyze_labels(
        self,
        values: List[List[float]],
        labels: List[int],
        k_neighbors: int = 5
    ) -> Dict[str, Any]:
        X = self._to_array(values)
        X_scaled = self._scale(X)
        suspicious = self.knn_label_consistency(
            X_scaled,
            labels=labels,
            k_neighbors=k_neighbors
        )

        results = []
        suspicious_indices = []

        for i in range(len(X)):
            item = {
                "index": i,
                "value": X[i].tolist(),
                "label": int(labels[i]),
                "suspicious": bool(suspicious[i]),
            }
            results.append(item)
            if suspicious[i] == 1:
                suspicious_indices.append(i)

        return {
            "total_samples": int(len(X)),
            "suspicious_count": int(np.sum(suspicious)),
            "suspicious_indices": suspicious_indices,
            "results": results,
        }