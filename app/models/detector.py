import numpy as np
from sklearn.ensemble import IsolationForest

class PoisoningDetector:
    def __init__(self, threshold=2.5):
        self.threshold = threshold
        self.model = IsolationForest(contamination=0.1, random_state=42)

    def fit(self, data):
        data = np.array(data).reshape(-1, 1)
        self.model.fit(data)

    def detect(self, data):
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)

        # z-score
        z_results = []
        for x in data:
            z_score = abs((x - mean) / std)
            z_results.append(z_score)

        # Isolation Forest
        data_reshaped = data.reshape(-1, 1)
        iso_preds = self.model.predict(data_reshaped)
        iso_preds = np.where(iso_preds == -1, 1, 0)

        results = []
        for i, x in enumerate(data):
            results.append({
                "value": float(x),
                "z_score": float(z_results[i]),
                "z_suspicious": z_results[i] > self.threshold,
                "iso_suspicious": bool(iso_preds[i])
            })

        return results