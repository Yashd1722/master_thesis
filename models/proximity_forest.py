import numpy as np
import joblib

MODEL_NAME  = "proximity_forest"
MODEL_CLASS = "ProximityForestNet"
IS_TSC      = True
MAX_TRAIN_SAMPLES = 10000


class ProximityForestNet:
    def __init__(self, ts_len: int, num_classes: int, **kwargs):
        from aeon.classification.distance_based import ProximityForest
        self.ts_len      = ts_len
        self.num_classes = num_classes
        hp = self.supported_hyperparameters()[0].copy()
        hp.update({k: v for k, v in kwargs.items() if k in hp})
        self._clf = ProximityForest(**hp)
        self._classes = None

    @classmethod
    def supported_hyperparameters(cls):
        return [{"n_trees": 50, "n_jobs": 4}]

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._clf.fit(X, y)
        self._classes = np.unique(y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(X)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
