"""
models/minirocket.py — MiniRocketClassifier wrapper (nn-dataset Net pattern).
IS_TSC = True  -> loaded by models/__init__.py as an aeon classifier.
"""
import numpy as np
import joblib

MODEL_NAME  = "minirocket"
MODEL_CLASS = "MiniRocketNet"
IS_TSC      = True
MAX_TRAIN_SAMPLES = 100000


class MiniRocketNet:
    """Net wrapper around aeon MiniRocketClassifier."""

    def __init__(self, ts_len: int, num_classes: int, **kwargs):
        from aeon.classification.convolution_based import MiniRocketClassifier
        self.ts_len      = ts_len
        self.num_classes = num_classes
        hp = self.supported_hyperparameters()[0].copy()
        hp.update({k: v for k, v in kwargs.items() if k in hp})
        self._clf = MiniRocketClassifier(**hp)
        self._classes = None

    @classmethod
    def supported_hyperparameters(cls):
        return [{"n_kernels": 10000, "n_jobs": 16}]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """X: (N, T) float32, y: (N,) int."""
        self._clf.fit(X, y)
        self._classes = np.unique(y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns (N, num_classes) probability matrix."""
        return self._clf.predict_proba(X)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
