"""
models/weasel.py — WEASEL wrapper (nn-dataset Net pattern).
IS_TSC = True  -> loaded by models/__init__.py as an aeon classifier.
"""
import numpy as np
import joblib

MODEL_NAME  = "weasel"
MODEL_CLASS = "WEASELNet"
IS_TSC      = True
MAX_TRAIN_SAMPLES = 50000


class WEASELNet:
    """Net wrapper around aeon WEASEL."""

    def __init__(self, ts_len: int, num_classes: int, **kwargs):
        from aeon.classification.dictionary_based import WEASEL
        self.ts_len      = ts_len
        self.num_classes = num_classes
        hp = self.supported_hyperparameters()[0].copy()
        hp.update({k: v for k, v in kwargs.items() if k in hp})
        self._clf = WEASEL(**hp)
        self._classes = None

    @classmethod
    def supported_hyperparameters(cls):
        return [{"window_inc": 4, "n_jobs": 8}]

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
