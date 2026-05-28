"""
models/ts_chief.py — RISTClassifier wrapper (nn-dataset Net pattern).
IS_TSC = True  -> loaded by models/__init__.py as an aeon classifier.

Note: aeon does not implement TS-CHIEF directly. RISTClassifier is the
closest available hybrid interval+shapelet ensemble in aeon and is used
as a strong TS-CHIEF substitute.
"""
import numpy as np
import joblib

MODEL_NAME  = "ts_chief"
MODEL_CLASS = "TSCHIEFNet"
IS_TSC      = True
MAX_TRAIN_SAMPLES = 50000


class TSCHIEFNet:
    """Net wrapper around aeon RISTClassifier (TS-CHIEF substitute)."""

    def __init__(self, ts_len: int, num_classes: int, **kwargs):
        from aeon.classification.hybrid import RISTClassifier
        self.ts_len      = ts_len
        self.num_classes = num_classes
        hp = self.supported_hyperparameters()[0].copy()
        hp.update({k: v for k, v in kwargs.items() if k in hp})
        self._clf = RISTClassifier(**hp)
        self._classes = None

    @classmethod
    def supported_hyperparameters(cls):
        # n_intervals and n_shapelets default to None (auto) in RISTClassifier
        return [{"n_jobs": 8}]

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
