"""
models/minirocket.py

MiniRocket — deterministic, extremely fast variant of ROCKET.
Dempster et al. (2021). Near-identical accuracy to ROCKET at 75x less compute.

Multivariate: accepts (N, C, L) where C >= 1.
IS_TSC = True — managed by the aeon training loop in train.py.
"""
import numpy as np
import joblib

MODEL_NAME        = "minirocket"
MODEL_CLASS       = "MiniRocketNet"
IS_TSC            = True
MAX_TRAIN_SAMPLES = 100000


class MiniRocketNet:
    """Thin wrapper around aeon MiniRocketClassifier."""

    def __init__(self, ts_len: int, num_classes: int,
                 n_kernels: int = 10000, n_jobs: int = 1, **kwargs):
        from aeon.classification.convolution_based import MiniRocketClassifier
        self.ts_len      = ts_len
        self.num_classes = num_classes
        self._clf = MiniRocketClassifier(n_kernels=n_kernels, n_jobs=n_jobs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """X: (N, C, L) or (N, L). y: (N,) int."""
        self._clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns (N, num_classes) probability matrix."""
        return self._clf.predict_proba(X)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
