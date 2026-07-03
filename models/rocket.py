"""
models/rocket.py

ROCKET — Random Convolutional KErnel Transform classifier.
Parrish et al. (2020). Fast and surprisingly effective on UCR/UEA benchmarks.

Multivariate: accepts (N, C, L) where C >= 1.
IS_TSC = True — managed by the aeon training loop in train.py.
"""
import numpy as np
import joblib

MODEL_NAME        = "rocket"
MODEL_CLASS       = "RocketNet"
IS_TSC            = True
MAX_TRAIN_SAMPLES = 50000    # 10k kernels × 2 features; OOM above ~50k on 60G nodes


class RocketNet:
    """Thin wrapper around aeon RocketClassifier."""

    def __init__(self, ts_len: int, num_classes: int,
                 n_kernels: int = 10000, n_jobs: int = 1, **kwargs):
        from aeon.classification.convolution_based import RocketClassifier
        self.ts_len      = ts_len
        self.num_classes = num_classes
        self._clf = RocketClassifier(n_kernels=n_kernels, n_jobs=n_jobs)

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
