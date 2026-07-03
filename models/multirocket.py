"""
models/multirocket.py

MultiRocket — extends MiniRocket with additional feature types (mean, variance,
positive-proportion), improving accuracy by ~2-3% on UCR benchmarks.
Tan et al. (2022).

Multivariate: accepts (N, C, L) where C >= 1.
IS_TSC = True — managed by the aeon training loop in train.py.
"""
import numpy as np
import joblib

MODEL_NAME        = "multirocket"
MODEL_CLASS       = "MultiRocketNet"
IS_TSC            = True
MAX_TRAIN_SAMPLES = 40000   # 49728 features × 80k → dual-copy peak ~52GB OOM on 60G nodes


class MultiRocketNet:
    """Thin wrapper around aeon MultiRocketClassifier."""

    def __init__(self, ts_len: int, num_classes: int,
                 n_kernels: int = 6250, n_jobs: int = 1, **kwargs):
        from aeon.classification.convolution_based import MultiRocketClassifier
        self.ts_len      = ts_len
        self.num_classes = num_classes
        self._clf = MultiRocketClassifier(n_kernels=n_kernels, n_jobs=n_jobs)

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
