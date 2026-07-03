"""
models/drcif.py

DrCIF — Diverse Representation Canonical Interval Forest.
Middlehurst et al. (2021). Interval-based ensemble; strong on short/noisy series
like the Bury residuals.

Multivariate: accepts (N, C, L) where C >= 1.
IS_TSC = True — managed by the aeon training loop in train.py.
"""
import numpy as np
import joblib

MODEL_NAME        = "drcif"
MODEL_CLASS       = "DrCIFNet"
IS_TSC            = True
MAX_TRAIN_SAMPLES = 20000    # 50k exceeded 4h wall time for ts_1500 on large_cpu


class DrCIFNet:
    """Thin wrapper around aeon DrCIFClassifier."""

    def __init__(self, ts_len: int, num_classes: int,
                 n_estimators: int = 100, n_jobs: int = 1, **kwargs):
        from aeon.classification.interval_based import DrCIFClassifier
        self.ts_len      = ts_len
        self.num_classes = num_classes
        self._clf = DrCIFClassifier(n_estimators=n_estimators, n_jobs=n_jobs)

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
