"""
models/rdst.py

RDST — Random Dilated Shapelet Transform classifier.
Replaces the deleted ShapeletTransformClassifier. RDST uses randomly sampled
shapelets at multiple dilations, making it both faster and more accurate than
classical shapelet search.

Multivariate: accepts (N, C, L) where C >= 1.
Requires float64 input (numba compiled kernel — will cast internally).
IS_TSC = True — managed by the aeon training loop in train.py.
"""
import numpy as np
import joblib

MODEL_NAME        = "rdst"
MODEL_CLASS       = "RDSTNet"
IS_TSC            = True
MAX_TRAIN_SAMPLES = 30000   # numba kernel is memory-intensive at large N


class RDSTNet:
    """Thin wrapper around aeon RDSTClassifier."""

    def __init__(self, ts_len: int, num_classes: int,
                 max_shapelets: int = 10000, n_jobs: int = 1, **kwargs):
        from aeon.classification.shapelet_based import RDSTClassifier
        self.ts_len      = ts_len
        self.num_classes = num_classes
        self._clf = RDSTClassifier(
            max_shapelets=max_shapelets,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """X: (N, C, L) or (N, L). y: (N,) int.
        Casts to float64 — RDST's numba kernel requires it."""
        self._clf.fit(X.astype(np.float64), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns (N, num_classes) probability matrix."""
        return self._clf.predict_proba(X.astype(np.float64))

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
