"""
models/hydra_multirocket.py

MultiRocketHydra classifier — combines MultiRocket's random convolution kernels
with Hydra's grouped convolution dictionary. In aeon benchmarks this is one of
the strongest and fastest SOTA TSC methods (top-3 on UCR/UEA archives).

Multivariate: accepts (N, C, L) where C >= 1.
IS_TSC = True — managed by the aeon training loop in train.py.
"""
import numpy as np
import joblib

MODEL_NAME        = "hydra_multirocket"
MODEL_CLASS       = "HydraMultiRocketNet"
IS_TSC            = True
MAX_TRAIN_SAMPLES = 100000   # feature transform is O(n); 100k fits in 60G


class HydraMultiRocketNet:
    """Thin wrapper around aeon MultiRocketHydraClassifier."""

    def __init__(self, ts_len: int, num_classes: int,
                 n_kernels: int = 6250, n_groups: int = 64,
                 n_jobs: int = 1, **kwargs):
        from aeon.classification.convolution_based import MultiRocketHydraClassifier
        self.ts_len      = ts_len
        self.num_classes = num_classes
        self._clf = MultiRocketHydraClassifier(
            n_kernels=n_kernels,
            n_groups=n_groups,
            n_jobs=n_jobs,
        )

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
