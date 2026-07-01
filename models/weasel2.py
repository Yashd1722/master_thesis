"""
models/weasel2.py

WEASEL-D (WEASEL_V2) — replaces the deleted BOSS, WEASEL, and TDE classifiers.
WEASEL_V2 uses a learned dictionary of discriminative symbolic words with
first-difference features. It is univariate; multivariate input is handled by
taking the first channel (the raw residual, channel 0).

IS_TSC = True — managed by the aeon training loop in train.py.
"""
import warnings
import numpy as np
import joblib

MODEL_NAME        = "weasel2"
MODEL_CLASS       = "WEASEL2Net"
IS_TSC            = True
MAX_TRAIN_SAMPLES = 50000


class WEASEL2Net:
    """Thin wrapper around aeon WEASEL_V2 (univariate)."""

    def __init__(self, ts_len: int, num_classes: int,
                 n_jobs: int = 1, **kwargs):
        from aeon.classification.dictionary_based import WEASEL_V2
        self.ts_len      = ts_len
        self.num_classes = num_classes
        self._clf = WEASEL_V2(n_jobs=n_jobs)

    def _to_univariate(self, X: np.ndarray) -> np.ndarray:
        """
        WEASEL_V2 is univariate. If X is (N, C, L), take channel 0 (raw residual).
        If X is already (N, L), pass through.
        """
        if X.ndim == 3:
            return X[:, 0, :]   # (N, L) — raw residual channel
        return X                 # already (N, L)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """X: (N, C, L) or (N, L). y: (N,) int."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*liblinear.*", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*n_jobs.*liblinear.*", category=UserWarning)
            self._clf.fit(self._to_univariate(X), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns (N, num_classes) probability matrix."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*liblinear.*", category=FutureWarning)
            return self._clf.predict_proba(self._to_univariate(X))

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
