"""
models/tsc.py — every aeon TSC classifier behind one thin wrapper.

All TSC models share the same interface (fit / predict_proba / save / load), so
they need one wrapper, not eight near-identical files. To add a model: add a row
to TSC_SPECS and a branch to _build_classifier.

MAX_TRAIN_SAMPLES caps are memory-safety gates tuned for 60 GB CPU nodes.
`univariate` models take channel 0 only; `float64` models need a float64 cast.
"""
import warnings
import numpy as np
import joblib

# name -> (max_train_samples, univariate, float64)
# max_samples caps are memory-safety gates for 60 GB CPU nodes. With
# use_4channel the multivariate models carry ~4x the input, so arsenal and
# multirocket are cut below their 1-channel caps (arsenal OOM'd even at 1ch).
# weasel2 is univariate (channel 0 only) so 4-channel does not affect it.
TSC_SPECS = {
    "minirocket":  {"max_samples": 100000, "univariate": False, "float64": False},
    "rocket":      {"max_samples": 40000,  "univariate": False, "float64": False},
    "multirocket": {"max_samples": 20000,  "univariate": False, "float64": False},
    "arsenal":     {"max_samples": 20000,  "univariate": False, "float64": False},
    "rdst":        {"max_samples": 30000,  "univariate": False, "float64": True},
    "weasel2":     {"max_samples": 20000,  "univariate": True,  "float64": False},
    "drcif":       {"max_samples": 20000,  "univariate": False, "float64": False},
}


def _build_classifier(name, n_jobs=1, **cfg):
    """Instantiate the underlying aeon estimator for `name` (lazy imports)."""
    if name in ("minirocket", "rocket", "multirocket", "arsenal"):
        from aeon.classification.convolution_based import (
            MiniRocketClassifier, RocketClassifier,
            MultiRocketClassifier, Arsenal)
        if name == "minirocket":
            return MiniRocketClassifier(n_kernels=cfg.get("n_kernels", 10000), n_jobs=n_jobs)
        if name == "rocket":
            return RocketClassifier(n_kernels=cfg.get("n_kernels", 10000), n_jobs=n_jobs)
        if name == "multirocket":
            return MultiRocketClassifier(n_kernels=cfg.get("n_kernels", 6250), n_jobs=n_jobs)
        return Arsenal(n_kernels=cfg.get("num_kernels", 2000), n_jobs=n_jobs)
    if name == "rdst":
        from aeon.classification.shapelet_based import RDSTClassifier
        return RDSTClassifier(max_shapelets=cfg.get("max_shapelets", 10000), n_jobs=n_jobs)
    if name == "weasel2":
        from aeon.classification.dictionary_based import WEASEL_V2
        return WEASEL_V2(n_jobs=n_jobs)
    if name == "drcif":
        from aeon.classification.interval_based import DrCIFClassifier
        return DrCIFClassifier(n_estimators=cfg.get("n_estimators", 100), n_jobs=n_jobs)
    raise ValueError(f"Unknown TSC model '{name}'")


class TSCModel:
    """Uniform wrapper around any aeon classifier. Accepts (N, C, L) or (N, L)."""

    def __init__(self, name, ts_len, num_classes, n_jobs=1, **cfg):
        spec = TSC_SPECS[name]
        self.name        = name
        self.num_classes = num_classes
        self.univariate  = spec["univariate"]
        self.float64     = spec["float64"]
        self._clf        = _build_classifier(name, n_jobs=n_jobs, **cfg)

    def _prep(self, X):
        if self.univariate and X.ndim == 3:
            X = X[:, 0, :]          # keep raw-residual channel only
        return X.astype(np.float64) if self.float64 else X

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._clf.fit(self._prep(X), y)
        return self

    def predict_proba(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._clf.predict_proba(self._prep(X))

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
