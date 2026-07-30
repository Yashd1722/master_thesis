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
import gc
import resource
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV

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
        self.n_jobs      = n_jobs
        self.cfg         = cfg

        if name in ("minirocket", "rocket", "multirocket"):
            self.use_custom_pipeline = True
            self._transformer = None
            self._scaler = None
            self._estimator = None
        else:
            self.use_custom_pipeline = False
            self._clf        = _build_classifier(name, n_jobs=n_jobs, **cfg)

    def _prep(self, X):
        if self.univariate and X.ndim == 3:
            X = X[:, 0, :]          # keep raw-residual channel only
        return X.astype(np.float64) if self.float64 else X

    def fit(self, X, y):
        X = self._prep(X)
        if self.use_custom_pipeline:
            # Print initial memory
            initial_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
            print(f"[{self.name}] Fitting started. Initial RSS: {initial_rss:.2f} GB")

            # 1. Instantiate transformer
            if self.name == "minirocket":
                from aeon.transformations.collection.convolution_based import MiniRocket
                self._transformer = MiniRocket(
                    n_kernels=self.cfg.get("n_kernels", 10000),
                    n_jobs=self.n_jobs,
                    random_state=self.cfg.get("random_state", None)
                )
            elif self.name == "rocket":
                from aeon.transformations.collection.convolution_based import Rocket
                self._transformer = Rocket(
                    n_kernels=self.cfg.get("n_kernels", 10000),
                    n_jobs=self.n_jobs,
                    random_state=self.cfg.get("random_state", None)
                )
            elif self.name == "multirocket":
                from aeon.transformations.collection.convolution_based import MultiRocket
                self._transformer = MultiRocket(
                    n_kernels=self.cfg.get("n_kernels", 6250),
                    n_jobs=self.n_jobs,
                    random_state=self.cfg.get("random_state", None)
                )

            # 2. Fit transformer
            self._transformer.fit(X, y)

            # 3. Transform in chunks
            chunk_size = 5000
            X_trans_parts = []
            for i in range(0, len(X), chunk_size):
                chunk = X[i : i + chunk_size]
                chunk_trans = self._transformer.transform(chunk).astype(np.float32)
                X_trans_parts.append(chunk_trans)
                del chunk_trans
                gc.collect()

                # Check RSS
                current_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
                if current_rss > 55.0:
                    print(f"[{self.name}] RSS {current_rss:.2f} GB exceeded 55 GB budget! Aborting gracefully.")
                    raise MemoryError("RSS exceeded 55 GB budget.")

            X_trans = np.concatenate(X_trans_parts, axis=0)
            del X_trans_parts
            gc.collect()

            # 4. Standard Scaler
            self._scaler = StandardScaler(with_mean=False)
            X_trans = self._scaler.fit_transform(X_trans)

            # 5. Fit Estimator (RidgeClassifierCV)
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)

            self._estimator = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            self._estimator.fit(X_trans, y)

            final_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
            print(f"[{self.name}] Fitting completed. Peak RSS: {final_rss:.2f} GB")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._clf.fit(X, y)
                self.classes_ = getattr(self._clf, "classes_", np.unique(y))
                self.n_classes_ = getattr(self._clf, "n_classes_", len(self.classes_))
        return self

    def predict_proba(self, X):
        X = self._prep(X)
        if self.use_custom_pipeline:
            chunk_size = 5000
            X_trans_parts = []
            for i in range(0, len(X), chunk_size):
                chunk = X[i : i + chunk_size]
                chunk_trans = self._transformer.transform(chunk).astype(np.float32)
                X_trans_parts.append(chunk_trans)
                del chunk_trans
                gc.collect()
            X_trans = np.concatenate(X_trans_parts, axis=0)
            X_trans = self._scaler.transform(X_trans)

            preds = self._estimator.predict(X_trans)
            dists = np.zeros((X.shape[0], self.n_classes_))
            for i in range(0, X.shape[0]):
                idx = np.where(self.classes_ == preds[i])[0]
                if len(idx) > 0:
                    dists[i, idx[0]] = 1.0
            return dists
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self._clf.predict_proba(X)

    def __setstate__(self, state):
        # Checkpoints pickled before use_custom_pipeline existed hold a plain aeon
        # ._clf for every model, including the rocket family. Absence of
        # _transformer is what marks them.
        state.setdefault("use_custom_pipeline", "_transformer" in state)
        self.__dict__.update(state)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
