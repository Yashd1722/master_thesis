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
    "minirocket":  {"max_samples": 80000,  "univariate": False, "float64": False},
    "rocket":      {"max_samples": 40000,  "univariate": False, "float64": False},
    "multirocket": {"max_samples": 30000,  "univariate": False, "float64": False},
    "arsenal":     {"max_samples": 30000,  "univariate": False, "float64": False},
    "rdst":        {"max_samples": 30000,  "univariate": False, "float64": True},
    "weasel2":     {"max_samples": 20000,  "univariate": True,  "float64": False},
    "drcif":       {"max_samples": 20000,  "univariate": False, "float64": False},
    "catch22":     {"max_samples": 30000,  "univariate": True,  "float64": True},
    "tde":         {"max_samples": 10000,  "univariate": True,  "float64": False},
    "pf":          {"max_samples": 5000,   "univariate": True,  "float64": False},

    # --- Bake Off (Bagnall et al. 2017) baselines, added for the thesis
    # comparison chapter. All univariate=True: the paper benchmarks univariate
    # UCR series, and it keeps these already-expensive models off the 4x cost
    # of use_4channel.
    #
    # The paper's six whole-series elastic-distance 1-NN baselines (euclidean,
    # dtw, dtw_cv, wdtw, twe, msm) plus EE are deliberately NOT here: a 1-NN
    # classifier's "training set" IS its memory, so cost scales as
    # n_train * n_query * L^2 at predict time. At this project's real scale
    # (ts_500 train ~475k, val ~20k) that's ~10^15 operations per model —
    # days, not hours — regardless of hardware, and no amount of subsampling
    # avoids it without also shrinking accuracy. Dropped rather than shipped
    # as a crippled/subsampled result.
    #
    # The remaining five build a fixed-size model instead of memorizing every
    # training series, so full-dataset training is tractable. max_samples is
    # None (uncapped — use the whole train split); cost is bounded by each
    # classifier's own time_limit_in_minutes contract instead (see
    # _build_classifier). These minute budgets are starting points, not
    # measured wall-clock — tune them from the first real run.
    #
    # boss is the exception: ContractableBOSS._fit subsamples 70% of
    # whatever it's given internally (aeon's _cboss.py), so the
    # time_limit_in_minutes/n_parameter_samples contract doesn't bound the
    # per-candidate SFA transform's memory at all — it OOM'd/MemoryError'd
    # repeatedly at full-dataset scale (both ts_500 and ts_1500) regardless
    # of n_parameter_samples. max_train_samples in config.yaml overrides
    # this None to actually cap what boss trains on.
    "tsf":  {"max_samples": None, "univariate": True, "float64": False},
    # st runs ShapeletTransformClassifier's numba shapelet-distance kernel,
    # which needs float64 — same reason rdst is float64 above. STC's search
    # cost is ~linear in row count (measured on real ts_500 data: 14.7s +
    # 0.022s/row) regardless of time_limit_in_minutes=0 (see the "st"
    # config.yaml comment) — uncapped that's ~11.5h at real scale (1.89M
    # rows), past the 8h SLURM walltime (confirmed: both timed out — see
    # sacct history). 200000 keeps STC's own fit to ~1.2h.
    "st":   {"max_samples": 200000, "univariate": True, "float64": True},
    "ls":   {"max_samples": None, "univariate": True, "float64": False},
    "boss": {"max_samples": None, "univariate": True, "float64": False},

    # --- Bake Off models with no maintained-library implementation, built
    # from scratch for this thesis (models/bakeoff_*.py). bop/saxvsm wrap
    # pyts (tested community code, just needs a scipy compat shim — see
    # models/_pyts_compat.py); tsbf/lps/fastshapelet are original
    # implementations against their papers, with no reference to check
    # against. All validated on UCR GunPoint before use — see
    # models/test_bakeoff_scratch.py. None are 1-NN-over-everything, so all
    # are uncapped like the five above.
    "bop":     {"max_samples": None, "univariate": True, "float64": False},
    "saxvsm":  {"max_samples": None, "univariate": True, "float64": False},
    "tsbf":    {"max_samples": None, "univariate": True, "float64": False},
    "lps":     {"max_samples": None, "univariate": True, "float64": False},
    "fastshapelet": {"max_samples": None, "univariate": True, "float64": False},
    "cif":     {"max_samples": 20000, "univariate": True, "float64": False},
    "mrsqm":   {"max_samples": 20000, "univariate": True, "float64": False},
    "grsf":    {"max_samples": 15000, "univariate": True, "float64": True},
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
    if name == "tde":
        from aeon.classification.dictionary_based import TemporalDictionaryEnsemble
        return TemporalDictionaryEnsemble(
            n_parameter_samples=cfg.get("n_parameter_samples", 250), n_jobs=n_jobs)
    if name == "drcif":
        from aeon.classification.interval_based import DrCIFClassifier
        return DrCIFClassifier(n_estimators=cfg.get("n_estimators", 100), n_jobs=n_jobs)
    if name == "catch22":
        from aeon.classification.feature_based import Catch22Classifier
        return Catch22Classifier(estimator=None, n_jobs=n_jobs)
    if name == "pf":
        from aeon.classification.distance_based import ProximityForest
        return ProximityForest(n_trees=cfg.get("n_estimators", 100), n_jobs=n_jobs)

    if name == "tsf":
        from aeon.classification.interval_based import TimeSeriesForestClassifier
        return TimeSeriesForestClassifier(
            n_estimators=cfg.get("n_estimators", 200),
            time_limit_in_minutes=cfg.get("time_limit_in_minutes"),
            contract_max_n_estimators=cfg.get("contract_max_n_estimators", 500),
            n_jobs=n_jobs)
    if name == "st":
        from aeon.classification.shapelet_based import ShapeletTransformClassifier
        return ShapeletTransformClassifier(
            n_shapelet_samples=cfg.get("n_shapelet_samples", 10000),
            max_shapelets=cfg.get("max_shapelets", None),
            time_limit_in_minutes=cfg.get("time_limit_in_minutes", 0),
            n_jobs=n_jobs)
    if name == "ls":
        # Gradient-based (tslearn/keras under the hood), not memory-based like
        # the dropped 1-NN methods — cost is ~linear in N per epoch, so the
        # full train set is tractable; max_iter is the tunable knob here
        # instead of a time contract.
        from aeon.classification.shapelet_based import LearningShapeletClassifier
        return LearningShapeletClassifier(
            max_iter=cfg.get("max_iter", 100), batch_size=cfg.get("batch_size", 256))
    if name == "boss":
        # BOSSEnsemble itself has no time contract and does not scale to
        # full-dataset training. ContractableBOSS (cBOSS) is aeon's own
        # scalable replacement for exactly this case — same dictionary-based
        # family, bounded by time_limit_in_minutes instead of series count.
        from aeon.classification.dictionary_based import ContractableBOSS
        return ContractableBOSS(
            time_limit_in_minutes=cfg.get("time_limit_in_minutes", 90),
            max_ensemble_size=cfg.get("max_ensemble_size", 50),
            n_parameter_samples=cfg.get("n_parameter_samples", 250),
            n_jobs=n_jobs)
    if name == "cif":
        from aeon.classification.interval_based import CanonicalIntervalForestClassifier
        return CanonicalIntervalForestClassifier(
            n_estimators=cfg.get("n_estimators", 100),
            time_limit_in_minutes=cfg.get("time_limit_in_minutes"),
            contract_max_n_estimators=cfg.get("contract_max_n_estimators", 500),
            n_jobs=n_jobs)
    if name == "mrsqm":
        from aeon.classification.dictionary_based import MrSQMClassifier
        return MrSQMClassifier(random_state=cfg.get("random_state", 0))
    if name == "grsf":
        from wildboar.ensemble import ShapeletForestClassifier
        return ShapeletForestClassifier(
            n_estimators=cfg.get("n_estimators", 100),
            n_shapelets=cfg.get("n_shapelets", "log2"),
            metric=cfg.get("metric", "euclidean"),
            n_jobs=n_jobs, random_state=cfg.get("random_state", 0))
    if name == "bop":
        from models.bakeoff_dictionary import BOPClassifier
        return BOPClassifier(
            window_size=cfg.get("window_size", 0.5), word_size=cfg.get("word_size", 0.5),
            n_bins=cfg.get("n_bins", 4), numerosity_reduction=cfg.get("numerosity_reduction", True),
            n_neighbors=cfg.get("n_neighbors", 1),
            fit_sample_size=cfg.get("fit_sample_size", 2000),
            chunk_size=cfg.get("chunk_size", 2000),
            random_state=cfg.get("random_state", 0), n_jobs=n_jobs)
    if name == "saxvsm":
        from models.bakeoff_dictionary import SAXVSMClassifier
        return SAXVSMClassifier(
            window_size=cfg.get("window_size", 0.5), word_size=cfg.get("word_size", 0.5),
            n_bins=cfg.get("n_bins", 4), numerosity_reduction=cfg.get("numerosity_reduction", True),
            fit_sample_size=cfg.get("fit_sample_size", 2000),
            chunk_size=cfg.get("chunk_size", 2000),
            random_state=cfg.get("random_state", 0))
    if name == "tsbf":
        from models.bakeoff_tsbf import TSBFClassifier
        return TSBFClassifier(
            n_bins=cfg.get("n_bins", 10), n_estimators=cfg.get("n_estimators", 200),
            min_interval_length=cfg.get("min_interval_length", 5),
            n_subseries=cfg.get("n_subseries", 20),
            max_fit_pool=cfg.get("max_fit_pool", 200000),
            rf_max_leaf_nodes=cfg.get("rf_max_leaf_nodes", 500),
            random_state=cfg.get("random_state", 0), n_jobs=n_jobs)
    if name == "lps":
        from models.bakeoff_lps import LPSClassifier
        return LPSClassifier(
            n_trees=cfg.get("n_trees", 200), n_segments=cfg.get("n_segments", 75),
            min_segment_length=cfg.get("min_segment_length", 5),
            max_leaf_nodes=cfg.get("max_leaf_nodes", 100),
            max_pool_size=cfg.get("max_pool_size", 20000),
            chunk_size=cfg.get("chunk_size", 20000),
            clf_max_leaf_nodes=cfg.get("clf_max_leaf_nodes", 500),
            max_fit_pool=cfg.get("max_fit_pool", 200000),
            random_state=cfg.get("random_state", 0), n_jobs=n_jobs)
    if name == "fastshapelet":
        from models.bakeoff_fastshapelet import FastShapeletClassifier
        return FastShapeletClassifier(
            n_shapelets=cfg.get("n_shapelets", 10), max_candidates=cfg.get("max_candidates", 200),
            sax_word_length=cfg.get("sax_word_length", 8), sax_alphabet_size=cfg.get("sax_alphabet_size", 4),
            n_masks=cfg.get("n_masks", 10),
            selection_sample_size=cfg.get("selection_sample_size", 500),
            clf_max_leaf_nodes=cfg.get("clf_max_leaf_nodes", 500),
            max_fit_pool=cfg.get("max_fit_pool", 200000),
            random_state=cfg.get("random_state", 0))
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
        if X.ndim == 3:
            if X.shape[1] > 1:
                # If multivariate (e.g. 4 EWS channels), flatten (N, C, L) -> (N, C*L)
                # so univariate bakeoff models consume all 4 feature channels
                N, C, L = X.shape
                X = X.reshape(N, C * L)
            else:
                X = X[:, 0, :]
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

            # 3. Transform in chunks with pre-allocated memory buffer (eliminates 50% RAM spike)
            chunk_size = 5000
            X_trans = None

            for i in range(0, len(X), chunk_size):
                chunk = X[i : i + chunk_size]
                chunk_trans = self._transformer.transform(chunk).astype(np.float32)

                if X_trans is None:
                    n_samples = len(X)
                    n_feats = chunk_trans.shape[1]
                    X_trans = np.empty((n_samples, n_feats), dtype=np.float32)

                X_trans[i : i + len(chunk)] = chunk_trans
                del chunk_trans
                gc.collect()

                # Check RSS
                current_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
                if current_rss > 55.0:
                    print(f"[{self.name}] RSS {current_rss:.2f} GB exceeded 55 GB budget! Aborting gracefully.")
                    raise MemoryError("RSS exceeded 55 GB budget.")

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
            X_trans = None
            for i in range(0, len(X), chunk_size):
                chunk = X[i : i + chunk_size]
                chunk_trans = self._transformer.transform(chunk).astype(np.float32)
                if X_trans is None:
                    X_trans = np.empty((len(X), chunk_trans.shape[1]), dtype=np.float32)
                X_trans[i : i + len(chunk)] = chunk_trans
                del chunk_trans
                gc.collect()
            X_trans = self._scaler.transform(X_trans)

            # Continuous probabilities via decision_function + softmax
            df = self._estimator.decision_function(X_trans)
            if df.ndim == 1:
                df = np.column_stack([-df, df])
            exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
            probs = exp_df / np.sum(exp_df, axis=1, keepdims=True)
            return probs
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
