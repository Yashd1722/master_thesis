"""
models/bakeoff_tsbf.py — Time Series Bag of Features (Baydogan, Runger & Tuv,
2013), built from scratch. No maintained library ships this.

This is a documented simplification, not a byte-for-byte reproduction:
  - The paper draws subsequences from a fixed multi-scale segmentation scheme;
    here intervals are randomly positioned/length (same generator style as
    aeon's TimeSeriesForestClassifier already used elsewhere in this repo),
    which is easier to validate and still captures the "many local windows,
    each described by simple stats" idea the algorithm rests on.
  - The paper's exact per-interval "relative position" feature is replaced
    with (mean, std, slope), the three stats TSF/TSBF both agree matter most.
Validated only against UCR GunPoint (see models/test_bakeoff_scratch.py) —
no reference implementation exists to check against directly.

Two-stage design (matches the paper's core idea):
  1. Random intervals -> (mean, std, slope) features -> a first Random Forest,
     whose out-of-bag class-probability estimate is computed per interval.
  2. Per series, bin the OOB probability estimates of its own intervals into
     a fixed-length histogram (the "bag of features" summary), concatenate
     whole-series (mean, std, slope), and train a second Random Forest on
     that per-series representation.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def _stats(seg):
    mean = seg.mean(axis=-1)
    std = seg.std(axis=-1)
    t = np.arange(seg.shape[-1])
    t_c = t - t.mean()
    denom = (t_c ** 2).sum() or 1.0
    slope = (seg * t_c).sum(axis=-1) / denom
    return mean, std, slope


class TSBFClassifier:
    def __init__(self, n_bins=10, n_estimators=200, min_interval_length=5,
                 n_subseries=20, max_fit_pool=200000, rf_max_leaf_nodes=500,
                 random_state=0, n_jobs=1):
        self.n_bins = n_bins
        self.n_estimators = n_estimators
        self.min_interval_length = min_interval_length
        self.n_subseries = n_subseries
        # n_subseries is fixed, so the interval-row count (n_series *
        # n_subseries) scales only with dataset size — at real project scale
        # that's tens of millions of rows for stage-1's Random Forest, which
        # OOM'd in practice. Cap the FIT pool; rows outside it get an
        # unbiased predict_proba instead of the (fit-only) OOB estimate —
        # they were never in stage-1's training set either way, so this
        # isn't a leakage shortcut, just a size cap.
        self.max_fit_pool = max_fit_pool
        # sklearn's RandomForestClassifier has no depth/leaf cap by default —
        # at ~1.9M rows, stage-2's forest grew to a 23GB checkpoint (near-1
        # sample per leaf, i.e. memorising the data). Capped for both the
        # checkpoint size and the overfitting that size implies.
        self.rf_max_leaf_nodes = rf_max_leaf_nodes
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _make_intervals(self, ts_len, rng):
        min_len = min(self.min_interval_length, max(2, ts_len // 2))
        starts = rng.integers(0, ts_len - min_len, size=self.n_subseries)
        lengths = rng.integers(min_len, max(min_len + 1, ts_len - starts.max()),
                                size=self.n_subseries)
        lengths = np.minimum(lengths, ts_len - starts)
        return list(zip(starts.tolist(), lengths.tolist()))

    def _interval_features(self, X, intervals):
        # X: (n, L) -> (n * n_subseries, 3), interval index per row
        feats = []
        for start, length in intervals:
            seg = X[:, start:start + length]
            mean, std, slope = _stats(seg)
            feats.append(np.stack([mean, std, slope], axis=1))
        return np.stack(feats, axis=1)  # (n, n_subseries, 3)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        n, ts_len = X.shape
        rng = np.random.default_rng(self.random_state)
        self.intervals_ = self._make_intervals(ts_len, rng)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        cls_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([cls_idx[v] for v in y])

        int_feats = self._interval_features(X, self.intervals_)  # (n, k, 3)
        k = int_feats.shape[1]
        flat_X = int_feats.reshape(n * k, 3)
        flat_y = np.repeat(y_idx, k)
        total = flat_X.shape[0]

        self._stage1 = RandomForestClassifier(
            n_estimators=self.n_estimators, oob_score=True, bootstrap=True,
            max_leaf_nodes=self.rf_max_leaf_nodes,
            random_state=self.random_state, n_jobs=self.n_jobs)
        fit_size = min(total, self.max_fit_pool)
        fit_idx = rng.choice(total, size=fit_size, replace=False)
        self._stage1.fit(flat_X[fit_idx], flat_y[fit_idx])

        probs = np.empty((total, self.n_classes_))
        probs[fit_idx] = np.nan_to_num(
            self._stage1.oob_decision_function_, nan=1.0 / self.n_classes_)
        rest = np.ones(total, dtype=bool)
        rest[fit_idx] = False
        if rest.any():
            probs[rest] = self._stage1.predict_proba(flat_X[rest])

        series_feats = self._series_representation(X, int_feats, probs, k, n)
        self._stage2 = RandomForestClassifier(
            n_estimators=self.n_estimators, max_leaf_nodes=self.rf_max_leaf_nodes,
            random_state=self.random_state, n_jobs=self.n_jobs)
        self._stage2.fit(series_feats, y_idx)
        return self

    def _series_representation(self, X, int_feats, oob_probs, k, n):
        oob_probs = oob_probs.reshape(n, k, self.n_classes_)
        # histogram of each class's predicted-probability mass across this
        # series' own intervals -> fixed-length (n_classes_ * n_bins) summary.
        # Equal-width bins, computed via one vectorised pass per class instead
        # of a Python loop over all n series (n reaches the millions at real
        # project scale — a per-series np.histogram call there is its own
        # meaningful cost, separate from the OOM this was fixed alongside).
        bin_idx = np.clip((oob_probs * self.n_bins).astype(np.int64), 0, self.n_bins - 1)
        hist = np.zeros((n, self.n_classes_, self.n_bins))
        series_idx = np.repeat(np.arange(n), k)
        for c in range(self.n_classes_):
            np.add.at(hist[:, c, :], (series_idx, bin_idx[:, :, c].reshape(-1)), 1)
        hist /= max(k, 1)
        hist = hist.reshape(n, -1)
        whole_mean, whole_std, whole_slope = _stats(X)
        return np.concatenate(
            [hist, whole_mean[:, None], whole_std[:, None], whole_slope[:, None]], axis=1)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        int_feats = self._interval_features(X, self.intervals_)
        k = int_feats.shape[1]
        flat_X = int_feats.reshape(n * k, 3)
        probs = self._stage1.predict_proba(flat_X)
        series_feats = self._series_representation(X, int_feats, probs, k, n)
        return self._stage2.predict_proba(series_feats)
