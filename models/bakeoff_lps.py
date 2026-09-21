"""
models/bakeoff_lps.py — Learned Pattern Similarity (Baydogan & Runger, 2016),
built from scratch. No maintained library ships this.

Documented simplification, not a byte-for-byte reproduction:
  - The paper uses a multi-scale segment-length scheme; here a single
    min_segment_length is used for every regression tree, which is easier to
    validate and keeps one fewer under-specified hyperparameter surface.
  - "Learn local dynamics" here means: given a window, predict its own last
    point from the rest (next-step autoregression) — the simplest faithful
    reading of "regression trees over lagged segments."
Validated only against UCR GunPoint (see models/test_bakeoff_scratch.py) —
no reference implementation exists to check against directly.

Design:
  1. Pool (input, target) autoregressive pairs from random windows across a
     bounded sample of training series (label-blind — this stage learns
     generic local waveform dynamics, not class-discriminative splits).
  2. Fit n_trees regression trees on bootstrap samples of that pool. Each
     tree's leaves partition local dynamics into recurring pattern types.
  3. Per series (every real series, processed in chunks): push its own
     windows through every tree, get a leaf id per (window, tree); build a
     sparse leaf-occupancy histogram per series (LPS's "pattern similarity"
     representation).
  4. Train a Random Forest classifier on that per-series histogram.

Two OOM rounds happened building this (see logs/lps_ts_500_train.log,
2986380_30 and 2986451_30) — worth stating plainly since both are real
lessons, not just resolved footnotes:
  - Round 1: the per-series histogram was a DENSE (n_series, total_leaves)
    array. At real project scale that's ~300GB of >99% zeros.
  - Round 2: switching to sparse fixed the *output*, but fit() still built
    windows for every series at once and accumulated all n_trees' worth of
    row/col index arrays before concatenating — same order of memory, just
    moved into intermediates instead of the final array.
Fixed by chunking over series everywhere: windows, per-tree leaf lookups,
and the sparse block construction are all bounded by chunk_size, never by
the full dataset size.
"""
import numpy as np
from scipy.sparse import coo_matrix, vstack as sparse_vstack
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier


def _sample_windows(X, n_segments, seg_len, rng):
    # X: (n, L) -> windows: (n, n_segments, seg_len)
    n, ts_len = X.shape
    seg_len = min(seg_len, max(2, ts_len - 1))
    max_start = ts_len - seg_len
    starts = rng.integers(0, max_start + 1, size=(n, n_segments))
    return np.stack([X[np.arange(n)[:, None], starts + o] for o in range(seg_len)], axis=-1)


class LPSClassifier:
    def __init__(self, n_trees=200, n_segments=75, min_segment_length=5,
                 max_leaf_nodes=100, max_pool_size=20000, chunk_size=20000,
                 clf_max_leaf_nodes=500, max_fit_pool=200000, random_state=0, n_jobs=1):
        self.n_trees = n_trees
        self.n_segments = n_segments
        self.min_segment_length = min_segment_length
        self.max_leaf_nodes = max_leaf_nodes
        self.max_pool_size = max_pool_size
        # Cap on the FINAL RandomForestClassifier (fit on every series'
        # sparse histogram) — the n_trees pattern-learning regressors above
        # already have their own max_leaf_nodes, but this last classifier
        # didn't, and at real project scale that meant an unbounded-depth
        # forest over ~1.9M rows (same failure mode as tsbf's stage-2).
        self.clf_max_leaf_nodes = clf_max_leaf_nodes
        # Row cap on the final classifier's fit set. Unlike tsbf's stage-2,
        # this classifier's feature width is total_leaves (n_trees *
        # max_leaf_nodes, e.g. 20000) — a WIDE sparse matrix, not a small
        # fixed-size vector. Its size scales with n_series * total_leaves,
        # so at real scale (~1.9M series) it OOM'd (~150GB) and separately
        # overflowed the int32 sparse index forced below (~7.9B nonzeros,
        # past int32's ~2.1B limit -> "negative dimensions" in sklearn).
        # clf_max_leaf_nodes alone doesn't fix either failure since it only
        # bounds tree depth, not row count. Capping rows here does both.
        self.max_fit_pool = max_fit_pool
        # Series processed per batch when building the leaf-histogram
        # representation — bounds windows/tree.apply()/sparse-block memory to
        # this many series at a time regardless of dataset size.
        self.chunk_size = chunk_size
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _fit_trees(self, X, rng):
        # Only enough series to fill max_pool_size rows ever get windowed —
        # the full dataset's windows are never materialised for tree fitting.
        n_series_needed = max(1, -(-self.max_pool_size // self.n_segments))  # ceil div
        idx = rng.choice(len(X), size=min(len(X), n_series_needed), replace=False)
        windows = _sample_windows(X[idx], self.n_segments, self.min_segment_length, rng)
        n, k, seg_len = windows.shape
        pooled_in = windows[:, :, :-1].reshape(n * k, seg_len - 1)
        pooled_tgt = windows[:, :, -1].reshape(n * k)
        pool_n = pooled_in.shape[0]
        base_pool = min(pool_n, self.max_pool_size)

        self._trees = []
        for t in range(self.n_trees):
            tree_rng = np.random.default_rng(self.random_state + t)
            boot_idx = tree_rng.integers(0, pool_n, size=base_pool)
            tree = DecisionTreeRegressor(
                max_leaf_nodes=self.max_leaf_nodes, random_state=self.random_state + t)
            tree.fit(pooled_in[boot_idx], pooled_tgt[boot_idx])
            self._trees.append(tree)

        self._leaf_offsets = np.concatenate(
            [[0], np.cumsum([t.get_n_leaves() for t in self._trees])]).astype(np.int32)
        # node id -> compact leaf column lookup, from tree structure directly
        # (no need to run .apply() over data just to discover which leaves exist)
        self._leaf_lookup = []
        for t in self._trees:
            lookup = np.full(t.tree_.node_count, -1, dtype=np.int32)
            leaf_ids = np.where(t.tree_.children_left == -1)[0]
            lookup[leaf_ids] = np.arange(len(leaf_ids), dtype=np.int32)
            self._leaf_lookup.append(lookup)

    def _histogram_block(self, X_chunk, rng):
        windows = _sample_windows(X_chunk, self.n_segments, self.min_segment_length, rng)
        n, k, seg_len = windows.shape
        inputs = windows[:, :, :-1].reshape(n * k, seg_len - 1)
        total_leaves = int(self._leaf_offsets[-1])
        # int32, not numpy's int64 default: sklearn's tree-based estimators
        # reject int64-indexed sparse input ("No support for np.int64 index
        # based sparse matrices") — n and total_leaves both fit comfortably
        # in int32's range regardless of dataset size, this is purely about
        # numpy's default dtype, not an actual size constraint.
        row_idx = np.repeat(np.arange(n, dtype=np.int32), k)

        block = None
        for t, tree in enumerate(self._trees):
            leaves = tree.apply(inputs)
            cols_t = self._leaf_lookup[t][leaves]
            valid = cols_t >= 0
            data = np.ones(valid.sum(), dtype=np.float64)
            part = coo_matrix(
                (data, (row_idx[valid], cols_t[valid] + self._leaf_offsets[t])),
                shape=(n, total_leaves)).tocsr()
            block = part if block is None else block + part
        return block / max(k, 1)

    def _series_histogram(self, X, rng):
        blocks = [self._histogram_block(X[s:e], rng)
                  for s in range(0, len(X), self.chunk_size)
                  for e in [min(s + self.chunk_size, len(X))]]
        result = sparse_vstack(blocks).tocsr()
        # vstack/tocsr can silently upcast indices back to int64 — force back
        # to int32 right before this reaches RandomForestClassifier.
        result.indices = result.indices.astype(np.int32)
        result.indptr = result.indptr.astype(np.int32)
        return result

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        self._fit_trees(X, rng)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        cls_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([cls_idx[v] for v in y])

        fit_idx = rng.choice(len(X), size=min(len(X), self.max_fit_pool), replace=False)
        feats = self._series_histogram(X[fit_idx], rng)
        self._clf = RandomForestClassifier(
            n_estimators=200, max_leaf_nodes=self.clf_max_leaf_nodes,
            random_state=self.random_state, n_jobs=self.n_jobs)
        self._clf.fit(feats, y_idx[fit_idx])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state + 999)
        feats = self._series_histogram(X, rng)
        return self._clf.predict_proba(feats)
