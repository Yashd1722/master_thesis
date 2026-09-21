"""
models/bakeoff_dictionary.py — BOP and SAX-VSM, wrapping pyts's tested
implementations (Lin, Khade & Li 2012 / Senin & Malinchik 2013) rather than
reimplementing SAX + histogram machinery pyts already gets right.

Both take 2D (n_samples, n_timestamps) input, matching what TSCModel's
univariate=True _prep already produces before calling into these.
"""
import numpy as np
from scipy.sparse import vstack as sparse_vstack
from . import _pyts_compat  # noqa: F401  (patches scipy before pyts import)


def _chunks(n, size):
    for start in range(0, n, size):
        yield start, min(start + size, n)


class BOPClassifier:
    """Bag of Patterns (Lin, Khade & Li 2012): SAX bag-of-words histogram per series."""

    def __init__(self, window_size=0.5, word_size=0.5, n_bins=4,
                 numerosity_reduction=True, n_neighbors=1,
                 use_diff=False, fit_sample_size=2000, chunk_size=500,
                 random_state=0, n_jobs=1):
        from pyts.transformation import BagOfPatterns
        from sklearn.neighbors import KNeighborsClassifier
        self._bop = BagOfPatterns(
            window_size=window_size, word_size=word_size, n_bins=n_bins,
            numerosity_reduction=numerosity_reduction)
        self._knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs)
        self.use_diff = use_diff
        self.fit_sample_size = fit_sample_size
        self.chunk_size = chunk_size
        self.random_state = random_state

    def _prep_diff(self, X):
        X = np.asarray(X)
        if self.use_diff and X.ndim == 2 and X.shape[1] > 2:
            diff = np.diff(X, axis=-1)
            return np.pad(diff, ((0, 0), (1, 0)), mode="edge")
        return X

    def _transform_chunked(self, X):
        X = self._prep_diff(X)
        n = len(X)
        parts = [self._bop.transform(X[s:e]) for s, e in _chunks(n, self.chunk_size)]
        return sparse_vstack(parts).tocsr()

    def fit(self, X, y):
        X = self._prep_diff(X)
        rng = np.random.default_rng(self.random_state)
        fit_idx = rng.choice(len(X), size=min(len(X), self.fit_sample_size), replace=False)
        self._bop.fit(X[fit_idx], np.asarray(y)[fit_idx])

        Xt = self._transform_chunked(X)
        self._knn.fit(Xt, y)
        self.classes_ = self._knn.classes_
        self.n_classes_ = len(self.classes_)
        return self

    def predict_proba(self, X):
        Xt = self._transform_chunked(X)
        return self._knn.predict_proba(Xt)


class SAXVSMClassifier:
    """SAX-VSM (Senin & Malinchik 2013): TF-IDF word vectors per class."""

    def __init__(self, window_size=0.5, word_size=0.5, n_bins=4,
                 numerosity_reduction=True, use_diff=False,
                 fit_sample_size=2000, chunk_size=500, random_state=0):
        from pyts.classification import SAXVSM
        self._clf = SAXVSM(
            window_size=window_size, word_size=word_size, n_bins=n_bins,
            numerosity_reduction=numerosity_reduction)
        self.use_diff = use_diff
        self.fit_sample_size = fit_sample_size
        self.chunk_size = chunk_size
        self.random_state = random_state

    def _prep_diff(self, X):
        X = np.asarray(X)
        if self.use_diff and X.ndim == 2 and X.shape[1] > 2:
            diff = np.diff(X, axis=-1)
            return np.pad(diff, ((0, 0), (1, 0)), mode="edge")
        return X

    def fit(self, X, y):
        X = self._prep_diff(X)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        classes = np.unique(y)
        per_class = max(1, self.fit_sample_size // len(classes))
        idx = np.concatenate([
            rng.choice(np.where(y == c)[0], min(per_class, (y == c).sum()), replace=False)
            for c in classes])
        self._clf.fit(X[idx], y[idx])
        self.classes_ = self._clf.classes_
        self.n_classes_ = len(self.classes_)
        return self

    def predict_proba(self, X):
        X = self._prep_diff(X)
        sims = np.concatenate([
            self._clf.decision_function(X[s:e])
            for s, e in _chunks(len(X), self.chunk_size)])
        sims = np.clip(sims, 0, None)
        row_sums = sims.sum(axis=1, keepdims=True)
        flat = (row_sums[:, 0] == 0)
        row_sums[row_sums == 0] = 1.0
        probs = sims / row_sums
        # a document with zero similarity to every class (rare, short/atypical
        # series) falls back to uniform rather than manufacturing a fake mode
        if flat.any():
            probs[flat] = 1.0 / sims.shape[1]
        return probs
