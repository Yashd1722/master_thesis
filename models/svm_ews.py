"""
models/svm_ews.py — SVM classifier on EWS features (Ma et al. 2025).

Unlike PyTorch models, SVM does not learn from raw time series.
It classifies pre-transition vs neutral using 4 handcrafted features:
  [mean_variance, mean_lag1_ac, kendall_tau_variance, kendall_tau_lag1_ac]

Ma 2025 finding: SVM was best classifier for MS66 (AUC=0.97).

─────────────────────────────────────────────────────────────────────────
CONTRACT FOR ADDING ANY NEW SKLEARN MODEL:

1. Add these three lines at module level (after the docstring):
       MODEL_NAME  = "my_model"
       MODEL_CLASS = "MyModelClass"
       IS_SKLEARN  = True

2. Implement this interface:
       fit(X: ndarray, y: ndarray)                      train on feature matrix
       predict_proba_numpy(X: ndarray) -> ndarray       (N, n_classes) probs
       save(path: Path)                                  save to .pkl
       load(path: Path) -> instance              [classmethod]
       _build_sklearn() -> sklearn pipeline       used by train.py grid search
       extract_features(...) -> ndarray (4,)     [staticmethod] one row per step
       to(device) -> self                         no-op, keep for API compat
       eval()     -> self                         no-op, keep for API compat

3. Add training hyperparameters to config.yaml under training.my_model

That is all. train.py and evaluate.py handle everything else automatically.
─────────────────────────────────────────────────────────────────────────
"""

MODEL_NAME  = "svm"
MODEL_CLASS = "SVMClassifier"
IS_SKLEARN  = True

import pickle
import numpy as np
from pathlib import Path


class SVMClassifier:
    """
    SVM on rolling-window EWS features.
    Binary classifier: 0 = neutral, 1 = pre_transition.
    """

    def __init__(self, ts_len: int = 500, num_classes: int = 2,
                 kernel: str = "rbf", C: float = 1.0,
                 gamma: str = "scale"):
        self.ts_len      = ts_len
        self.num_classes = num_classes
        self.kernel      = kernel
        self.C           = C
        self.gamma       = gamma
        self._model      = None   # sklearn Pipeline, set after fit()

    # ── Build sklearn pipeline ────────────────────────────────────────────────

    def _build_sklearn(self):
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svc",    SVC(
                kernel       = self.kernel,
                C            = self.C,
                gamma        = self.gamma,
                probability  = True,
                class_weight = "balanced",
                random_state = 42,
            )),
        ])

    # ── Feature extraction ────────────────────────────────────────────────────

    @staticmethod
    def extract_features(variance: np.ndarray, lag1_ac: np.ndarray,
                         ktau_var: float, ktau_ac: float) -> np.ndarray:
        """
        4-dimensional EWS feature vector for one window position.
        Called by evaluate.py once per rolling-window step.

        Parameters
        ----------
        variance  : variance values in the window
        lag1_ac   : lag-1 AC values in the window
        ktau_var  : Kendall tau trend of variance over all steps
        ktau_ac   : Kendall tau trend of lag-1 AC over all steps

        Returns (4,) float32 array.
        """
        return np.array([
            float(np.nanmean(variance)),
            float(np.nanmean(lag1_ac)),
            float(ktau_var),
            float(ktau_ac),
        ], dtype=np.float32)

    @staticmethod
    def extract_features_from_df(df) -> np.ndarray:
        """Extract features from a prediction CSV DataFrame."""
        return SVMClassifier.extract_features(
            df["variance"].values,
            df["lag1_ac"].values,
            float(df["ktau_variance"].iloc[0]) if "ktau_variance" in df else 0.0,
            float(df["ktau_lag1_ac"].iloc[0])  if "ktau_lag1_ac"  in df else 0.0,
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMClassifier":
        self._model = self._build_sklearn()
        self._model.fit(X, y)
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba_numpy(self, X: np.ndarray) -> np.ndarray:
        """
        Returns (N, 2) probabilities: [[p_neutral, p_pretrans], ...]
        Called by evaluate.py run_inference().
        """
        if self._model is None:
            raise RuntimeError("SVM not trained. Call fit() or load().")
        return self._model.predict_proba(X)

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model":       self._model,
                "ts_len":      self.ts_len,
                "num_classes": self.num_classes,
                "kernel":      self.kernel,
                "C":           self.C,
                "gamma":       self.gamma,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "SVMClassifier":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj          = cls(ts_len=state["ts_len"],
                           num_classes=state["num_classes"],
                           kernel=state["kernel"],
                           C=state["C"],
                           gamma=state["gamma"])
        obj._model   = state["model"]
        return obj

    # ── API compatibility (no-ops for sklearn, required by pipeline) ──────────

    def to(self, device):   return self
    def eval(self):         return self
    def train(self, mode=True): return self
    def parameters(self):   return iter([])
    def load_state_dict(self, *a, **kw): pass


if __name__ == "__main__":
    svm = SVMClassifier()
    X   = np.random.randn(100, 4).astype(np.float32)
    y   = np.array([0] * 50 + [1] * 50)
    svm.fit(X, y)
    probs = svm.predict_proba_numpy(X[:5])
    assert probs.shape == (5, 2), f"Expected (5,2), got {probs.shape}"
    print(f"SVMClassifier: OK  probs shape={probs.shape}")
