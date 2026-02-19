import sys
from pathlib import Path
import logging
import importlib
import inspect
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset

# Make project root importable (for models and src)
sys.path.insert(0, str(Path(__file__).parent.parent))


def project_root() -> Path:
    """Return absolute path to the project root."""
    return Path(__file__).resolve().parents[1]


def accuracy_np(y_true, y_pred):
    """NumPy accuracy (used in progress bar)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def list_available_models():
    """Scan models/ directory and return lowercased .py filenames (no __init__)."""
    models_dir = project_root() / "models"
    models_dir.mkdir(exist_ok=True)
    py_files = [p for p in models_dir.glob("*.py") if p.name != "__init__.py"]
    return sorted([p.stem.lower() for p in py_files])


def setup_logger(log_path: Path):
    """Create a logger that writes to both file and console."""
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear old handlers
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------- Bury‑style preprocessing ----------
def bury_random_zero_padding(X, pad_left_max, pad_right_max, rng):
    """
    Randomly zero out first `pad_left` and last `pad_right` timesteps.
    X: np.ndarray (T, F)
    """
    T = X.shape[0]
    pad_left = int(pad_left_max * rng.random()) if pad_left_max > 0 else 0
    pad_right = int(pad_right_max * rng.random()) if pad_right_max > 0 else 0

    if pad_left > 0:
        X[:pad_left, :] = 0.0
    if pad_right > 0:
        X[T - pad_right:T, :] = 0.0
    return X


def bury_mean_abs_normalize_nonzero(X, eps=1e-12):
    """Normalise by mean(abs(non‑zero entries)) over all features jointly."""
    mask = (X != 0.0)
    if not mask.any():
        return X
    mean_abs = float(np.mean(np.abs(X[mask])))
    if mean_abs < eps:
        return X
    X[mask] = X[mask] / mean_abs
    return X


def get_bury_pad_max(seq_len, pad_mode):
    """
    Return (pad_left_max, pad_right_max) according to Bury’s settings.
    pad_mode: 'both' or 'left'
    """
    if pad_mode == "both":
        return (225, 225) if seq_len == 500 else (725, 725)
    if pad_mode == "left":
        return (450, 0) if seq_len == 500 else (1450, 0)
    raise ValueError("pad_mode must be 'both' or 'left'")


# ---------- Streaming dataset ----------
class TSCSVDataset(IterableDataset):
    """Streaming dataset for time‑series CSV files (split by sequence_ID)."""

    def __init__(
        self,
        csv_path,
        feature_cols=("x", "Residuals"),
        seq_len=500,
        chunksize=300_000,
        apply_padding=False,
        pad_mode="both",
        apply_norm=False,
        seed=42,
    ):
        self.csv_path = str(csv_path)
        self.feature_cols = list(feature_cols)
        self.seq_len = seq_len
        self.chunksize = chunksize
        self.apply_padding = apply_padding
        self.pad_mode = pad_mode
        self.apply_norm = apply_norm
        self.seed = seed

    def __iter__(self):
        cols = ["sequence_ID", "Time", *self.feature_cols, "class_label"]
        buffer_df = None
        rng = np.random.default_rng(self.seed)
        pad_left_max, pad_right_max = get_bury_pad_max(self.seq_len, self.pad_mode)

        for chunk in pd.read_csv(self.csv_path, usecols=cols, chunksize=self.chunksize):
            chunk = chunk.sort_values(["sequence_ID", "Time"])

            if buffer_df is not None:
                chunk = pd.concat([buffer_df, chunk], ignore_index=True)
                buffer_df = None

            groups = list(chunk.groupby("sequence_ID", sort=False))
            if not groups:
                continue

            # Keep last (possibly incomplete) group as buffer
            last_sid, last_g = groups[-1]
            if len(last_g) < self.seq_len:
                buffer_df = last_g.copy()
                groups = groups[:-1]

            for sid, g in groups:
                g = g.iloc[: self.seq_len]
                if len(g) < self.seq_len:
                    continue

                X = g[self.feature_cols].to_numpy(dtype=np.float32)
                y = int(g["class_label"].iloc[0])

                if self.apply_padding:
                    X = bury_random_zero_padding(X, pad_left_max, pad_right_max, rng)
                if self.apply_norm:
                    X = bury_mean_abs_normalize_nonzero(X)

                yield torch.from_numpy(X), torch.tensor(y, dtype=torch.long)


# ---------- Dynamic model loader ----------
def _find_model_module_name(model_name: str) -> str:
    """Convert 'lstm' → 'models.lstm' after checking file existence."""
    model_name = model_name.lower()
    models_dir = project_root() / "models"
    py_files = [p for p in models_dir.glob("*.py") if p.name != "__init__.py"]
    for p in py_files:
        if p.stem.lower() == model_name:
            return f"models.{p.stem}"
    available = sorted([p.stem.lower() for p in py_files])
    raise ValueError(
        f"Unknown model '{model_name}'. Available: {available}"
    )


def _pick_model_class(module, model_class: str | None = None):
    """Pick a single nn.Module subclass from the module."""
    if model_class is not None:
        if not hasattr(module, model_class):
            raise ValueError(f"Class '{model_class}' not found in {module.__name__}")
        cls = getattr(module, model_class)
        if not (inspect.isclass(cls) and issubclass(cls, nn.Module)):
            raise ValueError(f"{module.__name__}.{model_class} is not a torch.nn.Module")
        return cls

    candidates = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        if issubclass(obj, nn.Module):
            candidates.append(obj)

    if len(candidates) == 1:
        return candidates[0]

    preferred = [c for c in candidates if c.__name__.endswith(("Classifier", "Model"))]
    if len(preferred) == 1:
        return preferred[0]

    if not candidates:
        raise ValueError(f"No nn.Module found in {module.__name__}")
    names = [c.__name__ for c in candidates]
    raise ValueError(f"Multiple classes: {names}. Use --model_class.")


def build_model(model_name, input_size, num_classes, model_kwargs_json=None, model_class=None):
    """Instantiate a model by name, passing input_size, num_classes and extra kwargs."""
    module_name = _find_model_module_name(model_name)
    module = importlib.import_module(module_name)
    ModelCls = _pick_model_class(module, model_class=model_class)

    extra_kwargs = {}
    if model_kwargs_json:
        extra_kwargs = json.loads(model_kwargs_json)
        if not isinstance(extra_kwargs, dict):
            raise ValueError("--model_kwargs must be a JSON object")

    return ModelCls(input_size=input_size, num_classes=num_classes, **extra_kwargs)
