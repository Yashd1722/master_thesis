import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

from models import list_available_models, build_model
from src.signal_injector import apply_forcing

def project_root() -> Path:
    """Return absolute path to the project root."""
    return Path(__file__).resolve().parents[1]


def accuracy_np(y_true, y_pred):
    """NumPy accuracy (used in progress bar)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


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
        forcing_config=None,
        seed=42,
    ):
        self.csv_path = Path(csv_path)
        self.feature_cols = list(feature_cols)
        self.seq_len = seq_len
        self.chunksize = chunksize
        self.apply_padding = apply_padding
        self.pad_mode = pad_mode
        self.apply_norm = apply_norm
        self.forcing_config = forcing_config or {}
        self.seed = seed

    def __iter__(self):
        cols = ["sequence_ID", "Time", *self.feature_cols, "class_label"]
        buffer_df = pd.DataFrame(columns=cols)
        rng = np.random.default_rng(self.seed)
        pad_left_max, pad_right_max = get_bury_pad_max(self.seq_len, self.pad_mode)

        for chunk in pd.read_csv(self.csv_path, usecols=cols, chunksize=self.chunksize):
            chunk = chunk.sort_values(["sequence_ID", "Time"])

            if not buffer_df.empty:
                chunk = pd.concat([buffer_df, chunk], ignore_index=True)

            # Group by sequence_ID, keeping the order in which they appear
            groups = chunk.groupby("sequence_ID", sort=False)
            group_keys = list(groups.groups.keys())

            # The last group might be incomplete in the current chunk
            last_sid = group_keys[-1]
            last_g = groups.get_group(last_sid)

            # If the last group is not complete, we save it for the next chunk
            if len(last_g) < self.seq_len:
                buffer_df = last_g
                group_keys = group_keys[:-1]
            else:
                buffer_df = pd.DataFrame(columns=cols)

            for sid in group_keys:
                g = groups.get_group(sid).iloc[:self.seq_len]
                if len(g) < self.seq_len:
                    continue

                X = g[self.feature_cols].to_numpy(dtype=np.float32)
                y = int(g["class_label"].iloc[0])

                if self.apply_padding:
                    X = bury_random_zero_padding(X, pad_left_max, pad_right_max, rng)

                if self.apply_norm:
                    X = bury_mean_abs_normalize_nonzero(X)

                X_tensor = torch.from_numpy(X)

                # Apply forcing if configured
                if self.forcing_config:
                    # apply_forcing expects (B, T, F)
                    X_tensor = apply_forcing(X_tensor.unsqueeze(0), **self.forcing_config).squeeze(0)

                yield X_tensor, torch.tensor(y, dtype=torch.long)


