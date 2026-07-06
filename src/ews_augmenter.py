"""
src/ews_augmenter.py — (N, L) residuals -> (N, 4, L) EWS feature channels:
    0 raw residual | 1 rolling variance | 2 rolling lag-1 AC | 3 rolling skewness

Variance and lag-1 AC are the canonical Critical Slowing Down indicators
(Scheffer 2009, Dakos 2012); skewness adds fold-vs-null signal (Bury 2021).
Each channel is z-normalised with TRAIN-set stats that must be saved and reused
for val/test/empirical (never refit on held-out data).

    X_aug, stats = augment_ews_channels(X_train_2d)   # training
    X_aug, _     = augment_ews_channels(X_val_2d, channel_stats=stats)  # eval
"""

import numpy as np
import pandas as pd

# Process this many series per pandas chunk to keep memory below ~200 MB.
_CHUNK_SIZE = 10_000


def _rolling_channels_chunk(chunk: np.ndarray, window: int) -> np.ndarray:
    """(B, L) -> (B, 4, L) [raw, var, lag1_ac, skew] for one chunk."""
    # (L, B) DataFrame — rolling operates down rows (time), one column per series.
    df = pd.DataFrame(chunk.T, dtype=np.float64)  # (L, B)
    roll = df.rolling(window, min_periods=2)

    var_ch  = roll.var().fillna(0.0)                      # (L, B)
    skew_ch = roll.skew().fillna(0.0)                     # (L, B)

    # Lag-1 Pearson autocorrelation between x[t] and x[t-1] within each window.
    # df.shift(1) shifts rows down so row t of df_lag contains x[t-1].
    lag1_ch = roll.corr(df.shift(1)).fillna(0.0)          # (L, B)

    # Stack to (B, 4, L)
    out = np.stack([
        chunk,              # raw residual — (B, L)
        var_ch.T.values,    # (B, L)
        lag1_ch.T.values,   # (B, L)
        skew_ch.T.values,   # (B, L)
    ], axis=1)              # -> (B, 4, L)

    return out.astype(np.float32)


def _compute_rolling_channels(X: np.ndarray, window: int) -> np.ndarray:
    """(N, L) -> (N, 4, L), processed in _CHUNK_SIZE batches for memory."""
    N, L = X.shape
    out  = np.empty((N, 4, L), dtype=np.float32)

    for start in range(0, N, _CHUNK_SIZE):
        end              = min(start + _CHUNK_SIZE, N)
        out[start:end]   = _rolling_channels_chunk(X[start:end].astype(np.float64), window)

    return out


def fit_channel_stats(X_aug: np.ndarray) -> dict:
    """Per-channel mean/std over samples+time from the TRAIN array -> {mean, std}."""
    mean = X_aug.mean(axis=(0, 2))
    std  = X_aug.std(axis=(0, 2))
    std  = np.where(std < 1e-8, 1.0, std)   # guard degenerate channels
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def apply_channel_norm(X_aug: np.ndarray, stats: dict) -> np.ndarray:
    """Z-normalise (N, 4, L) with per-channel stats broadcast over (1, 4, 1)."""
    mean = stats["mean"][np.newaxis, :, np.newaxis]
    std  = stats["std"][np.newaxis, :, np.newaxis]
    return ((X_aug - mean) / std).astype(np.float32)


def augment_ews_channels(X, window_frac=0.25, channel_stats=None):
    """(N, L) residuals -> z-normalised (N, 4, L) EWS channels.

    The rolling window uses only past values (causal — no future leakage).
    channel_stats: None on training (stats are fit and returned); pass the saved
    stats on val/test/empirical so held-out data is never used to fit them.
    """
    if X.ndim == 3 and X.shape[1] == 1:
        X = X[:, 0, :]
    window = max(2, int(X.shape[1] * window_frac))
    X_aug = _compute_rolling_channels(X, window)
    if channel_stats is None:
        channel_stats = fit_channel_stats(X_aug)
    return apply_channel_norm(X_aug, channel_stats), channel_stats
