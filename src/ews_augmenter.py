"""
src/ews_augmenter.py

Converts (N, L) residuals into (N, 4, L) Early Warning Signal feature channels.

Channel layout:
  0 — raw_residual    (unchanged input)
  1 — rolling_variance
  2 — rolling_lag1_ac  (lag-1 Pearson autocorrelation)
  3 — rolling_skewness

Science: variance and lag-1 AC are the canonical Critical Slowing Down (CSD)
indicators (Scheffer 2009, Dakos 2012). Both increase near a bifurcation as
the dominant eigenvalue of the linearised system approaches zero. Skewness
is asymmetric near fold bifurcations and provides extra signal for multi-class
discrimination (Bury et al. 2021).

Z-normalization: each channel is normalised using the TRAIN-set mean and std
computed via fit_channel_stats(). These stats MUST be saved and reused for
val/test/empirical data — never refit on held-out data (that would leak
distributional information across splits).

Usage (training):
    X_aug, stats = augment_ews_channels(X_train_2d)
    np.savez("ch_stats.npz", **stats)

Usage (val/test):
    stats = dict(np.load("ch_stats.npz"))
    X_aug, _ = augment_ews_channels(X_val_2d, channel_stats=stats)
"""

import numpy as np
import pandas as pd

# Process this many series per pandas chunk to keep memory below ~200 MB.
_CHUNK_SIZE = 10_000


def _rolling_channels_chunk(chunk: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling channels for one chunk of series.

    Args:
        chunk: (B, L) float64 array — a batch of B time series
        window: rolling window length in samples

    Returns:
        (B, 4, L) float32 array with channels [raw, var, lag1_ac, skew]
    """
    # Build (L, B) DataFrame — rolling operates along rows (time axis).
    # Each column is one time series.
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
    """
    Compute all 4 EWS channels for N series of length L.

    Processes in chunks of _CHUNK_SIZE to stay within memory budget.

    Args:
        X: (N, L) residual array
        window: rolling window length in samples

    Returns:
        (N, 4, L) float32 array
    """
    N, L = X.shape
    out  = np.empty((N, 4, L), dtype=np.float32)

    for start in range(0, N, _CHUNK_SIZE):
        end              = min(start + _CHUNK_SIZE, N)
        out[start:end]   = _rolling_channels_chunk(X[start:end].astype(np.float64), window)

    return out


def fit_channel_stats(X_aug: np.ndarray) -> dict:
    """
    Compute per-channel z-normalization stats from the augmented TRAINING array.

    Mean and std are computed over all samples (N) and time steps (L) jointly
    so that one scalar normalises each channel uniformly.

    Args:
        X_aug: (N, 4, L) float32 array — output of _compute_rolling_channels

    Returns:
        dict with 'mean' and 'std' arrays of shape (4,), dtype float32
    """
    # Average over axis 0 (samples) and axis 2 (time), leaving one value per channel.
    mean = X_aug.mean(axis=(0, 2))                          # (4,)
    std  = X_aug.std(axis=(0, 2))                           # (4,)
    std  = np.where(std < 1e-8, 1.0, std)                   # guard degenerate channels
    return {
        "mean": mean.astype(np.float32),
        "std":  std.astype(np.float32),
    }


def apply_channel_norm(X_aug: np.ndarray, stats: dict) -> np.ndarray:
    """
    Z-normalise (N, 4, L) array using pre-computed per-channel stats.

    Args:
        X_aug: (N, 4, L) float32 array
        stats:  dict with 'mean' and 'std' of shape (4,)

    Returns:
        (N, 4, L) normalised float32 array
    """
    mean = stats["mean"][np.newaxis, :, np.newaxis]         # broadcast: (1, 4, 1)
    std  = stats["std"][np.newaxis, :, np.newaxis]
    return ((X_aug - mean) / std).astype(np.float32)


def augment_ews_channels(
    X: np.ndarray,
    window_frac: float = 0.25,
    channel_stats: dict = None,
) -> tuple:
    """
    Convert (N, L) residuals to z-normalised (N, 4, L) EWS feature channels.

    The rolling window uses ONLY past values (no center=True) so that the
    features remain causal — no future data leaks into the feature channels.

    Args:
        X:             (N, L) float array — detrended residuals
        window_frac:   rolling window as a fraction of L (default 0.25)
        channel_stats: if None (training), computes and returns stats;
                       if provided (val/test), uses those stats directly.

    Returns:
        X_aug:         (N, 4, L) normalised float32 feature array
        channel_stats: dict with 'mean' and 'std' of shape (4,)
    """
    if X.ndim == 3 and X.shape[1] == 1:
        X = X[:, 0, :]              # squeeze channel dim if caller passes (N, 1, L)

    N, L   = X.shape
    window = max(2, int(L * window_frac))

    X_aug = _compute_rolling_channels(X, window)            # (N, 4, L)

    if channel_stats is None:
        # Training: compute normalisation stats from this set.
        channel_stats = fit_channel_stats(X_aug)

    X_norm = apply_channel_norm(X_aug, channel_stats)
    return X_norm, channel_stats
