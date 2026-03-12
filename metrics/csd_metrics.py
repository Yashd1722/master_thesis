# metrics/csd_metrics.py

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def rolling_variance(signal: np.ndarray, window: int) -> np.ndarray:
    if window < 2:
        raise ValueError("window must be >= 2 for rolling variance.")
    if len(signal) < window:
        raise ValueError("Signal length must be >= rolling window.")

    out = []
    for i in range(len(signal) - window + 1):
        out.append(float(np.var(signal[i:i + window], ddof=1)))
    return np.asarray(out, dtype=float)


def _lag1_autocorr(x: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    x0 = x[:-1]
    x1 = x[1:]
    if np.std(x0) == 0 or np.std(x1) == 0:
        return np.nan
    return float(np.corrcoef(x0, x1)[0, 1])


def rolling_ac1(signal: np.ndarray, window: int) -> np.ndarray:
    if window < 2:
        raise ValueError("window must be >= 2 for rolling AC1.")
    if len(signal) < window:
        raise ValueError("Signal length must be >= rolling window.")

    out = []
    for i in range(len(signal) - window + 1):
        out.append(_lag1_autocorr(signal[i:i + window]))
    return np.asarray(out, dtype=float)


def kendall_tau_trend(values: np.ndarray) -> float:
    if len(values) < 2:
        return np.nan
    idx = np.arange(len(values))
    tau, _ = kendalltau(idx, values, nan_policy="omit")
    return float(tau) if tau is not None else np.nan


def compute_csd_scores(signal, window_frac: float = 0.5, min_window: int = 20):
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if len(signal) == 0:
        raise ValueError("Signal is empty.")
    if np.isnan(signal).all():
        raise ValueError("Signal contains only NaN values.")

    signal = np.nan_to_num(signal, nan=float(np.nanmedian(signal)))
    window = max(min_window, int(round(len(signal) * window_frac)))
    window = min(window, len(signal))

    if window < 2:
        raise ValueError("Computed rolling window is too small.")

    var_series = rolling_variance(signal, window)
    ac1_series = rolling_ac1(signal, window)

    scores = {
        "signal_length": int(len(signal)),
        "window_size": int(window),
        "ktau_var": kendall_tau_trend(var_series),
        "ktau_ac1": kendall_tau_trend(ac1_series),
        "mean_var": float(np.nanmean(var_series)),
        "mean_ac1": float(np.nanmean(ac1_series)),
    }

    rolling_df = pd.DataFrame(
        {
            "window_index": np.arange(len(var_series)),
            "rolling_variance": var_series,
            "rolling_ac1": ac1_series,
        }
    )

    return scores, rolling_df
