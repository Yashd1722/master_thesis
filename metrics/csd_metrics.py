# metrics/csd_metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


@dataclass
class CSDResult:
    scores: Dict[str, float]
    rolling_df: pd.DataFrame
    null_summary_df: Optional[pd.DataFrame] = None


def _ensure_1d(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.ndim != 1:
        raise ValueError(f"Signal must be 1D, got shape={x.shape}")
    if x.size < 3:
        raise ValueError("Signal is too short for CSD computation.")
    return x


def _safe_variance(x: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    return float(np.var(x, ddof=1))


def _safe_lag1_autocorr(x: np.ndarray) -> float:
    if x.size < 3:
        return np.nan

    x0 = x[:-1]
    x1 = x[1:]

    std0 = np.std(x0)
    std1 = np.std(x1)

    if std0 < 1e-12 or std1 < 1e-12:
        return np.nan

    corr = np.corrcoef(x0, x1)[0, 1]
    return float(corr)


def _kendall_tau(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)

    valid_mask = np.isfinite(y)
    y = y[valid_mask]

    if y.size < 2:
        return np.nan

    x = np.arange(len(y), dtype=np.float64)
    tau, _ = kendalltau(x, y)

    if tau is None:
        return np.nan
    return float(tau)


def _compute_window_size(n: int, window_frac: float, min_window: int) -> int:
    if not (0 < window_frac <= 1):
        raise ValueError(f"window_frac must be in (0, 1], got {window_frac}")

    window = int(np.floor(n * window_frac))
    window = max(window, int(min_window))
    window = min(window, n)

    if window < 3:
        raise ValueError("Computed rolling window is too small.")
    return window


def _rolling_statistics(signal: np.ndarray, window_size: int) -> pd.DataFrame:
    x = _ensure_1d(signal)
    n = len(x)

    rows = []
    for end in range(window_size, n + 1):
        start = end - window_size
        window = x[start:end]

        rows.append(
            {
                "start_idx": int(start),
                "end_idx": int(end - 1),
                "center_idx": int((start + end - 1) // 2),
                "variance": _safe_variance(window),
                "ac1": _safe_lag1_autocorr(window),
            }
        )

    return pd.DataFrame(rows)


def _fit_ar1_residual(signal: np.ndarray, fit_fraction: float = 0.2) -> Tuple[float, float, float]:
    """
    Fit a simple AR(1) model on the early fraction of the series:

        x_t = c + phi * x_{t-1} + eps_t

    Returns:
        c, phi, sigma_eps
    """
    x = _ensure_1d(signal)
    n = len(x)

    fit_n = max(5, int(np.floor(n * fit_fraction)))
    fit_n = min(fit_n, n)

    fit_x = x[:fit_n]
    if len(fit_x) < 3:
        return 0.0, 0.0, 1.0

    y = fit_x[1:]
    x_prev = fit_x[:-1]

    A = np.column_stack([np.ones_like(x_prev), x_prev])

    try:
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        c = float(beta[0])
        phi = float(beta[1])
        residuals = y - (c + phi * x_prev)
        sigma_eps = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 1.0
        sigma_eps = max(sigma_eps, 1e-8)
    except Exception:
        c, phi, sigma_eps = 0.0, 0.0, 1.0

    return c, phi, sigma_eps


def _generate_ar1_null(signal: np.ndarray, fit_fraction: float = 0.2, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Generate one AR(1) surrogate as a simple null model.
    """
    x = _ensure_1d(signal)
    n = len(x)

    if rng is None:
        rng = np.random.default_rng()

    c, phi, sigma_eps = _fit_ar1_residual(x, fit_fraction=fit_fraction)

    phi = np.clip(phi, -0.99, 0.99)

    out = np.zeros(n, dtype=np.float64)
    out[0] = float(x[0])

    for t in range(1, n):
        eps = rng.normal(0.0, sigma_eps)
        out[t] = c + phi * out[t - 1] + eps

    return out


def compute_csd_scores(
    signal: np.ndarray,
    window_frac: float = 0.5,
    min_window: int = 20,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Main CSD computation for one 1D signal.

    Returns:
        scores dict:
            - sequence_length
            - window_size
            - n_windows
            - mean_var
            - mean_ac1
            - ktau_var
            - ktau_ac1
        rolling_df:
            DataFrame with rolling variance and AC(1)
    """
    x = _ensure_1d(signal)
    n = len(x)
    window_size = _compute_window_size(n=n, window_frac=window_frac, min_window=min_window)

    rolling_df = _rolling_statistics(x, window_size=window_size)

    var_series = rolling_df["variance"].to_numpy(dtype=np.float64)
    ac1_series = rolling_df["ac1"].to_numpy(dtype=np.float64)

    scores = {
        "sequence_length": int(n),
        "window_size": int(window_size),
        "n_windows": int(len(rolling_df)),
        "mean_var": float(np.nanmean(var_series)) if len(var_series) > 0 else np.nan,
        "mean_ac1": float(np.nanmean(ac1_series)) if len(ac1_series) > 0 else np.nan,
        "ktau_var": _kendall_tau(var_series),
        "ktau_ac1": _kendall_tau(ac1_series),
    }

    return scores, rolling_df


def compute_csd_with_null(
    signal: np.ndarray,
    window_frac: float = 0.5,
    min_window: int = 20,
    n_null: int = 100,
    null_fit_fraction: float = 0.2,
    random_seed: Optional[int] = 42,
) -> CSDResult:
    """
    CSD + null-model computation for one signal.

    Returns:
        CSDResult containing:
            - observed scores
            - rolling statistics DataFrame
            - null summary DataFrame
    """
    x = _ensure_1d(signal)
    scores, rolling_df = compute_csd_scores(
        signal=x,
        window_frac=window_frac,
        min_window=min_window,
    )

    rng = np.random.default_rng(random_seed)

    null_rows = []
    for i in range(n_null):
        surrogate = _generate_ar1_null(
            signal=x,
            fit_fraction=null_fit_fraction,
            rng=rng,
        )
        null_scores, _ = compute_csd_scores(
            signal=surrogate,
            window_frac=window_frac,
            min_window=min_window,
        )

        null_rows.append(
            {
                "null_index": int(i),
                "mean_var": null_scores["mean_var"],
                "mean_ac1": null_scores["mean_ac1"],
                "ktau_var": null_scores["ktau_var"],
                "ktau_ac1": null_scores["ktau_ac1"],
            }
        )

    null_summary_df = pd.DataFrame(null_rows)

    if not null_summary_df.empty:
        scores["null_mean_ktau_var"] = float(np.nanmean(null_summary_df["ktau_var"]))
        scores["null_mean_ktau_ac1"] = float(np.nanmean(null_summary_df["ktau_ac1"]))

        obs_ktau_var = scores["ktau_var"]
        obs_ktau_ac1 = scores["ktau_ac1"]

        if np.isfinite(obs_ktau_var):
            scores["pvalue_ktau_var"] = float(
                np.mean(null_summary_df["ktau_var"].to_numpy(dtype=np.float64) >= obs_ktau_var)
            )
        else:
            scores["pvalue_ktau_var"] = np.nan

        if np.isfinite(obs_ktau_ac1):
            scores["pvalue_ktau_ac1"] = float(
                np.mean(null_summary_df["ktau_ac1"].to_numpy(dtype=np.float64) >= obs_ktau_ac1)
            )
        else:
            scores["pvalue_ktau_ac1"] = np.nan
    else:
        scores["null_mean_ktau_var"] = np.nan
        scores["null_mean_ktau_ac1"] = np.nan
        scores["pvalue_ktau_var"] = np.nan
        scores["pvalue_ktau_ac1"] = np.nan

    return CSDResult(
        scores=scores,
        rolling_df=rolling_df,
        null_summary_df=null_summary_df,
    )
