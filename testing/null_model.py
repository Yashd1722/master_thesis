# testing/null_models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AR1Fit:
    phi: float
    sigma: float
    mean: float
    n_fit: int


def _as_1d_float(signal) -> np.ndarray:
    arr = np.asarray(signal, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        raise ValueError("Signal must contain at least 3 finite values for AR(1) fitting.")
    return arr


def fit_ar1_from_prefix(signal, fit_fraction: float = 0.2) -> AR1Fit:
    """
    Fit a simple AR(1) model using only the first portion of the signal,
    following the same high-level idea as Bury's empirical null construction.

    x_t = mean + phi * (x_{t-1} - mean) + eps_t
    eps_t ~ N(0, sigma^2)
    """
    x = _as_1d_float(signal)

    if not (0.0 < fit_fraction <= 1.0):
        raise ValueError("fit_fraction must be in (0, 1].")

    n_fit = max(3, int(round(len(x) * fit_fraction)))
    x_fit = x[:n_fit]

    x_prev = x_fit[:-1]
    x_next = x_fit[1:]

    mean = float(np.mean(x_fit))
    prev_centered = x_prev - mean
    next_centered = x_next - mean

    denom = float(np.dot(prev_centered, prev_centered))
    if denom <= 1e-12:
        phi = 0.0
    else:
        phi = float(np.dot(prev_centered, next_centered) / denom)

    # Keep the process stable.
    phi = float(np.clip(phi, -0.99, 0.99))

    residuals = next_centered - phi * prev_centered
    sigma = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
    sigma = max(sigma, 1e-8)

    return AR1Fit(phi=phi, sigma=sigma, mean=mean, n_fit=n_fit)


def simulate_ar1(
    length: int,
    fit: AR1Fit,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate one AR(1) surrogate series from fitted parameters."""
    if length < 3:
        raise ValueError("length must be at least 3.")

    rng = np.random.default_rng(seed)
    out = np.empty(length, dtype=float)
    out[0] = fit.mean

    noise = rng.normal(loc=0.0, scale=fit.sigma, size=length - 1)
    for t in range(1, length):
        out[t] = fit.mean + fit.phi * (out[t - 1] - fit.mean) + noise[t - 1]

    return out


def generate_null_surrogates(
    signal,
    n_surrogates: int = 100,
    fit_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, AR1Fit]:
    """
    Generate multiple AR(1) null surrogates from the first fit_fraction of the signal.

    Returns
    -------
    surrogates : np.ndarray
        Shape (n_surrogates, len(signal))
    fit : AR1Fit
        Fitted AR(1) parameters.
    """
    x = _as_1d_float(signal)
    if n_surrogates < 1:
        raise ValueError("n_surrogates must be >= 1.")

    fit = fit_ar1_from_prefix(x, fit_fraction=fit_fraction)
    surrogates = np.vstack(
        [simulate_ar1(len(x), fit, seed=seed + i) for i in range(n_surrogates)]
    )
    return surrogates, fit
