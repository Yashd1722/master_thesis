"""
metric/kendall_tau.py

Kendall tau rank correlation between time index and transition probability.

Science: Bury et al. (2021) use Kendall tau to measure whether the classifier's
estimated transition probability p_transition = 1 - P(null) rises monotonically
over the forced segment as the system approaches the bifurcation.

Convention: series MUST be sorted OLDEST-FIRST so that index 0 is the most
ancient sample and index T-1 is closest to the transition.
  - Positive tau → p_transition rises toward the transition (early warning present)
  - Negative tau → p_transition falls (inconsistent with CSD hypothesis)
  - tau ≈ 0     → no trend (model or data does not carry EWS signal)

Unit check (run at import):
    p_rising = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]  →  tau = +1.0
    p_const  = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  →  tau = 0.0 (std < threshold)
"""

import numpy as np
from scipy.stats import kendalltau


def compute_kendall_tau(p_transition: np.ndarray) -> float:
    """
    Kendall tau between ascending time index and p_transition.

    Args:
        p_transition: (T,) float array, sorted oldest-first.

    Returns:
        tau in [-1, 1].  Returns 0.0 for degenerate (constant or short) input.
    """
    p = np.asarray(p_transition, dtype=float)
    if len(p) < 3:
        return 0.0
    if np.std(p) < 1e-10:
        return 0.0
    tau, _ = kendalltau(np.arange(len(p)), p)
    return float(tau) if np.isfinite(tau) else 0.0


def compute_tau_ci(tau_list: list, confidence: float = 0.95) -> dict:
    """
    Mean and confidence interval for a list of Kendall tau values.

    Used to average tau across AR(1) null surrogates or across multiple
    forced series from different sapropels.

    Args:
        tau_list:   list of float tau values
        confidence: CI level (default 0.95)

    Returns:
        dict with 'mean', 'std', 'ci_low', 'ci_high'
    """
    arr  = np.asarray(tau_list, dtype=float)
    n    = len(arr)
    mean = float(arr.mean()) if n > 0 else float("nan")
    std  = float(arr.std())  if n > 0 else float("nan")

    if n < 2:
        return {"mean": mean, "std": 0.0, "ci_low": mean, "ci_high": mean, "n": n}

    from scipy.stats import t as t_dist
    alpha = 1.0 - confidence
    t_val = float(t_dist.ppf(1.0 - alpha / 2.0, df=n - 1))
    margin = t_val * std / np.sqrt(n)
    return {
        "mean":     round(mean, 6),
        "std":      round(std, 6),
        "ci_low":   round(mean - margin, 6),
        "ci_high":  round(mean + margin, 6),
        "n":        n,
    }


# --- Unit checks (verified at import) ----------------------------------------
_p_rising = np.linspace(0.1, 1.0, 10)
assert compute_kendall_tau(_p_rising) > 0, "tau must be positive for rising p_transition"
_p_const  = np.full(10, 0.5)
assert compute_kendall_tau(_p_const) == 0.0, "tau must be 0 for constant p_transition"
del _p_rising, _p_const
