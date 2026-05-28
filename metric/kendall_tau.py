import numpy as np
from scipy.stats import kendalltau


def compute_kendall_tau(series: np.ndarray) -> float:
    """
    Kendall tau trend statistic for a 1-D time series.
    Returns tau in [-1, 1]. Positive = increasing trend.
    """
    if len(series) < 4:
        return float("nan")
    idx = np.arange(len(series))
    tau, _ = kendalltau(idx, series)
    return float(tau)
