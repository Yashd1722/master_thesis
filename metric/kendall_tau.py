import numpy as np
from scipy.stats import kendalltau

def compute_kendall_tau(series: np.ndarray) -> float:
    if len(series) < 4: return float("nan")
    idx = np.arange(len(series))
    tau, _ = kendalltau(idx, series)
    return float(tau)
