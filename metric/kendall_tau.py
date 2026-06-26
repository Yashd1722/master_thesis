import numpy as np
from scipy.stats import kendalltau

def compute_kendall_tau(p_transition: np.ndarray) -> float:
    p = np.asarray(p_transition, dtype=float)
    if len(p) < 3:
        return 0.0
    if np.std(p) < 1e-10:   # constant series — model output degenerate
        return 0.0
    tau, _ = kendalltau(np.arange(len(p)), p)
    return float(tau) if np.isfinite(tau) else 0.0
