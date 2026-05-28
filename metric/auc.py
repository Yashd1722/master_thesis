import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """
    AUC-ROC. probs is p_transition (1-D) or (N, n_classes).
    For multi-class: p_transition = 1 - p_null column.
    Returns scalar AUC in [0, 1].
    """
    if probs.ndim == 2:
        probs = probs[:, -1]
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return float("nan")
