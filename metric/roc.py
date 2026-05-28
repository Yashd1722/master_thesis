import numpy as np
from sklearn.metrics import roc_curve


def compute_roc(labels: np.ndarray, probs: np.ndarray):
    """
    ROC curve. Returns (fpr, tpr, thresholds) as plain Python lists.
    probs may be 1-D p_transition or 2-D (N, n_classes).
    """
    if probs.ndim == 2:
        probs = probs[:, -1]
    fpr, tpr, thresh = roc_curve(labels, probs)
    return fpr.tolist(), tpr.tolist(), thresh.tolist()
