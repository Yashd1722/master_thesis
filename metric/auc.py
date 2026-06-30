"""
metric/auc.py

AUC metrics for binary and multi-class evaluation.

Binary AUC follows Bury et al. (2021): the classifier is evaluated on its ability
to distinguish forced (bifurcating) series from null (AR1 surrogate) series.
Transition probability is always  p_transition = 1 - P(null class),
where P(null class) = probs[:, NULL_IDX] with NULL_IDX = 3 (see src/constants.py).
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auc(labels_binary: np.ndarray, p_transition: np.ndarray) -> float:
    """
    Binary ROC-AUC between forced (label=1) and null (label=0) series.

    Args:
        labels_binary: (N,) int array — 1 for forced/bifurcation, 0 for null
        p_transition:  (N,) float array — transition probability per sample,
                       computed OUTSIDE this function as  1 - probs[:, NULL_IDX]

    Returns:
        AUC score in [0, 1], or nan if computation fails.
    """
    try:
        return float(roc_auc_score(labels_binary, np.asarray(p_transition, dtype=float)))
    except ValueError:
        return float("nan")


def ovr_macro_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    One-vs-rest (OVR) macro-averaged AUC for multi-class classification.

    Each class is scored against all others; the four AUC values are averaged.
    This is the multi-class equivalent of binary AUC and measures how well the
    model separates EACH bifurcation type from the rest.

    Args:
        y_true: (N,) int array — true class indices (0=fold,1=hopf,2=trans,3=null)
        probs:  (N, C) float array — per-class softmax probabilities

    Returns:
        Macro-averaged OVR AUC in [0, 1], or nan if computation fails.
    """
    try:
        return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")
