# metrics/roc_auc.py

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score


def compute_binary_roc_auc(y_true, y_score, score_name: str = "score"):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if len(y_true) == 0:
        raise ValueError("y_true is empty.")
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")
    if len(np.unique(y_true)) < 2:
        raise ValueError("ROC/AUC requires at least two classes in y_true.")

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_value = float(roc_auc_score(y_true, y_score))

    roc_df = pd.DataFrame(
        {
            "score_name": score_name,
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds,
        }
    )
    return roc_df, auc_value
