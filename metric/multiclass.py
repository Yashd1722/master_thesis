"""
metric/multiclass.py

Multi-class classification metrics that complement the binary AUC/tau from Bury.

These are the thesis's own contribution:
  - macro_f1      : 4-class macro-averaged F1 (equal weight per class).
  - ovr_macro_auc : re-exported here for one-stop import convenience.

The four classes are (canonical Bury ordering, see src/constants.py):
  0=fold, 1=hopf, 2=transcritical, 3=null
"""

import numpy as np
from sklearn.metrics import f1_score
from metric.auc import ovr_macro_auc   # re-export for convenience


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Macro-averaged F1 across all four bifurcation classes.

    Macro averaging treats each class equally regardless of its sample count.
    For a balanced dataset (equal samples per class) this equals micro F1.

    Args:
        y_true: (N,) int array — ground-truth class indices
        y_pred: (N,) int array — predicted class indices

    Returns:
        Macro F1 in [0, 1].
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
