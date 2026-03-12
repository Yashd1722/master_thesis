# metrics/confusion_matrix.py

from __future__ import annotations

import pandas as pd
from sklearn.metrics import confusion_matrix


def compute_binary_confusion_matrix(y_true, y_pred) -> pd.DataFrame:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return pd.DataFrame(
        [
            {"label": "tn", "value": int(tn)},
            {"label": "fp", "value": int(fp)},
            {"label": "fn", "value": int(fn)},
            {"label": "tp", "value": int(tp)},
        ]
    )
