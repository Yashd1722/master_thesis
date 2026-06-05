import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def compute_accuracy(labels: np.ndarray, preds: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds).tolist()
    }#
