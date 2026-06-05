import numpy as np
from sklearn.metrics import roc_auc_score

def compute_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    if probs.ndim == 2: probs = probs[:, -1]
    try: return float(roc_auc_score(labels, probs))
    except ValueError: return float("nan")
