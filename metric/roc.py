import numpy as np
from sklearn.metrics import roc_curve

def compute_roc(labels: np.ndarray, probs: np.ndarray):
    if probs.ndim == 2: probs = probs[:, -1]
    fpr, tpr, thresh = roc_curve(labels, probs)
    return fpr.tolist(), tpr.tolist(), thresh.tolist()
