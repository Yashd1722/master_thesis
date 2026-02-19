import numpy as np

NAME = "acc"

def compute(y_true, y_pred, num_classes=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())

