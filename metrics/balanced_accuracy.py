import numpy as np

NAME = "balanced_accuracy"

def compute(y_true, y_pred, num_classes: int):
    # balanced accuracy = macro recall
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    recalls = []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(rec)

    return float(np.mean(recalls))
