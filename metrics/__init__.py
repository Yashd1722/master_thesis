from .acc import compute as acc
from .f1_macro import compute as f1_macro
from .balanced_accuracy import compute as balanced_accuracy

METRICS = {
    "acc": acc,
    "f1_macro": f1_macro,
    "balanced_accuracy": balanced_accuracy,
}
