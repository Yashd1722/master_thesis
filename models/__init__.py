"""
models/__init__.py
==================
Model registry — the ONLY import any script needs.

Usage:
    from models import get_model, list_models

    model = get_model("cnn_lstm", ts_len=500, num_classes=4)
    model = get_model("lstm",     ts_len=500, num_classes=2)
    model = get_model("cnn",      ts_len=500, num_classes=4)

Models available:
    cnn_lstm  — Bury et al. (PNAS 2021) baseline — exact reproduction
    lstm      — Standalone LSTM  (Ma et al. 2025)
    cnn       — Standalone CNN   (Ma et al. 2025)

All models share the same interface:
    forward(x)       : x shape (B, 1, T) → logits (B, num_classes)
    predict_proba(x) : x shape (B, 1, T) → softmax probs (B, num_classes)

num_classes is set dynamically:
    4 → Bury mode  (fold / hopf / transcritical / null)
    2 → SDML mode  (neutral / pre_transition)
"""

from .cnn_lstm import CNNLSTM
from .lstm     import LSTMClassifier
from .cnn      import CNNClassifier

_REGISTRY = {
    "cnn_lstm": CNNLSTM,
    "lstm":     LSTMClassifier,
    "cnn":      CNNClassifier,
}


def get_model(name: str, ts_len: int, num_classes: int):
    """
    Instantiate a model by name.

    Parameters
    ----------
    name        : "cnn_lstm" | "lstm" | "cnn"
    ts_len      : input time series length (500 or 1500)
    num_classes : number of output classes (4 for Bury, 2 for SDML)

    Returns
    -------
    nn.Module with forward(x) and predict_proba(x) methods
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'.\n"
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](ts_len=ts_len, num_classes=num_classes)


def list_models():
    """Return list of all available model names."""
    return list(_REGISTRY.keys())
