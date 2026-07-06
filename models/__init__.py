"""
models/__init__.py — the model registry.

TSC models (aeon) all share one wrapper in models/tsc.py; DL models (PyTorch)
each have their own file. Everything is registered explicitly below — no
auto-discovery.

Interface every model exposes:
    fit(X, y) / predict_proba(X) / save(path) / load(path)   (TSC)
    forward(x) / predict_proba(x)                            (DL, nn.Module)
"""
from models.tsc import TSCModel, TSC_SPECS
from models.cnn_lstm import CNNLSTM
from models.lstm import LSTMClassifier
from models.inceptiontime import InceptionTime

# DL builders: name -> class (constructed with ts_len, num_classes)
_DL = {
    "cnn_lstm":      CNNLSTM,
    "lstm":          LSTMClassifier,
    "inceptiontime": InceptionTime,
}


def get_model(name, ts_len, num_classes, **kwargs):
    if name in TSC_SPECS:
        return TSCModel(name, ts_len=ts_len, num_classes=num_classes, **kwargs)
    if name in _DL:
        return _DL[name](ts_len=ts_len, num_classes=num_classes)
    raise ValueError(f"Unknown model '{name}'. Available: {list_models()}")


def list_models():
    return sorted(list(TSC_SPECS) + list(_DL))


def is_tsc_model(name):
    return name in TSC_SPECS


def get_max_train_samples(name):
    """MAX_TRAIN_SAMPLES for a TSC model, or None (DL / unlimited)."""
    return TSC_SPECS.get(name, {}).get("max_samples")
