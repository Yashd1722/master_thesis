"""
models/__init__.py — simple model registry.

To add a new model:
  1. Create models/my_model.py
  2. Set at module level:
       MODEL_NAME  = "my_model"    # key used in --model CLI arg and config.yaml
       MODEL_CLASS = "MyModelClass"
       IS_TSC      = True          # True for aeon classifiers, False for PyTorch
  3. Done — it will be auto-discovered here.

TSC (aeon) model interface:
  __init__(ts_len, num_classes, **kwargs)
  fit(X, y)              — X: (N, C, L) numpy array, y: (N,) int
  predict_proba(X)       — returns (N, num_classes) numpy array
  save(path) / load(path)

DL (PyTorch) model interface:
  __init__(ts_len, num_classes)
  forward(x)             — (B, 1, T) -> logits (B, num_classes)
  predict_proba(x)       — (B, 1, T) -> softmax (B, num_classes)
"""

import importlib
from pathlib import Path

_REGISTRY = {}   # name -> class
_TSC      = set()  # names of TSC (aeon) models


def _discover():
    """Scan models/*.py and register anything with MODEL_NAME + MODEL_CLASS."""
    models_dir = Path(__file__).parent
    for fpath in sorted(models_dir.glob("*.py")):
        if fpath.stem.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"models.{fpath.stem}")
        except Exception:
            continue
        name       = getattr(mod, "MODEL_NAME",  None)
        class_name = getattr(mod, "MODEL_CLASS", None)
        is_tsc     = getattr(mod, "IS_TSC",      False)
        if name and class_name:
            cls = getattr(mod, class_name, None)
            if cls:
                _REGISTRY[name] = cls
                if is_tsc:
                    _TSC.add(name)


_discover()


def get_model(name: str, ts_len: int, num_classes: int, **kwargs):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list_models()}")
    return _REGISTRY[name](ts_len=ts_len, num_classes=num_classes, **kwargs)


def list_models():
    return sorted(_REGISTRY.keys())


def is_tsc_model(name: str) -> bool:
    return name in _TSC


def get_max_train_samples(name: str):
    """Return MAX_TRAIN_SAMPLES for a model, or None if unlimited."""
    import sys as _sys
    cls = _REGISTRY.get(name)
    if cls is None:
        return None
    val = cls.__dict__.get("MAX_TRAIN_SAMPLES")
    if val is None:
        mod = _sys.modules.get(cls.__module__)
        if mod is not None:
            val = getattr(mod, "MAX_TRAIN_SAMPLES", None)
    return val
