"""
models/__init__.py — auto-discovering model registry.

To add a new model:
  1. Create models/my_model.py
  2. Set at module level:
       MODEL_NAME  = "my_model"
       MODEL_CLASS = "MyModelClass"
       IS_TSC      = False   # True for aeon classifiers
  3. Done.

PyTorch models interface:
  __init__(self, ts_len, num_classes)
  forward(x)        : (B, 1, T) -> logits (B, num_classes)
  predict_proba(x)  : (B, 1, T) -> softmax (B, num_classes)

TSC (aeon) models interface — Net class:
  __init__(self, ts_len, num_classes, **kwargs)
  fit(X, y)
  predict_proba(X)  : (N, num_classes)
  supported_hyperparameters() -> list[dict]
"""

import importlib
from pathlib import Path

_REGISTRY = {}
_TSC      = set()


def _discover():
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
    cls = _REGISTRY.get(name)
    if cls is None:
        return None
    return getattr(cls, "MAX_TRAIN_SAMPLES", None)
