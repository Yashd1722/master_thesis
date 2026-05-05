"""
models/__init__.py — auto-discovering model registry.

To add a new model:
  1. Create models/my_model.py
  2. Add these two lines at the top (after the docstring):
       MODEL_NAME  = "my_model"
       MODEL_CLASS = "MyModelClass"
       IS_SKLEARN  = False
  3. Done. No other file needs changing.

Interface all PyTorch models must implement:
  __init__(self, ts_len: int, num_classes: int)
  forward(x)        : (B, 1, T) -> logits  (B, num_classes)
  predict_proba(x)  : (B, 1, T) -> softmax (B, num_classes)
"""

import importlib
from pathlib import Path

_REGISTRY = {}
_SKLEARN  = set()


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
        is_sklearn = getattr(mod, "IS_SKLEARN",  False)
        if name and class_name:
            cls = getattr(mod, class_name, None)
            if cls:
                _REGISTRY[name] = cls
                if is_sklearn:
                    _SKLEARN.add(name)


_discover()


def get_model(name: str, ts_len: int, num_classes: int):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list_models()}")
    return _REGISTRY[name](ts_len=ts_len, num_classes=num_classes)


def list_models():
    return sorted(_REGISTRY.keys())


def is_sklearn_model(name: str) -> bool:
    return name in _SKLEARN
