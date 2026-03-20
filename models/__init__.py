import importlib
import inspect
import json
from pathlib import Path

import torch
import torch.nn as nn

def get_project_root() -> Path:
    """Return absolute path to the project root."""
    return Path(__file__).resolve().parents[1]

def list_available_models():
    """Scan models/ directory and return lowercased .py filenames (no __init__)."""
    models_dir = Path(__file__).resolve().parent
    py_files = [p for p in models_dir.glob("*.py") if p.name != "__init__.py"]
    return sorted([p.stem.lower() for p in py_files])

def _find_model_module_name(model_name: str) -> str:
    """Convert 'lstm' → 'models.lstm' after checking file existence."""
    model_name = model_name.lower()
    models_dir = Path(__file__).resolve().parent
    py_files = [p for p in models_dir.glob("*.py") if p.name != "__init__.py"]
    for p in py_files:
        if p.stem.lower() == model_name:
            return f"models.{p.stem}"
    available = sorted([p.stem.lower() for p in py_files])
    raise ValueError(
        f"Unknown model '{model_name}'. Available: {available}"
    )

def _pick_model_class(module, model_class: str | None = None):
    """Pick a single nn.Module subclass from the module."""
    if model_class is not None:
        if not hasattr(module, model_class):
            raise ValueError(f"Class '{model_class}' not found in {module.__name__}")
        cls = getattr(module, model_class)
        if not (inspect.isclass(cls) and issubclass(cls, nn.Module)):
            raise ValueError(f"{module.__name__}.{model_class} is not a torch.nn.Module")
        return cls

    candidates = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        if issubclass(obj, nn.Module):
            candidates.append(obj)

    if len(candidates) == 1:
        return candidates[0]

    preferred = [c for c in candidates if c.__name__.endswith(("Classifier", "Model"))]
    if len(preferred) == 1:
        return preferred[0]

    if not candidates:
        raise ValueError(f"No nn.Module found in {module.__name__}")
    names = [c.__name__ for c in candidates]
    raise ValueError(f"Multiple classes: {names}. Use --model_class.")

def build_model(model_name, input_size, num_classes, model_kwargs_json=None, model_class=None):
    """Instantiate a model by name, passing input_size, num_classes and extra kwargs."""
    module_name = _find_model_module_name(model_name)
    module = importlib.import_module(module_name)
    ModelCls = _pick_model_class(module, model_class=model_class)

    extra_kwargs = {}
    if model_kwargs_json:
        if isinstance(model_kwargs_json, str):
            extra_kwargs = json.loads(model_kwargs_json)
        elif isinstance(model_kwargs_json, dict):
            extra_kwargs = model_kwargs_json
        else:
            raise ValueError("model_kwargs_json must be a JSON string or dict")

    return ModelCls(input_size=input_size, num_classes=num_classes, **extra_kwargs)

def infer_input_dim_from_state_dict(state: dict) -> int | None:
    """Infer the input dimension from a loaded state_dict."""
    for key, value in state.items():
        if not hasattr(value, "shape"):
            continue
        shape = tuple(value.shape)

        # Common pattern for LSTM/GRU/CNN-LSTM input weights
        if ("weight_ih" in key or "weight_ih_l0" in key) and len(shape) == 2:
            return int(shape[1])

        # Pattern for conv layers
        if "conv" in key and "weight" in key and len(shape) >= 2:
            return int(shape[1])

        # Pattern for first layer features in CNNs
        if key == "features.0.weight" and len(shape) >= 2:
            return int(shape[1])

    return None

def load_model_from_checkpoint(checkpoint_path: Path, model_name: str, num_classes: int, device: torch.device, model_class: str | None = None, model_kwargs_json=None):
    """Load a model from a checkpoint, inferring input dimension if necessary."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device)
    input_dim = infer_input_dim_from_state_dict(state)

    if input_dim is None:
        raise RuntimeError(f"Could not infer input dimension from checkpoint: {checkpoint_path}")

    model = build_model(model_name, input_dim, num_classes, model_kwargs_json, model_class)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, input_dim
