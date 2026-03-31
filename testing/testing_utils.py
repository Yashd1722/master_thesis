# testing/testing_utils.py

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ---------------------------------------------------------------------
# repo root
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------
# optional project imports
# adapt these imports if your repo names differ
# ---------------------------------------------------------------------
from src.dataset_loader import load_dataset  # adjust only if needed
from models.CNN import CNNClassifier  # adjust only if needed
from models.LSTM import LSTMClassifier  # adjust only if needed
from models.CNN_LSTM import CNNLSTMClassifier  # adjust only if needed


# ---------------------------------------------------------------------
# basic utilities
# ---------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj: Dict[str, Any], save_path: Path) -> None:
    ensure_dir(save_path.parent)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_numpy_confusion_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: Path,
) -> None:
    ensure_dir(save_dir)
    np.save(save_dir / "y_true.npy", np.asarray(y_true, dtype=np.int64))
    np.save(save_dir / "y_pred.npy", np.asarray(y_pred, dtype=np.int64))


def summarise_run_config(**kwargs: Any) -> Dict[str, Any]:
    return dict(kwargs)


# ---------------------------------------------------------------------
# device
# ---------------------------------------------------------------------
def resolve_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device argument: {device_arg}")


# ---------------------------------------------------------------------
# checkpoint helpers
# ---------------------------------------------------------------------
def load_checkpoint_safely(checkpoint_path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint is not a dict: {checkpoint_path}")
    return checkpoint


def extract_checkpoint_metadata(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}

    for key in [
        "model_name",
        "dataset",
        "metric",
        "input_length",
        "seq_len",
        "num_classes",
        "class_names",
        "model_config",
    ]:
        if key in checkpoint:
            meta[key] = checkpoint[key]

    if "meta" in checkpoint and isinstance(checkpoint["meta"], dict):
        for k, v in checkpoint["meta"].items():
            meta.setdefault(k, v)

    return meta


def infer_model_name_from_checkpoint(checkpoint_path: Path, ckpt_meta: Dict[str, Any]) -> str:
    if "model_name" in ckpt_meta:
        return str(ckpt_meta["model_name"])

    stem = checkpoint_path.stem.lower()

    if "cnn_lstm" in stem:
        return "cnn_lstm"
    if "lstm" in stem:
        return "lstm"
    if "cnn" in stem:
        return "cnn"

    raise ValueError(
        "Could not infer model name from checkpoint metadata or filename. "
        "Please save 'model_name' in checkpoint."
    )


def resolve_class_names(num_classes: int, ckpt_meta: Dict[str, Any]) -> List[str]:
    if "class_names" in ckpt_meta and ckpt_meta["class_names"] is not None:
        class_names = list(ckpt_meta["class_names"])
        if len(class_names) == num_classes:
            return [str(x) for x in class_names]

    return [f"class_{i}" for i in range(num_classes)]


# ---------------------------------------------------------------------
# model building
# ---------------------------------------------------------------------
def _build_model(
    model_name: str,
    num_classes: int,
    input_size: int | None = None,
    input_length: int | None = None,
) -> torch.nn.Module:
    model_name = model_name.lower()
    effective_input_size = int(input_size if input_size is not None else input_length if input_length is not None else 2)

    if model_name == "cnn":
        return CNNClassifier(input_size=effective_input_size, num_classes=num_classes)

    if model_name == "lstm":
        return LSTMClassifier(input_size=effective_input_size, num_classes=num_classes)

    if model_name in {"cnn_lstm", "cnnlstm"}:
        return CNNLSTMClassifier(input_size=effective_input_size, num_classes=num_classes)

    raise ValueError(f"Unsupported model name: {model_name}")


def build_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    checkpoint_path: Path,
    model_name: str,
    num_classes: int,
    input_size: int = 2,
    input_length: int = 2,
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    model = _build_model(
        model_name=model_name,
        num_classes=num_classes,
        input_size=input_size,
        input_length=input_length,
    )

    state_dict = None
    for key in ["model_state_dict", "state_dict", "model"]:
        if key in checkpoint:
            state_dict = checkpoint[key]
            break

    # directly saved state dict (checkpoint is a state_dict itself)
    if state_dict is None and isinstance(checkpoint, dict):
        # heuristic: parameter keys typically contain '.' and 'weight' or 'bias'
        if any("." in str(k) for k in checkpoint.keys()):
            state_dict = checkpoint

    if state_dict is None:
        raise KeyError(
            f"No model state dict found in checkpoint: {checkpoint_path}. "
            "Expected one of: model_state_dict, state_dict, model (or raw state_dict)"
        )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------
# dataset containers
# ---------------------------------------------------------------------
@dataclass
class WindowSample:
    x: np.ndarray
    y: int
    series_id: str
    window_id: str


class WindowDataset(Dataset):
    def __init__(self, samples: List[WindowSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "x": torch.tensor(s.x, dtype=torch.float32),
            "y": torch.tensor(s.y, dtype=torch.long),
            "series_id": s.series_id,
            "window_id": s.window_id,
        }


class SeriesDataset:
    def __init__(self, series_items: List[Dict[str, Any]]) -> None:
        self.series_items = series_items


# ---------------------------------------------------------------------
# data formatting helpers
# ---------------------------------------------------------------------
def _ensure_1d_float32(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    return arr


def _pad_or_trim_1d(x: np.ndarray, target_length: int) -> np.ndarray:
    x = _ensure_1d_float32(x)

    if len(x) == target_length:
        return x

    if len(x) > target_length:
        return x[-target_length:]

    out = np.zeros(target_length, dtype=np.float32)
    out[-len(x):] = x
    return out


def _to_model_input(x_1d: np.ndarray, input_length: int) -> np.ndarray:
    """
    Returns shape (1, L) for conv/recurrent models expecting channel-first or simple sequence input.
    Adjust this if your models expect a different shape.
    """
    x_fixed = _pad_or_trim_1d(x_1d, input_length)
    return np.expand_dims(x_fixed, axis=0)  # (1, L)


def _extract_series_items_from_loaded_dataset(
    loaded: Any,
    dataset_name: str,
    split: str,
) -> Tuple[List[WindowSample], List[Dict[str, Any]]]:
    """
    This is the only repo-dependent section.
    It converts your project dataset loader output into:
    - sample/window dataset
    - full series dataset for progressive inference

    Expected flexible input patterns:
    1) dict with keys like:
       loaded["X_test"], loaded["y_test"], loaded["series_test"]
    2) dict with split nested:
       loaded["test"]["X"], loaded["test"]["y"], loaded["test"]["series"]
    3) already prepared objects:
       loaded["window_samples"], loaded["series_items"]
    """

    # Case A: already normalized by your loader
    if isinstance(loaded, dict) and "window_samples" in loaded and "series_items" in loaded:
        return loaded["window_samples"], loaded["series_items"]

    # Case B: split nested dict
    if isinstance(loaded, dict) and split in loaded and isinstance(loaded[split], dict):
        split_obj = loaded[split]

        X = split_obj.get("X", split_obj.get("x"))
        y = split_obj.get("y")
        series = split_obj.get("series", split_obj.get("series_items"))

        if X is not None and y is not None:
            window_samples: List[WindowSample] = []
            for i, (xi, yi) in enumerate(zip(X, y)):
                series_id = f"{dataset_name}_{split}_series_{i}"
                window_id = f"{series_id}_window_0"
                window_samples.append(
                    WindowSample(
                        x=_ensure_1d_float32(xi),
                        y=int(yi),
                        series_id=series_id,
                        window_id=window_id,
                    )
                )

            series_items: List[Dict[str, Any]] = []
            if series is not None:
                for i, item in enumerate(series):
                    if isinstance(item, dict):
                        signal = item.get("signal", item.get("x", item.get("series")))
                        label = item.get("label", item.get("y"))
                        transition_index = item.get("transition_index", None)
                        series_id = item.get("series_id", f"{dataset_name}_{split}_series_{i}")
                    else:
                        signal = item
                        label = None
                        transition_index = None
                        series_id = f"{dataset_name}_{split}_series_{i}"

                    series_items.append(
                        {
                            "series_id": str(series_id),
                            "signal": _ensure_1d_float32(signal),
                            "label": None if label is None else int(label),
                            "transition_index": transition_index,
                        }
                    )
            else:
                for i, (xi, yi) in enumerate(zip(X, y)):
                    series_items.append(
                        {
                            "series_id": f"{dataset_name}_{split}_series_{i}",
                            "signal": _ensure_1d_float32(xi),
                            "label": int(yi),
                            "transition_index": None,
                        }
                    )

            return window_samples, series_items

    # Case C: flat dict
    if isinstance(loaded, dict):
        X = loaded.get(f"X_{split}", loaded.get("X"))
        y = loaded.get(f"y_{split}", loaded.get("y"))
        series = loaded.get(f"series_{split}", loaded.get("series_items", loaded.get("series")))

        if X is not None and y is not None:
            window_samples = []
            for i, (xi, yi) in enumerate(zip(X, y)):
                series_id = f"{dataset_name}_{split}_series_{i}"
                window_id = f"{series_id}_window_0"
                window_samples.append(
                    WindowSample(
                        x=_ensure_1d_float32(xi),
                        y=int(yi),
                        series_id=series_id,
                        window_id=window_id,
                    )
                )

            series_items = []
            if series is not None:
                for i, item in enumerate(series):
                    if isinstance(item, dict):
                        signal = item.get("signal", item.get("x", item.get("series")))
                        label = item.get("label", item.get("y"))
                        transition_index = item.get("transition_index", None)
                        series_id = item.get("series_id", f"{dataset_name}_{split}_series_{i}")
                    else:
                        signal = item
                        label = None
                        transition_index = None
                        series_id = f"{dataset_name}_{split}_series_{i}"

                    series_items.append(
                        {
                            "series_id": str(series_id),
                            "signal": _ensure_1d_float32(signal),
                            "label": None if label is None else int(label),
                            "transition_index": transition_index,
                        }
                    )
            else:
                for i, (xi, yi) in enumerate(zip(X, y)):
                    series_items.append(
                        {
                            "series_id": f"{dataset_name}_{split}_series_{i}",
                            "signal": _ensure_1d_float32(xi),
                            "label": int(yi),
                            "transition_index": None,
                        }
                    )

            return window_samples, series_items

    raise ValueError(
        "Could not parse dataset loader output into window samples and series items. "
        "Please adapt _extract_series_items_from_loaded_dataset() to your exact loader format."
    )


def load_test_dataset_for_inference(
    dataset_name: str,
    split: str,
    input_length: int,
    num_classes: int,
) -> Dict[str, Any]:
    """
    Loads dataset through the project loader and converts it into:
    - window_dataset: for normal classification evaluation
    - series_dataset: for progressive per-series prediction
    """
    loaded = load_dataset(dataset_name=dataset_name)

    window_samples, series_items = _extract_series_items_from_loaded_dataset(
        loaded=loaded,
        dataset_name=dataset_name,
        split=split,
    )

    normalized_window_samples: List[WindowSample] = []
    for s in window_samples:
        normalized_window_samples.append(
            WindowSample(
                x=_to_model_input(s.x, input_length=input_length),
                y=int(s.y),
                series_id=str(s.series_id),
                window_id=str(s.window_id),
            )
        )

    return {
        "window_dataset": WindowDataset(normalized_window_samples),
        "series_dataset": SeriesDataset(series_items),
    }


# ---------------------------------------------------------------------
# batch / prediction helpers
# ---------------------------------------------------------------------
def collate_eval_batch(batch: Sequence[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, List[str]]]:
    x = torch.stack([item["x"] for item in batch], dim=0)
    y = torch.stack([item["y"] for item in batch], dim=0)

    meta = {
        "series_id": [str(item["series_id"]) for item in batch],
        "window_id": [str(item["window_id"]) for item in batch],
    }
    return x, y, meta


def _forward_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits


def predict_batch_probabilities(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    logits = _forward_logits(model, x)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


# ---------------------------------------------------------------------
# progressive series prediction
# ---------------------------------------------------------------------
def _build_progressive_reveal_indices(
    series_length: int,
    progressive_start_frac: float,
    progressive_end_frac: float,
    progressive_num_steps: int,
    min_prefix_len: int,
) -> List[int]:
    if series_length <= 0:
        return []

    start_idx = max(min_prefix_len, int(round(series_length * progressive_start_frac)))
    end_idx = max(start_idx, int(round(series_length * progressive_end_frac)))

    start_idx = min(start_idx, series_length)
    end_idx = min(end_idx, series_length)

    if progressive_num_steps <= 1:
        return [end_idx]

    indices = np.linspace(start_idx, end_idx, progressive_num_steps)
    indices = np.unique(indices.astype(int))
    indices = np.clip(indices, 1, series_length)

    return [int(v) for v in indices.tolist()]


def create_progressive_series_records(
    model: torch.nn.Module,
    device: torch.device,
    x_full: np.ndarray,
    series_id: str,
    y_true: Optional[int],
    transition_index: Optional[int],
    input_length: int,
    class_names: List[str],
    progressive_start_frac: float,
    progressive_end_frac: float,
    progressive_num_steps: int,
    min_prefix_len: int,
) -> List[Dict[str, Any]]:
    x_full = _ensure_1d_float32(x_full)
    reveal_indices = _build_progressive_reveal_indices(
        series_length=len(x_full),
        progressive_start_frac=progressive_start_frac,
        progressive_end_frac=progressive_end_frac,
        progressive_num_steps=progressive_num_steps,
        min_prefix_len=min_prefix_len,
    )

    if len(reveal_indices) == 0:
        return []

    records: List[Dict[str, Any]] = []

    model.eval()
    with torch.no_grad():
        for step_idx, reveal_index in enumerate(reveal_indices):
            prefix = x_full[:reveal_index]
            x_model = _to_model_input(prefix, input_length=input_length)
            x_tensor = torch.tensor(x_model, dtype=torch.float32, device=device).unsqueeze(0)

            probs = predict_batch_probabilities(model, x_tensor)[0]
            y_pred = int(np.argmax(probs))

            row: Dict[str, Any] = {
                "series_id": str(series_id),
                "step_idx": int(step_idx),
                "reveal_index": int(reveal_index),
                "reveal_fraction": float(reveal_index / max(len(x_full), 1)),
                "series_length": int(len(x_full)),
                "y_pred": y_pred,
                "pred_class_name": class_names[y_pred],
                "transition_index": None if transition_index is None else int(transition_index),
            }

            if y_true is not None:
                row["y_true"] = int(y_true)

            for class_idx, class_name in enumerate(class_names):
                row[f"prob_{class_name}"] = float(probs[class_idx])

            records.append(row)

    return records
