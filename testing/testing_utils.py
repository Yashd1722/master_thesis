# testing/testing_utils.py
from __future__ import annotations

import json
import random
import re
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
# imports from your project – adjust if needed
# ---------------------------------------------------------------------
from src.dataset_loader import load_dataset
from models.CNN import CNNClassifier
from models.LSTM import LSTMClassifier
from models.CNN_LSTM import CNNLSTMClassifier


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


def infer_num_classes_from_checkpoint_dict(checkpoint: Dict[str, Any]) -> Optional[int]:
    """Try to infer the number of output classes from a checkpoint or raw state_dict."""
    state_dict = None
    for key in ["model_state_dict", "state_dict", "model"]:
        if key in checkpoint:
            state_dict = checkpoint[key]
            break

    if state_dict is None and isinstance(checkpoint, dict):
        if any("." in str(k) for k in checkpoint.keys()):
            state_dict = checkpoint

    if not isinstance(state_dict, dict):
        return None

    # Prefer explicit classifier/fc/output/linear final layers.
    candidates: List[tuple[str, int, Optional[int]]] = []
    for k, v in state_dict.items():
        lk = k.lower()
        if not lk.endswith(".weight"):
            continue
        if any(x in lk for x in ("classifier", ".fc", "output", "linear")):
            m = re.search(r"(?:classifier|fc|output|linear)\.(\d+)\.weight$", lk)
            if m:
                idx = int(m.group(1))
            else:
                m2 = re.search(r"\.(\d+)\.weight$", lk)
                idx = int(m2.group(1)) if m2 else -1

            try:
                out0 = int(v.shape[0])
            except Exception:
                out0 = None

            candidates.append((lk, idx, out0))

    if candidates:
        candidates.sort(key=lambda t: (t[1] if t[1] is not None else -1), reverse=True)
        for _, _, out0 in candidates:
            if out0 is not None:
                return out0

    # final fallback: any 2-D weight parameter's output dim
    for k, v in state_dict.items():
        if k.lower().endswith(".weight"):
            try:
                if hasattr(v, "shape") and len(v.shape) == 2:
                    return int(v.shape[0])
            except Exception:
                continue

    return None


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

    if state_dict is None and isinstance(checkpoint, dict):
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
    x: np.ndarray      # shape (T, F)
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
# 2D data formatting helpers
# ---------------------------------------------------------------------
def _ensure_2d_float32(x: Any) -> np.ndarray:
    """Convert input to float32 array with shape (T, F)."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)   # single feature -> (T,1)
    # if already 2D, leave as is; if higher, raise error
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D input, got {arr.ndim}D")
    return arr


def _pad_or_trim_2d(x: np.ndarray, target_length: int) -> np.ndarray:
    """Pad or trim along the time axis (first dimension)."""
    T, F = x.shape
    if T == target_length:
        return x
    if T > target_length:
        return x[-target_length:, :]
    out = np.zeros((target_length, F), dtype=np.float32)
    out[-T:, :] = x
    return out


def _to_model_input(x_2d: np.ndarray, input_length: int) -> np.ndarray:
    """
    Prepare input for model: shape (1, L, F) where L = input_length.
    """
    x_fixed = _pad_or_trim_2d(x_2d, input_length)
    return np.expand_dims(x_fixed, axis=0)   # (1, L, F)


# ---------------------------------------------------------------------
# loader adapter
# ---------------------------------------------------------------------
def load_test_dataset_for_inference(
    dataset_name: str,
    split: str,
    input_length: int,
    num_classes: int,
) -> Dict[str, Any]:
    """
    Load dataset through project loader and convert to window/series format.
    Assumes the loader returns dict with keys 'X', 'y', 'series' or similar.
    Adapt this if your loader returns a different structure.
    """
    loaded = load_dataset(dataset_name)

    # Normalize older loader outputs that return (sequences, labels, feature_names)
    if isinstance(loaded, (list, tuple)) and len(loaded) >= 2:
        sequences = loaded[0]
        labels = loaded[1]
        # feature_names = loaded[2] if len(loaded) > 2 else None

        series_list = []
        for i, seq in enumerate(sequences):
            item = {
                "series_id": f"{dataset_name}_{split}_series_{i}",
                "signal": seq,
                "label": None if labels is None else int(labels[i]),
            }
            series_list.append(item)

        loaded = {"X": sequences, "y": labels, "series": series_list}

    # Try to extract window samples and series items from a dict
    window_samples: List[WindowSample] = []
    series_items: List[Dict[str, Any]] = []

    if isinstance(loaded, dict):
        X = loaded.get("X", loaded.get("x"))
        y = loaded.get("y")
        series = loaded.get("series", loaded.get("series_items"))

        if X is not None:
            # Convert each sample to 2D
            for i, xi in enumerate(X):
                xi_2d = _ensure_2d_float32(xi)
                yi = int(y[i]) if y is not None else 0
                series_id = f"{dataset_name}_{split}_series_{i}"
                window_id = f"{series_id}_window_0"
                window_samples.append(WindowSample(
                    x=xi_2d,
                    y=yi,
                    series_id=series_id,
                    window_id=window_id,
                ))

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

                signal_2d = _ensure_2d_float32(signal)
                series_items.append({
                    "series_id": str(series_id),
                    "signal": signal_2d,
                    "label": None if label is None else int(label),
                    "transition_index": transition_index,
                })
        else:
            # If no series list but we have windows, create series from windows
            for i, ws in enumerate(window_samples):
                series_items.append({
                    "series_id": ws.series_id,
                    "signal": ws.x,
                    "label": ws.y,
                    "transition_index": None,
                })

    if not window_samples and not series_items:
        raise ValueError("Could not parse dataset loader output. Check loader format.")

    # Convert window samples to model input shape (1, L, F)
    normalized_window_samples: List[WindowSample] = []
    for ws in window_samples:
        normalized_window_samples.append(WindowSample(
            x=_to_model_input(ws.x, input_length),
            y=ws.y,
            series_id=ws.series_id,
            window_id=ws.window_id,
        ))

    return {
        "window_dataset": WindowDataset(normalized_window_samples),
        "series_dataset": SeriesDataset(series_items),
    }


# ---------------------------------------------------------------------
# batch and prediction helpers
# ---------------------------------------------------------------------
def collate_eval_batch(batch: Sequence[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, List[str]]]:
    # Each item["x"] is shape (1, L, F) after preprocessing
    x = torch.stack([item["x"] for item in batch], dim=0).squeeze(1)  # -> (batch, L, F)
    y = torch.stack([item["y"] for item in batch], dim=0)              # -> (batch,)

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
# progressive series prediction (multi-feature)
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
    x_full: np.ndarray,          # shape (T, F)
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
    # Ensure x_full is 2D
    x_full = _ensure_2d_float32(x_full)
    total_len = x_full.shape[0]

    reveal_indices = _build_progressive_reveal_indices(
        series_length=total_len,
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
        for step_idx, reveal_idx in enumerate(reveal_indices):
            prefix = x_full[:reveal_idx]   # (prefix_len, F)
            x_model = _to_model_input(prefix, input_length)   # (1, L, F)
            x_tensor = torch.tensor(x_model, dtype=torch.float32, device=device)

            probs = predict_batch_probabilities(model, x_tensor)[0]  # (num_classes,)
            y_pred = int(np.argmax(probs))

            row: Dict[str, Any] = {
                "series_id": str(series_id),
                "step_idx": int(step_idx),
                "reveal_index": int(reveal_idx),
                "reveal_fraction": float(reveal_idx / max(total_len, 1)),
                "series_length": int(total_len),
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
