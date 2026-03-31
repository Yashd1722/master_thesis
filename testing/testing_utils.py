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

from src.dataset_loader import load_dataset
from models.CNN import CNNClassifier
from models.LSTM import LSTMClassifier
from models.CNN_LSTM import CNNLSTMClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def save_numpy_confusion_inputs(y_true: np.ndarray, y_pred: np.ndarray, save_dir: Path) -> None:
    ensure_dir(save_dir)
    np.save(save_dir / "y_true.npy", np.asarray(y_true, dtype=np.int64))
    np.save(save_dir / "y_pred.npy", np.asarray(y_pred, dtype=np.int64))


def summarise_run_config(**kwargs: Any) -> Dict[str, Any]:
    return dict(kwargs)


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
        "input_size",
    ]:
        if key in checkpoint:
            meta[key] = checkpoint[key]

    if "meta" in checkpoint and isinstance(checkpoint["meta"], dict):
        for key, value in checkpoint["meta"].items():
            meta.setdefault(key, value)

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
        "Please save 'model_name' in the checkpoint."
    )


def resolve_class_names(num_classes: int, ckpt_meta: Dict[str, Any]) -> List[str]:
    if "class_names" in ckpt_meta and ckpt_meta["class_names"] is not None:
        class_names = list(ckpt_meta["class_names"])
        if len(class_names) == num_classes:
            return [str(x) for x in class_names]
    return [f"class_{i}" for i in range(num_classes)]


def infer_num_classes_from_checkpoint_dict(checkpoint: Dict[str, Any]) -> Optional[int]:
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

    candidates: List[Tuple[str, int, Optional[int]]] = []
    for key, value in state_dict.items():
        key_lower = key.lower()
        if not key_lower.endswith(".weight"):
            continue

        if any(token in key_lower for token in ("classifier", ".fc", "output", "linear")):
            match = re.search(r"(?:classifier|fc|output|linear)\.(\d+)\.weight$", key_lower)
            layer_idx = int(match.group(1)) if match else -1
            out_dim = int(value.shape[0]) if hasattr(value, "shape") else None
            candidates.append((key_lower, layer_idx, out_dim))

    if candidates:
        candidates.sort(key=lambda item: item[1], reverse=True)
        for _, _, out_dim in candidates:
            if out_dim is not None:
                return out_dim

    for key, value in state_dict.items():
        if key.lower().endswith(".weight") and hasattr(value, "shape") and len(value.shape) == 2:
            return int(value.shape[0])

    return None


def _build_model(
    model_name: str,
    num_classes: int,
    input_size: int = 2,
) -> torch.nn.Module:
    model_name = model_name.lower()
    if model_name == "cnn":
        return CNNClassifier(input_size=input_size, num_classes=num_classes)
    if model_name == "lstm":
        return LSTMClassifier(input_size=input_size, num_classes=num_classes)
    if model_name in {"cnn_lstm", "cnnlstm"}:
        return CNNLSTMClassifier(input_size=input_size, num_classes=num_classes)
    raise ValueError(f"Unsupported model name: {model_name}")


def build_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    checkpoint_path: Path,
    model_name: str,
    num_classes: int,
    input_size: int = 2,
    input_length: int = 500,
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    _ = input_length
    model = _build_model(
        model_name=model_name,
        num_classes=num_classes,
        input_size=input_size,
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
            "Expected one of: model_state_dict, state_dict, model (or raw state_dict)."
        )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


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
        sample = self.samples[idx]
        return {
            "x": torch.tensor(sample.x, dtype=torch.float32),
            "y": torch.tensor(sample.y, dtype=torch.long),
            "series_id": sample.series_id,
            "window_id": sample.window_id,
        }


class SeriesDataset:
    def __init__(self, series_items: List[Dict[str, Any]]) -> None:
        self.series_items = series_items


def _ensure_2d_float32(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D input, got shape {arr.shape}")
    return arr


def _ensure_1d_float32(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _pad_or_trim_2d(x: np.ndarray, target_length: int) -> np.ndarray:
    time_steps, num_features = x.shape
    if time_steps == target_length:
        return x
    if time_steps > target_length:
        return x[-target_length:, :]

    out = np.zeros((target_length, num_features), dtype=np.float32)
    out[-time_steps:, :] = x
    return out


def _to_model_input(x_2d: np.ndarray, input_length: int) -> np.ndarray:
    x_fixed = _pad_or_trim_2d(x_2d, input_length)
    return np.expand_dims(x_fixed, axis=0)


def _copy_optional_series_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    optional_keys = [
        "raw_signal",
        "smooth_signal",
        "residual_signal",
        "time_index",
        "metadata",
    ]
    out: Dict[str, Any] = {}
    for key in optional_keys:
        if key not in item:
            continue
        value = item[key]
        if key == "metadata":
            out[key] = dict(value) if isinstance(value, dict) else value
        else:
            out[key] = _ensure_1d_float32(value)
    return out


def _build_series_item_from_array(
    dataset_name: str,
    split: str,
    series_idx: int,
    signal: Any,
    label: Optional[int],
) -> Dict[str, Any]:
    signal_2d = _ensure_2d_float32(signal)
    series_item: Dict[str, Any] = {
        "series_id": f"{dataset_name}_{split}_series_{series_idx}",
        "signal": signal_2d,
        "label": None if label is None else int(label),
        "transition_index": None,
    }

    if signal_2d.shape[1] >= 1:
        series_item["raw_signal"] = signal_2d[:, 0].astype(np.float32)
    if signal_2d.shape[1] >= 2:
        series_item["residual_signal"] = signal_2d[:, 1].astype(np.float32)
        series_item["smooth_signal"] = (signal_2d[:, 0] - signal_2d[:, 1]).astype(np.float32)
    else:
        series_item["smooth_signal"] = signal_2d[:, 0].astype(np.float32)

    series_item["time_index"] = np.arange(signal_2d.shape[0], dtype=np.float32)
    series_item["metadata"] = {"dataset_name": dataset_name, "split": split}
    return series_item


def load_test_dataset_for_inference(
    dataset_name: str,
    split: str,
    input_length: int,
    num_classes: int,
) -> Dict[str, Any]:
    _ = num_classes
    loaded = load_dataset(dataset_name=dataset_name, split=split, seq_len=input_length)

    if isinstance(loaded, (list, tuple)) and len(loaded) >= 2:
        sequences = loaded[0]
        labels = loaded[1]
        series_list = []
        for idx, sequence in enumerate(sequences):
            label = None if labels is None else int(labels[idx])
            series_list.append(
                _build_series_item_from_array(
                    dataset_name=dataset_name,
                    split=split,
                    series_idx=idx,
                    signal=sequence,
                    label=label,
                )
            )
        loaded = {"X": sequences, "y": labels, "series": series_list}

    window_samples: List[WindowSample] = []
    series_items: List[Dict[str, Any]] = []

    if not isinstance(loaded, dict):
        raise ValueError("Dataset loader output must be a dict or tuple/list.")

    X = loaded.get("X", loaded.get("x"))
    y = loaded.get("y")
    series = loaded.get("series", loaded.get("series_items"))

    if X is not None:
        for idx, xi in enumerate(X):
            xi_2d = _ensure_2d_float32(xi)
            yi = int(y[idx]) if y is not None else 0
            series_id = f"{dataset_name}_{split}_series_{idx}"
            window_id = f"{series_id}_window_0"
            window_samples.append(
                WindowSample(
                    x=_to_model_input(xi_2d, input_length),
                    y=yi,
                    series_id=series_id,
                    window_id=window_id,
                )
            )

    if series is not None:
        for idx, item in enumerate(series):
            if isinstance(item, dict):
                signal = item.get("signal", item.get("x", item.get("series")))
                label = item.get("label", item.get("y"))
                transition_index = item.get("transition_index", None)
                series_id = str(item.get("series_id", f"{dataset_name}_{split}_series_{idx}"))

                series_item = {
                    "series_id": series_id,
                    "signal": _ensure_2d_float32(signal),
                    "label": None if label is None else int(label),
                    "transition_index": None if transition_index is None else int(transition_index),
                }
                series_item.update(_copy_optional_series_fields(item))
            else:
                label = None if y is None else int(y[idx])
                series_item = _build_series_item_from_array(
                    dataset_name=dataset_name,
                    split=split,
                    series_idx=idx,
                    signal=item,
                    label=label,
                )
            series_items.append(series_item)
    else:
        for idx, sample in enumerate(window_samples):
            series_items.append(
                _build_series_item_from_array(
                    dataset_name=dataset_name,
                    split=split,
                    series_idx=idx,
                    signal=sample.x.squeeze(0),
                    label=sample.y,
                )
            )

    if not window_samples and not series_items:
        raise ValueError("Could not parse dataset loader output. Check loader format.")

    return {
        "window_dataset": WindowDataset(window_samples),
        "series_dataset": SeriesDataset(series_items),
    }


def collate_eval_batch(
    batch: Sequence[Dict[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, List[str]]]:
    x = torch.stack([item["x"] for item in batch], dim=0).squeeze(1)
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
    x_full = _ensure_2d_float32(x_full)
    total_len = x_full.shape[0]

    reveal_lengths = _build_progressive_reveal_indices(
        series_length=total_len,
        progressive_start_frac=progressive_start_frac,
        progressive_end_frac=progressive_end_frac,
        progressive_num_steps=progressive_num_steps,
        min_prefix_len=min_prefix_len,
    )
    if not reveal_lengths:
        return []

    records: List[Dict[str, Any]] = []
    model.eval()

    with torch.no_grad():
        for step_idx, reveal_len in enumerate(reveal_lengths):
            prefix = x_full[:reveal_len]
            x_model = _to_model_input(prefix, input_length)
            x_tensor = torch.tensor(x_model, dtype=torch.float32, device=device)

            probs = predict_batch_probabilities(model, x_tensor)[0]
            y_pred = int(np.argmax(probs))

            row: Dict[str, Any] = {
                "series_id": str(series_id),
                "step_idx": int(step_idx),
                "prefix_length": int(reveal_len),
                "reveal_index": int(reveal_len - 1),
                "reveal_fraction": float(reveal_len / max(total_len, 1)),
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
