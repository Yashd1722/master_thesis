# testing/testing_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import json
import logging
import numpy as np
import pandas as pd
import torch


CLASS_NAMES = ["fold", "hopf", "transcritical", "null"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def experiment_suffix(experiment: str) -> str:
    return "" if experiment == "base" else f"_{experiment}"


def build_checkpoint_name(model: str, train_dataset: str, metric: str, experiment: str) -> str:
    return f"{model}_{train_dataset}_{metric}{experiment_suffix(experiment)}.pt"


def build_run_name(model: str, train_dataset: str, metric: str, experiment: str, test_dataset: str) -> str:
    return f"{model}_{train_dataset}_{metric}{experiment_suffix(experiment)}_on_{test_dataset}"


def get_fixed_length_from_train_dataset(train_dataset: str) -> int:
    if train_dataset == "ts_500":
        return 500
    if train_dataset == "ts_1500":
        return 1500
    raise ValueError(f"Unsupported train_dataset: {train_dataset}")


def setup_logger(log_path: Path, logger_name: str) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def get_test_paths(run_name: str) -> dict:
    root = project_root()
    run_root = root / "test_results" / run_name

    dl_dir = run_root / "dl"
    csd_dir = run_root / "csd"
    compare_dir = run_root / "compare"
    log_dir = root / "test_logs"

    for path in [dl_dir, csd_dir, compare_dir, log_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "run_root": run_root,
        "dl_dir": dl_dir,
        "csd_dir": csd_dir,
        "compare_dir": compare_dir,
        "dl_log": log_dir / f"{run_name}_dl.log",
        "csd_log": log_dir / f"{run_name}_csd.log",
        "compare_log": log_dir / f"{run_name}_compare.log",
    }


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_checkpoint_state(checkpoint_path: Path, device: torch.device) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def class_index_to_name(idx: int) -> str:
    return IDX_TO_CLASS.get(int(idx), f"unknown_{idx}")


def normalize_label(label: Any) -> Optional[int]:
    if label is None:
        return None

    if isinstance(label, (int, np.integer)):
        return int(label)

    label_str = str(label).strip().lower()

    if label_str in CLASS_TO_IDX:
        return CLASS_TO_IDX[label_str]

    if label_str.isdigit():
        return int(label_str)

    return None


def ensure_2d(sample: Any) -> np.ndarray:
    arr = np.asarray(sample, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D input, got shape={arr.shape}")

    return arr


def zscore_per_feature(x: np.ndarray) -> np.ndarray:
    x = ensure_2d(x)

    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0

    return (x - mean) / std


def enforce_fixed_sequence_length(x: Any, target_length: int, mode: str = "last") -> np.ndarray:
    arr = ensure_2d(x)
    seq_len, feat_dim = arr.shape

    if seq_len == target_length:
        return arr

    if seq_len > target_length:
        if mode == "first":
            return arr[:target_length]
        return arr[-target_length:]

    pad_len = target_length - seq_len
    pad = np.zeros((pad_len, feat_dim), dtype=arr.dtype)

    if mode == "first":
        return np.concatenate([arr, pad], axis=0)
    return np.concatenate([pad, arr], axis=0)


def preprocess_for_dl(sample: Any, train_dataset: str, length_mode: str = "last") -> np.ndarray:
    """
    DL preprocessing:
    1. make sure sample is [T, C]
    2. z-score per feature/channel
    3. enforce fixed length based on training dataset
    """
    x = ensure_2d(sample)
    x = zscore_per_feature(x)
    fixed_length = get_fixed_length_from_train_dataset(train_dataset)
    x = enforce_fixed_sequence_length(x, fixed_length, mode=length_mode)
    return x


def sample_to_1d(sample: Any, feature_mode: str = "first") -> np.ndarray:
    """
    CSD preprocessing:
    Convert sample to a single 1D signal.

    feature_mode:
    - first: use first column
    - mean : mean across columns
    """
    arr = ensure_2d(sample)

    if arr.shape[1] == 1:
        return arr[:, 0].astype(np.float64)

    if feature_mode == "first":
        return arr[:, 0].astype(np.float64)

    if feature_mode == "mean":
        return arr.mean(axis=1).astype(np.float64)

    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def to_dl_tensor(sample: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert [T, C] -> tensor shaped [1, T, C]
    This is the safest default for recurrent models.
    Model-specific reshaping can still be handled in test.py if needed.
    """
    arr = ensure_2d(sample)
    tensor = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)
    return tensor


def extract_samples(dataset_obj: Any) -> List[Tuple[str, Any, Optional[int]]]:
    """
    Normalizes multiple possible dataset_loader outputs into:
    [(sample_name, sequence, label), ...]

    Supported:
    - tuple/list: (X, y, feature_names) or (X, y)
    - dict with keys like X/y/names
    """
    if dataset_obj is None:
        raise ValueError("load_dataset returned None")

    if isinstance(dataset_obj, (tuple, list)):
        if len(dataset_obj) < 1:
            raise ValueError("Dataset tuple/list is empty")

        sequences = dataset_obj[0]
        labels = dataset_obj[1] if len(dataset_obj) >= 2 else None

        if labels is None:
            labels = [None] * len(sequences)

        if len(labels) != len(sequences):
            raise ValueError("Mismatch between number of sequences and labels")

        return [
            (f"sample_{i}", sequences[i], normalize_label(labels[i]))
            for i in range(len(sequences))
        ]

    if isinstance(dataset_obj, dict):
        X = None
        for key in ["X", "x", "data", "features", "samples", "series"]:
            if key in dataset_obj:
                X = dataset_obj[key]
                break

        if X is None:
            raise KeyError("Dataset dict must contain one of: X, x, data, features, samples, series")

        names = None
        for key in ["names", "sample_ids", "ids", "file_names", "filenames"]:
            if key in dataset_obj:
                names = dataset_obj[key]
                break

        if names is None:
            names = [f"sample_{i}" for i in range(len(X))]

        labels = None
        for key in ["y", "labels", "targets"]:
            if key in dataset_obj:
                labels = dataset_obj[key]
                break

        if labels is None:
            labels = [None] * len(X)

        if len(names) != len(X):
            raise ValueError("Mismatch between number of names and sequences")

        if len(labels) != len(X):
            raise ValueError("Mismatch between number of labels and sequences")

        return [
            (str(names[i]), X[i], normalize_label(labels[i]))
            for i in range(len(X))
        ]

    raise TypeError(f"Unsupported dataset type: {type(dataset_obj)}")
