from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import torch


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    return value.strip("_")


def build_run_name(model_name: str, dataset_name: str, metric_name: str) -> str:
    if not model_name or not dataset_name or not metric_name:
        raise ValueError("model_name, dataset_name, and metric_name must all be non-empty.")
    return f"{safe_name(model_name)}_{safe_name(dataset_name)}_{safe_name(metric_name)}"


def setup_logger(log_path: Path, logger_name: Optional[str] = None) -> logging.Logger:
    ensure_dir(log_path.parent)

    if logger_name is None:
        logger_name = f"testing.{log_path.stem}"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def get_test_paths(run_name: str) -> dict[str, Path]:
    root = project_root()
    test_logs = ensure_dir(root / "test_logs")

    dl_dir = ensure_dir(root / "test_results" / "dl" / run_name)
    csd_dir = ensure_dir(root / "test_results" / "csd" / run_name)
    cmp_dir = ensure_dir(root / "test_results" / "comparison" / run_name)

    null_dir = ensure_dir(csd_dir / "null_models")
    null_scores_dir = ensure_dir(csd_dir / "null_scores")

    return {
        "root": root,
        "test_logs": test_logs,
        "dl_dir": dl_dir,
        "csd_dir": csd_dir,
        "cmp_dir": cmp_dir,
        "null_dir": null_dir,
        "null_scores_dir": null_scores_dir,
        "dl_log": test_logs / f"{run_name}_dl.log",
        "csd_log": test_logs / f"{run_name}_csd.log",
        "compare_log": test_logs / f"{run_name}_compare.log",
    }


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def save_json(data: dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_probability_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required probability columns: {missing}")


def add_transition_probability(
    df: pd.DataFrame,
    fold_col: str = "p_fold",
    hopf_col: str = "p_hopf",
    transcritical_col: str = "p_transcritical",
    out_col: str = "p_transition",
) -> pd.DataFrame:
    validate_probability_columns(df, [fold_col, hopf_col, transcritical_col])
    out = df.copy()
    out[out_col] = out[fold_col] + out[hopf_col] + out[transcritical_col]
    return out


def load_checkpoint_state(checkpoint_path: Path, device: torch.device) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint at {checkpoint_path} did not load as a dict.")
    return state


def class_index_to_name(idx: int) -> str:
    mapping = {
        0: "fold",
        1: "hopf",
        2: "transcritical",
        3: "null",
    }
    if idx not in mapping:
        raise ValueError(f"Unknown class index: {idx}")
    return mapping[idx]


def _coerce_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def sample_to_1d(x: Any, feature_mode: str = "first") -> np.ndarray:
    arr = _coerce_numpy(x)

    if arr.ndim == 1:
        out = arr.astype(float)

    elif arr.ndim == 2:
        if feature_mode == "first":
            out = arr[:, 0].astype(float)
        elif feature_mode == "mean":
            out = np.nanmean(arr.astype(float), axis=1)
        else:
            raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    elif arr.ndim == 3:
        arr2 = arr[0]
        if feature_mode == "first":
            out = arr2[:, 0].astype(float)
        elif feature_mode == "mean":
            out = np.nanmean(arr2.astype(float), axis=1)
        else:
            raise ValueError(f"Unsupported feature_mode: {feature_mode}")
    else:
        raise ValueError(f"Unsupported sample shape for 1D conversion: {arr.shape}")

    out = np.asarray(out, dtype=float).reshape(-1)
    out = out[np.isfinite(out)]

    if out.size < 3:
        raise ValueError("Signal has fewer than 3 finite values after conversion.")
    return out


def extract_samples(dataset_obj: Any) -> list[tuple[str, Any]]:
    if dataset_obj is None:
        raise ValueError("load_dataset returned None.")

    if isinstance(dataset_obj, tuple) and len(dataset_obj) == 3:
        sequences, _, _ = dataset_obj
        if sequences is None or len(sequences) == 0:
            raise ValueError("No sequences in dataset.")
        return [(f"sample_{i}", seq) for i, seq in enumerate(sequences)]

    if isinstance(dataset_obj, dict):
        for key in ("X", "x", "data", "features"):
            if key in dataset_obj:
                X = dataset_obj[key]
                break
        else:
            raise KeyError("Dataset dict missing one of: X, x, data, features")

        names = dataset_obj.get("names") or dataset_obj.get("file_names") or dataset_obj.get("filenames")
        if names is None:
            names = [f"sample_{i}" for i in range(len(X))]

        if len(names) != len(X):
            raise ValueError("Dataset names length does not match sample count.")

        return [(str(n), X[i]) for i, n in enumerate(names)]

    if isinstance(dataset_obj, (list, tuple)):
        out = []
        for i, item in enumerate(dataset_obj):
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("file_name") or item.get("filename") or f"sample_{i}")
                for key in ("x", "X", "data", "features"):
                    if key in item:
                        sample = item[key]
                        break
                else:
                    raise KeyError(f"No data key in sample dict at index {i}")
            else:
                name, sample = f"sample_{i}", item
            out.append((name, sample))
        return out

    raise TypeError(f"Unsupported dataset type: {type(dataset_obj)}")


def enforce_fixed_sequence_length(
    x: Any,
    target_length: int,
    mode: str = "last",
) -> np.ndarray:
    """
    Synthetic input -> fixed sequence length
    Empirical input -> do not call this helper

    mode:
        - 'last': keep last target_length points
        - 'first': keep first target_length points
    """
    arr = _coerce_numpy(x)

    if arr.ndim == 1:
        seq_len = arr.shape[0]
        feat_dim = None
    elif arr.ndim == 2:
        seq_len = arr.shape[0]
        feat_dim = arr.shape[1]
    else:
        raise ValueError(f"Unsupported shape for fixed-length enforcement: {arr.shape}")

    if seq_len == target_length:
        return arr

    if seq_len > target_length:
        if mode == "last":
            return arr[-target_length:]
        if mode == "first":
            return arr[:target_length]
        raise ValueError(f"Unsupported mode: {mode}")

    pad_len = target_length - seq_len
    if arr.ndim == 1:
        pad = np.zeros(pad_len, dtype=arr.dtype)
    else:
        pad = np.zeros((pad_len, feat_dim), dtype=arr.dtype)

    return np.concatenate([pad, arr], axis=0)
