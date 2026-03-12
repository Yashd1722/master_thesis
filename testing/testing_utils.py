# testing/testing_utils.py

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import torch


def project_root() -> Path:
    """Return repository root from testing/ folder."""
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> Path:
    """Create directory if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(value: str) -> str:
    """Convert a string to a filesystem-safe token."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    return value.strip("_")


def build_run_name(model_name: str, dataset_name: str, metric_name: str) -> str:
    """Build standard run name: model_dataset_metric."""
    if not model_name or not dataset_name or not metric_name:
        raise ValueError("model_name, dataset_name, and metric_name must all be non-empty.")
    return f"{safe_name(model_name)}_{safe_name(dataset_name)}_{safe_name(metric_name)}"


def setup_logger(log_path: Path, logger_name: Optional[str] = None) -> logging.Logger:
    """
    Create a file + console logger.

    Uses a dedicated logger name so multiple testing scripts do not clash.
    """
    ensure_dir(log_path.parent)

    if logger_name is None:
        logger_name = f"testing.{log_path.stem}"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers if script is re-run in same process.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def get_test_paths(run_name: str) -> dict[str, Path]:
    """
    Return standard paths for one testing run.
    """
    root = project_root()

    test_logs = ensure_dir(root / "test_logs")
    dl_dir = ensure_dir(root / "test_results" / "dl" / run_name)
    csd_dir = ensure_dir(root / "test_results" / "csd" / run_name)
    cmp_dir = ensure_dir(root / "test_results" / "comparison" / run_name)

    return {
        "root": root,
        "test_logs": test_logs,
        "dl_dir": dl_dir,
        "csd_dir": csd_dir,
        "cmp_dir": cmp_dir,
        "dl_log": test_logs / f"{run_name}_dl.log",
        "csd_log": test_logs / f"{run_name}_csd.log",
        "compare_log": test_logs / f"{run_name}_compare.log",
    }


def checkpoint_path_for(model_name: str, dataset_name: str, metric_name: str) -> Path:
    """
    Resolve the expected checkpoint path from the training naming convention.
    Example: checkpoints/lstm_ts_500_f1_macro.pt
    """
    root = project_root()
    ckpt = root / "checkpoints" / f"{build_run_name(model_name, dataset_name, metric_name)}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            f"Expected training checkpoint name format: model_dataset_metric.pt"
        )
    return ckpt


def list_checkpoints(
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    metric_name: Optional[str] = None,
) -> list[Path]:
    """
    List checkpoints with optional filtering by model, dataset, metric.
    """
    ckpt_dir = project_root() / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    files = sorted(ckpt_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No checkpoint files found in: {ckpt_dir}")

    def _match(p: Path) -> bool:
        stem = p.stem.lower()
        if model_name and not stem.startswith(safe_name(model_name) + "_"):
            return False
        if dataset_name and f"_{safe_name(dataset_name)}_" not in f"_{stem}_":
            return False
        if metric_name and not stem.endswith("_" + safe_name(metric_name)):
            return False
        return True

    matched = [p for p in files if _match(p)]
    if not matched:
        raise FileNotFoundError(
            "No matching checkpoints found for the requested filters: "
            f"model={model_name}, dataset={dataset_name}, metric={metric_name}"
        )
    return matched


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Safely save a dataframe to CSV."""
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def save_json(data: dict, path: Path) -> None:
    """Safely save a JSON file."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_probability_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> None:
    """Validate that probability columns exist."""
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
    """
    Add binary transition probability = fold + hopf + transcritical.
    """
    validate_probability_columns(df, [fold_col, hopf_col, transcritical_col])
    out = df.copy()
    out[out_col] = out[fold_col] + out[hopf_col] + out[transcritical_col]
    return out


def load_checkpoint_state(checkpoint_path: Path, device: torch.device) -> dict:
    """Load checkpoint state dict with explicit error handling."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    try:
        state = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint: {checkpoint_path}") from exc

    if not isinstance(state, dict):
        raise TypeError(
            f"Checkpoint at {checkpoint_path} did not load as a state_dict-like dict."
        )
    return state


def class_index_to_name(idx: int) -> str:
    """
    Map model output index to class name.
    Assumes 4-class setup:
    0=fold, 1=hopf, 2=transcritical, 3=null
    """
    mapping = {
        0: "fold",
        1: "hopf",
        2: "transcritical",
        3: "null",
    }
    if idx not in mapping:
        raise ValueError(f"Unknown class index: {idx}")
    return mapping[idx]
