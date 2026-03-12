from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch


# -----------------------------
# Project + path helpers
# -----------------------------
def project_root() -> Path:
    """Return repository root."""
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(value: str) -> str:
    """Convert text to a filesystem-safe name."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    return value.strip("_")


def build_run_name(model_name: str, dataset_name: str, metric_name: str) -> str:
    """Build standard run name: model_dataset_metric"""
    if not model_name or not dataset_name or not metric_name:
        raise ValueError("model_name, dataset_name, and metric_name must be non-empty.")
    return f"{safe_name(model_name)}_{safe_name(dataset_name)}_{safe_name(metric_name)}"


def get_test_paths(run_name: str) -> dict[str, Path]:
    """Return all standard output paths for one test run."""
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
    """Return checkpoint path from training naming convention."""
    ckpt = project_root() / "checkpoints" / f"{build_run_name(model_name, dataset_name, metric_name)}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


# -----------------------------
# Save/load helpers
# -----------------------------
def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def save_json(data: dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_checkpoint_state(checkpoint_path: Path, device: torch.device) -> dict:
    """Load PyTorch checkpoint safely."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint at {checkpoint_path} is not a valid state_dict.")

    return state


# -----------------------------
# Logger
# -----------------------------
def setup_logger(log_path: Path, logger_name: str | None = None) -> logging.Logger:
    """Create file + console logger."""
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

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


# -----------------------------
# Class mapping
# -----------------------------
CLASS_NAMES = {
    0: "fold",
    1: "hopf",
    2: "transcritical",
    3: "null",
}


def class_index_to_name(idx: int) -> str:
    """Convert class index to class name."""
    if idx not in CLASS_NAMES:
        raise ValueError(f"Unknown class index: {idx}")
    return CLASS_NAMES[idx]


# -----------------------------
# Probability helpers
# -----------------------------
def validate_probability_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing probability columns: {missing}")


def add_transition_probability(
    df: pd.DataFrame,
    fold_col: str = "p_fold",
    hopf_col: str = "p_hopf",
    transcritical_col: str = "p_transcritical",
    out_col: str = "p_transition",
) -> pd.DataFrame:
    """
    Add binary transition probability:
    p_transition = p_fold + p_hopf + p_transcritical
    """
    validate_probability_columns(df, [fold_col, hopf_col, transcritical_col])

    out = df.copy()
    out[out_col] = out[fold_col] + out[hopf_col] + out[transcritical_col]
    return out
