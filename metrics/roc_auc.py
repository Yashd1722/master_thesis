# metrics/roc_auc.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _to_numpy_1d(x: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr.reshape(-1)


def binary_labels_from_neutral(
    y_true: Any,
    neutral_label: int = 0,
) -> np.ndarray:
    """
    Convert multiclass labels into binary:
    neutral_label -> 0
    everything else -> 1
    """
    y_true_np = _to_numpy_1d(y_true, dtype=np.int64)
    return (y_true_np != neutral_label).astype(np.int64)


def filter_finite_scores(
    y_true_binary: Any,
    scores: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    y_true_binary_np = _to_numpy_1d(y_true_binary, dtype=np.int64)
    scores_np = _to_numpy_1d(scores, dtype=np.float64)

    if len(y_true_binary_np) != len(scores_np):
        raise ValueError(
            f"Length mismatch between y_true_binary ({len(y_true_binary_np)}) "
            f"and scores ({len(scores_np)})."
        )

    valid_mask = np.isfinite(scores_np)
    return y_true_binary_np[valid_mask], scores_np[valid_mask]


def roc_curve_binary(
    y_true_binary: Any,
    scores: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Manual ROC computation.

    Returns:
        thresholds, fpr, tpr
    """
    y_true_binary_np, scores_np = filter_finite_scores(y_true_binary, scores)

    if len(scores_np) == 0:
        return np.array([]), np.array([]), np.array([])

    unique_thresholds = np.unique(scores_np)[::-1]
    thresholds = np.concatenate(([np.inf], unique_thresholds, [-np.inf]))

    pos = np.sum(y_true_binary_np == 1)
    neg = np.sum(y_true_binary_np == 0)

    fpr_values: List[float] = []
    tpr_values: List[float] = []

    for thr in thresholds:
        y_pred_binary = (scores_np >= thr).astype(np.int64)

        tp = np.sum((y_pred_binary == 1) & (y_true_binary_np == 1))
        fp = np.sum((y_pred_binary == 1) & (y_true_binary_np == 0))

        tpr = tp / pos if pos > 0 else np.nan
        fpr = fp / neg if neg > 0 else np.nan

        tpr_values.append(tpr)
        fpr_values.append(fpr)

    return (
        thresholds.astype(np.float64),
        np.asarray(fpr_values, dtype=np.float64),
        np.asarray(tpr_values, dtype=np.float64),
    )


def auc_from_roc(
    fpr: Any,
    tpr: Any,
) -> float:
    """
    Compute AUC with trapezoidal rule after sorting by FPR.
    """
    fpr_np = _to_numpy_1d(fpr, dtype=np.float64)
    tpr_np = _to_numpy_1d(tpr, dtype=np.float64)

    if len(fpr_np) != len(tpr_np):
        raise ValueError(
            f"Length mismatch between fpr ({len(fpr_np)}) and tpr ({len(tpr_np)})."
        )

    valid_mask = np.isfinite(fpr_np) & np.isfinite(tpr_np)
    fpr_np = fpr_np[valid_mask]
    tpr_np = tpr_np[valid_mask]

    if len(fpr_np) < 2:
        return float("nan")

    order = np.argsort(fpr_np)
    fpr_sorted = fpr_np[order]
    tpr_sorted = tpr_np[order]

    return float(np.trapz(tpr_sorted, fpr_sorted))


def compute_roc_auc_binary(
    y_true_binary: Any,
    scores: Any,
) -> Dict[str, Any]:
    thresholds, fpr, tpr = roc_curve_binary(y_true_binary=y_true_binary, scores=scores)
    auc = auc_from_roc(fpr=fpr, tpr=tpr)

    return {
        "thresholds": thresholds,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
    }


def compute_roc_auc_from_neutral_multiclass(
    y_true: Any,
    scores: Any,
    neutral_label: int = 0,
) -> Dict[str, Any]:
    """
    Convert multiclass truth to binary neutral/non-neutral, then compute ROC/AUC.
    """
    y_true_binary = binary_labels_from_neutral(y_true=y_true, neutral_label=neutral_label)
    return compute_roc_auc_binary(y_true_binary=y_true_binary, scores=scores)


def make_roc_dataframe(
    thresholds: Any,
    fpr: Any,
    tpr: Any,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "threshold": _to_numpy_1d(thresholds, dtype=np.float64),
            "fpr": _to_numpy_1d(fpr, dtype=np.float64),
            "tpr": _to_numpy_1d(tpr, dtype=np.float64),
        }
    )


def save_roc_dataframe(
    thresholds: Any,
    fpr: Any,
    tpr: Any,
    save_path: Path,
) -> pd.DataFrame:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = make_roc_dataframe(thresholds=thresholds, fpr=fpr, tpr=tpr)
    df.to_csv(save_path, index=False)
    return df


def evaluate_multiple_score_columns(
    df: pd.DataFrame,
    y_true_col: str,
    score_cols: List[str],
    neutral_label: int = 0,
) -> pd.DataFrame:
    """
    Convenience helper:
    returns one summary row per score column with AUC.
    """
    if y_true_col not in df.columns:
        raise ValueError(f"Column '{y_true_col}' not found in dataframe.")

    rows: List[Dict[str, Any]] = []

    for score_col in score_cols:
        if score_col not in df.columns:
            continue

        result = compute_roc_auc_from_neutral_multiclass(
            y_true=df[y_true_col].values,
            scores=df[score_col].values,
            neutral_label=neutral_label,
        )

        rows.append(
            {
                "method": score_col,
                "auc": float(result["auc"]) if result["auc"] is not None else np.nan,
                "num_samples": int(np.sum(np.isfinite(_to_numpy_1d(df[score_col].values, dtype=np.float64)))),
            }
        )

    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
