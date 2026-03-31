# metrics/evaluation.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


CLASS_NAMES = ["fold", "hopf", "transcritical", "null"]


def _to_numpy_int(x: Iterable[int]) -> np.ndarray:
    return np.asarray(list(x), dtype=np.int64)


def _to_numpy_float_2d(x: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D probability array, got shape={arr.shape}")
    return arr


def save_json(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_confusion_matrix_csv(cm: List[List[int]], out_path: Path, class_names: Optional[List[str]] = None) -> None:
    class_names = class_names or CLASS_NAMES
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    df.to_csv(out_path)


def save_roc_curve_csv(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "fpr": np.asarray(fpr, dtype=np.float64),
            "tpr": np.asarray(tpr, dtype=np.float64),
            "threshold": np.asarray(thresholds, dtype=np.float64),
        }
    )
    df.to_csv(out_path, index=False)


def compute_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    class_names: Optional[List[str]] = None,
) -> Dict:
    class_names = class_names or CLASS_NAMES
    y_true = _to_numpy_int(y_true)
    y_pred = _to_numpy_int(y_pred)

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def compute_multiclass_roc_auc(
    y_true: Iterable[int],
    y_prob: Iterable[Iterable[float]],
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Multiclass ROC-AUC for DL outputs.
    One-vs-rest style, matching the usual comparison approach.
    """
    class_names = class_names or CLASS_NAMES
    y_true = _to_numpy_int(y_true)
    y_prob = _to_numpy_float_2d(y_prob)

    n_classes = len(class_names)
    if y_prob.shape[1] != n_classes:
        raise ValueError(
            f"Expected probability matrix second dimension = {n_classes}, got {y_prob.shape[1]}"
        )

    y_true_onehot = np.eye(n_classes, dtype=np.float64)[y_true]

    result = {
        "roc_auc_macro_ovr": None,
        "roc_auc_weighted_ovr": None,
        "per_class": {},
    }

    try:
        result["roc_auc_macro_ovr"] = float(
            roc_auc_score(y_true_onehot, y_prob, average="macro", multi_class="ovr")
        )
    except Exception:
        result["roc_auc_macro_ovr"] = None

    try:
        result["roc_auc_weighted_ovr"] = float(
            roc_auc_score(y_true_onehot, y_prob, average="weighted", multi_class="ovr")
        )
    except Exception:
        result["roc_auc_weighted_ovr"] = None

    for i, class_name in enumerate(class_names):
        y_bin = y_true_onehot[:, i]
        scores = y_prob[:, i]

        if len(np.unique(y_bin)) < 2:
            result["per_class"][class_name] = {
                "auc": None,
                "fpr": [],
                "tpr": [],
                "thresholds": [],
            }
            continue

        fpr, tpr, thresholds = roc_curve(y_bin, scores)
        try:
            auc_val = float(roc_auc_score(y_bin, scores))
        except Exception:
            auc_val = None

        result["per_class"][class_name] = {
            "auc": auc_val,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }

    return result


def make_binary_transition_labels(y_true: Iterable[int], null_class_idx: int = 3) -> np.ndarray:
    """
    Converts 4-class labels into binary:
    1 = transition class (fold/hopf/transcritical)
    0 = null
    """
    y_true = _to_numpy_int(y_true)
    return (y_true != int(null_class_idx)).astype(np.int64)


def compute_binary_roc(
    y_true_binary: Iterable[int],
    scores: Iterable[float],
) -> Dict:
    """
    Generic binary ROC/AUC helper for:
    - DL transition score
    - CSD ktau_var
    - CSD ktau_ac1
    """
    y_true_binary = _to_numpy_int(y_true_binary)
    scores = np.asarray(list(scores), dtype=np.float64)

    valid_mask = np.isfinite(scores)
    y_true_binary = y_true_binary[valid_mask]
    scores = scores[valid_mask]

    if len(y_true_binary) == 0:
        return {
            "auc": None,
            "fpr": [],
            "tpr": [],
            "thresholds": [],
            "n_used": 0,
        }

    if len(np.unique(y_true_binary)) < 2:
        return {
            "auc": None,
            "fpr": [],
            "tpr": [],
            "thresholds": [],
            "n_used": int(len(y_true_binary)),
        }

    fpr, tpr, thresholds = roc_curve(y_true_binary, scores)
    auc_val = float(roc_auc_score(y_true_binary, scores))

    return {
        "auc": auc_val,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "n_used": int(len(y_true_binary)),
    }


def compare_dl_vs_csd_binary_roc(
    merged_df: pd.DataFrame,
    true_label_col: str = "true_class_idx",
    dl_transition_col: str = "p_transition",
    csd_var_col: str = "ktau_var",
    csd_ac1_col: str = "ktau_ac1",
    null_class_idx: int = 3,
) -> Dict:
    """
    Main DL vs CSD comparison.

    Required columns:
    - true_class_idx
    - p_transition
    - ktau_var
    - ktau_ac1
    """
    required_cols = [true_label_col, dl_transition_col, csd_var_col, csd_ac1_col]
    missing = [c for c in required_cols if c not in merged_df.columns]
    if missing:
        raise KeyError(f"Missing required columns for DL vs CSD comparison: {missing}")

    eval_df = merged_df[required_cols].copy()
    eval_df = eval_df.dropna(subset=[true_label_col])

    y_true = eval_df[true_label_col].astype(int).to_numpy()
    y_true_binary = make_binary_transition_labels(y_true, null_class_idx=null_class_idx)

    dl_result = compute_binary_roc(y_true_binary, eval_df[dl_transition_col].to_numpy(dtype=np.float64))
    var_result = compute_binary_roc(y_true_binary, eval_df[csd_var_col].to_numpy(dtype=np.float64))
    ac1_result = compute_binary_roc(y_true_binary, eval_df[csd_ac1_col].to_numpy(dtype=np.float64))

    return {
        "n_samples": int(len(eval_df)),
        "dl_transition": dl_result,
        "csd_ktau_var": var_result,
        "csd_ktau_ac1": ac1_result,
    }


def plot_binary_roc_comparison(
    comparison_result: Dict,
    out_path: Path,
    title: str = "DL vs CSD ROC comparison",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))

    def _plot_one(result_key: str, label: str) -> None:
        payload = comparison_result.get(result_key, {})
        fpr = payload.get("fpr", [])
        tpr = payload.get("tpr", [])
        auc_val = payload.get("auc", None)

        if len(fpr) == 0 or len(tpr) == 0:
            return

        label_text = label if auc_val is None else f"{label} (AUC={auc_val:.3f})"
        plt.plot(fpr, tpr, linewidth=2, label=label_text)

    _plot_one("dl_transition", "DL transition")
    _plot_one("csd_ktau_var", "CSD Kendall tau variance")
    _plot_one("csd_ktau_ac1", "CSD Kendall tau AC(1)")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_multiclass_roc_per_class(
    roc_payload: Dict,
    out_dir: Path,
) -> None:
    """
    Saves one ROC plot per class for DL multiclass one-vs-rest.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    per_class = roc_payload.get("per_class", {})
    for class_name, payload in per_class.items():
        fpr = payload.get("fpr", [])
        tpr = payload.get("tpr", [])
        auc_val = payload.get("auc", None)

        if len(fpr) == 0 or len(tpr) == 0:
            continue

        plt.figure(figsize=(6, 5))
        label = class_name if auc_val is None else f"{class_name} (AUC={auc_val:.3f})"
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curve: {class_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"roc_curve_{class_name}.png", dpi=200)
        plt.close()
