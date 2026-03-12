# testing/compare_dl_vs_csd.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from testing.testing_utils import (
    build_run_name,
    get_test_paths,
    save_csv,
    setup_logger,
)
from metrics.roc_auc import compute_binary_roc_auc
from metrics.confusion_matrix import compute_binary_confusion_matrix
from metrics import METRICS


def _load_required_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{name} file is empty: {path}")
    return df


def _prepare_binary_labels(df: pd.DataFrame, label_col: str | None) -> pd.DataFrame:
    """
    If label_col exists, convert to binary:
      1 = transition-like (fold/hopf/transcritical or numeric 1)
      0 = null/stable (null or numeric 0)
    """
    out = df.copy()

    if label_col is None:
        return out

    if label_col not in out.columns:
        raise KeyError(f"Requested label column '{label_col}' not found in merged dataframe.")

    def to_binary(v):
        if pd.isna(v):
            return None
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"fold", "hopf", "transcritical", "transition", "positive", "1", "true", "yes"}:
                return 1
            if s in {"null", "stable", "negative", "0", "false", "no"}:
                return 0
            raise ValueError(f"Unsupported string label value: {v}")
        if int(v) in {0, 1}:
            return int(v)
        raise ValueError(f"Unsupported numeric label value: {v}")

    out["binary_label"] = out[label_col].apply(to_binary)
    if out["binary_label"].isna().any():
        raise ValueError("binary_label contains NaN after conversion.")
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare DL transition probability vs CSD scores.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument(
        "--label_col",
        default=None,
        help="Optional label column for binary evaluation. If absent, only merged outputs are saved.",
    )
    parser.add_argument(
        "--decision_threshold",
        type=float,
        default=0.5,
        help="Threshold for confusion matrix / threshold metrics on DL p_transition.",
    )
    args = parser.parse_args()

    run_name = build_run_name(args.model, args.dataset, args.metric)
    paths = get_test_paths(run_name)
    logger = setup_logger(paths["compare_log"], logger_name=f"testing.compare.{run_name}")

    logger.info(f"Run name: {run_name}")

    dl_path = paths["dl_dir"] / "per_series_predictions.csv"
    csd_path = paths["csd_dir"] / "csd_scores.csv"

    dl_df = _load_required_csv(dl_path, "DL predictions")
    csd_df = _load_required_csv(csd_path, "CSD scores")

    if "file_name" not in dl_df.columns or "file_name" not in csd_df.columns:
        raise KeyError("Both DL and CSD files must contain 'file_name'.")

    merged = pd.merge(dl_df, csd_df, on=["file_name", "dataset_name", "model_name", "metric_name"], how="inner")
    if merged.empty:
        raise ValueError("Merged DL vs CSD dataframe is empty. Check file_name alignment.")

    save_csv(merged, paths["cmp_dir"] / "dl_vs_csd_merged.csv")
    logger.info(f"Saved: {paths['cmp_dir'] / 'dl_vs_csd_merged.csv'}")

    # If no labels exist, stop after merge.
    if args.label_col is None:
        logger.info("No label_col provided. Skipping ROC/AUC/confusion-matrix computation.")
        logger.info("Done.")
        return

    merged = _prepare_binary_labels(merged, args.label_col)

    # ROC / AUC
    roc_rows = []
    auc_rows = []

    score_specs = [
        ("dl_p_transition", "p_transition"),
        ("csd_ktau_var", "ktau_var"),
        ("csd_ktau_ac1", "ktau_ac1"),
    ]

    for score_name, score_col in score_specs:
        if score_col not in merged.columns:
            logger.warning(f"Skipping missing score column: {score_col}")
            continue

        roc_df, auc_value = compute_binary_roc_auc(
            y_true=merged["binary_label"].tolist(),
            y_score=merged[score_col].tolist(),
            score_name=score_name,
        )
        roc_rows.append(roc_df)
        auc_rows.append({"score_name": score_name, "auc": auc_value})

    roc_results = pd.concat(roc_rows, ignore_index=True) if roc_rows else pd.DataFrame()
    auc_results = pd.DataFrame(auc_rows)

    save_csv(roc_results, paths["cmp_dir"] / "roc_results.csv")
    save_csv(auc_results, paths["cmp_dir"] / "auc_results.csv")

    # Confusion matrix + threshold metrics for DL only
    if "p_transition" not in merged.columns:
        raise KeyError("Merged dataframe is missing 'p_transition' required for threshold metrics.")

    y_true = merged["binary_label"].tolist()
    y_pred = (merged["p_transition"] >= args.decision_threshold).astype(int).tolist()

    cm_df = compute_binary_confusion_matrix(y_true=y_true, y_pred=y_pred)
    save_csv(cm_df, paths["cmp_dir"] / "confusion_matrix.csv")

    threshold_metrics = []
    for metric_name, metric_fn in METRICS.items():
        try:
            value = metric_fn(y_true, y_pred, 2) if metric_name != "acc" else sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
            threshold_metrics.append({"metric_name": metric_name, "value": value})
        except Exception:
            continue

    threshold_metrics_df = pd.DataFrame(threshold_metrics)
    save_csv(threshold_metrics_df, paths["cmp_dir"] / "threshold_metrics.csv")

    best_auc_row = auc_results.sort_values("auc", ascending=False).head(1) if not auc_results.empty else pd.DataFrame()
    final_summary = pd.DataFrame(
        [
            {
                "dataset_name": args.dataset,
                "model_name": args.model,
                "metric_name": args.metric,
                "num_samples": len(merged),
                "decision_threshold": args.decision_threshold,
                "best_auc_score_name": None if best_auc_row.empty else best_auc_row.iloc[0]["score_name"],
                "best_auc": None if best_auc_row.empty else best_auc_row.iloc[0]["auc"],
            }
        ]
    )
    save_csv(final_summary, paths["cmp_dir"] / "final_comparison_summary.csv")

    logger.info(f"Saved: {paths['cmp_dir'] / 'roc_results.csv'}")
    logger.info(f"Saved: {paths['cmp_dir'] / 'auc_results.csv'}")
    logger.info(f"Saved: {paths['cmp_dir'] / 'confusion_matrix.csv'}")
    logger.info(f"Saved: {paths['cmp_dir'] / 'threshold_metrics.csv'}")
    logger.info(f"Saved: {paths['cmp_dir'] / 'final_comparison_summary.csv'}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
