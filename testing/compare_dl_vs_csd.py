from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import METRICS
from metrics.confusion_matrix import compute_binary_confusion_matrix
from metrics.roc_auc import compute_binary_roc_auc
from testing.testing_utils import (
    build_run_name,
    get_test_paths,
    save_csv,
    setup_logger,
)


def _load_required_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{name} file is empty: {path}")
    return df


def _safe_metric(metric_name: str, y_true, y_pred):
    try:
        metric_fn = METRICS[metric_name]
        if metric_name == "acc":
            return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
        return metric_fn(y_true, y_pred, 2)
    except Exception:
        return None


def _compute_threshold_metrics(
    y_true: list[int],
    y_score: list[float],
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_pred = (pd.Series(y_score) >= threshold).astype(int).tolist()
    cm_df = compute_binary_confusion_matrix(y_true=y_true, y_pred=y_pred)

    rows = []
    for metric_name in METRICS.keys():
        value = _safe_metric(metric_name, y_true, y_pred)
        if value is not None:
            rows.append(
                {
                    "metric_name": metric_name,
                    "value": value,
                    "decision_threshold": threshold,
                }
            )

    return cm_df, pd.DataFrame(rows)


def _prepare_observed_merge(dl_df: pd.DataFrame, csd_df: pd.DataFrame) -> pd.DataFrame:
    required_dl = {
        "file_name",
        "dataset_name",
        "model_name",
        "metric_name",
        "p_transition",
        "p_fold",
        "p_hopf",
        "p_transcritical",
        "p_null",
        "predicted_class",
    }
    required_csd = {
        "file_name",
        "dataset_name",
        "model_name",
        "metric_name",
        "ktau_var",
        "ktau_ac1",
    }

    missing_dl = required_dl - set(dl_df.columns)
    missing_csd = required_csd - set(csd_df.columns)

    if missing_dl:
        raise KeyError(f"DL predictions missing columns: {sorted(missing_dl)}")
    if missing_csd:
        raise KeyError(f"CSD scores missing columns: {sorted(missing_csd)}")

    merged = pd.merge(
        dl_df,
        csd_df,
        on=["file_name", "dataset_name", "model_name", "metric_name"],
        how="inner",
        suffixes=("_dl", "_csd"),
    )
    if merged.empty:
        raise ValueError("Merged observed DL vs CSD dataframe is empty. Check file_name alignment.")

    return merged


def _make_observed_summary(observed_merged: pd.DataFrame) -> pd.DataFrame:
    predicted_class_counts = (
        observed_merged["predicted_class"]
        .value_counts(dropna=False)
        .rename_axis("predicted_class")
        .reset_index(name="count")
    )

    summary_rows = [
        {
            "summary_key": "num_rows",
            "summary_value": int(len(observed_merged)),
        },
        {
            "summary_key": "mean_p_transition",
            "summary_value": float(observed_merged["p_transition"].mean()),
        },
        {
            "summary_key": "mean_ktau_var",
            "summary_value": float(observed_merged["ktau_var"].mean()),
        },
        {
            "summary_key": "mean_ktau_ac1",
            "summary_value": float(observed_merged["ktau_ac1"].mean()),
        },
    ]

    for _, row in predicted_class_counts.iterrows():
        summary_rows.append(
            {
                "summary_key": f"predicted_class_count_{row['predicted_class']}",
                "summary_value": int(row["count"]),
            }
        )

    return pd.DataFrame(summary_rows)


def _compare_observed_vs_null(
    observed_merged: pd.DataFrame,
    null_csd_df: pd.DataFrame,
    decision_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_null = {"file_name", "dataset_name", "ktau_var", "ktau_ac1"}
    missing_null = required_null - set(null_csd_df.columns)
    if missing_null:
        raise KeyError(f"Null CSD scores missing columns: {sorted(missing_null)}")

    observed_eval = observed_merged[
        [
            "file_name",
            "dataset_name",
            "model_name",
            "metric_name",
            "p_transition",
            "ktau_var",
            "ktau_ac1",
        ]
    ].copy()
    observed_eval["binary_label"] = 1
    observed_eval["series_source"] = "observed"

    null_agg = (
        null_csd_df.groupby(["file_name", "dataset_name"], as_index=False)[["ktau_var", "ktau_ac1"]]
        .mean()
    )
    null_agg["binary_label"] = 0
    null_agg["series_source"] = "null"
    null_agg["model_name"] = observed_merged["model_name"].iloc[0]
    null_agg["metric_name"] = observed_merged["metric_name"].iloc[0]
    null_agg["p_transition"] = np.nan

    combined = pd.concat(
        [
            observed_eval,
            null_agg[
                [
                    "file_name",
                    "dataset_name",
                    "model_name",
                    "metric_name",
                    "p_transition",
                    "ktau_var",
                    "ktau_ac1",
                    "binary_label",
                    "series_source",
                ]
            ],
        ],
        ignore_index=True,
        sort=False,
    )

    roc_rows = []
    auc_rows = []

    for score_name, score_col in [
        ("dl_p_transition", "p_transition"),
        ("csd_ktau_var", "ktau_var"),
        ("csd_ktau_ac1", "ktau_ac1"),
    ]:
        score_df = combined.dropna(subset=[score_col]).copy()
        if score_df.empty:
            continue
        if score_df["binary_label"].nunique() < 2:
            continue

        roc_df, auc_value = compute_binary_roc_auc(
            y_true=score_df["binary_label"].tolist(),
            y_score=score_df[score_col].tolist(),
            score_name=score_name,
        )
        roc_rows.append(roc_df)
        auc_rows.append(
            {
                "score_name": score_name,
                "auc": auc_value,
            }
        )

    roc_df = pd.concat(roc_rows, ignore_index=True) if roc_rows else pd.DataFrame()
    auc_df = pd.DataFrame(auc_rows)

    threshold_metrics_df = pd.DataFrame()
    confusion_df = pd.DataFrame()

    dl_threshold_df = combined.dropna(subset=["p_transition"]).copy()
    if not dl_threshold_df.empty and dl_threshold_df["binary_label"].nunique() == 2:
        confusion_df, threshold_metrics_df = _compute_threshold_metrics(
            y_true=dl_threshold_df["binary_label"].tolist(),
            y_score=dl_threshold_df["p_transition"].tolist(),
            threshold=decision_threshold,
        )

    summary_rows = [
        {
            "summary_key": "num_rows",
            "summary_value": int(len(combined)),
        },
        {
            "summary_key": "num_observed_rows",
            "summary_value": int((combined["series_source"] == "observed").sum()),
        },
        {
            "summary_key": "num_null_rows",
            "summary_value": int((combined["series_source"] == "null").sum()),
        },
    ]

    if not auc_df.empty:
        best_row = auc_df.sort_values("auc", ascending=False).iloc[0]
        summary_rows.append(
            {
                "summary_key": "best_auc_score_name",
                "summary_value": best_row["score_name"],
            }
        )
        summary_rows.append(
            {
                "summary_key": "best_auc_value",
                "summary_value": float(best_row["auc"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    return combined, roc_df, auc_df, confusion_df, threshold_metrics_df, summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Merge DL and CSD outputs and compare observed vs optional null-model results."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--metric", required=True, help="Metric name")
    parser.add_argument(
        "--use_null",
        action="store_true",
        help="Use null CSD outputs if available and compute observed-vs-null comparison.",
    )
    parser.add_argument(
        "--decision_threshold",
        type=float,
        default=0.5,
        help="Threshold for binary decision metrics on p_transition.",
    )
    args = parser.parse_args()

    run_name = build_run_name(args.model, args.dataset, args.metric)
    paths = get_test_paths(run_name)
    logger = setup_logger(paths["compare_log"], logger_name=f"testing.compare.{run_name}")

    logger.info("Run name: %s", run_name)
    logger.info("Dataset: %s | Model: %s | Metric: %s", args.dataset, args.model, args.metric)
    logger.info("Use null: %s", args.use_null)

    dl_path = paths["dl_dir"] / "per_series_predictions.csv"
    csd_path = paths["csd_dir"] / "csd_scores.csv"

    dl_df = _load_required_csv(dl_path, "DL predictions")
    csd_df = _load_required_csv(csd_path, "Observed CSD scores")

    observed_merged = _prepare_observed_merge(dl_df, csd_df)
    observed_summary = _make_observed_summary(observed_merged)

    save_csv(observed_merged, paths["cmp_dir"] / "dl_vs_csd_observed_merged.csv")
    save_csv(observed_summary, paths["cmp_dir"] / "dl_vs_csd_observed_summary.csv")

    logger.info("Saved observed merged comparison outputs.")

    if args.use_null:
        null_scores_path = paths["null_scores_dir"] / "null_csd_scores.csv"
        null_csd_df = _load_required_csv(null_scores_path, "Null CSD scores")

        (
            combined_df,
            roc_df,
            auc_df,
            confusion_df,
            threshold_metrics_df,
            null_summary_df,
        ) = _compare_observed_vs_null(
            observed_merged=observed_merged,
            null_csd_df=null_csd_df,
            decision_threshold=args.decision_threshold,
        )

        save_csv(combined_df, paths["cmp_dir"] / "observed_vs_null_eval_table.csv")
        save_csv(roc_df, paths["cmp_dir"] / "roc_results_observed_vs_null.csv")
        save_csv(auc_df, paths["cmp_dir"] / "auc_results_observed_vs_null.csv")
        save_csv(confusion_df, paths["cmp_dir"] / "confusion_matrix_observed_vs_null_dl.csv")
        save_csv(
            threshold_metrics_df,
            paths["cmp_dir"] / "threshold_metrics_observed_vs_null_dl.csv",
        )
        save_csv(null_summary_df, paths["cmp_dir"] / "final_summary_observed_vs_null.csv")

        logger.info("Saved observed-vs-null comparison outputs.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
