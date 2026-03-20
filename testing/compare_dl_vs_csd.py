# testing/compare_dl_vs_csd.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.evaluation import (
    compare_dl_vs_csd_binary_roc,
    plot_binary_roc_comparison,
    save_json,
    save_roc_curve_csv,
)
from testing.testing_utils import build_run_name, get_test_paths, save_csv, setup_logger


def build_summary_row(args, merged_df: pd.DataFrame, roc_result: dict) -> dict:
    dl_auc = roc_result.get("dl_transition", {}).get("auc")
    csd_var_auc = roc_result.get("csd_ktau_var", {}).get("auc")
    csd_ac1_auc = roc_result.get("csd_ktau_ac1", {}).get("auc")

    row = {
        "dataset_name": args.dataset,
        "train_dataset_name": args.train_dataset,
        "model_name": args.model,
        "metric_name": args.metric,
        "experiment": args.experiment,
        "num_matched_files": int(len(merged_df)),
        "dl_transition_auc": None if dl_auc is None else float(dl_auc),
        "csd_ktau_var_auc": None if csd_var_auc is None else float(csd_var_auc),
        "csd_ktau_ac1_auc": None if csd_ac1_auc is None else float(csd_ac1_auc),
    }

    if "p_transition" in merged_df.columns:
        row["mean_p_transition"] = float(merged_df["p_transition"].mean())

    if "ktau_var" in merged_df.columns:
        row["mean_ktau_var"] = float(merged_df["ktau_var"].mean())

    if "ktau_ac1" in merged_df.columns:
        row["mean_ktau_ac1"] = float(merged_df["ktau_ac1"].mean())

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DL and CSD outputs")
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--train_dataset", required=True, type=str, choices=["ts_500", "ts_1500"])
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--metric", required=True, type=str)
    parser.add_argument(
        "--experiment",
        type=str,
        default="base",
        choices=["base", "trend", "season", "trend_season"],
    )
    args = parser.parse_args()

    run_name = build_run_name(
        model=args.model,
        train_dataset=args.train_dataset,
        metric=args.metric,
        experiment=args.experiment,
        test_dataset=args.dataset,
    )
    paths = get_test_paths(run_name)
    logger = setup_logger(paths["compare_log"], logger_name=f"testing.compare.{run_name}")

    dl_path = paths["dl_dir"] / "prediction_summary.csv"
    csd_path = paths["csd_dir"] / "csd_scores.csv"

    logger.info("Run name: %s", run_name)
    logger.info("DL file: %s", dl_path)
    logger.info("CSD file: %s", csd_path)

    if not dl_path.exists():
        raise FileNotFoundError(f"Missing DL predictions file: {dl_path}")

    if not csd_path.exists():
        raise FileNotFoundError(f"Missing CSD scores file: {csd_path}")

    dl_df = pd.read_csv(dl_path)
    csd_df = pd.read_csv(csd_path)

    if dl_df.empty:
        raise RuntimeError("DL prediction file is empty.")
    if csd_df.empty:
        raise RuntimeError("CSD score file is empty.")

    if "file_name" not in dl_df.columns:
        raise KeyError("DL file must contain 'file_name' column.")
    if "file_name" not in csd_df.columns:
        raise KeyError("CSD file must contain 'file_name' column.")

    keep_dl_cols = [
        col for col in [
            "file_name",
            "dataset_name",
            "train_dataset_name",
            "model_name",
            "metric_name",
            "experiment",
            "true_class_idx",
            "p_fold",
            "p_hopf",
            "p_transcritical",
            "p_null",
            "p_transition",
            "predicted_class",
        ]
        if col in dl_df.columns
    ]

    keep_csd_cols = [
        col for col in [
            "file_name",
            "sequence_length",
            "window_size",
            "n_windows",
            "mean_var",
            "mean_ac1",
            "ktau_var",
            "ktau_ac1",
            "null_mean_ktau_var",
            "null_mean_ktau_ac1",
            "pvalue_ktau_var",
            "pvalue_ktau_ac1",
        ]
        if col in csd_df.columns
    ]

    dl_df = dl_df[keep_dl_cols].copy()
    csd_df = csd_df[keep_csd_cols].copy()

    merged_df = dl_df.merge(csd_df, on="file_name", how="inner")

    if merged_df.empty:
        raise RuntimeError("Merged DL vs CSD dataframe is empty.")

    save_csv(merged_df, paths["compare_dir"] / "dl_vs_csd_merged.csv")
    logger.info("Saved merged comparison file.")

    metrics_payload = {
        "run_name": run_name,
        "dataset": args.dataset,
        "train_dataset": args.train_dataset,
        "model": args.model,
        "metric": args.metric,
        "experiment": args.experiment,
        "num_matched_files": int(len(merged_df)),
    }

    if "true_class_idx" in merged_df.columns and merged_df["true_class_idx"].notna().any():
        roc_result = compare_dl_vs_csd_binary_roc(
            merged_df=merged_df,
            true_label_col="true_class_idx",
            dl_transition_col="p_transition",
            csd_var_col="ktau_var",
            csd_ac1_col="ktau_ac1",
            null_class_idx=3,
        )
        metrics_payload["binary_transition_roc"] = roc_result

        dl_payload = roc_result.get("dl_transition", {})
        var_payload = roc_result.get("csd_ktau_var", {})
        ac1_payload = roc_result.get("csd_ktau_ac1", {})

        if len(dl_payload.get("fpr", [])) > 0:
            save_roc_curve_csv(
                np.asarray(dl_payload["fpr"], dtype=np.float64),
                np.asarray(dl_payload["tpr"], dtype=np.float64),
                np.asarray(dl_payload["thresholds"], dtype=np.float64),
                paths["compare_dir"] / "roc_curve_dl_transition.csv",
            )

        if len(var_payload.get("fpr", [])) > 0:
            save_roc_curve_csv(
                np.asarray(var_payload["fpr"], dtype=np.float64),
                np.asarray(var_payload["tpr"], dtype=np.float64),
                np.asarray(var_payload["thresholds"], dtype=np.float64),
                paths["compare_dir"] / "roc_curve_csd_ktau_var.csv",
            )

        if len(ac1_payload.get("fpr", [])) > 0:
            save_roc_curve_csv(
                np.asarray(ac1_payload["fpr"], dtype=np.float64),
                np.asarray(ac1_payload["tpr"], dtype=np.float64),
                np.asarray(ac1_payload["thresholds"], dtype=np.float64),
                paths["compare_dir"] / "roc_curve_csd_ktau_ac1.csv",
            )

        plot_binary_roc_comparison(
            comparison_result=roc_result,
            out_path=paths["compare_dir"] / "dl_vs_csd_roc.png",
            title=f"DL vs CSD ROC comparison ({run_name})",
        )

        summary_row = build_summary_row(args, merged_df, roc_result)
        save_csv(pd.DataFrame([summary_row]), paths["compare_dir"] / "dl_vs_csd_summary.csv")

        logger.info(
            "AUCs | DL transition=%s | CSD var=%s | CSD ac1=%s",
            str(roc_result.get("dl_transition", {}).get("auc")),
            str(roc_result.get("csd_ktau_var", {}).get("auc")),
            str(roc_result.get("csd_ktau_ac1", {}).get("auc")),
        )
    else:
        summary_row = {
            "dataset_name": args.dataset,
            "train_dataset_name": args.train_dataset,
            "model_name": args.model,
            "metric_name": args.metric,
            "experiment": args.experiment,
            "num_matched_files": int(len(merged_df)),
            "dl_transition_auc": None,
            "csd_ktau_var_auc": None,
            "csd_ktau_ac1_auc": None,
        }
        save_csv(pd.DataFrame([summary_row]), paths["compare_dir"] / "dl_vs_csd_summary.csv")
        logger.info("No usable true labels found. Saved merged file and summary without ROC/AUC.")

    save_json(metrics_payload, paths["compare_dir"] / "metrics.json")
    logger.info("Saved comparison outputs to %s", paths["compare_dir"])
    logger.info("Done.")


if __name__ == "__main__":
    main()
