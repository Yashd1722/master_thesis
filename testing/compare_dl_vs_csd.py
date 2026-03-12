from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from testing.testing_utils import (
    build_run_name,
    class_index_to_name,
    get_test_paths,
    save_csv,
    setup_logger,
)


def load_csv_required(path: Path, file_label: str) -> pd.DataFrame:
    """
    Load a CSV file safely.

    keep_default_na=False is important so that the string 'null'
    is not converted into NaN by pandas.
    """
    if not path.exists():
        raise FileNotFoundError(f"{file_label} not found: {path}")

    df = pd.read_csv(path, keep_default_na=False)

    if df.empty:
        raise ValueError(f"{file_label} is empty: {path}")

    return df


def ensure_predicted_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure predicted_class exists and matches predicted_class_idx.
    """
    out = df.copy()

    if "predicted_class_idx" not in out.columns:
        raise ValueError("DL file must contain 'predicted_class_idx'.")

    out["predicted_class_idx"] = pd.to_numeric(
        out["predicted_class_idx"], errors="raise"
    ).astype(int)

    if "predicted_class" not in out.columns:
        out["predicted_class"] = out["predicted_class_idx"].apply(class_index_to_name)
    else:
        out["predicted_class"] = out["predicted_class"].astype(str)
        blank_mask = out["predicted_class"].str.strip().eq("")
        out.loc[blank_mask, "predicted_class"] = out.loc[
            blank_mask, "predicted_class_idx"
        ].apply(class_index_to_name)

    return out


def standardize_name_column(df: pd.DataFrame, file_label: str) -> pd.DataFrame:
    """
    Rename whichever sample-id column exists to 'file_name'.
    """
    out = df.copy()

    candidate_cols = [
        "file_name",
        "sample_name",
        "sample",
        "series_name",
        "filename",
        "name",
    ]

    for col in candidate_cols:
        if col in out.columns:
            if col != "file_name":
                out = out.rename(columns={col: "file_name"})
            return out

    raise KeyError(
        f"{file_label} does not contain any valid sample-name column. "
        f"Tried: {candidate_cols}"
    )


def merge_dl_and_csd(dl_df: pd.DataFrame, csd_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge DL and CSD results on file_name.
    """
    dl_df = standardize_name_column(dl_df, "DL file")
    csd_df = standardize_name_column(csd_df, "CSD file")

    merged = pd.merge(
        dl_df,
        csd_df,
        on="file_name",
        how="outer",
        suffixes=("_dl", "_csd"),
    )

    return merged


def build_summary(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a simple one-row comparison summary.
    """
    summary = {
        "num_rows": [len(merged_df)],
    }

    if "predicted_class" in merged_df.columns:
        counts = merged_df["predicted_class"].value_counts(dropna=False).to_dict()
        for cls_name in ["fold", "hopf", "transcritical", "null"]:
            summary[f"count_predicted_{cls_name}"] = [counts.get(cls_name, 0)]

    if "p_transition" in merged_df.columns:
        summary["mean_p_transition"] = [
            pd.to_numeric(merged_df["p_transition"], errors="coerce").mean()
        ]

    for col in merged_df.columns:
        if "kendall_tau" in col:
            summary[f"mean_{col}"] = [
                pd.to_numeric(merged_df[col], errors="coerce").mean()
            ]

    return pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description="Compare DL and CSD outputs")
    parser.add_argument("--dataset", required=True, help="Test dataset name")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--metric", required=True, help="Metric name")
    args = parser.parse_args()

    run_name = build_run_name(args.model, args.dataset, args.metric)
    paths = get_test_paths(run_name)

    logger = setup_logger(
        paths["compare_log"],
        logger_name=f"testing.compare.{run_name}",
    )

    logger.info(f"Run name: {run_name}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Metric: {args.metric}")

    dl_file = paths["dl_dir"] / "prediction_summary.csv"
    csd_file = paths["csd_dir"] / "kendall_tau_results.csv"

    logger.info(f"Loading DL file: {dl_file}")
    dl_df = load_csv_required(dl_file, "DL prediction summary")
    dl_df = ensure_predicted_class(dl_df)

    logger.info(f"Loading CSD file: {csd_file}")
    csd_df = load_csv_required(csd_file, "CSD kendall tau results")

    logger.info("Merging DL and CSD results")
    merged_df = merge_dl_and_csd(dl_df, csd_df)

    logger.info("Building summary")
    summary_df = build_summary(merged_df)

    merged_out = paths["cmp_dir"] / "dl_vs_csd_merged.csv"
    summary_out = paths["cmp_dir"] / "dl_vs_csd_summary.csv"

    save_csv(merged_df, merged_out)
    save_csv(summary_df, summary_out)

    logger.info(f"Saved merged file: {merged_out}")
    logger.info(f"Saved summary file: {summary_out}")
    logger.info("Comparison finished successfully.")


if __name__ == "__main__":
    main()
