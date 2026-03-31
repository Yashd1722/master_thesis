# testing/compare_dl_vs_csd.py

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# repo root import setup
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.roc_auc import compute_roc_auc_binary, save_roc_dataframe  # noqa: E402
from testing.testing_utils import ensure_dir, save_json, set_seed  # noqa: E402

LOGGER = logging.getLogger("testing.compare_dl_vs_csd")


# ---------------------------------------------------------------------
# args
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare DL vs CSD outputs using ROC/AUC and merged per-series summaries."
    )

    parser.add_argument("--dl-run-dir", type=str, required=True, help="Path to results/testing/<dl_run_name>")
    parser.add_argument("--csd-run-dir", type=str, required=True, help="Path to results/testing_csd/<csd_run_name>")

    parser.add_argument(
        "--positive-class-index",
        type=int,
        default=1,
        help="Which DL class index should be treated as the positive class for ROC."
    )
    parser.add_argument(
        "--positive-class-name",
        type=str,
        default=None,
        help="Optional DL class name. If given, overrides positive-class-index lookup."
    )
    parser.add_argument(
        "--neutral-label",
        type=int,
        default=0,
        help="Label treated as neutral/non-transition in series-level truth."
    )

    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------
def setup_run_logging(log_file: Path, verbose: bool = False) -> None:
    ensure_dir(log_file.parent)

    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if root_logger.handlers:
        root_logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)


# ---------------------------------------------------------------------
# dirs
# ---------------------------------------------------------------------
def get_compare_output_dirs(run_dir: Path) -> Dict[str, Path]:
    logs_dir = run_dir / "logs"
    tables_dir = run_dir / "tables"
    plots_dir = run_dir / "plots"
    roc_dir = plots_dir / "roc"
    overview_dir = plots_dir / "overview"

    for path in [logs_dir, tables_dir, plots_dir, roc_dir, overview_dir]:
        ensure_dir(path)

    return {
        "logs_dir": logs_dir,
        "tables_dir": tables_dir,
        "plots_dir": plots_dir,
        "roc_dir": roc_dir,
        "overview_dir": overview_dir,
    }


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def load_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        LOGGER.warning("CSV is empty: %s", path)
    return df


def infer_positive_prob_column(
    df_dl_final: pd.DataFrame,
    positive_class_index: int,
    positive_class_name: Optional[str],
) -> str:
    prob_cols = sorted([c for c in df_dl_final.columns if c.startswith("prob_")])
    if not prob_cols:
        raise ValueError("No probability columns found in DL final predictions CSV.")

    if positive_class_name is not None:
        candidate = f"prob_{positive_class_name}"
        if candidate not in prob_cols:
            raise ValueError(
                f"Requested positive class column '{candidate}' not found. "
                f"Available columns: {prob_cols}"
            )
        return candidate

    if positive_class_index < 0 or positive_class_index >= len(prob_cols):
        raise ValueError(
            f"positive_class_index={positive_class_index} is out of range for columns {prob_cols}"
        )
    return prob_cols[positive_class_index]


def build_binary_ground_truth(
    y_true: pd.Series,
    neutral_label: int,
) -> np.ndarray:
    y_true = y_true.astype(float).fillna(neutral_label).astype(int)
    return (y_true != neutral_label).astype(int).values


def kendall_tau_direction_adjusted_score(x: pd.Series) -> np.ndarray:
    """
    Pass-through for now.
    Later, if needed, lag-1 AC can be sign-adjusted for oscillatory transitions.
    """
    return x.astype(float).values


def safe_method_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


# ---------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------
def plot_roc_curves(
    roc_items: List[Dict[str, np.ndarray]],
    save_path: Path,
    title: str = "ROC comparison: DL vs CSD",
) -> None:
    plt.figure(figsize=(7, 6))

    for item in roc_items:
        name = item["name"]
        fpr = item["fpr"]
        tpr = item["tpr"]
        auc = item["auc"]

        label = f"{name} (AUC={auc:.3f})" if np.isfinite(auc) else f"{name} (AUC=nan)"
        plt.plot(fpr, tpr, label=label)

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_auc_bar(auc_df: pd.DataFrame, save_path: Path) -> None:
    if auc_df.empty:
        return

    plt.figure(figsize=(7, 4))
    plt.bar(auc_df["method"], auc_df["auc"])
    plt.xlabel("method")
    plt.ylabel("AUC")
    plt.title("AUC comparison")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# merge logic
# ---------------------------------------------------------------------
def prepare_merged_comparison_df(
    dl_run_dir: Path,
    csd_run_dir: Path,
    positive_class_index: int,
    positive_class_name: Optional[str],
    neutral_label: int,
) -> Tuple[pd.DataFrame, str]:
    dl_final_path = dl_run_dir / "final_series_predictions.csv"
    csd_final_path = csd_run_dir / "final_series_csd_predictions.csv"

    df_dl = load_csv_required(dl_final_path)
    df_csd = load_csv_required(csd_final_path)

    if "series_id" not in df_dl.columns:
        raise ValueError("DL final predictions must contain 'series_id'.")
    if "series_id" not in df_csd.columns:
        raise ValueError("CSD final predictions must contain 'series_id'.")

    positive_prob_col = infer_positive_prob_column(
        df_dl_final=df_dl,
        positive_class_index=positive_class_index,
        positive_class_name=positive_class_name,
    )

    merge_cols_dl = ["series_id", positive_prob_col]
    if "y_true" in df_dl.columns:
        merge_cols_dl.append("y_true")
    if "y_pred" in df_dl.columns:
        merge_cols_dl.append("y_pred")
    if "pred_class_name" in df_dl.columns:
        merge_cols_dl.append("pred_class_name")

    df_dl_small = df_dl[merge_cols_dl].copy()
    df_dl_small = df_dl_small.rename(columns={positive_prob_col: "dl_positive_score"})

    keep_csd_cols = ["series_id"]
    for col in ["final_variance_tau", "final_lag1_ac_tau", "final_variance", "final_lag1_ac"]:
        if col in df_csd.columns:
            keep_csd_cols.append(col)

    df_csd_small = df_csd[keep_csd_cols].copy()

    merged = pd.merge(df_dl_small, df_csd_small, on="series_id", how="inner")

    if merged.empty:
        raise ValueError("Merged DL and CSD comparison dataframe is empty. Check matching series_id values.")

    if "y_true" not in merged.columns:
        raise ValueError(
            "Series-level truth column 'y_true' not found in DL final predictions. "
            "Need y_true to compute ROC/AUC."
        )

    merged["y_true_binary"] = build_binary_ground_truth(
        y_true=merged["y_true"],
        neutral_label=neutral_label,
    )

    if "final_variance_tau" in merged.columns:
        merged["variance_tau_score"] = kendall_tau_direction_adjusted_score(merged["final_variance_tau"])
    else:
        merged["variance_tau_score"] = np.nan

    if "final_lag1_ac_tau" in merged.columns:
        merged["lag1_ac_tau_score"] = kendall_tau_direction_adjusted_score(merged["final_lag1_ac_tau"])
    else:
        merged["lag1_ac_tau_score"] = np.nan

    return merged, positive_prob_col


# ---------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------
def compute_all_rocs(
    merged_df: pd.DataFrame,
    output_tables_dir: Path,
) -> Tuple[List[Dict[str, np.ndarray]], pd.DataFrame]:
    y_true_binary = merged_df["y_true_binary"].astype(int).values

    methods = [
        ("DL", merged_df["dl_positive_score"].astype(float).values),
        ("Variance Kendall tau", merged_df["variance_tau_score"].astype(float).values),
        ("Lag-1 AC Kendall tau", merged_df["lag1_ac_tau_score"].astype(float).values),
    ]

    roc_items: List[Dict[str, np.ndarray]] = []
    auc_rows: List[Dict[str, float]] = []

    for name, scores in methods:
        result = compute_roc_auc_binary(
            y_true_binary=y_true_binary,
            scores=scores,
        )

        thresholds = result["thresholds"]
        fpr = result["fpr"]
        tpr = result["tpr"]
        auc = result["auc"]

        roc_items.append(
            {
                "name": name,
                "thresholds": thresholds,
                "fpr": fpr,
                "tpr": tpr,
                "auc": auc,
            }
        )

        auc_rows.append(
            {
                "method": name,
                "auc": auc,
            }
        )

        save_roc_dataframe(
            thresholds=thresholds,
            fpr=fpr,
            tpr=tpr,
            save_path=output_tables_dir / f"roc_{safe_method_name(name)}.csv",
        )

    auc_df = pd.DataFrame(auc_rows).sort_values("auc", ascending=False).reset_index(drop=True)
    return roc_items, auc_df


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    set_seed(42)

    dl_run_dir = Path(args.dl_run_dir)
    csd_run_dir = Path(args.csd_run_dir)

    if not dl_run_dir.exists():
        raise FileNotFoundError(f"DL run directory not found: {dl_run_dir}")
    if not csd_run_dir.exists():
        raise FileNotFoundError(f"CSD run directory not found: {csd_run_dir}")

    run_name = (
        args.run_name
        if args.run_name is not None
        else f"compare_{dl_run_dir.name}_vs_{csd_run_dir.name}"
    )

    run_dir = Path(args.results_root) / "comparison" / run_name
    output_dirs = get_compare_output_dirs(run_dir)

    log_file = output_dirs["logs_dir"] / "compare_dl_vs_csd.log"
    setup_run_logging(log_file=log_file, verbose=args.verbose)

    LOGGER.info("=" * 80)
    LOGGER.info("Starting DL vs CSD comparison")
    LOGGER.info("=" * 80)
    LOGGER.info("DL run dir           : %s", dl_run_dir)
    LOGGER.info("CSD run dir          : %s", csd_run_dir)
    LOGGER.info("Positive class index : %d", args.positive_class_index)
    LOGGER.info("Positive class name  : %s", args.positive_class_name)
    LOGGER.info("Neutral label        : %d", args.neutral_label)
    LOGGER.info("Comparison run dir   : %s", run_dir)

    config = {
        "dl_run_dir": str(dl_run_dir),
        "csd_run_dir": str(csd_run_dir),
        "positive_class_index": args.positive_class_index,
        "positive_class_name": args.positive_class_name,
        "neutral_label": args.neutral_label,
        "run_name": run_name,
    }
    save_json(config, run_dir / "run_config.json")

    LOGGER.info("-" * 80)
    LOGGER.info("[1/3] Preparing merged comparison dataframe")
    LOGGER.info("-" * 80)

    merged_df, positive_prob_col = prepare_merged_comparison_df(
        dl_run_dir=dl_run_dir,
        csd_run_dir=csd_run_dir,
        positive_class_index=args.positive_class_index,
        positive_class_name=args.positive_class_name,
        neutral_label=args.neutral_label,
    )

    merged_csv_path = run_dir / "merged_dl_csd_comparison.csv"
    merged_df.to_csv(merged_csv_path, index=False)
    merged_df.to_csv(output_dirs["tables_dir"] / "merged_dl_csd_comparison.csv", index=False)

    LOGGER.info("Merged rows           : %d", len(merged_df))
    LOGGER.info("Merged series count   : %d", merged_df["series_id"].nunique())
    LOGGER.info("DL positive prob col  : %s", positive_prob_col)

    LOGGER.info("-" * 80)
    LOGGER.info("[2/3] Computing ROC and AUC")
    LOGGER.info("-" * 80)

    roc_items, auc_df = compute_all_rocs(
        merged_df=merged_df,
        output_tables_dir=output_dirs["tables_dir"],
    )

    auc_csv_path = run_dir / "auc_summary.csv"
    auc_df.to_csv(auc_csv_path, index=False)
    auc_df.to_csv(output_dirs["tables_dir"] / "auc_summary.csv", index=False)

    LOGGER.info("AUC summary:")
    LOGGER.info("\n%s", auc_df.to_string(index=False))

    LOGGER.info("-" * 80)
    LOGGER.info("[3/3] Plot generation")
    LOGGER.info("-" * 80)

    plot_roc_curves(
        roc_items=roc_items,
        save_path=output_dirs["roc_dir"] / "roc_comparison.png",
        title="ROC comparison: DL vs CSD",
    )

    plot_auc_bar(
        auc_df=auc_df,
        save_path=output_dirs["overview_dir"] / "auc_comparison.png",
    )

    summary = {
        "num_merged_rows": int(len(merged_df)),
        "num_series": int(merged_df["series_id"].nunique()),
        "positive_prob_column": positive_prob_col,
        "best_method": str(auc_df.iloc[0]["method"]) if not auc_df.empty else None,
        "best_auc": float(auc_df.iloc[0]["auc"]) if not auc_df.empty else None,
    }
    save_json(summary, run_dir / "summary.json")

    LOGGER.info("=" * 80)
    LOGGER.info("Comparison completed successfully")
    LOGGER.info("All outputs saved in: %s", run_dir)
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
