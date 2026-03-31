from __future__ import annotations

import argparse
import json
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

from metrics.roc_auc import compute_roc_auc_binary, save_roc_dataframe
from testing.testing_utils import ensure_dir, save_json, set_seed

LOGGER = logging.getLogger("testing.compare_dl_vs_csd")


# ---------------------------------------------------------------------
# args
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare DL and CSD outputs and optionally build combined per-series plots."
    )
    parser.add_argument(
        "--dl-run-dir",
        type=str,
        required=True,
        help="Path to results/testing/<run_name>",
    )
    parser.add_argument(
        "--csd-run-dir",
        type=str,
        required=True,
        help="Path to results/testing_csd/<run_name>",
    )
    parser.add_argument(
        "--positive-class-index",
        type=int,
        default=1,
        help="Which DL probability column to treat as the positive class if class name is not given.",
    )
    parser.add_argument(
        "--positive-class-name",
        type=str,
        default=None,
        help="Optional DL class name. If given, overrides positive-class-index.",
    )
    parser.add_argument(
        "--neutral-label",
        type=int,
        default=0,
        help="Series-level label treated as neutral / non-transition.",
    )
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--make-per-series-plots", action="store_true")
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
    panel_dir = plots_dir / "per_series_panels"

    for path in [logs_dir, tables_dir, plots_dir, roc_dir, overview_dir, panel_dir]:
        ensure_dir(path)

    return {
        "logs_dir": logs_dir,
        "tables_dir": tables_dir,
        "plots_dir": plots_dir,
        "roc_dir": roc_dir,
        "overview_dir": overview_dir,
        "panel_dir": panel_dir,
    }


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def safe_series_name(series_id: str) -> str:
    return str(series_id).replace("/", "_").replace("\\", "_").replace(" ", "_")


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
    prob_cols = sorted([col for col in df_dl_final.columns if col.startswith("prob_")])
    if not prob_cols:
        raise ValueError("No probability columns found in DL final predictions CSV.")

    if positive_class_name is not None:
        requested = f"prob_{positive_class_name}"
        if requested not in prob_cols:
            raise ValueError(
                f"Requested probability column '{requested}' not found. "
                f"Available columns: {prob_cols}"
            )
        return requested

    if positive_class_index < 0 or positive_class_index >= len(prob_cols):
        raise ValueError(
            f"positive_class_index={positive_class_index} is out of range for columns {prob_cols}"
        )

    return prob_cols[positive_class_index]


def build_binary_ground_truth(y_true: pd.Series, neutral_label: int) -> np.ndarray:
    y_true = y_true.astype(float).fillna(neutral_label).astype(int)
    return (y_true != neutral_label).astype(int).values


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

    dl_keep_cols = ["series_id", positive_prob_col]
    for col in ["y_true", "y_pred", "pred_class_name"]:
        if col in df_dl.columns:
            dl_keep_cols.append(col)

    csd_keep_cols = ["series_id"]
    for col in ["final_variance_tau", "final_lag1_ac_tau", "final_variance", "final_lag1_ac"]:
        if col in df_csd.columns:
            csd_keep_cols.append(col)

    df_dl_small = df_dl[dl_keep_cols].copy().rename(columns={positive_prob_col: "dl_positive_score"})
    df_csd_small = df_csd[csd_keep_cols].copy()

    merged = pd.merge(df_dl_small, df_csd_small, on="series_id", how="inner")

    if merged.empty:
        raise ValueError(
            "Merged DL and CSD comparison dataframe is empty. "
            "Check that series_id values match between runs."
        )

    if "y_true" in merged.columns:
        merged["y_true_binary"] = build_binary_ground_truth(
            y_true=merged["y_true"],
            neutral_label=neutral_label,
        )
    else:
        merged["y_true_binary"] = np.nan

    merged["variance_tau_score"] = (
        merged["final_variance_tau"].astype(float) if "final_variance_tau" in merged.columns else np.nan
    )
    merged["lag1_ac_tau_score"] = (
        merged["final_lag1_ac_tau"].astype(float) if "final_lag1_ac_tau" in merged.columns else np.nan
    )

    return merged, positive_prob_col


# ---------------------------------------------------------------------
# roc / auc
# ---------------------------------------------------------------------
def compute_all_rocs(
    merged_df: pd.DataFrame,
    output_tables_dir: Path,
) -> Tuple[List[Dict[str, np.ndarray]], pd.DataFrame]:
    if "y_true_binary" not in merged_df.columns:
        return [], pd.DataFrame(columns=["method", "auc"])

    methods = [
        ("DL", "dl_positive_score"),
        ("Variance Kendall tau", "variance_tau_score"),
        ("Lag-1 AC Kendall tau", "lag1_ac_tau_score"),
    ]

    roc_items: List[Dict[str, np.ndarray]] = []
    auc_rows: List[Dict[str, float]] = []

    for method_name, score_col in methods:
        if score_col not in merged_df.columns:
            continue

        valid_mask = (
            pd.to_numeric(merged_df["y_true_binary"], errors="coerce").notna()
            & pd.to_numeric(merged_df[score_col], errors="coerce").notna()
        )

        if valid_mask.sum() == 0:
            LOGGER.warning("Skipping ROC for %s because no valid rows were found.", method_name)
            continue

        y_true_binary = merged_df.loc[valid_mask, "y_true_binary"].astype(int).values
        scores = merged_df.loc[valid_mask, score_col].astype(float).values

        if len(np.unique(y_true_binary)) < 2:
            LOGGER.warning(
                "Skipping ROC for %s because the valid subset does not contain both classes.",
                method_name,
            )
            continue

        result = compute_roc_auc_binary(
            y_true_binary=y_true_binary,
            scores=scores,
        )

        roc_items.append(
            {
                "name": method_name,
                "thresholds": result["thresholds"],
                "fpr": result["fpr"],
                "tpr": result["tpr"],
                "auc": result["auc"],
            }
        )

        auc_rows.append(
            {
                "method": method_name,
                "auc": result["auc"],
                "num_valid_rows": int(valid_mask.sum()),
            }
        )

        safe_method_name = method_name.lower().replace(" ", "_").replace("-", "_")
        save_roc_dataframe(
            thresholds=result["thresholds"],
            fpr=result["fpr"],
            tpr=result["tpr"],
            save_path=output_tables_dir / f"roc_{safe_method_name}.csv",
        )

    auc_df = pd.DataFrame(auc_rows)
    if not auc_df.empty:
        auc_df = auc_df.sort_values("auc", ascending=False).reset_index(drop=True)

    return roc_items, auc_df


# ---------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------
def plot_roc_curves(
    roc_items: List[Dict[str, np.ndarray]],
    save_path: Path,
    title: str = "ROC comparison: DL vs CSD",
) -> None:
    if not roc_items:
        return

    plt.figure(figsize=(7, 6))

    for item in roc_items:
        auc = item["auc"]
        label = f"{item['name']} (AUC={auc:.3f})" if np.isfinite(auc) else f"{item['name']} (AUC=nan)"
        plt.plot(item["fpr"], item["tpr"], label=label)

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_per_series_panel(
    signal_df: pd.DataFrame,
    dl_df: pd.DataFrame,
    csd_df: pd.DataFrame,
    save_path: Path,
    title: str,
    transition_index: Optional[float],
) -> None:
    if signal_df.empty or dl_df.empty or csd_df.empty:
        return

    probability_cols = sorted([col for col in dl_df.columns if col.startswith("prob_")])

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)

    # raw + smooth
    axes[0].plot(
        signal_df["local_index"],
        signal_df["raw_signal"],
        color="black",
        linewidth=1.0,
        label="Raw",
    )
    if "smooth_signal" in signal_df.columns:
        axes[0].plot(
            signal_df["local_index"],
            signal_df["smooth_signal"],
            color="gray",
            linewidth=1.4,
            label="Smooth",
        )
    axes[0].set_ylabel("Signal")
    axes[0].set_title(title)
    axes[0].legend(loc="upper right")

    # DL probabilities
    for col in probability_cols:
        class_name = col.replace("prob_", "", 1)
        axes[1].plot(dl_df["reveal_index"], dl_df[col], label=class_name)
    axes[1].set_ylabel("DL probability")
    if probability_cols:
        axes[1].legend(loc="upper right")

    # indicators
    axes[2].plot(csd_df["reveal_index"], csd_df["variance"], label="Variance")
    axes[2].plot(csd_df["reveal_index"], csd_df["lag1_ac"], label="Lag-1 AC")
    axes[2].set_ylabel("Indicator")
    axes[2].legend(loc="upper right")

    # kendall tau
    axes[3].plot(csd_df["reveal_index"], csd_df["variance_tau"], label="Variance tau")
    axes[3].plot(csd_df["reveal_index"], csd_df["lag1_ac_tau"], label="Lag-1 AC tau")
    axes[3].set_xlabel("Local index")
    axes[3].set_ylabel("Kendall tau")
    axes[3].legend(loc="upper right")

    if transition_index is not None:
        x_max = float(signal_df["local_index"].max())
        for ax in axes:
            ax.axvline(x=transition_index, linestyle="--", linewidth=1.2)
            if transition_index < x_max:
                ax.axvspan(transition_index, x_max, color="gray", alpha=0.08)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def build_per_series_combined_plots(
    dl_run_dir: Path,
    csd_run_dir: Path,
    output_panel_dir: Path,
) -> int:
    dl_progressive_path = dl_run_dir / "all_series_progressive_predictions.csv"
    csd_progressive_path = csd_run_dir / "all_series_csd_predictions.csv"
    signal_dir = csd_run_dir / "series_signals"

    if not dl_progressive_path.exists() or not csd_progressive_path.exists() or not signal_dir.exists():
        LOGGER.warning(
            "Progressive CSVs or the series_signals directory were not found. "
            "Skipping combined per-series plots."
        )
        return 0

    dl_df = load_csv_required(dl_progressive_path)
    csd_df = load_csv_required(csd_progressive_path)

    if "series_id" not in dl_df.columns or "series_id" not in csd_df.columns:
        LOGGER.warning("Progressive CSVs must contain 'series_id'. Skipping combined per-series plots.")
        return 0

    common_series = sorted(set(dl_df["series_id"].unique()) & set(csd_df["series_id"].unique()))
    num_plots = 0

    for series_id in common_series:
        safe_name = safe_series_name(series_id)
        signal_path = signal_dir / f"{safe_name}.csv"
        if not signal_path.exists():
            LOGGER.warning("Signal CSV missing for series_id=%s", series_id)
            continue

        signal_df = pd.read_csv(signal_path)
        dl_series_df = (
            dl_df[dl_df["series_id"] == series_id]
            .sort_values("reveal_index")
            .reset_index(drop=True)
        )
        csd_series_df = (
            csd_df[csd_df["series_id"] == series_id]
            .sort_values("reveal_index")
            .reset_index(drop=True)
        )

        transition_index = None
        for frame in [signal_df, dl_series_df, csd_series_df]:
            if "transition_index" in frame.columns:
                valid = pd.to_numeric(frame["transition_index"], errors="coerce").dropna().values
                if len(valid) > 0:
                    transition_index = float(valid[0])
                    break

        plot_per_series_panel(
            signal_df=signal_df,
            dl_df=dl_series_df,
            csd_df=csd_series_df,
            save_path=output_panel_dir / f"{safe_name}.png",
            title=f"Series {series_id}",
            transition_index=transition_index,
        )
        num_plots += 1

    return num_plots


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

    run_name = args.run_name or f"compare_{dl_run_dir.name}_vs_{csd_run_dir.name}"
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
    LOGGER.info("Run directory        : %s", run_dir)

    config = {
        "dl_run_dir": str(dl_run_dir),
        "csd_run_dir": str(csd_run_dir),
        "positive_class_index": args.positive_class_index,
        "positive_class_name": args.positive_class_name,
        "neutral_label": args.neutral_label,
        "run_name": run_name,
        "make_per_series_plots": bool(args.make_per_series_plots),
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

    LOGGER.info("Merged rows         : %d", len(merged_df))
    LOGGER.info("Merged series count : %d", merged_df["series_id"].nunique())
    LOGGER.info("Positive prob col   : %s", positive_prob_col)

    LOGGER.info("-" * 80)
    LOGGER.info("[2/3] ROC / AUC computation")
    LOGGER.info("-" * 80)
    roc_items, auc_df = compute_all_rocs(
        merged_df=merged_df,
        output_tables_dir=output_dirs["tables_dir"],
    )

    if not auc_df.empty:
        auc_df.to_csv(output_dirs["tables_dir"] / "auc_summary.csv", index=False)
        plot_roc_curves(
            roc_items=roc_items,
            save_path=output_dirs["roc_dir"] / "dl_vs_csd_roc.png",
        )
        LOGGER.info("Saved ROC/AUC outputs to: %s", output_dirs["roc_dir"])
    else:
        LOGGER.info("ROC/AUC skipped because valid binary labels were not available.")

    LOGGER.info("-" * 80)
    LOGGER.info("[3/3] Optional per-series combined plots")
    LOGGER.info("-" * 80)
    num_panel_plots = 0
    if args.make_per_series_plots:
        num_panel_plots = build_per_series_combined_plots(
            dl_run_dir=dl_run_dir,
            csd_run_dir=csd_run_dir,
            output_panel_dir=output_dirs["panel_dir"],
        )
        LOGGER.info("Saved %d combined per-series plots to: %s", num_panel_plots, output_dirs["panel_dir"])
    else:
        LOGGER.info("Per-series combined plots skipped.")

    summary = {
        "num_merged_rows": int(len(merged_df)),
        "num_series": int(merged_df["series_id"].nunique()),
        "positive_prob_column": positive_prob_col,
        "num_roc_methods": int(len(roc_items)),
        "num_per_series_plots": int(num_panel_plots),
        "best_method": None if auc_df.empty else str(auc_df.iloc[0]["method"]),
    }
    save_json(summary, run_dir / "summary.json")

    LOGGER.info(json.dumps(summary, indent=2))
    LOGGER.info("=" * 80)
    LOGGER.info("Comparison completed successfully")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
