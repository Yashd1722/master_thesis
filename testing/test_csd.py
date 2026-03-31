# testing/test_csd.py

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# repo root import setup
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.testing_utils import (  # noqa: E402
    ensure_dir,
    load_test_dataset_for_inference,
    save_json,
    set_seed,
)

LOGGER = logging.getLogger("testing.test_csd")


# ---------------------------------------------------------------------
# args
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test CSD indicators progressively on synthetic/empirical datasets."
    )

    parser.add_argument("--dataset", type=str, required=True, help="Dataset token: ts_500, ts_1500, pangaea_923197")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--rolling-window-frac", type=float, default=0.50)
    parser.add_argument("--progressive-start-frac", type=float, default=0.60)
    parser.add_argument("--progressive-end-frac", type=float, default=1.00)
    parser.add_argument("--progressive-num-steps", type=int, default=40)
    parser.add_argument("--min-prefix-len", type=int, default=64)
    parser.add_argument("--gaussian-smooth-sigma", type=float, default=0.0)

    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--save-series-csv", action="store_true")
    parser.add_argument("--make-plots", action="store_true")
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
# folder helpers
# ---------------------------------------------------------------------
def get_csd_output_dirs(run_dir: Path) -> Dict[str, Path]:
    logs_dir = run_dir / "logs"
    tables_dir = run_dir / "tables"
    plots_dir = run_dir / "plots"
    indicator_dir = plots_dir / "indicator_curves"
    tau_dir = plots_dir / "kendall_tau_curves"
    overview_dir = plots_dir / "overview"
    series_csv_dir = run_dir / "series_predictions"

    for path in [logs_dir, tables_dir, plots_dir, indicator_dir, tau_dir, overview_dir, series_csv_dir]:
        ensure_dir(path)

    return {
        "logs_dir": logs_dir,
        "tables_dir": tables_dir,
        "plots_dir": plots_dir,
        "indicator_dir": indicator_dir,
        "tau_dir": tau_dir,
        "overview_dir": overview_dir,
        "series_csv_dir": series_csv_dir,
    }


# ---------------------------------------------------------------------
# math helpers
# ---------------------------------------------------------------------
def gaussian_kernel1d(sigma: float, radius: Optional[int] = None) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float64)

    if radius is None:
        radius = max(1, int(round(4 * sigma)))

    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel


def gaussian_smooth_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if sigma <= 0 or len(x) < 3:
        return x.copy()

    kernel = gaussian_kernel1d(sigma=sigma)
    pad = len(kernel) // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    smoothed = np.convolve(x_pad, kernel, mode="valid")
    return smoothed


def kendall_tau(x: np.ndarray) -> float:
    """
    Simple Kendall tau-a implementation without scipy.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = len(x)
    if n < 2:
        return np.nan

    concordant = 0
    discordant = 0

    for i in range(n - 1):
        diff = x[i + 1:] - x[i]
        concordant += np.sum(diff > 0)
        discordant += np.sum(diff < 0)

    denom = n * (n - 1) / 2
    if denom == 0:
        return np.nan
    return (concordant - discordant) / denom


def compute_variance(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) < 2:
        return np.nan
    return float(np.var(x, ddof=1))


def compute_lag1_autocorrelation(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) < 3:
        return np.nan

    x0 = x[:-1]
    x1 = x[1:]

    x0_mean = np.mean(x0)
    x1_mean = np.mean(x1)

    num = np.sum((x0 - x0_mean) * (x1 - x1_mean))
    den = np.sqrt(np.sum((x0 - x0_mean) ** 2) * np.sum((x1 - x1_mean) ** 2))

    if den == 0:
        return np.nan
    return float(num / den)


def rolling_indicator_values(
    x: np.ndarray,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        indices, rolling_variance, rolling_lag1_ac
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = len(x)

    if n < window_size or window_size < 3:
        return np.array([]), np.array([]), np.array([])

    indices = []
    vars_ = []
    acs_ = []

    for end_idx in range(window_size, n + 1):
        window = x[end_idx - window_size:end_idx]
        indices.append(end_idx)
        vars_.append(compute_variance(window))
        acs_.append(compute_lag1_autocorrelation(window))

    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(vars_, dtype=np.float64),
        np.asarray(acs_, dtype=np.float64),
    )


def progressive_reveal_indices(
    series_length: int,
    progressive_start_frac: float,
    progressive_end_frac: float,
    progressive_num_steps: int,
    min_prefix_len: int,
) -> List[int]:
    if series_length <= 0:
        return []

    start_idx = max(min_prefix_len, int(round(series_length * progressive_start_frac)))
    end_idx = max(start_idx, int(round(series_length * progressive_end_frac)))

    start_idx = min(start_idx, series_length)
    end_idx = min(end_idx, series_length)

    if progressive_num_steps <= 1:
        return [end_idx]

    values = np.linspace(start_idx, end_idx, progressive_num_steps)
    values = np.unique(values.astype(int))
    values = np.clip(values, 1, series_length)

    return [int(v) for v in values.tolist()]


# ---------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------
def plot_indicator_curves(
    series_df: pd.DataFrame,
    save_path: Path,
    transition_index: Optional[float] = None,
    title: Optional[str] = None,
) -> None:
    if series_df.empty:
        return

    plt.figure(figsize=(10, 6))

    if "reveal_index" in series_df.columns and "variance" in series_df.columns:
        plt.plot(series_df["reveal_index"], series_df["variance"], label="variance")
    if "reveal_index" in series_df.columns and "lag1_ac" in series_df.columns:
        plt.plot(series_df["reveal_index"], series_df["lag1_ac"], label="lag1_ac")

    if transition_index is not None:
        plt.axvline(x=transition_index, linestyle="--", linewidth=1.5, label="transition")

    plt.xlabel("reveal_index")
    plt.ylabel("indicator value")
    plt.title(title or "CSD indicator curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_tau_curves(
    series_df: pd.DataFrame,
    save_path: Path,
    transition_index: Optional[float] = None,
    title: Optional[str] = None,
) -> None:
    if series_df.empty:
        return

    plt.figure(figsize=(10, 6))

    if "reveal_index" in series_df.columns and "variance_tau" in series_df.columns:
        plt.plot(series_df["reveal_index"], series_df["variance_tau"], label="variance_tau")
    if "reveal_index" in series_df.columns and "lag1_ac_tau" in series_df.columns:
        plt.plot(series_df["reveal_index"], series_df["lag1_ac_tau"], label="lag1_ac_tau")

    if transition_index is not None:
        plt.axvline(x=transition_index, linestyle="--", linewidth=1.5, label="transition")

    plt.xlabel("reveal_index")
    plt.ylabel("kendall tau")
    plt.title(title or "Kendall tau curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_final_tau_distribution(df_final: pd.DataFrame, save_path: Path) -> None:
    if df_final.empty:
        return

    plt.figure(figsize=(8, 4))

    vals1 = df_final["final_variance_tau"].dropna().values if "final_variance_tau" in df_final.columns else np.array([])
    vals2 = df_final["final_lag1_ac_tau"].dropna().values if "final_lag1_ac_tau" in df_final.columns else np.array([])

    bins = 20
    if len(vals1) > 0:
        plt.hist(vals1, bins=bins, alpha=0.6, label="final_variance_tau")
    if len(vals2) > 0:
        plt.hist(vals2, bins=bins, alpha=0.6, label="final_lag1_ac_tau")

    plt.xlabel("kendall tau")
    plt.ylabel("count")
    plt.title("Final Kendall tau distribution across series")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# per-series evaluation
# ---------------------------------------------------------------------
def evaluate_single_series_csd(
    x_full: np.ndarray,
    series_id: str,
    y_true: Optional[int],
    transition_index: Optional[int],
    rolling_window_frac: float,
    progressive_start_frac: float,
    progressive_end_frac: float,
    progressive_num_steps: int,
    min_prefix_len: int,
    gaussian_smooth_sigma: float,
) -> List[Dict[str, Any]]:
    x_full = np.asarray(x_full, dtype=np.float64).reshape(-1)
    n = len(x_full)

    reveal_idxs = progressive_reveal_indices(
        series_length=n,
        progressive_start_frac=progressive_start_frac,
        progressive_end_frac=progressive_end_frac,
        progressive_num_steps=progressive_num_steps,
        min_prefix_len=min_prefix_len,
    )

    if len(reveal_idxs) == 0:
        return []

    records: List[Dict[str, Any]] = []

    for step_idx, reveal_idx in enumerate(reveal_idxs):
        prefix = x_full[:reveal_idx]

        smooth_prefix = gaussian_smooth_1d(prefix, sigma=gaussian_smooth_sigma)
        residual = prefix - smooth_prefix

        window_size = max(3, int(round(len(prefix) * rolling_window_frac)))
        roll_idx, roll_var, roll_ac = rolling_indicator_values(residual, window_size=window_size)

        var_tau = kendall_tau(roll_var) if len(roll_var) >= 2 else np.nan
        ac_tau = kendall_tau(roll_ac) if len(roll_ac) >= 2 else np.nan

        row: Dict[str, Any] = {
            "series_id": str(series_id),
            "step_idx": int(step_idx),
            "reveal_index": int(reveal_idx),
            "reveal_fraction": float(reveal_idx / max(n, 1)),
            "series_length": int(n),
            "rolling_window_size": int(window_size),
            "variance": float(roll_var[-1]) if len(roll_var) > 0 else np.nan,
            "lag1_ac": float(roll_ac[-1]) if len(roll_ac) > 0 else np.nan,
            "variance_tau": float(var_tau) if not np.isnan(var_tau) else np.nan,
            "lag1_ac_tau": float(ac_tau) if not np.isnan(ac_tau) else np.nan,
            "transition_index": None if transition_index is None else int(transition_index),
        }

        if y_true is not None:
            row["y_true"] = int(y_true)

        records.append(row)

    return records


def run_csd_evaluation(
    series_dataset: Any,
    rolling_window_frac: float,
    progressive_start_frac: float,
    progressive_end_frac: float,
    progressive_num_steps: int,
    min_prefix_len: int,
    gaussian_smooth_sigma: float,
) -> pd.DataFrame:
    all_records: List[Dict[str, Any]] = []
    total_series = len(series_dataset.series_items)

    for idx, item in enumerate(series_dataset.series_items, start=1):
        series_id = str(item["series_id"])
        x_full = np.asarray(item["signal"], dtype=np.float64).reshape(-1)
        y_true = item.get("label", None)
        transition_index = item.get("transition_index", None)

        LOGGER.info(
            "CSD evaluation [%d/%d] | series_id=%s | length=%d",
            idx,
            total_series,
            series_id,
            len(x_full),
        )

        recs = evaluate_single_series_csd(
            x_full=x_full,
            series_id=series_id,
            y_true=y_true,
            transition_index=transition_index,
            rolling_window_frac=rolling_window_frac,
            progressive_start_frac=progressive_start_frac,
            progressive_end_frac=progressive_end_frac,
            progressive_num_steps=progressive_num_steps,
            min_prefix_len=min_prefix_len,
            gaussian_smooth_sigma=gaussian_smooth_sigma,
        )

        if len(recs) == 0:
            LOGGER.warning("No CSD records created for series_id=%s", series_id)
            continue

        all_records.extend(recs)

    return pd.DataFrame(all_records)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_name = args.run_name if args.run_name is not None else f"csd_{args.dataset}"
    run_dir = Path(args.results_root) / "testing_csd" / run_name
    output_dirs = get_csd_output_dirs(run_dir)

    log_file = output_dirs["logs_dir"] / "test_csd.log"
    setup_run_logging(log_file=log_file, verbose=args.verbose)

    LOGGER.info("=" * 80)
    LOGGER.info("Starting CSD testing run")
    LOGGER.info("=" * 80)
    LOGGER.info("Dataset             : %s", args.dataset)
    LOGGER.info("Split               : %s", args.split)
    LOGGER.info("Rolling window frac : %.4f", args.rolling_window_frac)
    LOGGER.info("Progressive start   : %.4f", args.progressive_start_frac)
    LOGGER.info("Progressive end     : %.4f", args.progressive_end_frac)
    LOGGER.info("Progressive steps   : %d", args.progressive_num_steps)
    LOGGER.info("Min prefix len      : %d", args.min_prefix_len)
    LOGGER.info("Smooth sigma        : %.4f", args.gaussian_smooth_sigma)
    LOGGER.info("Run directory       : %s", run_dir)
    LOGGER.info("Log file            : %s", log_file)

    config_summary = {
        "run_name": run_name,
        "dataset": args.dataset,
        "split": args.split,
        "rolling_window_frac": args.rolling_window_frac,
        "progressive_start_frac": args.progressive_start_frac,
        "progressive_end_frac": args.progressive_end_frac,
        "progressive_num_steps": args.progressive_num_steps,
        "min_prefix_len": args.min_prefix_len,
        "gaussian_smooth_sigma": args.gaussian_smooth_sigma,
    }
    save_json(config_summary, run_dir / "run_config.json")

    LOGGER.info("Loading dataset...")
    loaded = load_test_dataset_for_inference(
        dataset_name=args.dataset,
        split=args.split,
        input_length=500,
        num_classes=3,
    )
    series_dataset = loaded["series_dataset"]

    LOGGER.info("-" * 80)
    LOGGER.info("[1/2] Progressive CSD evaluation")
    LOGGER.info("-" * 80)

    csd_df = run_csd_evaluation(
        series_dataset=series_dataset,
        rolling_window_frac=args.rolling_window_frac,
        progressive_start_frac=args.progressive_start_frac,
        progressive_end_frac=args.progressive_end_frac,
        progressive_num_steps=args.progressive_num_steps,
        min_prefix_len=args.min_prefix_len,
        gaussian_smooth_sigma=args.gaussian_smooth_sigma,
    )

    if csd_df.empty:
        LOGGER.warning("No CSD outputs were generated.")
        LOGGER.info("Finished.")
        return

    combined_csv_path = run_dir / "all_series_csd_predictions.csv"
    csd_df.to_csv(combined_csv_path, index=False)
    csd_df.to_csv(output_dirs["tables_dir"] / "all_series_csd_predictions.csv", index=False)
    LOGGER.info("Saved combined CSD predictions: %s", combined_csv_path)

    final_df = (
        csd_df
        .sort_values(["series_id", "reveal_index"])
        .groupby("series_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
        .rename(columns={
            "variance_tau": "final_variance_tau",
            "lag1_ac_tau": "final_lag1_ac_tau",
            "variance": "final_variance",
            "lag1_ac": "final_lag1_ac",
        })
    )

    final_csv_path = run_dir / "final_series_csd_predictions.csv"
    final_df.to_csv(final_csv_path, index=False)
    final_df.to_csv(output_dirs["tables_dir"] / "final_series_csd_predictions.csv", index=False)
    LOGGER.info("Saved final CSD predictions: %s", final_csv_path)

    if args.save_series_csv:
        for series_id, series_df in csd_df.groupby("series_id", sort=True):
            safe_name = str(series_id).replace("/", "_").replace("\\", "_").replace(" ", "_")
            series_df.sort_values("reveal_index").to_csv(
                output_dirs["series_csv_dir"] / f"{safe_name}.csv",
                index=False,
            )
        LOGGER.info("Saved per-series CSD CSV files: %s", output_dirs["series_csv_dir"])

    LOGGER.info("-" * 80)
    LOGGER.info("[2/2] Plot generation")
    LOGGER.info("-" * 80)

    if args.make_plots:
        for series_id, series_df in csd_df.groupby("series_id", sort=True):
            series_df = series_df.sort_values("reveal_index").reset_index(drop=True)
            safe_name = str(series_id).replace("/", "_").replace("\\", "_").replace(" ", "_")

            transition_index = None
            if "transition_index" in series_df.columns:
                vals = series_df["transition_index"].dropna().values
                if len(vals) > 0:
                    transition_index = float(vals[0])

            plot_indicator_curves(
                series_df=series_df,
                save_path=output_dirs["indicator_dir"] / f"{safe_name}.png",
                transition_index=transition_index,
                title=f"Series {series_id} - CSD indicators",
            )

            plot_tau_curves(
                series_df=series_df,
                save_path=output_dirs["tau_dir"] / f"{safe_name}.png",
                transition_index=transition_index,
                title=f"Series {series_id} - Kendall tau curves",
            )

        plot_final_tau_distribution(
            df_final=final_df,
            save_path=output_dirs["overview_dir"] / "final_tau_distribution.png",
        )
        LOGGER.info("Saved CSD plots under: %s", output_dirs["plots_dir"])
    else:
        LOGGER.info("Plot generation skipped.")

    summary = {
        "num_series": int(csd_df["series_id"].nunique()),
        "num_rows": int(len(csd_df)),
        "num_final_rows": int(len(final_df)),
    }
    save_json(summary, run_dir / "summary.json")

    LOGGER.info("=" * 80)
    LOGGER.info("CSD testing completed successfully")
    LOGGER.info("All outputs saved in: %s", run_dir)
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
