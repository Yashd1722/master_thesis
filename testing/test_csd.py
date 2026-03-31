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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.testing_utils import ensure_dir, load_test_dataset_for_inference, save_json, set_seed

LOGGER = logging.getLogger("testing.test_csd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute variance, lag-1 AC, and Kendall tau progressively for one dataset."
    )
    parser.add_argument("--dataset", type=str, required=True, help="ts_500, ts_1500, or pangaea_923197")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--rolling-window-frac", type=float, default=0.50)
    parser.add_argument("--progressive-start-frac", type=float, default=0.60)
    parser.add_argument("--progressive-end-frac", type=float, default=1.00)
    parser.add_argument("--progressive-num-steps", type=int, default=40)
    parser.add_argument("--min-prefix-len", type=int, default=64)

    parser.add_argument(
        "--gaussian-smooth-sigma",
        type=float,
        default=-1.0,
        help="Use -1 for dataset-specific automatic sigma.",
    )

    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-series-csv", action="store_true")
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


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


def get_csd_output_dirs(run_dir: Path) -> Dict[str, Path]:
    logs_dir = run_dir / "logs"
    tables_dir = run_dir / "tables"
    plots_dir = run_dir / "plots"
    indicator_dir = plots_dir / "indicator_curves"
    tau_dir = plots_dir / "kendall_tau_curves"
    overview_dir = plots_dir / "overview"
    panel_dir = plots_dir / "per_series_panels"
    series_csv_dir = run_dir / "series_predictions"
    series_signal_dir = run_dir / "series_signals"

    for path in [
        logs_dir,
        tables_dir,
        plots_dir,
        indicator_dir,
        tau_dir,
        overview_dir,
        panel_dir,
        series_csv_dir,
        series_signal_dir,
    ]:
        ensure_dir(path)

    return {
        "logs_dir": logs_dir,
        "tables_dir": tables_dir,
        "plots_dir": plots_dir,
        "indicator_dir": indicator_dir,
        "tau_dir": tau_dir,
        "overview_dir": overview_dir,
        "panel_dir": panel_dir,
        "series_csv_dir": series_csv_dir,
        "series_signal_dir": series_signal_dir,
    }


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
    return np.convolve(x_pad, kernel, mode="valid")


def kendall_tau(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x)
    x = x[valid]

    n = len(x)
    if n < 2:
        return np.nan

    concordant = 0
    discordant = 0
    for i in range(n - 1):
        diff = x[i + 1 :] - x[i]
        concordant += int(np.sum(diff > 0))
        discordant += int(np.sum(diff < 0))

    denom = n * (n - 1) / 2
    if denom == 0:
        return np.nan
    return float((concordant - discordant) / denom)


def compute_variance(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) < 2:
        return np.nan
    return float(np.var(x, ddof=1))


def compute_lag1_autocorrelation(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) < 3:
        return np.nan
    if np.std(x) == 0:
        return np.nan

    x0 = x[:-1]
    x1 = x[1:]

    x0_mean = np.mean(x0)
    x1_mean = np.mean(x1)
    numerator = np.sum((x0 - x0_mean) * (x1 - x1_mean))
    denominator = np.sqrt(np.sum((x0 - x0_mean) ** 2) * np.sum((x1 - x1_mean) ** 2))

    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def rolling_indicator_values(x: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = len(x)
    if n < window_size or window_size < 3:
        return np.array([]), np.array([]), np.array([])

    end_indices: List[int] = []
    variances: List[float] = []
    lag1_values: List[float] = []

    for end_idx in range(window_size, n + 1):
        window = x[end_idx - window_size : end_idx]
        end_indices.append(end_idx - 1)
        variances.append(compute_variance(window))
        lag1_values.append(compute_lag1_autocorrelation(window))

    return (
        np.asarray(end_indices, dtype=np.int64),
        np.asarray(variances, dtype=np.float64),
        np.asarray(lag1_values, dtype=np.float64),
    )


def progressive_reveal_lengths(
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


def safe_series_name(series_id: str) -> str:
    return str(series_id).replace("/", "_").replace("\\", "_").replace(" ", "_")


def resolve_sigma(dataset_name: str, user_sigma: float, metadata: Optional[Dict[str, Any]]) -> float:
    if user_sigma >= 0:
        return float(user_sigma)

    if isinstance(metadata, dict) and "default_gaussian_sigma" in metadata:
        try:
            return float(metadata["default_gaussian_sigma"])
        except Exception:
            pass

    if str(dataset_name).lower() == "pangaea_923197":
        return 20.0

    return 0.0


def build_series_signal_frame(
    raw_signal: np.ndarray,
    sigma: float,
    time_index: Optional[np.ndarray] = None,
    transition_index: Optional[int] = None,
) -> pd.DataFrame:
    raw_signal = np.asarray(raw_signal, dtype=np.float64).reshape(-1)
    local_index = np.arange(len(raw_signal), dtype=np.int64)
    smooth_signal = gaussian_smooth_1d(raw_signal, sigma=sigma)
    residual_signal = raw_signal - smooth_signal

    df = pd.DataFrame(
        {
            "local_index": local_index,
            "raw_signal": raw_signal,
            "smooth_signal": smooth_signal,
            "residual_signal": residual_signal,
        }
    )

    if time_index is not None:
        time_index = np.asarray(time_index, dtype=np.float64).reshape(-1)
        if len(time_index) == len(raw_signal):
            df["time_value"] = time_index

    if transition_index is not None:
        df["transition_index"] = int(transition_index)

    return df


def evaluate_single_series_csd(
    raw_signal: np.ndarray,
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
    raw_signal = np.asarray(raw_signal, dtype=np.float64).reshape(-1)
    total_length = len(raw_signal)

    reveal_lengths = progressive_reveal_lengths(
        series_length=total_length,
        progressive_start_frac=progressive_start_frac,
        progressive_end_frac=progressive_end_frac,
        progressive_num_steps=progressive_num_steps,
        min_prefix_len=min_prefix_len,
    )
    if not reveal_lengths:
        return []

    records: List[Dict[str, Any]] = []
    for step_idx, reveal_len in enumerate(reveal_lengths):
        prefix_raw = raw_signal[:reveal_len]
        prefix_smooth = gaussian_smooth_1d(prefix_raw, sigma=gaussian_smooth_sigma)
        prefix_residual = prefix_raw - prefix_smooth

        window_size = max(3, int(round(reveal_len * rolling_window_frac)))
        _, rolling_var, rolling_ac = rolling_indicator_values(prefix_residual, window_size=window_size)

        variance_tau = kendall_tau(rolling_var) if len(rolling_var) >= 2 else np.nan
        lag1_tau = kendall_tau(rolling_ac) if len(rolling_ac) >= 2 else np.nan

        row: Dict[str, Any] = {
            "series_id": str(series_id),
            "step_idx": int(step_idx),
            "prefix_length": int(reveal_len),
            "reveal_index": int(reveal_len - 1),
            "reveal_fraction": float(reveal_len / max(total_length, 1)),
            "series_length": int(total_length),
            "rolling_window_size": int(window_size),
            "variance": float(rolling_var[-1]) if len(rolling_var) > 0 else np.nan,
            "lag1_ac": float(rolling_ac[-1]) if len(rolling_ac) > 0 else np.nan,
            "variance_tau": float(variance_tau) if np.isfinite(variance_tau) else np.nan,
            "lag1_ac_tau": float(lag1_tau) if np.isfinite(lag1_tau) else np.nan,
            "transition_index": None if transition_index is None else int(transition_index),
        }
        if y_true is not None:
            row["y_true"] = int(y_true)

        records.append(row)

    return records


def plot_indicator_curves(
    series_df: pd.DataFrame,
    save_path: Path,
    transition_index: Optional[float] = None,
    title: Optional[str] = None,
) -> None:
    if series_df.empty:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(series_df["reveal_index"], series_df["variance"], label="Variance")
    plt.plot(series_df["reveal_index"], series_df["lag1_ac"], label="Lag-1 AC")

    if transition_index is not None:
        plt.axvline(x=transition_index, linestyle="--", linewidth=1.5, label="Transition")

    plt.xlabel("Local index")
    plt.ylabel("Indicator value")
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

    plt.figure(figsize=(10, 5))
    plt.plot(series_df["reveal_index"], series_df["variance_tau"], label="Variance tau")
    plt.plot(series_df["reveal_index"], series_df["lag1_ac_tau"], label="Lag-1 AC tau")

    if transition_index is not None:
        plt.axvline(x=transition_index, linestyle="--", linewidth=1.5, label="Transition")

    plt.xlabel("Local index")
    plt.ylabel("Kendall tau")
    plt.title(title or "Kendall tau curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_csd_overview(
    signal_df: pd.DataFrame,
    csd_df: pd.DataFrame,
    save_path: Path,
    transition_index: Optional[float] = None,
    title: Optional[str] = None,
) -> None:
    if signal_df.empty or csd_df.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(signal_df["local_index"], signal_df["raw_signal"], color="black", linewidth=1.0, label="Raw")
    axes[0].plot(signal_df["local_index"], signal_df["smooth_signal"], color="gray", linewidth=1.5, label="Smooth")
    axes[0].set_ylabel("Signal")
    axes[0].set_title(title or "Series overview")
    axes[0].legend(loc="upper right")

    axes[1].plot(csd_df["reveal_index"], csd_df["variance"], label="Variance")
    axes[1].plot(csd_df["reveal_index"], csd_df["lag1_ac"], label="Lag-1 AC")
    axes[1].set_ylabel("Indicator")
    axes[1].legend(loc="upper right")

    axes[2].plot(csd_df["reveal_index"], csd_df["variance_tau"], label="Variance tau")
    axes[2].plot(csd_df["reveal_index"], csd_df["lag1_ac_tau"], label="Lag-1 AC tau")
    axes[2].set_xlabel("Local index")
    axes[2].set_ylabel("Kendall tau")
    axes[2].legend(loc="upper right")

    if transition_index is not None:
        for ax in axes:
            ax.axvline(x=transition_index, linestyle="--", linewidth=1.2)
            x_max = float(signal_df["local_index"].max())
            if transition_index < x_max:
                ax.axvspan(transition_index, x_max, color="gray", alpha=0.08)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_final_tau_distribution(df_final: pd.DataFrame, save_path: Path) -> None:
    if df_final.empty:
        return

    plt.figure(figsize=(8, 4))
    values_1 = df_final["final_variance_tau"].dropna().values if "final_variance_tau" in df_final else np.array([])
    values_2 = df_final["final_lag1_ac_tau"].dropna().values if "final_lag1_ac_tau" in df_final else np.array([])

    if len(values_1) > 0:
        plt.hist(values_1, bins=20, alpha=0.6, label="Final variance tau")
    if len(values_2) > 0:
        plt.hist(values_2, bins=20, alpha=0.6, label="Final lag-1 AC tau")

    plt.xlabel("Kendall tau")
    plt.ylabel("Count")
    plt.title("Final Kendall tau distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def run_csd_evaluation(
    dataset_name: str,
    series_dataset: Any,
    rolling_window_frac: float,
    progressive_start_frac: float,
    progressive_end_frac: float,
    progressive_num_steps: int,
    min_prefix_len: int,
    user_sigma: float,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    all_records: List[Dict[str, Any]] = []
    series_signal_frames: Dict[str, pd.DataFrame] = {}

    total_series = len(series_dataset.series_items)
    for idx, item in enumerate(series_dataset.series_items, start=1):
        series_id = str(item["series_id"])
        raw_signal = item.get("raw_signal")
        if raw_signal is None:
            signal = np.asarray(item["signal"], dtype=np.float64)
            raw_signal = signal[:, 0] if signal.ndim == 2 else signal.reshape(-1)

        metadata = item.get("metadata", {})
        sigma = resolve_sigma(dataset_name=dataset_name, user_sigma=user_sigma, metadata=metadata)
        transition_index = item.get("transition_index", None)
        y_true = item.get("label", None)
        time_index = item.get("time_index", None)

        LOGGER.info(
            "CSD evaluation [%d/%d] | series_id=%s | length=%d | sigma=%.3f",
            idx,
            total_series,
            series_id,
            len(raw_signal),
            sigma,
        )

        signal_df = build_series_signal_frame(
            raw_signal=raw_signal,
            sigma=sigma,
            time_index=time_index,
            transition_index=transition_index,
        )
        series_signal_frames[series_id] = signal_df

        records = evaluate_single_series_csd(
            raw_signal=raw_signal,
            series_id=series_id,
            y_true=y_true,
            transition_index=transition_index,
            rolling_window_frac=rolling_window_frac,
            progressive_start_frac=progressive_start_frac,
            progressive_end_frac=progressive_end_frac,
            progressive_num_steps=progressive_num_steps,
            min_prefix_len=min_prefix_len,
            gaussian_smooth_sigma=sigma,
        )
        if not records:
            LOGGER.warning("No CSD records created for series_id=%s", series_id)
            continue

        all_records.extend(records)

    return pd.DataFrame(all_records), series_signal_frames


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
    LOGGER.info("Smooth sigma arg    : %.4f", args.gaussian_smooth_sigma)
    LOGGER.info("Run directory       : %s", run_dir)

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

    loaded = load_test_dataset_for_inference(
        dataset_name=args.dataset,
        split=args.split,
        input_length=500,
        num_classes=3,
    )
    series_dataset = loaded["series_dataset"]

    csd_df, series_signal_frames = run_csd_evaluation(
        dataset_name=args.dataset,
        series_dataset=series_dataset,
        rolling_window_frac=args.rolling_window_frac,
        progressive_start_frac=args.progressive_start_frac,
        progressive_end_frac=args.progressive_end_frac,
        progressive_num_steps=args.progressive_num_steps,
        min_prefix_len=args.min_prefix_len,
        user_sigma=args.gaussian_smooth_sigma,
    )

    if csd_df.empty:
        LOGGER.warning("No CSD outputs were generated.")
        return

    combined_csv_path = run_dir / "all_series_csd_predictions.csv"
    csd_df.to_csv(combined_csv_path, index=False)
    csd_df.to_csv(output_dirs["tables_dir"] / "all_series_csd_predictions.csv", index=False)

    final_df = (
        csd_df.sort_values(["series_id", "reveal_index"])
        .groupby("series_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
        .rename(
            columns={
                "variance_tau": "final_variance_tau",
                "lag1_ac_tau": "final_lag1_ac_tau",
                "variance": "final_variance",
                "lag1_ac": "final_lag1_ac",
            }
        )
    )

    final_csv_path = run_dir / "final_series_csd_predictions.csv"
    final_df.to_csv(final_csv_path, index=False)
    final_df.to_csv(output_dirs["tables_dir"] / "final_series_csd_predictions.csv", index=False)

    for series_id, signal_df in series_signal_frames.items():
        safe_name = safe_series_name(series_id)
        signal_df.to_csv(output_dirs["series_signal_dir"] / f"{safe_name}.csv", index=False)

    if args.save_series_csv:
        for series_id, series_df in csd_df.groupby("series_id", sort=True):
            safe_name = safe_series_name(series_id)
            series_df.sort_values("reveal_index").to_csv(
                output_dirs["series_csv_dir"] / f"{safe_name}.csv",
                index=False,
            )

    if args.make_plots:
        for series_id, series_df in csd_df.groupby("series_id", sort=True):
            series_df = series_df.sort_values("reveal_index").reset_index(drop=True)
            signal_df = series_signal_frames.get(series_id, pd.DataFrame())
            safe_name = safe_series_name(series_id)

            transition_index = None
            valid_transition = series_df["transition_index"].dropna().values if "transition_index" in series_df else []
            if len(valid_transition) > 0:
                transition_index = float(valid_transition[0])

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
            plot_csd_overview(
                signal_df=signal_df,
                csd_df=series_df,
                save_path=output_dirs["panel_dir"] / f"{safe_name}.png",
                transition_index=transition_index,
                title=f"Series {series_id}",
            )

        plot_final_tau_distribution(
            df_final=final_df,
            save_path=output_dirs["overview_dir"] / "final_tau_distribution.png",
        )

    summary = {
        "num_series": int(csd_df["series_id"].nunique()),
        "num_rows": int(len(csd_df)),
        "num_final_rows": int(len(final_df)),
        "num_signal_csv": int(len(series_signal_frames)),
    }
    save_json(summary, run_dir / "summary.json")

    LOGGER.info("Saved CSD outputs to: %s", run_dir)
    LOGGER.info(json.dumps(summary, indent=2))
    LOGGER.info("CSD testing completed successfully")


if __name__ == "__main__":
    main()
