# testing/test_csd.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.csd_metrics import compute_csd_scores, compute_csd_with_null
from src.dataset_loader import load_dataset
from testing.testing_utils import (
    extract_samples,
    build_run_name,
    get_test_paths,
    sample_to_1d,
    save_csv,
    setup_logger,
)


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    std = np.std(x)
    if std < 1e-12:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CSD indicators on synthetic or empirical data")
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
    parser.add_argument("--window_frac", type=float, default=0.5)
    parser.add_argument("--min_window", type=int, default=20)
    parser.add_argument("--feature_mode", type=str, default="first", choices=["first", "mean"])
    parser.add_argument("--normalize", action="store_true", help="Z-score the 1D signal before CSD computation")
    parser.add_argument("--generate_null", action="store_true")
    parser.add_argument("--n_null", type=int, default=100)
    parser.add_argument("--null_fit_fraction", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    run_name = build_run_name(
        model=args.model,
        train_dataset=args.train_dataset,
        metric=args.metric,
        experiment=args.experiment,
        test_dataset=args.dataset,
    )
    paths = get_test_paths(run_name)
    logger = setup_logger(paths["csd_log"], logger_name=f"testing.csd.{run_name}")

    logger.info("Run name: %s", run_name)
    logger.info("Dataset: %s", args.dataset)
    logger.info("Train dataset: %s", args.train_dataset)
    logger.info("Model: %s", args.model)
    logger.info("Metric: %s", args.metric)
    logger.info("Experiment: %s", args.experiment)
    logger.info("Window frac: %.3f", args.window_frac)
    logger.info("Min window: %d", args.min_window)
    logger.info("Feature mode: %s", args.feature_mode)
    logger.info("Normalize signal: %s", args.normalize)
    logger.info("Generate null: %s", args.generate_null)

    dataset_obj = load_dataset(args.dataset)
    samples = extract_samples(dataset_obj)
    logger.info("Loaded %d samples", len(samples))

    csd_rows = []
    rolling_rows = []
    null_rows = []

    for sample_idx, (sample_name, sample_x, sample_y) in enumerate(samples):
        try:
            signal = sample_to_1d(sample_x, feature_mode=args.feature_mode)

            if args.normalize:
                signal = zscore_1d(signal)

            if args.generate_null:
                result = compute_csd_with_null(
                    signal=signal,
                    window_frac=args.window_frac,
                    min_window=args.min_window,
                    n_null=args.n_null,
                    null_fit_fraction=args.null_fit_fraction,
                    random_seed=args.random_seed,
                )
                scores = result.scores
                rolling_df = result.rolling_df
                null_df = result.null_summary_df
            else:
                scores, rolling_df = compute_csd_scores(
                    signal=signal,
                    window_frac=args.window_frac,
                    min_window=args.min_window,
                )
                null_df = None

            csd_row = {
                "file_name": sample_name,
                "dataset_name": args.dataset,
                "train_dataset_name": args.train_dataset,
                "model_name": args.model,
                "metric_name": args.metric,
                "experiment": args.experiment,
                "sample_index": sample_idx,
                "true_class_idx": None if sample_y is None else int(sample_y),
                **scores,
            }
            csd_rows.append(csd_row)

            if rolling_df is not None and not rolling_df.empty:
                tmp_roll = rolling_df.copy()
                tmp_roll.insert(0, "file_name", sample_name)
                tmp_roll.insert(1, "dataset_name", args.dataset)
                tmp_roll.insert(2, "sample_index", sample_idx)
                rolling_rows.append(tmp_roll)

            if null_df is not None and not null_df.empty:
                tmp_null = null_df.copy()
                tmp_null.insert(0, "file_name", sample_name)
                tmp_null.insert(1, "dataset_name", args.dataset)
                tmp_null.insert(2, "sample_index", sample_idx)
                null_rows.append(tmp_null)

            logger.info(
                "Sample=%s | len=%d | ktau_var=%.4f | ktau_ac1=%.4f",
                sample_name,
                len(signal),
                float(scores.get("ktau_var", np.nan)),
                float(scores.get("ktau_ac1", np.nan)),
            )

        except Exception as exc:
            logger.exception("Failed on sample '%s': %s", sample_name, exc)

    if not csd_rows:
        raise RuntimeError("No CSD outputs were produced.")

    csd_df = pd.DataFrame(csd_rows)
    rolling_df = pd.concat(rolling_rows, ignore_index=True) if rolling_rows else pd.DataFrame()
    null_df = pd.concat(null_rows, ignore_index=True) if null_rows else pd.DataFrame()

    save_csv(csd_df, paths["csd_dir"] / "csd_scores.csv")
    save_csv(rolling_df, paths["csd_dir"] / "rolling_statistics.csv")

    if not null_df.empty:
        save_csv(null_df, paths["csd_dir"] / "null_summary.csv")

    logger.info("Saved CSD scores to %s", paths["csd_dir"] / "csd_scores.csv")
    logger.info("Saved rolling statistics to %s", paths["csd_dir"] / "rolling_statistics.csv")

    if not null_df.empty:
        logger.info("Saved null summary to %s", paths["csd_dir"] / "null_summary.csv")

    logger.info("Done.")


if __name__ == "__main__":
    main()
