from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.csd_metrics import compute_csd_scores
from src.dataset_loader import load_dataset
from testing.null_models import generate_null_surrogates
from testing.testing_utils import (
    build_run_name,
    enforce_fixed_sequence_length,
    extract_samples,
    get_test_paths,
    sample_to_1d,
    save_csv,
    save_json,
    setup_logger,
)


def _compute_one_signal(signal: np.ndarray, window_frac: float, min_window: int):
    scores, rolling_df = compute_csd_scores(
        signal=signal,
        window_frac=window_frac,
        min_window=min_window,
    )
    return scores, rolling_df


def main():
    parser = argparse.ArgumentParser(
        description="Compute CSD metrics for synthetic or empirical data with optional null generation."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name for src.dataset_loader.load_dataset()")
    parser.add_argument("--model", required=True, help="Model name for run naming")
    parser.add_argument("--metric", required=True, help="Metric name for run naming")

    # separate rolling window fraction for CSD
    parser.add_argument("--window_frac", type=float, default=0.5, help="Rolling window fraction")
    parser.add_argument("--min_window", type=int, default=20, help="Minimum rolling window size")

    parser.add_argument(
        "--feature_mode",
        choices=["first", "mean"],
        default="first",
        help="For multivariate input: first feature or mean across features",
    )

    # synthetic fixed input length
    parser.add_argument(
        "--fixed_length",
        type=int,
        default=None,
        help="Synthetic only: force sequences to a fixed length, e.g. 500 or 1500.",
    )
    parser.add_argument(
        "--length_mode",
        choices=["last", "first"],
        default="last",
        help="How to crop if a synthetic sequence is longer than fixed_length.",
    )

    # optional null generation
    parser.add_argument("--generate_null", action="store_true", help="Generate AR(1) null surrogates")
    parser.add_argument("--n_null", type=int, default=100, help="Number of null surrogates per sample")
    parser.add_argument("--null_fit_fraction", type=float, default=0.2, help="Fit AR(1) on first fraction of signal")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for null generation")

    args = parser.parse_args()

    run_name = build_run_name(args.model, args.dataset, args.metric)
    paths = get_test_paths(run_name)
    logger = setup_logger(paths["csd_log"], logger_name=f"testing.csd.{run_name}")

    logger.info("Run name: %s", run_name)
    logger.info("Dataset: %s", args.dataset)
    logger.info("window_frac=%s | min_window=%s", args.window_frac, args.min_window)
    logger.info("feature_mode=%s", args.feature_mode)
    logger.info("fixed_length=%s | length_mode=%s", args.fixed_length, args.length_mode)
    logger.info("generate_null=%s", args.generate_null)

    dataset_obj = load_dataset(args.dataset)
    samples = extract_samples(dataset_obj)
    logger.info("Loaded %d samples.", len(samples))

    is_synthetic = args.dataset in {"ts_500", "ts_1500"}

    observed_rows = []
    observed_rolling_rows = []

    null_fit_rows = []
    null_score_rows = []
    null_summary_rows = []
    null_rolling_rows = []

    for sample_idx, (sample_name, sample_x) in enumerate(samples):
        try:
            if is_synthetic and args.fixed_length is not None:
                sample_x = enforce_fixed_sequence_length(
                    sample_x,
                    target_length=args.fixed_length,
                    mode=args.length_mode,
                )

            signal = sample_to_1d(sample_x, feature_mode=args.feature_mode)

            scores, rolling_df = _compute_one_signal(
                signal=signal,
                window_frac=args.window_frac,
                min_window=args.min_window,
            )

            observed_rows.append(
                {
                    "file_name": sample_name,
                    "dataset_name": args.dataset,
                    "model_name": args.model,
                    "metric_name": args.metric,
                    "series_kind": "observed",
                    "sample_index": sample_idx,
                    "signal_length": len(signal),
                    **scores,
                }
            )

            if rolling_df is not None and not rolling_df.empty:
                tmp = rolling_df.copy()
                tmp.insert(0, "file_name", sample_name)
                tmp.insert(1, "dataset_name", args.dataset)
                tmp.insert(2, "series_kind", "observed")
                observed_rolling_rows.append(tmp)

            if args.generate_null:
                surrogates, fit = generate_null_surrogates(
                    signal=signal,
                    n_surrogates=args.n_null,
                    fit_fraction=args.null_fit_fraction,
                    seed=args.seed + sample_idx * 1000,
                )

                null_fit_rows.append(
                    {
                        "file_name": sample_name,
                        "dataset_name": args.dataset,
                        "fit_fraction": args.null_fit_fraction,
                        "n_null": args.n_null,
                        "phi": fit.phi,
                        "sigma": fit.sigma,
                        "mean": fit.mean,
                        "n_fit": fit.n_fit,
                        "signal_length": len(signal),
                    }
                )

                per_sample_null_metrics = []
                for null_idx, null_signal in enumerate(surrogates):
                    null_scores, null_rolling_df = _compute_one_signal(
                        signal=null_signal,
                        window_frac=args.window_frac,
                        min_window=args.min_window,
                    )

                    row = {
                        "file_name": sample_name,
                        "dataset_name": args.dataset,
                        "series_kind": "null",
                        "null_id": null_idx,
                        "sample_index": sample_idx,
                        "signal_length": len(null_signal),
                        **null_scores,
                    }
                    null_score_rows.append(row)
                    per_sample_null_metrics.append(row)

                    if null_rolling_df is not None and not null_rolling_df.empty:
                        tmp_null = null_rolling_df.copy()
                        tmp_null.insert(0, "file_name", sample_name)
                        tmp_null.insert(1, "dataset_name", args.dataset)
                        tmp_null.insert(2, "series_kind", "null")
                        tmp_null.insert(3, "null_id", null_idx)
                        null_rolling_rows.append(tmp_null)

                per_sample_null_df = pd.DataFrame(per_sample_null_metrics)
                null_summary_rows.append(
                    {
                        "file_name": sample_name,
                        "dataset_name": args.dataset,
                        "n_null": args.n_null,
                        "observed_ktau_var": scores.get("ktau_var"),
                        "observed_ktau_ac1": scores.get("ktau_ac1"),
                        "null_mean_ktau_var": per_sample_null_df["ktau_var"].mean(),
                        "null_std_ktau_var": per_sample_null_df["ktau_var"].std(ddof=1),
                        "null_mean_ktau_ac1": per_sample_null_df["ktau_ac1"].mean(),
                        "null_std_ktau_ac1": per_sample_null_df["ktau_ac1"].std(ddof=1),
                    }
                )

        except Exception as exc:
            logger.exception("Failed on sample '%s': %s", sample_name, exc)

    if not observed_rows:
        raise RuntimeError("No observed CSD outputs were produced.")

    observed_df = pd.DataFrame(observed_rows)
    observed_rolling_df = (
        pd.concat(observed_rolling_rows, ignore_index=True) if observed_rolling_rows else pd.DataFrame()
    )

    kendall_tau_df = observed_df[
        ["file_name", "dataset_name", "series_kind", "ktau_var", "ktau_ac1"]
    ].copy()

    summary_df = pd.DataFrame(
        [
            {
                "dataset_name": args.dataset,
                "model_name": args.model,
                "metric_name": args.metric,
                "num_samples": len(observed_df),
                "mean_ktau_var": observed_df["ktau_var"].mean(),
                "mean_ktau_ac1": observed_df["ktau_ac1"].mean(),
                "feature_mode": args.feature_mode,
                "window_frac": args.window_frac,
                "min_window": args.min_window,
                "fixed_length": args.fixed_length,
                "generate_null": args.generate_null,
                "n_null": args.n_null if args.generate_null else 0,
            }
        ]
    )

    save_csv(observed_df, paths["csd_dir"] / "csd_scores.csv")
    save_csv(observed_rolling_df, paths["csd_dir"] / "rolling_statistics.csv")
    save_csv(kendall_tau_df, paths["csd_dir"] / "kendall_tau_results.csv")
    save_csv(summary_df, paths["csd_dir"] / "csd_summary.csv")

    if args.generate_null:
        null_fit_df = pd.DataFrame(null_fit_rows)
        null_scores_df = pd.DataFrame(null_score_rows)
        null_summary_df = pd.DataFrame(null_summary_rows)
        null_rolling_df = (
            pd.concat(null_rolling_rows, ignore_index=True) if null_rolling_rows else pd.DataFrame()
        )

        save_csv(null_fit_df, paths["null_dir"] / "null_fit_parameters.csv")
        save_csv(null_scores_df, paths["null_scores_dir"] / "null_csd_scores.csv")
        save_csv(null_summary_df, paths["null_scores_dir"] / "null_summary.csv")
        save_csv(null_rolling_df, paths["null_scores_dir"] / "null_rolling_statistics.csv")

        save_json(
            {
                "dataset": args.dataset,
                "model": args.model,
                "metric": args.metric,
                "window_frac": args.window_frac,
                "min_window": args.min_window,
                "feature_mode": args.feature_mode,
                "fixed_length": args.fixed_length,
                "length_mode": args.length_mode,
                "n_null": args.n_null,
                "null_fit_fraction": args.null_fit_fraction,
                "seed": args.seed,
            },
            paths["null_dir"] / "null_config.json",
        )

    logger.info("Saved CSD outputs to %s", paths["csd_dir"])
    logger.info("Done.")


if __name__ == "__main__":
    main()
