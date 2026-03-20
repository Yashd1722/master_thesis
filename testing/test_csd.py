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
from testing.null_model import generate_null_surrogates
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
    parser = argparse.ArgumentParser(description="Compute CSD metrics for synthetic or empirical data.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--window_frac", type=float, default=0.5)
    parser.add_argument("--min_window", type=int, default=20)
    parser.add_argument("--feature_mode", choices=["first", "mean"], default="first")
    parser.add_argument("--fixed_length", type=int, default=None)
    parser.add_argument("--length_mode", choices=["last", "first"], default="last")
    parser.add_argument("--generate_null", action="store_true")
    parser.add_argument("--n_null", type=int, default=100)
    parser.add_argument("--null_fit_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_name = build_run_name(args.model, args.dataset, args.metric)
    paths = get_test_paths(run_name)
    logger = setup_logger(paths["csd_log"], logger_name=f"testing.csd.{run_name}")

    logger.info(f"Computing CSD metrics for {args.dataset}")

    dataset_obj = load_dataset(args.dataset)
    samples = extract_samples(dataset_obj)
    logger.info(f"Loaded {len(samples)} samples.")

    is_synthetic = args.dataset in {"ts_500", "ts_1500"}
    observed_rows, observed_rolling_rows = [], []
    null_rows = [] # combined null results for simplicity

    for sample_idx, (sample_name, sample_x) in enumerate(samples):
        try:
            if is_synthetic and args.fixed_length:
                sample_x = enforce_fixed_sequence_length(sample_x, args.fixed_length, args.length_mode)

            signal = sample_to_1d(sample_x, feature_mode=args.feature_mode)
            scores, rolling_df = _compute_one_signal(signal, args.window_frac, args.min_window)

            observed_rows.append({"file_name": sample_name, "dataset_name": args.dataset, "sample_index": sample_idx, "signal_length": len(signal), **scores})

            if rolling_df is not None and not rolling_df.empty:
                rolling_df.insert(0, "file_name", sample_name)
                observed_rolling_rows.append(rolling_df)

            if args.generate_null:
                # Note: this will fail if testing.null_model is missing
                surrogates, fit = generate_null_surrogates(signal, n_surrogates=args.n_null, fit_fraction=args.null_fit_fraction, seed=args.seed + sample_idx)
                for null_idx, null_signal in enumerate(surrogates):
                    null_scores, _ = _compute_one_signal(null_signal, args.window_frac, args.min_window)
                    null_rows.append({"file_name": sample_name, "null_id": null_idx, **null_scores})

        except Exception as exc:
            logger.error(f"Failed on sample '{sample_name}': {exc}")

    if not observed_rows:
        raise RuntimeError("No CSD results produced.")

    save_csv(pd.DataFrame(observed_rows), paths["csd_dir"] / "csd_scores.csv")
    if observed_rolling_rows:
        save_csv(pd.concat(observed_rolling_rows, ignore_index=True), paths["csd_dir"] / "rolling_statistics.csv")

    if null_rows:
        save_csv(pd.DataFrame(null_rows), paths["null_scores_dir"] / "null_csd_scores.csv")

    logger.info(f"Saved CSD results to {paths['csd_dir']}")


if __name__ == "__main__":
    main()
