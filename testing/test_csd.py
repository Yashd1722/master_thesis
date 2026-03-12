# testing/test_csd.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_loader import load_dataset
from testing.testing_utils import (
    build_run_name,
    get_test_paths,
    save_csv,
    setup_logger,
)
from metrics.csd_metrics import compute_csd_scores   # single import


def _to_numpy_1d(x: Any):
    """Convert sample to a 1D numpy array."""
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        import numpy as np
        arr = np.asarray(x)

    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2:
        # Use first feature column for empirical signal
        return arr[:, 0].astype(float)
    if arr.ndim == 3:
        # If batch dimension exists, use first batch and first feature
        return arr[0, :, 0].astype(float)

    raise ValueError(f"Unsupported sample shape for CSD: {arr.shape}")


def _extract_samples(dataset_obj: Any) -> list[tuple[str, Any]]:
    """Extract (name, sample) pairs from the dataset loader output."""
    if dataset_obj is None:
        raise ValueError("load_dataset returned None.")

    # Handle training loader style: (sequences, labels, feature_names)
    if isinstance(dataset_obj, tuple) and len(dataset_obj) == 3:
        sequences, _, _ = dataset_obj
        if sequences is None or not sequences:
            raise ValueError("No sequences in dataset")
        return [(f"sample_{i}", seq) for i, seq in enumerate(sequences)]

    # Dict style
    if isinstance(dataset_obj, dict):
        for key in ('X', 'x', 'data', 'features'):
            if key in dataset_obj:
                X = dataset_obj[key]
                break
        else:
            raise KeyError("Dataset dict missing one of: X, x, data, features")

        names = dataset_obj.get('names') or dataset_obj.get('file_names') or dataset_obj.get('filenames')
        if names is None:
            names = [f"sample_{i}" for i in range(len(X))]

        if len(names) != len(X):
            raise ValueError("Dataset names length does not match sample count.")

        return [(str(n), X[i]) for i, n in enumerate(names)]

    # List/tuple of samples
    if isinstance(dataset_obj, (list, tuple)):
        out = []
        for i, item in enumerate(dataset_obj):
            if isinstance(item, dict):
                name = str(item.get('name') or item.get('file_name') or item.get('filename') or f"sample_{i}")
                for key in ('x', 'X', 'data', 'features'):
                    if key in item:
                        sample = item[key]
                        break
                else:
                    raise KeyError(f"No data key in sample dict at index {i}")
            elif isinstance(item, (list, tuple)):
                if len(item) == 0:
                    raise ValueError(f"Empty sample tuple/list at index {i}.")
                if len(item) >= 2 and isinstance(item[0], str):
                    name, sample = item[0], item[1]
                else:
                    name, sample = f"sample_{i}", item[0]
            else:
                name, sample = f"sample_{i}", item
            out.append((name, sample))

        if not out:
            raise ValueError("No samples extracted")
        return out

    raise TypeError(f"Unsupported dataset type: {type(dataset_obj)}")


def main():
    parser = argparse.ArgumentParser(description="Compute CSD metrics on unlabeled empirical data.")
    parser.add_argument("--dataset", required=True, help="Dataset name for src.dataset_loader.load_dataset()")
    parser.add_argument("--model", required=True, help="Model name, only for run-name consistency")
    parser.add_argument("--metric", required=True, help="Metric name, only for run-name consistency")
    parser.add_argument("--window_frac", type=float, default=0.5, help="Rolling window fraction of series length")
    parser.add_argument("--min_window", type=int, default=20, help="Minimum rolling window size")
    args = parser.parse_args()

    run_name = build_run_name(args.model, args.dataset, args.metric)
    paths = get_test_paths(run_name)
    logger = setup_logger(paths["csd_log"], logger_name=f"testing.csd.{run_name}")

    logger.info(f"Run name: {run_name}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Window fraction: {args.window_frac} | Min window: {args.min_window}")

    logger.info("Loading empirical dataset via src.dataset_loader.load_dataset(...)")
    dataset_obj = load_dataset(args.dataset)
    samples = _extract_samples(dataset_obj)
    logger.info(f"Loaded {len(samples)} empirical samples.")

    csd_rows = []
    rolling_rows = []

    for sample_name, sample_x in samples:
        signal = _to_numpy_1d(sample_x)
        scores, rolling_df = compute_csd_scores(
            signal=signal,
            window_frac=args.window_frac,
            min_window=args.min_window,
        )

        csd_rows.append(
            {
                "file_name": sample_name,
                "dataset_name": args.dataset,
                "model_name": args.model,
                "metric_name": args.metric,
                **scores,
            }
        )

        if rolling_df is not None and not rolling_df.empty:
            tmp = rolling_df.copy()
            tmp.insert(0, "file_name", sample_name)
            tmp.insert(1, "dataset_name", args.dataset)
            rolling_rows.append(tmp)

    csd_df = pd.DataFrame(csd_rows)
    if csd_df.empty:
        raise ValueError("No CSD outputs were produced.")

    rolling_df = pd.concat(rolling_rows, ignore_index=True) if rolling_rows else pd.DataFrame()

    kendall_tau_df = csd_df[
        ["file_name", "dataset_name", "ktau_var", "ktau_ac1"]
    ].copy()

    summary_df = pd.DataFrame(
        [
            {
                "dataset_name": args.dataset,
                "model_name": args.model,
                "metric_name": args.metric,
                "num_samples": len(csd_df),
                "mean_ktau_var": csd_df["ktau_var"].mean(),
                "mean_ktau_ac1": csd_df["ktau_ac1"].mean(),
                "mean_window_size": csd_df["window_size"].mean() if "window_size" in csd_df.columns else None,
            }
        ]
    )

    save_csv(csd_df, paths["csd_dir"] / "csd_scores.csv")
    save_csv(rolling_df, paths["csd_dir"] / "rolling_statistics.csv")
    save_csv(kendall_tau_df, paths["csd_dir"] / "kendall_tau_results.csv")
    save_csv(summary_df, paths["csd_dir"] / "csd_summary.csv")

    logger.info(f"Saved: {paths['csd_dir'] / 'csd_scores.csv'}")
    logger.info(f"Saved: {paths['csd_dir'] / 'rolling_statistics.csv'}")
    logger.info(f"Saved: {paths['csd_dir'] / 'kendall_tau_results.csv'}")
    logger.info(f"Saved: {paths['csd_dir'] / 'csd_summary.csv'}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
