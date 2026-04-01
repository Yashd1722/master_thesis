"""
src/build_cache.py
==================
Run ONCE before training to eliminate the I/O bottleneck.

Problem:
    dataset_loader.py reads 475,000 individual CSV files per epoch.
    At 0.3ms per file = 2.4 min per epoch, GPU sits idle.

Solution:
    Load all CSVs once → apply normalisation → store in a single
    numpy memmap file. Training then reads from RAM not disk.
    Expected speedup: 2.6 min/epoch → 5-15 sec/epoch.

Output files:
    dataset/ts_500/combined/cache_residuals.npy   shape (N, 500)
    dataset/ts_500/combined/cache_labels.npy      shape (N,)
    dataset/ts_1500/combined/cache_residuals.npy  shape (N, 1500)
    dataset/ts_1500/combined/cache_labels.npy     shape (N,)

Usage:
    python src/build_cache.py --dataset ts_500
    python src/build_cache.py --dataset ts_1500
    python src/build_cache.py --dataset both      # build both

Run time: ~5-10 minutes once. Training is then fast forever.
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def load_config(config_path: str = "config.yaml") -> dict:
    candidates = [
        Path(config_path),
        REPO_ROOT / "config.yaml",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("config.yaml not found")


def build_cache(dataset_name: str, cfg: dict) -> None:
    """
    Build numpy cache for one dataset (ts_500 or ts_1500).

    Steps:
      1. Read labels.csv to get sequence_ID → class_label
      2. Sort residual files numerically
      3. Load each CSV, extract Residuals column, normalise
      4. Stack into single numpy array
      5. Save as .npy files
    """
    ds_cfg   = cfg["datasets"][dataset_name]
    ts_len   = ds_cfg["ts_length"]
    path_key = ds_cfg["path_key"]
    base_dir = REPO_ROOT / cfg["paths"][path_key] / "combined"

    labels_path = base_dir / "labels.csv"
    resids_dir  = base_dir / "output_resids"
    cache_X     = base_dir / "cache_residuals.npy"
    cache_y     = base_dir / "cache_labels.npy"

    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found: {labels_path}")
    if not resids_dir.exists():
        raise FileNotFoundError(f"output_resids/ not found: {resids_dir}")

    # ── Check if cache already exists ─────────────────────────────────────────
    if cache_X.exists() and cache_y.exists():
        X = np.load(cache_X, mmap_mode="r")
        y = np.load(cache_y)
        print(f"  Cache already exists: {dataset_name}")
        print(f"    X shape: {X.shape}  y shape: {y.shape}")
        ans = input("  Rebuild? [y/N]: ").strip().lower()
        if ans != "y":
            print("  Skipping.")
            return

    # ── Load labels ───────────────────────────────────────────────────────────
    labels_df = (
        pd.read_csv(labels_path)
        .sort_values("sequence_ID")
        .reset_index(drop=True)
    )

    # ── Discover + sort residual files ────────────────────────────────────────
    resid_files = sorted(
        resids_dir.glob("resids*.csv"),
        key=lambda p: int(p.stem.replace("resids", ""))
    )

    n = min(len(resid_files), len(labels_df))
    print(f"\n  Dataset  : {dataset_name}")
    print(f"  ts_len   : {ts_len}")
    print(f"  N files  : {n}")
    print(f"  Building cache (this takes ~5-10 min) ...")

    # ── Allocate output arrays ────────────────────────────────────────────────
    X = np.zeros((n, ts_len), dtype=np.float32)
    y = labels_df["class_label"].values[:n].astype(np.int64)

    t0 = time.time()
    errors = 0

    for i, fpath in enumerate(resid_files[:n]):
        try:
            df        = pd.read_csv(fpath)
            residuals = df["Residuals"].values.astype(np.float32)

            # Normalise by mean absolute value (Bury 2021)
            denom = np.mean(np.abs(residuals))
            if denom > 1e-10:
                residuals = residuals / denom

            # Crop or pad to ts_len (should always be exact length)
            curr_len = len(residuals)
            if curr_len >= ts_len:
                X[i] = residuals[-ts_len:]
            else:
                X[i, -curr_len:] = residuals

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [warn] Error loading {fpath.name}: {e}")

        # Progress every 50k files
        if (i + 1) % 50000 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (n - i - 1) / rate / 60
            print(f"  {i+1:>6}/{n}  |  {rate:.0f} files/sec  |  ETA {eta:.1f} min")

    elapsed = time.time() - t0
    print(f"\n  Loaded {n} files in {elapsed/60:.1f} min  ({errors} errors)")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"  Saving {cache_X.name} ...")
    np.save(cache_X, X)
    print(f"  Saving {cache_y.name} ...")
    np.save(cache_y, y)

    size_mb = (cache_X.stat().st_size + cache_y.stat().st_size) / 1e6
    print(f"\n  Cache saved: {size_mb:.0f} MB")
    print(f"    {cache_X}")
    print(f"    {cache_y}")
    print(f"\n  Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    class_names = cfg["datasets"][dataset_name]["class_names"]
    for cls, cnt in zip(unique, counts):
        name = class_names[cls] if cls < len(class_names) else str(cls)
        print(f"    {cls} ({name:15s}): {cnt:>6}  ({cnt/n*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-build numpy cache from residual CSVs."
    )
    parser.add_argument(
        "--dataset", type=str, default="both",
        choices=["ts_500", "ts_1500", "both"],
        help="Which dataset to cache (default: both)"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    datasets = (
        ["ts_500", "ts_1500"]
        if args.dataset == "both"
        else [args.dataset]
    )

    for ds in datasets:
        build_cache(ds, cfg)

    print("\nCache building complete.")
    print("Now restart training — expect ~5-15 sec/epoch instead of 2.6 min.")


if __name__ == "__main__":
    main()
