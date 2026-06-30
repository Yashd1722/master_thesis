"""
src/preprocess_bury_data.py

Preprocesses Bury et al. (2021) synthetic time series into .npz files
ready for model training and evaluation.

Class ordering (canonical, from Bury's label CSV):
    fold=0, hopf=1, transcritical=2, null=3

Run once before any training:
    python src/preprocess_bury_data.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Canonical class ordering matches Bury et al. (2021) exactly.
# Import from constants so the whole codebase agrees on one definition.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.constants import CLASS_NAMES, NULL_IDX

LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
# {"fold": 0, "hopf": 1, "transcritical": 2, "null": 3}


def preprocess_dataset(raw_dir: str, out_dir: str, ts_length: int):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Processing ts_{ts_length} ---")

    # Bury's data lives in <dataset>/combined/ with three files:
    #   labels.csv  (sequence_ID, class_label)
    #   groups.csv  (sequence_ID, dataset_ID: 1=train 2=val 3=test)
    #   output_resids/<N>.csv  (one file per time series, last column = Residuals)
    combined_dir = raw_dir / "combined"
    labels_path  = combined_dir / "labels.csv"
    groups_path  = combined_dir / "groups.csv"
    ts_dir       = combined_dir / "output_resids"

    if not (labels_path.exists() and groups_path.exists() and ts_dir.exists()):
        print(f"[ERROR] Missing required files in {raw_dir}")
        print(f"  Expected: {labels_path}, {groups_path}, and {ts_dir}/")
        return

    labels_df = pd.read_csv(labels_path)
    groups_df = pd.read_csv(groups_path)

    # Force sequence_ID to string so merges are reliable across CSV variants.
    labels_df["sequence_ID"] = labels_df["sequence_ID"].astype(str).str.strip()
    groups_df["sequence_ID"] = groups_df["sequence_ID"].astype(str).str.strip()

    df = pd.merge(labels_df, groups_df, on="sequence_ID", how="inner")

    # Map string labels → integers using the canonical ordering.
    # Raise immediately on any unknown label — never silently assign 0.
    if df["class_label"].dtype == object:
        raw_labels = df["class_label"].astype(str).str.lower().str.strip()
        unknown    = set(raw_labels.unique()) - set(LABEL_MAP.keys())
        if unknown:
            raise ValueError(
                f"Unknown class labels in {labels_path}: {unknown}\n"
                f"Expected one of: {list(LABEL_MAP.keys())}"
            )
        df["label_int"] = raw_labels.map(LABEL_MAP).astype(int)
    else:
        df["label_int"] = df["class_label"].astype(int)

    def map_split(g):
        g_str = str(g).strip()
        if g_str == "1": return "train"
        if g_str == "2": return "val"
        if g_str == "3": return "test"
        return "unknown"

    df["split"] = df["dataset_ID"].map(map_split)

    label_lookup = dict(zip(df["sequence_ID"], df["label_int"]))
    split_lookup = dict(zip(df["sequence_ID"], df["split"]))

    csv_files = sorted(ts_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} time series CSVs in {ts_dir.name}. Loading...")

    data = {"train": {"X": [], "y": []},
            "val":   {"X": [], "y": []},
            "test":  {"X": [], "y": []}}

    for f in tqdm(csv_files, desc=f"ts_{ts_length}"):
        # Extract the integer ID from filenames like "resids1.csv", "resids_001.csv".
        match = re.search(r"\d+", f.stem)
        if not match:
            continue
        file_id = match.group()

        label = label_lookup.get(file_id)
        split = split_lookup.get(file_id)

        if label is None or split not in data:
            continue

        try:
            # Bury's CSVs have a header row; the residuals are always the last column.
            df_ts = pd.read_csv(f)
            ts    = df_ts.iloc[:, -1].values.astype(np.float32)
            if len(ts) == ts_length:
                data[split]["X"].append(ts)
                data[split]["y"].append(label)
        except Exception:
            continue

    # Save one compressed .npz per split.
    for split_name in ["train", "val", "test"]:
        if not data[split_name]["X"]:
            print(f"[WARNING] No data for split '{split_name}'")
            continue

        X_split = np.array(data[split_name]["X"], dtype=np.float32)
        y_split = np.array(data[split_name]["y"], dtype=np.int64)

        # Add channel dimension for PyTorch Conv1D/LSTM: (N, 1, L)
        X_split = np.expand_dims(X_split, axis=1)

        out_path = out_dir / f"{split_name}_{ts_length}.npz"
        np.savez_compressed(out_path, X=X_split, y=y_split)
        print(f"  Saved {split_name}: X={X_split.shape}, y={y_split.shape}")
        print(f"  Class counts: {dict(zip(*np.unique(y_split, return_counts=True)))}")

    _verify_null_index(out_dir, ts_length)


def _verify_null_index(out_dir: Path, ts_length: int):
    """Assert that the 'null' class sits at index NULL_IDX in the saved data."""
    test_path = out_dir / f"test_{ts_length}.npz"
    if not test_path.exists():
        return
    data  = np.load(test_path)
    y     = data["y"]
    counts = np.bincount(y, minlength=len(CLASS_NAMES))
    print(f"\n  Verification (test_{ts_length}.npz) — class counts by index:")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"    idx {idx} ({name}): {counts[idx]}")
    assert counts[NULL_IDX] > 0, (
        f"null class (idx={NULL_IDX}) has 0 samples — label map is wrong!"
    )
    print(f"  OK: null is at index {NULL_IDX} as expected.")


if __name__ == "__main__":
    preprocess_dataset("dataset/ts_500",  "dataset/processed", ts_length=500)
    preprocess_dataset("dataset/ts_1500", "dataset/processed", ts_length=1500)
    print("\nPreprocessing complete. Run training next.")
