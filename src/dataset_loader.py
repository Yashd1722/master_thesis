from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_project_root() -> Path:
    # This file is in .../Main/src/, so root is one folder up
    return Path(__file__).resolve().parents[1]


# -------------------------
# TS datasets: ts_1500 / ts_500
# -------------------------
def load_ts_dataset(csv_path: Path, feature_cols=("x", "Residuals")):
    """
    Loads a TS dataset CSV.

    Expected columns:
      sequence_ID, Time, x, Residuals, class_label

    Returns:
      sequences: list[np.ndarray]  (one array per sequence, shape (T, F))
      labels: np.ndarray          (shape (N,))
      feature_names: list[str]
    """
    df = pd.read_csv(csv_path)

    # ensure time order inside each sequence
    df = df.sort_values(["sequence_ID", "Time"])

    sequences = []
    labels = []

    groups = df.groupby("sequence_ID")
    for seq_id, g in tqdm(groups, total=groups.ngroups, desc="Building TS sequences"):
        X = g[list(feature_cols)].to_numpy(dtype=np.float32)
        y = int(g["class_label"].iloc[0])  # one label per sequence
        sequences.append(X)
        labels.append(y)

    labels = np.array(labels, dtype=np.int64)
    return sequences, labels, list(feature_cols)


# -------------------------
# PANGAEA dataset: folder of CSVs
# -------------------------
def load_pangaea_dataset(folder_path: Path, time_col="Age [ka BP]"):
    """
    Loads PANGAEA dataset from a folder containing multiple CSVs.
    Each CSV file is treated as one sequence.

    Returns:
      sequences: list[np.ndarray]
      labels: None  (no class label in pangaea)
      feature_names: list[str]
    """
    files = sorted(folder_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {folder_path}")

    sequences = []
    feature_names = None

    for f in tqdm(files, desc="Loading PANGAEA CSV files"):
        df = pd.read_csv(f)

        # sort by time if available
        if time_col in df.columns:
            df = df.sort_values(time_col)

        # drop known non-feature columns if they exist
        drop_cols = [c for c in [time_col, "Depth sed [m]", "Method comm"] if c in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")

        # make everything numeric; non-numeric becomes NaN
        df = df.apply(pd.to_numeric, errors="coerce")

        # fill missing values with column median
        df = df.fillna(df.median(numeric_only=True))

        # lock feature columns based on first file
        if feature_names is None:
            feature_names = list(df.columns)

        # enforce same column order
        df = df[feature_names]

        X = df.to_numpy(dtype=np.float32)
        sequences.append(X)

    return sequences, None, feature_names


# -------------------------
# Universal loader
# -------------------------
def load_dataset(name: str):
    """
    name: "ts_1500" | "ts_500" | "pangaea_923197"
    """
    root = get_project_root()
    name = name.strip().lower()

    if name == "ts_1500":
        csv_path = root / "dataset" / "ts_1500" / "ts_1500_final.csv"
        return load_ts_dataset(csv_path)

    if name == "ts_500":
        csv_path = root / "dataset" / "ts_500" / "ts_500_final.csv"
        return load_ts_dataset(csv_path)

    if name == "pangaea_923197":
        folder = root / "dataset" / "pangaea_923197" / "datasets" / "clean_dataset"
        return load_pangaea_dataset(folder)

    raise ValueError("Unknown dataset. Use: ts_1500, ts_500, pangaea_923197")


# -------------------------
# Standalone usage
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Universal dataset loader (standalone)")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ts_1500", "ts_500", "pangaea_923197"],
    )
    args = parser.parse_args()

    sequences, labels, feature_names = load_dataset(args.dataset)

    print("\n=== DATASET INFO ===")
    print("Dataset:", args.dataset)
    print("Num sequences:", len(sequences))
    print("Num features:", len(feature_names))
    print("Feature names:", feature_names)
    print("Example sequence shape:", sequences[0].shape)

    if labels is None:
        print("Labels: None")
    else:
        print("Labels shape:", labels.shape)
        u, c = np.unique(labels, return_counts=True)
        print("Class counts:", dict(zip(u.tolist(), c.tolist())))
    print()


if __name__ == "__main__":
    main()
