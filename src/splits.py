import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


# -------------------------
# Core split (simple + same functionality)
# -------------------------
def split_data(num_sequences, labels=None, train=0.7, val=0.15, test=0.15, seed=42):
    """
    Split by sequence index (not by rows).

    - If labels is None: random split
    - If labels is provided: stratified split (balanced class distribution)

    Returns:
      train_idx, val_idx, test_idx
    """
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("train + val + test must equal 1.0")

    rng = np.random.default_rng(seed)
    idx = np.arange(num_sequences)

    # Random split (no labels)
    if labels is None:
        rng.shuffle(idx)
        n_train = int(train * num_sequences)
        n_val = int(val * num_sequences)
        return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    # Stratified split (labels provided)
    labels = np.asarray(labels)
    if len(labels) != num_sequences:
        raise ValueError("labels length must match num_sequences")

    train_parts, val_parts, test_parts = [], [], []
    for c in np.unique(labels):
        idx_c = idx[labels == c]
        rng.shuffle(idx_c)

        n_c = len(idx_c)
        n_train_c = int(train * n_c)
        n_val_c = int(val * n_c)

        train_parts.append(idx_c[:n_train_c])
        val_parts.append(idx_c[n_train_c:n_train_c + n_val_c])
        test_parts.append(idx_c[n_train_c + n_val_c:])

    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
    val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=int)
    test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def apply_split(sequences, labels, train_idx, val_idx, test_idx):
    """
    Returns:
      X_train, y_train, X_val, y_val, X_test, y_test
    """
    X_train = [sequences[i] for i in train_idx]
    X_val = [sequences[i] for i in val_idx]
    X_test = [sequences[i] for i in test_idx]

    if labels is None:
        return X_train, None, X_val, None, X_test, None

    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]
    return X_train, y_train, X_val, y_val, X_test, y_test


# -------------------------
# Helper: class counts printing
# -------------------------
def class_counts(y):
    u, c = np.unique(y, return_counts=True)
    return dict(zip(u.tolist(), c.tolist()))


# -------------------------
# NEW: Create split CSV files per dataset (reuse if exist) – simplified
# -------------------------
def get_project_root() -> Path:
    # src/ -> project root
    return Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------
# Dataset‑specific processing functions (modular helpers)
# ----------------------------------------------------------------------
def _process_ts_dataset(dataset_name, dataset_folder, splits, seed, chunksize, out_paths):
    """Handle TS datasets (ts_500, ts_1500) with stratified split on sequence_ID."""
    train_path, val_path, test_path = out_paths
    train_ratio, val_ratio, test_ratio = splits
    final_csv = dataset_folder / f"{dataset_name}_final.csv"
    if not final_csv.exists():
        raise FileNotFoundError(f"Missing file: {final_csv}")

    print(f"\nCreating split files for {dataset_name} from:")
    print(" ", final_csv)

    # Step 1: Build sequence-level labels (memory‑safe)
    seq_label = {}
    usecols = ["sequence_ID", "class_label"]
    for chunk in tqdm(pd.read_csv(final_csv, usecols=usecols, chunksize=chunksize),
                      desc="Reading labels (chunks)", unit="chunk", dynamic_ncols=True):
        for sid, g in chunk.groupby("sequence_ID", sort=False):
            if sid not in seq_label:
                seq_label[sid] = int(g["class_label"].iloc[0])

    seq_ids = np.array(sorted(seq_label.keys()))
    labels = np.array([seq_label[sid] for sid in seq_ids])

    # Step 2: Stratified split on sequence IDs
    train_idx, val_idx, test_idx = split_data(
        num_sequences=len(seq_ids), labels=labels,
        train=train_ratio, val=val_ratio, test=test_ratio, seed=seed
    )

    # Print class distribution
    print("\nClass distribution (sequence-level):")
    print(" Train:", class_counts(labels[train_idx]))
    print(" Val:  ", class_counts(labels[val_idx]))
    print(" Test: ", class_counts(labels[test_idx]))

    train_ids = set(seq_ids[train_idx].tolist())
    val_ids = set(seq_ids[val_idx].tolist())
    test_ids = set(seq_ids[test_idx].tolist())

    # Step 3: Stream and write split CSVs
    header_written = {"train": False, "val": False, "test": False}
    for chunk in tqdm(pd.read_csv(final_csv, chunksize=chunksize),
                      desc="Writing split CSVs", unit="chunk", dynamic_ncols=True):
        for part_name, ids_set, path in [("train", train_ids, train_path),
                                         ("val", val_ids, val_path),
                                         ("test", test_ids, test_path)]:
            mask = chunk["sequence_ID"].isin(ids_set)
            if mask.any():
                chunk_part = chunk[mask]
                chunk_part.to_csv(path, mode="a", index=False,
                                  header=not header_written[part_name])
                header_written[part_name] = True


def _process_pangaea_dataset(dataset_name, dataset_folder, splits, seed, chunksize, out_paths):
    """Handle PANGAEA dataset: split file list and concatenate into train/val/test CSVs."""
    train_path, val_path, test_path = out_paths
    train_ratio, val_ratio, test_ratio = splits
    clean_folder = dataset_folder / "datasets" / "clean_dataset"
    if not clean_folder.exists():
        raise FileNotFoundError(f"Missing folder: {clean_folder}")

    files = sorted(clean_folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found in: {clean_folder}")

    # Split file indices (no labels)
    train_idx, val_idx, test_idx = split_data(
        num_sequences=len(files), labels=None,
        train=train_ratio, val=val_ratio, test=test_ratio, seed=seed
    )

    # Helper to concatenate a list of files into a single CSV
    def _concat_files(file_list, out_path):
        first = True
        for f in tqdm(file_list, desc=f"Writing {out_path.name}", dynamic_ncols=True):
            df = pd.read_csv(f)
            df["source_file"] = f.name  # keep trace
            df.to_csv(out_path, mode="a", index=False, header=first)
            first = False

    # Remove any partially created output files
    for p in (train_path, val_path, test_path):
        if p.exists():
            p.unlink()

    _concat_files([files[i] for i in train_idx], train_path)
    _concat_files([files[i] for i in val_idx], val_path)
    _concat_files([files[i] for i in test_idx], test_path)


# ----------------------------------------------------------------------
# Main public function to create/use split CSV files
# ----------------------------------------------------------------------
def create_or_use_split_csv(dataset_name: str, train=0.7, val=0.15, test=0.15, seed=42, chunksize=500_000):
    """
    Creates split CSV files for a given dataset if they don't exist.
    Returns paths to train, val, test CSV files.
    """
    root = get_project_root()
    dataset_folder = root / "dataset" / dataset_name
    train_path = dataset_folder / f"{dataset_name}_train.csv"
    val_path = dataset_folder / f"{dataset_name}_val.csv"
    test_path = dataset_folder / f"{dataset_name}_test.csv"

    # If split files already exist, just return them
    if train_path.exists() and val_path.exists() and test_path.exists():
        print("Split files already exist. Using existing:")
        for p in (train_path, val_path, test_path):
            print(" ", p)
        return train_path, val_path, test_path

    # Ensure dataset folder exists
    dataset_folder.mkdir(parents=True, exist_ok=True)
    splits = (train, val, test)

    # Dispatch to dataset-specific processor
    processors = {
        "ts_500": _process_ts_dataset,
        "ts_1500": _process_ts_dataset,
        "pangaea_923197": _process_pangaea_dataset,
    }
    if dataset_name not in processors:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Supported: {list(processors.keys())}")

    processors[dataset_name](
        dataset_name=dataset_name,
        dataset_folder=dataset_folder,
        splits=splits,
        seed=seed,
        chunksize=chunksize,
        out_paths=(train_path, val_path, test_path),
    )

    print("\nSaved split files:")
    for p in (train_path, val_path, test_path):
        print(" ", p)
    return train_path, val_path, test_path


# -------------------------
# Standalone usage (kept + added optional dataset split creation)
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Simple splitter (standalone)")

    # Original usage (unchanged):
    parser.add_argument("--n", type=int, help="number of sequences (random split demo)")

    # NEW optional usage:
    parser.add_argument("--dataset", type=str, choices=["ts_500", "ts_1500", "pangaea_923197"],
                        help="If provided, creates/uses split CSV files for that dataset")

    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # If dataset is provided -> create/use split CSV files
    if args.dataset is not None:
        create_or_use_split_csv(
            dataset_name=args.dataset,
            train=args.train, val=args.val, test=args.test,
            seed=args.seed
        )
        return

    # Otherwise keep old behavior (random split by n)
    if args.n is None:
        print("Use either --n <num_sequences> OR --dataset <name>")
        return

    train_idx, val_idx, test_idx = split_data(
        num_sequences=args.n,
        labels=None,
        train=args.train,
        val=args.val,
        test=args.test,
        seed=args.seed,
    )

    print("\n=== SPLIT INFO ===")
    print("n =", args.n)
    print("train/val/test sizes:", len(train_idx), len(val_idx), len(test_idx))
    print("first 10 train idx:", train_idx[:10])
    print()


if __name__ == "__main__":
    main()
