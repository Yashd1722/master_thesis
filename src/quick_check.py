import argparse
import pandas as pd
import numpy as np

from dataset_loader import load_dataset
from splits import create_or_use_split_csv


def seq_level_class_counts(csv_path):
    """
    Fast + correct for TS datasets:
    counts classes by UNIQUE sequence_ID (not by rows).
    """
    df = pd.read_csv(csv_path, usecols=["sequence_ID", "class_label"])
    df = df.drop_duplicates(subset=["sequence_ID"])  # one label per sequence
    u, c = np.unique(df["class_label"].values, return_counts=True)
    return dict(zip(u.tolist(), c.tolist()))


def main():
    parser = argparse.ArgumentParser(description="Quick check: load dataset + ensure splits exist")
    parser.add_argument("--dataset", required=True,
                        choices=["ts_1500", "ts_500", "pangaea_923197"])
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1) Ensure split CSVs exist (create if missing)
    train_csv, val_csv, test_csv = create_or_use_split_csv(
        dataset_name=args.dataset,
        train=args.train,
        val=args.val,
        test=args.test,
        seed=args.seed
    )

    # 2) Load dataset into sequences (same as before)
    sequences, labels, feature_names = load_dataset(args.dataset)

    print("\n=== DATASET INFO ===")
    print("Dataset:", args.dataset)
    print("Num sequences:", len(sequences))
    print("Num features:", len(feature_names))
    print("Feature names:", feature_names)

    print("\nExample sequence shapes (first 3):")
    for i in range(min(3, len(sequences))):
        print(f"  seq[{i}] shape:", sequences[i].shape)

    if labels is None:
        print("\nLabels: None")
    else:
        print("\nLabels shape:", labels.shape)
        u, c = np.unique(labels, return_counts=True)
        print("Overall class counts:", dict(zip(u.tolist(), c.tolist())))

    # 3) Print split file paths
    print("\n=== SPLIT FILES ===")
    print("Train:", train_csv)
    print("Val:  ", val_csv)
    print("Test: ", test_csv)

    # 4) Print class distribution inside each split CSV (TS datasets only)
    if args.dataset in ["ts_1500", "ts_500"]:
        print("\n=== CLASS DISTRIBUTION (SEQUENCE-LEVEL) ===")
        print("Train:", seq_level_class_counts(train_csv))
        print("Val:  ", seq_level_class_counts(val_csv))
        print("Test: ", seq_level_class_counts(test_csv))

    else:
        print("\n(No class distribution: pangaea_923197 has no class labels)")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
