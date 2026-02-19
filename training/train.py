import argparse
from pathlib import Path
import time
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.splits import create_or_use_split_csv
from metrics import METRICS


# -------------------------
# Small helpers
# -------------------------
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def accuracy_np(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


# -------------------------
# Streaming dataset for TS split CSVs (memory-safe)
# Assumption: rows are ordered by (sequence_ID, Time) in your split files.
# -------------------------
class TSCSVDataset(IterableDataset):
    def __init__(self, csv_path, feature_cols=("x", "Residuals"), seq_len=500, chunksize=300_000):
        self.csv_path = str(csv_path)
        self.feature_cols = list(feature_cols)
        self.seq_len = seq_len
        self.chunksize = chunksize

    def __iter__(self):
        cols = ["sequence_ID", "Time", *self.feature_cols, "class_label"]
        buffer_df = None

        for chunk in pd.read_csv(self.csv_path, usecols=cols, chunksize=self.chunksize):
            chunk = chunk.sort_values(["sequence_ID", "Time"])

            if buffer_df is not None:
                chunk = pd.concat([buffer_df, chunk], ignore_index=True)
                buffer_df = None

            groups = list(chunk.groupby("sequence_ID", sort=False))
            if not groups:
                continue

            # keep last group as buffer (may be incomplete)
            last_sid, last_g = groups[-1]
            if len(last_g) < self.seq_len:
                buffer_df = last_g.copy()
                groups = groups[:-1]

            for sid, g in groups:
                g = g.iloc[: self.seq_len]
                if len(g) < self.seq_len:
                    continue

                X = g[self.feature_cols].to_numpy(dtype=np.float32)  # (T,F)
                y = int(g["class_label"].iloc[0])
                yield torch.from_numpy(X), torch.tensor(y, dtype=torch.long)


# -------------------------
# Model loader (simple)
# -------------------------
def build_model(model_name, input_size, num_classes):
    model_name = model_name.lower()

    if model_name == "lstm":
        from models.LSTM import LSTMClassifier
        return LSTMClassifier(input_size=input_size, num_classes=num_classes)

    raise ValueError("Unknown model. Use: lstm")


# -------------------------
# Train / Val loops
# -------------------------
def run_one_epoch(model, loader, optimizer, device, train_mode=True):
    """
    Returns:
      avg_loss, avg_acc
    """
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    y_true_all = []
    y_pred_all = []
    n_batches = 0

    if train_mode:
        model.train()
        pbar = tqdm(loader, desc="Train", dynamic_ncols=True)
    else:
        model.eval()
        pbar = tqdm(loader, desc="Val  ", dynamic_ncols=True)

    for X, y in pbar:
        X = X.to(device)
        y = y.to(device)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            logits = model(X)
            loss = loss_fn(logits, y)

            if train_mode:
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=1)

        total_loss += float(loss.item())
        n_batches += 1

        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(preds.detach().cpu().numpy())

        # update progress bar text
        if n_batches % 20 == 0:
            y_true_np = np.concatenate(y_true_all)
            y_pred_np = np.concatenate(y_pred_all)
            pbar.set_postfix(loss=total_loss / n_batches, acc=accuracy_np(y_true_np, y_pred_np))

    y_true_np = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=int)
    y_pred_np = np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=int)

    avg_loss = total_loss / max(1, n_batches)
    avg_acc = accuracy_np(y_true_np, y_pred_np) if len(y_true_np) else 0.0
    return avg_loss, avg_acc, y_true_np, y_pred_np


# -------------------------
# Early stopping (optional)
# -------------------------
class EarlyStopping:
    def __init__(self, patience=5, mode="max"):
        self.patience = patience
        self.mode = mode  # "max" for metric, "min" for loss
        self.best = None
        self.bad = 0

    def step(self, value):
        if self.best is None:
            self.best = value
            return False

        improved = (value > self.best) if self.mode == "max" else (value < self.best)
        if improved:
            self.best = value
            self.bad = 0
            return False

        self.bad += 1
        return self.bad >= self.patience


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ts_500", "ts_1500"])
    parser.add_argument("--model", default="lstm", choices=["lstm"])
    parser.add_argument("--metric", default="f1_macro", choices=list(METRICS.keys()))

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--seq_len", type=int, default=500)
    parser.add_argument("--chunksize", type=int, default=300_000)
    parser.add_argument("--num_classes", type=int, default=4)

    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--patience", type=int, default=5)

    args = parser.parse_args()

    root = project_root()
    (root / "results").mkdir(exist_ok=True)
    (root / "checkpoints").mkdir(exist_ok=True)

    # Ensure split CSVs exist (cached)
    train_csv, val_csv, test_csv = create_or_use_split_csv(args.dataset)

    # DataLoaders (streaming)
    feature_cols = ("x", "Residuals")
    input_size = len(feature_cols)

    train_ds = TSCSVDataset(train_csv, feature_cols=feature_cols, seq_len=args.seq_len, chunksize=args.chunksize)
    val_ds   = TSCSVDataset(val_csv,   feature_cols=feature_cols, seq_len=args.seq_len, chunksize=args.chunksize)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, num_workers=0)

    # Model / optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = build_model(args.model, input_size=input_size, num_classes=args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Metric function
    metric_fn = METRICS[args.metric]

    # Saving names
    run_name = f"{args.model}_{args.dataset}_{args.metric}"
    ckpt_path = root / "checkpoints" / f"{run_name}.pt"
    results_csv = root / "results" / f"{run_name}_training.csv"  # <-- as you asked

    # Early stopping on validation metric
    stopper = EarlyStopping(patience=args.patience, mode="max") if args.early_stop else None
    best_val_metric = None

    rows = []  # store epoch results

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc, _, _ = run_one_epoch(model, train_loader, optimizer, device, train_mode=True)

        with torch.no_grad():
            val_loss, val_acc, y_true, y_pred = run_one_epoch(model, val_loader, optimizer=None, device=device, train_mode=False)

        val_metric = metric_fn(y_true, y_pred, args.num_classes) if args.metric != "acc" else float(val_acc)

        print(f"\nEpoch {epoch}/{args.epochs}  (time: {time.time()-t0:.1f}s)")
        print(f"  Train -> loss: {train_loss:.6f} | acc: {train_acc:.6f}")
        print(f"  Val   -> loss: {val_loss:.6f} | acc: {val_acc:.6f} | {args.metric}: {val_metric:.6f}")

        rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            f"val_{args.metric}": val_metric,
        })

        # Save best checkpoint (by val_metric)
        if best_val_metric is None or val_metric > best_val_metric:
            best_val_metric = val_metric
            torch.save(model.state_dict(), ckpt_path)
            print("  ✅ Saved best checkpoint:", ckpt_path)

        # Early stopping
        if stopper is not None:
            if stopper.step(val_metric):
                print(f"  🛑 Early stopping (patience={args.patience})")
                break

        # Save results CSV every epoch (safe)
        pd.DataFrame(rows).to_csv(results_csv, index=False)

    print("\nSaved training log CSV:", results_csv)


if __name__ == "__main__":
    main()
