import argparse
import sys
import time
import json
from pathlib import Path
import pandas as pd
import torch

# Make project root importable (for src and models)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.splits import create_or_use_split_csv
from metrics import METRICS
from helper import (
    project_root,
    list_available_models,
    setup_logger,
    TSCSVDataset,
    build_model,
)
from evaluate import run_one_epoch


class EarlyStopping:
    """Simple early stopping based on a monitored metric."""
    def __init__(self, patience=5, mode="max"):
        self.patience = patience
        self.mode = mode
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


def run_experiment(dataset_name, model_name, metric_name, args):
    """Run a single training experiment."""
    root = project_root()
    for sub in ["results", "checkpoints", "logs"]:
        (root / sub).mkdir(exist_ok=True)

    run_name = f"{model_name}_{dataset_name}_{metric_name}"
    log_path = root / "logs" / f"{run_name}.log"
    ckpt_path = root / "checkpoints" / f"{run_name}.pt"
    results_csv = root / "results" / f"{run_name}_training.csv"

    logger = setup_logger(log_path)
    logger.info(f"Run name: {run_name}")
    logger.info(f"Dataset: {dataset_name} | Model: {model_name} | Metric: {metric_name}")
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Results CSV: {results_csv}")

    train_csv, val_csv, test_csv = create_or_use_split_csv(dataset_name)

    feature_cols = ("x", "Residuals")
    input_size = len(feature_cols)

    train_ds = TSCSVDataset(
        train_csv,
        feature_cols=feature_cols,
        seq_len=args.seq_len,
        chunksize=args.chunksize,
        apply_padding=args.bury_padding,
        pad_mode=args.bury_pad_mode,
        apply_norm=args.bury_norm,
        seed=args.data_seed,
    )
    val_ds = TSCSVDataset(
        val_csv,
        feature_cols=feature_cols,
        seq_len=args.seq_len,
        chunksize=args.chunksize,
        apply_padding=args.bury_padding,
        pad_mode=args.bury_pad_mode,
        apply_norm=args.bury_norm,
        seed=args.data_seed + 1,
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = build_model(
        model_name,
        input_size=input_size,
        num_classes=args.num_classes,
        model_kwargs_json=args.model_kwargs,
        model_class=args.model_class,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    metric_fn = METRICS[metric_name]

    stopper = EarlyStopping(patience=args.patience, mode="max") if args.early_stop else None
    best_val_metric = None
    rows = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc, _, _ = run_one_epoch(
            model, train_loader, optimizer, device, train_mode=True
        )

        with torch.no_grad():
            val_loss, val_acc, y_true, y_pred = run_one_epoch(
                model, val_loader, optimizer=None, device=device, train_mode=False
            )

        val_metric = metric_fn(y_true, y_pred, args.num_classes) if metric_name != "acc" else float(val_acc)

        logger.info(f"Epoch {epoch}/{args.epochs} (time: {time.time()-t0:.1f}s)")
        logger.info(f"  Train -> loss: {train_loss:.6f} | acc: {train_acc:.6f}")
        logger.info(f"  Val   -> loss: {val_loss:.6f} | acc: {val_acc:.6f} | {metric_name}: {val_metric:.6f}")

        rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            f"val_{metric_name}": val_metric,
        })

        if best_val_metric is None or val_metric > best_val_metric:
            best_val_metric = val_metric
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f" Saved best checkpoint: {ckpt_path}")

        if stopper is not None and stopper.step(val_metric):
            logger.info(f"  Early stopping (patience={args.patience})")
            break

        pd.DataFrame(rows).to_csv(results_csv, index=False)

    logger.info(f"Saved training log CSV: {results_csv}")
    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser()

    # Experiment selection (if omitted, run all combinations)
    parser.add_argument("--dataset", default=None, choices=["ts_500", "ts_1500"])
    parser.add_argument("--model", default=None, help="Model file name in models/ (e.g., lstm, gru)")
    parser.add_argument("--metric", default=None, choices=list(METRICS.keys()))

    # Model selection helpers
    parser.add_argument("--model_class", default=None, help="Exact class name inside the model file")
    parser.add_argument("--model_kwargs", default=None, help="JSON dict of extra model args")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq_len", type=int, default=500)
    parser.add_argument("--chunksize", type=int, default=300_000)
    parser.add_argument("--num_classes", type=int, default=4)

    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--patience", type=int, default=5)

    # Bury-style preprocessing
    parser.add_argument("--bury_padding", action="store_true", help="Apply random zero padding")
    parser.add_argument("--bury_pad_mode", default="both", choices=["both", "left"], help="Padding mode")
    parser.add_argument("--bury_norm", action="store_true", help="Apply mean‑abs normalisation")
    parser.add_argument("--data_seed", type=int, default=42, help="Seed for padding randomness")

    args = parser.parse_args()

    datasets = ["ts_500", "ts_1500"] if args.dataset is None else [args.dataset]
    metrics = list(METRICS.keys()) if args.metric is None else [args.metric]

    available_models = list_available_models()
    if not available_models:
        raise RuntimeError("No model files found in models/")
    models = available_models if args.model is None else [args.model.lower()]

    total = len(datasets) * len(models) * len(metrics)
    print(f"Running {total} experiment(s): {len(models)} model(s) × {len(datasets)} dataset(s) × {len(metrics)} metric(s)")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Metrics: {metrics}")

    for model_name in models:
        for dataset_name in datasets:
            for metric_name in metrics:
                run_experiment(dataset_name, model_name, metric_name, args)


if __name__ == "__main__":
    main()
