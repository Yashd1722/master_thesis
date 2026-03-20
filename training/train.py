import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.splits import create_or_use_split_csv
from metrics import METRICS
from models import list_available_models, build_model
from helper import project_root, setup_logger, TSCSVDataset


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


def build_run_name(model_name, dataset_name, metric_name, args):
    base = f"{model_name}_{dataset_name}_{metric_name}"
    if args.trend:
        base += "_trend"
    if args.seasonality:
        base += "_season"
    return base


def run_epoch(model, loader, device, optimizer=None, args=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0
    y_true_all, y_pred_all = [], []

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if train_mode:
                optimizer.zero_grad()

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            if train_mode:
                loss.backward()
                optimizer.step()

            preds = torch.argmax(logits, dim=1)
            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == y).sum().item()
            total_samples += batch_size

            y_true_all.extend(y.cpu().tolist())
            y_pred_all.extend(preds.cpu().tolist())

    return total_loss / total_samples, total_correct / total_samples, y_true_all, y_pred_all


def run_experiment(dataset_name, model_name, metric_name, args):
    """Run a single training experiment."""
    root = project_root()
    for sub in ["results", "checkpoints", "logs"]:
        (root / sub).mkdir(exist_ok=True)

    run_name = build_run_name(model_name, dataset_name, metric_name, args)
    ckpt_path = root / "checkpoints" / f"{run_name}.pt"
    results_csv = root / "results" / f"{run_name}_training.csv"
    logger = setup_logger(root / "logs" / f"{run_name}.log")

    logger.info(f"Experiment: {run_name} | Dataset: {dataset_name} | Model: {model_name}")

    try:
        train_csv, val_csv, _ = create_or_use_split_csv(dataset_name)
    except Exception as e:
        logger.error(f"Failed to prepare splits for dataset {dataset_name}: {e}")
        return

    if not train_csv.exists() or not val_csv.exists():
        logger.error(f"Dataset split files not found: {train_csv} or {val_csv}")
        return

    forcing_config = {
        "trend": args.trend,
        "seasonality": args.seasonality,
        "trend_strength": args.trend_strength,
        "season_amp": args.season_amp,
        "season_period": args.season_period,
    }

    ds_kwargs = dict(
        feature_cols=("x", "Residuals"),
        seq_len=args.seq_len,
        chunksize=args.chunksize,
        apply_padding=args.bury_padding,
        pad_mode=args.bury_pad_mode,
        apply_norm=args.bury_norm,
        forcing_config=forcing_config,
        seed=args.data_seed,
    )

    train_loader = DataLoader(TSCSVDataset(train_csv, **ds_kwargs), batch_size=args.batch_size)
    val_loader = DataLoader(TSCSVDataset(val_csv, **ds_kwargs), batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_name, input_size=2, num_classes=args.num_classes,
        model_kwargs_json=args.model_kwargs, model_class=args.model_class
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    metric_fn = METRICS[metric_name]
    stopper = EarlyStopping(patience=args.patience, mode="max") if args.early_stop else None

    best_val_metric, rows = 0.0, []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, device, optimizer, args)
        val_loss, val_acc, y_true, y_pred = run_epoch(model, val_loader, device, None, args)

        val_metric = metric_fn(y_true, y_pred, args.num_classes) if metric_name != "acc" else float(val_acc)
        logger.info(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val {metric_name}: {val_metric:.4f} | Time: {time.time()-t0:.1f}s")

        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, f"val_{metric_name}": val_metric})

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"New best model saved to {ckpt_path}")

        if stopper and stopper.step(val_metric):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    pd.DataFrame(rows).to_csv(results_csv, index=False)
    logger.info(f"Experiment complete. Results saved to {results_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, choices=["ts_500", "ts_1500"])
    parser.add_argument("--model", default=None, help="Model name in models/")
    parser.add_argument("--metric", default=None, choices=list(METRICS.keys()))
    parser.add_argument("--model_class", default=None)
    parser.add_argument("--model_kwargs", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq_len", type=int, default=500)
    parser.add_argument("--chunksize", type=int, default=300_000)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--bury_padding", action="store_true")
    parser.add_argument("--bury_pad_mode", default="both", choices=["both", "left"])
    parser.add_argument("--bury_norm", action="store_true")
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--trend", action="store_true")
    parser.add_argument("--seasonality", action="store_true")
    parser.add_argument("--trend_strength", type=float, default=1.0)
    parser.add_argument("--season_amp", type=float, default=0.4)
    parser.add_argument("--season_period", type=int, default=50)

    args = parser.parse_args()
    datasets = [args.dataset] if args.dataset else ["ts_500", "ts_1500"]
    metrics = [args.metric] if args.metric else list(METRICS.keys())
    models = [args.model.lower()] if args.model else list_available_models()

    for m in models:
        for d in datasets:
            for met in metrics:
                run_experiment(d, m, met, args)


if __name__ == "__main__":
    main()
