"""
training/train.py
=================
Unified training script for all DL (PyTorch) and TSC (aeon) models.

Usage:
    python training/train.py --model cnn_lstm   --dataset ts_500
    python training/train.py --model rocket     --dataset ts_500
    python training/train.py --model hivecote   --dataset ts_1500

Output:
    checkpoints/{model}_{dataset}_v{variant}_best.ckpt  (DL)
    checkpoints/{model}_{dataset}_best.pkl               (TSC)
    logs/{model}_{dataset}_train.log
"""

# Set NUMBA_NUM_THREADS from sched_getaffinity BEFORE any numba/torch import
import os as _os
try:
    _n_aff = len(_os.sched_getaffinity(0))
except AttributeError:
    _n_aff = _os.cpu_count() or 8
_os.environ["NUMBA_NUM_THREADS"] = str(_n_aff)
_os.environ.setdefault("OMP_NUM_THREADS", "8")
del _os, _n_aff

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models import get_model, list_models, is_tsc_model, get_max_train_samples
from src.dataset_loader import load_config, get_dataset_info, get_dataloader


# =============================================================================
#  Logging
# =============================================================================

def setup_log(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(log_path.stem)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    for h in [logging.FileHandler(log_path, mode="w"),
               logging.StreamHandler(sys.stdout)]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# =============================================================================
#  DL training helpers
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


def _train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss, preds_all, labels_all = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(y)
        preds_all.extend(logits.argmax(1).cpu().numpy())
        labels_all.extend(y.cpu().numpy())
    n = len(labels_all)
    return total_loss / n, accuracy_score(labels_all, preds_all), \
           f1_score(labels_all, preds_all, average="macro", zero_division=0)


@torch.no_grad()
def _eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, preds_all, labels_all = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        preds_all.extend(logits.argmax(1).cpu().numpy())
        labels_all.extend(y.cpu().numpy())
    n = len(labels_all)
    return (total_loss / n,
            accuracy_score(labels_all, preds_all),
            f1_score(labels_all, preds_all, average="macro", zero_division=0),
            confusion_matrix(labels_all, preds_all).tolist())


# =============================================================================
#  Train DL model (one pad_variant)
# =============================================================================

def train_dl_variant(model_name, dataset_name, pad_variant, cfg, device, logger):
    tr_cfg  = cfg["training"][model_name]
    ds_info = get_dataset_info(dataset_name, cfg)
    ts_len  = ds_info["ts_length"]
    n_cls   = ds_info["num_classes"]
    seed    = cfg["project"]["seed"]

    ckpt_dir = REPO_ROOT / cfg["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / cfg["naming"]["checkpoint_dl"].format(
        model=model_name, dataset=dataset_name, variant=pad_variant)

    train_dl = get_dataloader(dataset_name, "train", cfg,
                               pad_variant=pad_variant,
                               batch_size=tr_cfg["batch_size"], seed=seed)
    val_dl   = get_dataloader(dataset_name, "val",   cfg,
                               pad_variant=pad_variant,
                               batch_size=tr_cfg["batch_size"], seed=seed)
    test_dl  = get_dataloader(dataset_name, "test",  cfg,
                               pad_variant=pad_variant,
                               batch_size=tr_cfg["batch_size"], seed=seed)

    torch.manual_seed(seed)
    model    = get_model(model_name, ts_len=ts_len, num_classes=n_cls).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Parameters: {n_params:,}  ckpt: {ckpt_path.name}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=tr_cfg["lr"],
                            weight_decay=tr_cfg["weight_decay"])
    scheduler = None
    if tr_cfg.get("scheduler") == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=tr_cfg["scheduler_patience"],
            factor=tr_cfg["scheduler_factor"], min_lr=tr_cfg["min_lr"])

    best_val_f1, patience_ctr = -1.0, 0
    history = {k: [] for k in ["train_loss", "train_acc", "train_f1",
                                 "val_loss", "val_acc", "val_f1"]}
    t0 = time.time()

    for epoch in range(1, tr_cfg["epochs"] + 1):
        tr_loss, tr_acc, tr_f1 = _train_epoch(
            model, train_dl, optimizer, criterion, device, tr_cfg["grad_clip"])
        vl_loss, vl_acc, vl_f1, _ = _eval_epoch(model, val_dl, criterion, device)

        for k, v in [("train_loss", tr_loss), ("train_acc", tr_acc),
                      ("train_f1", tr_f1), ("val_loss", vl_loss),
                      ("val_acc", vl_acc), ("val_f1", vl_f1)]:
            history[k].append(round(v, 6))

        if scheduler:
            scheduler.step(vl_loss)
        if vl_f1 > best_val_f1:
            best_val_f1, patience_ctr = vl_f1, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_ctr += 1

        if epoch % 50 == 0 or epoch == 1:
            logger.info(f"  Ep {epoch:5d} | tr_f1={tr_f1:.4f} | "
                        f"vl_f1={vl_f1:.4f} | best={best_val_f1:.4f} | "
                        f"{(time.time()-t0)/60:.1f}min")
        if patience_ctr >= tr_cfg["patience"]:
            logger.info(f"  Early stop at epoch {epoch}")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                      weights_only=True))
    te_loss, te_acc, te_f1, te_cm = _eval_epoch(model, test_dl, criterion, device)
    logger.info(f"  Test acc={te_acc:.4f} f1={te_f1:.4f} "
                f"time={(time.time()-t0)/60:.1f}min")

    return {
        "model": model_name, "dataset": dataset_name,
        "pad_variant": pad_variant, "n_params": n_params,
        "best_val_f1": round(best_val_f1, 6),
        "test_acc": round(te_acc, 6), "test_f1": round(te_f1, 6),
        "confusion_matrix": te_cm,
        "training_time_min": round((time.time() - t0) / 60, 2),
        "history": history,
    }


# =============================================================================
#  Train TSC model
# =============================================================================

def train_tsc(model_name, dataset_name, cfg, logger):
    ds_info = get_dataset_info(dataset_name, cfg)
    ts_len  = ds_info["ts_length"]
    n_cls   = ds_info["num_classes"]
    tr_cfg  = cfg["training"].get(model_name, {})

    ckpt_dir  = REPO_ROOT / cfg["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / cfg["naming"]["checkpoint_tsc"].format(
        model=model_name, dataset=dataset_name)

    seed = cfg["project"]["seed"]
    train_dl = get_dataloader(dataset_name, "train", cfg,
                               pad_variant=1, batch_size=4096, seed=seed)
    val_dl   = get_dataloader(dataset_name, "val",   cfg,
                               pad_variant=1, batch_size=4096, seed=seed)

    logger.info("  Loading train data into memory...")
    X_train, y_train = [], []
    for x, y in train_dl:
        X_train.append(x.squeeze(1).numpy())
        y_train.append(y.numpy())
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    X_val, y_val = [], []
    for x, y in val_dl:
        X_val.append(x.squeeze(1).numpy())
        y_val.append(y.numpy())
    X_val = np.concatenate(X_val)
    y_val = np.concatenate(y_val)

    # Drop flat series (std <= 1e-7) — aeon rejects constant input
    tr_mask = X_train.std(axis=1) > 1e-7
    vl_mask = X_val.std(axis=1)   > 1e-7
    if (~tr_mask).sum():
        logger.info("  Dropping %d flat train series" % int((~tr_mask).sum()))
        X_train, y_train = X_train[tr_mask], y_train[tr_mask]
    if (~vl_mask).sum():
        logger.info("  Dropping %d flat val series" % int((~vl_mask).sum()))
        X_val, y_val = X_val[vl_mask], y_val[vl_mask]

    logger.info(f"  Train: {X_train.shape}  Val: {X_val.shape}")

    # Stratified subsample for computationally expensive models
    max_samp = get_max_train_samples(model_name)
    if max_samp and len(X_train) > max_samp:
        rng = np.random.default_rng(seed)
        classes = np.unique(y_train)
        per_cls = max_samp // len(classes)
        idx = []
        for c in classes:
            cidx = np.where(y_train == c)[0]
            idx.append(rng.choice(cidx, min(per_cls, len(cidx)), replace=False))
        idx = np.concatenate(idx)
        rng.shuffle(idx)
        X_train, y_train = X_train[idx], y_train[idx]
        logger.info(f"  Subsampled train to {len(X_train)} (MAX_TRAIN_SAMPLES={max_samp})")

    # Pass config hyperparams so the Net class can override defaults
    model = get_model(model_name, ts_len=ts_len, num_classes=n_cls, **tr_cfg)
    logger.info(f"  Fitting {model_name}...")
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    val_probs = model.predict_proba(X_val)
    val_preds = val_probs.argmax(axis=1)
    val_acc   = float(accuracy_score(y_val, val_preds))
    val_f1    = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
    logger.info(f"  Val acc={val_acc:.4f} f1={val_f1:.4f} "
                f"time={train_time/60:.1f}min")

    model.save(ckpt_path)
    logger.info(f"  Saved: {ckpt_path.name}")

    return {
        "model": model_name, "dataset": dataset_name,
        "val_acc": round(val_acc, 6), "val_f1": round(val_f1, 6),
        "training_time_min": round(train_time / 60, 2),
    }


# =============================================================================
#  Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, choices=list_models())
    parser.add_argument("--dataset", required=True,
                        choices=["ts_500", "ts_1500"])
    parser.add_argument("--config",  default="config.yaml")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    log_dir = REPO_ROOT / cfg["paths"]["logs"]
    log_dir.mkdir(parents=True, exist_ok=True)
    logger  = setup_log(log_dir / f"{args.model}_{args.dataset}_train.log")

    logger.info(f"Model  : {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"IS_TSC : {is_tsc_model(args.model)}")

    if is_tsc_model(args.model):
        metrics = train_tsc(args.model, args.dataset, cfg, logger)
        logger.info(f"Done — val_f1={metrics['val_f1']:.4f}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device : {device}")
        pad_variants = cfg["training"][args.model]["pad_variants"]
        for v in pad_variants:
            m = train_dl_variant(args.model, args.dataset, v, cfg, device, logger)
            logger.info(f"  v{v} done — test_f1={m['test_f1']:.4f}")


if __name__ == "__main__":
    main()
