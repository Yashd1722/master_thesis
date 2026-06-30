"""
training/train.py
Unified training script for all DL (PyTorch) and TSC (aeon) models.
"""
import os as _os

# Set thread counts BEFORE importing numpy, numba, or aeon — those libraries
# read these env vars at import time. Use SLURM_CPUS_PER_TASK when available
# (set by the scheduler); fall back to CPU affinity count on login nodes.
_n_threads = int(
    _os.environ.get("SLURM_CPUS_PER_TASK", None)
    or len(getattr(_os, "sched_getaffinity", lambda _: [])(0) or [])
    or _os.cpu_count()
    or 4
)
_os.environ["NUMBA_NUM_THREADS"] = str(_n_threads)
_os.environ["OMP_NUM_THREADS"]   = str(_n_threads)
_os.environ["MKL_NUM_THREADS"]   = str(_n_threads)
del _os, _n_threads

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models import get_model, list_models, is_tsc_model, get_max_train_samples
from src.constants import CLASS_NAMES, NULL_IDX
from src.dataset_loader import load_config, get_dataset_info, get_dataloader

# Number of parallel workers: always read from SLURM so we never waste or
# exceed the CPU allocation. Falls back to 4 for interactive sessions.
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

# =============================================================================
# Logging
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
# DL training helpers
# =============================================================================
def _train_epoch(model, loader, optimizer, criterion, device, grad_clip, epoch, total_epochs):
    model.train()
    total_loss, preds_all, labels_all = 0.0, [], []
    bar = tqdm(loader, desc=f"Ep {epoch}/{total_epochs} [train]",
               leave=False, dynamic_ncols=True)
    for x, y in bar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(y)
        preds_all.extend(logits.argmax(1).cpu().numpy())
        labels_all.extend(y.cpu().numpy())
        bar.set_postfix(loss=f"{loss.item():.4f}")
    n = len(labels_all)
    return (total_loss / n,
            accuracy_score(labels_all, preds_all),
            f1_score(labels_all, preds_all, average="macro", zero_division=0))


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
    n  = len(labels_all)
    cm = confusion_matrix(labels_all, preds_all,
                           labels=list(range(len(CLASS_NAMES)))).tolist()
    return (total_loss / n,
            accuracy_score(labels_all, preds_all),
            f1_score(labels_all, preds_all, average="macro", zero_division=0),
            cm)

# =============================================================================
# Train DL model (one pad_variant)
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
    t0          = time.time()
    total_epochs = tr_cfg["epochs"]

    epoch_bar = tqdm(range(1, total_epochs + 1), desc=f"{model_name} v{pad_variant}",
                     unit="ep", dynamic_ncols=True)
    for epoch in epoch_bar:
        tr_loss, tr_acc, tr_f1 = _train_epoch(
            model, train_dl, optimizer, criterion, device,
            tr_cfg["grad_clip"], epoch, total_epochs)
        vl_loss, vl_acc, vl_f1, _ = _eval_epoch(model, val_dl, criterion, device)

        for k, v in [("train_loss", tr_loss), ("train_acc", tr_acc),
                     ("train_f1",  tr_f1),   ("val_loss",  vl_loss),
                     ("val_acc",   vl_acc),   ("val_f1",   vl_f1)]:
            history[k].append(round(v, 6))

        if scheduler:
            scheduler.step(vl_loss)
        if vl_f1 > best_val_f1:
            best_val_f1, patience_ctr = vl_f1, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_ctr += 1

        epoch_bar.set_postfix(
            tr_f1=f"{tr_f1:.3f}", vl_f1=f"{vl_f1:.3f}",
            best=f"{best_val_f1:.3f}", pat=patience_ctr)

        if epoch % 50 == 0 or epoch == 1:
            logger.info(f"  Ep {epoch:5d} | tr_f1={tr_f1:.4f} | "
                        f"vl_f1={vl_f1:.4f} | best={best_val_f1:.4f} | "
                        f"{(time.time() - t0) / 60:.1f} min")
        if patience_ctr >= tr_cfg["patience"]:
            logger.info(f"  Early stop at epoch {epoch}")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                       weights_only=True))
    te_loss, te_acc, te_f1, te_cm = _eval_epoch(model, test_dl, criterion, device)
    logger.info(f"  Test acc={te_acc:.4f}  macro-F1={te_f1:.4f}  "
                f"time={(time.time() - t0) / 60:.1f} min")
    logger.info(f"  Confusion matrix (rows=true, cols=pred):")
    for i, row in enumerate(te_cm):
        logger.info(f"    {CLASS_NAMES[i]:>15s}: {row}")

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
# Train TSC model
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
    logger.info(f"  n_jobs={N_JOBS}  (SLURM_CPUS_PER_TASK or default=4)")

    # Load train and val splits into numpy arrays.
    # Using a large batch size so each DataLoader loop is effectively one pass.
    logger.info("  Loading train and val data into memory...")
    train_dl = get_dataloader(dataset_name, "train", cfg,
                               pad_variant=1, batch_size=8192, seed=seed)
    val_dl   = get_dataloader(dataset_name, "val",   cfg,
                               pad_variant=1, batch_size=8192, seed=seed)

    X_train = np.concatenate([x.squeeze(1).numpy() for x, _ in train_dl])
    y_train = np.concatenate([y.numpy()             for _, y in train_dl])
    X_val   = np.concatenate([x.squeeze(1).numpy() for x, _ in val_dl])
    y_val   = np.concatenate([y.numpy()             for _, y in val_dl])

    # Drop flat series — aeon rejects constant input (std ≤ 1e-7).
    tr_mask = X_train.std(axis=1) > 1e-7
    vl_mask = X_val.std(axis=1)   > 1e-7
    if (~tr_mask).any():
        logger.info(f"  Dropping {(~tr_mask).sum()} flat train series")
        X_train, y_train = X_train[tr_mask], y_train[tr_mask]
    if (~vl_mask).any():
        logger.info(f"  Dropping {(~vl_mask).sum()} flat val series")
        X_val, y_val = X_val[vl_mask], y_val[vl_mask]

    logger.info(f"  Train shape: {X_train.shape}  Val shape: {X_val.shape}")

    # ------------------------------------------------------------------
    # Optional EWS feature augmentation (config flag use_4channel).
    # When ON: expands (N, L) -> (N, 4, L) with [residual, variance,
    # lag-1 AC, skewness] channels, each z-normalised independently.
    # When OFF (default): data stays as (N, 1, L) — univariate.
    # ------------------------------------------------------------------
    use_4ch = cfg.get("inference", {}).get("use_4channel", False)
    if use_4ch:
        from src.ews_augmenter import augment_ews_channels
        logger.info("  EWS augmentation ON (use_4channel=true) — expanding to 4 channels")
        X_train = augment_ews_channels(X_train)   # -> (N, 4, L)
        X_val   = augment_ews_channels(X_val)
        logger.info(f"  Augmented shape: {X_train.shape}")
    else:
        # Keep univariate: (N, L) -> (N, 1, L) so all models get consistent shape.
        X_train = X_train[:, np.newaxis, :]
        X_val   = X_val[:, np.newaxis, :]

    # ------------------------------------------------------------------
    # Stratified subsample when a model defines MAX_TRAIN_SAMPLES.
    # This is the single, explicit memory-safety gate — no duplicate blocks.
    # ------------------------------------------------------------------
    max_samp = get_max_train_samples(model_name)
    if max_samp and len(X_train) > max_samp:
        rng      = np.random.default_rng(seed)
        classes  = np.unique(y_train)
        per_cls  = max_samp // len(classes)
        idx      = np.concatenate([
            rng.choice(np.where(y_train == c)[0], min(per_cls, (y_train == c).sum()),
                       replace=False)
            for c in classes
        ])
        rng.shuffle(idx)
        X_train, y_train = X_train[idx], y_train[idx]
        logger.info(f"  Subsampled to {len(X_train)} (MAX_TRAIN_SAMPLES={max_samp})")

    # Inject n_jobs from environment so we never hardcode 16.
    tr_cfg = dict(tr_cfg)   # copy so we don't mutate config
    tr_cfg["n_jobs"] = N_JOBS

    model = get_model(model_name, ts_len=ts_len, num_classes=n_cls, **tr_cfg)
    logger.info(f"  Fitting {model_name} on {len(X_train)} series...")
    sys.stdout.flush()
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    val_probs = model.predict_proba(X_val)
    val_preds = val_probs.argmax(axis=1)
    val_acc   = float(accuracy_score(y_val, val_preds))
    val_f1    = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
    val_cm    = confusion_matrix(y_val, val_preds,
                                  labels=list(range(n_cls))).tolist()

    logger.info(f"  Val acc={val_acc:.4f}  macro-F1={val_f1:.4f}  "
                f"time={train_time / 60:.1f} min")
    logger.info(f"  Confusion matrix (rows=true, cols=pred):")
    for i, row in enumerate(val_cm):
        logger.info(f"    {CLASS_NAMES[i]:>15s}: {row}")

    model.save(ckpt_path)
    logger.info(f"  Saved: {ckpt_path.name}")

    return {
        "model": model_name, "dataset": dataset_name,
        "val_acc": round(val_acc, 6), "val_f1": round(val_f1, 6),
        "val_confusion_matrix": val_cm,
        "training_time_min": round(train_time / 60, 2),
    }

# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, choices=list_models())
    parser.add_argument("--dataset", required=True, choices=["ts_500", "ts_1500"])
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
