"""
training/train.py
Minimal, unified training script for DL (PyTorch) and TSC (aeon) models.
Features:
- Checks for existing checkpoints and skips training if found (Caching).
- Automatically generates result.json for both new and cached models.
- HPC thread-safe.
"""
import os, sys, json, argparse, logging, time
from pathlib import Path

# =============================================================================
# 1. HPC THREAD SAFETY (Must be before any numpy/torch/aeon imports)
# =============================================================================
try:
    _n_aff = len(os.sched_getaffinity(0))
except AttributeError:
    _n_aff = os.cpu_count() or 8
os.environ["NUMBA_NUM_THREADS"] = str(_n_aff)
os.environ["OMP_NUM_THREADS"] = str(_n_aff)
os.environ["MKL_NUM_THREADS"] = str(_n_aff)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models import get_model, list_models, is_tsc_model

# Safe fallback if get_max_train_samples is missing from models/__init__.py
try:
    from models import get_max_train_samples
except ImportError:
    def get_max_train_samples(model_name): return 20000

from src.dataset_loader import load_config, get_dataset_info, get_dataloader

# =============================================================================
# 2. UTILITIES
# =============================================================================
def json_safe(obj):
    """Converts numpy types to native Python types for JSON saving."""
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.generic): return obj.item()
    raise TypeError

def setup_log(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(log_path.stem)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
        for h in [logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)]:
            h.setFormatter(fmt)
            logger.addHandler(h)
    return logger

def save_results(metrics: dict, cfg: dict):
    out_dir = REPO_ROOT / cfg["paths"]["results"] / f"{metrics['model']}_{metrics['dataset']}_train"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w") as f:
        json.dump(metrics, f, indent=4, default=json_safe)

# =============================================================================
# 3. DEEP LEARNING (PYTORCH) HELPERS
# =============================================================================
def _train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss, preds_all, labels_all = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(y)
        preds_all.extend(logits.argmax(1).cpu().numpy())
        labels_all.extend(y.cpu().numpy())
    n = len(labels_all)
    return total_loss / n, accuracy_score(labels_all, preds_all), f1_score(labels_all, preds_all, average="macro", zero_division=0)

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
    return total_loss / n, accuracy_score(labels_all, preds_all), f1_score(labels_all, preds_all, average="macro", zero_division=0), confusion_matrix(labels_all, preds_all).tolist()

# =============================================================================
# 4. MAIN TRAINING ROUTINES (WITH CACHING)
# =============================================================================
def train_dl_model(model_name, dataset_name, pad_variant, cfg, device, logger):
    tr_cfg = cfg["training"].get(model_name, {})
    ds_info = get_dataset_info(dataset_name, cfg)
    ckpt_dir = REPO_ROOT / cfg["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / cfg["naming"]["checkpoint_dl"].format(model=model_name, dataset=dataset_name, variant=pad_variant)

    # --- CACHING MECHANISM ---
    if ckpt_path.exists():
        logger.info(f"  ⏭️ Checkpoint already exists: {ckpt_path.name}. Skipping training.")
        logger.info("  Loading existing checkpoint to evaluate test set...")
        model = get_model(model_name, ts_len=ds_info["ts_length"], num_classes=ds_info["num_classes"]).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()
        
        test_dl = get_dataloader(dataset_name, "test", cfg, pad_variant, tr_cfg.get("batch_size", 64))
        criterion = nn.CrossEntropyLoss()
        te_loss, te_acc, te_f1, te_cm = _eval_epoch(model, test_dl, criterion, device)
        
        return {
            "model": model_name, "dataset": dataset_name, "target": "train", 
            "pad_variant": pad_variant, "status": "skipped_cached",
            "test_acc": round(te_acc, 4), "test_f1": round(te_f1, 4), "confusion_matrix": te_cm
        }

    # --- NORMAL TRAINING LOOP ---
    batch_size = tr_cfg.get("batch_size", 64)
    train_dl = get_dataloader(dataset_name, "train", cfg, pad_variant, batch_size)
    val_dl = get_dataloader(dataset_name, "val", cfg, pad_variant, batch_size)
    test_dl = get_dataloader(dataset_name, "test", cfg, pad_variant, batch_size)

    torch.manual_seed(cfg["project"]["seed"])
    model = get_model(model_name, ts_len=ds_info["ts_length"], num_classes=ds_info["num_classes"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Parameters: {n_params:,} | Checkpoint: {ckpt_path.name}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=tr_cfg.get("lr", 0.001), weight_decay=tr_cfg.get("weight_decay", 0.0))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=tr_cfg.get("scheduler_patience", 10), factor=0.5, min_lr=1e-6) if tr_cfg.get("scheduler") == "reduce_on_plateau" else None

    best_val_f1, patience_ctr, t0 = -1.0, 0, time.time()
    for epoch in range(1, tr_cfg.get("epochs", 100) + 1):
        tr_loss, tr_acc, tr_f1 = _train_epoch(model, train_dl, optimizer, criterion, device, tr_cfg.get("grad_clip", 1.0))
        vl_loss, vl_acc, vl_f1, _ = _eval_epoch(model, val_dl, criterion, device)
        if scheduler: scheduler.step(vl_loss)
        if vl_f1 > best_val_f1:
            best_val_f1, patience_ctr = vl_f1, 0
            torch.save(model.state_dict(), ckpt_path)
        else: 
            patience_ctr += 1
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"  Ep {epoch:3d} | tr_f1={tr_f1:.4f} | vl_f1={vl_f1:.4f} | best={best_val_f1:.4f} | {(time.time()-t0)/60:.1f}min")
        if patience_ctr >= tr_cfg.get("patience", 30):
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    te_loss, te_acc, te_f1, te_cm = _eval_epoch(model, test_dl, criterion, device)
    train_time = (time.time() - t0) / 60
    logger.info(f"  Test acc={te_acc:.4f} f1={te_f1:.4f} | Total time={train_time:.1f}min")
    
    return {
        "model": model_name, "dataset": dataset_name, "target": "train", "pad_variant": pad_variant, 
        "n_params": n_params, "best_val_f1": round(best_val_f1, 4), "test_acc": round(te_acc, 4), 
        "test_f1": round(te_f1, 4), "confusion_matrix": te_cm, "training_time_min": round(train_time, 2)
    }

def train_tsc_model(model_name, dataset_name, cfg, logger):
    ds_info = get_dataset_info(dataset_name, cfg)
    tr_cfg = cfg["training"].get(model_name, {})
    ckpt_dir = REPO_ROOT / cfg["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / cfg["naming"]["checkpoint_tsc"].format(model=model_name, dataset=dataset_name)

    # --- CACHING MECHANISM ---
    import joblib
    if ckpt_path.exists():
        logger.info(f"  ⏭️ Checkpoint already exists: {ckpt_path.name}. Skipping training.")
        logger.info("  Loading existing checkpoint to evaluate validation set...")
        model = joblib.load(ckpt_path)
        
        val_dl = get_dataloader(dataset_name, "val", cfg, 1, 4096)
        X_val_list, y_val_list = [], []
        for x, y in val_dl:
            X_val_list.append(x.squeeze(1).numpy())
            y_val_list.append(y.numpy())
        X_val, y_val = np.concatenate(X_val_list), np.concatenate(y_val_list)
        
        vl_mask = X_val.std(axis=1) > 1e-4
        X_val, y_val = X_val[vl_mask], y_val[vl_mask]
        
        val_preds = model.predict_proba(X_val).argmax(axis=1)
        val_acc = accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        
        return {
            "model": model_name, "dataset": dataset_name, "target": "train", "status": "skipped_cached",
            "val_acc": round(val_acc, 4), "val_f1": round(val_f1, 4)
        }

    # --- NORMAL TRAINING LOOP ---
    logger.info("  Loading train/val data into memory...")
    train_dl = get_dataloader(dataset_name, "train", cfg, 1, 4096)
    val_dl = get_dataloader(dataset_name, "val", cfg, 1, 4096)
    
    X_train_list, y_train_list = [], []
    for x, y in train_dl:
        X_train_list.append(x.squeeze(1).numpy())
        y_train_list.append(y.numpy())
    X_train, y_train = np.concatenate(X_train_list), np.concatenate(y_train_list)
    
    X_val_list, y_val_list = [], []
    for x, y in val_dl:
        X_val_list.append(x.squeeze(1).numpy())
        y_val_list.append(y.numpy())
    X_val, y_val = np.concatenate(X_val_list), np.concatenate(y_val_list)
    
    # Drop flat series (aeon rejects constant input)
    tr_mask, vl_mask = X_train.std(axis=1) > 1e-4, X_val.std(axis=1) > 1e-4
    X_train, y_train = X_train[tr_mask], y_train[tr_mask]
    X_val, y_val = X_val[vl_mask], y_val[vl_mask]

    max_samp = get_max_train_samples(model_name)
    if max_samp and len(X_train) > max_samp:
        idx = np.random.default_rng(42).choice(len(X_train), max_samp, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        logger.info(f"  Subsampled train to {len(X_train)} samples")

    logger.info(f"  Fitting {model_name}...")
    model = get_model(model_name, ts_len=ds_info["ts_length"], num_classes=ds_info["num_classes"], **tr_cfg)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = (time.time() - t0) / 60
    
    val_preds = model.predict_proba(X_val).argmax(axis=1)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
    logger.info(f"  Val acc={val_acc:.4f} f1={val_f1:.4f} | Total time={train_time:.1f}min")
    model.save(ckpt_path)
    
    return {
        "model": model_name, "dataset": dataset_name, "target": "train", 
        "val_acc": round(val_acc, 4), "val_f1": round(val_f1, 4), "training_time_min": round(train_time, 2)
    }

# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list_models())
    parser.add_argument("--dataset", required=True, choices=["ts_500", "ts_1500"])
    args = parser.parse_args()
    
    cfg = load_config()
    logger = setup_log(REPO_ROOT / cfg["paths"]["logs"] / f"{args.model}_{args.dataset}_train.log")
    logger.info(f"Model: {args.model} | Dataset: {args.dataset} | IS_TSC: {is_tsc_model(args.model)}")

    if is_tsc_model(args.model): 
        metrics = train_tsc_model(args.model, args.dataset, cfg, logger)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")
        for v in cfg["training"].get(args.model, {}).get("pad_variants", [1]):
            metrics = train_dl_model(args.model, args.dataset, v, cfg, device, logger)
            
    save_results(metrics, cfg)
    logger.info(f"✅ Saved training metrics to results/{metrics['model']}_{metrics['dataset']}_train/result.json")

if __name__ == "__main__":
    main()
