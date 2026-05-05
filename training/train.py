"""
training/train.py
=================
Universal training script for all models and datasets.

What one run does:
    For each dataset in [ts_500, ts_1500]:
        For each pad_variant in [1, 2]:
            Train model → save checkpoint + metrics + log

Called by train_array.sh with --model argument.
Never hardcodes model names, paths, or hyperparameters.
Everything comes from config.yaml.

Usage:
    python training/train.py --model cnn_lstm
    python training/train.py --model lstm
    python training/train.py --model cnn
    python training/train.py --model cnn_lstm --mode sdml --dataset sdml_MS21

Output files (all naming from config.yaml):
    checkpoints/{model}_{dataset}_v{variant}_best.ckpt
    metrics/{model}_{dataset}_v{variant}_train_metrics.json
    logs/{model}_{dataset}_v{variant}_train.log
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

# ── Repo root on path ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models import get_model, list_models, is_sklearn_model
from src.dataset_loader import get_dataloader, get_dataset_info, load_config


# =============================================================================
#  Logging setup
# =============================================================================

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%H:%M:%S")
    # File handler
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# =============================================================================
#  One epoch
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * len(y)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    n        = len(all_labels)
    avg_loss = total_loss / n
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)

        total_loss += loss.item() * len(y)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    n        = len(all_labels)
    avg_loss = total_loss / n
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm       = confusion_matrix(all_labels, all_preds).tolist()
    return avg_loss, acc, f1, cm


# =============================================================================
#  Single training run (one model × one dataset × one pad_variant)
# =============================================================================

def train_single(
    model_name:  str,
    dataset_name: str,
    pad_variant: int,
    cfg:         dict,
    mode_key:    str,
    device:      torch.device,
    logger:      logging.Logger,
) -> dict:
    """
    Train one model on one dataset with one padding variant.

    Returns metrics dict saved to metrics/.
    """
    # Per-model hyperparameters (cnn_lstm=Bury, lstm/cnn=Ma 2025)
    tr_cfg  = cfg["training"][model_name]
    ds_info = get_dataset_info(dataset_name, cfg)
    ts_len  = ds_info["ts_length"]
    n_cls   = ds_info["num_classes"]
    seed    = cfg["project"]["seed"]

    logger.info(
        f"\n{'─'*60}\n"
        f"  model={model_name}  dataset={dataset_name}  "
        f"pad_variant={pad_variant}  num_classes={n_cls}\n"
        f"{'─'*60}"
    )

    # ── Paths from config naming ───────────────────────────────────────────────
    tag       = f"{model_name}_{dataset_name}_v{pad_variant}"
    ckpt_dir  = REPO_ROOT / cfg["paths"]["checkpoints"]
    met_dir   = REPO_ROOT / cfg["paths"]["metrics"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"{tag}_best.ckpt"
    met_path  = met_dir  / f"{tag}_train_metrics.json"

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader = get_dataloader(dataset_name, "train", cfg,
                                  pad_variant=pad_variant,
                                  batch_size=tr_cfg["batch_size"],
                                  seed=seed)
    val_loader   = get_dataloader(dataset_name, "val",   cfg,
                                  pad_variant=pad_variant,
                                  batch_size=tr_cfg["batch_size"],
                                  seed=seed)
    test_loader  = get_dataloader(dataset_name, "test",  cfg,
                                  pad_variant=pad_variant,
                                  batch_size=tr_cfg["batch_size"],
                                  seed=seed)

    logger.info(
        f"  Batches — train:{len(train_loader)} "
        f"val:{len(val_loader)} test:{len(test_loader)}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    model     = get_model(model_name, ts_len=ts_len, num_classes=n_cls).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Parameters: {n_params:,}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr           = tr_cfg["lr"],
        weight_decay = tr_cfg["weight_decay"],
    )

    scheduler = None
    if tr_cfg.get("scheduler") == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience = tr_cfg["scheduler_patience"],
            factor   = tr_cfg["scheduler_factor"],
            min_lr   = tr_cfg["min_lr"],
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1  = -1.0
    patience_ctr = 0
    history      = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss":   [], "val_acc":   [], "val_f1":   [],
    }
    t_start = time.time()

    for epoch in range(1, tr_cfg["epochs"] + 1):

        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, tr_cfg["grad_clip"]
        )
        vl_loss, vl_acc, vl_f1, _ = eval_one_epoch(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(round(tr_loss, 6))
        history["train_acc"].append( round(tr_acc,  6))
        history["train_f1"].append(  round(tr_f1,   6))
        history["val_loss"].append(  round(vl_loss, 6))
        history["val_acc"].append(   round(vl_acc,  6))
        history["val_f1"].append(    round(vl_f1,   6))

        # Scheduler step
        if scheduler is not None:
            scheduler.step(vl_loss)

        # Checkpoint on best val F1
        if vl_f1 > best_val_f1:
            best_val_f1  = vl_f1
            patience_ctr = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_ctr += 1

        # Log every 50 epochs
        if epoch % 50 == 0 or epoch == 1:
            elapsed = (time.time() - t_start) / 60
            logger.info(
                f"  Epoch {epoch:5d}/{tr_cfg['epochs']} | "
                f"tr_loss={tr_loss:.4f} tr_f1={tr_f1:.4f} | "
                f"vl_loss={vl_loss:.4f} vl_f1={vl_f1:.4f} | "
                f"best={best_val_f1:.4f} | "
                f"{elapsed:.1f}min"
            )

        # Early stopping
        if patience_ctr >= tr_cfg["patience"]:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    total_time = (time.time() - t_start) / 60

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                     weights_only=True))
    te_loss, te_acc, te_f1, te_cm = eval_one_epoch(
        model, test_loader, criterion, device
    )
    logger.info(
        f"\n  Test results: loss={te_loss:.4f} "
        f"acc={te_acc:.4f} f1={te_f1:.4f}"
    )
    logger.info(f"  Training time: {total_time:.1f} min")
    logger.info(f"  Checkpoint: {ckpt_path.name}")

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics = {
        "model":        model_name,
        "dataset":      dataset_name,
        "pad_variant":  pad_variant,
        "num_classes":  n_cls,
        "ts_length":    ts_len,
        "n_params":     n_params,
        "best_epoch":   int(np.argmax(history["val_f1"])) + 1,
        "best_val_f1":  round(best_val_f1, 6),
        "test_loss":    round(te_loss, 6),
        "test_acc":     round(te_acc,  6),
        "test_f1":      round(te_f1,   6),
        "confusion_matrix": te_cm,
        "training_time_min": round(total_time, 2),
        "history":      history,
    }
    with open(met_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Metrics: {met_path.name}")

    return metrics


# =============================================================================
#  Main — loops over datasets and pad_variants
# =============================================================================

def train_svm(model_name: str, cfg: dict, logger: logging.Logger) -> dict:
    """
    Train SVM on PANGAEA AAFT surrogates.
    Called by main() when is_sklearn_model(model_name) is True.
    Uses sklearn GridSearchCV — no epochs, no DataLoader.
    Saves checkpoints/svm_{core}_best.pkl per core.
    """
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from scipy.stats import kendalltau
    from models.svm_ews import SVMClassifier

    svm_cfg  = cfg["training"]["svm"]
    ckpt_dir = REPO_ROOT / cfg["paths"]["checkpoints"]
    met_dir  = REPO_ROOT / cfg["paths"]["metrics"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    for core_name in cfg["slurm"]["test_cores"]:
        clean_dir = REPO_ROOT / cfg["paths"]["pangaea_clean"] / core_name
        X_rows, y_rows = [], []

        for cls_name, label in [("neutral", 0), ("pre_transition", 1)]:
            cls_dir = clean_dir / cls_name
            if not cls_dir.exists():
                logger.warning(f"  Missing surrogate dir: {cls_dir}")
                continue
            files = sorted(cls_dir.glob("surrogate_*.csv"))
            logger.info(f"  {core_name}/{cls_name}: {len(files)} surrogates")
            for fpath in files:
                try:
                    import pandas as pd
                    series = pd.read_csv(fpath, header=None).values.flatten()
                    if len(series) < 20:
                        continue
                    n, win = len(series), max(5, len(series) // 2)
                    variances, ac1s = [], []
                    for pos in range(win, n):
                        seg = series[pos-win:pos]
                        variances.append(float(np.var(seg, ddof=1)))
                        if np.std(seg[:-1]) > 1e-10:
                            ac1s.append(float(np.corrcoef(seg[:-1], seg[1:])[0,1]))
                        else:
                            ac1s.append(0.0)
                    if not variances:
                        continue
                    step_idx = np.arange(len(variances))
                    ktau_v, _ = kendalltau(step_idx, variances)
                    ktau_a, _ = kendalltau(step_idx, ac1s)
                    feat = SVMClassifier.extract_features(
                        np.array(variances), np.array(ac1s),
                        float(ktau_v), float(ktau_a)
                    )
                    X_rows.append(feat)
                    y_rows.append(label)
                except Exception as e:
                    logger.debug(f"  Skipping {fpath.name}: {e}")

        if not X_rows:
            logger.warning(f"  No features for {core_name} — run pangea_cleaner.py")
            continue

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_rows, dtype=np.int64)
        logger.info(f"  {core_name}: {len(X)} samples "
                    f"(neutral={int((y==0).sum())}, pre_tran={int((y==1).sum())})")

        param_grid = {"svc__C": svm_cfg["C_values"],
                      "svc__gamma": svm_cfg["gamma_values"]}
        cv     = StratifiedKFold(n_splits=svm_cfg["cv_folds"],
                                  shuffle=True, random_state=42)
        base   = SVMClassifier(num_classes=2, kernel=svm_cfg["kernel"])
        search = GridSearchCV(base._build_sklearn(), param_grid,
                               cv=cv, scoring=svm_cfg["scoring"],
                               n_jobs=-1, verbose=0)
        t0 = time.time()
        search.fit(X, y)
        logger.info(f"  Best params: {search.best_params_}  "
                    f"CV AUC={search.best_score_:.4f}  "
                    f"({time.time()-t0:.1f}s)")

        best_C     = search.best_params_["svc__C"]
        best_gamma = search.best_params_["svc__gamma"]
        final      = SVMClassifier(num_classes=2, kernel=svm_cfg["kernel"],
                                    C=best_C, gamma=best_gamma)
        final.fit(X, y)
        probs   = final.predict_proba_numpy(X)
        auc     = float(roc_auc_score(y, probs[:, 1]))
        ckpt_path = ckpt_dir / f"svm_{core_name}_best.pkl"
        final.save(ckpt_path)
        logger.info(f"  Saved: {ckpt_path.name}  train AUC={auc:.4f}")

        metrics = {"model": "svm", "core": core_name,
                   "cv_auc": round(search.best_score_, 4),
                   "train_auc": round(auc, 4),
                   "best_C": best_C, "best_gamma": str(best_gamma)}
        met_path = met_dir / f"svm_{core_name}_train_metrics.json"
        with open(met_path, "w") as f:
            json.dump(metrics, f, indent=2)
        all_metrics[core_name] = metrics

    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Universal training script — PyTorch and SVM models."
    )
    parser.add_argument("--model",  type=str, required=True,
                        choices=list_models())
    parser.add_argument("--mode",   type=str, default="bury",
                        choices=["bury", "sdml"])
    parser.add_argument("--dataset",type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    log_dir = REPO_ROOT / cfg["paths"]["logs"]
    log_dir.mkdir(parents=True, exist_ok=True)
    logger  = setup_logging(log_dir / f"{args.model}_{args.mode}_train.log")
    logger.info(f"Model  : {args.model}")
    logger.info(f"Mode   : {args.mode}")

    # ── SVM: sklearn path — trains on PANGAEA surrogates, no GPU needed ───────
    if is_sklearn_model(args.model):
        logger.info("SVM detected — training on PANGAEA AAFT surrogates")
        all_metrics = train_svm(args.model, cfg, logger)
        logger.info(f"\nSVM training complete: {list(all_metrics.keys())}")
        return

    # ── PyTorch path ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device : {device}")

    if args.mode == "bury":
        datasets     = cfg["slurm"]["train_datasets"]
        pad_variants = cfg["training"][args.model]["pad_variants"]
    else:
        if args.dataset is None:
            parser.error("--dataset required for --mode sdml")
        datasets     = [args.dataset]
        pad_variants = cfg["training"]["sdml"]["pad_variants"]

    logger.info(f"Datasets : {datasets}")
    logger.info(f"Variants : {pad_variants}")

    all_metrics = []
    for dataset_name in datasets:
        for pad_variant in pad_variants:
            run_tag    = f"{args.model}_{dataset_name}_v{pad_variant}"
            run_logger = setup_logging(log_dir / f"{run_tag}_train.log")
            try:
                metrics = train_single(
                    model_name   = args.model,
                    dataset_name = dataset_name,
                    pad_variant  = pad_variant,
                    cfg          = cfg,
                    mode_key     = args.mode,
                    device       = device,
                    logger       = run_logger,
                )
                all_metrics.append(metrics)
                logger.info(f"  {run_tag} | test_f1={metrics['test_f1']:.4f}")
            except Exception as e:
                logger.error(f"  FAILED {run_tag}: {e}", exc_info=True)

    logger.info(f"\n{'='*50}  SUMMARY  {'='*50}")
    for m in all_metrics:
        logger.info(f"  {m['dataset']} v{m['pad_variant']} | "
                    f"val_f1={m['best_val_f1']:.4f} | "
                    f"test_f1={m['test_f1']:.4f}")


if __name__ == "__main__":
    main()
