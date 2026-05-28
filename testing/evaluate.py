"""
testing/evaluate.py
===================
Unified evaluation for all DL and TSC models.
After each experiment saves:
    results/{model}_{dataset}_auc/result.json  + roc_curve.png
    results/{model}_{dataset}_accuracy/result.json  + confusion_matrix.png
    results/{model}_{dataset}_kendall_tau/result.json  + kendall_tau.png  (PANGAEA)

Usage:
    python testing/evaluate.py --model cnn_lstm --dataset ts_500
    python testing/evaluate.py --model rocket   --dataset ts_500
    python testing/evaluate.py --model rocket   --dataset ts_500 --target pangaea
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models import get_model, list_models, is_tsc_model
from src.dataset_loader import load_config, get_dataset_info, get_dataloader
from src.rolling_window import run_all_sapropels, ELEMENTS
from metric.auc import compute_auc
from metric.roc import compute_roc
from metric.accuracy import compute_accuracy
from metric.kendall_tau import compute_kendall_tau


# =============================================================================
#  Logging
# =============================================================================

def setup_log(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(log_path.stem)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for h in [logging.FileHandler(log_path, mode="w"),
               logging.StreamHandler(sys.stdout)]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# =============================================================================
#  Experiment result folder
#  results/{model}_{dataset}_{metric}/
# =============================================================================

def experiment_dir(model, dataset, metric, cfg) -> Path:
    name = cfg["naming"]["experiment_dir"].format(
        model=model, dataset=dataset, metric=metric)
    d = REPO_ROOT / cfg["paths"]["results"] / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_result(data: dict, out_dir: Path):
    path = out_dir / "result.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


# =============================================================================
#  Load checkpoint
# =============================================================================

def load_dl_models(model_name, dataset_name, ts_len, num_classes, cfg, device, logger):
    ckpt_dir     = REPO_ROOT / cfg["paths"]["checkpoints"]
    pad_variants = cfg["training"][model_name]["pad_variants"]
    models = []
    for v in pad_variants:
        name  = cfg["naming"]["checkpoint_dl"].format(
            model=model_name, dataset=dataset_name, variant=v)
        fpath = ckpt_dir / name
        if not fpath.exists():
            logger.warning(f"  Not found: {name}")
            continue
        m = get_model(model_name, ts_len=ts_len, num_classes=num_classes)
        m.load_state_dict(torch.load(fpath, map_location=device, weights_only=True))
        m.to(device).eval()
        models.append(m)
        logger.info(f"  Loaded: {name}")
    if not models:
        raise RuntimeError(f"No checkpoints for {model_name}/{dataset_name}")
    return models


def load_tsc_model(model_name, dataset_name, cfg, logger):
    import joblib
    ckpt_dir = REPO_ROOT / cfg["paths"]["checkpoints"]
    name     = cfg["naming"]["checkpoint_tsc"].format(
        model=model_name, dataset=dataset_name)
    fpath = ckpt_dir / name
    if not fpath.exists():
        raise RuntimeError(f"TSC checkpoint not found: {name}")
    model = joblib.load(fpath)
    logger.info(f"  Loaded: {name}")
    return model


# =============================================================================
#  Inference
# =============================================================================

@torch.no_grad()
def dl_predict_proba(models, X_np, device, batch_size=256):
    """X_np: (N, T) -> returns (N, n_classes) mean ensemble probs."""
    X_t = torch.tensor(X_np, dtype=torch.float32).unsqueeze(1)
    all_probs = []
    for m in models:
        probs = []
        for i in range(0, len(X_t), batch_size):
            probs.append(m.predict_proba(X_t[i:i+batch_size].to(device)).cpu().numpy())
        all_probs.append(np.concatenate(probs))
    return np.mean(all_probs, axis=0)


# =============================================================================
#  Evaluate on Zenodo test split -> saves auc + accuracy experiments
# =============================================================================

def evaluate_zenodo(model_name, dataset_name, cfg, device, logger):
    ds_info     = get_dataset_info(dataset_name, cfg)
    ts_len      = ds_info["ts_length"]
    num_classes = ds_info["num_classes"]
    class_names = ds_info["class_names"]
    null_idx    = class_names.index("null") if "null" in class_names else -1

    if is_tsc_model(model_name):
        model = load_tsc_model(model_name, dataset_name, cfg, logger)
    else:
        models = load_dl_models(model_name, dataset_name, ts_len,
                                 num_classes, cfg, device, logger)

    test_dl = get_dataloader(dataset_name, "test", cfg, pad_variant=1,
                              batch_size=1024, num_workers=0)

    X_all, y_all = [], []
    for x, y in test_dl:
        X_all.append(x.squeeze(1).numpy())
        y_all.append(y.numpy())
    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)

    t0 = time.time()
    if is_tsc_model(model_name):
        probs = model.predict_proba(X_all)
    else:
        probs = dl_predict_proba(models, X_all, device)
    inf_time = time.time() - t0

    p_transition = (1.0 - probs[:, null_idx] if null_idx >= 0
                    else probs[:, -1])
    labels_binary = (y_all != null_idx).astype(int) if null_idx >= 0 \
                    else (y_all > 0).astype(int)

    auc  = compute_auc(labels_binary, p_transition)
    fpr, tpr, thresh = compute_roc(labels_binary, p_transition)
    preds = probs.argmax(axis=1)
    acc_metrics = compute_accuracy(y_all, preds)

    base = {"model": model_name, "dataset": dataset_name,
            "num_classes": num_classes, "class_names": class_names,
            "inference_time_sec": round(inf_time, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # ── AUC experiment ────────────────────────────────────────────────────────
    auc_dir  = experiment_dir(model_name, dataset_name, "auc", cfg)
    auc_data = {**base, "metric": "auc", "auc": round(auc, 6),
                "roc_fpr": fpr, "roc_tpr": tpr, "roc_thresh": thresh}
    save_result(auc_data, auc_dir)
    logger.info(f"  AUC={auc:.4f}  -> {auc_dir.name}/result.json")

    # ── Accuracy experiment ───────────────────────────────────────────────────
    acc_dir  = experiment_dir(model_name, dataset_name, "accuracy", cfg)
    acc_data = {**base, "metric": "accuracy", **acc_metrics}
    save_result(acc_data, acc_dir)
    logger.info(f"  Acc={acc_metrics['accuracy']:.4f}  -> {acc_dir.name}/result.json")

    # ── Plot immediately after saving ─────────────────────────────────────────
    from testing.plot_figures import plot_roc, plot_confusion_matrix
    plot_roc(auc_data, auc_dir)
    plot_confusion_matrix(acc_data, acc_dir, class_names)

    return {"auc": auc_data, "accuracy": acc_data}


# =============================================================================
#  Evaluate on PANGAEA -> saves auc + kendall_tau experiments per core
# =============================================================================

def evaluate_pangaea(model_name, dataset_name, cfg, device, logger):
    ds_info     = get_dataset_info(dataset_name, cfg)
    ts_len      = ds_info["ts_length"]
    num_classes = ds_info["num_classes"]
    class_names = ds_info["class_names"]
    null_idx    = class_names.index("null") if "null" in class_names else -1

    if is_tsc_model(model_name):
        model = load_tsc_model(model_name, dataset_name, cfg, logger)
        predict_fn = model.predict_proba
    else:
        models = load_dl_models(model_name, dataset_name, ts_len,
                                 num_classes, cfg, device, logger)
        predict_fn = lambda X: dl_predict_proba(models, X, device)

    cores = list(cfg["pangaea"]["cores"].keys())
    all_results = {}

    for core_name in cores:
        logger.info(f"\n  Core: {core_name}")
        sap_results = run_all_sapropels(core_name, cfg, ts_len)
        if not sap_results:
            logger.warning(f"  No data for {core_name}")
            continue

        for sap_id, sap_res in sap_results.items():
            for element, elem_res in sap_res.items():
                for seg_type, result in elem_res.items():
                    if not result.dl_inputs:
                        continue
                    X = np.stack(result.dl_inputs)
                    probs = predict_fn(X)

                    p_trans = (1.0 - probs[:, null_idx] if null_idx >= 0
                               else probs[:, -1])
                    labels  = np.ones(len(p_trans), dtype=int)
                    if seg_type == "neutral":
                        labels = np.zeros(len(p_trans), dtype=int)

                    auc = compute_auc(labels, p_trans)
                    fpr, tpr, _ = compute_roc(labels, p_trans)
                    ktau = compute_kendall_tau(p_trans)

                    tag = f"{core_name}_{sap_id}_{element}_{seg_type}"
                    base = {"model": model_name, "dataset": dataset_name,
                            "core": core_name, "sapropel": sap_id,
                            "element": element, "segment": seg_type,
                            "n_steps": len(p_trans),
                            "p_transition": p_trans.tolist(),
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

                    auc_dir  = experiment_dir(model_name, f"pangaea_{tag}", "auc", cfg)
                    auc_data = {**base, "metric": "auc",
                                "auc": round(auc, 6),
                                "roc_fpr": fpr, "roc_tpr": tpr}
                    save_result(auc_data, auc_dir)

                    ktau_dir  = experiment_dir(model_name, f"pangaea_{tag}", "kendall_tau", cfg)
                    ktau_data = {**base, "metric": "kendall_tau",
                                 "kendall_tau": round(ktau, 6),
                                 "variance": result.variance.tolist(),
                                 "lag1_ac": result.lag1_ac.tolist()}
                    save_result(ktau_data, ktau_dir)

                    from testing.plot_figures import plot_roc, plot_pangaea_series
                    plot_roc(auc_data, auc_dir)
                    plot_pangaea_series(ktau_data, result, ktau_dir)

                    logger.info(f"    {tag}: AUC={auc:.3f} tau={ktau:.3f}")
                    all_results[tag] = {"auc": auc, "kendall_tau": ktau}

    return all_results


# =============================================================================
#  Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, choices=list_models())
    parser.add_argument("--dataset", required=True, choices=["ts_500", "ts_1500"])
    parser.add_argument("--target",  default="zenodo",
                        choices=["zenodo", "pangaea"])
    parser.add_argument("--config",  default="config.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = REPO_ROOT / cfg["paths"]["logs"]
    log_dir.mkdir(parents=True, exist_ok=True)
    logger  = setup_log(
        log_dir / f"{args.model}_{args.dataset}_{args.target}_eval.log")

    logger.info(f"Model  : {args.model}  IS_TSC={is_tsc_model(args.model)}")
    logger.info(f"Dataset: {args.dataset}  Target: {args.target}")
    logger.info(f"Device : {device}")

    if args.target == "zenodo":
        evaluate_zenodo(args.model, args.dataset, cfg, device, logger)
    else:
        evaluate_pangaea(args.model, args.dataset, cfg, device, logger)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
