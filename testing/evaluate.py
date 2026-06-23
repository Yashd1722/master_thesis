"""
testing/evaluate.py
Unified evaluation for all DL and TSC models.
FIX: 
1. Renamed 'p_transition_forced' to 'p_transition' to match plot_figures.py.
2. Includes NumpyEncoder to safely save NumPy arrays to JSON.
"""
import argparse
import json
import logging
import sys
import time
import warnings
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
from testing.plot_figures import plot_roc, plot_confusion_matrix, plot_pangaea_series

# =============================================================================
# JSON Encoder for NumPy types (Prevents 'ndarray is not JSON serializable')
# =============================================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)

# =============================================================================
# Logging
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
# Experiment result folder
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
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    return path

# =============================================================================
# Load checkpoint
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
# Inference
# =============================================================================
@torch.no_grad()
def dl_predict_proba(models, X_np, device, batch_size=256):
    """X_np: (N, T) -> returns (N, n_classes) mean ensemble probs."""
    X_t = torch.tensor(X_np, dtype=torch.float32).unsqueeze(1)
    all_probs = []
    for m in models:
        probs = []
        for i in range(0, len(X_t), batch_size):
            probs.append(m(X_t[i:i+batch_size].to(device)).softmax(dim=1).cpu().numpy())
        all_probs.append(np.concatenate(probs))
    return np.mean(all_probs, axis=0)

# =============================================================================
# Evaluate on Zenodo test split
# =============================================================================
def evaluate_zenodo(model_name, dataset_name, cfg, device, logger):
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

    test_dl = get_dataloader(dataset_name, "test", cfg, pad_variant=1,
                              batch_size=1024)

    X_all, y_all = [], []
    for x, y in test_dl:
        X_all.append(x.squeeze(1).numpy())
        y_all.append(y.numpy())
    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)

    # Filter flat series for TSC models to prevent aeon ValueError
    if is_tsc_model(model_name):
        stds = X_all.std(axis=1)
        valid_mask = stds > 1e-7
        n_dropped = len(X_all) - valid_mask.sum()
        if n_dropped > 0:
            logger.warning(f"  Dropping {n_dropped} flat test series (std <= 1e-7)")
        X_all = X_all[valid_mask]
        y_all = y_all[valid_mask]
        
    if len(X_all) == 0:
        logger.error("  No valid samples remaining after filtering.")
        return

    t0 = time.time()
    probs = predict_fn(X_all)
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

    auc_dir  = experiment_dir(model_name, dataset_name, "auc", cfg)
    auc_data = {**base, "metric": "auc", "auc": round(auc, 6),
                 "roc_fpr": fpr, "roc_tpr": tpr, "roc_thresh": thresh}
    save_result(auc_data, auc_dir)
    logger.info(f"  AUC={auc:.4f}  -> {auc_dir.name}/result.json")

    acc_dir  = experiment_dir(model_name, dataset_name, "accuracy", cfg)
    acc_data = {**base, "metric": "accuracy", **acc_metrics}
    save_result(acc_data, acc_dir)
    logger.info(f"  Acc={acc_metrics['accuracy']:.4f}  -> {acc_dir.name}/result.json")

    plot_roc(auc_data, auc_dir)
    plot_confusion_matrix(acc_data, acc_dir, class_names)

    return {"auc": auc_data, "accuracy": acc_data}

# =============================================================================
# Evaluate on PANGAEA
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

    # Helper to safely extract data from dict or object
    def get_data(res, key):
        if res is None: return []
        if isinstance(res, dict): return res.get(key, [])
        return getattr(res, key, [])

    for core_name in cores:
        logger.info(f"\n  Core: {core_name}")
        sap_results = run_all_sapropels(core_name, cfg, ts_len)
        if not sap_results:
            logger.warning(f"  No data for {core_name}")
            continue

        for sap_id, sap_res in sap_results.items():
            for element, elem_res in sap_res.items():
                
                forced_res = elem_res.get("forced")
                neutral_res = elem_res.get("neutral")
                
                forced_inputs = get_data(forced_res, "dl_inputs")
                neutral_inputs = get_data(neutral_res, "dl_inputs")
                
                if not forced_inputs:
                    continue
                
                # 1. Inference on Forced (Positive Class)
                X_forced = np.stack(forced_inputs)
                probs_forced = predict_fn(X_forced)
                p_trans_forced = (1.0 - probs_forced[:, null_idx] if null_idx >= 0 
                                  else probs_forced[:, -1])
                
                # 2. Inference on Neutral (Negative Class)
                p_trans_neutral = np.array([])
                if neutral_inputs:
                    X_neutral = np.stack(neutral_inputs)
                    probs_neutral = predict_fn(X_neutral)
                    p_trans_neutral = (1.0 - probs_neutral[:, null_idx] if null_idx >= 0 
                                       else probs_neutral[:, -1])
                
                # 3. Compute AUC (Forced vs Neutral)
                auc = float('nan')
                fpr, tpr = [0, 1], [0, 1]
                if len(p_trans_neutral) > 0:
                    y_true = np.concatenate([np.ones(len(p_trans_forced)), 
                                             np.zeros(len(p_trans_neutral))])
                    y_score = np.concatenate([p_trans_forced, p_trans_neutral])
                    try:
                        auc = compute_auc(y_true, y_score)
                        fpr, tpr, _ = compute_roc(y_true, y_score)
                    except Exception:
                        pass
                
                # 4. Compute Kendall Tau (Trend in Forced)
                tau = compute_kendall_tau(p_trans_forced)
                
                # 5. Save Results
                tag = f"{core_name}_{sap_id}_{element}"
                base = {"model": model_name, "dataset": dataset_name,
                         "core": core_name, "sapropel": sap_id,
                         "element": element,
                         "n_forced": len(p_trans_forced),
                         "n_neutral": len(p_trans_neutral),
                         "p_transition": p_trans_forced.tolist(), # FIX: Key is now exactly 'p_transition'
                         "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
                
                # Save AUC (Combined)
                auc_dir = experiment_dir(model_name, f"pangaea_{tag}", "auc", cfg)
                auc_data = {**base, "metric": "auc", "segment": "combined",
                             "auc": round(auc, 6), "roc_fpr": fpr, "roc_tpr": tpr}
                save_result(auc_data, auc_dir)
                
                # Save Kendall Tau (Forced trend)
                ktau_dir = experiment_dir(model_name, f"pangaea_{tag}", "kendall_tau", cfg)
                ktau_data = {**base, "metric": "kendall_tau", "segment": "forced",
                              "kendall_tau": round(tau, 6),
                              "variance": get_data(forced_res, "variance"),
                              "lag1_ac": get_data(forced_res, "lag1_ac")}
                save_result(ktau_data, ktau_dir)
                
                # Plot
                plot_roc(auc_data, auc_dir)
                
                class SimpleResult:
                    def __init__(self, var, ac):
                        self.variance = var
                        self.lag1_ac = ac
                simple_result = SimpleResult(get_data(forced_res, "variance"), 
                                             get_data(forced_res, "lag1_ac"))
                plot_pangaea_series(ktau_data, simple_result, ktau_dir)
                
                logger.info(f"    {tag}: AUC={auc:.3f} tau={tau:.3f}")
                all_results[tag] = {"auc": auc, "kendall_tau": tau}

    return all_results

# =============================================================================
# Main
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
