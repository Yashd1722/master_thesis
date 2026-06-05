"""
testing/evaluate.py
Unified evaluation for Zenodo and PANGAEA. 
Features:
- Skips already evaluated models (Caching)
- Checks for checkpoint existence before loading data
- Filters flat series for TSC models to prevent aeon ValueError
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch, joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models import get_model, list_models, is_tsc_model
from src.dataset_loader import load_config, get_dataset_info, get_dataloader
from src.rolling_window import run_all_sapropels

from metric.accuracy import compute_accuracy
from metric.auc import compute_auc
from metric.roc import compute_roc
from metric.kendall_tau import compute_kendall_tau

def json_safe(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.generic): return obj.item()
    raise TypeError

def get_ckpt_path(model_name, dataset_name, cfg):
    """Constructs the checkpoint path without loading the model."""
    ckpt_dir = REPO_ROOT / cfg["paths"]["checkpoints"]
    if is_tsc_model(model_name):
        return ckpt_dir / cfg["naming"]["checkpoint_tsc"].format(model=model_name, dataset=dataset_name)
    else:
        return ckpt_dir / cfg["naming"]["checkpoint_dl"].format(model=model_name, dataset=dataset_name, variant=1)

def load_model(model_name, dataset_name, cfg, device):
    """Loads the model only after verifying the checkpoint exists."""
    ckpt_path = get_ckpt_path(model_name, dataset_name, cfg)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
    ds_info = get_dataset_info(dataset_name, cfg)
    if is_tsc_model(model_name):
        return joblib.load(ckpt_path), "tsc"
    else:
        model = get_model(model_name, ts_len=ds_info["ts_length"], num_classes=ds_info["num_classes"])
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        return model.to(device).eval(), "dl"

def predict_proba(model, X_np, m_type, device, null_idx):
    if m_type == "tsc": 
        probs = model.predict_proba(X_np)
    else:
        X_t = torch.tensor(X_np, dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad(): probs = model(X_t).softmax(dim=1).cpu().numpy()
    return 1.0 - probs[:, null_idx] if null_idx >= 0 else probs[:, -1]

def evaluate_zenodo(model_name, dataset_name, cfg, device):
    # 1. CACHING: Skip if already evaluated
    out_dir = REPO_ROOT / cfg["paths"]["results"] / f"{model_name}_{dataset_name}_zenodo"
    out_file = out_dir / "result.json"
    if out_file.exists():
        print(f"⏭️  Skipping {model_name}/{dataset_name}/zenodo (Already evaluated)")
        return

    # 2. CHECKPOINT CHECK: Fail fast if missing
    ckpt_path = get_ckpt_path(model_name, dataset_name, cfg)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path.name}")

    ds_info = get_dataset_info(dataset_name, cfg)
    null_idx = ds_info["class_names"].index("null") if "null" in ds_info["class_names"] else -1
    model, m_type = load_model(model_name, dataset_name, cfg, device)
    
    test_dl = get_dataloader(dataset_name, "test", cfg, 1, 1024)
    X_all = np.concatenate([x.squeeze(1).numpy() for x, y in test_dl])
    y_all = np.concatenate([y.numpy() for x, y in test_dl])
    
    # 3. FLAT SERIES FILTER: Prevents aeon ValueError
    if m_type == "tsc":
        stds = X_all.std(axis=1)
        valid_mask = stds > 1e-7
        n_dropped = len(X_all) - valid_mask.sum()
        if n_dropped > 0:
            print(f"  ⚠️ Dropping {n_dropped} flat test series for TSC inference")
        X_all = X_all[valid_mask]
        y_all = y_all[valid_mask]
        
    if len(X_all) == 0:
        print("  ❌ No valid test samples remaining.")
        return
    
    t0 = time.time()
    probs = predict_proba(model, X_all, m_type, device, null_idx)
    inf_time = time.time() - t0
    
    preds = (probs > 0.5).astype(int) if null_idx >= 0 else probs.argmax(axis=1)
    labels_bin = (y_all != null_idx).astype(int) if null_idx >= 0 else (y_all > 0).astype(int)
    
    fpr, tpr, thresh = compute_roc(labels_bin, probs)
    acc_dict = compute_accuracy(y_all, preds)
    
    results = {
        "model": model_name, "dataset": dataset_name, "target": "zenodo",
        "inference_time_sec": round(inf_time, 2), "auc": compute_auc(labels_bin, probs),
        "roc": {"fpr": fpr, "tpr": tpr}, "accuracy": acc_dict["accuracy"],
        "macro_f1": acc_dict["macro_f1"], "confusion_matrix": acc_dict["confusion_matrix"]
    }
    
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f: json.dump(results, f, indent=4, default=json_safe)
    print(f"✅ Saved Zenodo metrics to {out_dir}")

def evaluate_pangaea(model_name, dataset_name, cfg, device):
    # 1. CACHING: Skip if already evaluated
    out_dir = REPO_ROOT / cfg["paths"]["results"] / f"{model_name}_{dataset_name}_pangaea"
    out_file = out_dir / "result.json"
    if out_file.exists():
        print(f"⏭️  Skipping {model_name}/{dataset_name}/pangaea (Already evaluated)")
        return

    # 2. CHECKPOINT CHECK: Fail fast if missing
    ckpt_path = get_ckpt_path(model_name, dataset_name, cfg)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path.name}")

    ds_info = get_dataset_info(dataset_name, cfg)
    null_idx = ds_info["class_names"].index("null") if "null" in ds_info["class_names"] else -1
    model, m_type = load_model(model_name, dataset_name, cfg, device)
    
    core_data = {}
    for core_name in cfg["pangaea"]["cores"]:
        sap_results = run_all_sapropels(core_name, cfg, ds_info["ts_length"])
        core_data[core_name] = {}
        for sap_id, elements in sap_results.items():
            core_data[core_name][sap_id] = {}
            for element, segs in elements.items():
                if "forced" not in segs: continue
                forced_res = segs["forced"]
                X = np.stack(forced_res["dl_inputs"]) 
                
                # 3. FLAT SERIES FILTER for PANGAEA TSC
                if m_type == "tsc":
                    stds = X.std(axis=1)
                    valid_mask = stds > 1e-7
                    X = X[valid_mask]
                    if len(X) == 0: continue
                    
                p_trans = predict_proba(model, X, m_type, device, null_idx)
                
                # Align EWS metrics if we filtered out flat series
                if m_type == "tsc" and len(X) < len(forced_res["variance"]):
                    var_filt = forced_res["variance"][valid_mask].tolist()
                    ac_filt = forced_res["lag1_ac"][valid_mask].tolist()
                else:
                    var_filt = forced_res["variance"].tolist()
                    ac_filt = forced_res["lag1_ac"].tolist()
                
                core_data[core_name][sap_id][element] = {
                    "p_transition": p_trans.tolist(), "variance": var_filt,
                    "lag1_ac": ac_filt, "ktau_var": forced_res["ktau_var"],
                    "ktau_ac": forced_res["ktau_ac"], "ktau_p_trans": compute_kendall_tau(p_trans)
                }
                
    results = {"model": model_name, "dataset": dataset_name, "target": "pangaea", "cores": core_data}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f: json.dump(results, f, indent=4, default=json_safe)
    print(f"✅ Saved PANGAEA metrics to {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list_models())
    parser.add_argument("--dataset", required=True, choices=["ts_500", "ts_1500"])
    parser.add_argument("--target", default="zenodo", choices=["zenodo", "pangaea"])
    args = parser.parse_args()
    
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.target == "zenodo": evaluate_zenodo(args.model, args.dataset, cfg, device)
    else: evaluate_pangaea(args.model, args.dataset, cfg, device)

if __name__ == "__main__":
    main()
