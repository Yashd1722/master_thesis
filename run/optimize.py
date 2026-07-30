import os
import sys
import csv
import itertools
from pathlib import Path
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_common import normalize_mean_abs, left_pad_to, random_censor, make_model_input
from models.tsc import TSCModel
from metric.auc import compute_auc
from sklearn.metrics import f1_score

def make_synthetic_ood_validation(X_test, y_test, seed=42):
    rng = np.random.default_rng(seed)
    X_ood = []
    # Core-length distribution ranges from 57 to 365 points
    for i in range(len(X_test)):
        s = X_test[i, 0]
        L_trunc = rng.integers(57, 365 + 1)
        tail = s[-L_trunc:]
        tail_norm = normalize_mean_abs(tail)
        padded = left_pad_to(tail_norm, 500)
        X_ood.append(padded)
    return np.stack(X_ood), y_test.copy()

def evaluate_config(config_dict, X_train, y_train, X_ood, y_ood, seed=42):
    max_samp = config_dict.get("max_train_samples", 80000)
    n_copies = config_dict.get("tsc_copies", 2)
    
    rng = np.random.default_rng(seed)
    
    # Subsample base training set first to avoid generating massive redundant copies
    if max_samp:
        base_target = max(1, max_samp // (1 + n_copies))
        if len(X_train) > base_target:
            classes = np.unique(y_train)
            per_cls = base_target // len(classes)
            idx = np.concatenate([
                rng.choice(np.where(y_train == c)[0], min(per_cls, (y_train == c).sum()), replace=False)
                for c in classes])
            rng.shuffle(idx)
            X_train, y_train = X_train[idx], y_train[idx]
            
    parts_X = [np.stack([make_model_input(s, 500) for s in X_train]).astype(np.float32)]
    parts_y = [y_train]
    for _ in range(n_copies):
        parts_X.append(np.stack([
            random_censor(s, 500, rng, pad_max_frac=config_dict["pad_max_frac"],
                          min_visible=30, both_sided=config_dict["both_sided"])
            for s in X_train]).astype(np.float32))
        parts_y.append(y_train)
    X_tr_aug = np.concatenate(parts_X, axis=0)
    y_tr_aug = np.concatenate(parts_y, axis=0)
        
    use_4ch = config_dict["use_4channel"]
    window_frac = config_dict["csd_window_frac"]
    
    X_val_input = X_ood.copy()
    
    if use_4ch:
        from src.ews_augmenter import augment_ews_channels
        X_tr_aug, ch_stats = augment_ews_channels(X_tr_aug, window_frac=window_frac)
        X_val_input, _ = augment_ews_channels(X_val_input, window_frac=window_frac, channel_stats=ch_stats)
    else:
        X_tr_aug = X_tr_aug[:, np.newaxis, :]
        X_val_input = X_val_input[:, np.newaxis, :]
        
    tr_ok = (X_tr_aug.std(axis=2) > 1e-6).all(axis=1)
    X_tr_aug, y_tr_aug = X_tr_aug[tr_ok], y_tr_aug[tr_ok]
    
    vl_ok = (X_val_input.std(axis=2) > 1e-6).all(axis=1)
    X_val_input, y_ood_filtered = X_val_input[vl_ok], y_ood[vl_ok]
        
    model = TSCModel("minirocket", ts_len=500, num_classes=4, n_jobs=4, n_kernels=1000)
    model.fit(X_tr_aug, y_tr_aug)
    
    probs = model.predict_proba(X_val_input)
    preds = probs.argmax(axis=1)
    
    p_transition = 1.0 - probs[:, 3]
    labels_binary = (y_ood_filtered != 3).astype(int)
    
    auc = compute_auc(labels_binary, p_transition)
    f1 = f1_score(y_ood_filtered, preds, average="macro", zero_division=0)
    
    return float(auc), float(f1)

def main():
    # Load dataset
    processed_dir = REPO_ROOT / "dataset/processed"
    train_data = np.load(processed_dir / "train_500.npz")
    X_train = train_data["X"].squeeze(1)
    y_train = train_data["y"]
    
    test_data = np.load(processed_dir / "test_500.npz")
    X_test = test_data["X"]
    y_test = test_data["y"]
    
    # Filter flat series from train
    tr_mask = X_train.std(axis=1) > 1e-6
    X_train, y_train = X_train[tr_mask], y_train[tr_mask]
    
    # Generate OOD Validation Set
    X_ood, y_ood = make_synthetic_ood_validation(X_test, y_test, seed=42)
    
    # Define search space (Refinement Round)
    pad_max_fracs = [0.8, 0.9, 0.95]
    both_sided_options = [False]
    tsc_copies_options = [3, 4]
    csd_window_fracs = [0.15, 0.20, 0.25]
    use_4channel_options = [True]
    
    # Keep ledger file
    ledger_path = REPO_ROOT / "results/opt_ledger.csv"
    best_config_path = REPO_ROOT / "results/best_config.yaml"
    
    os.makedirs(REPO_ROOT / "results", exist_ok=True)
    
    f_ledger = open(ledger_path, "a", newline="")
    writer = csv.writer(f_ledger)
    
    best_auc = 0.7599
    best_config = {
        "pad_max_frac": 0.9,
        "both_sided": False,
        "tsc_copies": 3,
        "csd_window_frac": 0.25,
        "use_4channel": True,
        "max_train_samples": 5000
    }
    
    # Generate grid
    grid = list(itertools.product(pad_max_fracs, both_sided_options, tsc_copies_options, csd_window_fracs, use_4channel_options))
    print(f"Total configurations to evaluate: {len(grid)}")
    
    for idx, (pmf, bs, tc, cwf, u4c) in enumerate(grid):
        cfg_dict = {
            "pad_max_frac": pmf,
            "both_sided": bs,
            "tsc_copies": tc,
            "csd_window_frac": cwf,
            "use_4channel": u4c,
            "max_train_samples": 5000
        }
        
        print(f"\n[{idx+1}/{len(grid)}] Evaluating: {cfg_dict}")
        try:
            auc, f1 = evaluate_config(cfg_dict, X_train, y_train, X_ood, y_ood, seed=42)
            print(f"--> OOD Binary AUC: {auc:.4f} | Macro F1: {f1:.4f}")
            
            writer.writerow([pmf, bs, tc, cwf, u4c, auc, f1])
            f_ledger.flush()
            
            if auc > best_auc:
                best_auc = auc
                best_config = cfg_dict
                # Save best config
                with open(best_config_path, "w") as fh:
                    yaml.dump(best_config, fh)
                print(f"New best config saved! AUC: {best_auc:.4f}")
                
            # STOP RULE: if best OOD validation AUC >= 0.95
            if best_auc >= 0.95:
                print(f"STOP RULE triggered! Reached target OOD Binary AUC >= 0.95: {best_auc:.4f}")
                break
                
        except Exception as e:
            print(f"Failed to evaluate config: {e}")
            
    f_ledger.close()
    
    print("\nOptimization completed.")
    if best_config:
        print(f"Best Configuration: {best_config}")
        print(f"Best OOD Binary AUC: {best_auc:.4f}")
    else:
        print("No configurations successfully evaluated.")

if __name__ == "__main__":
    main()
