import os
import sys
import shutil
import json
import subprocess
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

def load_best_config():
    best_config_path = REPO_ROOT / "results/best_config.yaml"
    with open(best_config_path, "r") as f:
        return yaml.safe_load(f)

def update_global_config(best_cfg):
    config_path = REPO_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    cfg["augmentation"]["pad_max_frac"] = float(best_cfg["pad_max_frac"])
    cfg["augmentation"]["tsc_copies"] = int(best_cfg["tsc_copies"])
    cfg["augmentation"]["both_sided"] = bool(best_cfg["both_sided"])
    
    cfg["inference"]["rolling_window_frac"] = float(best_cfg["csd_window_frac"])
    cfg["inference"]["rolling_window_frac_augment"] = float(best_cfg["csd_window_frac"])
    cfg["inference"]["use_4channel"] = bool(best_cfg["use_4channel"])
    
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print("config.yaml updated with best configurations.")

def get_baseline_f1(model_name, dataset_name):
    baseline_path = REPO_ROOT / "results/baseline_metrics.json"
    if not baseline_path.exists():
        return 0.0
    with open(baseline_path, "r") as f:
        baseline = json.load(f)
    key = f"synthetic_{model_name}_{dataset_name}"
    if key in baseline:
        return baseline[key].get("macro_f1", 0.0)
    return 0.0

def main():
    best_cfg = load_best_config()
    update_global_config(best_cfg)
    
    models = ["minirocket", "rocket", "multirocket", "arsenal", "rdst", "weasel2", "drcif", "cnn_lstm", "lstm", "inceptiontime"]
    datasets = ["ts_500", "ts_1500"]
    
    ckpt_dir = REPO_ROOT / "checkpoints"
    backup_dir = REPO_ROOT / "checkpoints_backup"
    backup_dir.mkdir(exist_ok=True)
    
    python_exe = sys.executable or "python3"
    
    for model in models:
        for dataset in datasets:
            print(f"\n==================================================")
            print(f"Processing: {model} on {dataset}")
            print(f"==================================================")
            
            # Retrieve baseline F1
            base_f1 = get_baseline_f1(model, dataset)
            print(f"Baseline Macro-F1: {base_f1:.6f}")
            
            # Backup current files if they exist
            files_to_backup = [
                f"{model}_{dataset}_best.pkl",
                f"{model}_{dataset}_best.pt",
                f"{model}_{dataset}_best_ch_stats.npz",
                f"{model}_{dataset}_train_metrics.json"
            ]
            
            backed_up = []
            for fname in files_to_backup:
                src = ckpt_dir / fname
                if src.exists():
                    shutil.copy2(src, backup_dir / fname)
                    backed_up.append(fname)
            
            # Run train.py
            cmd = [python_exe, "training/train.py", "--model", model, "--dataset", dataset, "--force"]
            print(f"Executing: {' '.join(cmd)}")
            
            try:
                # Set thread environment variables matching SLURM or system settings
                env = os.environ.copy()
                n_cpus = env.get("SLURM_CPUS_PER_TASK", "16")
                env["NUMBA_NUM_THREADS"] = n_cpus
                env["OMP_NUM_THREADS"] = n_cpus
                env["MKL_NUM_THREADS"] = n_cpus
                
                res = subprocess.run(cmd, env=env, check=True)
                
                # Check new validation F1
                metrics_path = ckpt_dir / f"{model}_{dataset}_train_metrics.json"
                if not metrics_path.exists():
                    print(f"Error: new metrics file not found for {model} on {dataset}.")
                    raise FileNotFoundError()
                    
                with open(metrics_path, "r") as f:
                    new_metrics = json.load(f)
                new_f1 = new_metrics.get("val_f1", 0.0)
                print(f"New validation Macro-F1: {new_f1:.6f}")
                
                # Gate vs baseline (R6): keep old config/model if it regresses
                if new_f1 < base_f1:
                    print(f"[REVERT WARNING] New Macro-F1 ({new_f1:.6f}) regressed from baseline ({base_f1:.6f})!")
                    print("Restoring baseline checkpoints...")
                    for fname in files_to_backup:
                        bk = backup_dir / fname
                        if bk.exists():
                            shutil.copy2(bk, ckpt_dir / fname)
                        else:
                            # If it did not exist before, remove any newly created files
                            dest = ckpt_dir / fname
                            if dest.exists():
                                dest.unlink()
                else:
                    print(f"[PROMOTED] Model improved or matched baseline (New: {new_f1:.6f} >= Base: {base_f1:.6f})")
                    
            except Exception as e:
                print(f"Training failed or error during post-processing for {model} on {dataset}: {e}")
                print("Restoring baseline checkpoints due to failure...")
                for fname in files_to_backup:
                    bk = backup_dir / fname
                    if bk.exists():
                        shutil.copy2(bk, ckpt_dir / fname)
                        
            # Clean backup files for this model+dataset
            for fname in files_to_backup:
                bk = backup_dir / fname
                if bk.exists():
                    bk.unlink()
                    
    # Remove backup dir
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
        
    print("\nAll models trained and processed.")

if __name__ == "__main__":
    main()
