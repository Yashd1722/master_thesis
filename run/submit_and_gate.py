import os
import sys
import time
import shutil
import json
import subprocess
import re
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

def get_running_jobs(user):
    try:
        res = subprocess.run(["squeue", "-u", user, "-h", "-o", "%F"], capture_output=True, text=True, check=True)
        job_ids = res.stdout.strip().split()
        return set(job_ids)
    except Exception as e:
        print(f"Error checking squeue: {e}")
        return set()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-jobs", nargs="+", help="Wait for specific existing Slurm job IDs and skip submission.")
    args = parser.parse_args()

    best_cfg = load_best_config()
    update_global_config(best_cfg)
    
    ckpt_dir = REPO_ROOT / "checkpoints"
    backup_dir = REPO_ROOT / "checkpoints_backup"
    
    models = ["minirocket", "rocket", "multirocket", "arsenal", "rdst", "weasel2", "drcif", "cnn_lstm", "lstm", "inceptiontime"]
    datasets = ["ts_500", "ts_1500"]
    
    backup_dir.mkdir(exist_ok=True)
    
    # 1. Backup all current checkpoints
    print("Backing up current checkpoints...")
    for model in models:
        for dataset in datasets:
            files_to_backup = [
                f"{model}_{dataset}_best.pkl",
                f"{model}_{dataset}_best_ch_stats.npz",
                f"{model}_{dataset}_v1_best.ckpt",
                f"{model}_{dataset}_v2_best.ckpt"
            ]
            for fname in files_to_backup:
                src = ckpt_dir / fname
                if src.exists():
                    shutil.copy2(src, backup_dir / fname)

    if args.wait_jobs:
        submitted_jobs = args.wait_jobs
        print(f"Skipping submission. Gating will wait for existing jobs: {submitted_jobs}")
    else:
        # 2. Submit SLURM jobs
        print("Submitting Slurm jobs...")
        submitted_jobs = []
        
        try:
            res_tsc = subprocess.run(["sbatch", "training/train_tsc_array.sh"], capture_output=True, text=True, check=True)
            m = re.search(r"Submitted batch job (\d+)", res_tsc.stdout)
            if m:
                submitted_jobs.append(m.group(1))
                print(f"Submitted TSC batch job: {m.group(1)}")
        except Exception as e:
            print(f"Failed to submit TSC jobs: {e}")
            
        try:
            res_dl = subprocess.run(["sbatch", "training/train_dl_array.sh"], capture_output=True, text=True, check=True)
            m = re.search(r"Submitted batch job (\d+)", res_dl.stdout)
            if m:
                submitted_jobs.append(m.group(1))
                print(f"Submitted DL batch job: {m.group(1)}")
        except Exception as e:
            print(f"Failed to submit DL jobs: {e}")
            
        if not submitted_jobs:
            print("No jobs submitted. Aborting.")
            return
        
    # 3. Wait for jobs to finish
    user = os.environ.get("USER", "s466553")
    print(f"Waiting for Slurm jobs {submitted_jobs} to finish...")
    
    while True:
        running = get_running_jobs(user)
        active = running.intersection(submitted_jobs)
        if not active:
            print("All submitted Slurm jobs have finished.")
            break
        print(f"Still running: {list(active)}. Sleeping 60s...")
        time.sleep(60)
        
    # 4. Gate vs Baseline
    print("\nGATING PHASE STARTED")
    for model in models:
        for dataset in datasets:
            base_f1 = get_baseline_f1(model, dataset)
            metrics_path = REPO_ROOT / "results" / f"{model}_{dataset}_train_metrics.json"
            
            files = [
                f"{model}_{dataset}_best.pkl",
                f"{model}_{dataset}_best_ch_stats.npz",
                f"{model}_{dataset}_v1_best.ckpt",
                f"{model}_{dataset}_v2_best.ckpt"
            ]
            
            new_f1 = 0.0
            if metrics_path.exists():
                try:
                    with open(metrics_path, "r") as f:
                        new_metrics = json.load(f)
                    if isinstance(new_metrics, list):
                        new_f1 = max(m.get("best_val_f1", 0.0) for m in new_metrics)
                    else:
                        new_f1 = new_metrics.get("val_f1", 0.0)
                except Exception as e:
                    print(f"Could not read metrics for {model} on {dataset}: {e}")
            
            print(f"Model: {model} | Dataset: {dataset} | Base F1: {base_f1:.6f} | New F1: {new_f1:.6f}")
            
            if new_f1 < base_f1:
                print(f"  [REVERT] Regression detected! Restoring baseline.")
                for fname in files:
                    bk = backup_dir / fname
                    dest = ckpt_dir / fname
                    if bk.exists():
                        shutil.copy2(bk, dest)
                    else:
                        if dest.exists():
                            dest.unlink()
            else:
                print(f"  [KEEP] Model improved or matched baseline.")
                
    # 5. Cleanup backup
    print("\nCleaning up backup files...")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    print("Done!")

if __name__ == "__main__":
    main()
