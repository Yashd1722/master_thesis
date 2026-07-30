import os
import sys
import json
import csv
import numpy as np
import scipy.stats as stats
from pathlib import Path
from sklearn.metrics import roc_curve, auc

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

def compute_auc_score(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(auc(fpr, tpr))

def main():
    test_result_dir = REPO_ROOT / "test_result"
    if not test_result_dir.exists():
        print("test_result directory does not exist.")
        return
        
    records = []
    
    intervals = {
        "EARLY": (0.6, 0.8),
        "LATE": (0.8, 1.0),
        "FULL": (0.0, 1.0)
    }
    
    for d in sorted(test_result_dir.iterdir()):
        if not d.is_dir():
            continue
        if not d.name.endswith("_pangaea"):
            continue
            
        rfile = d / "result.json"
        if not rfile.exists():
            continue
            
        try:
            with open(rfile, "r") as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"Failed to read {rfile}: {e}")
            continue
            
        model = data.get("model", d.name.replace("_pangaea", ""))
        core = data.get("core", "")
        sap = data.get("sapropel", "")
        element = data.get("element", "")
        
        p_trans_f = np.array(data.get("p_transition", []))
        p_trans_n = np.array(data.get("p_transition_null", []))
        null_win_counts = data.get("null_window_counts", [])
        
        N_forced = len(p_trans_f)
        if N_forced == 0:
            continue
            
        # Calculate early/late/full metrics
        for name, (start_rel, end_rel) in intervals.items():
            # Get indices for forced segment falling into the relative interval
            # Relative position of step i is i / (N_forced - 1)
            indices_forced = [i for i in range(N_forced) if start_rel <= i / (N_forced - 1) <= end_rel]
            if not indices_forced:
                continue
                
            p_f_segment = p_trans_f[indices_forced]
            
            # Select corresponding null segment values
            p_n_segment_list = []
            offset = 0
            for n_win in null_win_counts:
                # Find indices within the null window
                idx_null = [offset + i for i in indices_forced if i < n_win]
                p_n_segment_list.extend(p_trans_n[idx_null])
                offset += n_win
                
            p_n_segment = np.array(p_n_segment_list)
            
            # Binary AUC
            y_true = np.concatenate([np.ones(len(p_f_segment)), np.zeros(len(p_n_segment))])
            y_score = np.concatenate([p_f_segment, p_n_segment])
            auc_val = compute_auc_score(y_true, y_score)
            
            # Kendall tau on forced segment
            steps = np.arange(len(p_f_segment))
            tau, _ = stats.kendalltau(steps, p_f_segment)
            
            records.append({
                "model": model,
                "core": core,
                "sapropel": sap,
                "element": element,
                "interval": name,
                "auc": round(auc_val, 4) if not np.isnan(auc_val) else "nan",
                "tau": round(float(tau), 4) if not np.isnan(tau) else "nan"
            })
            
    # Write output to CSV
    out_csv = REPO_ROOT / "results/bury_comparison.csv"
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "core", "sapropel", "element", "interval", "auc", "tau"])
        writer.writeheader()
        writer.writerows(records)
        
    print(f"Bury comparison results saved to {out_csv}")
    
    # Print summary table by model and interval
    print("\nSummary of Empirical Performance by Model and Relative-Position Interval:")
    print("-" * 80)
    print(f"{'Model':<20} | {'Interval':<10} | {'Mean AUC':<10} | {'Mean Kendall Tau':<15}")
    print("-" * 80)
    
    unique_models = sorted(list(set(r["model"] for r in records)))
    for m in unique_models:
        for name in ["EARLY", "LATE", "FULL"]:
            match = [r for r in records if r["model"] == m and r["interval"] == name]
            aucs = [float(r["auc"]) for r in match if r["auc"] != "nan"]
            taus = [float(r["tau"]) for r in match if r["tau"] != "nan"]
            mean_auc = f"{np.mean(aucs):.4f}" if aucs else "nan"
            mean_tau = f"{np.mean(taus):.4f}" if taus else "nan"
            print(f"{m:<20} | {name:<10} | {mean_auc:<10} | {mean_tau:<15}")
        print("-" * 80)

if __name__ == "__main__":
    main()
