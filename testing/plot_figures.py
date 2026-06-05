"""
testing/plot_figures.py
Generates the scientific paper figures from the unified result.json files.
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.dataset_loader import load_config

def load_all_results(results_dir: Path, target: str) -> list:
    records = []
    for d in results_dir.iterdir():
        if d.is_dir() and d.name.endswith(f"_{target}"):
            f = d / "result.json"
            if f.exists():
                with open(f) as fp: records.append(json.load(fp))
    return records

def plot_roc_all_models(results_dir: Path, out_dir: Path, dataset: str):
    records = [r for r in load_all_results(results_dir, "zenodo") if r.get("dataset") == dataset]
    if not records: return
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in records:
        ax.plot(r["roc"]["fpr"], r["roc"]["tpr"], lw=1.5, label=f"{r['model']} ({r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {dataset}"); ax.legend(fontsize=8)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"fig1_roc_{dataset}.png", dpi=150); plt.close(fig)

def plot_pangaea_ews(results_dir: Path, out_dir: Path):
    for r in load_all_results(results_dir, "pangaea"):
        for core, saps in r.get("cores", {}).items():
            for sap, elems in saps.items():
                for elem, data in elems.items():
                    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
                    steps = np.arange(len(data["p_transition"]))
                    
                    axes[0].plot(steps, data["variance"], color="#E07B1A")
                    axes[0].set_ylabel("Variance")
                    axes[0].text(0.05, 0.9, f"Kendall τ = {data['ktau_var']:.2f}", transform=axes[0].transAxes)
                    
                    axes[1].plot(steps, data["lag1_ac"], color="#2D8A4E")
                    axes[1].set_ylabel("Lag-1 AC")
                    axes[1].text(0.05, 0.9, f"Kendall τ = {data['ktau_ac']:.2f}", transform=axes[1].transAxes)
                    
                    axes[2].plot(steps, data["p_transition"], color="#7165D0")
                    axes[2].axhline(0.5, color="gray", ls="--")
                    axes[2].set_ylabel("p(transition)"); axes[2].set_ylim(0, 1)
                    axes[2].set_xlabel("Rolling Window Step")
                    
                    fig.suptitle(f"{r['model']} | {core} / {sap} / {elem}")
                    fig.tight_layout()
                    core_dir = out_dir / core; core_dir.mkdir(parents=True, exist_ok=True)
                    fig.savefig(core_dir / f"{r['model']}_{sap}_{elem}_ews.png", dpi=150)
                    plt.close(fig)

def plot_speed_vs_auc(results_dir: Path, out_dir: Path):
    zenodo_records = load_all_results(results_dir, "zenodo")
    train_records = load_all_results(results_dir, "train")
    if not zenodo_records or not train_records: return
    
    model_auc, model_time = {}, {}
    for r in zenodo_records: model_auc[f"{r['model']}_{r['dataset']}"] = r.get("auc", float("nan"))
    for r in train_records: model_time[f"{r['model']}_{r['dataset']}"] = r.get("training_time_min", float("nan"))
    
    keys = [k for k in model_auc if k in model_time]
    if not keys: return
    
    tsc_names = {"rocket", "minirocket", "multirocket", "arsenal", "knn_dtw", "boss", "weasel", "shapelet", "proximity_forest", "ts_chief", "drcif", "tde", "hivecote"}
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for k in keys:
        m = k.split("_")[0]
        col = "#E07B1A" if m in tsc_names else "#1f77b4"
        ax.scatter(model_time[k], model_auc[k], color=col, s=80, zorder=3)
        ax.annotate(m, (model_time[k], model_auc[k]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        
    ax.set_xlabel("Training time (min)"); ax.set_ylabel("AUC")
    ax.set_title("Speed vs AUC Trade-off")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#1f77b4", label="DL"), Patch(color="#E07B1A", label="TSC")])
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig5_speed_vs_auc.png", dpi=150); plt.close(fig)

def main():
    cfg = load_config()
    results_dir = REPO_ROOT / cfg["paths"]["results"]
    comp_dir = results_dir / "comparison"
    
    print("Generating Paper Figures...")
    for ds in ["ts_500", "ts_1500"]: plot_roc_all_models(results_dir, comp_dir, ds)
    plot_pangaea_ews(results_dir, comp_dir)
    plot_speed_vs_auc(results_dir, comp_dir)
    print(f"✅ Done! Check {comp_dir}")

if __name__ == "__main__":
    main()
