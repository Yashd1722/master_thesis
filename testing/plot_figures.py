"""
testing/plot_figures.py
=======================
Per-experiment plots (called inline after evaluate.py saves result.json)
+ comparison plots across all models (called once all experiments done).

Per-experiment plots saved to: results/{model}_{dataset}_{metric}/
Comparison plots saved to:     results/comparison/
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
#  Per-experiment plots  (called immediately after each experiment)
# =============================================================================

def plot_roc(data: dict, out_dir: Path):
    """ROC curve for one model/dataset/experiment."""
    fpr = np.array(data["roc_fpr"])
    tpr = np.array(data["roc_tpr"])
    auc = data.get("auc", 0.0)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, color="#1f77b4",
            label=f"{data['model']} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#AAAAAA", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {data['model']} / {data['dataset']}")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(data: dict, out_dir: Path, class_names: list):
    cm = np.array(data["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {data['model']} / {data['dataset']}")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_pangaea_series(data: dict, result, out_dir: Path):
    """p_transition + variance + lag1-AC over time for one PANGAEA segment."""
    p_trans  = np.array(data["p_transition"])
    variance = np.array(data.get("variance", []))
    lag1_ac  = np.array(data.get("lag1_ac", []))
    steps    = np.arange(len(p_trans))

    n_panels = 1 + (1 if len(variance) else 0) + (1 if len(lag1_ac) else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 2.5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0
    if len(variance):
        axes[ax_idx].plot(steps, variance, color="#E07B1A")
        axes[ax_idx].set_ylabel("Variance")
        ax_idx += 1
    if len(lag1_ac):
        axes[ax_idx].plot(steps, lag1_ac, color="#2D8A4E")
        axes[ax_idx].set_ylabel("Lag-1 AC")
        ax_idx += 1

    axes[ax_idx].plot(steps, p_trans, color="#7165D0")
    axes[ax_idx].axhline(0.5, color="#AAAAAA", lw=1, ls="--")
    axes[ax_idx].set_ylabel("p(transition)")
    axes[ax_idx].set_ylim(0, 1)
    axes[ax_idx].set_xlabel("Rolling window step")

    tag = f"{data['core']} / {data['sapropel']} / {data['element']} / {data['segment']}"
    axes[0].set_title(f"{data['model']} — {tag}")
    fig.tight_layout()
    fig.savefig(out_dir / "pangaea_series.png", dpi=150)
    plt.close(fig)


# =============================================================================
#  Comparison plots  (called once all experiments are done)
#  Reads all result.json files from results/
# =============================================================================

def load_all_results(results_dir: Path, metric: str) -> list:
    """Load all result.json files matching a given metric."""
    records = []
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.endswith(f"_{metric}"):
            continue
        rfile = exp_dir / "result.json"
        if rfile.exists():
            with open(rfile) as f:
                records.append(json.load(f))
    return records


def plot_roc_all_models(results_dir: Path, out_dir: Path, dataset: str,
                        model_colors: dict):
    """Fig 1: ROC curves for all models on one dataset (Bury Fig. 2 analog)."""
    records = [r for r in load_all_results(results_dir, "auc")
               if r.get("dataset") == dataset and "core" not in r]
    if not records:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    for r in records:
        fpr = np.array(r["roc_fpr"])
        tpr = np.array(r["roc_tpr"])
        col = model_colors.get(r["model"], "#888888")
        ax.plot(fpr, tpr, lw=1.5, color=col,
                label=f"{r['model']} ({r['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="#AAAAAA", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curves — all models / {dataset}")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"fig1_roc_all_models_{dataset}.png", dpi=150)
    plt.close(fig)


def plot_auc_heatmap(results_dir: Path, out_dir: Path):
    """Fig 2: AUC heatmap — models x datasets (Bury Fig. 3 analog)."""
    records = [r for r in load_all_results(results_dir, "auc")
               if "core" not in r]
    if not records:
        return

    models   = sorted(set(r["model"]   for r in records))
    datasets = sorted(set(r["dataset"] for r in records))

    matrix = np.full((len(models), len(datasets)), np.nan)
    for r in records:
        i = models.index(r["model"])
        j = datasets.index(r["dataset"])
        matrix[i, j] = r["auc"]

    fig, ax = plt.subplots(figsize=(max(4, len(datasets) * 2),
                                     max(5, len(models) * 0.4)))
    im = ax.imshow(matrix, vmin=0.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    for i in range(len(models)):
        for j in range(len(datasets)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    fig.colorbar(im, label="AUC")
    ax.set_title("AUC Heatmap — all models x datasets")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig2_auc_heatmap.png", dpi=150)
    plt.close(fig)


def plot_model_ranking(results_dir: Path, out_dir: Path):
    """Fig 4: Mean AUC bar chart (Ma 2025 Fig. 3 analog)."""
    records = [r for r in load_all_results(results_dir, "auc")
               if "core" not in r]
    if not records:
        return

    from collections import defaultdict
    model_aucs = defaultdict(list)
    for r in records:
        model_aucs[r["model"]].append(r["auc"])

    models = sorted(model_aucs, key=lambda m: -np.mean(model_aucs[m]))
    means  = [np.mean(model_aucs[m]) for m in models]
    stds   = [np.std(model_aucs[m])  for m in models]

    tsc_names = {"rocket", "minirocket", "multirocket", "arsenal", "knn_dtw",
                 "boss", "weasel", "shapelet", "proximity_forest", "ts_chief",
                 "drcif", "tde", "hivecote"}
    colors = ["#E07B1A" if m in tsc_names else "#1f77b4" for m in models]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 0.6), 5))
    ax.bar(models, means, yerr=stds, color=colors,
           capsize=4, edgecolor="white", linewidth=0.5)
    ax.axhline(0.5, color="#AAAAAA", lw=1, ls="--", label="Chance")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Mean AUC (across datasets)")
    ax.set_title("Model Ranking — Mean AUC")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#1f77b4", label="DL"),
                        Patch(color="#E07B1A", label="TSC")])
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig4_model_ranking.png", dpi=150)
    plt.close(fig)


def plot_pangaea_all_models(results_dir: Path, out_dir: Path, core: str):
    """Fig 3: p_transition for all models on PANGAEA cores (Bury Fig. 4 analog)."""
    records = [r for r in load_all_results(results_dir, "kendall_tau")
               if r.get("core") == core]
    if not records:
        return

    sapropels = sorted(set(r["sapropel"] for r in records))

    fig, axes = plt.subplots(len(sapropels), 1,
                              figsize=(10, 3 * len(sapropels)), sharex=False)
    if len(sapropels) == 1:
        axes = [axes]

    for ax, sap in zip(axes, sapropels):
        sap_recs = [r for r in records if r["sapropel"] == sap]
        for r in sap_recs:
            p = np.array(r["p_transition"])
            ax.plot(np.linspace(0, 1, len(p)), p, lw=1, alpha=0.7,
                    label=r["model"])
        ax.axhline(0.5, color="#AAAAAA", lw=1, ls="--")
        ax.set_ylim(0, 1)
        ax.set_ylabel("p(transition)")
        ax.set_title(f"{core} / {sap}")
        ax.legend(fontsize=6, ncol=3)

    axes[-1].set_xlabel("Normalised time (0=start, 1=transition)")
    fig.suptitle(f"PANGAEA predictions — {core}", fontsize=12)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"fig3_pangaea_{core}.png", dpi=150)
    plt.close(fig)


def plot_speed_vs_auc(results_dir: Path, out_dir: Path, train_log_dir: Path):
    """Fig 5: Training time vs AUC scatter."""
    records = [r for r in load_all_results(results_dir, "auc")
               if "core" not in r]
    if not records:
        return

    from collections import defaultdict
    tsc_names = {"rocket", "minirocket", "multirocket", "arsenal", "knn_dtw",
                 "boss", "weasel", "shapelet", "proximity_forest", "ts_chief",
                 "drcif", "tde", "hivecote"}

    model_auc  = defaultdict(list)
    model_time = {}
    for r in records:
        model_auc[r["model"]].append(r.get("auc", float("nan")))

    for log_f in train_log_dir.glob("*_train.log"):
        model = log_f.stem.replace("_train", "").rsplit("_", 1)[0]
        try:
            text = log_f.read_text()
            for line in text.splitlines():
                if "time=" in line and "min" in line:
                    t = float(line.split("time=")[-1].replace("min", "").strip())
                    model_time[model] = t
        except Exception:
            pass

    models = [m for m in model_auc if m in model_time]
    if not models:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for m in models:
        auc = np.nanmean(model_auc[m])
        t   = model_time[m]
        col = "#E07B1A" if m in tsc_names else "#1f77b4"
        ax.scatter(t, auc, color=col, s=80, zorder=3)
        ax.annotate(m, (t, auc), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")

    ax.set_xlabel("Training time (min)")
    ax.set_ylabel("Mean AUC")
    ax.set_title("Speed vs AUC Trade-off")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#1f77b4", label="DL"),
                        Patch(color="#E07B1A", label="TSC")])
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig5_speed_vs_auc.png", dpi=150)
    plt.close(fig)


# =============================================================================
#  CLI — generate all comparison plots from saved results
# =============================================================================

def main():
    import argparse
    from src.dataset_loader import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg         = load_config(args.config)
    results_dir = REPO_ROOT / cfg["paths"]["results"]
    comp_dir    = results_dir / "comparison"
    log_dir     = REPO_ROOT / cfg["paths"]["logs"]

    dl_models  = cfg["models"]["dl"]
    tsc_models = cfg["models"]["tsc"]
    dl_cols  = ["#1f77b4", "#E07B1A", "#2D8A4E", "#7165D0"]
    tsc_cols = ["#17BECF", "#BCBD22", "#8C564B", "#E377C2",
                "#7F7F7F", "#AEC7E8", "#FFBB78", "#98DF8A",
                "#FF9896", "#C5B0D5", "#C49C94", "#F7B6D2", "#C7C7C7"]
    all_colors = {}
    for m, c in zip(dl_models, dl_cols):
        all_colors[m] = c
    for m, c in zip(tsc_models, tsc_cols):
        all_colors[m] = c

    for ds in cfg["slurm"]["train_datasets"]:
        print(f"ROC all models — {ds}")
        plot_roc_all_models(results_dir, comp_dir, ds, all_colors)

    print("AUC heatmap")
    plot_auc_heatmap(results_dir, comp_dir)

    print("Model ranking")
    plot_model_ranking(results_dir, comp_dir)

    for core in cfg["slurm"]["test_cores"]:
        print(f"PANGAEA — {core}")
        plot_pangaea_all_models(results_dir, comp_dir, core)

    print("Speed vs AUC")
    plot_speed_vs_auc(results_dir, comp_dir, log_dir)

    print(f"\nComparison plots saved to: {comp_dir}")


if __name__ == "__main__":
    main()
