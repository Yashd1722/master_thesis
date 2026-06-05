"""
testing/plot_figures.py
Generates publication-quality figures for all models, datasets, and elements.
Matches the style of Bury et al. (2021) Figures 1 and 2.
"""
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Suppress warnings
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]

# Colors matching Bury et al.
COLOR_MODEL = "#7165D0"  # Purple (SDML/DL)
COLOR_VAR = "#E07B1A"    # Orange (Variance)
COLOR_AC = "#2D8A4E"     # Green (Lag-1 AC)
COLOR_RANDOM = "#AAAAAA" # Gray

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def plot_zenodo_results(json_path, out_dir):
    """
    Generates ROC Curve and Confusion Matrix for Zenodo (Test Set) results.
    Matches Bury Fig 2 (ROC) and Fig 1 F/G (Confusion Matrix).
    """
    data = load_json(json_path)
    model = data.get("model", "Unknown")
    dataset = data.get("dataset", "Unknown")
    
    # 1. ROC Curve Plot
    if "roc" in data:
        fig, ax = plt.subplots(figsize=(6, 6))
        fpr = np.array(data["roc"]["fpr"])
        tpr = np.array(data["roc"]["tpr"])
        auc = data.get("auc", 0.0)
        
        # Plot Model ROC (Purple)
        ax.plot(fpr, tpr, color=COLOR_MODEL, lw=2, label=f"{model} (AUC={auc:.3f})")
        
        # Random classifier line
        ax.plot([0, 1], [0, 1], color=COLOR_RANDOM, linestyle="--", lw=1.5, label="Random")
        
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curve — {model} / {dataset}", fontsize=14)
        ax.legend(loc="lower right", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Save ROC
        roc_path = out_dir / f"{model}_{dataset}_zenodo_roc.png"
        fig.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✅ Saved ROC: {roc_path.name}")

    # 2. Confusion Matrix Plot
    if "confusion_matrix" in data:
        cm = np.array(data["confusion_matrix"])
        class_names = ["Null", "Fold", "Hopf", "Transcritical"] # Standard Bury classes
        
        fig, ax = plt.subplots(figsize=(6, 5))
        # Use a blue colormap similar to Bury
        cmap = plt.cm.Blues
        
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               xlabel='Predicted Label',
               ylabel='True Label',
               title=f'Confusion Matrix — {model} / {dataset}')
        
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations.
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        cm_path = out_dir / f"{model}_{dataset}_zenodo_cm.png"
        fig.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✅ Saved CM:  {cm_path.name}")

def plot_pangaea_results(json_path, out_dir):
    """
    Generates 3-panel EWS plots for PANGAEA data.
    Matches Bury Fig 1 A-E (Variance, AC, Probability over time).
    Iterates through all Cores, Sapropels, and Elements.
    """
    data = load_json(json_path)
    model = data.get("model", "Unknown")
    dataset = data.get("dataset", "Unknown")
    
    if "cores" not in data:
        return

    for core_name, saps in data["cores"].items():
        for sap_id, elements in saps.items():
            for element, elem_data in elements.items():
                
                p_trans = np.array(elem_data.get("p_transition", []))
                variance = np.array(elem_data.get("variance", []))
                lag1_ac = np.array(elem_data.get("lag1_ac", []))
                
                if len(p_trans) == 0:
                    continue
                
                steps = np.arange(len(p_trans))
                ktau_var = elem_data.get("ktau_var", 0.0)
                ktau_ac = elem_data.get("ktau_ac", 0.0)
                
                # Create 3-panel figure
                fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
                fig.suptitle(f"{model.upper()} | {core_name} / {sap_id} / {element}", fontsize=14, y=0.98)
                
                # Panel 1: Variance
                axes[0].plot(steps, variance, color=COLOR_VAR, linewidth=2)
                axes[0].set_ylabel("Variance", fontsize=11)
                axes[0].text(0.05, 0.9, f"Kendall τ = {ktau_var:.2f}", transform=axes[0].transAxes, fontsize=10)
                axes[0].grid(True, alpha=0.3)
                
                # Panel 2: Lag-1 AC
                axes[1].plot(steps, lag1_ac, color=COLOR_AC, linewidth=2)
                axes[1].set_ylabel("Lag-1 AC", fontsize=11)
                axes[1].text(0.05, 0.9, f"Kendall τ = {ktau_ac:.2f}", transform=axes[1].transAxes, fontsize=10)
                axes[1].grid(True, alpha=0.3)
                
                # Panel 3: Probability
                axes[2].plot(steps, p_trans, color=COLOR_MODEL, linewidth=2)
                axes[2].axhline(0.5, color=COLOR_RANDOM, linestyle="--", linewidth=1.5)
                axes[2].set_ylabel("p(transition)", fontsize=11)
                axes[2].set_xlabel("Rolling Window Step", fontsize=11)
                axes[2].set_ylim(0, 1.05)
                axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save figure
                # Name: model_dataset_core_sap_element.png
                filename = f"{model}_{dataset}_{core_name}_{sap_id}_{element}.png"
                save_path = out_dir / filename
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  ✅ Saved EWS: {filename}")

def main():
    results_dir = REPO_ROOT / "results"
    if not results_dir.exists():
        print(f" Results directory not found: {results_dir}")
        return

    print(" Starting comprehensive plot generation...")
    
    # Find all result.json files
    for json_path in sorted(results_dir.rglob("result.json")):
        out_dir = json_path.parent
        print(f"\n📂 Processing: {out_dir.name}")
        
        try:
            data = load_json(json_path)
            target = data.get("target", "unknown")
            
            if target == "zenodo":
                plot_zenodo_results(json_path, out_dir)
            elif target == "pangaea":
                plot_pangaea_results(json_path, out_dir)
            else:
                # Fallback for older formats
                if "cores" in data:
                    plot_pangaea_results(json_path, out_dir)
                elif "roc" in data or "confusion_matrix" in data:
                    plot_zenodo_results(json_path, out_dir)
                    
        except Exception as e:
            print(f"  ⚠️ Error processing {json_path}: {e}")

    print("\n🎉 All plots generated successfully!")

if __name__ == "__main__":
    main()
