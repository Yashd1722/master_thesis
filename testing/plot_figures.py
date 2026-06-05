"""
testing/plot_figures.py
Generates Bury 2021 style figures for PANGAEA empirical data.
Fig 1: Time series of Variance, Lag-1 AC, and 4-class probabilities.
Fig 2: ROC AUC summary with inset bar chart of late-window class proportions.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# Bury 2021 Colors
COLORS = {
    "Fold": "#7165D0",          # Purple
    "Hopf": "#E07B1A",          # Orange
    "Transcritical": "#17BECF", # Cyan
    "Null": "#AAAAAA",          # Gray
    "Variance": "#D62728",      # Red for indicators
    "AC": "#2CA02C",            # Green for indicators
}

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def plot_bury_fig1(data, out_path):
    """
    Replicates Bury Fig 1: Time series of Variance, Lag-1 AC, and Class Probabilities.
    """
    variance = np.array(data["variance"])
    lag1_ac = np.array(data["lag1_ac"])
    probs = np.array(data["per_class_probs"]) # Shape: (N, 4)
    class_names = data.get("class_names", ["null", "fold", "hopf", "transcritical"])
    
    steps = np.arange(len(variance))
    ktau_var = data.get("ktau_var", 0.0)
    ktau_ac = data.get("ktau_ac", 0.0)
    
    # Create 3-panel figure (Variance, AC, Probabilities)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"{data['model']} | {data['core']} / {data['sapropel']} / {data['element']}", 
                 fontsize=14, fontweight='bold')
    
    # Panel 1: Variance
    axes[0].plot(steps, variance, color=COLORS["Variance"], lw=2)
    axes[0].set_ylabel("Variance", fontsize=12)
    axes[0].text(0.05, 0.9, f"Kendall τ = {ktau_var:.2f}", 
                 transform=axes[0].transAxes, fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Lag-1 AC
    axes[1].plot(steps, lag1_ac, color=COLORS["AC"], lw=2)
    axes[1].set_ylabel("Lag-1 AC", fontsize=12)
    axes[1].text(0.05, 0.9, f"Kendall τ = {ktau_ac:.2f}", 
                 transform=axes[1].transAxes, fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Class Probabilities
    color_map = {
        "null": COLORS["Null"],
        "fold": COLORS["Fold"],
        "hopf": COLORS["Hopf"],
        "transcritical": COLORS["Transcritical"]
    }
    
    for i, cls in enumerate(class_names):
        label = cls.capitalize()
        color = color_map.get(cls, "#000000")
        ls = "--" if cls == "null" else "-"
        axes[2].plot(steps, probs[:, i], color=color, lw=2, label=label, ls=ls)
        
    axes[2].set_ylabel("Probability", fontsize=12)
    axes[2].set_xlabel("Rolling Window Step", fontsize=12)
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved Fig1: {out_path.name}")

def plot_bury_fig2(data, out_path):
    """
    Replicates Bury Fig 2 style: AUC summary with inset bar chart of late-window class proportions.
    """
    auc_any = data.get("roc_auc_any", 0.0)
    late_probs = np.array(data.get("late_window_mean_probs", [0.25, 0.25, 0.25, 0.25]))
    class_names = data.get("class_names", ["null", "fold", "hopf", "transcritical"])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Display the AUC prominently
    ax.text(0.5, 0.5, f"AUC (Any Transition)\n= {auc_any:.3f}", 
            ha='center', va='center', fontsize=20, fontweight='bold', 
            transform=ax.transAxes, color="#7165D0")
    
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Summary — {data['model']} | {data['element']}", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5) # Random classifier line
    ax.grid(True, alpha=0.3)
    
    # Inset Bar Chart: Late window mean probabilities
    color_map = {
        "null": COLORS["Null"],
        "fold": COLORS["Fold"],
        "hopf": COLORS["Hopf"],
        "transcritical": COLORS["Transcritical"]
    }
    bar_colors = [color_map.get(cls, "#000000") for cls in class_names]
    bar_labels = [c.capitalize() for c in class_names]
    
    # Create inset axes [left, bottom, width, height]
    ax_inset = fig.add_axes([0.55, 0.15, 0.35, 0.35]) 
    bars = ax_inset.bar(bar_labels, late_probs, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax_inset.set_ylabel("Proportion", fontsize=10)
    ax_inset.set_title("Late Window\n(80-100%)", fontsize=10, fontweight='bold')
    ax_inset.set_ylim(0, 1)
    ax_inset.tick_params(axis='x', labelsize=8, rotation=45)
    ax_inset.tick_params(axis='y', labelsize=8)
    
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ Saved Fig2: {out_path.name}")

def main():
    results_dir = REPO_ROOT / "results"
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return

    print("🎨 Starting generation of Bury-style PANGAEA plots...")
    
    # Find all pangaea result.json files
    for json_path in sorted(results_dir.rglob("*_pangaea/result.json")):
        data = load_json(json_path)
        out_dir = json_path.parent # Save in the same directory as result.json
        
        model = data["model"]
        dataset = data["dataset"]
        class_names = data.get("class_names", ["null", "fold", "hopf", "transcritical"])
        
        print(f"\n📂 Processing {json_path.parent.name}...")
        
        for core, saps in data["cores"].items():
            for sap, elements in saps.items():
                for element, elem_data in elements.items():
                    # Add metadata to elem_data for plotting
                    plot_data = {**elem_data, "model": model, "dataset": dataset, 
                                 "core": core, "sapropel": sap, "element": element,
                                 "class_names": class_names}
                    
                    # 1. Plot Fig 1 Style (Time Series)
                    fig1_name = f"{model}_{dataset}_{core}_{sap}_{element}_fig1.png"
                    fig1_path = out_dir / fig1_name
                    try:
                        plot_bury_fig1(plot_data, fig1_path)
                    except Exception as e:
                        print(f"  ⚠️ Failed to plot Fig1 for {element}: {e}")
                        
                    # 2. Plot Fig 2 Style (ROC Summary + Inset)
                    fig2_name = f"{model}_{dataset}_{core}_{sap}_{element}_fig2.png"
                    fig2_path = out_dir / fig2_name
                    try:
                        plot_bury_fig2(plot_data, fig2_path)
                    except Exception as e:
                        print(f"  ⚠️ Failed to plot Fig2 for {element}: {e}")

    print("\n🎉 All Bury-style plots generated successfully!")
    print(f"📁 Check your results folders: {results_dir}")

if __name__ == "__main__":
    main()
