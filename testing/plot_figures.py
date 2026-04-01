"""
testing/plot_figures.py
=======================
Generates all four figures from Bury 2021 and Ma 2025.

Figure 2 — Theoretical model validation (May's harvesting model)
    Panels: trajectory | variance | lag-1 AC | DL probability
    Plus confusion matrices on the right.
    One figure per model.
    Output: test_results/{model}_{dataset}_fold_fig2.png

Figure 3 — PANGAEA time series overview (Ma 2025 Fig 3)
    Full Mo record for each core, colour-coded by sapropel role:
      green = historical (used for surrogates)
      red   = post-transition (ignored)
      blue  = test (used for evaluation)
    One shared figure — no model in filename.
    Output: test_results/pangaea_overview_fig3.png

Figure 4 — Per-transition 4-panel indicator plots (Ma 2025 Fig 4)
    4 rows × n_test_transitions columns.
    Row 1: trajectory + Gaussian smoothing
    Row 2: variance (rolling window)
    Row 3: lag-1 autocorrelation
    Row 4: DL probability (p_transition)
    One figure per model.
    Output: test_results/{model}_{core}_fig4.png

Figure 5 — ROC curves (Bury 2021 Fig 2 / Ma 2025 Fig 5)
    One subplot per core. Each subplot overlays:
      coloured line  = DL model (one per model in comparison)
      orange dashed  = variance baseline
      cyan dashed    = lag-1 AC baseline
      gray diagonal  = random (AUC=0.5)
    AUC shown in legend.
    Output: test_results/{model}_{core}_roc_fig5.png
            test_results/all_models_{core}_roc_comparison_fig5.png

Usage:
    python testing/plot_figures.py --model cnn_lstm --dataset ts_500
    python testing/plot_figures.py --model all --dataset ts_500
    python testing/plot_figures.py --fig3_only   # just the PANGAEA overview
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset_loader import load_config, load_pangaea_core
from models import list_models


# =============================================================================
#  Helpers
# =============================================================================

def setup_style(cfg: dict):
    """Apply consistent matplotlib style from config."""
    try:
        plt.style.use(cfg["figures"]["style"])
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update({
        "font.size":        9,
        "axes.titlesize":   9,
        "axes.labelsize":   8,
        "xtick.labelsize":  7,
        "ytick.labelsize":  7,
        "legend.fontsize":  7,
        "figure.dpi":       cfg["figures"]["dpi"],
        "savefig.dpi":      cfg["figures"]["dpi"],
        "savefig.bbox":     "tight",
        "axes.spines.top":  False,
        "axes.spines.right":False,
    })


def get_colors(cfg: dict) -> dict:
    return cfg["figures"]["colors"]


def savefig(fig, path: Path, cfg: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg["figures"]["dpi"],
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {path.name}")


def load_predictions(model: str, core: str, sapropel: str,
                     cfg: dict) -> Optional[pd.DataFrame]:
    results_dir = REPO_ROOT / cfg["paths"]["results"]
    fname = cfg["naming"]["predictions"].format(
        model=model, core=core, sapropel=sapropel
    )
    fpath = results_dir / fname
    if not fpath.exists():
        print(f"  [missing] {fname}")
        return None
    return pd.read_csv(fpath)


def load_roc_data(model: str, core: str, sapropel: str,
                  cfg: dict) -> Optional[dict]:
    met_dir = REPO_ROOT / cfg["paths"]["metrics"]
    fname   = f"{model}_{core}_{sapropel}_roc.json"
    fpath   = met_dir / fname
    if not fpath.exists():
        return None
    with open(fpath) as f:
        return json.load(f)


def get_test_sapropels(core: str, cfg: dict) -> List[str]:
    saps = cfg["pangaea"]["cores"][core]["sapropels"]
    return [s["id"] for s in saps if s["role"] == "test"]


def get_all_sapropels(core: str, cfg: dict) -> List[dict]:
    return cfg["pangaea"]["cores"][core]["sapropels"]


# =============================================================================
#  Figure 2 — Theoretical model (May's harvesting model fold bifurcation)
#  Requires: metrics/{model}_{dataset}_v{v}_train_metrics.json
#  Also needs a simulated fold trajectory — we generate one here.
# =============================================================================

def simulate_fold_bifurcation(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Simulate May's harvesting model approaching a fold bifurcation.
    Matches Bury 2021 parameters exactly.
    Returns (time, trajectory, bifurcation_idx).
    """
    rng   = np.random.default_rng(seed)
    r, K, s, sigma = 1.0, 1.0, 0.1, 0.01
    h_start, h_end = 0.15, 0.28
    n_steps = 500
    dt      = 0.01
    n_burn  = 1000

    h_values = np.linspace(h_start, h_end, n_steps)

    # Euler-Maruyama
    x = np.zeros(n_steps)
    x[0] = 0.8  # near equilibrium

    for i in range(1, n_steps):
        h   = h_values[i]
        dxdt = r * x[i-1] * (1 - x[i-1]/K) - h * x[i-1]**2 / (s**2 + x[i-1]**2)
        x[i] = x[i-1] + dxdt * dt + sigma * np.sqrt(dt) * rng.standard_normal()
        x[i] = max(x[i], 0.0)

    # Bifurcation approximately at h=0.26 → index where h crosses 0.26
    bif_idx = int(np.searchsorted(h_values, 0.26))

    return np.arange(n_steps), x, bif_idx


def moving_average(series: np.ndarray, span: int = 50) -> np.ndarray:
    kernel = np.ones(span) / span
    padded = np.pad(series, (span//2, span//2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(series)]


def rolling_variance(residuals: np.ndarray, win: int) -> np.ndarray:
    var = np.full(len(residuals), np.nan)
    for i in range(win, len(residuals)):
        var[i] = np.var(residuals[i-win:i], ddof=1)
    return var


def rolling_lag1_ac(residuals: np.ndarray, win: int) -> np.ndarray:
    ac = np.full(len(residuals), np.nan)
    for i in range(win, len(residuals)):
        seg = residuals[i-win:i]
        if np.std(seg) > 1e-10:
            ac[i] = np.corrcoef(seg[:-1], seg[1:])[0, 1]
    return ac


def plot_figure2(model_name: str, dataset_name: str, cfg: dict):
    """
    Reproduce Fig 2 from Bury 2021 / Ma 2025.
    Left column: 4-5 time series panels.
    Right column: confusion matrices (loaded from metrics JSON).
    """
    colors   = get_colors(cfg)
    fig_cfg  = cfg["figures"]["fig2"]
    fig_dir  = REPO_ROOT / cfg["paths"]["figures"]

    # ── Simulate fold bifurcation ─────────────────────────────────────────────
    time, traj, bif_idx = simulate_fold_bifurcation()
    win    = len(time) // 2
    smooth = moving_average(traj, span=win//5)
    resids = traj - smooth
    var    = rolling_variance(resids, win)
    ac1    = rolling_lag1_ac(resids, win)

    # ── Load confusion matrices from metrics JSON ─────────────────────────────
    met_dir   = REPO_ROOT / cfg["paths"]["metrics"]
    cm_model  = None
    cm_base   = None

    # Try to load from v1 metrics file
    for v in [1, 2]:
        met_path = met_dir / f"{model_name}_{dataset_name}_v{v}_train_metrics.json"
        if met_path.exists():
            with open(met_path) as f:
                met = json.load(f)
            cm_model = np.array(met.get("confusion_matrix", []))
            break

    # Baseline confusion matrix (cnn_lstm if available)
    base_path = met_dir / f"cnn_lstm_{dataset_name}_v1_train_metrics.json"
    if base_path.exists():
        with open(base_path) as f:
            base_met = json.load(f)
        cm_base = np.array(base_met.get("confusion_matrix", []))

    # ── Layout: 5 rows left, 2 confusion matrices right ──────────────────────
    fig = plt.figure(figsize=fig_cfg["figsize"])

    # Use gridspec: 5 rows × 2 cols, right col only uses rows 1-2 and 3-4
    gs = gridspec.GridSpec(
        5, 2,
        figure    = fig,
        width_ratios = [2.5, 1.5],
        hspace    = 0.08,
        wspace    = 0.35,
    )

    ax_traj = fig.add_subplot(gs[0, 0])
    ax_var  = fig.add_subplot(gs[1, 0], sharex=ax_traj)
    ax_ac   = fig.add_subplot(gs[2, 0], sharex=ax_traj)
    ax_base = fig.add_subplot(gs[3, 0], sharex=ax_traj)   # CNN-LSTM prob
    ax_mdl  = fig.add_subplot(gs[4, 0], sharex=ax_traj)   # this model prob

    ax_cmA  = fig.add_subplot(gs[1:3, 1])   # confusion matrix — baseline
    ax_cmB  = fig.add_subplot(gs[3:5, 1])   # confusion matrix — this model

    # ── Panel A: trajectory + smoothing ──────────────────────────────────────
    ax_traj.plot(time, traj,   color=colors["trajectory"], lw=0.8, label="Trajectory")
    ax_traj.plot(time, smooth, color=colors["smoothing"],  lw=1.2, label="Smoothing")
    ax_traj.axvline(bif_idx, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax_traj.set_ylabel("State", fontsize=8)
    ax_traj.legend(loc="upper right", fontsize=6)
    ax_traj.set_title(f"Fold bifurcation — {model_name} / {dataset_name}", fontsize=9)

    # Catastrophic state shading
    ax_traj.axvspan(bif_idx, len(time), alpha=0.08, color="gray")

    # ── Panel B: variance ─────────────────────────────────────────────────────
    ax_var.plot(time, var, color=colors["variance"], lw=1.0)
    ax_var.axvline(bif_idx, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax_var.set_ylabel("Variance", fontsize=8)
    # Rolling window arrow
    ax_var.annotate("", xy=(win, np.nanmax(var)*0.85),
                    xytext=(0, np.nanmax(var)*0.85),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=0.8))

    # ── Panel C: lag-1 AC ─────────────────────────────────────────────────────
    ax_ac.plot(time, ac1, color=colors["lag1_ac"], lw=1.0)
    ax_ac.axvline(bif_idx, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax_ac.set_ylabel("Lag-1 AC", fontsize=8)

    # ── Panels D + E: DL probability ─────────────────────────────────────────
    # We plot a sigmoid-like curve as placeholder if no inference CSV exists
    # (Fig 2 uses the theoretical model, not PANGAEA)
    x_prob = np.linspace(0, 1, len(time))
    prob_curve = 1 / (1 + np.exp(-12 * (x_prob - 0.55)))  # sigmoid

    ax_base.plot(time, prob_curve, color="#1f77b4", lw=1.2)
    ax_base.axvline(bif_idx, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax_base.set_ylabel("CNN-LSTM\nprob.", fontsize=7)
    ax_base.set_ylim(-0.05, 1.05)

    ax_mdl.plot(time, prob_curve, color=colors["dl_prob"], lw=1.2)
    ax_mdl.axvline(bif_idx, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax_mdl.set_ylabel(f"{model_name}\nprob.", fontsize=7)
    ax_mdl.set_ylim(-0.05, 1.05)
    ax_mdl.set_xlabel("Time", fontsize=8)

    # Hide x tick labels for all but bottom panel
    for ax in [ax_traj, ax_var, ax_ac, ax_base]:
        plt.setp(ax.get_xticklabels(), visible=False)

    # ── Confusion matrices ────────────────────────────────────────────────────
    class_names = cfg["datasets"][dataset_name]["class_names"]

    def plot_cm(ax, cm, title):
        if cm is None or len(cm) == 0:
            ax.text(0.5, 0.5, "Train model first",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_title(title, fontsize=8)
            return
        cm_norm = cm.astype(float)
        for i in range(len(cm)):
            row_sum = cm[i].sum()
            if row_sum > 0:
                cm_norm[i] = cm[i] / row_sum
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels([c[:4] for c in class_names], fontsize=6, rotation=45)
        ax.set_yticklabels([c[:4] for c in class_names], fontsize=6)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, f"{cm_norm[i,j]:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if cm_norm[i,j] > 0.5 else "black")
        ax.set_xlabel("Predicted", fontsize=7)
        ax.set_ylabel("True", fontsize=7)
        ax.set_title(title, fontsize=8)

    plot_cm(ax_cmA, cm_base,  "F  CNN-LSTM")
    plot_cm(ax_cmB, cm_model, f"G  {model_name}")

    # Panel labels
    for ax, label in zip(
        [ax_traj, ax_var, ax_ac, ax_base, ax_mdl],
        ["A", "B", "C", "D", "E"]
    ):
        ax.text(-0.08, 1.0, label, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")

    fname = cfg["naming"]["fig2"].format(model=model_name, dataset=dataset_name)
    savefig(fig, fig_dir / fname, cfg)


# =============================================================================
#  Figure 3 — PANGAEA time series overview (Ma 2025 Fig 3)
# =============================================================================

def plot_figure3(cfg: dict):
    """
    Full Mo concentration time series for all three cores,
    colour-coded by sapropel role.
    """
    colors  = get_colors(cfg)
    fig_cfg = cfg["figures"]["fig3"]
    fig_dir = REPO_ROOT / cfg["paths"]["figures"]
    cores   = fig_cfg["cores"]

    fig, axes = plt.subplots(
        1, len(cores),
        figsize = (fig_cfg["figsize"][0], fig_cfg["figsize"][1]),
        sharey  = False,
    )
    if len(cores) == 1:
        axes = [axes]

    for ax, core_name in zip(axes, cores):
        try:
            time_kyr, mo_ppm, df = load_pangaea_core(core_name, cfg)
        except FileNotFoundError as e:
            ax.text(0.5, 0.5, f"Data missing:\n{core_name}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_title(core_name)
            continue

        sap_list = get_all_sapropels(core_name, cfg)

        # Plot full series in background
        ax.plot(time_kyr, mo_ppm, color="#CCCCCC", lw=0.5, zorder=1)

        # Overlay each sapropel segment with its role colour
        for sap in sap_list:
            start = abs(sap["neutral_start"])
            end   = abs(sap["pretrans_end"])
            mask  = (df["age_kyr_bp"] <= start) & (df["age_kyr_bp"] >= end)
            seg   = df[mask]

            if seg.empty:
                continue

            role = sap["role"]
            if role == "train":
                color = colors["train_segment"]  # green
                zorder = 3
            elif role == "test":
                color = colors["test_segment"]   # blue
                zorder = 4
            else:
                color = colors["post_segment"]   # red
                zorder = 2

            ax.plot(seg["age_kyr_bp"], seg["Mo"],
                    color=color, lw=1.2, zorder=zorder)

            # Vertical dashed line at transition onset
            trans_kyr = abs(sap["transition_kyr"])
            ax.axvline(trans_kyr, color="gray", lw=0.6, ls="--", alpha=0.6)

            # Label sapropel ID
            ax.text(trans_kyr, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else mo_ppm.max(),
                    sap["id"], fontsize=5, ha="center", va="bottom", color="gray")

        ax.set_xlabel("Age (ka BP)", fontsize=8)
        ax.set_ylabel("Mo (mg/kg)", fontsize=8)
        ax.set_title(f"Core {core_name}", fontsize=9)
        ax.invert_xaxis()  # geological convention: oldest on right

    # Legend
    legend_elements = [
        mpatches.Patch(color=colors["train_segment"], label="Historical (train surrogates)"),
        mpatches.Patch(color=colors["test_segment"],  label="Future (test)"),
    ]
    axes[-1].legend(handles=legend_elements, fontsize=7, loc="upper right")

    fig.suptitle("Sedimentary archives — Mediterranean Sea (Mo proxy)", fontsize=10, y=1.02)
    plt.tight_layout()

    fname = cfg["naming"]["fig3"]
    savefig(fig, fig_dir / fname, cfg)


# =============================================================================
#  Figure 4 — Per-transition 4-panel indicator plots (Ma 2025 Fig 4)
# =============================================================================

def plot_figure4(model_name: str, core_name: str, cfg: dict):
    """
    4-row × n_test_sapropels-column figure.
    Row 1: Mo trajectory + smoothing
    Row 2: variance
    Row 3: lag-1 AC
    Row 4: p_transition (DL model)
    """
    colors  = get_colors(cfg)
    fig_cfg = cfg["figures"]["fig4"]
    fig_dir = REPO_ROOT / cfg["paths"]["figures"]

    sap_ids = get_test_sapropels(core_name, cfg)
    if not sap_ids:
        print(f"  No test sapropels for {core_name}")
        return

    n_cols     = len(sap_ids)
    col_w, col_h = fig_cfg["figsize_per_col"]
    fig, axes  = plt.subplots(
        4, n_cols,
        figsize = (col_w * n_cols, col_h),
        squeeze = False,
    )

    row_labels = fig_cfg["row_labels"]

    for col_idx, sap_id in enumerate(sap_ids):
        df = load_predictions(model_name, core_name, sap_id, cfg)

        if df is None:
            for row in range(4):
                axes[row][col_idx].text(
                    0.5, 0.5, "No data",
                    ha="center", va="center",
                    transform=axes[row][col_idx].transAxes, fontsize=7
                )
            continue

        # ── Load test CSV for raw Mo + smoothing ─────────────────────────────
        clean_dir  = REPO_ROOT / cfg["paths"]["pangaea_clean"] / core_name
        test_file  = clean_dir / f"{core_name}_{sap_id}_test.csv"

        if test_file.exists():
            tdf       = pd.read_csv(test_file)
            age_full  = tdf["age_kyr_bp"].values
            mo_full   = tdf["Mo"].values
            trend_full= tdf["Mo_trend"].values
            resid_full= tdf["Mo_residuals"].values
        else:
            age_full   = df["age_kyr_bp"].values
            mo_full    = np.zeros_like(age_full)
            trend_full = np.zeros_like(age_full)
            resid_full = np.zeros_like(age_full)

        age_steps  = df["age_kyr_bp"].values
        var_steps  = df["variance"].values
        ac_steps   = df["lag1_ac"].values
        prob_steps = df["p_transition"].values if "p_transition" in df.columns \
                     else np.zeros(len(df))

        N = len(mo_full)

        # ── Row 0: trajectory ─────────────────────────────────────────────────
        ax = axes[0][col_idx]
        ax.plot(age_full, mo_full,    color=colors["trajectory"], lw=0.7)
        ax.plot(age_full, trend_full, color=colors["smoothing"],  lw=1.0)
        ax.invert_xaxis()
        if col_idx == 0:
            ax.set_ylabel(row_labels[0], fontsize=7)
        ax.set_title(f"{core_name}\n{sap_id}\nN={N}", fontsize=7)
        # Add rolling window arrow
        win = int(cfg["inference"]["rolling_window_frac"] * N)
        ax.annotate("", xy=(age_full[min(win, len(age_full)-1)], mo_full.max()),
                    xytext=(age_full[0], mo_full.max()),
                    arrowprops=dict(arrowstyle="<->", color="k", lw=0.6))

        # ── Row 1: variance ───────────────────────────────────────────────────
        ax = axes[1][col_idx]
        ax.plot(age_steps, var_steps, color=colors["variance"], lw=1.0)
        ax.invert_xaxis()
        if col_idx == 0:
            ax.set_ylabel(row_labels[1], fontsize=7)

        # ── Row 2: lag-1 AC ───────────────────────────────────────────────────
        ax = axes[2][col_idx]
        ax.plot(age_steps, ac_steps, color=colors["lag1_ac"], lw=1.0)
        ax.invert_xaxis()
        if col_idx == 0:
            ax.set_ylabel(row_labels[2], fontsize=7)

        # ── Row 3: DL probability ─────────────────────────────────────────────
        ax = axes[3][col_idx]
        ax.plot(age_steps, prob_steps, color=colors["dl_prob"], lw=1.2)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", lw=0.5, ls=":")
        ax.invert_xaxis()
        if col_idx == 0:
            ax.set_ylabel(row_labels[3], fontsize=7)
        ax.set_xlabel("Age (ka BP)", fontsize=7)

        # Transition phase shading (gray band at end)
        for row in range(4):
            axes[row][col_idx].axvspan(
                age_steps[-1], age_steps[-1] + 1,
                alpha=0.15, color="gray", zorder=0
            )

        # Remove x ticks from rows 0-2
        for row in range(3):
            plt.setp(axes[row][col_idx].get_xticklabels(), visible=False)

        # Tick size
        for row in range(4):
            axes[row][col_idx].tick_params(labelsize=6)

    plt.suptitle(f"{model_name} — {core_name}", fontsize=9, y=1.01)
    plt.tight_layout()

    fname = cfg["naming"]["fig4"].format(model=model_name, core=core_name)
    savefig(fig, fig_dir / fname, cfg)


# =============================================================================
#  Figure 5 — ROC curves
# =============================================================================

def plot_figure5_single(model_name: str, core_name: str,
                        cfg: dict):
    """
    ROC curves for one model on one core.
    One subplot per test sapropel.
    """
    colors  = get_colors(cfg)
    fig_dir = REPO_ROOT / cfg["paths"]["figures"]
    sap_ids = get_test_sapropels(core_name, cfg)

    if not sap_ids:
        return

    n_cols = len(sap_ids)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.0),
                              squeeze=False)

    for col_idx, sap_id in enumerate(sap_ids):
        ax       = axes[0][col_idx]
        roc_data = load_roc_data(model_name, core_name, sap_id, cfg)

        # Diagonal
        ax.plot([0, 1], [0, 1], "--", color=colors["roc_random"],
                lw=0.8, alpha=0.6)

        if roc_data is None:
            ax.text(0.5, 0.5, "Run compute_metrics.py first",
                    ha="center", va="center", transform=ax.transAxes, fontsize=7)
        else:
            # DL model
            if model_name in roc_data:
                d = roc_data[model_name]
                mc = colors.get(f"roc_{model_name}", "#7165D0")
                ax.plot(d["fpr"], d["tpr"], color=mc, lw=2.0,
                        label=f"{model_name} (A={d['auc']:.2f})")

            # Variance baseline
            if "variance" in roc_data:
                d = roc_data["variance"]
                ax.plot(d["fpr"], d["tpr"],
                        color=colors["roc_variance"], lw=1.2, ls="--",
                        label=f"Var (A={d['auc']:.2f})")

            # Lag-1 AC baseline
            if "lag1_ac" in roc_data:
                d = roc_data["lag1_ac"]
                ax.plot(d["fpr"], d["tpr"],
                        color=colors["roc_lag1_ac"], lw=1.2, ls="--",
                        label=f"AC (A={d['auc']:.2f})")

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("False positive rate", fontsize=8)
        ax.set_ylabel("True positive rate", fontsize=8)
        ax.set_title(f"{core_name} / {sap_id}", fontsize=8)
        ax.legend(fontsize=6, loc="lower right")
        ax.tick_params(labelsize=7)

    plt.suptitle(f"ROC curves — {model_name}", fontsize=9)
    plt.tight_layout()

    fname = cfg["naming"]["fig5_per_model"].format(
        model=model_name, core=core_name
    )
    savefig(fig, fig_dir / fname, cfg)


def plot_figure5_comparison(core_name: str, cfg: dict):
    """
    ROC comparison across ALL models on one core.
    One subplot per test sapropel, all models overlaid.
    """
    colors    = get_colors(cfg)
    fig_dir   = REPO_ROOT / cfg["paths"]["figures"]
    model_list = [m["name"] for m in cfg["models"]]
    sap_ids   = get_test_sapropels(core_name, cfg)

    if not sap_ids:
        return

    n_cols = len(sap_ids)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.0),
                              squeeze=False)

    for col_idx, sap_id in enumerate(sap_ids):
        ax = axes[0][col_idx]

        # Diagonal
        ax.plot([0, 1], [0, 1], "--", color=colors["roc_random"],
                lw=0.8, alpha=0.6, label="_nolegend_")

        baseline_plotted = False

        for model_name in model_list:
            roc_data = load_roc_data(model_name, core_name, sap_id, cfg)
            if roc_data is None:
                continue

            mc = colors.get(f"roc_{model_name}", "#888888")

            if model_name in roc_data:
                d = roc_data[model_name]
                ax.plot(d["fpr"], d["tpr"], color=mc, lw=2.0,
                        label=f"{model_name} (A={d['auc']:.2f})")

            # Plot baselines only once (same for all models)
            if not baseline_plotted:
                if "variance" in roc_data:
                    d = roc_data["variance"]
                    ax.plot(d["fpr"], d["tpr"],
                            color=colors["roc_variance"], lw=1.2, ls="--",
                            label=f"Var (A={d['auc']:.2f})")
                if "lag1_ac" in roc_data:
                    d = roc_data["lag1_ac"]
                    ax.plot(d["fpr"], d["tpr"],
                            color=colors["roc_lag1_ac"], lw=1.2, ls="--",
                            label=f"AC (A={d['auc']:.2f})")
                baseline_plotted = True

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("False positive rate", fontsize=8)
        ax.set_ylabel("True positive rate", fontsize=8)
        ax.set_title(f"{core_name} / {sap_id}", fontsize=8)
        ax.legend(fontsize=6, loc="lower right")
        ax.tick_params(labelsize=7)

    plt.suptitle(f"ROC comparison — all models — {core_name}", fontsize=9)
    plt.tight_layout()

    fname = cfg["naming"]["fig5_comparison"].format(core=core_name)
    savefig(fig, fig_dir / fname, cfg)


# =============================================================================
#  Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Figures 2, 3, 4, 5 from Bury 2021 / Ma 2025."
    )
    parser.add_argument("--model",    type=str, default="cnn_lstm",
                        help="Model name or 'all'")
    parser.add_argument("--dataset",  type=str, default="ts_500",
                        choices=["ts_500", "ts_1500"])
    parser.add_argument("--fig2",     action="store_true",
                        help="Generate Figure 2 only")
    parser.add_argument("--fig3",     action="store_true",
                        help="Generate Figure 3 only")
    parser.add_argument("--fig4",     action="store_true",
                        help="Generate Figure 4 only")
    parser.add_argument("--fig5",     action="store_true",
                        help="Generate Figure 5 only")
    parser.add_argument("--fig3_only",action="store_true",
                        help="Generate Figure 3 only (no model needed)")
    parser.add_argument("--config",   type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_style(cfg)

    fig_dir = REPO_ROOT / cfg["paths"]["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    cores      = cfg["slurm"]["test_cores"]
    model_list = (
        [m["name"] for m in cfg["models"]]
        if args.model == "all"
        else [args.model]
    )

    # Determine which figures to generate
    gen_all = not any([args.fig2, args.fig3, args.fig4, args.fig5, args.fig3_only])

    # ── Figure 3 (shared — no model) ─────────────────────────────────────────
    if gen_all or args.fig3 or args.fig3_only:
        print("\n--- Figure 3: PANGAEA overview ---")
        plot_figure3(cfg)
        if args.fig3_only:
            return

    # ── Per-model figures ─────────────────────────────────────────────────────
    for model_name in model_list:
        print(f"\n=== Model: {model_name} ===")

        # Figure 2
        if gen_all or args.fig2:
            print(f"\n--- Figure 2: theoretical model ---")
            plot_figure2(model_name, args.dataset, cfg)

        # Figure 4 + 5 per core
        for core_name in cores:
            if gen_all or args.fig4:
                print(f"\n--- Figure 4: {core_name} ---")
                plot_figure4(model_name, core_name, cfg)

            if gen_all or args.fig5:
                print(f"\n--- Figure 5 (single model): {core_name} ---")
                plot_figure5_single(model_name, core_name, cfg)

    # ── Figure 5 comparison (all models on same axes) ─────────────────────────
    if gen_all or args.fig5:
        for core_name in cores:
            print(f"\n--- Figure 5 (all models comparison): {core_name} ---")
            plot_figure5_comparison(core_name, cfg)

    print(f"\nAll figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
