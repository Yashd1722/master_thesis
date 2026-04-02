"""
testing/plot_figures.py
=======================
Generates all figures matching Bury 2021 and Ma 2025 exactly.

Figure 3 — PANGAEA time series overview (Ma 2025 Fig 3)
  Mo concentration per core, colour-coded by sapropel role:
    GREEN = historical (used for SDML surrogates)
    RED   = within-sapropel post-onset (not used)
    BLUE  = future test segments
  Vertical dashed lines at sapropel onsets.
  X-axis: negative ka BP, oldest LEFT, youngest RIGHT.
  Output: test_results/pangaea_overview_fig3.png

Figure 4 — Per-element 4-panel indicator plots (Ma 2025 Fig 4)
  One subplot per (element × test sapropel).
  4 rows: trajectory+smoothing | variance | lag-1 AC | p_transition
  Output: test_results/{model}_{core}_{element}_fig4.png

Figure 5 — ROC curves (Bury 2021 Fig 2 / Ma 2025 Fig 5)
  Per core, per element.
  Overlays: model curve + variance + lag-1 AC + diagonal.
  Inset: bar chart of mean p_transition for forced vs neutral.
  N shown on each subplot.
  Output: test_results/{model}_{core}_{element}_roc_fig5.png
          test_results/all_models_{core}_{element}_roc_fig5.png

Figure 2 — Theoretical model (fold bifurcation)
  5 panels: trajectory | variance | lag-1 AC | CNN-LSTM prob | model prob
  Confusion matrices on right.
  Output: test_results/{model}_{dataset}_fold_fig2.png

Usage:
  python testing/plot_figures.py --fig3_only
  python testing/plot_figures.py --model cnn_lstm --dataset ts_500
  python testing/plot_figures.py --model all --dataset ts_500
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
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset_loader import load_config, _build_zenodo_index, _split_index
from src.rolling_window import ELEMENTS, PRIMARY_ELEMENT
from models import list_models, get_model
import torch


# =============================================================================
#  Style helpers
# =============================================================================

def setup_style(cfg: dict):
    try:
        plt.style.use(cfg["figures"]["style"])
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update({
        "font.size":         9,
        "axes.titlesize":    9,
        "axes.labelsize":    8,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   7,
        "figure.dpi":        cfg["figures"]["dpi"],
        "savefig.dpi":       cfg["figures"]["dpi"],
        "savefig.bbox":      "tight",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def C(cfg, key):
    """Shorthand for cfg["figures"]["colors"][key]."""
    return cfg["figures"]["colors"][key]


def savefig(fig, path: Path, cfg: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg["figures"]["dpi"],
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {path.name}")


def get_test_sapropels(core: str, cfg: dict) -> List[dict]:
    return [s for s in cfg["pangaea"]["cores"][core]["sapropels"]
            if s["role"] == "test"]


def get_all_sapropels(core: str, cfg: dict) -> List[dict]:
    return cfg["pangaea"]["cores"][core]["sapropels"]


# =============================================================================
#  Data loaders
# =============================================================================

def load_xrf_full(core_name: str, cfg: dict) -> Optional[pd.DataFrame]:
    """Load full calibratedXRF file for Figure 3."""
    from src.pangea_cleaner import load_xrf, smooth_all_elements
    try:
        df = load_xrf(core_name, cfg)
        return smooth_all_elements(df, cfg["pangaea"]["bandwidth_years"])
    except FileNotFoundError as e:
        print(f"  [warn] {e}")
        return None


def load_segment_csv(core_name: str, sap_id: str, seg_type: str,
                     cfg: dict) -> Optional[pd.DataFrame]:
    """Load a segment CSV saved by pangea_cleaner.py."""
    clean_dir = REPO_ROOT / cfg["paths"]["pangaea_clean"] / core_name
    fname     = f"{core_name}_{sap_id}_{seg_type}.csv"
    fpath     = clean_dir / fname
    return pd.read_csv(fpath) if fpath.exists() else None


def load_pred(model: str, core: str, sapropel: str,
              element: str, seg_type: str, cfg: dict) -> Optional[pd.DataFrame]:
    results_dir = REPO_ROOT / cfg["paths"]["results"]
    fname = f"{model}_{core}_{sapropel}_{element}_{seg_type}.csv"
    fpath = results_dir / fname
    return pd.read_csv(fpath) if fpath.exists() else None


def load_roc(model: str, core: str, element: str,
             cfg: dict) -> Optional[dict]:
    met_dir = REPO_ROOT / cfg["paths"]["metrics"]
    fname   = f"{model}_{core}_{element}_roc.json"
    fpath   = met_dir / fname
    if not fpath.exists():
        return None
    with open(fpath) as f:
        return json.load(f)


# =============================================================================
#  FIGURE 3 — PANGAEA time series overview (Ma 2025 Fig 3)
# =============================================================================

def plot_figure3(cfg: dict):
    """
    Reproduces Ma 2025 Fig 3 exactly.
    Shows Mo concentration for each core with colour-coded sapropel roles.
    Green = historical (train surrogates)
    Red   = within-sapropel post-onset (not used)
    Blue  = test segments
    """
    fig_cfg = cfg["figures"]["fig3"]
    fig_dir = REPO_ROOT / cfg["paths"]["figures"]
    cores   = fig_cfg["cores"]

    fig, axes = plt.subplots(1, len(cores),
                              figsize=(fig_cfg["figsize"][0],
                                       fig_cfg["figsize"][1]))
    if len(cores) == 1:
        axes = [axes]

    panel_labels = ["A", "B", "C"]

    for col_idx, (ax, core_name) in enumerate(zip(axes, cores)):
        df = load_xrf_full(core_name, cfg)

        if df is None:
            ax.text(0.5, 0.5, f"Data missing\n{core_name}",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        age  = df["age_kyr_bp"].values
        mo   = df["Mo"].values

        # Plot full series in light gray background
        ax.plot(-age, mo, color="#CCCCCC", lw=0.4, zorder=1)

        # Colour-code each sapropel
        for sap in get_all_sapropels(core_name, cfg):
            sap_id = sap["id"]
            role   = sap["role"]

            # Full sapropel window = pretrans_start to pretrans_end
            p_start = abs(sap["pretrans_start"])
            p_end   = abs(sap["pretrans_end"])

            # Neutral (pre-sapropel) window
            n_start = abs(sap["neutral_start"])
            n_end   = abs(sap["neutral_end"])

            # Green: neutral/historical segment
            mask_n = (age <= n_start) & (age >= n_end)
            seg_n  = df[mask_n]
            if not seg_n.empty:
                color = C(cfg, "train_segment") if role == "train" else C(cfg, "test_segment")
                ax.plot(-seg_n["age_kyr_bp"], seg_n["Mo"],
                        color=color, lw=0.8, zorder=3)

            # Blue or red: the sapropel itself
            mask_s = (age <= p_start) & (age >= p_end)
            seg_s  = df[mask_s]
            if not seg_s.empty:
                if role == "test":
                    color = C(cfg, "test_segment")      # blue
                else:
                    color = C(cfg, "post_segment")      # red (historical peak)
                ax.plot(-seg_s["age_kyr_bp"], seg_s["Mo"],
                        color=color, lw=0.8, zorder=4)

            # Vertical dashed line at transition onset
            trans = -abs(sap["transition_kyr"])
            ax.axvline(trans, color="black", lw=0.7, ls="--", alpha=0.8, zorder=5)

            # Sapropel label above the dashed line
            y_top = np.nanmax(mo) * 1.02
            ax.text(trans, y_top, sap_id, fontsize=6,
                    ha="center", va="bottom", color="black")

        ax.set_xlabel("Time (ka BP)", fontsize=8)
        ax.set_ylabel("Mo [ppm]", fontsize=8)
        ax.set_title(f"Core {core_name}", fontsize=9)

        # Panel label
        ax.text(-0.10, 1.05, panel_labels[col_idx],
                transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

    # Legend
    legend_elements = [
        mpatches.Patch(color=C(cfg, "train_segment"),
                       label="Historical (train surrogates)"),
        mpatches.Patch(color=C(cfg, "post_segment"),
                       label="Post-transition (not used)"),
        mpatches.Patch(color=C(cfg, "test_segment"),
                       label="Future test"),
    ]
    axes[-1].legend(handles=legend_elements, fontsize=7,
                    loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fname = cfg["naming"]["fig3"]
    savefig(fig, fig_dir / fname, cfg)


# =============================================================================
#  FIGURE 4 — per-element 4-panel indicator plots (Ma 2025 Fig 4)
# =============================================================================

def plot_figure4(model_name: str, core_name: str, cfg: dict):
    """
    One figure per (model, core).
    Rows: trajectory | variance | lag-1 AC | p_transition
    Columns: one per test sapropel × one per element.
    """
    fig_cfg  = cfg["figures"]["fig4"]
    fig_dir  = REPO_ROOT / cfg["paths"]["figures"]
    test_saps= get_test_sapropels(core_name, cfg)

    if not test_saps:
        return

    # One figure per element (keeps figures readable)
    for element in ELEMENTS:
        n_cols = len(test_saps)
        col_w, col_h = fig_cfg["figsize_per_col"]
        fig, axes = plt.subplots(
            4, n_cols,
            figsize=(col_w * n_cols, col_h),
            squeeze=False,
        )
        row_labels = fig_cfg["row_labels"]

        for col_idx, sap in enumerate(test_saps):
            sap_id = sap["id"]

            # Load forced segment (has raw + smoothed + residuals)
            seg_df = load_segment_csv(core_name, sap_id, "forced", cfg)
            pred_f = load_pred(model_name, core_name, sap_id,
                               element, "forced", cfg)

            # ── Row 0: trajectory + smoothing ─────────────────────────────────
            ax = axes[0][col_idx]
            if seg_df is not None and element in seg_df.columns:
                age_full  = -seg_df["age_kyr_bp"].values  # negative for x-axis
                mo_full   = seg_df[element].values
                trend_col = f"{element}_trend"
                trend     = seg_df[trend_col].values if trend_col in seg_df.columns \
                            else np.zeros_like(mo_full)
                ax.plot(age_full, mo_full,  color=C(cfg,"trajectory"), lw=0.7)
                ax.plot(age_full, trend,    color=C(cfg,"smoothing"),  lw=1.0)
                N = len(mo_full)
                # Rolling window arrow
                win = max(2, int(cfg["inference"]["rolling_window_frac"] * N))
                ax.annotate("",
                    xy=(age_full[min(win, N-1)], np.nanmax(mo_full) * 0.95),
                    xytext=(age_full[0], np.nanmax(mo_full) * 0.95),
                    arrowprops=dict(arrowstyle="<->", color="k", lw=0.6))
                ax.text(0.5, 0.98, f"N={N}",
                        transform=ax.transAxes, fontsize=6,
                        ha="center", va="top")
            else:
                ax.text(0.5, 0.5, "No data",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=7)

            ax.set_title(f"{sap_id}", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(f"{element}\n{row_labels[0]}", fontsize=7)

            # ── Row 1: variance ────────────────────────────────────────────────
            ax = axes[1][col_idx]
            if pred_f is not None and "variance" in pred_f.columns:
                age_s = -pred_f["age_kyr_bp"].values
                ax.plot(age_s, pred_f["variance"].values,
                        color=C(cfg,"variance"), lw=1.0)
            if col_idx == 0:
                ax.set_ylabel(row_labels[1], fontsize=7)

            # ── Row 2: lag-1 AC ────────────────────────────────────────────────
            ax = axes[2][col_idx]
            if pred_f is not None and "lag1_ac" in pred_f.columns:
                age_s = -pred_f["age_kyr_bp"].values
                ax.plot(age_s, pred_f["lag1_ac"].values,
                        color=C(cfg,"lag1_ac"), lw=1.0)
            if col_idx == 0:
                ax.set_ylabel(row_labels[2], fontsize=7)

            # ── Row 3: DL probability ──────────────────────────────────────────
            ax = axes[3][col_idx]
            if pred_f is not None and "p_transition" in pred_f.columns:
                age_s = -pred_f["age_kyr_bp"].values
                ax.plot(age_s, pred_f["p_transition"].values,
                        color=C(cfg,"dl_prob"), lw=1.2)
                ax.set_ylim(-0.05, 1.05)
                ax.axhline(0.5, color="gray", lw=0.4, ls=":")
            if col_idx == 0:
                ax.set_ylabel(row_labels[3], fontsize=7)
            ax.set_xlabel("Time (ka BP)", fontsize=7)

            # Transition shading
            for row in range(4):
                axes[row][col_idx].tick_params(labelsize=6)
            for row in range(3):
                plt.setp(axes[row][col_idx].get_xticklabels(), visible=False)

        plt.suptitle(f"{model_name} — {core_name} — {element}",
                     fontsize=9, y=1.01)
        plt.tight_layout()

        fname = f"{model_name}_{core_name}_{element}_fig4.png"
        savefig(fig, fig_dir / fname, cfg)


# =============================================================================
#  FIGURE 5 — ROC curves matching paper style (Ma 2025 Fig 5)
# =============================================================================

def _roc_inset(ax, mean_p_forced: float, mean_p_neutral: float):
    """Add bar chart inset. Uses numeric y positions to avoid unit conflict."""
    inset  = ax.inset_axes([0.52, 0.05, 0.45, 0.32])
    ypos   = [0.0, 1.0]          # numeric positions, not strings
    vals   = [mean_p_neutral, mean_p_forced]
    colors = ["#888888", "#E07B1A"]
    inset.barh(ypos, vals, color=colors, height=0.6)
    inset.set_xlim(0, 1)
    inset.set_ylim(-0.5, 1.5)
    inset.set_yticks(ypos)
    inset.set_yticklabels(["Neutral", "Pre-tran"], fontsize=5)
    inset.set_xlabel("Prop.", fontsize=5)
    inset.tick_params(labelsize=5)
    for yp, val in zip(ypos, vals):
        inset.text(min(val + 0.03, 0.95), yp,
                   f"{val:.2f}", va="center", ha="left", fontsize=5)


def plot_figure5_single(model_name: str, core_name: str, cfg: dict):
    """
    ROC curves for one model on one core.
    Produces ONE FIGURE PER ELEMENT — individual, clean, matching paper style.
    Output: {model}_{core}_{element}_roc_fig5.png
    """
    fig_dir = REPO_ROOT / cfg["paths"]["figures"]

    for element in ELEMENTS:
        roc_data = load_roc(model_name, core_name, element, cfg)

        fig, ax = plt.subplots(figsize=(4.5, 4.5))

        # Diagonal
        ax.plot([0, 1], [0, 1], "--", color="#AAAAAA", lw=0.8, alpha=0.6)

        if roc_data is None:
            ax.text(0.5, 0.5, "Run compute_metrics.py first",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_title(f"{model_name} — {core_name} — {element}", fontsize=9)
            fname = f"{model_name}_{core_name}_{element}_roc_fig5.png"
            savefig(fig, fig_dir / fname, cfg)
            continue

        n_label = roc_data.get("N", "?")
        mc      = C(cfg, f"roc_{model_name}")

        if model_name in roc_data:
            d = roc_data[model_name]
            ax.plot(d["fpr"], d["tpr"], color=mc, lw=2.0,
                    label=f"$A_{{DL}}$={d['auc']:.2f}")

        if "variance" in roc_data:
            d = roc_data["variance"]
            ax.plot(d["fpr"], d["tpr"],
                    color=C(cfg, "roc_variance"), lw=1.2, ls="--",
                    label=f"$A_{{Var}}$={d['auc']:.2f}")

        if "lag1_ac" in roc_data:
            d = roc_data["lag1_ac"]
            ax.plot(d["fpr"], d["tpr"],
                    color=C(cfg, "roc_lag1_ac"), lw=1.2, ls="--",
                    label=f"$A_{{AC}}$={d['auc']:.2f}")

        _roc_inset(ax,
                   roc_data.get("mean_p_forced",  0.5),
                   roc_data.get("mean_p_neutral", 0.5))

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("False positive rate", fontsize=9)
        ax.set_ylabel("True positive rate", fontsize=9)
        ax.set_title(f"{core_name} — {element}", fontsize=9)
        ax.text(0.05, 0.05, f"N={n_label}", transform=ax.transAxes, fontsize=8)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
        ax.tick_params(labelsize=8)

        plt.suptitle(f"{model_name}", fontsize=9, y=1.01)
        plt.tight_layout()

        fname = f"{model_name}_{core_name}_{element}_roc_fig5.png"
        savefig(fig, fig_dir / fname, cfg)


def plot_figure5_comparison(core_name: str, element: str, cfg: dict):
    """
    ROC comparison across ALL models for one (core, element).
    One plot — all model curves + baselines overlaid.
    """
    fig_dir    = REPO_ROOT / cfg["paths"]["figures"]
    model_list = [m["name"] for m in cfg["models"]]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#AAAAAA", lw=0.8, alpha=0.6)

    baseline_done = False
    n_label       = 0

    for model_name in model_list:
        roc_data = load_roc(model_name, core_name, element, cfg)
        if roc_data is None:
            continue

        mc = C(cfg, f"roc_{model_name}")
        if model_name in roc_data:
            d = roc_data[model_name]
            ax.plot(d["fpr"], d["tpr"], color=mc, lw=2.0,
                    label=f"{model_name} ($A$={d['auc']:.2f})")
        n_label = roc_data.get("N", n_label)

        if not baseline_done:
            if "variance" in roc_data:
                d = roc_data["variance"]
                ax.plot(d["fpr"], d["tpr"],
                        color=C(cfg,"roc_variance"), lw=1.2, ls="--",
                        label=f"Var ($A$={d['auc']:.2f})")
            if "lag1_ac" in roc_data:
                d = roc_data["lag1_ac"]
                ax.plot(d["fpr"], d["tpr"],
                        color=C(cfg,"roc_lag1_ac"), lw=1.2, ls="--",
                        label=f"AC ($A$={d['auc']:.2f})")
            baseline_done = True

            _roc_inset(ax,
                       roc_data.get("mean_p_forced", 0.5),
                       roc_data.get("mean_p_neutral", 0.5))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("False positive", fontsize=9)
    ax.set_ylabel("True positive", fontsize=9)
    ax.set_title(f"{core_name} / {element}", fontsize=9)
    ax.text(0.05, 0.05, f"N={n_label}", transform=ax.transAxes, fontsize=8)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    fname = f"all_models_{core_name}_{element}_roc_fig5.png"
    savefig(fig, fig_dir / fname, cfg)


# =============================================================================
#  FIGURE 2 — Theoretical model (fold bifurcation)
# =============================================================================

# =============================================================================
#  Load one test sample per class from ts_500/ts_1500 cache
# =============================================================================

def load_test_sample_per_class(dataset_name: str, cfg: dict,
                                class_idx: int) -> Optional[np.ndarray]:
    """
    Load one residual sample of a given class from the numpy cache.
    Uses the test split so these are genuinely held-out samples.
    The ts_500 test set contains real simulated fold/hopf/transcritical/null
    trajectories — this is the correct "theoretical test" data from Zenodo.

    Returns normalised residuals array of shape (ts_len,) or None.
    """
    from src.dataset_loader import load_config as _lc, _build_zenodo_index, _split_index
    ds_cfg   = cfg["datasets"][dataset_name]
    ts_len   = ds_cfg["ts_length"]
    base_dir = REPO_ROOT / cfg["paths"][ds_cfg["path_key"]] / "combined"
    cache_X  = base_dir / "cache_residuals.npy"
    cache_y  = base_dir / "cache_labels.npy"

    if not cache_X.exists():
        print(f"  Cache not found: {cache_X}. Run build_cache.py first.")
        return None

    X = np.load(cache_X, mmap_mode="r")
    y = np.load(cache_y, mmap_mode="r")

    # Get test indices (reproducible split)
    index_df = _build_zenodo_index(base_dir)
    splits   = _split_index(index_df,
                             ds_cfg["train_frac"],
                             ds_cfg["val_frac"],
                             seed=cfg["project"]["seed"])
    test_df  = splits["test"]

    # Find first sample of the requested class in the test split
    cls_rows = test_df[test_df["class_label"] == class_idx]
    if cls_rows.empty:
        print(f"  No test sample found for class {class_idx}")
        return None

    cache_pos = int(cls_rows.iloc[0]["cache_pos"])
    return X[cache_pos].copy()   # shape (ts_len,)


@torch.no_grad()
def run_rolling_on_sample(residuals: np.ndarray, model,
                           device: torch.device,
                           ts_len: int, n_steps: int = 100) -> dict:
    """
    Run rolling window on one residual series and get model predictions.
    Returns dict with keys: variance, lag1_ac, probs (n_steps, n_classes)
    """
    from src.rolling_window import _variance, _lag1_ac, prepare_dl_input
    import torch

    n   = len(residuals)
    win = n // 2

    positions = np.linspace(win, n, n_steps, dtype=int)
    positions = np.clip(positions, win, n)

    variances, ac1s, dl_inputs = [], [], []
    for pos in positions:
        seg = residuals[pos - win: pos]
        variances.append(_variance(seg))
        ac1s.append(_lag1_ac(seg))
        dl_inputs.append(prepare_dl_input(residuals, pos, ts_len))

    # Run model
    X   = np.stack(dl_inputs)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)

    all_probs = []
    for i in range(0, len(X_t), 64):
        batch = X_t[i:i+64]
        all_probs.append(
            torch.softmax(model(batch), dim=-1).cpu().numpy()
        )
    probs = np.concatenate(all_probs, axis=0)

    return {
        "positions": positions,
        "variance":  np.array(variances),
        "lag1_ac":   np.array(ac1s),
        "probs":     probs,
    }


def plot_figure2(model_name: str, dataset_name: str, cfg: dict):
    """
    Reproduces Fig 2 from Bury 2021 / Fig 2 from Ma 2025.

    Uses REAL ts_500 test samples (Zenodo data) — one per bifurcation class.
    Runs actual trained model on each sample.
    No simulation needed — the Zenodo ts_500 IS the theoretical test data.

    Layout matching Ma 2025 Fig 2:
      Left: trajectory | variance | AC(1) | CNN-LSTM prob | this model prob
      Right: confusion matrix F (CNN-LSTM) and G (this model)
    """
    import torch
    fig_dir     = REPO_ROOT / cfg["paths"]["figures"]
    colors      = cfg["figures"]["colors"]
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_info     = cfg["datasets"][dataset_name]
    ts_len      = ds_info["ts_length"]
    class_names = ds_info["class_names"]
    met_dir     = REPO_ROOT / cfg["paths"]["metrics"]

    # ── Load confusion matrices from saved metrics ────────────────────────────
    cm_base, cm_model = None, None
    for v in [1, 2]:
        p = met_dir / f"cnn_lstm_{dataset_name}_v{v}_train_metrics.json"
        if p.exists():
            with open(p) as f:
                cm_base = np.array(json.load(f).get("confusion_matrix", []))
            break
    for v in cfg["training"][model_name]["pad_variants"]:
        p = met_dir / f"{model_name}_{dataset_name}_v{v}_train_metrics.json"
        if p.exists():
            with open(p) as f:
                cm_model = np.array(json.load(f).get("confusion_matrix", []))
            break

    # ── Load checkpoints ──────────────────────────────────────────────────────
    ckpt_dir    = REPO_ROOT / cfg["paths"]["checkpoints"]
    num_classes = ds_info["num_classes"]

    def load_first_ckpt(mname):
        for v in cfg["training"][mname]["pad_variants"]:
            p = ckpt_dir / f"{mname}_{dataset_name}_v{v}_best.ckpt"
            if p.exists():
                m = get_model(mname, ts_len=ts_len, num_classes=num_classes)
                m.load_state_dict(torch.load(p, map_location=device,
                                             weights_only=True))
                return m.to(device).eval()
        return None

    model_base  = load_first_ckpt("cnn_lstm")
    model_this  = load_first_ckpt(model_name)

    # ── Load one fold test sample from Zenodo ts_500 ──────────────────────────
    # class 0 = fold (matches paper Fig 2 which shows a fold bifurcation)
    fold_idx = class_names.index("fold") if "fold" in class_names else 0
    residuals = load_test_sample_per_class(dataset_name, cfg, fold_idx)

    if residuals is None:
        print(f"  Cannot load test sample — skipping Figure 2")
        return

    # ── Run rolling window + model inference ──────────────────────────────────
    ews_base = (run_rolling_on_sample(residuals, model_base, device, ts_len)
                if model_base else None)
    ews_this = (run_rolling_on_sample(residuals, model_this, device, ts_len)
                if model_this else None)

    if ews_this is None:
        print(f"  No checkpoint for {model_name} — skipping Figure 2")
        return

    pos       = ews_this["positions"]
    variance  = ews_this["variance"]
    ac1       = ews_this["lag1_ac"]
    probs_this= ews_this["probs"]   # (n_steps, n_classes)
    probs_base= ews_base["probs"] if ews_base else probs_this

    # p_transition = 1 - p_null
    null_idx     = class_names.index("null") if "null" in class_names else -1
    p_trans_this = 1.0 - probs_this[:, null_idx]
    p_trans_base = 1.0 - probs_base[:, null_idx]

    # Bifurcation index = where p_transition starts rising sharply
    bif_pos = pos[np.argmax(p_trans_this > 0.5)] if np.any(p_trans_this > 0.5)               else pos[-1]

    # ── Layout: 5 panels left + 2 confusion matrices right ───────────────────
    fig = plt.figure(figsize=(10, 8))
    gs  = gridspec.GridSpec(5, 2, figure=fig,
                             width_ratios=[2.5, 1.5],
                             hspace=0.06, wspace=0.35)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[1, 0], sharex=ax_A)
    ax_C = fig.add_subplot(gs[2, 0], sharex=ax_A)
    ax_D = fig.add_subplot(gs[3, 0], sharex=ax_A)
    ax_E = fig.add_subplot(gs[4, 0], sharex=ax_A)
    ax_F = fig.add_subplot(gs[1:3, 1])
    ax_G = fig.add_subplot(gs[3:5, 1])

    def shade(ax):
        ax.axvspan(bif_pos, pos[-1], alpha=0.10, color="gray", zorder=0)
    def vline(ax):
        ax.axvline(bif_pos, color="gray", lw=0.8, ls="--", alpha=0.7)

    # Panel A: trajectory + rolling window arrow
    win = len(residuals) // 2
    ax_A.plot(np.arange(len(residuals)), residuals,
              color=colors["trajectory"], lw=0.7)
    vline(ax_A); shade(ax_A)
    ax_A.set_ylabel("State", fontsize=8)
    ax_A.set_title(f"Fold bifurcation — {dataset_name} test sample", fontsize=9)
    ax_A.annotate("", xy=(win, residuals.max()*0.85),
                  xytext=(0, residuals.max()*0.85),
                  arrowprops=dict(arrowstyle="<->", color="k", lw=0.7))

    # Panel B: variance (1e-4 scale matching paper)
    ax_B.plot(pos, variance, color=colors["variance"], lw=1.0)
    vline(ax_B); shade(ax_B)
    ax_B.set_ylabel("Variance", fontsize=8)

    # Panel C: lag-1 AC
    ax_C.plot(pos, ac1, color=colors["lag1_ac"], lw=1.0)
    vline(ax_C); shade(ax_C)
    ax_C.set_ylabel("AC(1)", fontsize=8)

    # Panel D: CNN-LSTM probability
    ax_D.plot(pos, p_trans_base, color="#1f77b4", lw=1.2)
    vline(ax_D); shade(ax_D)
    ax_D.set_ylabel("DL", fontsize=8)
    ax_D.set_ylim(-0.05, 1.05)

    # Panel E: this model probability
    ax_E.plot(pos, p_trans_this, color=colors["dl_prob"], lw=1.2)
    vline(ax_E); shade(ax_E)
    ax_E.set_ylabel(model_name.upper(), fontsize=8)
    ax_E.set_ylim(-0.05, 1.05)
    ax_E.set_xlabel("Time", fontsize=8)

    for ax, lbl in zip([ax_A, ax_B, ax_C, ax_D, ax_E],
                        ["A", "B", "C", "D", "E"]):
        ax.text(-0.08, 1.0, lbl, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")
    for ax in [ax_A, ax_B, ax_C, ax_D]:
        plt.setp(ax.get_xticklabels(), visible=False)

    # Confusion matrices F and G
    def plot_cm(ax, cm, title, label):
        ax.text(-0.15, 1.0, label, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")
        ax.set_title(title, fontsize=8)
        if cm is None or len(cm) == 0:
            ax.text(0.5, 0.5, "Train model first",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            return
        cm_n = cm.astype(float)
        for i in range(len(cm)):
            s = cm[i].sum()
            if s > 0:
                cm_n[i] = cm[i] / s
        ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
        ticks = range(len(class_names))
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels([c[:4] for c in class_names], fontsize=6, rotation=45)
        ax.set_yticklabels([c[:4] for c in class_names], fontsize=6)
        ax.set_xlabel("Predicted label", fontsize=7)
        ax.set_ylabel("True label", fontsize=7)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, f"{cm_n[i,j]:.3f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if cm_n[i,j] > 0.5 else "black")

    plot_cm(ax_F, cm_base,  "DL classifier (CNN-LSTM)", "F")
    plot_cm(ax_G, cm_model, f"SDML classifier ({model_name})", "G")

    fname = cfg["naming"]["fig2"].format(model=model_name, dataset=dataset_name)
    savefig(fig, fig_dir / fname, cfg)


# =============================================================================
#  Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="cnn_lstm",
                        help="Model name or 'all'")
    parser.add_argument("--dataset",  default="ts_500",
                        choices=["ts_500", "ts_1500"])
    parser.add_argument("--fig2",     action="store_true")
    parser.add_argument("--fig3_only",action="store_true")
    parser.add_argument("--fig4",     action="store_true")
    parser.add_argument("--fig5",     action="store_true")
    parser.add_argument("--config",   default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_style(cfg)

    fig_dir = REPO_ROOT / cfg["paths"]["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    cores      = cfg["slurm"]["test_cores"]
    model_list = (
        [m["name"] for m in cfg["models"]]
        if args.model == "all" else [args.model]
    )
    gen_all = not any([args.fig2, args.fig3_only, args.fig4, args.fig5])

    # ── Figure 3 ──────────────────────────────────────────────────────────────
    if gen_all or args.fig3_only:
        print("\n--- Figure 3: PANGAEA overview ---")
        plot_figure3(cfg)
        if args.fig3_only:
            return

    for model_name in model_list:
        print(f"\n=== Model: {model_name} ===")

        # ── Figure 2 ──────────────────────────────────────────────────────────
        if gen_all or args.fig2:
            print(f"\n--- Figure 2: theoretical model ---")
            plot_figure2(model_name, args.dataset, cfg)

        for core_name in cores:
            # ── Figure 4 ──────────────────────────────────────────────────────
            if gen_all or args.fig4:
                print(f"\n--- Figure 4: {core_name} (all elements) ---")
                plot_figure4(model_name, core_name, cfg)

            # ── Figure 5 single model ─────────────────────────────────────────
            if gen_all or args.fig5:
                print(f"\n--- Figure 5 single: {core_name} ---")
                plot_figure5_single(model_name, core_name, cfg)

    # ── Figure 5 all-model comparison — one file per (core, element) ──────────
    if gen_all or args.fig5:
        for element in ELEMENTS:
            for core_name in cores:
                print(f"\n--- Figure 5 comparison: {core_name}/{element} ---")
                plot_figure5_comparison(core_name, element, cfg)

    print(f"\nAll figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
