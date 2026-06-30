"""
testing/plot_figures.py

Two kinds of plots:
  1. Inline — called immediately after evaluate.py saves a result.json.
  2. Summary — called once after all models are evaluated; one loop drives
     every (model, core, sapropel, element) combination to guarantee every
     combination gets the identical set of figures.

Inline plots:   plot_roc, plot_confusion_matrix, plot_pangaea_series
Summary CLI:    python testing/plot_figures.py [--config config.yaml]
  Produces FIG1–FIG5 in results/comparison/

Filename convention: {model}_{core}_{sapropel}_{element}_{figname}.png
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# =============================================================================
# Style constants — fixed across every figure so all plots are comparable
# =============================================================================

# Canonical model order: TSC first (fast → slow), then DL
MODEL_ORDER = [
    "minirocket", "multirocket", "rocket", "arsenal", "drcif",
    "hydra_multirocket", "rdst", "weasel2",
    "cnn_lstm", "lstm", "inceptiontime",
]

# Fixed per-model color palette
MODEL_COLORS = {
    "minirocket":        "#1f77b4",
    "multirocket":       "#aec7e8",
    "rocket":            "#ff7f0e",
    "arsenal":           "#ffbb78",
    "drcif":             "#2ca02c",
    "hydra_multirocket": "#98df8a",
    "rdst":              "#d62728",
    "weasel2":           "#ff9896",
    "cnn_lstm":          "#9467bd",
    "lstm":              "#c5b0d5",
    "inceptiontime":     "#8c564b",
}

# Fixed per-class color palette (canonical Bury ordering)
CLASS_COLORS = {
    "fold":          "#E07B1A",
    "hopf":          "#1f77b4",
    "transcritical": "#2D8A4E",
    "null":          "#7165D0",
}

# Fixed element order
ELEMENT_ORDER = ["Al", "Ba", "Mo", "Ti", "U"]

_CHANCE_COLOR = "#AAAAAA"
_DPI = 150


def _model_color(model: str) -> str:
    return MODEL_COLORS.get(model, "#888888")


def _sort_models(models):
    """Return models in canonical MODEL_ORDER; unknown models appended at end."""
    known   = [m for m in MODEL_ORDER if m in models]
    unknown = sorted(m for m in models if m not in MODEL_ORDER)
    return known + unknown


def _sort_elements(elements):
    known   = [e for e in ELEMENT_ORDER if e in elements]
    unknown = sorted(e for e in elements if e not in ELEMENT_ORDER)
    return known + unknown


# =============================================================================
# Data loading helpers
# =============================================================================

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_zenodo_results(results_dir: Path) -> list:
    """
    Load all Zenodo (synthetic test set) result.json files.

    Handles both new directories (*_zenodo) and old-format (*_auc).
    New format has 'binary_auc'; old format has 'auc'. Normalised to 'binary_auc'.
    """
    records = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        rfile = d / "result.json"
        if not rfile.exists():
            continue
        if d.name.endswith("_zenodo") or d.name.endswith("_auc"):
            r = _load_json(rfile)
            # Normalise field name
            if "auc" in r and "binary_auc" not in r:
                r["binary_auc"] = r["auc"]
            if "core" not in r:    # skip pangaea entries
                records.append(r)
    return records


def load_pangaea_results(results_dir: Path) -> list:
    """
    Load all PANGAEA (empirical) result.json files.

    Handles new-format (*_pangaea) and old-format (*_auc/*_kendall_tau) dirs.
    Merges AUC + tau from old dirs if present. Normalises to 'binary_auc'/'kendall_tau'.
    """
    # New format: one file per (model, core, sap, element) with all metrics
    records = []
    seen    = set()

    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        rfile = d / "result.json"
        if not rfile.exists():
            continue

        if d.name.endswith("_pangaea"):
            r = _load_json(rfile)
            if "binary_auc" not in r and "auc" in r:
                r["binary_auc"] = r["auc"]
            key = (r.get("model"), r.get("core"), r.get("sapropel"), r.get("element"))
            seen.add(key)
            records.append(r)

    # Old format: separate _auc and _kendall_tau directories — merge them
    auc_data  = {}
    tau_data  = {}
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        rfile = d / "result.json"
        if not rfile.exists():
            continue
        r = _load_json(rfile)
        if "core" not in r:
            continue
        key = (r.get("model"), r.get("core"), r.get("sapropel"), r.get("element"))
        if key in seen:   # already loaded from new format
            continue
        if d.name.endswith("_auc"):
            auc_data[key] = r
        elif d.name.endswith("_kendall_tau"):
            tau_data[key] = r

    all_keys = set(auc_data) | set(tau_data)
    for key in all_keys:
        if key in seen:
            continue
        merged = dict(auc_data.get(key, tau_data.get(key, {})))
        if key in auc_data and key in tau_data:
            merged.update({k: v for k, v in tau_data[key].items()
                           if k not in merged})
        if "auc" in merged and "binary_auc" not in merged:
            merged["binary_auc"] = merged["auc"]
        records.append(merged)
        seen.add(key)

    return records


# =============================================================================
# Inline plots — called immediately after evaluate.py
# =============================================================================

def plot_roc(data: dict, out_dir: Path, filename: str = "roc_curve.png"):
    """FIG2-style ROC curve for one result (forced vs null)."""
    fpr = np.array(data.get("roc_fpr", [0, 1]))
    tpr = np.array(data.get("roc_tpr", [0, 1]))
    # Accept both old ('auc') and new ('binary_auc') field names
    auc = data.get("binary_auc", data.get("auc", float("nan")))
    model = data.get("model", "?")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, color=_model_color(model),
            label=f"{model}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=_CHANCE_COLOR, lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {model}")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=_DPI)
    plt.close(fig)


def plot_confusion_matrix(data: dict, out_dir: Path, class_names: list,
                          filename: str = "confusion_matrix.png"):
    """Confusion matrix heatmap for one zenodo experiment."""
    cm = np.array(data.get("confusion_matrix", []))
    if cm.size == 0:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > threshold else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {data.get('model', '?')}")
    fig.colorbar(im)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=_DPI)
    plt.close(fig)


def plot_pangaea_series(data: dict, _unused, out_dir: Path,
                        filename: str = "pangaea_series.png"):
    """
    Quick inline plot for one PANGAEA segment: variance + AC + p_transition.
    Called immediately from evaluate_pangaea after each result is saved.
    For the full 4-panel FIG1, use plot_fig1_pangaea (summary CLI).
    """
    p_trans  = np.array(data.get("p_transition", []))
    variance = np.array(data.get("variance", []))
    lag1_ac  = np.array(data.get("lag1_ac", []))
    ages     = np.array(data.get("ages_kyr_bp", np.arange(len(p_trans))))

    n_panels = sum([len(variance) > 0, len(lag1_ac) > 0, len(p_trans) > 0])
    if n_panels == 0:
        return
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 2.5 * n_panels),
                              sharex=True)
    if n_panels == 1:
        axes = [axes]

    idx = 0
    if len(variance):
        axes[idx].plot(ages, variance, color="#E07B1A")
        axes[idx].set_ylabel("Variance")
        idx += 1
    if len(lag1_ac):
        axes[idx].plot(ages, lag1_ac, color="#2D8A4E")
        axes[idx].set_ylabel("Lag-1 AC")
        idx += 1
    if len(p_trans):
        axes[idx].plot(ages, p_trans, color=_model_color(data.get("model", "")))
        axes[idx].axhline(0.5, color=_CHANCE_COLOR, lw=1, ls="--")
        axes[idx].set_ylabel("p(transition)")
        axes[idx].set_ylim(0, 1)

    axes[-1].set_xlabel("Age (kyr BP)")
    core = data.get("core", "?")
    sap  = data.get("sapropel", "?")
    elem = data.get("element", "?")
    axes[0].set_title(f"{data.get('model','?')} — {core}/{sap}/{elem}")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=_DPI)
    plt.close(fig)


# =============================================================================
# Summary figures — produced by the CLI after all evaluations complete
# =============================================================================

def plot_fig1_pangaea(record: dict, pangaea_clean_dir: Path, out_dir: Path):
    """
    FIG1: 4-panel stacked for one (model, core, sapropel, element).

    Panels:
      (a) raw proxy + Gaussian trend  — read from CSV
      (b) residuals                   — read from CSV
      (c) rolling variance + lag-1 AC — from result.json
      (d) transition probability      — from result.json

    x-axis: Age (kyr BP). Time flows left → right, transition at RIGHT edge.
    This matches Bury (2021) Fig. 3 style.
    """
    model    = record.get("model", "?")
    core     = record.get("core",  "?")
    sap      = record.get("sapropel", "?")
    element  = record.get("element", "?")
    ages     = np.array(record.get("ages_kyr_bp", []))
    p_trans  = np.array(record.get("p_transition", []))
    variance = np.array(record.get("variance", []))
    lag1_ac  = np.array(record.get("lag1_ac", []))

    # Try to load raw proxy + Gaussian trend from CSV
    csv_path = pangaea_clean_dir / core / f"{core}_{sap}_forced.csv"
    raw_proxy = raw_trend = raw_resid = raw_ages = None
    try:
        import pandas as pd
        df  = pd.read_csv(csv_path).sort_values("age_kyr_bp")
        raw_ages  = df["age_kyr_bp"].values
        raw_proxy = df.get(element, df.get(f"{element}", None))
        raw_trend = df.get(f"{element}_trend", None)
        raw_resid = df.get(f"{element}_residuals", None)
        if raw_proxy is not None:
            raw_proxy = raw_proxy.values
        if raw_trend is not None:
            raw_trend = raw_trend.values
        if raw_resid is not None:
            raw_resid = raw_resid.values
    except Exception:
        pass

    has_raw = (raw_proxy is not None and raw_ages is not None)
    n_panels = 2 + has_raw * 2     # always 2 (c,d); + 2 if CSV available (a,b)

    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 2.8 * n_panels),
                              sharex=True)
    if n_panels == 1:
        axes = [axes]

    panel = 0

    if has_raw:
        # (a) raw proxy + Gaussian trend
        ax = axes[panel]
        ax.plot(raw_ages, raw_proxy, color="black", lw=0.8, label=element)
        if raw_trend is not None:
            ax.plot(raw_ages, raw_trend, color="#E07B1A", lw=1.5, label="Gaussian trend")
        ax.set_ylabel(f"{element} (raw)")
        ax.legend(fontsize=7, loc="upper left")
        ax.text(0.02, 0.92, "(a)", transform=ax.transAxes, fontsize=9, va="top")
        panel += 1

        # (b) residuals
        ax = axes[panel]
        if raw_resid is not None:
            ax.plot(raw_ages, raw_resid, color="black", lw=0.8)
        ax.axhline(0, color=_CHANCE_COLOR, lw=0.8, ls="--")
        ax.set_ylabel("Residuals")
        ax.text(0.02, 0.92, "(b)", transform=ax.transAxes, fontsize=9, va="top")
        panel += 1

    # (c) rolling variance + lag-1 AC (dual y-axis for different scales)
    ax = axes[panel]
    ax.text(0.02, 0.92, "(c)", transform=ax.transAxes, fontsize=9, va="top")
    if len(variance):
        ax.plot(ages, variance, color="#E07B1A", lw=1.5, label="Variance")
        ax.set_ylabel("Variance", color="#E07B1A")
        ax.tick_params(axis="y", labelcolor="#E07B1A")
    if len(lag1_ac):
        ax2 = ax.twinx()
        ax2.plot(ages, lag1_ac, color="#2D8A4E", lw=1.5, label="Lag-1 AC")
        ax2.set_ylabel("Lag-1 AC", color="#2D8A4E")
        ax2.tick_params(axis="y", labelcolor="#2D8A4E")
        ax2.set_ylim(-1, 1)
    lines = ([mpatches.Patch(color="#E07B1A", label="Variance")] if len(variance) else [])
    lines += ([mpatches.Patch(color="#2D8A4E", label="Lag-1 AC")] if len(lag1_ac) else [])
    if lines:
        ax.legend(handles=lines, fontsize=7, loc="upper left")
    panel += 1

    # (d) transition probability
    ax = axes[panel]
    ax.plot(ages, p_trans, color=_model_color(model), lw=1.5,
            label="p(transition)")
    ax.axhline(0.5, color=_CHANCE_COLOR, lw=1, ls="--", label="0.5 threshold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("p(transition)")
    ax.legend(fontsize=7, loc="upper left")
    ax.text(0.02, 0.92, "(d)", transform=ax.transAxes, fontsize=9, va="top")

    # Shared x-axis — time flows left → right (oldest at left = most negative age)
    # The x-axis is NOT inverted because kyr BP values are already ordered so
    # that smaller (more negative) = older = left, approaching 0 = younger = right.
    axes[-1].set_xlabel("Age (kyr BP)")

    tau = record.get("kendall_tau", float("nan"))
    auc = record.get("binary_auc", record.get("auc", float("nan")))
    fig.suptitle(
        f"FIG1 — {model} | {core}/{sap}/{element}  "
        f"[AUC={auc:.3f}  τ={tau:.3f}]",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{model}_{core}_{sap}_{element}_fig1.png"
    fig.savefig(out_dir / fname, dpi=_DPI)
    plt.close(fig)


def plot_fig2_roc(record: dict, out_dir: Path):
    """
    FIG2: ROC curve (forced vs AR(1) null) for one (model, core, sapropel, element).
    AUC shown in legend.
    """
    model   = record.get("model", "?")
    core    = record.get("core", "?")
    sap     = record.get("sapropel", "?")
    element = record.get("element", "?")
    fpr     = np.array(record.get("roc_fpr", [0, 1]))
    tpr     = np.array(record.get("roc_tpr", [0, 1]))
    auc     = record.get("binary_auc", record.get("auc", float("nan")))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, color=_model_color(model),
            label=f"{model}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=_CHANCE_COLOR, lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"FIG2 — ROC: {model} | {core}/{sap}/{element}")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{model}_{core}_{sap}_{element}_fig2_roc.png"
    fig.savefig(out_dir / fname, dpi=_DPI)
    plt.close(fig)


def plot_fig3_auc_heatmap(pangaea_records: list, out_dir: Path):
    """
    FIG3: AUC heatmap — rows=models, cols=elements.

    Each cell is the mean AUC over all (core, sapropel) combinations for that
    (model, element) pair. Missing combinations show as grey.
    """
    # Collect AUC values indexed by (model, element)
    auc_vals = defaultdict(list)
    for r in pangaea_records:
        model   = r.get("model")
        element = r.get("element")
        auc     = r.get("binary_auc", r.get("auc", float("nan")))
        if model and element and not np.isnan(auc):
            auc_vals[(model, element)].append(auc)

    models   = _sort_models(set(m for m, _ in auc_vals))
    elements = _sort_elements(set(e for _, e in auc_vals))

    if not models or not elements:
        return

    matrix = np.full((len(models), len(elements)), np.nan)
    for i, m in enumerate(models):
        for j, e in enumerate(elements):
            vals = auc_vals.get((m, e), [])
            if vals:
                matrix[i, j] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(max(4, len(elements) * 1.5),
                                    max(4, len(models) * 0.5)))
    im = ax.imshow(matrix, vmin=0.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(elements)))
    ax.set_xticklabels(elements)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    for i in range(len(models)):
        for j in range(len(elements)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    fig.colorbar(im, label="Mean AUC (over cores/sapropels)")
    ax.set_xlabel("Element")
    ax.set_ylabel("Model")
    ax.set_title("FIG3 — AUC Heatmap: models × elements")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig3_auc_heatmap.png", dpi=_DPI)
    plt.close(fig)


def plot_fig4_kendall_tau(pangaea_records: list, out_dir: Path):
    """
    FIG4: Kendall tau per model — mean ± CI across (core, sapropel, element).

    Positive tau = rising p_transition toward transition (good EWS signal).
    Bar color indicates TSC (orange) or DL (blue). Null tau mean shown as dashed line.
    """
    tau_by_model      = defaultdict(list)
    null_tau_by_model = defaultdict(list)

    for r in pangaea_records:
        model = r.get("model")
        tau   = r.get("kendall_tau", float("nan"))
        null_tau_mean = r.get("tau_null_mean", float("nan"))
        if model and not np.isnan(tau):
            tau_by_model[model].append(tau)
        if model and not np.isnan(null_tau_mean):
            null_tau_by_model[model].append(null_tau_mean)

    models = _sort_models(set(tau_by_model))
    if not models:
        return

    means   = [float(np.mean(tau_by_model[m]))  for m in models]
    stds    = [float(np.std(tau_by_model[m]))   for m in models]
    n_obs   = [len(tau_by_model[m])              for m in models]

    # 95 % CI using t-distribution
    from scipy.stats import t as t_dist
    ci_margins = []
    for std, n in zip(stds, n_obs):
        if n > 1:
            ci_margins.append(t_dist.ppf(0.975, df=n - 1) * std / np.sqrt(n))
        else:
            ci_margins.append(0.0)

    dl_set = {"cnn_lstm", "lstm", "inceptiontime"}
    colors = ["#1f77b4" if m in dl_set else "#E07B1A" for m in models]

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 0.8), 5))
    x = np.arange(len(models))
    ax.bar(x, means, yerr=ci_margins, color=colors, capsize=5,
           edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axhline(0, color=_CHANCE_COLOR, lw=1, ls="--", label="τ = 0 (no trend)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Kendall τ (mean ± 95% CI)")
    ax.set_ylim(-1, 1)
    ax.set_title("FIG4 — Kendall τ per model (mean over cores/sapropels/elements)")
    ax.legend(handles=[
        mpatches.Patch(color="#1f77b4", label="DL models"),
        mpatches.Patch(color="#E07B1A", label="TSC models"),
    ], fontsize=8)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig4_kendall_tau.png", dpi=_DPI)
    plt.close(fig)


def plot_fig5_roc_overlay(pangaea_records: list, core: str, element: str,
                          out_dir: Path):
    """
    FIG5: ROC overlay — all models on one axes for one (core, element).

    Averages over sapropels for each model. One file per core × element.
    """
    # Group by (model, sapropel) → list of (fpr, tpr, auc)
    model_curves = defaultdict(list)
    for r in pangaea_records:
        if r.get("core") != core or r.get("element") != element:
            continue
        model = r.get("model")
        fpr   = np.array(r.get("roc_fpr", []))
        tpr   = np.array(r.get("roc_tpr", []))
        auc   = r.get("binary_auc", r.get("auc", float("nan")))
        if len(fpr) >= 2 and not np.isnan(auc):
            model_curves[model].append((fpr, tpr, auc))

    if not model_curves:
        return

    models = _sort_models(set(model_curves))
    fig, ax = plt.subplots(figsize=(7, 6))

    for model in models:
        curves = model_curves[model]
        mean_auc = float(np.mean([auc for _, _, auc in curves]))
        # Interpolate all curves to a common FPR grid, then average TPR
        fpr_grid = np.linspace(0, 1, 200)
        tpr_grid = np.zeros_like(fpr_grid)
        for fpr, tpr, _ in curves:
            tpr_grid += np.interp(fpr_grid, fpr, tpr)
        tpr_grid /= len(curves)

        ax.plot(fpr_grid, tpr_grid, lw=1.5, color=_model_color(model),
                label=f"{model} (AUC={mean_auc:.3f})")

    ax.plot([0, 1], [0, 1], "--", color=_CHANCE_COLOR, lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"FIG5 — ROC overlay: {core} / {element}")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"fig5_roc_overlay_{core}_{element}.png"
    fig.savefig(out_dir / fname, dpi=_DPI)
    plt.close(fig)


# =============================================================================
# Summary CLI
# =============================================================================

def main():
    import argparse
    from src.dataset_loader import load_config

    parser = argparse.ArgumentParser(
        description="Generate all summary figures from saved result.json files.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg          = load_config(args.config)
    results_dir  = REPO_ROOT / cfg["paths"]["results"]
    comp_dir     = results_dir / "comparison"
    pangaea_clean = REPO_ROOT / cfg["paths"]["pangaea_clean"]

    print(f"Loading results from: {results_dir}")
    zenodo_records  = load_zenodo_results(results_dir)
    pangaea_records = load_pangaea_results(results_dir)

    print(f"  Zenodo records:  {len(zenodo_records)}")
    print(f"  Pangaea records: {len(pangaea_records)}")

    cores    = sorted(set(r["core"]    for r in pangaea_records if "core"    in r))
    elements = sorted(set(r["element"] for r in pangaea_records if "element" in r))
    models   = sorted(set(r["model"]   for r in pangaea_records if "model"   in r))
    sapropels_by_core = defaultdict(set)
    for r in pangaea_records:
        if "core" in r and "sapropel" in r:
            sapropels_by_core[r["core"]].add(r["sapropel"])

    # ── FIG1 + FIG2: one per (model, core, sapropel, element) ─────────────────
    print("\nFIG1 + FIG2 per (model, core, sapropel, element)...")
    done_fig12 = 0
    for r in pangaea_records:
        model   = r.get("model", "")
        core    = r.get("core", "")
        sap     = r.get("sapropel", "")
        element = r.get("element", "")
        if not all([model, core, sap, element]):
            continue
        exp_dir = comp_dir / f"{model}_{core}_{sap}_{element}"
        plot_fig1_pangaea(r, pangaea_clean, exp_dir)
        plot_fig2_roc(r, exp_dir)
        done_fig12 += 1
    print(f"  Done: {done_fig12} figure pairs")

    # ── FIG3: AUC heatmap (models × elements) ─────────────────────────────────
    print("\nFIG3 — AUC heatmap...")
    plot_fig3_auc_heatmap(pangaea_records, comp_dir)
    print(f"  Saved: {comp_dir / 'fig3_auc_heatmap.png'}")

    # ── FIG4: Kendall tau per model ────────────────────────────────────────────
    print("\nFIG4 — Kendall tau per model...")
    plot_fig4_kendall_tau(pangaea_records, comp_dir)
    print(f"  Saved: {comp_dir / 'fig4_kendall_tau.png'}")

    # ── FIG5: Per-core ROC overlay (all models, one file per core × element) ──
    print("\nFIG5 — Per-core ROC overlay...")
    done_fig5 = 0
    for core in cores:
        for element in _sort_elements(elements):
            plot_fig5_roc_overlay(pangaea_records, core, element, comp_dir)
            done_fig5 += 1
    print(f"  Done: {done_fig5} overlay plots")

    print(f"\nAll comparison figures saved to: {comp_dir}")


if __name__ == "__main__":
    main()
