"""
testing/plot_figures.py

Inline plots (called from evaluate.py after each result.json is saved):
    plot_roc, plot_confusion_matrix, plot_pangaea_series

Summary CLI:
    python testing/plot_figures.py [--config config.yaml]
  Produces two things:
    1. test_result/comparison/<model>_<dataset>_<core>_<sapropel>_<element>/
       *_fig2.png  — Bury et al. (2021) PNAS Fig. 2 style: ROC + favoured-class inset
       *_stack.png — Ma et al. (2025, Comms Phys) Fig. 3 style, signal rows
       only: raw proxy and p(transition) vs. age (no variance / lag-1 AC /
       Kendall-tau rows). One of each per (model, dataset, core, sapropel,
       element), for every Mo/U PANGAEA segment.
    2. test_result/<model>_<dataset>_zenodo/confusion_matrix.png
       Row-normalised confusion matrix, one per (model, dataset), for the
       labeled synthetic (Zenodo) test set -- the only evaluation in this
       pipeline with a known true class, so the only one a confusion
       matrix is meaningful for.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
CLEAN = REPO_ROOT / "dataset/pangaea_923197/datasets/clean_dataset"

# Redox proxies only — the anoxia bifurcation is a redox transition (see
# src/rolling_window.py ELEMENTS). The PANGAEA stack panel is skipped for any
# other element even if stale result.json files for them still exist on disk.
REDOX_ELEMENTS = {"Mo", "U"}

# =============================================================================
# Style constants
# =============================================================================

# Per-model colour (internal comparison registry — distinct hues, not run
# through the colourblind-safety validator).
MODEL_COLORS = {
    "minirocket": "#1f77b4", "multirocket": "#aec7e8", "rocket": "#ff7f0e",
    "arsenal": "#ffbb78", "drcif": "#2ca02c", "rdst": "#d62728",
    "weasel2": "#ff9896", "cnn_lstm": "#9467bd", "lstm": "#c5b0d5",
    "inceptiontime": "#8c564b", "patchtst": "#5254a3", "resnet": "#bd9e39",
    "tcn": "#e7969c", "rnn_fcn": "#a1d99b", "tsf": "#dbdb8d", "st": "#17becf",
}

# Per-class colour (canonical Bury ordering)
CLASS_COLORS = {
    "fold":          "#E07B1A",
    "hopf":          "#1f77b4",
    "transcritical": "#2D8A4E",
    "null":          "#7165D0",
}

_CHANCE_COLOR = "#AAAAAA"
_DPI = 150

# Models passing both gates: synthetic macro-AUC >= 0.80 (every available
# length) AND real PANGAEA mean AUC >= 0.75 (every complete, full-length
# evaluation) — see chapters/chapter_8_verification.md for how this list
# was derived from results/summary/{zenodo,pangaea_by_model}.csv.
TRUSTWORTHY = {
    "cnn_lstm", "lstm", "inceptiontime", "patchtst", "resnet", "tcn", "rnn_fcn",
    "arsenal", "tsf", "st", "grsf", "rocket", "minirocket", "mrsqm", "catch22",
}


def _model_color(model: str) -> str:
    return MODEL_COLORS.get(model, "#888888")


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# =============================================================================
# Inline plots — called immediately after evaluate.py
# =============================================================================

def plot_roc(data: dict, out_dir: Path, filename: str = "roc_curve.png"):
    """ROC curve for one result (forced vs null)."""
    fpr = np.array(data.get("roc_fpr", [0, 1]))
    tpr = np.array(data.get("roc_tpr", [0, 1]))
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
    """Row-normalised confusion matrix heatmap for one zenodo experiment.

    Rows = true class, cols = predicted. Colour + big number are the
    row-normalised fraction (recall on the diagonal); raw count shown below.
    Row-normalising is what makes this readable under the class imbalance in
    the Bury synthetic set.
    """
    cm = np.array(data.get("confusion_matrix", []), dtype=float)
    if cm.size == 0:
        return
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j] * 100:.0f}%\n{int(cm[i, j])}",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    acc = data.get("accuracy", float("nan"))
    ax.set_title(f"Confusion Matrix — {data.get('model', '?')}  (acc={acc:.3f})")
    fig.colorbar(im, label="Row-normalised fraction")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=_DPI)
    plt.close(fig)


def plot_pangaea_series(data: dict, _unused, out_dir: Path,
                        filename: str = "pangaea_series.png"):
    """Quick inline plot for one PANGAEA segment: variance + AC + p_transition."""
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
# FIG_STACK — Ma et al. (2025, Comms Phys) style stacked-indicator panel
# =============================================================================

def plot_fig_stack_ma(record: dict, out_dir: Path):
    """
    One panel per (model, dataset, core, sapropel, element), in the style of
    Ma et al. (2025) Fig. 3, reduced to the two signal rows: raw proxy on
    top, this model's p(transition) below, sharing one x-axis vs. age. No
    rolling variance / lag-1 AC / Kendall-tau rows -- those are the
    classical indicators, not signals, and are deliberately left out here.
    The raw proxy comes straight from the same forced.csv the model was
    evaluated on; p(transition) is already in `record` (written by
    testing/evaluate.py), so this is pure plotting, no new computation.
    """
    core, sap, element = record.get("core"), record.get("sapropel"), record.get("element")
    model, dataset      = record.get("model", "?"), record.get("dataset", "?")
    if not all([core, sap, element]):
        return

    forced_csv = CLEAN / core / f"{core}_{sap}_forced.csv"
    if not forced_csv.exists() or element not in pd.read_csv(forced_csv, nrows=0).columns:
        return
    df_raw = pd.read_csv(forced_csv)

    p_trans = np.array(record.get("p_transition", []))
    ages    = np.array(record.get("ages_kyr_bp", []))
    if not len(p_trans):
        return

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.2), sharex=True)
    axes[0].plot(df_raw["age_kyr_bp"], df_raw[element], color="#1f4e8c", lw=1)
    axes[0].set_ylabel(f"{element} [ppm]")
    axes[0].set_title(f"{model} | {dataset} | {core}/{sap}/{element}", fontsize=9)

    axes[1].plot(ages, p_trans, color=_model_color(model))
    axes[1].axhline(0.5, color=_CHANCE_COLOR, lw=1, ls="--")
    axes[1].set_ylabel("p(transition)")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Age (kyr BP)")

    # Age runs oldest -> youngest left-to-right in the raw file; flip so the
    # transition (age 0 end of the segment) sits on the right, matching the
    # "time running out" reading direction used throughout this thesis's figures.
    for ax in axes:
        ax.invert_xaxis()

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{model}_{dataset}_{core}_{sap}_{element}_stack.png"
    fig.savefig(out_dir / fname, dpi=_DPI)
    plt.close(fig)


# =============================================================================
# FIG2 — Bury et al. (2021) PNAS Fig. 2 style panel
# =============================================================================

def _favored_class_freq(record: dict):
    """Fraction of forced windows in which each class is the argmax DL
    prediction. Returns length-4 array [fold, hopf, transcritical, null], or
    None if the per-class probability tracks are missing."""
    classes = ["fold", "hopf", "transcritical", "null"]
    if not all(record.get(f"p_{c}") for c in classes):
        return None
    probs = np.column_stack([np.asarray(record[f"p_{c}"], float) for c in classes])
    if probs.size == 0:
        return None
    favored = probs.argmax(axis=1)
    return np.bincount(favored, minlength=4) / len(favored)


def plot_fig2_bury(record: dict, out_dir: Path):
    """
    FIG2 — Bury et al. (2021) PNAS Fig. 2 style panel, one per
    (model, dataset, core, sapropel, element).

      main axes : ROC curve (forced vs AR(1) null), AUC in legend, chance diagonal
      inset     : frequency of the *favored* (argmax) DL class over the forced
                  windows — bars f(old) / h(opf) / t(ranscritical) / n(ull)
    """
    model   = record.get("model", "?")
    dataset = record.get("dataset", "?")
    core    = record.get("core", "?")
    sap     = record.get("sapropel", "?")
    element = record.get("element", "?")
    fpr     = np.array(record.get("roc_fpr", [0, 1]))
    tpr     = np.array(record.get("roc_tpr", [0, 1]))
    auc     = record.get("binary_auc", record.get("auc", float("nan")))

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], "--", color=_CHANCE_COLOR, lw=1, zorder=1)
    ax.plot(fpr, tpr, lw=2.2, color=_model_color(model), zorder=3, clip_on=False,
            solid_capstyle="round")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"FIG2 — {model} | {dataset} | {core}/{sap}/{element}", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # AUC as text (Bury Fig. 2 style — no legend box to collide with the inset)
    ax.text(0.05, 0.94, f"AUC = {auc:.3f}", ha="left", va="top",
            fontsize=11, transform=ax.transAxes,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    # ── inset: favored-class frequency (Bury Fig. 2 inset, in the empty
    #    lower-right triangle below the diagonal) ──────────────────────────────
    freq = _favored_class_freq(record)
    if freq is not None:
        classes = ["fold", "hopf", "transcritical", "null"]
        axin = ax.inset_axes([0.58, 0.12, 0.36, 0.34])
        bars = axin.bar(range(4), freq, color=[CLASS_COLORS[c] for c in classes],
                        edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, freq):
            axin.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}",
                      ha="center", va="bottom", fontsize=6)
        axin.set_xticks(range(4))
        axin.set_xticklabels(["f", "h", "t", "n"], fontsize=8)
        axin.set_ylim(0, 1.15)
        axin.set_yticks([0, 0.5, 1])
        axin.tick_params(axis="y", labelsize=7)
        axin.set_title("favored DL class", fontsize=7)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{model}_{dataset}_{core}_{sap}_{element}_fig2.png"
    fig.savefig(out_dir / fname, dpi=_DPI)
    plt.close(fig)


# The 7 sapropels this thesis actually scores (config.yaml `role: test`,
# Chapter 8, Section 8.2.1) — an exact match to Ma et al. (2025)'s own
# sapropel selection, not Bury et al. (2021)'s full 13.
TEST_SAPROPELS = [
    ("64PE406E1", "S3"), ("64PE406E1", "S4"), ("64PE406E1", "S5"), ("64PE406E1", "S6"),
    ("MS21", "S1"), ("MS66", "S1"), ("MS66", "S3"),
]


def _pool_sapropel_model_element(records: list, core: str, sap: str, model: str, element: str):
    """Pool every run (both sequence lengths, where available) of ONE model on
    ONE element for one sapropel into a single ROC curve + favoured-class
    frequency. Same ensembling logic as testing/collect_results.py's
    collect_pangaea_ensemble(), but scoped to a single (model, element)
    instead of the trustworthy set pooled across both elements.

    Returns (fpr, tpr, auc, freq[4]) or None if no data.
    """
    from metric.auc import compute_auc
    from metric.roc import compute_roc

    matches = [r for r in records if r.get("core") == core and r.get("sapropel") == sap
               and r.get("model") == model and r.get("element") == element]
    pf_list = [r["p_transition"] for r in matches if r.get("p_transition")]
    pn_list = [r["p_transition_null"] for r in matches if r.get("p_transition_null")]
    if not pf_list or not pn_list:
        return None
    n_f, n_n = min(len(x) for x in pf_list), min(len(x) for x in pn_list)
    ens_f = np.mean([np.asarray(x[:n_f]) for x in pf_list], axis=0)
    ens_n = np.mean([np.asarray(x[:n_n]) for x in pn_list], axis=0)
    y_true = np.concatenate([np.ones(n_f), np.zeros(n_n)])
    y_score = np.concatenate([ens_f, ens_n])

    fav_counts = np.zeros(4)
    for r in matches:
        if all(r.get(f"p_{c}") for c in ["fold", "hopf", "transcritical", "null"]):
            probs = np.column_stack([np.asarray(r[f"p_{c}"], float)
                                     for c in ["fold", "hopf", "transcritical", "null"]])
            fav_counts += np.bincount(probs.argmax(axis=1), minlength=4)

    auc = compute_auc(y_true, y_score)
    fpr, tpr, _ = compute_roc(y_true, y_score)
    freq = fav_counts / fav_counts.sum() if fav_counts.sum() else np.zeros(4)
    return fpr, tpr, auc, freq


def plot_fig2_grid(records: list, out_path: Path, model: str, element: str):
    """Multi-panel ROC grid for ONE model on ONE element: one lettered panel
    (a)-(g) per test sapropel (TEST_SAPROPELS), each showing the ROC curve
    (forced vs. AR(1) null, pooled across available sequence lengths) with a
    favoured-class frequency inset, exactly like a single-record
    plot_fig2_bury() panel.
    """
    classes = ["fold", "hopf", "transcritical", "null"]
    letters = "abcdefg"
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    axes = axes.ravel()

    for i, (core, sap) in enumerate(TEST_SAPROPELS):
        ax = axes[i]
        pooled = _pool_sapropel_model_element(records, core, sap, model, element)
        ax.plot([0, 1], [0, 1], "--", color=_CHANCE_COLOR, lw=1, zorder=1)
        if pooled is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        else:
            fpr, tpr, auc, freq = pooled
            ax.plot(fpr, tpr, lw=2.2, color=_model_color(model), zorder=3, solid_capstyle="round")
            ax.text(0.05, 0.94, f"AUC = {auc:.3f}", ha="left", va="top", fontsize=10,
                    transform=ax.transAxes, bbox=dict(boxstyle="round", fc="white", ec="0.7"))
            axin = ax.inset_axes([0.58, 0.10, 0.38, 0.34])
            bars = axin.bar(range(4), freq, color=[CLASS_COLORS[c] for c in classes],
                            edgecolor="white", linewidth=0.5)
            for b, v in zip(bars, freq):
                axin.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}",
                          ha="center", va="bottom", fontsize=6)
            axin.set_xticks(range(4))
            axin.set_xticklabels(["f", "h", "t", "n"], fontsize=7)
            axin.set_ylim(0, 1.15)
            axin.set_yticks([0, 0.5, 1])
            axin.tick_params(axis="y", labelsize=6)
        ax.set_title(f"({letters[i]}) {core} / {sap}", fontsize=11, loc="left")
        ax.set_xlabel("False Positive Rate", fontsize=8)
        ax.set_ylabel("True Positive Rate", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # 8th grid cell unused (7 sapropels, 2x4 grid) — hide it
    axes[7].axis("off")

    fig.suptitle(f"{model} — {element} — ROC and favoured bifurcation type, by sapropel", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=_DPI)
    plt.close(fig)
    return len(TEST_SAPROPELS)


# =============================================================================
# Loaders
# =============================================================================

def _load_pangaea_by_dataset(results_dir: Path) -> list:
    """All *_pangaea result.json, keyed on (model, dataset, core, sapropel,
    element) so both sequence lengths (ts_500 / ts_1500) survive."""
    records, seen = [], set()
    for d in sorted(results_dir.iterdir()):
        if not (d.is_dir() and d.name.endswith("_pangaea")):
            continue
        rfile = d / "result.json"
        if not rfile.exists():
            continue
        r = _load_json(rfile)
        if "binary_auc" not in r and "auc" in r:
            r["binary_auc"] = r["auc"]
        key = (r.get("model"), r.get("dataset"), r.get("core"),
               r.get("sapropel"), r.get("element"))
        if key not in seen:
            seen.add(key)
            records.append(r)
    return records


def _load_zenodo_results(results_dir: Path) -> list:
    """All *_zenodo result.json — the labeled synthetic test set, one per
    (model, dataset). This is the only evaluation in this pipeline with a
    known true class per example, so it's the only one a confusion matrix
    is meaningful for (PANGAEA's true bifurcation type is unknown -- that's
    the whole point of Chapter 8's cross-model consensus analysis)."""
    records = []
    for d in sorted(results_dir.iterdir()):
        if not (d.is_dir() and d.name.endswith("_zenodo")):
            continue
        rfile = d / "result.json"
        if rfile.exists() and "confusion_matrix" in (r := _load_json(rfile)):
            records.append(r)
    return records


def plot_zenodo_acc_vs_auc(summary_csv: Path, out_path: Path):
    """4-class macro-AUC vs. 4-class accuracy, one point per model, on the
    labeled synthetic (Zenodo) test set — the one evaluation in this pipeline
    with a true label, so both metrics are genuinely computable (unlike
    PANGAEA, which has no known bifurcation type — see Chapter 8).

    Uses the longer available sequence length per model (ts_1500 where it
    exists, else ts_500), since that is the better-trained checkpoint.
    """
    df = pd.read_csv(summary_csv)
    best = (df.sort_values("dataset", ascending=False)  # ts_500 < ts_1500 lexically
              .drop_duplicates("model", keep="first"))

    fig, ax = plt.subplots(figsize=(7, 6))
    for _, row in best.iterrows():
        m = row["model"]
        color = "#2D8A4E" if m in TRUSTWORTHY else _CHANCE_COLOR
        ax.scatter(row["accuracy"], row["macro_auc_ovr"], color=color, s=35, zorder=3)
        ax.annotate(m, (row["accuracy"], row["macro_auc_ovr"]), fontsize=6.5,
                    xytext=(3, 2), textcoords="offset points")

    ax.axhline(0.5, color=_CHANCE_COLOR, ls=":", lw=0.8)
    ax.axvline(0.25, color=_CHANCE_COLOR, ls=":", lw=0.8, label="4-class chance accuracy (0.25)")
    ax.set_xlabel("4-class accuracy (argmax, single threshold)")
    ax.set_ylabel("4-class macro-AUC (one-vs-rest, threshold-independent)")
    ax.set_title("Synthetic (Zenodo) results: AUC vs. accuracy per model\n"
                 "(best available sequence length; green = trustworthy roster)")
    ax.set_xlim(0.2, 0.9)
    ax.set_ylim(0.45, 1.0)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=_DPI)
    plt.close(fig)
    return len(best)


# =============================================================================
# Summary CLI
# =============================================================================

def main():
    import argparse
    import shutil
    import yaml

    parser = argparse.ArgumentParser(
        description="Generate the FIG2 and stacked indicator panels for every "
                    "PANGAEA (model, dataset, core, sapropel, element), and the "
                    "confusion matrix for every labeled Zenodo (model, dataset).")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(REPO_ROOT / args.config))
    results_dir = REPO_ROOT / cfg["paths"]["test_results"]
    comp_dir    = results_dir / "comparison"

    # This directory holds nothing but output from this script, so it's safe
    # to wipe wholesale on every run -- replaces stale results instead of
    # accumulating them (e.g. filenames from a figure type that's since been
    # dropped, like the old *_roc3.png, would otherwise sit there forever).
    if comp_dir.exists():
        shutil.rmtree(comp_dir)

    print(f"Loading results from: {results_dir}")

    records = _load_pangaea_by_dataset(results_dir)
    print(f"  Pangaea records (dataset-aware): {len(records)}")
    done = skipped = 0
    for r in records:
        model, ds = r.get("model", ""), r.get("dataset", "")
        core, sap, element = r.get("core", ""), r.get("sapropel", ""), r.get("element", "")
        if not all([model, ds, core, sap, element]) or element not in REDOX_ELEMENTS:
            skipped += 1
            continue
        exp_dir = comp_dir / f"{model}_{ds}_{core}_{sap}_{element}"
        plot_fig2_bury(r, exp_dir)
        plot_fig_stack_ma(r, exp_dir)
        done += 1
    print(f"  fig2 + stack — {done} panel sets written to {comp_dir}  ({skipped} skipped, non-Mo/U)")

    zrecords = _load_zenodo_results(results_dir)
    print(f"  Zenodo records (labeled, confusion matrix available): {len(zrecords)}")
    zdone = 0
    for r in zrecords:
        model, ds = r.get("model", "?"), r.get("dataset", "?")
        class_names = r.get("class_names") or ["fold", "hopf", "transcritical", "null"]
        plot_confusion_matrix(r, results_dir / f"{model}_{ds}_zenodo", class_names)
        zdone += 1
    print(f"  confusion_matrix — {zdone} panels written to {results_dir}/<model>_<dataset>_zenodo/")

    summary_csv = REPO_ROOT / "results/summary/zenodo.csv"
    if summary_csv.exists():
        fig_out = REPO_ROOT / "figures/fig8_zenodo_acc_vs_auc"
        n = plot_zenodo_acc_vs_auc(summary_csv, fig_out)
        print(f"  zenodo_acc_vs_auc — {n} models plotted to {fig_out}.pdf/.png")
    else:
        print(f"  Skipping zenodo_acc_vs_auc — {summary_csv} not found")

    grid_records = _load_pangaea_by_dataset(results_dir)
    grid_records = [r for r in grid_records if r.get("element") in ("Mo", "U")]
    grid_dir = comp_dir / "pangaea_grids"
    grid_pairs = sorted({(r.get("model"), r.get("element")) for r in grid_records
                         if r.get("model")})
    grid_done = 0
    for model, element in grid_pairs:
        out = grid_dir / f"{model}_{element}_grid"
        plot_fig2_grid(grid_records, out, model, element)
        grid_done += 1
    print(f"  fig2_grid — {grid_done} (model, element) grids written to {grid_dir}/")


if __name__ == "__main__":
    main()
