"""
testing/fig_bury_final_probs.py

Reproduce the "final bifurcation probabilities" the way Bury et al. (2021, PNAS,
test_empirical/anoxia/compute_roc.py) and Ma et al. (2025, Comms Phys, SDML
07_compute_roc_ktau_dl.py) do it, from our already-saved PANGAEA result.json —
no retraining, no re-evaluation.

Bury recipe (per forced trajectory):
  1. ensemble-mean 4-class softmax over sliding windows        (already in result.json:
       p_fold / p_hopf / p_transcritical / p_null)
  2. keep only the LATE interval  = last 20% before the transition   (pred_interval_rel=[0.8,1.0])
  3. take 10 evenly-spaced predictions in that interval
  4. favoured bifurcation = argmax(fold, hopf, transcritical, null) per prediction
  5. pool every forced prediction -> count F / H / T / N   (Fig. 2 inset)
     bif_prob = fold+hopf+transcritical = 1 - null           (ROC transition indicator)

Ma recipe: same pooling, report P(transition)=bif_prob as ensemble mean +/- 95% CI.
(Our result.json stored only the ensemble mean per window, so the CI here is across
the pooled forced predictions, n reported — not across ensemble members. For the
across-members CI, re-run evaluate.py storing probs_*.std(0); that is a re-eval, not
a retrain.)

Usage:
  python testing/fig_bury_final_probs.py                       # trustworthy models, U+Mo (Bury's proxies)
  python testing/fig_bury_final_probs.py --models all --elements all
  python testing/fig_bury_final_probs.py --models cnn_lstm,inceptiontime --interval 0.6
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CLASSES = ["fold", "hopf", "transcritical", "null"]
CLASS_COLORS = ["#E07B1A", "#1f77b4", "#2D8A4E", "#7165D0"]
_DPI = 150

# Models that pass the synthetic gate (macro-AUC >= 0.80, every available length)
# and the PANGAEA gate (primary Mo+U mean AUC >= 0.75, every full 14-segment length) —
# recomputed directly from results/summary/{zenodo,pangaea_by_model}.csv.
TRUSTWORTHY = ["cnn_lstm", "lstm", "inceptiontime", "patchtst", "resnet", "tcn",
               "rnn_fcn", "arsenal", "tsf", "st", "grsf", "rocket",
               "minirocket", "mrsqm", "catch22"]
# Bury anoxia used only the redox-sensitive proxies
BURY_ELEMENTS = ["U", "Mo"]

# Bury et al. (2021) published favoured-bifurcation counts on the SAME Mediterranean
# anoxia data (test_empirical/anoxia/data/roc/df_bif_pred_counts_*.csv), for reference.
BURY_PUBLISHED = {"early": np.array([184, 33, 40, 3]),   # fold, hopf, branch, null
                  "late":  np.array([175, 44, 41, 0])}


def _late_points(arr, interval, n=10):
    """Last `1-interval` fraction of `arr`, resampled to `n` evenly-spaced points.

    Index-based: our sliding windows are evenly spaced in the resampled series, so
    this matches Bury's time-based [interval, 1.0] slice to within one window.
    """
    arr = np.asarray(arr, float)
    if len(arr) == 0:
        return np.empty(0)
    lo = int(np.floor(len(arr) * interval))
    tail = arr[lo:]
    if len(tail) <= n:
        return tail
    idx = np.round(np.linspace(0, len(tail) - 1, n)).astype(int)
    return tail[idx]


def collect(results_dir: Path, models, elements, interval):
    """Pool late-interval predictions across every matching forced segment."""
    fav_counts = np.zeros(4, dtype=int)
    bif_prob_forced, bif_prob_null = [], []
    n_segments = 0

    for d in sorted(results_dir.iterdir()):
        if not (d.is_dir() and d.name.endswith("_pangaea")):
            continue
        rf = d / "result.json"
        if not rf.exists():
            continue
        r = json.loads(rf.read_text())
        if models and r.get("model") not in models:
            continue
        if elements and r.get("element") not in elements:
            continue
        if not all(r.get(f"p_{c}") for c in CLASSES):
            continue

        # (n_windows, 4) ensemble-mean softmax, then late interval, 10 points
        P = np.column_stack([r[f"p_{c}"] for c in CLASSES])
        lo = int(np.floor(len(P) * interval))
        tail = P[lo:]
        if len(tail) > 10:
            idx = np.round(np.linspace(0, len(tail) - 1, 10)).astype(int)
            tail = tail[idx]
        if len(tail) == 0:
            continue
        n_segments += 1

        fav = tail.argmax(axis=1)
        for k in fav:
            fav_counts[k] += 1
        bif_prob_forced.extend(1.0 - tail[:, 3])          # = fold+hopf+transcritical

        # null side: p_transition_null laid out as surrogate blocks
        ptn = r.get("p_transition_null") or []
        counts = r.get("null_window_counts") or []
        off = 0
        for c in counts:
            seg = ptn[off:off + c]
            off += c
            bif_prob_null.extend(_late_points(seg, interval, 10))

    return fav_counts, np.array(bif_prob_forced), np.array(bif_prob_null), n_segments


def _mean_ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    m = x.mean()
    ci = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return m, ci


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="test_result")
    ap.add_argument("--models", default="trustworthy",
                    help="'trustworthy' (default), 'all', or comma list")
    ap.add_argument("--elements", default="UMo",
                    help="'UMo' (Bury's proxies, default), 'all', or comma list")
    ap.add_argument("--interval", type=float, default=0.8,
                    help="late-prediction interval start fraction (Bury=0.8)")
    ap.add_argument("--out", default="test_result/comparison/bury_final_probs")
    args = ap.parse_args()

    results_dir = REPO_ROOT / args.results
    models = (None if args.models == "all"
              else TRUSTWORTHY if args.models == "trustworthy"
              else args.models.split(","))
    elements = (None if args.elements == "all"
                else BURY_ELEMENTS if args.elements == "UMo"
                else args.elements.split(","))

    fav, bpf, bpn, nseg = collect(results_dir, models, elements, args.interval)
    total = fav.sum()
    freq = fav / total if total else fav * 0.0

    m_f, ci_f = _mean_ci(bpf)
    m_n, ci_n = _mean_ci(bpn)

    print(f"\nBury/Ma-style final probabilities  "
          f"(models={args.models}, elements={args.elements}, interval=[{args.interval},1.0])")
    print(f"  segments pooled : {nseg}")
    print(f"  forced preds    : {total}   (10 per segment, late interval)")
    print("  favoured bifurcation  count   freq")
    for c, k, fq in zip(["F (fold)", "H (hopf)", "T (transcritical)", "N (null)"], fav, freq):
        print(f"    {c:20s} {k:6d}  {fq:5.2f}")
    print(f"\n  P(transition) = fold+hopf+transcritical  (Ma-style, 95% CI over pooled preds)")
    print(f"    forced : {m_f:.3f} +/- {ci_f:.3f}   (n={len(bpf)})")
    print(f"    null   : {m_n:.3f} +/- {ci_n:.3f}   (n={len(bpn)})")

    bp = BURY_PUBLISHED["late"] if args.interval >= 0.8 else BURY_PUBLISHED["early"]
    bpn_ = bp / bp.sum()
    print(f"\n  Bury et al. (2021) published on the same anoxia data (F/H/T/N):")
    print(f"    counts {bp.tolist()}   freq {[round(x, 2) for x in bpn_.tolist()]}")

    # ── figure: Bury Fig. 2 inset (counts) + Ma P(transition) mean±CI ──────────
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    x = np.arange(4)
    ax1.bar(x - 0.2, freq, width=0.4, color=CLASS_COLORS, edgecolor="white",
            label="this work")
    bp = BURY_PUBLISHED["late"] if args.interval >= 0.8 else BURY_PUBLISHED["early"]
    ax1.bar(x + 0.2, bp / bp.sum(), width=0.4, color="none", edgecolor="black",
            hatch="///", label="Bury et al. 2021")
    for i, (fq, k) in enumerate(zip(freq, fav)):
        ax1.text(i - 0.2, fq, f"{fq:.0%}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["F", "H", "T", "N"])
    ax1.set_ylabel("favoured-bifurcation frequency")
    ax1.set_title("Bury Fig. 2 inset — favoured type")
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=7)

    ax2.bar(["forced", "AR(1) null"], [m_f, m_n], yerr=[ci_f, ci_n],
            color=["#1f77b4", "#AAAAAA"], edgecolor="white", capsize=6)
    ax2.axhline(0.5, color="#AAAAAA", ls="--", lw=1)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("P(transition) = fold+hopf+transcritical")
    ax2.set_title("Ma-style mean ± 95% CI")

    fig.suptitle(f"models={args.models}  elements={args.elements}  "
                 f"interval=[{args.interval},1.0]  ({nseg} segments)", fontsize=9)
    fig.tight_layout()
    png = out.with_suffix(".png")
    fig.savefig(png, dpi=_DPI)
    plt.close(fig)

    # Bury's df_bif_pred_counts CSV
    csv = out.with_suffix(".csv")
    csv.write_text("fold,hopf,branch,null\n" + ",".join(map(str, fav.tolist())) + "\n")
    print(f"\n  saved: {png}\n         {csv}")


if __name__ == "__main__":
    main()
