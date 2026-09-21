"""
testing/compare_bury.py

Side-by-side comparison of our PANGAEA anoxia result with Bury et al. (2021, PNAS),
computed on the SAME cores (MS21, MS66, 64PE406E1), SAME proxies (Mo, U), SAME
late-20% prediction interval. Pure post-processing of our result.json — no models run.

Bury's numbers are read from their published repo
(github.com/ThomasMBury/deep-early-warnings-pnas, test_empirical/anoxia/data/roc/).
"""

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["fold", "hopf", "transcritical", "null"]
TRUSTWORTHY = ["cnn_lstm", "lstm", "inceptiontime", "patchtst", "resnet", "tcn",
               "rnn_fcn", "arsenal", "tsf", "st", "grsf", "rocket"]
DL_ONLY = TRUSTWORTHY[:7]

# Classical EWS baselines on our data (from testing/classical_ews_auc.py; model-free,
# recompute there if the residual CSVs change).
OUR_VAR_AUC = 0.895
OUR_AC_AUC = 0.516

# ── Bury et al. (2021) published anoxia values, late interval [0.8, 1.0] ──────
BURY = {
    "auc_dl": 0.991, "auc_var": 0.899, "auc_ac": 0.594,
    "fav_counts": np.array([175, 44, 41, 0]),          # fold, hopf, transcritical, null
    "cores": "MS21, MS66, 64PE406E1", "n_transitions": 13, "seq_len": "500",
    "classifier": "1 CNN-LSTM arch, ~20-net ensemble, normal-form training",
}


def _late(a, frac=0.8, n=10):
    a = np.asarray(a, float)
    lo = int(len(a) * frac)
    t = a[lo:]
    if len(t) > n:
        t = t[np.round(np.linspace(0, len(t) - 1, n)).astype(int)]
    return t


def gather(models):
    fav = np.zeros(4)
    yf, yn = [], []
    segs = set()
    for f in glob.glob(str(REPO_ROOT / "test_result" / "*_pangaea" / "result.json")):
        d = json.loads(Path(f).read_text())
        if d.get("model") not in models or d.get("element") not in ("Mo", "U"):
            continue
        if not all(d.get(f"p_{c}") for c in CLASSES):
            continue
        P = np.column_stack([d[f"p_{c}"] for c in CLASSES])
        tail = P[int(len(P) * 0.8):]
        if len(tail) > 10:
            tail = tail[np.round(np.linspace(0, len(tail) - 1, 10)).astype(int)]
        for k in tail.argmax(1):
            fav[k] += 1
        yf.extend(1 - tail[:, 3])
        segs.add((d["core"], d["sapropel"]))
        ptn = d.get("p_transition_null") or []
        off = 0
        for c in (d.get("null_window_counts") or []):
            yn.extend(_late(ptn[off:off + c]))
            off += c
    y = np.r_[np.ones(len(yf)), np.zeros(len(yn))]
    auc = roc_auc_score(y, np.r_[yf, yn])
    return fav, auc, np.mean(yf), np.mean(yn), len(segs), len(yf), len(yn)


def main():
    fav, auc, mf, mn, nseg, nf, nn = gather(TRUSTWORTHY)
    favd, aucd, *_ = gather(DL_ONLY)
    our_freq = fav / fav.sum()
    bury_freq = BURY["fav_counts"] / BURY["fav_counts"].sum()

    print("\n============ Our anoxia result  vs  Bury et al. (2021) ============\n")
    print(f"{'':26s}{'Bury 2021':>22s}{'This work':>22s}")
    print(f"{'cores':26s}{BURY['cores']:>22s}{'MS21, MS66, 64PE406E1':>22s}")
    print(f"{'proxies':26s}{'Mo, U':>22s}{'Mo, U':>22s}")
    print(f"{'transitions':26s}{BURY['n_transitions']:>22d}{nseg:>22d}")
    print(f"{'sequence length':26s}{BURY['seq_len']:>22s}{'500 + 1500':>22s}")
    print(f"{'pred interval':26s}{'last 20%':>22s}{'last 20%':>22s}")
    print(f"\n{'DETECTION':26s}")
    print(f"{'  pooled ROC AUC (DL)':26s}{BURY['auc_dl']:>22.3f}{auc:>22.3f}   (DL-only {aucd:.3f})")
    print(f"{'  variance AUC':26s}{BURY['auc_var']:>22.3f}{OUR_VAR_AUC:>22.3f}   (testing/classical_ews_auc.py)")
    print(f"{'  lag-1 AC AUC':26s}{BURY['auc_ac']:>22.3f}{OUR_AC_AUC:>22.3f}   (testing/classical_ews_auc.py)")
    print(f"{'  P(trans) forced':26s}{'~1.0':>22s}{mf:>22.3f}")
    print(f"{'  P(trans) AR(1) null':26s}{'~0.5':>22s}{mn:>22.3f}")
    print(f"\n{'FAVOURED TYPE (late 20%)':26s}")
    for i, c in enumerate(CLASSES):
        print(f"{'  ' + c:26s}{bury_freq[i]*100:>21.0f}%{our_freq[i]*100:>21.0f}%")
    print(f"\n  Bury verdict : FOLD   |   Our verdict : "
          f"{CLASSES[our_freq.argmax()].upper()}")

    # ── figure ──────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(4)
    ax1.bar(x - 0.2, bury_freq, 0.4, label="Bury et al. 2021", color="#888")
    ax1.bar(x + 0.2, our_freq, 0.4, label="This work",
            color=["#E07B1A", "#1f77b4", "#2D8A4E", "#7165D0"])
    ax1.set_xticks(x); ax1.set_xticklabels(["F", "H", "T", "N"])
    ax1.set_ylabel("favoured-bifurcation frequency")
    ax1.set_title("Favoured type — anoxia (Mo, U, late 20%)")
    ax1.legend(fontsize=8); ax1.set_ylim(0, 1)

    groups = ["Deep\nlearning", "Variance\ntrend", "Lag-1 AC\ntrend"]
    bury_v = [BURY["auc_dl"], BURY["auc_var"], BURY["auc_ac"]]
    our_v = [auc, OUR_VAR_AUC, OUR_AC_AUC]
    gx = np.arange(3)
    ax2.bar(gx - 0.2, bury_v, 0.4, label="Bury et al. 2021", color="#888")
    ax2.bar(gx + 0.2, our_v, 0.4, label="This work", color="#1f77b4")
    ax2.axhline(0.5, color="#aaa", ls="--", lw=1)
    ax2.set_xticks(gx); ax2.set_xticklabels(groups)
    ax2.set_ylim(0.4, 1.03); ax2.set_ylabel("ROC AUC (forced vs AR(1) null)")
    ax2.set_title("Detection: DL vs classical indicators")
    ax2.legend(fontsize=8)
    for i, (b, o) in enumerate(zip(bury_v, our_v)):
        ax2.text(i - 0.2, b + 0.01, f"{b:.2f}", ha="center", fontsize=7)
        ax2.text(i + 0.2, o + 0.01, f"{o:.2f}", ha="center", fontsize=7)

    fig.suptitle("Anoxia / sapropel onsets — this work vs Bury et al. (2021)", fontsize=11)
    fig.tight_layout()
    out = REPO_ROOT / "test_result/comparison/compare_bury.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  saved: {out}")


if __name__ == "__main__":
    main()
