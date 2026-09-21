"""
testing/consensus_bif_type.py

Per-sapropel bifurcation-type CONSENSUS across the trustworthy model ensemble,
weighted by each model's synthetic (Zenodo) macro-F1 — the honest way to get a
concrete type statement out of an unstable per-model argmax. No retraining, pure
post-processing of the saved result.json files.

Method (Bury's late-interval convention):
  * prediction unit = (model, proxy in {U,Mo}, sequence length, late-20% window)
  * favoured class  = argmax(fold, hopf, transcritical, null) of the ensemble-mean
                      softmax for that unit
  * each unit's vote is weighted by the model's Zenodo macro-F1 (a model that
    demonstrably classifies type well on labelled data counts more)
  * per (core, sapropel): weighted frequency of each favoured class
  * bootstrap 95% CI by resampling prediction units
  * assign a type only if  top freq >= THRESHOLD  and its CI low > 2nd class freq;
    otherwise "undetermined"

Usage:  python testing/consensus_bif_type.py [--threshold 0.6] [--elements U,Mo]
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["fold", "hopf", "transcritical", "null"]
TRUSTWORTHY = ["cnn_lstm", "lstm", "inceptiontime", "patchtst", "resnet", "tcn",
               "rnn_fcn", "arsenal", "tsf", "st", "grsf", "rocket",
               "minirocket", "mrsqm", "catch22"]


def zenodo_macro_f1(results_dir: Path) -> dict:
    acc = {}
    for f in glob.glob(str(results_dir / "*_zenodo" / "result.json")):
        d = json.loads(Path(f).read_text())
        acc.setdefault(d.get("model"), []).append(
            d.get("macro_f1", d.get("accuracy", np.nan)))
    return {m: float(np.nanmean(v)) for m, v in acc.items()}


def late_favoured(P, interval=0.8, n=10):
    lo = int(np.floor(len(P) * interval))
    tail = P[lo:]
    if len(tail) > n:
        tail = tail[np.round(np.linspace(0, len(tail) - 1, n)).astype(int)]
    return tail.argmax(axis=1) if len(tail) else np.empty(0, int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="test_result")
    ap.add_argument("--elements", default="U,Mo")
    ap.add_argument("--threshold", type=float, default=0.60)
    ap.add_argument("--interval", type=float, default=0.8)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    results_dir = REPO_ROOT / args.results
    elems = set(args.elements.split(","))
    w_model = zenodo_macro_f1(results_dir)

    # per (core, sap): list of (favoured_class_idx, weight)
    votes: dict = {}
    for f in glob.glob(str(results_dir / "*_pangaea" / "result.json")):
        d = json.loads(Path(f).read_text())
        m = d.get("model")
        if m not in TRUSTWORTHY or d.get("element") not in elems:
            continue
        if not all(d.get(f"p_{c}") for c in CLASSES):
            continue
        P = np.column_stack([d[f"p_{c}"] for c in CLASSES])
        w = w_model.get(m, 0.5)
        key = (d.get("core"), d.get("sapropel"))
        for k in late_favoured(P, args.interval):
            votes.setdefault(key, []).append((int(k), w))

    rng = np.random.default_rng(0)
    print(f"\nConsensus bifurcation type  (trustworthy ensemble, proxies={sorted(elems)}, "
          f"weight=Zenodo macro-F1, threshold={args.threshold})\n")
    hdr = f"{'core/sapropel':16s}{'n':>5s}   " + "".join(f"{c:>14s}" for c in CLASSES) + "   verdict"
    print(hdr)
    print("-" * len(hdr))

    for key in sorted(votes):
        units = votes[key]
        idx = np.array([u[0] for u in units])
        wt = np.array([u[1] for u in units])

        def wfreq(sel):
            w = wt[sel]
            return np.array([w[idx[sel] == k].sum() for k in range(4)]) / w.sum()

        point = wfreq(np.ones(len(units), bool))
        boot = np.array([wfreq(rng.integers(0, len(units), len(units)))
                         for _ in range(args.boot)])
        ci_lo = np.percentile(boot, 2.5, axis=0)

        order = np.argsort(point)[::-1]
        top, second = order[0], order[1]
        if point[top] >= args.threshold and ci_lo[top] > point[second]:
            verdict = f"{CLASSES[top].upper()}  ({point[top]*100:.0f}%, CI≥{ci_lo[top]*100:.0f}%)"
        else:
            verdict = f"undetermined ({CLASSES[top]} {point[top]*100:.0f}% vs {CLASSES[second]} {point[second]*100:.0f}%)"

        row = f"{key[0]+'/'+key[1]:16s}{len(units):5d}   " + \
              "".join(f"{p*100:13.0f}%" for p in point) + f"   {verdict}"
        print(row)

    print("\n(n = weighted prediction units = models x proxies x lengths x late windows)")


if __name__ == "__main__":
    main()
