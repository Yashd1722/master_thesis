"""Cross-model agreement on PANGAEA 4-class bifurcation-type votes.

No ground-truth bifurcation type exists for real anoxia transitions, so the
only available proxy for "is this type-label trustworthy" is whether
independently-trained architectures agree with each other. This computes,
per (core, sapropel, element), the late-band (80-100% of the way to
transition) majority-vote class across a fixed set of models that already
passed the ground-truth synthetic check (results/summary/zenodo.csv strong
on both datasets), and reports how often they actually agree.

Two extra statistics go beyond raw vote-counting (which is all Bury's own
anoxia analysis does):
  - Shannon entropy of the models' averaged probability vector per segment,
    which uses each model's actual confidence instead of collapsing it to
    a single winner.
  - Fleiss' kappa across all segments, which corrects raw agreement for the
    agreement you'd expect from chance alone given the class frequencies.

    python testing/bif_cross_model_agreement.py [--config config.yaml]

Output:
    results/summary/pangaea_cross_model_agreement.csv
        one row per (core, sapropel, element): n_models, majority class,
        agreement fraction, vote tally, ensemble entropy (bits, 0-2).
    Fleiss' kappa is printed as a single summary statistic.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.constants import load_config
from testing.bif_common import (
    CLASSES, TRUSTWORTHY_MODELS, iter_pangaea_results, late_band_avg_probs,
)


def shannon_entropy_bits(p):
    return -sum(x * math.log2(x) for x in p if x > 0)


def fleiss_kappa(tallies):
    n_total = sum(sum(t.values()) for t in tallies)
    if n_total == 0:
        return float("nan")
    p_j = {c: sum(t[c] for t in tallies) / n_total for c in CLASSES}
    p_i_weighted_sum = 0.0
    for t in tallies:
        n_i = sum(t.values())
        if n_i < 2:
            continue
        p_i = sum(n_ij * (n_ij - 1) for n_ij in t.values()) / (n_i * (n_i - 1))
        p_i_weighted_sum += n_i * p_i
    p_bar = p_i_weighted_sum / n_total
    p_e = sum(v * v for v in p_j.values())
    if p_e >= 1.0:
        return float("nan")
    return (p_bar - p_e) / (1 - p_e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--min_models", type=int, default=3)
    parser.add_argument("--elements", default=None,
                        help="comma-separated element filter, e.g. Mo,U (default: all)")
    args = parser.parse_args()

    cfg = load_config(REPO_ROOT / args.config)
    test_root = REPO_ROOT / cfg["paths"]["test_results"]

    per_segment = {}
    elements = set(args.elements.split(",")) if args.elements else None
    for r in iter_pangaea_results(test_root, TRUSTWORTHY_MODELS, elements):
        probs = late_band_avg_probs(r)
        if probs is None:
            continue
        key = (r["core"], r["sapropel"], r["element"])
        per_segment.setdefault(key, {})[f'{r["model"]}_{r["dataset"]}'] = probs

    rows = []
    agree_fracs = []
    tallies = []
    for (core, sap, element), model_probs in sorted(per_segment.items()):
        if len(model_probs) < args.min_models:
            continue
        tally = {c: 0 for c in CLASSES}
        for probs in model_probs.values():
            tally[CLASSES[probs.index(max(probs))]] += 1
        top_cls = max(tally, key=tally.get)
        frac = tally[top_cls] / len(model_probs)
        agree_fracs.append(frac)
        tallies.append(tally)

        ensemble = [sum(p[i] for p in model_probs.values()) / len(model_probs)
                    for i in range(len(CLASSES))]
        entropy = shannon_entropy_bits(ensemble)

        rows.append({
            "core": core, "sapropel": sap, "element": element,
            "n_models": len(model_probs), "majority_class": top_cls,
            "agreement_frac": round(frac, 3),
            "entropy_bits": round(entropy, 3),
            "entropy_norm": round(entropy / math.log2(len(CLASSES)), 3),
            **{f"votes_{c}": tally[c] for c in CLASSES},
        })

    out_dir = REPO_ROOT / "results" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{'_'.join(sorted(elements))}" if elements else ""
    out_path = out_dir / f"pangaea_cross_model_agreement{suffix}.csv"
    with open(out_path, "w", newline="") as f:
        fieldnames = ["core", "sapropel", "element", "n_models", "majority_class",
                      "agreement_frac", "entropy_bits", "entropy_norm"] + [
                      f"votes_{c}" for c in CLASSES]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    mean_agree = sum(agree_fracs) / len(agree_fracs) if agree_fracs else float("nan")
    mean_entropy = sum(r["entropy_norm"] for r in rows) / len(rows) if rows else float("nan")
    kappa = fleiss_kappa(tallies)
    print(f"{len(rows)} segments written to {out_path}")
    print(f"mean cross-model agreement: {mean_agree:.3f}")
    print(f"min: {min(agree_fracs):.3f}  max: {max(agree_fracs):.3f}")
    print(f"mean normalised ensemble entropy: {mean_entropy:.3f}  (0=all models agree, 1=max confusion)")
    print(f"Fleiss' kappa across all segments: {kappa:.3f}  "
          f"(0=no better than chance, 1=perfect agreement)")


if __name__ == "__main__":
    main()
