"""Bifurcation-type vote counts on PANGAEA forced segments (Bury-style).

No ground-truth bifurcation type exists for real anoxia transitions, so
this reports the same thing Bury's own repo reports for the anoxia case
(test_empirical/anoxia/compute_roc.py): the argmax class ("favoured
bifurcation") of each forced window's probability vector, tabulated
separately for an early band (60-80% of the way from segment start to
transition) and a late band (80-100%, i.e. just before transition).

    python testing/bif_vote_counts.py [--config config.yaml]

Output:
    results/summary/pangaea_bif_votes.csv   one row per (model, band)
"""
import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.constants import load_config
from testing.bif_common import CLASSES, LATE_BAND, band_probs, iter_pangaea_results

BANDS = {"early": (0.6, 0.8), "late": LATE_BAND}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(REPO_ROOT / args.config)
    test_root = REPO_ROOT / cfg["paths"]["test_results"]

    counts = {}  # (model, band) -> {class: n}
    for r in iter_pangaea_results(test_root):
        model = r["model"]
        for band, (lo, hi) in BANDS.items():
            for p in band_probs(r, lo, hi):
                key = (model, band)
                counts.setdefault(key, {c: 0 for c in CLASSES})
                counts[key][CLASSES[p.index(max(p))]] += 1

    rows = []
    for (model, band), c in sorted(counts.items()):
        total = sum(c.values())
        row = {"model": model, "band": band, "n": total}
        for cls in CLASSES:
            row[cls] = c[cls]
            row[f"{cls}_pct"] = round(100 * c[cls] / total, 1) if total else 0.0
        rows.append(row)

    out_dir = REPO_ROOT / "results" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pangaea_bif_votes.csv"
    with open(out_path, "w", newline="") as f:
        fieldnames = ["model", "band", "n"] + [
            x for cls in CLASSES for x in (cls, f"{cls}_pct")
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} rows written to {out_path}")


if __name__ == "__main__":
    main()
