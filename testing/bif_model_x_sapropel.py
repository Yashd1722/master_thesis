"""PANGAEA late-band bifurcation-type vote, per (model, dataset) x sapropel.

Same late-band argmax logic as bif_cross_model_agreement.py, pivoted the
other way round: one row per trained checkpoint, one column per sapropel,
majority class across that sapropel's 5 elements with its vote fraction.
Useful for spotting a single architecture flip-flopping between its ts_500
and ts_1500 checkpoints, which the per-segment view doesn't show directly.

    python testing/bif_model_x_sapropel.py [--config config.yaml]

Output:
    results/summary/pangaea_model_x_sapropel.csv
"""
import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.constants import load_config
from testing.bif_common import (
    CLASSES, TRUSTWORTHY_MODELS, iter_pangaea_results, late_band_avg_probs,
)


def late_band_class(r):
    avg = late_band_avg_probs(r)
    if avg is None:
        return None
    return CLASSES[avg.index(max(avg))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(REPO_ROOT / args.config)
    test_root = REPO_ROOT / cfg["paths"]["test_results"]

    per_model_sap = {}
    for r in iter_pangaea_results(test_root, TRUSTWORTHY_MODELS):
        cls = late_band_class(r)
        if cls is None:
            continue
        key = (r["model"], r["dataset"], r["core"], r["sapropel"])
        per_model_sap.setdefault(key, {c: 0 for c in CLASSES})[cls] += 1

    saps = sorted({(k[2], k[3]) for k in per_model_sap})
    model_ds = sorted({(k[0], k[1]) for k in per_model_sap})

    rows = []
    for model, ds in model_ds:
        row = {"model": model, "dataset": ds}
        for core, sap in saps:
            tally = per_model_sap.get((model, ds, core, sap))
            col = f"{core}_{sap}"
            if not tally:
                row[col] = ""
                continue
            top = max(tally, key=tally.get)
            n = sum(tally.values())
            row[col] = f"{top}({tally[top]}/{n})"
        rows.append(row)

    out_dir = REPO_ROOT / "results" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pangaea_model_x_sapropel.csv"
    fieldnames = ["model", "dataset"] + [f"{c}_{s}" for c, s in saps]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} rows, {len(saps)} sapropel columns written to {out_path}")


if __name__ == "__main__":
    main()
