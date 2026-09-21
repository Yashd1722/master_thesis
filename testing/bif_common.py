"""Shared constants and helpers for the testing/bif_*.py PANGAEA
bifurcation-type analysis scripts (bif_vote_counts, bif_cross_model_agreement,
bif_model_x_sapropel). Kept as one small module so the three scripts don't
each reimplement the same result-loading and late-band-probability logic.
"""
import json
from pathlib import Path

CLASSES = ["fold", "hopf", "transcritical", "null"]
LATE_BAND = (0.8, 1.0)
TRUSTWORTHY_MODELS = {
    "cnn_lstm", "lstm", "inceptiontime", "patchtst", "resnet", "tcn", "rnn_fcn",
    "arsenal", "tsf", "st", "grsf", "rocket", "minirocket", "mrsqm", "catch22",
}


def iter_pangaea_results(test_root, models_filter=None, elements=None):
    for d in sorted(Path(test_root).iterdir()):
        if not (d.is_dir() and d.name.endswith("_pangaea")):
            continue
        try:
            r = json.loads((d / "result.json").read_text())
        except Exception:
            continue
        if "p_fold" not in r:
            continue
        if models_filter and r["model"] not in models_filter:
            continue
        if elements and r["element"] not in elements:
            continue
        yield r


def band_probs(r, lo, hi):
    ages = r["ages_kyr_bp"]
    if len(ages) < 2:
        return []
    t_start, t_trans = ages[0], ages[-1]
    span = t_start - t_trans
    if span <= 0:
        return []
    probs = list(zip(r["p_fold"], r["p_hopf"], r["p_transcritical"], r["p_null"]))
    return [p for age, p in zip(ages, probs) if lo <= (t_start - age) / span <= hi]


def late_band_avg_probs(r):
    late = band_probs(r, *LATE_BAND)
    if not late:
        return None
    return [sum(x) / len(late) for x in zip(*late)]
