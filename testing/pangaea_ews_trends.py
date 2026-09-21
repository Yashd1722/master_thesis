"""testing/pangaea_ews_trends.py

Classical EWS trend check on the raw PANGAEA signal itself: for each test
sapropel (Chapter 8, config.yaml `role: test`), compute the Kendall-tau trend
of rolling variance and rolling lag-1 autocorrelation over the forced
pre-transition window -- the same statistic src/rolling_window.py computes
per-window (ktau_var, ktau_ac), applied here to the full window's raw
variance/lag1_ac arrays already stored in every result.json (identical across
models for a given (core, sapropel, element), since these are properties of
the raw signal, not of any classifier).

A significant, strongly positive tau for both variance and lag-1 AC is the
textbook critical-slowing-down signature. This script exists to check whether
that signature is actually present, sapropel by sapropel, rather than assume
it -- and to see whether its presence or absence lines up with which
bifurcation type the trustworthy model roster favours there (Chapter 8,
Section 8.4).

Usage: python testing/pangaea_ews_trends.py
"""
import csv
import glob
import json
import sys
from pathlib import Path

from scipy.stats import kendalltau

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

TEST_SAPROPELS = [
    ("64PE406E1", "S3"), ("64PE406E1", "S4"), ("64PE406E1", "S5"), ("64PE406E1", "S6"),
    ("MS21", "S1"), ("MS66", "S1"), ("MS66", "S3"),
]
ELEMENTS = ("Mo", "U")


def main():
    test_root = REPO_ROOT / "test_result"
    out_rows = []

    for core, sap in TEST_SAPROPELS:
        for element in ELEMENTS:
            matches = glob.glob(str(test_root / f"*_pangaea_{core}_{sap}_{element}_pangaea" / "result.json"))
            if not matches:
                continue
            d = json.loads(Path(matches[0]).read_text())
            var, ac = d.get("variance"), d.get("lag1_ac")
            if not var or not ac:
                continue
            idx = list(range(len(var)))
            tau_var, p_var = kendalltau(idx, var)
            tau_ac, p_ac = kendalltau(idx, ac)
            out_rows.append({
                "core": core, "sapropel": sap, "element": element, "n_windows": len(var),
                "var_start": round(var[0], 4), "var_end": round(var[-1], 4),
                "tau_var": round(float(tau_var), 4), "p_var": round(float(p_var), 4),
                "ac_start": round(ac[0], 4), "ac_end": round(ac[-1], 4),
                "tau_ac": round(float(tau_ac), 4), "p_ac": round(float(p_ac), 4),
            })

    out = REPO_ROOT / "results/summary/pangaea_ews_trends.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"{len(out_rows)} (sapropel, element) rows written to {out}\n")
    print(f"{'segment':22s} {'n':>3s} {'var trend':>20s} {'tau_var':>9s} {'ac trend':>18s} {'tau_ac':>9s}  sig?")
    for r in out_rows:
        seg = f"{r['core']}/{r['sapropel']}/{r['element']}"
        var_trend = f"{r['var_start']:.2f}->{r['var_end']:.2f}"
        ac_trend = f"{r['ac_start']:.2f}->{r['ac_end']:.2f}"
        sig = "both p<.001" if r["p_var"] < 0.001 and r["p_ac"] < 0.001 else "check p"
        print(f"{seg:22s} {r['n_windows']:>3d} {var_trend:>20s} {r['tau_var']:>9.3f} {ac_trend:>18s} {r['tau_ac']:>9.3f}  {sig}")


if __name__ == "__main__":
    main()
