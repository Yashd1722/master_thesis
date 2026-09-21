"""
testing/classical_ews_auc.py

Bury's "plain" early-warning baselines on OUR data: how well does a rising trend
in rolling variance / lag-1 autocorrelation separate the real pre-sapropel
records from their AR(1) null surrogates?  (Bury 2021 anoxia: variance AUC 0.90,
lag-1 AC AUC 0.59.)

No models, no retraining — just rolling statistics on the residual CSVs:
  dataset/.../clean_dataset/<core>/<core>_<sap>_forced.csv        (<el>_residuals)
  dataset/.../clean_dataset/<core>/<core>_<sap>_<el>_ar1_null.csv (null_000..)

Recipe (matches Bury test_empirical/anoxia/compute_roc.py):
  * rolling variance & lag-1 AC over the residuals (window = ROLL_FRAC of segment)
  * restrict to the last 20% before the transition
  * indicator = Kendall tau of that statistic vs time  (does the warning rise?)
  * ROC: forced trajectories truth=1, null surrogates truth=0
"""

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN = REPO_ROOT / "dataset/pangaea_923197/datasets/clean_dataset"
ELEMENTS = ["Mo", "U"]
ROLL_FRAC = 0.5        # rolling-window length as a fraction of the segment
LATE_FRAC = 0.8        # keep the last 20% before the transition (Bury)


def _roll_stats(x):
    """Rolling variance and lag-1 AC of 1-D array x (window = ROLL_FRAC*len)."""
    s = pd.Series(x).dropna()
    w = max(5, int(len(s) * ROLL_FRAC))
    var = s.rolling(w).var()
    ac = s.rolling(w).apply(lambda v: pd.Series(v).autocorr(lag=1), raw=False)
    return var.values, ac.values


def _late_tau(stat):
    """Kendall tau of the last 20% of `stat` vs time (nan-safe)."""
    stat = np.asarray(stat, float)
    stat = stat[~np.isnan(stat)]
    if len(stat) < 5:
        return np.nan
    tail = stat[int(len(stat) * LATE_FRAC):]
    if len(tail) < 5:
        return np.nan
    tau, _ = kendalltau(np.arange(len(tail)), tail)
    return tau


def main():
    tau_var, tau_ac, truth = [], [], []
    n_forced = n_null = 0

    for forced_csv in sorted(glob.glob(str(CLEAN / "*" / "*_forced.csv"))):
        m = re.match(r"(.+)_(S\d+)_forced\.csv", Path(forced_csv).name)
        if not m:
            continue
        core, sap = m.group(1), m.group(2)
        df_f = pd.read_csv(forced_csv)

        for el in ELEMENTS:
            rcol = f"{el}_residuals"
            if rcol not in df_f.columns:
                continue
            null_csv = CLEAN / core / f"{core}_{sap}_{el}_ar1_null.csv"
            if not null_csv.exists():
                continue

            # forced trajectory  -> truth 1
            v, a = _roll_stats(df_f[rcol].values)
            tv, ta = _late_tau(v), _late_tau(a)
            if not (np.isnan(tv) and np.isnan(ta)):
                tau_var.append(tv); tau_ac.append(ta); truth.append(1)
                n_forced += 1

            # each AR(1) null surrogate -> truth 0
            df_n = pd.read_csv(null_csv)
            for c in [c for c in df_n.columns if c.startswith("null_")]:
                v, a = _roll_stats(df_n[c].values)
                tv, ta = _late_tau(v), _late_tau(a)
                if not (np.isnan(tv) and np.isnan(ta)):
                    tau_var.append(tv); tau_ac.append(ta); truth.append(0)
                    n_null += 1

    tau_var = np.array(tau_var, float)
    tau_ac = np.array(tau_ac, float)
    truth = np.array(truth)

    def auc(ind):
        ok = ~np.isnan(ind)
        return roc_auc_score(truth[ok], ind[ok])

    print(f"\nClassical EWS baselines on our data  (cores MS21/MS66/64PE406E1, "
          f"proxies {ELEMENTS}, late {int((1-LATE_FRAC)*100)}%)")
    print(f"  forced trajectories : {n_forced}")
    print(f"  null surrogates     : {n_null}")
    print(f"  Variance  (Kendall-tau trend) AUC : {auc(tau_var):.3f}   [Bury 2021: 0.90]")
    print(f"  Lag-1 AC  (Kendall-tau trend) AUC : {auc(tau_ac):.3f}   [Bury 2021: 0.59]")


if __name__ == "__main__":
    main()
