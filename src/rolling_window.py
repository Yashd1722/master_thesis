"""
src/rolling_window.py
Rolling window EWS engine for PANGAEA empirical testing.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from scipy.stats import kendalltau
from typing import List, Dict, Optional
import yaml

logger = logging.getLogger(__name__)

ELEMENTS        = ["Al", "Ba", "Mo", "Ti", "U"]
PRIMARY_ELEMENT = "Mo"


@dataclass
class RollingWindowResult:
    positions:     np.ndarray
    ages_kyr_bp:   np.ndarray
    variance:      np.ndarray
    lag1_ac:       np.ndarray
    ktau_variance: float
    ktau_lag1_ac:  float
    dl_inputs:     List[np.ndarray] = field(default_factory=list)
    dl_probs:      Optional[np.ndarray] = None
    p_transition:  Optional[np.ndarray] = None
    core_name:     str = ""
    sapropel_id:   str = ""
    element:       str = "Mo"
    segment_type:  str = "forced"
    n_series:      int = 0

    def to_dataframe(self, class_names: List[str]) -> pd.DataFrame:
        df = pd.DataFrame({
            "position":      self.positions,
            "age_kyr_bp":    self.ages_kyr_bp,
            "variance":      self.variance,
            "lag1_ac":       self.lag1_ac,
            "ktau_variance": self.ktau_variance,
            "ktau_lag1_ac":  self.ktau_lag1_ac,
            "segment_type":  self.segment_type,
            "element":       self.element,
        })
        if self.dl_probs is not None:
            for i, name in enumerate(class_names):
                df[f"p_{name}"] = self.dl_probs[:, i]
        if self.p_transition is not None:
            df["p_transition"] = self.p_transition
        return df


def _variance(series: np.ndarray) -> float:
    return float(np.var(series, ddof=1)) if len(series) >= 2 else 0.0


def _lag1_ac(series: np.ndarray) -> float:
    if len(series) < 3:
        return 0.0
    x, y = series[:-1], series[1:]
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def prepare_dl_input(residuals: np.ndarray, position: int,
                     ts_len: int) -> np.ndarray:
    """Right-aligned, normalised, left-padded to ts_len."""
    seg   = residuals[:position].astype(np.float64)
    denom = np.mean(np.abs(seg))
    if denom > 1e-10:
        seg = seg / denom
    n = len(seg)
    if n >= ts_len:
        return seg[-ts_len:].astype(np.float32)
    out      = np.zeros(ts_len, dtype=np.float32)
    out[-n:] = seg.astype(np.float32)
    return out


def compute_rolling_ews(
    residuals:    np.ndarray,
    ages_kyr_bp:  np.ndarray,
    element:      str,
    core_name:    str,
    sapropel_id:  str,
    segment_type: str,
    cfg:          dict,
    ts_len:       int,
) -> RollingWindowResult:
    """
    Compute rolling window EWS for one element on one segment.
    Returns RollingWindowResult with dl_inputs ready for model inference.
    """
    inf_cfg = cfg["inference"]
    n       = len(residuals)
    win     = max(10, int(inf_cfg["rolling_window_frac"] * n))

    n_steps   = inf_cfg["prediction_steps"]
    start_pos = win
    end_pos   = n

    positions = np.clip(
        np.linspace(start_pos, end_pos, n_steps, dtype=int), win, n
    )

    variances = np.zeros(n_steps)
    ac1s      = np.zeros(n_steps)
    step_ages = np.zeros(n_steps)
    dl_inputs = []

    for i, pos in enumerate(positions):
        win_seg       = residuals[pos - win: pos]
        variances[i]  = _variance(win_seg)
        ac1s[i]       = _lag1_ac(win_seg)
        step_ages[i]  = ages_kyr_bp[pos - 1] if pos - 1 < len(ages_kyr_bp) else np.nan
        dl_inputs.append(prepare_dl_input(residuals, pos, ts_len))

    step_idx     = np.arange(n_steps)
    ktau_var, _  = kendalltau(step_idx, variances)
    ktau_ac, _   = kendalltau(step_idx, ac1s)

    return RollingWindowResult(
        positions      = positions,
        ages_kyr_bp    = step_ages,
        variance       = variances,
        lag1_ac        = ac1s,
        ktau_variance  = float(ktau_var),
        ktau_lag1_ac   = float(ktau_ac),
        dl_inputs      = dl_inputs,
        core_name      = core_name,
        sapropel_id    = sapropel_id,
        element        = element,
        segment_type   = segment_type,
        n_series       = n,
    )


def _combine_null_results(results: List[RollingWindowResult]) -> RollingWindowResult:
    all_positions = np.concatenate([r.positions   for r in results])
    all_ages      = np.concatenate([r.ages_kyr_bp for r in results])
    all_variance  = np.concatenate([r.variance    for r in results])
    all_ac        = np.concatenate([r.lag1_ac     for r in results])
    all_dl_inputs = []
    for r in results:
        all_dl_inputs.extend(r.dl_inputs)

    step_idx     = np.arange(len(all_variance))
    ktau_var, _  = kendalltau(step_idx, all_variance)
    ktau_ac, _   = kendalltau(step_idx, all_ac)

    return RollingWindowResult(
        positions      = all_positions,
        ages_kyr_bp    = all_ages,
        variance       = all_variance,
        lag1_ac        = all_ac,
        ktau_variance  = float(ktau_var),
        ktau_lag1_ac   = float(ktau_ac),
        dl_inputs      = all_dl_inputs,
        core_name      = results[0].core_name,
        sapropel_id    = results[0].sapropel_id,
        element        = results[0].element,
        segment_type   = "neutral",
        n_series       = len(all_positions),
    )


def run_all_sapropels(
    core_name: str,
    cfg:       dict,
    ts_len:    int,
) -> Dict[str, Dict[str, Dict[str, RollingWindowResult]]]:
    """
    Process all test sapropels of one core.
    All 5 elements x (forced + neutral) segments.

    Returns:
        {sap_id: {element: {"forced": Result, "neutral": Result}}}
    """
    repo_root  = Path(__file__).resolve().parents[1]
    clean_dir  = repo_root / cfg["paths"]["pangaea_clean"] / core_name
    sapropels  = cfg["pangaea"]["cores"][core_name]["sapropels"]
    test_saps  = [s for s in sapropels if s["role"] == "test"]

    if not test_saps:
        logger.warning(f"No test sapropels for {core_name}")
        return {}

    all_results: Dict = {}

    for sap in test_saps:
        sap_id      = sap["id"]
        sap_results: Dict = {}

        for element in ELEMENTS:
            elem_results: Dict = {}

            forced_path = clean_dir / f"{core_name}_{sap_id}_forced.csv"
            if not forced_path.exists():
                logger.warning(
                    f"Missing forced file for {core_name}/{sap_id}. "
                    f"Run: python src/pangea_cleaner.py --core {core_name}"
                )
                sap_results[element] = elem_results
                continue

            df_forced = pd.read_csv(forced_path)
            resid_col = f"{element}_residuals"

            if resid_col not in df_forced.columns:
                sap_results[element] = elem_results
                continue

            forced_residuals = df_forced[resid_col].values.astype(np.float64)
            forced_ages      = df_forced["age_kyr_bp"].values.astype(np.float64)
            valid_f          = ~np.isnan(forced_residuals)

            if valid_f.sum() >= 10:
                r_valid  = forced_residuals[valid_f]
                a_valid  = forced_ages[valid_f]
                # Sort oldest-first so rolling window approaches transition at END
                sort_idx = np.argsort(a_valid)[::-1]
                result_forced = compute_rolling_ews(
                    residuals    = r_valid[sort_idx],
                    ages_kyr_bp  = a_valid[sort_idx],
                    element      = element,
                    core_name    = core_name,
                    sapropel_id  = sap_id,
                    segment_type = "forced",
                    cfg          = cfg,
                    ts_len       = ts_len,
                )
                elem_results["forced"] = result_forced

            null_path = clean_dir / f"{core_name}_{sap_id}_{element}_ar1_null.csv"

            if null_path.exists() and valid_f.sum() >= 10:
                df_null   = pd.read_csv(null_path)
                null_cols = [c for c in df_null.columns if c.startswith("null_")]
                null_ages = (df_null["age_kyr_bp"].values.astype(np.float64)
                             if "age_kyr_bp" in df_null.columns
                             else forced_ages[valid_f])

                all_null_results = []
                for null_col in null_cols:
                    null_resids = df_null[null_col].values.astype(np.float64)
                    valid_n     = ~np.isnan(null_resids)
                    if valid_n.sum() < 10:
                        continue
                    n_valid  = null_resids[valid_n]
                    na_valid = null_ages[valid_n] if len(null_ages) == len(null_resids) else null_ages[:valid_n.sum()]
                    ns_idx   = np.argsort(na_valid)[::-1]
                    r = compute_rolling_ews(
                        residuals    = n_valid[ns_idx],
                        ages_kyr_bp  = na_valid[ns_idx],
                        element      = element,
                        core_name    = core_name,
                        sapropel_id  = sap_id,
                        segment_type = "neutral",
                        cfg          = cfg,
                        ts_len       = ts_len,
                    )
                    all_null_results.append(r)

                if all_null_results:
                    elem_results["neutral"] = _combine_null_results(all_null_results)
                    logger.info(
                        f"  {core_name}/{sap_id}/{element}/neutral: "
                        f"{len(all_null_results)} AR(1) series x "
                        f"{len(all_null_results[0].positions)} steps"
                    )
            else:
                if not null_path.exists():
                    logger.warning(
                        f"AR(1) null file missing: {null_path.name}. "
                        f"Re-run: python src/pangea_cleaner.py --core {core_name}"
                    )

            sap_results[element] = elem_results
        all_results[sap_id] = sap_results

    return all_results


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--core",   default="MS21",
                        choices=["MS21", "MS66", "64PE406E1"])
    parser.add_argument("--ts_len", type=int, default=500)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg_path = Path(__file__).resolve().parents[1] / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print(f"\nRolling window check: {args.core}  ts_len={args.ts_len}\n")
    results = run_all_sapropels(args.core, cfg, args.ts_len)

    for sap_id, sap_res in results.items():
        print(f"  Sapropel {sap_id}:")
        for elem, elem_res in sap_res.items():
            for seg_type, res in elem_res.items():
                print(
                    f"    {elem:3s}/{seg_type:8s}: "
                    f"N={res.n_series:4d}  steps={len(res.positions)}  "
                    f"ktau_var={res.ktau_variance:+.3f}  "
                    f"ktau_ac={res.ktau_lag1_ac:+.3f}"
                )
    print("\nOK.")
