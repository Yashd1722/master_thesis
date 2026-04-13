"""
src/rolling_window.py
=====================
Universal rolling window engine for ALL empirical testing.

Processes ALL 5 elements: Al, Ba, Mo, Ti, U
Processes BOTH segment types: forced (pre-transition) and neutral
Neutral segments are required for proper ROC computation matching the paper.

Paper protocol (Bury 2021 / Ma 2025):
  - 40 equally-spaced predictions between start_frac and 100%
  - Rolling window = 50% of segment length
  - Forced (pre-transition) = positive class for ROC
  - Neutral (far from transition) = negative class for ROC

Public API:
  run_all_sapropels(core_name, cfg, ts_len)
      → {sap_id: {element: {"forced": Result, "neutral": Result}}}
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

ELEMENTS       = ["Al", "Ba", "Mo", "Ti", "U"]
PRIMARY_ELEMENT = "Mo"


# =============================================================================
#  Result container
# =============================================================================

@dataclass
class RollingWindowResult:
    """Outputs from one rolling window run on one segment of one element."""
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


# =============================================================================
#  Signal processing helpers
# =============================================================================

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


# =============================================================================
#  Core rolling window
# =============================================================================

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

    # KEY FIX: always sweep from the FIRST valid position (win) to the END.
    # Using start_frac = 0.80 on a 130-point series gives only 26 steps
    # which makes variance/AC look flat. We need the full sweep to see
    # the CSD trend clearly — exactly as Bury 2021 and Ma 2025 do.
    # The start_frac (60% or 80%) is only used for ROC threshold sweeping
    # NOT for the time series indicator plots.
    n_steps   = inf_cfg["prediction_steps"]
    start_pos = win          # always start from first valid window position
    end_pos   = n            # always go to end of series

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


# =============================================================================
#  Run all sapropels — all elements — both segment types
# =============================================================================

def _combine_null_results(results: List[RollingWindowResult]) -> RollingWindowResult:
    """
    Combine multiple null series results into one RollingWindowResult
    by stacking all predictions. This gives N_null × n_steps total
    predictions for the negative class in ROC.
    """
    all_positions  = np.concatenate([r.positions     for r in results])
    all_ages       = np.concatenate([r.ages_kyr_bp   for r in results])
    all_variance   = np.concatenate([r.variance      for r in results])
    all_ac         = np.concatenate([r.lag1_ac       for r in results])
    all_dl_inputs  = []
    for r in results:
        all_dl_inputs.extend(r.dl_inputs)

    from scipy.stats import kendalltau as _kt
    step_idx     = np.arange(len(all_variance))
    ktau_var, _  = _kt(step_idx, all_variance)
    ktau_ac, _   = _kt(step_idx, all_ac)

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
    All 5 elements × (forced + neutral) segments.

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
        sap_id     = sap["id"]
        sap_results: Dict = {}

        for element in ELEMENTS:
            elem_results: Dict = {}

            # ── Forced segment ─────────────────────────────────────────────────
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
                result_forced = compute_rolling_ews(
                    residuals    = forced_residuals[valid_f],
                    ages_kyr_bp  = forced_ages[valid_f],
                    element      = element,
                    core_name    = core_name,
                    sapropel_id  = sap_id,
                    segment_type = "forced",
                    cfg          = cfg,
                    ts_len       = ts_len,
                )
                elem_results["forced"] = result_forced

            # ── AR(1) null series (Bury 2021 protocol) ────────────────────────
            # Load the 10 AR(1) null series generated by pangea_cleaner.py
            # Each null series is one column in the CSV.
            # We run rolling window on EACH null series separately,
            # then combine all predictions as the negative class for ROC.
            null_path = clean_dir / f"{core_name}_{sap_id}_{element}_ar1_null.csv"

            if null_path.exists() and valid_f.sum() >= 10:
                df_null = pd.read_csv(null_path)
                null_cols = [c for c in df_null.columns if c.startswith("null_")]
                null_ages = df_null["age_kyr_bp"].values.astype(np.float64)                             if "age_kyr_bp" in df_null.columns                             else forced_ages[valid_f]

                all_null_results = []
                for null_col in null_cols:
                    null_resids = df_null[null_col].values.astype(np.float64)
                    valid_n     = ~np.isnan(null_resids)
                    if valid_n.sum() < 10:
                        continue
                    r = compute_rolling_ews(
                        residuals    = null_resids[valid_n],
                        ages_kyr_bp  = null_ages[valid_n],
                        element      = element,
                        core_name    = core_name,
                        sapropel_id  = sap_id,
                        segment_type = "neutral",
                        cfg          = cfg,
                        ts_len       = ts_len,
                    )
                    all_null_results.append(r)

                if all_null_results:
                    # Combine: stack all null predictions into one result
                    # by concatenating positions/variance/ac/dl_inputs
                    combined = _combine_null_results(all_null_results)
                    elem_results["neutral"] = combined
                    logger.info(
                        f"  {core_name}/{sap_id}/{element}/neutral: "
                        f"{len(all_null_results)} AR(1) series × "
                        f"{len(all_null_results[0].positions)} steps = "
                        f"{len(combined.dl_inputs)} total predictions"
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


# =============================================================================
#  Quick check
# =============================================================================

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
