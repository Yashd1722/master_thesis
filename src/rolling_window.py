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
    inf_cfg    = cfg["inference"]
    n          = len(residuals)
    win        = max(10, int(inf_cfg["rolling_window_frac"] * n))
    start_frac = inf_cfg["start_frac"].get(core_name,
                                            inf_cfg["start_frac"]["default"])
    n_steps    = inf_cfg["prediction_steps"]

    start_pos  = max(win, int(start_frac * n))
    end_pos    = min(n,   int(inf_cfg["end_frac"] * n))
    if start_pos >= end_pos:
        start_pos, end_pos = win, n

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

            for seg_type, fname in [
                ("forced",       f"{core_name}_{sap_id}_forced.csv"),
                ("neutral",      f"{core_name}_{sap_id}_neutral_test.csv"),
            ]:
                fpath = clean_dir / fname
                if not fpath.exists():
                    logger.warning(
                        f"Missing: {fname}. "
                        f"Run: python src/pangea_cleaner.py --core {core_name}"
                    )
                    continue

                df        = pd.read_csv(fpath)
                resid_col = f"{element}_residuals"

                if resid_col not in df.columns:
                    logger.debug(f"{element} residuals not in {fname}")
                    continue

                residuals = df[resid_col].values.astype(np.float64)
                ages      = df["age_kyr_bp"].values.astype(np.float64)

                # Drop NaN
                valid = ~np.isnan(residuals)
                if valid.sum() < 10:
                    logger.warning(
                        f"{core_name}/{sap_id}/{element}/{seg_type}: "
                        f"only {valid.sum()} valid points"
                    )
                    continue

                result = compute_rolling_ews(
                    residuals    = residuals[valid],
                    ages_kyr_bp  = ages[valid],
                    element      = element,
                    core_name    = core_name,
                    sapropel_id  = sap_id,
                    segment_type = seg_type,
                    cfg          = cfg,
                    ts_len       = ts_len,
                )
                elem_results[seg_type] = result

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
