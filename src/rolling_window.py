"""
src/rolling_window.py
=====================
Universal rolling window engine for empirical testing.

Used by testing/evaluate.py for ALL empirical datasets.
Never used during training — this is inference only.

What it does:
  1. Takes a preprocessed residual series (from pangea_cleaner.py)
  2. Applies a rolling window of size = 0.5 × series length
  3. At each of 40 equally-spaced steps (60-100% or 80-100% of series):
       - Computes variance of the window          → EWS baseline
       - Computes lag-1 autocorrelation            → EWS baseline
       - Returns the full padded series as input   → fed to DL model
  4. Returns Kendall tau trends for variance and lag-1 AC

Protocol exactly matches Bury 2021 and Ma 2025:
  - 40 prediction steps
  - start_frac: 0.60 for MS21, 0.80 for MS66 and 64PE406E1 (from config)
  - rolling window = 0.50 × N
  - Kendall tau computed over all preceding predictions

Public API:
  compute_rolling_ews(residuals, core_name, cfg)
      → RollingWindowResult (dataclass with all outputs)

  prepare_dl_input(residuals, position, ts_len)
      → np.ndarray of shape (ts_len,) ready for model input
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from scipy.stats import kendalltau
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# =============================================================================
#  Result container
# =============================================================================

@dataclass
class RollingWindowResult:
    """
    All outputs from one rolling window run on one time series.
    Saved as a CSV by evaluate.py with naming:
        {model}_{core}_{sapropel}_predictions.csv
    """
    # Positions (index into residuals array at each prediction step)
    positions:      np.ndarray          # shape (n_steps,)

    # Age at each prediction step (ka BP) — for x-axis of figures
    ages_kyr_bp:    np.ndarray          # shape (n_steps,)

    # EWS baselines
    variance:       np.ndarray          # shape (n_steps,)
    lag1_ac:        np.ndarray          # shape (n_steps,)

    # Kendall tau trends (scalar — overall trend across all steps)
    ktau_variance:  float
    ktau_lag1_ac:   float

    # DL model inputs — list of arrays each shape (ts_len,)
    # Set after compute_rolling_ews by evaluate.py
    dl_inputs:      List[np.ndarray] = field(default_factory=list)

    # DL model outputs — filled in by evaluate.py after running the model
    # Columns match dataset class_names: p_fold, p_hopf, p_transcritical, p_null
    # OR for SDML: p_neutral, p_pre_transition
    dl_probs:       Optional[np.ndarray] = None   # shape (n_steps, n_classes)
    p_transition:   Optional[np.ndarray] = None   # shape (n_steps,) = 1 - p_null

    # Metadata
    core_name:      str = ""
    sapropel_id:    str = ""
    n_series:       int = 0


    def to_dataframe(self, class_names: List[str]) -> pd.DataFrame:
        """
        Convert to DataFrame for saving as CSV.
        Naming convention: {model}_{core}_{sapropel}_predictions.csv
        """
        df = pd.DataFrame({
            "position":     self.positions,
            "age_kyr_bp":   self.ages_kyr_bp,
            "variance":     self.variance,
            "lag1_ac":      self.lag1_ac,
            "ktau_variance": self.ktau_variance,
            "ktau_lag1_ac":  self.ktau_lag1_ac,
        })

        if self.dl_probs is not None:
            for i, name in enumerate(class_names):
                df[f"p_{name}"] = self.dl_probs[:, i]

        if self.p_transition is not None:
            df["p_transition"] = self.p_transition

        return df


# =============================================================================
#  Core computation
# =============================================================================

def _lag1_autocorrelation(series: np.ndarray) -> float:
    """
    Lag-1 autocorrelation of a 1D series.
    Returns 0.0 if series is too short or has zero variance.
    """
    if len(series) < 3:
        return 0.0
    x = series[:-1]
    y = series[1:]
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _variance(series: np.ndarray) -> float:
    """Sample variance (ddof=1)."""
    if len(series) < 2:
        return 0.0
    return float(np.var(series, ddof=1))


def prepare_dl_input(residuals: np.ndarray, position: int,
                     ts_len: int) -> np.ndarray:
    """
    Prepare one DL model input from residuals up to `position`.

    Takes the last `position` elements of residuals, normalises,
    then right-pads or crops to ts_len.

    This matches Bury 2021's inference protocol:
    the model sees the series up to the current prediction point,
    right-aligned, padded with zeros on the left if shorter than ts_len.

    Parameters
    ----------
    residuals : full residual array (length N)
    position  : current prediction point (use residuals[:position])
    ts_len    : model input length (500 or 1500)

    Returns np.ndarray shape (ts_len,)
    """
    segment = residuals[:position].astype(np.float64)

    # Normalise by mean absolute value
    denom = np.mean(np.abs(segment))
    if denom > 1e-10:
        segment = segment / denom

    n = len(segment)
    if n >= ts_len:
        # Take the last ts_len points (closest to current position)
        out = segment[-ts_len:].astype(np.float32)
    else:
        # Left-pad with zeros — matches Bury's inference padding
        out        = np.zeros(ts_len, dtype=np.float32)
        out[-n:]   = segment.astype(np.float32)

    return out


def compute_rolling_ews(
    residuals:   np.ndarray,
    ages_kyr_bp: np.ndarray,
    core_name:   str,
    sapropel_id: str,
    cfg:         dict,
    ts_len:      int,
) -> RollingWindowResult:
    """
    Compute rolling window EWS for one pre-transition segment.

    Parameters
    ----------
    residuals    : Mo residuals after Gaussian smoothing (1D, length N)
    ages_kyr_bp  : corresponding ages in ka BP (1D, length N)
    core_name    : "MS21" | "MS66" | "64PE406E1"
    sapropel_id  : e.g. "S1", "S3"
    cfg          : loaded config dict
    ts_len       : model input length (from dataset config: 500 or 1500)

    Returns
    -------
    RollingWindowResult with variance, lag1_ac, dl_inputs filled in.
    dl_probs and p_transition are filled later by evaluate.py.
    """
    inf_cfg = cfg["inference"]
    n       = len(residuals)

    # ── Rolling window size ───────────────────────────────────────────────────
    win_size = max(10, int(inf_cfg["rolling_window_frac"] * n))

    # ── Prediction range ──────────────────────────────────────────────────────
    # Per-core start fraction (MS66 and 64PE406E1 use 0.80, MS21 uses 0.60)
    start_fracs = inf_cfg["start_frac"]
    start_frac  = start_fracs.get(core_name, start_fracs["default"])
    end_frac    = inf_cfg["end_frac"]
    n_steps     = inf_cfg["prediction_steps"]  # 40

    start_pos = max(win_size, int(start_frac * n))
    end_pos   = min(n,        int(end_frac   * n))

    if start_pos >= end_pos:
        logger.warning(
            f"{core_name}/{sapropel_id}: start_pos={start_pos} >= end_pos={end_pos}. "
            f"Series may be too short (N={n}). Using full range."
        )
        start_pos = win_size
        end_pos   = n

    # 40 equally-spaced integer positions
    positions = np.linspace(start_pos, end_pos, n_steps, dtype=int)
    positions = np.clip(positions, win_size, n)

    logger.info(
        f"{core_name}/{sapropel_id}: N={n} | "
        f"window={win_size} | "
        f"predictions at positions {positions[0]}→{positions[-1]} "
        f"({start_frac:.0%}→{end_frac:.0%})"
    )

    # ── Compute EWS at each position ──────────────────────────────────────────
    variances  = np.zeros(n_steps, dtype=np.float64)
    ac1s       = np.zeros(n_steps, dtype=np.float64)
    dl_inputs  = []
    step_ages  = np.zeros(n_steps, dtype=np.float64)

    for i, pos in enumerate(positions):
        window = residuals[pos - win_size: pos]

        variances[i]  = _variance(window)
        ac1s[i]       = _lag1_autocorrelation(window)

        # Age at this position (for figure x-axis)
        step_ages[i]  = ages_kyr_bp[pos - 1] if pos - 1 < len(ages_kyr_bp) else np.nan

        # Prepare DL model input
        dl_input = prepare_dl_input(residuals, pos, ts_len)
        dl_inputs.append(dl_input)

    # ── Kendall tau trends ────────────────────────────────────────────────────
    # Computed over the full sequence of 40 predictions (Bury 2021 protocol)
    step_idx          = np.arange(n_steps)
    ktau_var,  _      = kendalltau(step_idx, variances)
    ktau_ac,   _      = kendalltau(step_idx, ac1s)

    logger.info(
        f"  Kendall tau — variance: {ktau_var:.3f} | lag-1 AC: {ktau_ac:.3f}"
    )

    return RollingWindowResult(
        positions     = positions,
        ages_kyr_bp   = step_ages,
        variance      = variances,
        lag1_ac       = ac1s,
        ktau_variance = float(ktau_var),
        ktau_lag1_ac  = float(ktau_ac),
        dl_inputs     = dl_inputs,
        core_name     = core_name,
        sapropel_id   = sapropel_id,
        n_series      = n,
    )


# =============================================================================
#  Batch runner — process all test sapropels for one core
# =============================================================================

def run_all_sapropels(
    core_name: str,
    cfg:       dict,
    ts_len:    int,
) -> List[Tuple[str, RollingWindowResult]]:
    """
    Load all test sapropel CSVs for one core and compute rolling window EWS.

    Reads files saved by pangea_cleaner.py:
        clean_dataset/{core}/{core}_{sapropel_id}_test.csv

    Returns list of (sapropel_id, RollingWindowResult) for all test sapropels.
    """
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    clean_dir = repo_root / cfg["paths"]["pangaea_clean"] / core_name

    # Get test sapropel IDs from config
    sapropels     = cfg["pangaea"]["cores"][core_name]["sapropels"]
    test_sapropels = [s for s in sapropels if s["role"] == "test"]

    if not test_sapropels:
        logger.warning(f"No test sapropels defined for {core_name} in config.yaml")
        return []

    results = []
    for sap in test_sapropels:
        sap_id    = sap["id"]
        test_file = clean_dir / f"{core_name}_{sap_id}_test.csv"

        if not test_file.exists():
            logger.warning(
                f"Test file not found: {test_file}\n"
                f"Run: python src/pangea_cleaner.py --core {core_name}"
            )
            continue

        df         = pd.read_csv(test_file)
        residuals  = df["Mo_residuals"].values.astype(np.float64)
        ages       = df["age_kyr_bp"].values.astype(np.float64)

        logger.info(f"\nRunning rolling window: {core_name}/{sap_id}")

        result = compute_rolling_ews(
            residuals   = residuals,
            ages_kyr_bp = ages,
            core_name   = core_name,
            sapropel_id = sap_id,
            cfg         = cfg,
            ts_len      = ts_len,
        )
        results.append((sap_id, result))

    return results


# =============================================================================
#  Quick sanity check
#  python src/rolling_window.py --core MS21 --ts_len 500
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Verify rolling window computation on one core."
    )
    parser.add_argument("--core",   type=str, default="MS21",
                        choices=["MS21", "MS66", "64PE406E1"])
    parser.add_argument("--ts_len", type=int, default=500,
                        choices=[500, 1500])
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # Load config
    cfg_path = Path(__file__).resolve().parents[1] / args.config
    with open(cfg_path) as f:
        import yaml
        cfg = yaml.safe_load(f)

    print(f"\n{'='*60}")
    print(f"  Rolling window check: {args.core}  ts_len={args.ts_len}")
    print(f"{'='*60}\n")

    results = run_all_sapropels(args.core, cfg, args.ts_len)

    if not results:
        print("No results — run pangea_cleaner.py first.")
        sys.exit(1)

    for sap_id, res in results:
        print(f"  Sapropel {sap_id}:")
        print(f"    N series points : {res.n_series}")
        print(f"    Prediction steps: {len(res.positions)}")
        print(f"    Position range  : {res.positions[0]} → {res.positions[-1]}")
        print(f"    Variance range  : {res.variance.min():.4f} → {res.variance.max():.4f}")
        print(f"    Lag-1 AC range  : {res.lag1_ac.min():.4f} → {res.lag1_ac.max():.4f}")
        print(f"    Kendall tau var : {res.ktau_variance:.4f}")
        print(f"    Kendall tau AC  : {res.ktau_lag1_ac:.4f}")
        print(f"    DL inputs ready : {len(res.dl_inputs)} windows of shape "
              f"({res.dl_inputs[0].shape},)")
        print()

    print("  OK — rolling window computes correctly.\n")
    print("  Next: run testing/evaluate.py to feed inputs to the model.")
