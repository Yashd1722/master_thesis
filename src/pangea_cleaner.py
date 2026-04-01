"""
src/pangea_cleaner.py
=====================
Preprocesses PANGAEA Mediterranean sediment data for both:

  1. Bury mode  — Gaussian smooth + extract residuals per sapropel
                  → used by rolling_window.py for testing only
                  → saves: clean_dataset/{core}/{core}_{sapropel}_test.csv

  2. SDML mode  — additionally generates AAFT surrogate data from
                  historical (train) sapropels
                  → used by dataset_loader.py for training
                  → saves: clean_dataset/{core}/neutral/surrogate_*.csv
                            clean_dataset/{core}/pre_transition/surrogate_*.csv

Data used (confirmed from terminal):
    calibratedXRF files — high resolution (~7000-7500 rows per core)
    MS21PC_calibratedXRF.csv      7460 rows
    MS66PC_calibratedXRF.csv      5489 rows
    64PE406-E1_calibratedXRF.csv  7672 rows
    Columns: Age [ka BP], Depth sed [m], Al, Ba, Mo, Ti, U [mg/kg]
    Signal : Mo [mg/kg]  (proxy for anoxia — Hennekam 2020)

Preprocessing follows Hennekam 2020 / Ma 2025 exactly:
    - Gaussian kernel smoothing (bandwidth = 900 years)
    - Residuals = raw Mo - smoothed Mo
    - Rolling window = 0.5 × segment length

Usage:
    python src/pangea_cleaner.py                    # all cores, AAFT surrogates
    python src/pangea_cleaner.py --core MS21        # single core
    python src/pangea_cleaner.py --no_surrogate     # clean only, no surrogates
    python src/pangea_cleaner.py --method IAAFT1    # different surrogate method
"""

import argparse
import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from typing import Tuple, List, Dict, Optional

logger = logging.getLogger(__name__)


# =============================================================================
#  Core → XRF filename mapping
#  We always use calibratedXRF (high resolution), NOT calibrationICP-MS
# =============================================================================
CORE_XRF_FILES = {
    "MS21":      "MS21PC_calibratedXRF.csv",
    "MS66":      "MS66PC_calibratedXRF.csv",
    "64PE406E1": "64PE406-E1_calibratedXRF.csv",
}


# =============================================================================
#  Config helper
# =============================================================================

def load_config(config_path: str = "config.yaml") -> dict:
    import yaml
    candidates = [
        Path(config_path),
        Path(__file__).resolve().parents[1] / "config.yaml",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"config.yaml not found.")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# =============================================================================
#  Step 1 — Load XRF data
# =============================================================================

def load_xrf(core_name: str, cfg: dict) -> pd.DataFrame:
    """
    Load the calibrated XRF CSV for one core.
    Sorts by age ascending (oldest BP value first, i.e. largest ka BP first).
    Drops rows with NaN Mo.

    Returns DataFrame with clean column names:
        age_kyr_bp, depth_m, Al, Ba, Mo, Ti, U
    """
    clean_dir = _repo_root() / cfg["paths"]["pangaea_clean"]
    filename  = CORE_XRF_FILES[core_name]
    filepath  = clean_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"XRF file not found: {filepath}\n"
            f"Available: {[f.name for f in clean_dir.glob('*.csv')]}"
        )

    df = pd.read_csv(filepath)

    # Rename to clean internal column names
    df = df.rename(columns={
        "Age [ka BP]":   "age_kyr_bp",
        "Depth sed [m]": "depth_m",
        "Al [mg/kg]":    "Al",
        "Ba [mg/kg]":    "Ba",
        "Mo [mg/kg]":    "Mo",
        "Ti [mg/kg]":    "Ti",
        "U [mg/kg]":     "U",
    })

    # Sort by age: largest ka BP = oldest = earliest in geological time
    # We sort descending so index 0 = oldest, last = most recent
    df = df.sort_values("age_kyr_bp", ascending=False).reset_index(drop=True)

    before = len(df)
    df     = df.dropna(subset=["Mo"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"{core_name}: dropped {dropped} NaN Mo rows")

    logger.info(
        f"Loaded {core_name}: {len(df)} rows | "
        f"Age [{df['age_kyr_bp'].max():.1f} → {df['age_kyr_bp'].min():.1f}] ka BP | "
        f"Mo [{df['Mo'].min():.3f} → {df['Mo'].max():.3f}] mg/kg"
    )
    return df


# =============================================================================
#  Step 2 — Gaussian smoothing (Hennekam 2020 / Ma 2025)
# =============================================================================

def gaussian_smooth(values: np.ndarray, age_kyr: np.ndarray,
                    bandwidth_yr: float) -> np.ndarray:
    """
    Gaussian kernel smoothing.
    bandwidth_yr is in YEARS (config stores 900), age_kyr is in ka BP.
    Converts bandwidth to ka for consistent units.

    For each point i, the trend is a weighted average of all points,
    with weights given by a Gaussian centred at age[i].

    Returns residuals = values - trend.
    """
    bandwidth_ka = bandwidth_yr / 1000.0  # convert years → ka

    n     = len(values)
    trend = np.zeros(n, dtype=np.float64)

    for i in range(n):
        weights = norm.pdf(age_kyr, loc=age_kyr[i], scale=bandwidth_ka)
        w_sum   = weights.sum()
        if w_sum < 1e-12:
            trend[i] = values[i]
        else:
            trend[i] = np.dot(weights / w_sum, values)

    return values - trend


def smooth_core(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply Gaussian smoothing to Mo column of a core DataFrame.
    Adds columns: Mo_trend, Mo_residuals.
    """
    bandwidth_yr = cfg["pangaea"]["bandwidth_years"]  # 900 years

    mo_values = df["Mo"].values.astype(np.float64)
    age_kyr   = df["age_kyr_bp"].values.astype(np.float64)

    residuals    = gaussian_smooth(mo_values, age_kyr, bandwidth_yr)
    trend        = mo_values - residuals

    df = df.copy()
    df["Mo_trend"]     = trend
    df["Mo_residuals"] = residuals

    logger.info(
        f"Smoothed Mo: residuals range "
        f"[{residuals.min():.4f}, {residuals.max():.4f}] mg/kg"
    )
    return df


# =============================================================================
#  Step 3 — Split by sapropel boundaries (from config.yaml)
# =============================================================================

def extract_sapropel_segment(df: pd.DataFrame, sapropel: dict,
                              segment: str) -> pd.DataFrame:
    """
    Extract a segment of the core DataFrame for one sapropel period.

    segment : "neutral"      → far from transition (early part of inter-sapropel)
              "pre_transition" → close to transition (late part before onset)
              "full"         → the entire pre-transition window (for testing)

    Age boundaries come from config.yaml pangaea.cores.{core}.sapropels.
    Note: age_kyr_bp values are negative in config (e.g. -25.0 = 25 ka BP).
    The XRF data stores positive values. We use absolute values for comparison.
    """
    # Config stores negative ages (e.g. -25.0 means 25 ka BP)
    # XRF stores positive ages (e.g. 25.0 ka BP)
    # Convert: abs(config_age) = xrf_age

    if segment == "neutral":
        start = abs(sapropel["neutral_start"])
        end   = abs(sapropel["neutral_end"])
    elif segment == "pre_transition":
        start = abs(sapropel["pretrans_start"])
        end   = abs(sapropel["pretrans_end"])
    elif segment == "full":
        # Full pre-transition window for testing
        start = abs(sapropel["neutral_start"])
        end   = abs(sapropel["pretrans_end"])
    else:
        raise ValueError(f"Unknown segment '{segment}'")

    # age_kyr_bp in df is positive, larger = older
    # segment: from 'start' ka BP down to 'end' ka BP
    # (start > end because start is older)
    mask = (df["age_kyr_bp"] <= start) & (df["age_kyr_bp"] >= end)
    seg  = df[mask].copy().reset_index(drop=True)

    logger.debug(
        f"  Segment '{segment}': {len(seg)} rows | "
        f"age [{seg['age_kyr_bp'].max():.2f} → {seg['age_kyr_bp'].min():.2f}] ka BP"
    )
    return seg


# =============================================================================
#  Step 4 — AAFT surrogate generation (Ma 2025)
# =============================================================================

def _aaft_surrogate(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Amplitude-Adjusted Fourier Transform (AAFT) surrogate.
    Preserves both the power spectrum and histogram distribution.
    Algorithm: Theiler et al. 1992, Schreiber & Schmitz 1996.

    Steps:
      1. Rank-order map series to Gaussian
      2. Randomise phases of Gaussian series
      3. Rank-order map back to original histogram
    """
    n    = len(series)
    rank = np.argsort(np.argsort(series))  # ranks of original

    # Step 1: map to Gaussian via rank ordering
    gaussian  = np.sort(rng.standard_normal(n))
    rescaled  = gaussian[rank]

    # Step 2: randomise phases
    fft       = np.fft.rfft(rescaled)
    phases    = rng.uniform(0, 2 * np.pi, len(fft))
    fft_surr  = np.abs(fft) * np.exp(1j * phases)
    surr_gauss = np.fft.irfft(fft_surr, n=n)

    # Step 3: map back to original histogram
    surr_rank  = np.argsort(np.argsort(surr_gauss))
    sorted_orig = np.sort(series)
    surrogate   = sorted_orig[surr_rank]

    return surrogate


def _ft_surrogate(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fourier Transform surrogate — randomises phases only."""
    fft      = np.fft.rfft(series)
    phases   = rng.uniform(0, 2 * np.pi, len(fft))
    fft_surr = np.abs(fft) * np.exp(1j * phases)
    return np.fft.irfft(fft_surr, n=len(series))


def _rp_surrogate(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random Permutation surrogate — shuffles values."""
    return rng.permutation(series)


def _iaaft_surrogate(series: np.ndarray, rng: np.random.Generator,
                     max_iter: int = 100,
                     preserve: str = "histogram") -> np.ndarray:
    """
    Iterative AAFT surrogate (Schreiber & Schmitz 1996).
    preserve='histogram' → IAAFT1 (exact histogram)
    preserve='spectrum'  → IAAFT2 (exact power spectrum)
    """
    n            = len(series)
    target_fft   = np.abs(np.fft.rfft(series))
    sorted_orig  = np.sort(series)

    # Initialise with random permutation
    surr = rng.permutation(series)

    for _ in range(max_iter):
        # Match power spectrum
        fft      = np.fft.rfft(surr)
        phases   = np.angle(fft)
        fft_new  = target_fft * np.exp(1j * phases)
        surr     = np.fft.irfft(fft_new, n=n)

        if preserve == "histogram":
            # Match histogram
            rank = np.argsort(np.argsort(surr))
            surr = sorted_orig[rank]

    return surr


SURROGATE_METHODS = {
    "RP":     _rp_surrogate,
    "FT":     _ft_surrogate,
    "AAFT":   _aaft_surrogate,
    "IAAFT1": lambda s, r: _iaaft_surrogate(s, r, preserve="histogram"),
    "IAAFT2": lambda s, r: _iaaft_surrogate(s, r, preserve="spectrum"),
}


def generate_surrogates(residuals: np.ndarray, n_surrogates: int,
                        method: str, seed: int = 42) -> List[np.ndarray]:
    """
    Generate n_surrogates surrogate time series from residuals.

    Parameters
    ----------
    residuals    : 1D numpy array of Mo residuals (already smoothed)
    n_surrogates : number of surrogates to generate
    method       : "RP" | "FT" | "AAFT" | "IAAFT1" | "IAAFT2"
    seed         : reproducibility seed

    Returns list of 1D numpy arrays, each same length as residuals.
    """
    if method not in SURROGATE_METHODS:
        raise ValueError(
            f"Unknown surrogate method '{method}'. "
            f"Available: {list(SURROGATE_METHODS.keys())}"
        )

    surr_fn   = SURROGATE_METHODS[method]
    rng       = np.random.default_rng(seed)
    surrogates = []

    for i in range(n_surrogates):
        s = surr_fn(residuals.copy(), rng)
        surrogates.append(s.astype(np.float32))

    return surrogates


# =============================================================================
#  Step 5 — Save outputs
# =============================================================================

def save_test_segment(seg_df: pd.DataFrame, core_name: str,
                      sapropel_id: str, cfg: dict) -> Path:
    """
    Save a test sapropel segment for use by rolling_window.py / evaluate.py.
    Output: clean_dataset/{core}/{core}_{sapropel_id}_test.csv
    Columns: age_kyr_bp, Mo, Mo_trend, Mo_residuals
    """
    out_dir = (
        _repo_root() / cfg["paths"]["pangaea_clean"] / core_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{core_name}_{sapropel_id}_test.csv"
    cols     = ["age_kyr_bp", "Mo", "Mo_trend", "Mo_residuals"]
    seg_df[cols].to_csv(out_path, index=False)
    logger.info(f"  Saved test segment → {out_path.name}  ({len(seg_df)} rows)")
    return out_path


def save_surrogates(surrogates: List[np.ndarray], core_name: str,
                    segment_type: str, sapropel_id: str,
                    cfg: dict, start_idx: int = 0) -> int:
    """
    Save surrogate time series as individual CSVs.

    Output dir:
        clean_dataset/{core}/neutral/           for segment_type="neutral"
        clean_dataset/{core}/pre_transition/    for segment_type="pre_transition"

    File naming: surrogate_{idx:05d}.csv  (single column, no header)

    Returns number of files saved.
    """
    out_dir = (
        _repo_root()
        / cfg["paths"]["pangaea_clean"]
        / core_name
        / segment_type
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, surr in enumerate(surrogates):
        idx      = start_idx + i
        out_path = out_dir / f"surrogate_{idx:05d}.csv"
        pd.DataFrame(surr).to_csv(out_path, index=False, header=False)

    logger.info(
        f"  Saved {len(surrogates)} '{segment_type}' surrogates "
        f"from {core_name}/{sapropel_id} → {out_dir.name}/"
    )
    return len(surrogates)


# =============================================================================
#  Main pipeline — per core
# =============================================================================

def process_core(core_name: str, cfg: dict,
                 generate_surr: bool = True,
                 method: str = "AAFT") -> None:
    """
    Full processing pipeline for one core.

    For each sapropel in config:
      role=train → extract neutral + pre_transition segments
                   → generate AAFT surrogates (SDML training data)
      role=test  → extract full pre-transition window
                   → save clean CSV (for rolling_window / evaluate)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing core: {core_name}")
    logger.info(f"{'='*60}")

    # ── Load + smooth ─────────────────────────────────────────────────────────
    df = load_xrf(core_name, cfg)
    df = smooth_core(df, cfg)

    # ── Get sapropel config for this core ─────────────────────────────────────
    core_key = core_name  # "MS21", "MS66", "64PE406E1"
    sapropels = cfg["pangaea"]["cores"][core_key]["sapropels"]

    n_surr   = cfg["surrogate"]["n_surrogates"]
    surr_seed = cfg["project"]["seed"]

    # Counters so surrogate filenames don't collide across sapropels
    neutral_count    = 0
    pretrans_count   = 0

    for sap in sapropels:
        sap_id = sap["id"]
        role   = sap["role"]  # "train" or "test"

        logger.info(f"\n  Sapropel {sap_id} ({role})")

        if role == "test":
            # ── Test sapropel: save clean segment for evaluate.py ─────────────
            seg = extract_sapropel_segment(df, sap, segment="full")
            if len(seg) < 10:
                logger.warning(
                    f"  {sap_id}: only {len(seg)} rows in test segment — skipping"
                )
                continue
            save_test_segment(seg, core_name, sap_id, cfg)

        elif role == "train" and generate_surr:
            # ── Train sapropel: generate surrogates for SDML ──────────────────

            # Neutral segment (far from transition — first half of inter-sapropel)
            neutral_seg = extract_sapropel_segment(df, sap, segment="neutral")
            if len(neutral_seg) >= 10:
                neutral_resids = neutral_seg["Mo_residuals"].values
                surrs = generate_surrogates(
                    neutral_resids, n_surr, method,
                    seed=surr_seed + neutral_count
                )
                neutral_count += save_surrogates(
                    surrs, core_name, "neutral", sap_id, cfg,
                    start_idx=neutral_count
                )
            else:
                logger.warning(
                    f"  {sap_id} neutral: only {len(neutral_seg)} rows — skipping"
                )

            # Pre-transition segment (close to transition — second half)
            pretrans_seg = extract_sapropel_segment(df, sap, segment="pre_transition")
            if len(pretrans_seg) >= 10:
                pretrans_resids = pretrans_seg["Mo_residuals"].values
                surrs = generate_surrogates(
                    pretrans_resids, n_surr, method,
                    seed=surr_seed + pretrans_count + 100000
                )
                pretrans_count += save_surrogates(
                    surrs, core_name, "pre_transition", sap_id, cfg,
                    start_idx=pretrans_count
                )
            else:
                logger.warning(
                    f"  {sap_id} pre_transition: only {len(pretrans_seg)} rows — skipping"
                )

        elif role == "train" and not generate_surr:
            logger.info(f"  {sap_id}: skipping surrogate generation (--no_surrogate)")

    logger.info(f"\n  {core_name} done:")
    logger.info(f"    neutral surrogates    : {neutral_count}")
    logger.info(f"    pre_transition surr.  : {pretrans_count}")


# =============================================================================
#  Entry point
# =============================================================================

def main():
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(levelname)s | %(message)s",
        handlers= [logging.StreamHandler(sys.stdout)]
    )

    parser = argparse.ArgumentParser(
        description="Preprocess PANGAEA data and generate SDML surrogates."
    )
    parser.add_argument(
        "--core",
        type    = str,
        default = "all",
        choices = ["all", "MS21", "MS66", "64PE406E1"],
        help    = "Which core to process (default: all)"
    )
    parser.add_argument(
        "--method",
        type    = str,
        default = "AAFT",
        choices = list(SURROGATE_METHODS.keys()),
        help    = "Surrogate generation method (default: AAFT)"
    )
    parser.add_argument(
        "--no_surrogate",
        action  = "store_true",
        help    = "Only clean and save test segments, skip surrogate generation"
    )
    parser.add_argument(
        "--config",
        type    = str,
        default = "config.yaml",
        help    = "Path to config.yaml"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    cores = (
        list(CORE_XRF_FILES.keys())
        if args.core == "all"
        else [args.core]
    )

    for core in cores:
        process_core(
            core_name    = core,
            cfg          = cfg,
            generate_surr= not args.no_surrogate,
            method       = args.method,
        )

    logger.info("\nAll done.")
    logger.info(
        "Next step: python src/dataset_loader.py --dataset sdml_MS21 "
        "to verify surrogate files loaded correctly."
    )


if __name__ == "__main__":
    main()
