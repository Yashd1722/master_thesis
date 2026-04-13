"""
src/pangea_cleaner.py
=====================
Preprocesses PANGAEA Mediterranean sediment data.

Outputs per sapropel per core:
  For TRAINING (SDML):
    clean_dataset/{core}/neutral/surrogate_*.csv
    clean_dataset/{core}/pre_transition/surrogate_*.csv

  For TESTING (rolling window evaluation):
    clean_dataset/{core}/{core}_{sap}_forced.csv
        columns: age_kyr_bp, Al, Ba, Mo, Ti, U,
                 Al_trend, Ba_trend, Mo_trend, Ti_trend, U_trend,
                 Al_residuals, Ba_residuals, Mo_residuals, Ti_residuals, U_residuals
    clean_dataset/{core}/{core}_{sap}_neutral_test.csv
        same columns — neutral segment for ROC null class

Data files used (calibratedXRF — high resolution):
  MS21PC_calibratedXRF.csv       7460 rows
  MS66PC_calibratedXRF.csv       5489 rows
  64PE406-E1_calibratedXRF.csv   7672 rows
  All columns: Age [ka BP], Depth sed [m], Al, Ba, Mo, Ti, U [mg/kg]

Usage:
  python src/pangea_cleaner.py                  # all cores
  python src/pangea_cleaner.py --core MS21      # one core
  python src/pangea_cleaner.py --no_surrogate   # test segments only
"""

import argparse
import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from typing import List, Tuple, Optional
import yaml

logger = logging.getLogger(__name__)

ELEMENTS = ["Al", "Ba", "Mo", "Ti", "U"]

# XRF files — HIGH RESOLUTION, these are what we analyse
CORE_XRF_FILES = {
    "MS21":      "MS21PC_calibratedXRF.csv",
    "MS66":      "MS66PC_calibratedXRF.csv",
    "64PE406E1": "64PE406-E1_calibratedXRF.csv",
}


# =============================================================================
#  Config
# =============================================================================

def load_config(config_path: str = "config.yaml") -> dict:
    candidates = [
        Path(config_path),
        Path(__file__).resolve().parents[1] / "config.yaml",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("config.yaml not found")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# =============================================================================
#  Step 1 — Load XRF data
# =============================================================================

def load_xrf(core_name: str, cfg: dict) -> pd.DataFrame:
    """
    Load calibratedXRF CSV.
    Returns DataFrame sorted oldest→youngest (largest ka BP first).
    All 5 element columns guaranteed to exist.
    """
    clean_dir = _repo_root() / cfg["paths"]["pangaea_clean"]
    filepath  = clean_dir / CORE_XRF_FILES[core_name]

    if not filepath.exists():
        raise FileNotFoundError(
            f"XRF file not found: {filepath}\n"
            f"Available: {[f.name for f in clean_dir.glob('*.csv')]}"
        )

    df = pd.read_csv(filepath)

    # Rename to clean names
    df = df.rename(columns={
        "Age [ka BP]":   "age_kyr_bp",
        "Depth sed [m]": "depth_m",
        "Al [mg/kg]":    "Al",
        "Ba [mg/kg]":    "Ba",
        "Mo [mg/kg]":    "Mo",
        "Ti [mg/kg]":    "Ti",
        "U [mg/kg]":     "U",
    })

    # Drop any extra columns (e.g. MS66 has "Method comm")
    keep = ["age_kyr_bp", "depth_m"] + ELEMENTS
    df   = df[[c for c in keep if c in df.columns]].copy()

    # Sort: largest ka BP first = oldest first
    df = df.sort_values("age_kyr_bp", ascending=False).reset_index(drop=True)

    # Log stats
    logger.info(
        f"{core_name}: {len(df)} rows | "
        f"age [{df['age_kyr_bp'].max():.1f} → {df['age_kyr_bp'].min():.1f}] ka BP"
    )
    return df


# =============================================================================
#  Step 2 — Gaussian smoothing for all elements
# =============================================================================

def smooth_all_elements(df: pd.DataFrame, bandwidth_yr: float) -> pd.DataFrame:
    """
    Apply Gaussian kernel smooth to all 5 elements.
    Adds columns: {elem}_trend and {elem}_residuals for each element.
    bandwidth_yr is in years (config: 900).
    """
    bandwidth_ka = bandwidth_yr / 1000.0
    age_kyr      = df["age_kyr_bp"].values.astype(np.float64)
    df           = df.copy()

    for elem in ELEMENTS:
        if elem not in df.columns:
            logger.warning(f"Element {elem} not in DataFrame — skipping")
            df[f"{elem}_trend"]     = np.nan
            df[f"{elem}_residuals"] = np.nan
            continue

        values = df[elem].values.astype(np.float64)
        trend  = np.zeros(len(values))

        for i in range(len(values)):
            w = norm.pdf(age_kyr, loc=age_kyr[i], scale=bandwidth_ka)
            s = w.sum()
            trend[i] = np.dot(w / s, values) if s > 1e-12 else values[i]

        df[f"{elem}_trend"]     = trend
        df[f"{elem}_residuals"] = values - trend

    logger.info(f"  Smoothed {len(ELEMENTS)} elements (bandwidth={bandwidth_yr} yr)")
    return df


# =============================================================================
#  Step 3 — Extract segments by age boundary
# =============================================================================

def extract_segment(df: pd.DataFrame, start_ka: float, end_ka: float) -> pd.DataFrame:
    """
    Extract rows where start_ka >= age_kyr_bp >= end_ka.
    start_ka > end_ka (start is older).
    Config stores negative ages (e.g. -25.0) → take abs().
    """
    start = abs(start_ka)
    end   = abs(end_ka)
    mask  = (df["age_kyr_bp"] <= start) & (df["age_kyr_bp"] >= end)
    return df[mask].copy().reset_index(drop=True)


# =============================================================================
#  Step 4 — AAFT surrogate generation (for SDML training)
# =============================================================================

def _aaft_surrogate(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """AAFT surrogate — preserves histogram and power spectrum."""
    n        = len(series)
    rank     = np.argsort(np.argsort(series))
    gaussian = np.sort(rng.standard_normal(n))
    rescaled = gaussian[rank]

    fft      = np.fft.rfft(rescaled)
    phases   = rng.uniform(0, 2 * np.pi, len(fft))
    fft_surr = np.abs(fft) * np.exp(1j * phases)
    surr_g   = np.fft.irfft(fft_surr, n=n)

    surr_rank   = np.argsort(np.argsort(surr_g))
    sorted_orig = np.sort(series)
    return sorted_orig[surr_rank].astype(np.float32)


def _rp_surrogate(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(series).astype(np.float32)


def _ft_surrogate(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    fft      = np.fft.rfft(series)
    phases   = rng.uniform(0, 2 * np.pi, len(fft))
    fft_surr = np.abs(fft) * np.exp(1j * phases)
    return np.fft.irfft(fft_surr, n=len(series)).astype(np.float32)


def _iaaft_surrogate(series: np.ndarray, rng: np.random.Generator,
                     max_iter: int = 100,
                     preserve: str = "histogram") -> np.ndarray:
    n           = len(series)
    target_fft  = np.abs(np.fft.rfft(series))
    sorted_orig = np.sort(series)
    surr        = rng.permutation(series)
    for _ in range(max_iter):
        fft     = np.fft.rfft(surr)
        phases  = np.angle(fft)
        fft_new = target_fft * np.exp(1j * phases)
        surr    = np.fft.irfft(fft_new, n=n)
        if preserve == "histogram":
            rank = np.argsort(np.argsort(surr))
            surr = sorted_orig[rank]
    return surr.astype(np.float32)


SURROGATE_METHODS = {
    "RP":     _rp_surrogate,
    "FT":     _ft_surrogate,
    "AAFT":   _aaft_surrogate,
    "IAAFT1": lambda s, r: _iaaft_surrogate(s, r, preserve="histogram"),
    "IAAFT2": lambda s, r: _iaaft_surrogate(s, r, preserve="spectrum"),
}

# =============================================================================
#  AR(1) null series generation — Bury 2021 protocol
# =============================================================================

def fit_ar1(series: np.ndarray) -> Tuple[float, float]:
    """
    Fit AR(1) model to a series. Returns (phi, sigma) where:
      phi   = lag-1 autocorrelation coefficient
      sigma = noise standard deviation
    """
    if len(series) < 3:
        return 0.0, float(np.std(series))
    phi   = float(np.corrcoef(series[:-1], series[1:])[0, 1])
    resid = series[1:] - phi * series[:-1]
    sigma = float(np.std(resid))
    return phi, sigma


def generate_ar1_null_series(residuals: np.ndarray,
                              n_null: int,
                              seed: int = 42) -> List[np.ndarray]:
    """
    Generate AR(1) null series matching Bury 2021 protocol exactly:
      "Null time series of the same length are generated from an
       AR(1) process fit to the initial 20% of the data."

    The AR(1) fit preserves the autocorrelation structure of the
    real geological data, so the model cannot trivially distinguish
    null from forced by autocorrelation alone. This is the correct
    null hypothesis for ROC construction.

    Parameters
    ----------
    residuals : pre-transition residuals (full length)
    n_null    : number of null series to generate (Bury uses 10 per series)
    seed      : reproducibility seed

    Returns list of null series, each same length as residuals.
    """
    n     = len(residuals)
    # Fit AR(1) to INITIAL 20% only (Bury 2021 protocol)
    init  = residuals[:max(3, int(0.20 * n))]
    phi, sigma = fit_ar1(init)

    rng  = np.random.default_rng(seed)
    nulls = []

    for _ in range(n_null):
        null    = np.zeros(n, dtype=np.float64)
        null[0] = rng.normal(0, sigma)
        for t in range(1, n):
            null[t] = phi * null[t-1] + rng.normal(0, sigma)
        nulls.append(null.astype(np.float32))

    return nulls


def save_null_series(null_series: List[np.ndarray],
                     ages: np.ndarray,
                     core_name: str,
                     sapropel_id: str,
                     element: str,
                     cfg: dict) -> Path:
    """
    Save AR(1) null series as a single CSV for use by rolling_window.py.
    Each column = one null series.
    Output: {core}/{core}_{sap}_{element}_ar1_null.csv
    """
    out_dir = _repo_root() / cfg["paths"]["pangaea_clean"] / core_name
    out_dir.mkdir(parents=True, exist_ok=True)

    fname   = f"{core_name}_{sapropel_id}_{element}_ar1_null.csv"
    df_null = pd.DataFrame(
        np.stack(null_series, axis=1),
        columns=[f"null_{i:03d}" for i in range(len(null_series))]
    )
    df_null.insert(0, "age_kyr_bp", ages[:len(null_series[0])])
    df_null.to_csv(out_dir / fname, index=False)
    logger.info(f"    Saved {fname}  ({len(null_series)} null series × {len(null_series[0])} steps)")
    return out_dir / fname


def generate_surrogates(residuals: np.ndarray, n_surr: int,
                        method: str, seed: int) -> List[np.ndarray]:
    fn  = SURROGATE_METHODS[method]
    rng = np.random.default_rng(seed)
    return [fn(residuals.copy(), rng) for _ in range(n_surr)]


# =============================================================================
#  Step 5 — Save outputs
# =============================================================================

def save_test_segment(seg_df: pd.DataFrame, core_name: str,
                      sapropel_id: str, segment_type: str,
                      cfg: dict) -> Path:
    """
    Save a test segment (forced or neutral) for rolling_window.py.

    segment_type: "forced" or "neutral_test"
    Columns saved: age_kyr_bp, Al, Ba, Mo, Ti, U,
                   Al_trend...U_trend, Al_residuals...U_residuals
    """
    out_dir = _repo_root() / cfg["paths"]["pangaea_clean"] / core_name
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{core_name}_{sapropel_id}_{segment_type}.csv"
    cols  = ["age_kyr_bp"] + ELEMENTS
    for elem in ELEMENTS:
        if f"{elem}_trend" in seg_df.columns:
            cols += [f"{elem}_trend", f"{elem}_residuals"]

    cols = [c for c in cols if c in seg_df.columns]
    seg_df[cols].to_csv(out_dir / fname, index=False)
    logger.info(f"    Saved {fname}  ({len(seg_df)} rows)")
    return out_dir / fname


def save_surrogates(surrogates: List[np.ndarray], core_name: str,
                    segment_type: str, cfg: dict,
                    start_idx: int = 0) -> int:
    """
    Save AAFT surrogates to clean_dataset/{core}/{segment_type}/
    """
    out_dir = (
        _repo_root()
        / cfg["paths"]["pangaea_clean"]
        / core_name
        / segment_type
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, surr in enumerate(surrogates):
        pd.DataFrame(surr).to_csv(
            out_dir / f"surrogate_{start_idx + i:05d}.csv",
            index=False, header=False
        )
    return len(surrogates)


# =============================================================================
#  Main pipeline
# =============================================================================

def process_core(core_name: str, cfg: dict,
                 generate_surr: bool = True,
                 method: str = "AAFT") -> None:
    """Full pipeline for one core."""
    logger.info(f"\n{'='*60}\nProcessing: {core_name}\n{'='*60}")

    bandwidth_yr = cfg["pangaea"]["bandwidth_years"]
    n_surr       = cfg["surrogate"]["n_surrogates"]
    seed         = cfg["project"]["seed"]
    sapropels    = cfg["pangaea"]["cores"][core_name]["sapropels"]

    # ── Load + smooth ─────────────────────────────────────────────────────────
    df = load_xrf(core_name, cfg)
    df = smooth_all_elements(df, bandwidth_yr)

    neutral_count  = 0
    pretrans_count = 0

    for sap in sapropels:
        sap_id = sap["id"]
        role   = sap["role"]
        logger.info(f"\n  Sapropel {sap_id}  role={role}")

        if role == "test":
            # ── Save FORCED segment (pre-transition) ──────────────────────────
            forced_seg = extract_segment(
                df,
                start_ka = sap["pretrans_start"],
                end_ka   = sap["pretrans_end"],
            )
            if len(forced_seg) >= 10:
                save_test_segment(forced_seg, core_name, sap_id,
                                  "forced", cfg)
            else:
                logger.warning(f"  {sap_id} forced: only {len(forced_seg)} rows")
                forced_seg = None

            # ── Generate AR(1) null series (Bury 2021 protocol) ───────────────
            # "Null time series generated from AR(1) process fit to
            #  the initial 20% of the data" — Bury 2021 Methods
            # This is the CORRECT null for ROC — NOT inter-sapropel periods.
            # Inter-sapropel data has geological structure that the model
            # classifies as bifurcation-type, giving AUC ≈ 0.50.
            # AR(1) null preserves autocorrelation but has no CSD trend.
            if forced_seg is not None and len(forced_seg) >= 10:
                ages_forced = forced_seg["age_kyr_bp"].values
                for elem in ELEMENTS:
                    resid_col = f"{elem}_residuals"
                    if resid_col not in forced_seg.columns:
                        continue
                    resids = forced_seg[resid_col].dropna().values
                    if len(resids) < 10:
                        continue
                    # Generate 10 AR(1) null series (Bury uses 10 per event)
                    null_series = generate_ar1_null_series(
                        resids, n_null=10,
                        seed=seed + hash(f"{core_name}{sap_id}{elem}") % 100000
                    )
                    save_null_series(
                        null_series, ages_forced,
                        core_name, sap_id, elem, cfg
                    )

        elif role == "train" and generate_surr:
            # ── Generate AAFT surrogates from Mo residuals ────────────────────
            neutral_seg  = extract_segment(df, sap["neutral_start"],
                                           sap["neutral_end"])
            pretrans_seg = extract_segment(df, sap["pretrans_start"],
                                           sap["pretrans_end"])

            for seg, seg_type, counter in [
                (neutral_seg,  "neutral",        neutral_count),
                (pretrans_seg, "pre_transition", pretrans_count),
            ]:
                if "Mo_residuals" not in seg.columns or len(seg) < 10:
                    continue
                resids = seg["Mo_residuals"].dropna().values
                surrs  = generate_surrogates(resids, n_surr, method,
                                             seed=seed + counter)
                n_saved = save_surrogates(surrs, core_name, seg_type,
                                          cfg, start_idx=counter)
                if seg_type == "neutral":
                    neutral_count += n_saved
                else:
                    pretrans_count += n_saved

    logger.info(f"\n  {core_name} done:")
    logger.info(f"    neutral surrogates   : {neutral_count}")
    logger.info(f"    pre_transition surr. : {pretrans_count}")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    parser = argparse.ArgumentParser(
        description="Preprocess PANGAEA XRF data and generate SDML surrogates."
    )
    parser.add_argument("--core", default="all",
                        choices=["all", "MS21", "MS66", "64PE406E1"])
    parser.add_argument("--method", default="AAFT",
                        choices=list(SURROGATE_METHODS.keys()))
    parser.add_argument("--no_surrogate", action="store_true")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg   = load_config(args.config)
    cores = list(CORE_XRF_FILES.keys()) if args.core == "all" else [args.core]

    for core in cores:
        process_core(core, cfg,
                     generate_surr = not args.no_surrogate,
                     method        = args.method)

    logger.info("\nAll done.")
    logger.info("Next: python src/rolling_window.py --core MS21")


if __name__ == "__main__":
    main()
