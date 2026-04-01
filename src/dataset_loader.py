"""
src/dataset_loader.py
=====================
Universal DataLoader for ALL datasets in this thesis.

Actual data structure (confirmed from terminal):
------------------------------------------------
ts_500/combined/
    labels.csv              columns: sequence_ID, class_label (0=fold,1=hopf,2=transcritical,3=null)
    groups.csv              columns: sequence_ID, dataset_ID  (not used for training)
    output_resids/
        resids100000.csv    columns: Time, Residuals  (501 lines = 1 header + 500 rows)
        resids100001.csv
        ...

ts_1500/combined/           same structure, 1500 data rows per file

pangaea_923197/
    datasets/clean_dataset/
        MS21PC_calibrationICP-MS.csv       columns: Age [ka BP], Depth sed [m], Al, Ba, Mo, Ti, U [mg/kg]
        MS66PC_calibrationICP-OES_MS.csv
        64PE406-E1_calibrationICP-MS.csv
        (XRF files exist but we use ICP-MS — calibrated Mo values)

PANGAEA is used for TESTING ONLY in Bury mode.
In SDML mode, pangea_cleaner.py generates surrogate CSVs from
historical sapropels, which are then loaded here for training.

Public API — the ONLY functions any other script should call:
    get_dataloader(dataset_name, split, cfg, pad_variant, ...)  -> DataLoader
    get_dataset_info(dataset_name, cfg)                         -> dict
    load_pangaea_core(core_name, cfg)                           -> (time, mo, raw_df)

Every DataLoader yields:
    x : FloatTensor  (B, 1, T)   normalised residuals, channel-first for Conv1D
    y : LongTensor   (B,)         class index
"""

import logging
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
#  PANGAEA core → ICP-MS filename mapping
#  We always use the calibrated ICP-MS/ICP-OES files, not the raw XRF scans.
# =============================================================================
PANGAEA_CORE_FILES = {
    "MS21":      "MS21PC_calibrationICP-MS.csv",
    "MS66":      "MS66PC_calibrationICP-OES_MS.csv",
    "64PE406E1": "64PE406-E1_calibrationICP-MS.csv",
}

# class_label integer → bifurcation name (Bury 2021 ordering from labels.csv)
LABEL_MAP = {0: "fold", 1: "hopf", 2: "transcritical", 3: "null"}


# =============================================================================
#  Config helpers
# =============================================================================

def load_config(config_path: str = "config.yaml") -> dict:
    """Load config.yaml — searches repo root automatically."""
    candidates = [
        Path(config_path),
        Path(__file__).resolve().parents[1] / "config.yaml",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(
        f"config.yaml not found. Tried: {[str(c) for c in candidates]}"
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(cfg: dict, key: str) -> Path:
    return _repo_root() / cfg["paths"][key]


# =============================================================================
#  Preprocessing
#  Residuals from Zenodo are already Lowess-detrended by Bury.
#  We ONLY normalise — never detrend again.
# =============================================================================

def normalise(residuals: np.ndarray) -> np.ndarray:
    """
    Divide by mean absolute value (Bury 2021 normalisation).
    Makes all series scale-invariant before feeding to the model.
    Returns residuals unchanged if near-zero (all-zero padding regions).
    """
    denom = np.mean(np.abs(residuals))
    if denom < 1e-10:
        return residuals.copy().astype(np.float32)
    return (residuals / denom).astype(np.float32)


# ── Padding variants (Bury 2021 Section: DL Algorithm Architecture) ───────────

def pad_both_sides(series: np.ndarray, ts_len: int,
                   rng: np.random.Generator) -> np.ndarray:
    """
    Variant 1 — zeros on BOTH sides.
    Teaches the model to recognise bifurcation features anywhere in the window.
    The split between left and right padding is uniformly random.
    """
    n = len(series)
    if n >= ts_len:
        return series[-ts_len:].astype(np.float32)
    pad_total = ts_len - n
    left      = int(rng.integers(0, pad_total + 1))
    right     = pad_total - left
    return np.concatenate([
        np.zeros(left,  dtype=np.float32),
        series.astype(np.float32),
        np.zeros(right, dtype=np.float32),
    ])


def pad_left_only(series: np.ndarray, ts_len: int,
                  rng: np.random.Generator) -> np.ndarray:
    """
    Variant 2 — zeros on LEFT side only.
    Series is always right-aligned — the model always sees the series
    end at the right edge, matching the real inference scenario.
    """
    n = len(series)
    if n >= ts_len:
        return series[-ts_len:].astype(np.float32)
    left = ts_len - n
    return np.concatenate([
        np.zeros(left, dtype=np.float32),
        series.astype(np.float32),
    ])


# =============================================================================
#  File index builders
# =============================================================================

def _build_zenodo_index(base_dir: Path) -> pd.DataFrame:
    """
    Build a DataFrame mapping each residual file to its class label.

    Reads labels.csv (sequence_ID, class_label) and matches to
    output_resids/resids*.csv files by sorting both by their numeric ID.

    Returns DataFrame with columns: filepath, class_label
    """
    labels_path = base_dir / "labels.csv"
    resids_dir  = base_dir / "output_resids"

    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found at {labels_path}")
    if not resids_dir.exists():
        raise FileNotFoundError(f"output_resids/ not found at {resids_dir}")

    # Load labels — sort by sequence_ID ascending
    labels_df = pd.read_csv(labels_path)
    labels_df = labels_df.sort_values("sequence_ID").reset_index(drop=True)

    # Discover all residual files — sort by numeric suffix ascending
    resid_files = sorted(
        resids_dir.glob("resids*.csv"),
        key=lambda p: int(p.stem.replace("resids", ""))
    )

    if len(resid_files) == 0:
        raise FileNotFoundError(
            f"No resids*.csv files found in {resids_dir}.\n"
            f"Make sure output_resids.zip is unzipped."
        )

    if len(resid_files) != len(labels_df):
        logger.warning(
            f"File count mismatch: {len(resid_files)} residual files "
            f"but {len(labels_df)} label rows. "
            f"Using min({len(resid_files)}, {len(labels_df)})."
        )
        n = min(len(resid_files), len(labels_df))
        resid_files = resid_files[:n]
        labels_df   = labels_df.iloc[:n]

    # Positional match: both are sorted by their numeric ID ascending
    # cache_pos = position in the numpy cache array (0..N-1)
    # This column MUST be preserved through splitting so __getitem__ reads
    # the correct row from cache_residuals.npy
    n_matched = min(len(resid_files), len(labels_df))
    index_df = pd.DataFrame({
        "filepath":    [str(f) for f in resid_files[:n_matched]],
        "class_label": labels_df["class_label"].values[:n_matched],
        "cache_pos":   np.arange(n_matched, dtype=np.int64),
    })

    logger.info(
        f"Zenodo index built: {len(index_df)} series | "
        f"class distribution: { dict(index_df['class_label'].value_counts().sort_index()) }"
    )
    return index_df


def _build_sdml_index(clean_dir: Path, core: str) -> pd.DataFrame:
    """
    Build file index for SDML surrogate data generated by pangea_cleaner.py.

    Expected layout (created by pangea_cleaner.py):
        pangaea_clean/{core}/neutral/surrogate_*.csv        label=0
        pangaea_clean/{core}/pre_transition/surrogate_*.csv label=1

    Returns DataFrame with columns: filepath, class_label
    """
    core_dir = clean_dir / core
    if not core_dir.exists():
        raise FileNotFoundError(
            f"SDML surrogate directory not found: {core_dir}\n"
            f"Run: python src/pangea_cleaner.py --core {core}"
        )

    rows = []
    for cls_name, label in [("neutral", 0), ("pre_transition", 1)]:
        cls_dir = core_dir / cls_name
        if not cls_dir.exists():
            logger.warning(f"SDML class dir not found: {cls_dir}")
            continue
        files = sorted(cls_dir.glob("surrogate_*.csv"))
        if not files:
            logger.warning(f"No surrogate files in {cls_dir}")
            continue
        for f in files:
            rows.append({"filepath": str(f), "class_label": label})
        logger.info(f"  SDML {core}/{cls_name}: {len(files)} surrogates")

    if not rows:
        raise RuntimeError(
            f"No SDML surrogate files found for core {core} at {clean_dir}.\n"
            f"Run: python src/pangea_cleaner.py --core {core}"
        )

    return pd.DataFrame(rows)


# =============================================================================
#  Train / Val / Test split
# =============================================================================

def _split_index(index_df: pd.DataFrame,
                 train_frac: float, val_frac: float,
                 seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Deterministic stratified split.
    Stratified by class_label so every split has the same class distribution.

    Returns dict with keys: "train", "val", "test"
    """
    rng    = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}

    for label in sorted(index_df["class_label"].unique()):
        cls_df = index_df[index_df["class_label"] == label].copy()
        cls_df = cls_df.sample(frac=1, random_state=int(rng.integers(0, 99999)))
        n      = len(cls_df)
        n_tr   = int(n * train_frac)
        n_va   = int(n * val_frac)

        splits["train"].append(cls_df.iloc[:n_tr])
        splits["val"].append(  cls_df.iloc[n_tr: n_tr + n_va])
        splits["test"].append( cls_df.iloc[n_tr + n_va:])

    return {k: pd.concat(v).reset_index(drop=True) for k, v in splits.items()}


# =============================================================================
#  PyTorch Dataset classes
# =============================================================================

class ZenodoEWSDataset(Dataset):
    """
    Dataset for Zenodo synthetic data (ts_500 / ts_1500).

    FAST PATH: loads from numpy cache built by src/build_cache.py.
    Entire dataset lives in RAM. __getitem__ is a pure array slice.
    Expected: ~5-15 sec/epoch instead of 2.6 min/epoch.

    FALLBACK: if cache not found, loads CSVs one-by-one (slow).
    Run: python src/build_cache.py --dataset ts_500

    Parameters
    ----------
    index_df     : DataFrame with columns filepath, class_label
    base_dir     : path to combined/ (contains cache files)
    ts_len       : time series length (500 or 1500)
    pad_variant  : 1 = both sides random, 2 = left only
    seed         : base seed for padding randomisation
    """

    def __init__(self, index_df: pd.DataFrame, base_dir: Path,
                 ts_len: int, pad_variant: int = 1, seed: int = 42):
        self.records     = index_df.reset_index(drop=True)
        self.ts_len      = ts_len
        self.pad_variant = pad_variant
        self.base_seed   = seed

        cache_X = base_dir / "cache_residuals.npy"
        cache_y = base_dir / "cache_labels.npy"

        if cache_X.exists() and cache_y.exists():
            logger.info(f"  Loading cache from {cache_X.name} ...")
            # mmap_mode='r': all DataLoader workers share the same
            # memory-mapped file — no per-worker copy, no extra RAM
            self._X         = np.load(cache_X, mmap_mode='r')
            self._y         = np.load(cache_y, mmap_mode='r')
            self._use_cache = True
            # cache_pos column gives the correct row in _X/_y for this split
            self._cache_pos = self.records["cache_pos"].values.astype(np.int64)
            # Pre-compute ALL padding offsets at init time — eliminates
            # np.random.default_rng() overhead inside __getitem__
            rng = np.random.default_rng(seed)
            n   = len(self.records)
            self._pad_offsets = rng.integers(0, ts_len, size=n).astype(np.int32)
            logger.info(f"  Cache loaded: {self._X.shape}  split={n}")
        else:
            logger.warning(
                f"Cache not found at {cache_X}. Using slow per-file loading. "
                f"Run: python src/build_cache.py --dataset "
                f"ts_500 or ts_1500 first."
            )
            self._use_cache  = False
            self._pad_offsets = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._use_cache:
            # Fast RAM lookup using correct cache position
            pos       = int(self._cache_pos[idx])
            residuals = np.array(self._X[pos], dtype=np.float32)
            label     = int(self._y[pos])
        else:
            row = self.records.iloc[idx]
            try:
                df        = pd.read_csv(row["filepath"])
                residuals = df["Residuals"].values.astype(np.float32)
            except Exception as e:
                logger.error(f"Failed to load {row['filepath']}: {e}")
                residuals = np.zeros(self.ts_len, dtype=np.float32)
            label = int(row["class_label"])

        # Use pre-computed padding offset — no RNG object creation per item
        n = len(residuals)
        if n >= self.ts_len:
            padded = residuals[-self.ts_len:]
        elif self.pad_variant == 1:
            # Both sides: use pre-computed left pad amount
            left  = int(self._pad_offsets[idx]) % (self.ts_len - n + 1)
            right = self.ts_len - n - left
            padded = np.concatenate([
                np.zeros(left,  dtype=np.float32),
                residuals,
                np.zeros(right, dtype=np.float32),
            ])
        else:
            # Left only: right-align
            padded = np.zeros(self.ts_len, dtype=np.float32)
            padded[-n:] = residuals

        x = torch.tensor(padded, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(label,  dtype=torch.long)
        return x, y


class SDMLSurrogateDataset(Dataset):
    """
    Dataset for SDML surrogate data (generated from PANGAEA by pangea_cleaner.py).
    Binary: 0 = neutral, 1 = pre_transition.

    Surrogates are already detrended — only normalise here.
    No padding needed since pangea_cleaner.py outputs fixed-length surrogates.

    Parameters
    ----------
    index_df : DataFrame with columns filepath, class_label
    ts_len   : crop/pad all surrogates to this length
    """

    def __init__(self, index_df: pd.DataFrame, ts_len: int):
        self.records = index_df.reset_index(drop=True)
        self.ts_len  = ts_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.records.iloc[idx]

        try:
            df  = pd.read_csv(row["filepath"], header=None)
            raw = df.iloc[:, 0].values.astype(np.float64)
        except Exception as e:
            logger.error(f"Failed to load {row['filepath']}: {e}")
            raw = np.zeros(self.ts_len, dtype=np.float64)

        normed = normalise(raw)

        # Right-align: crop to ts_len or left-pad if shorter
        n = len(normed)
        if n >= self.ts_len:
            out = normed[-self.ts_len:].astype(np.float32)
        else:
            out       = np.zeros(self.ts_len, dtype=np.float32)
            out[-n:]  = normed.astype(np.float32)

        x = torch.tensor(out,  dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(int(row["class_label"]), dtype=torch.long)
        return x, y


# =============================================================================
#  Public API — training
# =============================================================================

def get_dataset_info(dataset_name: str, cfg: dict) -> dict:
    """
    Return dataset metadata without loading any files.

    Returns
    -------
    dict: ts_length, num_classes, class_names, mode (bury | sdml)
    """
    ds_cfg = cfg["datasets"].get(dataset_name)
    if ds_cfg is None:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(cfg['datasets'].keys())}"
        )
    mode = "sdml" if dataset_name.startswith("sdml_") else "bury"
    return {
        "ts_length":   ds_cfg["ts_length"],
        "num_classes": ds_cfg["num_classes"],
        "class_names": ds_cfg["class_names"],
        "train_frac":  ds_cfg["train_frac"],
        "val_frac":    ds_cfg["val_frac"],
        "mode":        mode,
    }


def get_dataloader(
    dataset_name: str,
    split:        str,
    cfg:          dict,
    pad_variant:  int          = 1,
    batch_size:   Optional[int] = None,
    num_workers:  int          = 4,
    seed:         int          = 42,
) -> DataLoader:
    """
    Universal DataLoader factory — the ONLY function training/train.py calls.

    Parameters
    ----------
    dataset_name : "ts_500" | "ts_1500" | "sdml_MS21" | "sdml_MS66" | "sdml_64PE406E1"
    split        : "train" | "val" | "test"
    cfg          : loaded config dict
    pad_variant  : 1 (both sides) or 2 (left only) — Bury mode only
    batch_size   : overrides config value if provided
    num_workers  : DataLoader worker processes
    seed         : reproducibility seed

    Returns
    -------
    DataLoader yielding (x, y):
        x : FloatTensor (B, 1, T)
        y : LongTensor  (B,)
    """
    assert split in ("train", "val", "test"), \
        f"split must be train/val/test, got '{split}'"

    ds_cfg = cfg["datasets"].get(dataset_name)
    if ds_cfg is None:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(cfg['datasets'].keys())}"
        )

    is_sdml    = dataset_name.startswith("sdml_")
    ts_len     = ds_cfg["ts_length"]
    train_frac = ds_cfg["train_frac"]
    val_frac   = ds_cfg["val_frac"]
    # ── Resolve batch size from config defaults when not overridden ──────────
    # get_dataloader does not know the model name, so uses fallback defaults.
    # train.py always passes batch_size explicitly from the per-model config.
    if batch_size is None:
        default_key = "sdml_default" if is_sdml else "bury"
        batch_size  = cfg["training"]["defaults"][default_key]["batch_size"]

    # ── Build file index ──────────────────────────────────────────────────────
    if is_sdml:
        core      = ds_cfg["core"]
        clean_dir = _resolve(cfg, "pangaea_clean")
        index_df  = _build_sdml_index(clean_dir, core)
    else:
        # Zenodo data lives at paths.ts_500 or paths.ts_1500 + /combined/
        base_dir = _resolve(cfg, ds_cfg["path_key"]) / "combined"
        index_df = _build_zenodo_index(base_dir)

    # ── Split ─────────────────────────────────────────────────────────────────
    splits    = _split_index(index_df, train_frac, val_frac, seed)
    split_df  = splits[split]

    logger.info(
        f"[{dataset_name}] {split}: {len(split_df)} samples | "
        f"pad_variant={pad_variant if not is_sdml else 'N/A (SDML)'}"
    )

    # ── Build Dataset ─────────────────────────────────────────────────────────
    split_seed = seed + {"train": 0, "val": 1, "test": 2}[split]

    if is_sdml:
        dataset = SDMLSurrogateDataset(split_df, ts_len)
    else:
        # Pass base_dir so Dataset can find the numpy cache
        dataset = ZenodoEWSDataset(
            split_df, base_dir,
            ts_len      = ts_len,
            pad_variant = pad_variant,
            seed        = split_seed,
        )

    is_train = (split == "train")
    return DataLoader(
        dataset,
        batch_size       = batch_size,
        shuffle          = is_train,
        num_workers      = num_workers,
        pin_memory       = torch.cuda.is_available(),
        drop_last        = is_train,
        persistent_workers = (num_workers > 0),  # keep workers alive between epochs
        prefetch_factor    = 2 if num_workers > 0 else None,
    )


# =============================================================================
#  Public API — PANGAEA inference (testing only)
#  Used by testing/evaluate.py — NOT used during training.
# =============================================================================

def load_pangaea_core(core_name: str, cfg: dict) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load the full ICP-MS calibrated time series for one PANGAEA core.
    PANGAEA is NEVER used for training — this function is for testing only.

    Parameters
    ----------
    core_name : "MS21" | "MS66" | "64PE406E1"
    cfg       : loaded config dict

    Returns
    -------
    time_kyr  : np.ndarray  — Age [ka BP], sorted ascending (oldest first)
    mo_ppm    : np.ndarray  — Mo [mg/kg] values
    df        : pd.DataFrame — full dataframe with all columns
    """
    if core_name not in PANGAEA_CORE_FILES:
        raise ValueError(
            f"Unknown core '{core_name}'. "
            f"Available: {list(PANGAEA_CORE_FILES.keys())}"
        )

    clean_dir = _resolve(cfg, "pangaea_clean")
    filename  = PANGAEA_CORE_FILES[core_name]
    filepath  = clean_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"PANGAEA file not found: {filepath}\n"
            f"Expected: {filename}\n"
            f"Available files: {[f.name for f in clean_dir.glob('*.csv')]}"
        )

    df = pd.read_csv(filepath)

    # Rename columns to clean internal names
    rename = {
        "Age [ka BP]":   "age_kyr_bp",
        "Depth sed [m]": "depth_m",
        "Al [mg/kg]":    "Al",
        "Ba [mg/kg]":    "Ba",
        "Mo [mg/kg]":    "Mo",
        "Ti [mg/kg]":    "Ti",
        "U [mg/kg]":     "U",
    }
    df = df.rename(columns=rename)

    # Sort by age ascending (oldest → most recent)
    df = df.sort_values("age_kyr_bp").reset_index(drop=True)

    # Drop rows where Mo is NaN
    before = len(df)
    df     = df.dropna(subset=["Mo"]).reset_index(drop=True)
    if len(df) < before:
        logger.warning(f"Dropped {before - len(df)} rows with NaN Mo in {core_name}")

    time_kyr = df["age_kyr_bp"].values.astype(np.float64)
    mo_ppm   = df["Mo"].values.astype(np.float64)

    logger.info(
        f"Loaded {core_name}: {len(df)} data points | "
        f"Age range: [{time_kyr.min():.1f}, {time_kyr.max():.1f}] ka BP | "
        f"Mo range: [{mo_ppm.min():.3f}, {mo_ppm.max():.3f}] mg/kg"
    )

    return time_kyr, mo_ppm, df


# =============================================================================
#  Quick sanity check
#  Run directly to verify data loads without errors:
#    python src/dataset_loader.py --dataset ts_500
#    python src/dataset_loader.py --dataset sdml_MS21
#    python src/dataset_loader.py --pangaea MS21
# =============================================================================

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name: ts_500 | ts_1500 | sdml_MS21 ...")
    parser.add_argument("--split",   type=str, default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--pad_variant", type=int, default=1, choices=[1, 2])
    parser.add_argument("--pangaea", type=str, default=None,
                        help="Load and inspect a PANGAEA core: MS21 | MS66 | 64PE406E1")
    parser.add_argument("--config",  type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── PANGAEA inspection ────────────────────────────────────────────────────
    if args.pangaea:
        print(f"\n{'='*60}")
        print(f"  PANGAEA core : {args.pangaea}")
        print(f"{'='*60}")
        time_kyr, mo_ppm, df = load_pangaea_core(args.pangaea, cfg)
        print(f"  Rows         : {len(df)}")
        print(f"  Age range    : {time_kyr.min():.2f} → {time_kyr.max():.2f} ka BP")
        print(f"  Mo range     : {mo_ppm.min():.3f} → {mo_ppm.max():.3f} mg/kg")
        print(f"  Columns      : {list(df.columns)}")
        print(f"\n  First 3 rows:")
        print(df.head(3).to_string())
        print()
        raise SystemExit(0)

    # ── Training dataset inspection ───────────────────────────────────────────
    if args.dataset is None:
        parser.print_help()
        raise SystemExit(0)

    print(f"\n{'='*60}")
    print(f"  Dataset      : {args.dataset}")
    print(f"  Split        : {args.split}")
    print(f"  Pad variant  : {args.pad_variant}")
    print(f"{'='*60}")

    info = get_dataset_info(args.dataset, cfg)
    print(f"\n  ts_length    : {info['ts_length']}")
    print(f"  num_classes  : {info['num_classes']}")
    print(f"  class_names  : {info['class_names']}")
    print(f"  mode         : {info['mode']}")

    loader = get_dataloader(
        dataset_name = args.dataset,
        split        = args.split,
        cfg          = cfg,
        pad_variant  = args.pad_variant,
        num_workers  = 0,
    )

    print(f"\n  Batches in {args.split} loader : {len(loader)}")

    x_batch, y_batch = next(iter(loader))
    print(f"  x shape      : {tuple(x_batch.shape)}  dtype={x_batch.dtype}")
    print(f"  y shape      : {tuple(y_batch.shape)}  dtype={y_batch.dtype}")
    print(f"  x range      : [{x_batch.min():.4f}, {x_batch.max():.4f}]")
    print(f"  x mean abs   : {x_batch.abs().mean():.4f}")

    unique, counts = torch.unique(y_batch, return_counts=True)
    print(f"\n  Class distribution (first batch):")
    for cls_idx, cnt in zip(unique.tolist(), counts.tolist()):
        name = info["class_names"][cls_idx]
        print(f"    {cls_idx} ({name:15s}): {cnt}")

    print(f"\n  OK — {args.dataset}/{args.split} loads correctly.\n")
