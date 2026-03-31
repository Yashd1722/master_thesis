from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

REPO_ROOT = Path(__file__).resolve().parents[1]

SYNTHETIC_DATASETS = {"ts_500", "ts_1500"}
DEFAULT_PANGAEA_PROXIES = [
    "Al [mg/kg]",
    "Ba [mg/kg]",
    "Mo [mg/kg]",
    "Ti [mg/kg]",
    "U [mg/kg]",
]


def gaussian_smooth_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if sigma <= 0 or x.size < 3:
        return x.copy()
    return gaussian_filter1d(x, sigma=sigma, mode="nearest")


def _to_numeric_filled(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    values = values.interpolate(limit_direction="both").bfill().ffill()
    if values.isna().any():
        values = values.fillna(float(values.median()) if not values.dropna().empty else 0.0)
    return values.to_numpy(dtype=np.float64)


def _safe_time_index(df: pd.DataFrame, time_col: Optional[str]) -> np.ndarray:
    if time_col is not None and time_col in df.columns:
        time_values = pd.to_numeric(df[time_col], errors="coerce").astype(float)
        if time_values.notna().any():
            time_values = time_values.interpolate(limit_direction="both").bfill().ffill()
            return time_values.to_numpy(dtype=np.float64)
    return np.arange(len(df), dtype=np.float64)


def _estimate_gaussian_sigma(
    time_values: np.ndarray,
    time_col: Optional[str],
    bandwidth_years: float,
    fallback_sigma: float = 20.0,
) -> float:
    time_values = np.asarray(time_values, dtype=np.float64).reshape(-1)
    if time_values.size < 3:
        return float(fallback_sigma)

    diffs = np.diff(time_values)
    diffs = np.abs(diffs[np.isfinite(diffs)])
    diffs = diffs[diffs > 0]

    if diffs.size == 0:
        return float(fallback_sigma)

    spacing = float(np.median(diffs))
    is_ka = bool(time_col) and "ka" in time_col.lower()
    spacing_years = spacing * 1000.0 if is_ka else spacing

    if spacing_years <= 0:
        return float(fallback_sigma)

    sigma_samples = bandwidth_years / spacing_years
    if not np.isfinite(sigma_samples) or sigma_samples <= 0:
        return float(fallback_sigma)

    return float(max(1.0, sigma_samples))


def _build_synthetic_dataset(
    dataset_folder: Path,
    dataset_name: str,
    split: str,
    seq_len: int,
) -> Dict[str, object]:
    csv_path = dataset_folder / f"{dataset_name}_{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = ["sequence_ID", "Time", "x", "Residuals", "class_label"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    df = df.sort_values(["sequence_ID", "Time"]).reset_index(drop=True)

    series_items: List[Dict[str, object]] = []
    windows: List[np.ndarray] = []
    labels: List[int] = []

    for sequence_id, group in df.groupby("sequence_ID", sort=True):
        group = group.sort_values("Time").reset_index(drop=True)

        raw_signal = pd.to_numeric(group["x"], errors="coerce").to_numpy(dtype=np.float64)
        residual_signal = pd.to_numeric(group["Residuals"], errors="coerce").to_numpy(dtype=np.float64)
        smooth_signal = raw_signal - residual_signal
        time_index = pd.to_numeric(group["Time"], errors="coerce").to_numpy(dtype=np.float64)
        label = int(group["class_label"].iloc[0])

        model_signal = np.column_stack([raw_signal, residual_signal]).astype(np.float32)

        series_items.append(
            {
                "series_id": str(sequence_id),
                "signal": model_signal,
                "label": label,
                "transition_index": None,
                "raw_signal": raw_signal.astype(np.float32),
                "smooth_signal": smooth_signal.astype(np.float32),
                "residual_signal": residual_signal.astype(np.float32),
                "time_index": time_index.astype(np.float32),
                "metadata": {
                    "dataset_name": dataset_name,
                    "split": split,
                    "default_gaussian_sigma": 0.0,
                    "source": "synthetic",
                },
            }
        )

        if len(model_signal) != seq_len:
            raise ValueError(
                f"Sequence {sequence_id} in {csv_path} has length {len(model_signal)} "
                f"but seq_len={seq_len}."
            )

        windows.append(model_signal)
        labels.append(label)

    if not windows:
        raise RuntimeError(f"No synthetic windows were created from {csv_path}.")

    return {
        "X": np.stack(windows, axis=0).astype(np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "series": series_items,
    }


def _build_pangaea_dataset(
    dataset_folder: Path,
    seq_len: int,
    proxy_cols: Optional[List[str]] = None,
    time_col: str = "Age [ka BP]",
    gaussian_bandwidth_years: float = 900.0,
    use_raw_for_model: bool = False,
) -> Dict[str, object]:
    clean_folder = dataset_folder / "datasets" / "clean_dataset"
    if not clean_folder.exists():
        raise FileNotFoundError(f"Clean dataset folder not found: {clean_folder}")

    core_files = sorted(clean_folder.glob("*_calibratedXRF.csv"))
    if not core_files:
        raise FileNotFoundError(f"No calibrated XRF CSV files found in {clean_folder}")

    proxies = proxy_cols or DEFAULT_PANGAEA_PROXIES
    all_windows: List[np.ndarray] = []
    all_series: List[Dict[str, object]] = []

    for core_file in core_files:
        core_name = core_file.stem.replace("_calibratedXRF", "")
        df = pd.read_csv(core_file)

        time_index = _safe_time_index(df, time_col)
        sigma = _estimate_gaussian_sigma(
            time_values=time_index,
            time_col=time_col,
            bandwidth_years=gaussian_bandwidth_years,
            fallback_sigma=20.0,
        )

        for proxy in proxies:
            if proxy not in df.columns:
                continue

            raw_signal = _to_numeric_filled(df[proxy])
            smooth_signal = gaussian_smooth_1d(raw_signal, sigma=sigma)
            residual_signal = raw_signal - smooth_signal

            if use_raw_for_model:
                model_signal = np.column_stack([raw_signal, residual_signal]).astype(np.float32)
            else:
                model_signal = np.column_stack(
                    [residual_signal, np.zeros_like(residual_signal)]
                ).astype(np.float32)

            series_id = f"{core_name}_{proxy}"
            all_series.append(
                {
                    "series_id": series_id,
                    "signal": model_signal,
                    "label": None,
                    "transition_index": None,
                    "raw_signal": raw_signal.astype(np.float32),
                    "smooth_signal": smooth_signal.astype(np.float32),
                    "residual_signal": residual_signal.astype(np.float32),
                    "time_index": time_index.astype(np.float32),
                    "metadata": {
                        "dataset_name": "pangaea_923197",
                        "core_name": core_name,
                        "proxy_name": proxy,
                        "time_col": time_col,
                        "gaussian_bandwidth_years": float(gaussian_bandwidth_years),
                        "default_gaussian_sigma": float(sigma),
                        "source": "empirical",
                    },
                }
            )

            total_length = model_signal.shape[0]
            num_windows = max(0, total_length - seq_len + 1)
            for start in range(num_windows):
                all_windows.append(model_signal[start : start + seq_len])

    if not all_windows:
        raise RuntimeError("No PANGAEA windows were created. Check seq_len and input files.")

    return {
        "X": np.stack(all_windows, axis=0).astype(np.float32),
        "y": None,
        "series": all_series,
    }


def load_dataset(
    dataset_name: str,
    split: str = "test",
    seq_len: int = 500,
) -> Dict[str, object]:
    dataset_name = str(dataset_name).strip().lower()
    dataset_folder = REPO_ROOT / "dataset" / dataset_name

    if dataset_name in SYNTHETIC_DATASETS:
        return _build_synthetic_dataset(
            dataset_folder=dataset_folder,
            dataset_name=dataset_name,
            split=split,
            seq_len=seq_len,
        )

    if dataset_name == "pangaea_923197":
        return _build_pangaea_dataset(
            dataset_folder=dataset_folder,
            seq_len=seq_len,
        )

    supported = sorted(list(SYNTHETIC_DATASETS) + ["pangaea_923197"])
    raise ValueError(f"Unknown dataset: {dataset_name}. Supported datasets: {supported}")
