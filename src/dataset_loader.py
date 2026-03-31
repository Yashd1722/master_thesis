# src/dataset_loader.py
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d  # you may need to install scipy

REPO_ROOT = Path(__file__).resolve().parents[1]

def gaussian_smooth_1d(x, sigma):
    """Gaussian smoothing with edge handling."""
    if sigma <= 0:
        return x.copy()
    return gaussian_filter1d(x, sigma, mode='nearest')

def load_dataset(dataset_name: str, split: str = "test", seq_len: int = 500):
    """
    Unified loader for all datasets.

    For synthetic datasets (ts_500, ts_1500): reads the split CSV.
    For pangaea_923197: detrends each proxy, computes residuals, and creates
    a dummy second feature (zeros). Returns a series for each (core, proxy).
    """
    dataset_folder = REPO_ROOT / "dataset" / dataset_name

    if dataset_name in ["ts_500", "ts_1500"]:
        # ==================== Synthetic datasets ====================
        csv_path = dataset_folder / f"{dataset_name}_{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        required = ["sequence_ID", "Time", "x", "Residuals", "class_label"]
        if not all(col in df.columns for col in required):
            raise ValueError(f"CSV missing required columns: {required}")

        # Build series list (full signals)
        series_list = []
        for sid, group in df.groupby("sequence_ID"):
            group = group.sort_values("Time")
            signal = group[["x", "Residuals"]].values.astype(np.float32)  # (T, 2)
            label = int(group["class_label"].iloc[0])
            series_list.append({
                "series_id": str(sid),
                "signal": signal,
                "label": label,
                "transition_index": None,
            })

        # Convert to windows (each row is already a window)
        X = df[["x", "Residuals"]].values.reshape(-1, seq_len, 2).astype(np.float32)
        y = df["class_label"].values.astype(np.int64)[::seq_len]
        return {"X": X, "y": y, "series": series_list}

    elif dataset_name == "pangaea_923197":
        # ==================== Pangaea (empirical) dataset ====================
        clean_folder = dataset_folder / "datasets" / "clean_dataset"
        if not clean_folder.exists():
            raise FileNotFoundError(f"Clean dataset folder not found: {clean_folder}")

        core_files = sorted(clean_folder.glob("*_calibratedXRF.csv"))
        if not core_files:
            raise FileNotFoundError(f"No *calibratedXRF.csv files found in {clean_folder}")

        # All geochemical proxies
        proxy_cols = ["Al [mg/kg]", "Ba [mg/kg]", "Mo [mg/kg]", "Ti [mg/kg]", "U [mg/kg]"]
        # Gaussian filter sigma = bandwidth / 2.35? For now use a fixed sigma (900 years / spacing ~ 10-50 years)
        # We'll use a sigma of 20 (typical for 900-year filter with ~45-year spacing). You can adjust.
        sigma = 20.0   # corresponds roughly to 900-year smoothing with ~45-year resolution

        all_windows = []
        all_series = []

        for core_file in core_files:
            core_name = core_file.stem.replace("_calibratedXRF", "")
            df = pd.read_csv(core_file)

            for proxy in proxy_cols:
                if proxy not in df.columns:
                    print(f"Warning: Core {core_name} missing column {proxy}; skipping")
                    continue

                # 1. Extract raw proxy values
                raw = df[proxy].values.astype(np.float64)

                # 2. Detrend with Gaussian filter
                smoothed = gaussian_smooth_1d(raw, sigma=sigma)

                # 3. Compute residuals
                residual = (raw - smoothed).astype(np.float32)

                # 4. Create 2-feature input: [residual, zeros]
                signal_2d = np.column_stack([residual, np.zeros_like(residual)])  # shape (T, 2)

                series_id = f"{core_name}_{proxy}"
                series_item = {
                    "series_id": series_id,
                    "signal": signal_2d,
                    "label": None,
                    "transition_index": None,
                }
                all_series.append(series_item)

                # 5. Create sliding windows
                T = signal_2d.shape[0]
                n_windows = max(0, T - seq_len + 1)
                for i in range(n_windows):
                    window = signal_2d[i:i+seq_len]  # (seq_len, 2)
                    all_windows.append(window)

        if not all_windows:
            raise RuntimeError("No windows created from pangaea data")

        X = np.stack(all_windows, axis=0)   # (n_windows, seq_len, 2)
        y = None
        return {"X": X, "y": y, "series": all_series}

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: ts_500, ts_1500, pangaea_923197")
