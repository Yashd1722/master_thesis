"""
src/rolling_window.py
Calculates Early Warning Signals (EWS) for PANGAEA core data.
FIX: Now also loads neutral (null) series so AUC can be computed!
"""
import numpy as np
import pandas as pd
from pathlib import Path
from metric.kendall_tau import compute_kendall_tau

ELEMENTS = ["Al", "Ba", "Mo", "Ti", "U"]

def calculate_variance(window_data):
    if len(window_data) < 2: return 0.0
    return float(np.var(window_data, ddof=1))

def calculate_lag1_autocorrelation(window_data):
    if len(window_data) < 3: return 0.0
    today = window_data[:-1]
    tomorrow = window_data[1:]
    if np.std(today) < 1e-10 or np.std(tomorrow) < 1e-10: return 0.0
    return float(np.corrcoef(today, tomorrow)[0, 1])

def format_for_neural_network(residuals, current_position, target_length):
    segment = residuals[:current_position].astype(np.float64)
    average_size = np.mean(np.abs(segment))
    if average_size > 1e-10:
        segment = segment / average_size
        
    current_length = len(segment)
    if current_length >= target_length:
        return segment[-target_length:].astype(np.float32)
        
    padded = np.zeros(target_length, dtype=np.float32)
    padded[-current_length:] = segment.astype(np.float32)
    return padded

def analyze_one_segment(residuals, ages, element, core, sapropel, segment_type, cfg, ts_len):
    total_points = len(residuals)
    window_size = max(10, int(cfg["inference"]["rolling_window_frac"] * total_points))
    num_steps = cfg["inference"]["prediction_steps"]
    checkpoints = np.linspace(window_size, total_points, num_steps, dtype=int)
    
    variances, autocorrelations, ages_at_checkpoints, model_inputs = [], [], [], []
    
    for pos in checkpoints:
        window = residuals[pos - window_size : pos]
        variances.append(calculate_variance(window))
        autocorrelations.append(calculate_lag1_autocorrelation(window))
        ages_at_checkpoints.append(ages[pos - 1] if pos - 1 < len(ages) else np.nan)
        model_inputs.append(format_for_neural_network(residuals, pos, ts_len))
        
    variances = np.array(variances)
    autocorrelations = np.array(autocorrelations)
    
    return {
        "positions": checkpoints, "ages": np.array(ages_at_checkpoints),
        "variance": variances, "lag1_ac": autocorrelations,
        "ktau_var": compute_kendall_tau(variances), "ktau_ac": compute_kendall_tau(autocorrelations),
        "dl_inputs": model_inputs, "core": core, "sapropel": sapropel,
        "element": element, "segment_type": segment_type
    }

def run_all_sapropels(core_name, cfg, ts_len):
    repo_root = Path(__file__).resolve().parents[1]
    clean_dir = repo_root / cfg["paths"]["pangaea_clean"] / core_name
    sapropels = cfg["pangaea"]["cores"][core_name]["sapropels"]
    test_sapropels = [s for s in sapropels if s["role"] == "test"]
    all_results = {}
    
    for sap in test_sapropels:
        sap_id = sap["id"]
        sap_results = {}
        for element in ELEMENTS:
            elem_results = {}
            
            # 1. Load Forced Data
            forced_path = clean_dir / f"{core_name}_{sap_id}_forced.csv"
            if not forced_path.exists(): continue
                
            df_f = pd.read_csv(forced_path)
            col_name = f"{element}_residuals"
            if col_name not in df_f.columns: continue
                
            valid_data = ~np.isnan(df_f[col_name].values)
            if valid_data.sum() >= 10:
                elem_results["forced"] = analyze_one_segment(
                    residuals=df_f[col_name].values[valid_data], ages=df_f["age_kyr_bp"].values[valid_data],
                    element=element, core=core_name, sapropel=sap_id, segment_type="forced", cfg=cfg, ts_len=ts_len
                )
            
            # 2. Load Neutral (Null) Data -> THIS FIXES THE NAN AUC ISSUE!
            null_path = clean_dir / f"{core_name}_{sap_id}_{element}_ar1_null.csv"
            if null_path.exists():
                df_n = pd.read_csv(null_path)
                null_cols = [c for c in df_n.columns if c.startswith("null_")]
                if null_cols:
                    # Use the first null series for the rolling window analysis
                    first_null_col = null_cols[0]
                    valid_n = ~np.isnan(df_n[first_null_col].values)
                    if valid_n.sum() >= 10:
                        elem_results["neutral"] = analyze_one_segment(
                            residuals=df_n[first_null_col].values[valid_n], ages=df_n["age_kyr_bp"].values[valid_n],
                            element=element, core=core_name, sapropel=sap_id, segment_type="neutral", cfg=cfg, ts_len=ts_len
                        )
                        
            sap_results[element] = elem_results
        all_results[sap_id] = sap_results
    return all_results
