"""
src/preprocess_bury_data.py
"""
import argparse
import re
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def preprocess_dataset(raw_dir: str, out_dir: str, ts_length: int):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Processing ts_{ts_length} ---")
    
    # 1. Define exact paths. STRICTLY use output_resids/ as per Bury et al. (2021)
    combined_dir = raw_dir / "combined"
    labels_path = combined_dir / "labels.csv"
    groups_path = combined_dir / "groups.csv"
    ts_dir = combined_dir / "output_resids"
        
    if not (labels_path.exists() and groups_path.exists() and ts_dir.exists()):
        print(f"❌ Error: Missing required files in {raw_dir}")
        print(f"   Expected: {labels_path}, {groups_path}, and {ts_dir}")
        return

    # 2. Load metadata and force sequence_ID to string to guarantee a successful merge
    labels_df = pd.read_csv(labels_path)
    groups_df = pd.read_csv(groups_path)
    
    labels_df["sequence_ID"] = labels_df["sequence_ID"].astype(str).str.strip()
    groups_df["sequence_ID"] = groups_df["sequence_ID"].astype(str).str.strip()
    
    df = pd.merge(labels_df, groups_df, on="sequence_ID", how="inner")
    
    # 3. Map class labels to integers (0: null, 1: fold, 2: hopf, 3: transcritical)
    if df["class_label"].dtype == "object":
        label_map = {"null": 0, "fold": 1, "hopf": 2, "transcritical": 3}
        df["label_int"] = df["class_label"].astype(str).str.lower().str.strip().map(label_map).fillna(0).astype(int)
    else:
        df["label_int"] = df["class_label"].astype(int)
        
    # 4. Map dataset_ID to official Bury splits (1=train, 2=val, 3=test)
    def map_split(g):
        g_str = str(g).strip()
        if g_str == "1": return "train"
        if g_str == "2": return "val"
        if g_str == "3": return "test"
        return "unknown"
        
    df["split"] = df["dataset_ID"].map(map_split)
    
    # Create fast lookup dictionaries
    label_lookup = dict(zip(df["sequence_ID"], df["label_int"]))
    split_lookup = dict(zip(df["sequence_ID"], df["split"]))
    
    # 5. Process time series files
    csv_files = sorted(list(ts_dir.glob("*.csv")))
    print(f"Found {len(csv_files)} time series CSVs in {ts_dir.name}. Loading...")
    
    data = {"train": {"X": [], "y": []}, "val": {"X": [], "y": []}, "test": {"X": [], "y": []}}
    
    for f in tqdm(csv_files, desc=f"Loading ts_{ts_length}"):
        # Extract ONLY the digits from the filename (handles 'resids1', 'resids_1', etc.)
        match = re.search(r'\d+', f.stem)
        if not match:
            continue
        file_id = match.group()
        
        label = label_lookup.get(file_id)
        split = split_lookup.get(file_id)
        
        if label is None or split not in ["train", "val", "test"]:
            continue
            
        try:
            # FIX: Read with header, extract only the Residuals column (last column)
            # This prevents the header row from being counted as data and causing shape mismatches
            df_ts = pd.read_csv(f)
            ts = df_ts.iloc[:, -1].values.astype(np.float32)
            
            if len(ts) == ts_length:
                data[split]["X"].append(ts)
                data[split]["y"].append(label)
        except Exception:
            continue
            
    # 6. Save compressed .npz files by split
    for split_name in ["train", "val", "test"]:
        if len(data[split_name]["X"]) == 0:
            print(f"⚠️ Warning: No data for split '{split_name}'")
            continue
            
        X_split = np.array(data[split_name]["X"], dtype=np.float32)
        y_split = np.array(data[split_name]["y"], dtype=np.int64)
        
        # Add channel dimension for PyTorch Conv1D/LSTM: (N, 1, L)
        X_split = np.expand_dims(X_split, axis=1)
        
        out_path = out_dir / f"{split_name}_{ts_length}.npz"
        np.savez_compressed(out_path, X=X_split, y=y_split)
        print(f"✅ Saved {split_name}: X={X_split.shape}, y={y_split.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    preprocess_dataset("dataset/ts_500", "dataset/processed", ts_length=500)
    preprocess_dataset("dataset/ts_1500", "dataset/processed", ts_length=1500)
    
    print("\n🎉 Preprocessing complete! You can now run training.")
