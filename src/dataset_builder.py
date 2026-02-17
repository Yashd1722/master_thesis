import os
import pandas as pd
from tqdm import tqdm

BASE = "/home/s466553/Master_thesis/Main/dataset/ts_500/combined"
LABELS = os.path.join(BASE, "labels.csv")
OUT_RESIDS = os.path.join(BASE, "output_resids", "output_resids")
OUT_SIMS = os.path.join(BASE, "output_sims", "output_sims")

labels = pd.read_csv(LABELS)

rows = []
for seq_id, class_label in tqdm(labels[["sequence_ID", "class_label"]].values):
    res_path = os.path.join(OUT_RESIDS, f"resids{seq_id}.csv")
    sim_path = os.path.join(OUT_SIMS, f"tseries{seq_id}.csv")
    if not (os.path.exists(res_path) and os.path.exists(sim_path)):
        continue

    res = pd.read_csv(res_path)   # Time, Residuals
    sim = pd.read_csv(sim_path)   # Time, x
    merged = pd.merge(sim, res, on="Time", how="inner")

    merged["sequence_ID"] = int(seq_id)
    merged["class_label"] = int(class_label)

    # order columns: sequence_ID, Time, x, Residuals, class_label
    merged = merged[["sequence_ID", "Time", "x", "Residuals", "class_label"]]
    rows.append(merged)

flat = pd.concat(rows, ignore_index=True)
flat.to_csv("/home/s466553/Master_thesis/Main/dataset/ts_500_final.csv", index=False)
