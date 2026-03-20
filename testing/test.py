import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import load_model_from_checkpoint
from src.dataset_loader import load_dataset
from testing.testing_utils import (
    add_transition_probability,
    build_run_name,
    class_index_to_name,
    enforce_fixed_sequence_length,
    extract_samples,
    get_test_paths,
    save_csv,
    setup_logger,
)


def coerce_sample_feature_dim(sample_x, expected_input_dim: int):
    arr = np.asarray(sample_x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected sample with ndim 1 or 2, got shape {arr.shape}")

    seq_len, feat_dim = arr.shape
    if feat_dim == expected_input_dim:
        return arr
    if feat_dim > expected_input_dim:
        return arr[:, :expected_input_dim]

    pad = np.zeros((seq_len, expected_input_dim - feat_dim), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Run DL inference on synthetic or empirical datasets.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train_dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fixed_length", type=int, default=None)
    parser.add_argument("--length_mode", choices=["last", "first"], default="last")
    args = parser.parse_args()

    device = torch.device(args.device)
    run_name = build_run_name(args.model, args.dataset, args.metric)
    paths = get_test_paths(run_name)
    logger = setup_logger(paths["dl_log"], logger_name=f"testing.dl.{run_name}")

    checkpoint_path = paths["root"] / "checkpoints" / f"{args.model}_{args.train_dataset}_{args.metric}.pt"
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    model, expected_input_dim = load_model_from_checkpoint(checkpoint_path, args.model, num_classes=4, device=device)
    logger.info(f"Loaded model from {checkpoint_path} with inferred input dim: {expected_input_dim}")

    dataset_obj = load_dataset(args.dataset)
    samples = extract_samples(dataset_obj)
    logger.info(f"Loaded {len(samples)} samples from {args.dataset}")

    rows = []
    is_synthetic = args.dataset in {"ts_500", "ts_1500"}

    with torch.no_grad():
        for sample_name, sample_x in samples:
            try:
                if is_synthetic and args.fixed_length:
                    sample_x = enforce_fixed_sequence_length(sample_x, args.fixed_length, args.length_mode)

                x_np = coerce_sample_feature_dim(sample_x, expected_input_dim)
                x_tensor = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)

                probs = F.softmax(model(x_tensor), dim=1).squeeze(0).cpu().numpy()
                pred_idx = int(np.argmax(probs))

                rows.append({
                    "file_name": sample_name, "dataset_name": args.dataset, "model_name": args.model,
                    "metric_name": args.metric, "sequence_length_used": int(x_np.shape[0]),
                    "input_dim_used": int(x_np.shape[1]), "p_fold": float(probs[0]),
                    "p_hopf": float(probs[1]), "p_transcritical": float(probs[2]),
                    "p_null": float(probs[3]), "predicted_class_idx": pred_idx,
                    "predicted_class": class_index_to_name(pred_idx)
                })
            except Exception as e:
                logger.error(f"Failed on sample '{sample_name}': {e}")

    if not rows:
        raise RuntimeError("No inference results produced.")

    df = add_transition_probability(pd.DataFrame(rows))
    save_csv(df, paths["dl_dir"] / "per_series_predictions.csv")

    # Simple aggregations
    agg_cols = ["p_fold", "p_hopf", "p_transcritical", "p_null", "p_transition"]
    save_csv(df.groupby(["dataset_name", "model_name", "metric_name"], as_index=False)[agg_cols].mean(),
             paths["dl_dir"] / "checkpointwise_predictions.csv")

    logger.info(f"Saved DL results to {paths['dl_dir']}")


if __name__ == "__main__":
    main()
