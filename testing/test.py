from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.CNN import CNNClassifier
from models.LSTM import LSTMClassifier
from models.CNN_LSTM import CNNLSTMClassifier
from src.dataset_loader import load_dataset
from testing.testing_utils import (
    add_transition_probability,
    build_run_name,
    class_index_to_name,
    enforce_fixed_sequence_length,
    extract_samples,
    get_test_paths,
    load_checkpoint_state,
    save_csv,
    setup_logger,
)


def build_model(model_name: str, input_dim: int, num_classes: int = 4):
    model_name = model_name.lower()

    if model_name == "cnn":
        return CNNClassifier(input_size=input_dim, num_classes=num_classes)
    if model_name == "lstm":
        return LSTMClassifier(input_size=input_dim, num_classes=num_classes)
    if model_name in {"cnn_lstm", "cnnlstm"}:
        return CNNLSTMClassifier(input_size=input_dim, num_classes=num_classes)

    raise ValueError(f"Unsupported model name: {model_name}")


def infer_input_dim_from_state_dict(state: dict) -> int | None:
    for key, value in state.items():
        if not hasattr(value, "shape"):
            continue
        shape = tuple(value.shape)

        # Common pattern for LSTM/CNN-LSTM input weights
        if "weight_ih" in key and len(shape) == 2:
            return int(shape[1])

        # Fallback for conv layers
        if "conv" in key and "weight" in key and len(shape) >= 2:
            return int(shape[1])

        # New: handle features.0.weight (first conv layer in many CNNs)
        if key == "features.0.weight" and len(shape) >= 2:
            return int(shape[1])

    return None


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
    parser.add_argument("--dataset", required=True, help="Dataset name for src.dataset_loader.load_dataset()")
    parser.add_argument("--train_dataset", required=True, help="Dataset token used in checkpoint naming")
    parser.add_argument("--model", required=True, help="Model name: cnn, lstm, cnn_lstm")
    parser.add_argument("--metric", required=True, help="Metric name used in checkpoint naming")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument(
        "--fixed_length",
        type=int,
        default=None,
        help="Synthetic only: force sequences to a fixed length, e.g. 500 or 1500.",
    )
    parser.add_argument(
        "--length_mode",
        choices=["last", "first"],
        default="last",
        help="How to crop if a synthetic sequence is longer than fixed_length.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    run_name = build_run_name(args.model, args.dataset, args.metric)
    paths = get_test_paths(run_name)
    logger = setup_logger(paths["dl_log"], logger_name=f"testing.dl.{run_name}")

    logger.info("Run name: %s", run_name)
    logger.info("Dataset: %s", args.dataset)
    logger.info("Train dataset token: %s", args.train_dataset)
    logger.info("Model: %s | Metric: %s", args.model, args.metric)
    logger.info("Device: %s", device)
    logger.info("Fixed length: %s", args.fixed_length)
    logger.info("Length mode: %s", args.length_mode)

    checkpoint_path = paths["root"] / "checkpoints" / f"{args.model}_{args.train_dataset}_{args.metric}.pt"
    state = load_checkpoint_state(checkpoint_path, device=device)

    expected_input_dim = infer_input_dim_from_state_dict(state)
    if expected_input_dim is None:
        print("Checkpoint state_dict keys:", list(state.keys()))
        raise RuntimeError("Could not infer expected input_dim from checkpoint state_dict.")

    logger.info("Expected input dim inferred from checkpoint: %d", expected_input_dim)

    model = build_model(args.model, expected_input_dim, num_classes=4)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    dataset_obj = load_dataset(args.dataset)
    samples = extract_samples(dataset_obj)
    logger.info("Loaded %d samples.", len(samples))

    is_synthetic = args.dataset in {"ts_500", "ts_1500"}

    rows = []
    with torch.no_grad():
        for sample_idx, (sample_name, sample_x) in enumerate(samples):
            try:
                if is_synthetic and args.fixed_length is not None:
                    sample_x = enforce_fixed_sequence_length(
                        sample_x,
                        target_length=args.fixed_length,
                        mode=args.length_mode,
                    )

                x_np = coerce_sample_feature_dim(sample_x, expected_input_dim)
                x_tensor = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)

                logits = model(x_tensor)
                probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

                pred_idx = int(np.argmax(probs))
                pred_name = class_index_to_name(pred_idx)

                rows.append(
                    {
                        "file_name": sample_name,
                        "dataset_name": args.dataset,
                        "model_name": args.model,
                        "metric_name": args.metric,
                        "sample_index": sample_idx,
                        "sequence_length_used": int(x_np.shape[0]),
                        "input_dim_used": int(x_np.shape[1]),
                        "p_fold": float(probs[0]),
                        "p_hopf": float(probs[1]),
                        "p_transcritical": float(probs[2]),
                        "p_null": float(probs[3]),
                        "predicted_class_idx": pred_idx,
                        "predicted_class": pred_name,
                    }
                )

            except Exception as exc:
                logger.exception("Failed on sample '%s': %s", sample_name, exc)

    if not rows:
        raise RuntimeError("No inference rows were produced.")

    per_series_df = pd.DataFrame(rows)
    per_series_df = add_transition_probability(per_series_df)

    checkpointwise_df = (
        per_series_df.groupby(["dataset_name", "model_name", "metric_name"], as_index=False)[
            ["p_fold", "p_hopf", "p_transcritical", "p_null", "p_transition"]
        ]
        .mean()
    )

    aggregated_df = (
        per_series_df.groupby(["file_name"], as_index=False)[
            ["p_fold", "p_hopf", "p_transcritical", "p_null", "p_transition"]
        ]
        .mean()
    )
    aggregated_df["predicted_class"] = aggregated_df[
        ["p_fold", "p_hopf", "p_transcritical", "p_null"]
    ].idxmax(axis=1).str.replace("p_", "", regex=False)

    prediction_summary_df = per_series_df.sort_values(
        by=["p_transition", "file_name"], ascending=[False, True]
    ).reset_index(drop=True)

    save_csv(per_series_df, paths["dl_dir"] / "per_series_predictions.csv")
    save_csv(checkpointwise_df, paths["dl_dir"] / "checkpointwise_predictions.csv")
    save_csv(aggregated_df, paths["dl_dir"] / "aggregated_predictions.csv")
    save_csv(prediction_summary_df, paths["dl_dir"] / "prediction_summary.csv")

    logger.info("Saved DL outputs to %s", paths["dl_dir"])
    logger.info("Done.")


if __name__ == "__main__":
    main()
