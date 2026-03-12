from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_loader import load_dataset
from testing.testing_utils import (
    add_transition_probability,
    build_run_name,
    checkpoint_path_for,
    class_index_to_name,
    get_test_paths,
    load_checkpoint_state,
    save_csv,
    setup_logger,
)


def build_model(
    model_name: str,
    input_size: int,
    num_classes: int = 4,
    model_kwargs_json: str | None = None,
    model_class: str | None = None,
):
    """
    Dynamically import and create a model from models/.
    """
    base = model_name.replace("-", "_")

    candidate_modules = [
        f"models.{base}",
        f"models.{base.lower()}",
        f"models.{base.upper()}",
        f"models.{base.title()}",
    ]

    module = None
    for module_name in candidate_modules:
        try:
            module = importlib.import_module(module_name)
            break
        except ImportError:
            continue

    if module is None:
        raise ImportError(f"Could not import model module for '{model_name}'")

    model_kwargs = json.loads(model_kwargs_json) if model_kwargs_json else {}

    if model_class:
        cls = getattr(module, model_class)
        return cls(input_size=input_size, num_classes=num_classes, **model_kwargs)

    # First matching class name
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and name.lower() == base.lower():
            return obj(input_size=input_size, num_classes=num_classes, **model_kwargs)

    # Fallback: first class in module
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type):
            return obj(input_size=input_size, num_classes=num_classes, **model_kwargs)

    raise RuntimeError(f"No model class found in module {module.__name__}")


def extract_samples(dataset_obj: Any) -> list[tuple[str, Any]]:
    """
    Convert dataset_loader output into:
    [(file_name, sample), ...]
    """
    if dataset_obj is None:
        raise ValueError("load_dataset returned None")

    # Case 1: (sequences, labels, feature_names)
    if isinstance(dataset_obj, tuple) and len(dataset_obj) == 3:
        sequences, _, _ = dataset_obj
        if sequences is None or len(sequences) == 0:
            raise ValueError("No sequences found in dataset")
        return [(f"sample_{i}", seq) for i, seq in enumerate(sequences)]

    # Case 2: dict with data and optional names
    if isinstance(dataset_obj, dict):
        data = None
        for key in ["X", "x", "data", "features"]:
            if key in dataset_obj:
                data = dataset_obj[key]
                break

        if data is None:
            raise KeyError("Dataset dict must contain one of: X, x, data, features")

        names = (
            dataset_obj.get("names")
            or dataset_obj.get("file_names")
            or dataset_obj.get("filenames")
        )

        if names is None:
            names = [f"sample_{i}" for i in range(len(data))]

        if len(names) != len(data):
            raise ValueError("Number of names does not match number of samples")

        return [(str(name), data[i]) for i, name in enumerate(names)]

    # Case 3: plain list
    if isinstance(dataset_obj, list):
        return [(f"sample_{i}", sample) for i, sample in enumerate(dataset_obj)]

    raise TypeError(f"Unsupported dataset type: {type(dataset_obj)}")


def infer_input_size(state_dict: dict) -> int:
    """
    Infer expected input_size from checkpoint weights.
    """
    candidate_keys = [
        "conv.0.weight",
        "features.0.weight",
        "cnn.0.weight",
        "lstm.weight_ih_l0",
        "gru.weight_ih_l0",
        "rnn.weight_ih_l0",
    ]

    for key in candidate_keys:
        if key in state_dict:
            return state_dict[key].shape[1]

    raise KeyError("Could not infer input_size from checkpoint state_dict")


def force_shape(sample: Any, expected_features: int) -> torch.Tensor:
    """
    Convert input into tensor shape [1, seq_len, expected_features]
    """
    if not isinstance(sample, torch.Tensor):
        sample = torch.tensor(sample, dtype=torch.float32)

    if sample.ndim == 1:
        sample = sample.unsqueeze(-1)  # [seq_len] -> [seq_len, 1]

    if sample.ndim not in (2, 3):
        raise ValueError(f"Unsupported input shape: {sample.shape}")

    current_features = sample.shape[-1]

    if current_features > expected_features:
        sample = sample[..., :expected_features]

    elif current_features < expected_features:
        pad_size = expected_features - current_features
        last_feature = sample[..., -1:]

        if sample.ndim == 2:
            pad = last_feature.repeat(1, pad_size)
            sample = torch.cat([sample, pad], dim=1)
        else:
            pad = last_feature.repeat(1, 1, pad_size)
            sample = torch.cat([sample, pad], dim=-1)

    if sample.ndim == 2:
        sample = sample.unsqueeze(0)

    return sample


def run_inference(
    model,
    device: torch.device,
    samples: list[tuple[str, Any]],
    num_classes: int,
    expected_features: int,
) -> pd.DataFrame:
    """
    Run model inference and return one row per sample.
    """
    rows = []
    model.eval()

    with torch.no_grad():
        for file_name, sample in samples:
            x = force_shape(sample, expected_features).to(device)
            logits = model(x)

            if logits.ndim != 2 or logits.shape[0] != 1:
                raise ValueError(f"Unexpected model output shape: {logits.shape}")

            probs = F.softmax(logits, dim=1).squeeze(0).cpu()

            if probs.numel() != num_classes:
                raise ValueError(f"Expected {num_classes} classes, got {probs.numel()}")

            predicted_idx = int(torch.argmax(probs).item())

            rows.append(
                {
                    "file_name": file_name,
                    "p_fold": probs[0].item(),
                    "p_hopf": probs[1].item(),
                    "p_transcritical": probs[2].item(),
                    "p_null": probs[3].item(),
                    "predicted_class_idx": predicted_idx,
                    "predicted_class": class_index_to_name(predicted_idx),
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    return add_transition_probability(df)


def main():
    parser = argparse.ArgumentParser(description="Run DL inference on test dataset")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train_dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--model_class", default=None)
    parser.add_argument("--model_kwargs", default=None)
    parser.add_argument("--num_classes", type=int, default=4)
    args = parser.parse_args()

    run_name = build_run_name(args.model, args.dataset, args.metric)
    paths = get_test_paths(run_name)

    logger = setup_logger(paths["dl_log"], logger_name=f"testing.dl.{run_name}")
    logger.info(f"Run: {run_name}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    dataset_obj = load_dataset(args.dataset)
    samples = extract_samples(dataset_obj)
    logger.info(f"Loaded {len(samples)} samples")

    checkpoint_path = checkpoint_path_for(args.model, args.train_dataset, args.metric)
    state_dict = load_checkpoint_state(checkpoint_path, device=device)

    expected_features = infer_input_size(state_dict)
    logger.info(f"Checkpoint expects input_size={expected_features}")

    model = build_model(
        model_name=args.model,
        input_size=expected_features,
        num_classes=args.num_classes,
        model_kwargs_json=args.model_kwargs,
        model_class=args.model_class,
    ).to(device)

    model.load_state_dict(state_dict)

    df = run_inference(
        model=model,
        device=device,
        samples=samples,
        num_classes=args.num_classes,
        expected_features=expected_features,
    )

    # Add metadata columns
    df.insert(0, "checkpoint_name", checkpoint_path.name)
    df.insert(0, "train_dataset_name", args.train_dataset)
    df.insert(0, "metric_name", args.metric)
    df.insert(0, "model_name", args.model)
    df.insert(0, "dataset_name", args.dataset)

    # 1. Per-sample predictions
    save_csv(df, paths["dl_dir"] / "per_series_predictions.csv")

    # 2. Checkpoint summary
    checkpoint_summary = pd.DataFrame(
        [
            {
                "checkpoint_name": checkpoint_path.name,
                "model_name": args.model,
                "dataset_name": args.dataset,
                "train_dataset_name": args.train_dataset,
                "metric_name": args.metric,
                "num_samples": len(df),
                "mean_p_fold": df["p_fold"].mean(),
                "mean_p_hopf": df["p_hopf"].mean(),
                "mean_p_transcritical": df["p_transcritical"].mean(),
                "mean_p_null": df["p_null"].mean(),
                "mean_p_transition": df["p_transition"].mean(),
            }
        ]
    )
    save_csv(checkpoint_summary, paths["dl_dir"] / "checkpointwise_predictions.csv")

    # 3. Aggregated predictions
    prob_cols = ["p_fold", "p_hopf", "p_transcritical", "p_null"]

    aggregated = (
        df.groupby("file_name", as_index=False)[prob_cols + ["p_transition"]]
        .mean()
    )

    aggregated["predicted_class_idx"] = aggregated[prob_cols].values.argmax(axis=1)
    aggregated["predicted_class"] = aggregated["predicted_class_idx"].apply(class_index_to_name)

    save_csv(aggregated, paths["dl_dir"] / "aggregated_predictions.csv")

    # 4. Final summary
    summary = aggregated.copy()
    summary["prediction_confidence"] = summary[prob_cols].max(axis=1)
    summary = summary.sort_values("prediction_confidence", ascending=False)

    save_csv(summary, paths["dl_dir"] / "prediction_summary.csv")

    logger.info("Done.")


if __name__ == "__main__":
    main()
