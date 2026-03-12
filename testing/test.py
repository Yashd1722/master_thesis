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


def build_model(model_name: str, input_size: int, num_classes: int = 4,
                model_kwargs_json: str | None = None, model_class: str | None = None):
    base = model_name.replace("-", "_")
    candidates = [
        f"models.{base}",
        f"models.{base.upper()}",
        f"models.{base.lower()}",
        f"models.{base.title()}",
        f"models.{base.replace('_', '').upper()}",
    ]
    if "_" in base:
        candidates.append(f"models.{'_'.join(p.upper() for p in base.split('_'))}")

    module = None
    for m in candidates:
        try:
            module = importlib.import_module(m)
            break
        except ImportError:
            continue
    if module is None:
        raise ImportError(f"Could not import model module for '{model_name}'")

    kwargs = json.loads(model_kwargs_json) if model_kwargs_json else {}
    if model_class:
        cls = getattr(module, model_class)
        return cls(input_size=input_size, num_classes=num_classes, **kwargs)

    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type):
            if name.lower() == base.lower():
                return obj(input_size=input_size, num_classes=num_classes, **kwargs)
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type):
            return obj(input_size=input_size, num_classes=num_classes, **kwargs)
    raise RuntimeError(f"No model class in module {module.__name__}")


def _extract_samples(ds):
    if ds is None:
        raise ValueError("load_dataset returned None")
    if isinstance(ds, tuple) and len(ds) == 3:
        seqs = ds[0]
        if seqs is None or not seqs:
            raise ValueError("No sequences in dataset")
        return [(f"sample_{i}", s) for i, s in enumerate(seqs)]
    if isinstance(ds, dict):
        for key in ('X', 'x', 'data', 'features'):
            if key in ds:
                X = ds[key]
                break
        else:
            raise KeyError("Missing X/x/data/features in dataset dict")
        names = ds.get('names') or ds.get('file_names') or ds.get('filenames')
        if names is None:
            names = [f"sample_{i}" for i in range(len(X))]
        return [(str(n), X[i]) for i, n in enumerate(names)]
    if isinstance(ds, (list, tuple)):
        out = []
        for i, item in enumerate(ds):
            if isinstance(item, dict):
                name = str(item.get('name') or item.get('file_name') or item.get('filename') or f"sample_{i}")
                for key in ('x', 'X', 'data', 'features'):
                    if key in item:
                        sample = item[key]
                        break
                else:
                    raise KeyError(f"No data key in sample dict at index {i}")
            elif isinstance(item, (list, tuple)):
                if len(item) >= 2 and isinstance(item[0], str):
                    name, sample = item[0], item[1]
                else:
                    name, sample = f"sample_{i}", item[0]
            else:
                name, sample = f"sample_{i}", item
            if sample is not None:
                out.append((name, sample))
        if not out:
            raise ValueError("No samples extracted")
        return out
    raise TypeError(f"Unsupported dataset type: {type(ds)}")


def infer_input_size(state_dict: dict) -> int:
    keys = [
        "conv.0.weight", "features.0.weight", "cnn.0.weight",
        "lstm.weight_ih_l0", "gru.weight_ih_l0", "rnn.weight_ih_l0"
    ]
    for k in keys:
        if k in state_dict:
            return state_dict[k].shape[1]
    raise KeyError(f"Could not infer input_size from state_dict keys: {list(state_dict.keys())[:10]}")


def force_shape(x: Any, expected: int) -> torch.Tensor:
    """Convert sample to [1, seq_len, expected] tensor."""
    if x is None:
        raise TypeError("Sample is None")
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    # Ensure 2D [seq_len, features] or 3D [1, seq_len, features]
    if x.ndim == 1:
        x = x.unsqueeze(-1)                # -> [seq_len, 1]
    if x.ndim not in (2, 3):
        raise ValueError(f"Unsupported shape {x.shape}")

    # Adjust feature dimension to match expected
    feat = x.shape[-1]
    if feat > expected:
        x = x[..., :expected]
    elif feat < expected:
        if feat == 1:
            # Repeat the single feature
            if x.ndim == 2:
                x = x.repeat(1, expected)
            else:  # 3D
                x = x.repeat(1, 1, expected)
        else:
            # Pad with last feature
            pad_size = expected - feat
            last = x[..., -1:]
            if x.ndim == 2:
                pad = last.repeat(1, pad_size)
                x = torch.cat([x, pad], dim=1)
            else:  # 3D
                pad = last.repeat(1, 1, pad_size)
                x = torch.cat([x, pad], dim=-1)

    # Add batch dimension if missing (i.e., input was 2D)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    return x


def run_inference(model, device, samples, num_classes, expected):
    rows = []
    model.eval()
    with torch.no_grad():
        for name, sample in samples:
            x = force_shape(sample, expected).to(device)
            logits = model(x)
            if logits.ndim != 2 or logits.shape[0] != 1:
                raise ValueError(f"Model output shape {logits.shape} not [1, num_classes]")
            probs = F.softmax(logits, dim=1).squeeze(0).cpu()
            if probs.numel() != num_classes:
                raise ValueError(f"Expected {num_classes} outputs, got {probs.numel()}")
            pred = int(torch.argmax(probs))
            rows.append({
                "file_name": name,
                "p_fold": probs[0].item(),
                "p_hopf": probs[1].item(),
                "p_transcritical": probs[2].item(),
                "p_null": probs[3].item(),
                "predicted_class_idx": pred,
                "predicted_class": class_index_to_name(pred),
            })
    df = pd.DataFrame(rows)
    return add_transition_probability(df) if not df.empty else df


def main():
    parser = argparse.ArgumentParser()
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
    logger.info(f"Run: {run_name} | Dataset: {args.dataset} | Model: {args.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    ds = load_dataset(args.dataset)
    samples = _extract_samples(ds)
    logger.info(f"Loaded {len(samples)} samples")

    ckpt = checkpoint_path_for(args.model, args.train_dataset, args.metric)
    state = load_checkpoint_state(ckpt, device=device)
    expected = infer_input_size(state)
    logger.info(f"Checkpoint expects input_size={expected}")

    model = build_model(args.model, expected, args.num_classes,
                        args.model_kwargs, args.model_class).to(device)
    model.load_state_dict(state)

    df = run_inference(model, device, samples, args.num_classes, expected)

    for col, val in [("dataset_name", args.dataset), ("model_name", args.model),
                     ("metric_name", args.metric), ("train_dataset_name", args.train_dataset),
                     ("checkpoint_name", ckpt.name)]:
        df.insert(0, col, val)

    per = paths["dl_dir"] / "per_series_predictions.csv"
    save_csv(df, per)

    ckpt_sum = pd.DataFrame([{
        "checkpoint_name": ckpt.name, "model_name": args.model, "dataset_name": args.dataset,
        "train_dataset_name": args.train_dataset, "metric_name": args.metric,
        "num_samples": len(df), "mean_p_fold": df["p_fold"].mean(),
        "mean_p_hopf": df["p_hopf"].mean(), "mean_p_transcritical": df["p_transcritical"].mean(),
        "mean_p_null": df["p_null"].mean(), "mean_p_transition": df["p_transition"].mean()
    }])
    save_csv(ckpt_sum, paths["dl_dir"] / "checkpointwise_predictions.csv")

    agg = df.groupby("file_name", as_index=False)[
        ["p_fold", "p_hopf", "p_transcritical", "p_null", "p_transition"]].mean()
    cols = ["p_fold", "p_hopf", "p_transcritical", "p_null"]
    agg["predicted_class"] = agg[cols].idxmax(axis=1).map({
        "p_fold": "fold", "p_hopf": "hopf", "p_transcritical": "transcritical", "p_null": "null"
    })
    save_csv(agg, paths["dl_dir"] / "aggregated_predictions.csv")

    summary = agg.copy()
    summary["prediction_confidence"] = summary[cols].max(axis=1)
    summary = summary.sort_values("prediction_confidence", ascending=False)
    save_csv(summary, paths["dl_dir"] / "prediction_summary.csv")

    logger.info("Done.")


if __name__ == "__main__":
    main()
