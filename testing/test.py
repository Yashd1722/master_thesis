# testing/test.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.evaluation import (
    compute_classification_metrics,
    compute_multiclass_roc_auc,
    save_confusion_matrix_csv,
    save_json,
    save_roc_curve_csv,
)
from models.CNN import CNNClassifier
from models.LSTM import LSTMClassifier
from models.CNN_LSTM import CNNLSTMClassifier
from src.dataset_loader import load_dataset
from testing.testing_utils import (
    CLASS_NAMES,
    build_checkpoint_name,
    build_run_name,
    class_index_to_name,
    extract_samples,
    get_test_paths,
    load_checkpoint_state,
    preprocess_for_dl,
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
    """
    Infer model input feature dimension from checkpoint weights.
    """
    for key, value in state.items():
        if not hasattr(value, "shape"):
            continue

        shape = tuple(value.shape)

        if "weight_ih" in key and len(shape) == 2:
            return int(shape[1])

        if key == "features.0.weight" and len(shape) >= 2:
            return int(shape[1])

        if "conv" in key and "weight" in key and len(shape) >= 2:
            return int(shape[1])

    return None


def coerce_feature_dim(x: np.ndarray, expected_input_dim: int) -> np.ndarray:
    """
    Ensure x has shape [T, expected_input_dim].
    If there are more features, keep the first ones.
    If there are fewer, zero-pad feature dimension.
    """
    arr = np.asarray(x, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array after preprocessing, got shape={arr.shape}")

    seq_len, feat_dim = arr.shape

    if feat_dim == expected_input_dim:
        return arr

    if feat_dim > expected_input_dim:
        return arr[:, :expected_input_dim]

    pad = np.zeros((seq_len, expected_input_dim - feat_dim), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=1)


def add_transition_probability(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["p_transition"] = 1.0 - out["p_null"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DL testing on synthetic or empirical datasets")
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--train_dataset", required=True, type=str, choices=["ts_500", "ts_1500"])
    parser.add_argument("--model", required=True, type=str, choices=["cnn", "lstm", "cnn_lstm"])
    parser.add_argument("--metric", required=True, type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--experiment", default="base", type=str, choices=["base", "trend", "season", "trend_season"])
    parser.add_argument("--length_mode", default="last", type=str, choices=["first", "last"])
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    run_name = build_run_name(
        model=args.model,
        train_dataset=args.train_dataset,
        metric=args.metric,
        experiment=args.experiment,
        test_dataset=args.dataset,
    )
    paths = get_test_paths(run_name)
    logger = setup_logger(paths["dl_log"], logger_name=f"testing.dl.{run_name}")

    checkpoint_name = build_checkpoint_name(
        model=args.model,
        train_dataset=args.train_dataset,
        metric=args.metric,
        experiment=args.experiment,
    )
    checkpoint_path = paths["root"] / "checkpoints" / checkpoint_name

    logger.info("Run name: %s", run_name)
    logger.info("Dataset: %s", args.dataset)
    logger.info("Train dataset: %s", args.train_dataset)
    logger.info("Model: %s", args.model)
    logger.info("Metric: %s", args.metric)
    logger.info("Experiment: %s", args.experiment)
    logger.info("Length mode: %s", args.length_mode)
    logger.info("Device: %s", device)
    logger.info("Checkpoint: %s", checkpoint_path)

    state = load_checkpoint_state(checkpoint_path, device=device)
    expected_input_dim = infer_input_dim_from_state_dict(state)
    if expected_input_dim is None:
        raise RuntimeError("Could not infer input_dim from checkpoint state_dict.")

    logger.info("Expected input_dim from checkpoint: %d", expected_input_dim)

    model = build_model(args.model, input_dim=expected_input_dim, num_classes=4)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    dataset_obj = load_dataset(args.dataset)
    samples = extract_samples(dataset_obj)
    logger.info("Loaded %d samples", len(samples))

    rows = []
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[List[float]] = []

    with torch.no_grad():
        for sample_idx, (sample_name, sample_x, sample_y) in enumerate(samples):
            try:
                x_np = preprocess_for_dl(
                    sample=sample_x,
                    train_dataset=args.train_dataset,
                    length_mode=args.length_mode,
                )
                x_np = coerce_feature_dim(x_np, expected_input_dim)

                x_tensor = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T, F]
                logits = model(x_tensor)
                probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

                pred_idx = int(np.argmax(probs))
                pred_name = class_index_to_name(pred_idx)

                row = {
                    "file_name": sample_name,
                    "dataset_name": args.dataset,
                    "train_dataset_name": args.train_dataset,
                    "model_name": args.model,
                    "metric_name": args.metric,
                    "experiment": args.experiment,
                    "sample_index": sample_idx,
                    "sequence_length_used": int(x_np.shape[0]),
                    "input_dim_used": int(x_np.shape[1]),
                    "p_fold": float(probs[0]),
                    "p_hopf": float(probs[1]),
                    "p_transcritical": float(probs[2]),
                    "p_null": float(probs[3]),
                    "predicted_class_idx": pred_idx,
                    "predicted_class": pred_name,
                    "true_class_idx": None if sample_y is None else int(sample_y),
                    "true_class": None if sample_y is None else class_index_to_name(int(sample_y)),
                }
                rows.append(row)

                if sample_y is not None:
                    y_true.append(int(sample_y))
                    y_pred.append(pred_idx)
                    y_prob.append(probs.tolist())

                if sample_idx < 10:
                    logger.info(
                        "Sample=%s | len=%d | probs=[%.4f, %.4f, %.4f, %.4f] | pred=%s",
                        sample_name,
                        int(x_np.shape[0]),
                        float(probs[0]),
                        float(probs[1]),
                        float(probs[2]),
                        float(probs[3]),
                        pred_name,
                    )

            except Exception as exc:
                logger.exception("Failed on sample '%s': %s", sample_name, exc)

    if not rows:
        raise RuntimeError("No inference rows were produced.")

    per_series_df = pd.DataFrame(rows)
    per_series_df = add_transition_probability(per_series_df)

    prediction_summary_df = (
        per_series_df.sort_values(by=["p_transition", "file_name"], ascending=[False, True])
        .reset_index(drop=True)
    )

    aggregated_df = (
        per_series_df.groupby(["file_name"], as_index=False)[
            ["p_fold", "p_hopf", "p_transcritical", "p_null", "p_transition"]
        ].mean()
    )
    aggregated_df["predicted_class"] = (
        aggregated_df[["p_fold", "p_hopf", "p_transcritical", "p_null"]]
        .idxmax(axis=1)
        .str.replace("p_", "", regex=False)
    )

    save_csv(per_series_df, paths["dl_dir"] / "per_series_predictions.csv")
    save_csv(prediction_summary_df, paths["dl_dir"] / "prediction_summary.csv")
    save_csv(aggregated_df, paths["dl_dir"] / "aggregated_predictions.csv")

    logger.info("Saved prediction files to %s", paths["dl_dir"])

    metrics_payload = {
        "run_name": run_name,
        "dataset": args.dataset,
        "train_dataset": args.train_dataset,
        "model": args.model,
        "metric": args.metric,
        "experiment": args.experiment,
        "num_samples": int(len(per_series_df)),
    }

    if y_true:
        cls_metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            class_names=CLASS_NAMES,
        )
        roc_metrics = compute_multiclass_roc_auc(
            y_true=y_true,
            y_prob=y_prob,
            class_names=CLASS_NAMES,
        )

        metrics_payload.update(cls_metrics)
        metrics_payload["roc_auc_macro_ovr"] = roc_metrics["roc_auc_macro_ovr"]
        metrics_payload["roc_auc_weighted_ovr"] = roc_metrics["roc_auc_weighted_ovr"]

        save_confusion_matrix_csv(
            cls_metrics["confusion_matrix"],
            paths["dl_dir"] / "confusion_matrix.csv",
            class_names=CLASS_NAMES,
        )

        for class_name, payload in roc_metrics["per_class"].items():
            fpr = payload.get("fpr", [])
            tpr = payload.get("tpr", [])
            thresholds = payload.get("thresholds", [])

            if len(fpr) == 0:
                continue

            save_roc_curve_csv(
                np.asarray(fpr, dtype=np.float64),
                np.asarray(tpr, dtype=np.float64),
                np.asarray(thresholds, dtype=np.float64),
                paths["dl_dir"] / "roc_curves" / f"roc_curve_{class_name}.csv",
            )

        logger.info("Accuracy: %.6f", cls_metrics["accuracy"])
        logger.info("Balanced accuracy: %.6f", cls_metrics["balanced_accuracy"])
        logger.info("F1 macro: %.6f", cls_metrics["f1_macro"])
        logger.info("ROC AUC macro OVR: %s", str(roc_metrics["roc_auc_macro_ovr"]))
    else:
        logger.info("No labels found for dataset=%s. Saved predictions only.", args.dataset)

    save_json(metrics_payload, paths["dl_dir"] / "metrics.json")
    logger.info("Saved metrics to %s", paths["dl_dir"] / "metrics.json")
    logger.info("Done.")


if __name__ == "__main__":
    main()
