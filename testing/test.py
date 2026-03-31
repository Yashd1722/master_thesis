# testing/test.py

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------
# repo root import setup
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.acc import compute  # noqa: E402
from metrics.f1_macro import compute  # noqa: E402
from src.plot_testing_results import get_testing_output_dirs, run_all_testing_plots  # noqa: E402
from testing.testing_utils import (  # noqa: E402
    build_model_from_checkpoint,
    collate_eval_batch,
    create_progressive_series_records,
    ensure_dir,
    extract_checkpoint_metadata,
    infer_model_name_from_checkpoint,
    load_checkpoint_safely,
    load_test_dataset_for_inference,
    predict_batch_probabilities,
    resolve_class_names,
    resolve_device,
    save_json,
    save_numpy_confusion_inputs,
    set_seed,
    summarise_run_config,
)

LOGGER = logging.getLogger("testing.test")


# ---------------------------------------------------------------------
# args
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test DL checkpoints with sample-level and progressive per-series predictions."
    )

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset token: ts_500, ts_1500, pangaea_923197")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")
    parser.add_argument("--seed", type=int, default=42)

    # progressive prediction
    parser.add_argument("--progressive-start-frac", type=float, default=0.60)
    parser.add_argument("--progressive-end-frac", type=float, default=1.00)
    parser.add_argument("--progressive-num-steps", type=int, default=40)
    parser.add_argument("--min-prefix-len", type=int, default=64)

    # outputs
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-summary-csv", action="store_true")
    parser.add_argument("--save-series-csv", action="store_true")
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------
def setup_run_logging(log_file: Path, verbose: bool = False) -> None:
    ensure_dir(log_file.parent)

    level = logging.DEBUG if verbose else logging.INFO

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # clear old handlers to avoid duplicate logs
    if root_logger.handlers:
        root_logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)


# ---------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "acc": float(compute_acc(y_true, y_pred)),
        "f1_macro": float(compute_f1_macro(y_true, y_pred)),
        "precision_macro": float(compute_precision_macro(y_true, y_pred)),
        "recall_macro": float(compute_recall_macro(y_true, y_pred)),
    }


# ---------------------------------------------------------------------
# sample-level evaluation
# ---------------------------------------------------------------------
def run_sample_level_evaluation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Dict[str, Any]:
    model.eval()

    all_probs: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_series_ids: List[str] = []
    all_window_ids: List[str] = []

    with torch.no_grad():
        for batch in loader:
            x, y, meta = collate_eval_batch(batch)

            x = x.to(device)
            probs = predict_batch_probabilities(model=model, x=x)
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.append(preds)
            all_targets.append(y.cpu().numpy())

            batch_series_ids = meta.get("series_id", [])
            batch_window_ids = meta.get("window_id", [])

            all_series_ids.extend([str(v) for v in batch_series_ids])
            all_window_ids.extend([str(v) for v in batch_window_ids])

    probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, len(class_names)))
    preds_np = np.concatenate(all_preds, axis=0) if all_preds else np.empty((0,), dtype=int)
    targets_np = np.concatenate(all_targets, axis=0) if all_targets else np.empty((0,), dtype=int)

    metrics = compute_metrics(targets_np, preds_np)

    rows: List[Dict[str, Any]] = []
    for i in range(len(preds_np)):
        row: Dict[str, Any] = {
            "series_id": all_series_ids[i] if i < len(all_series_ids) else f"series_{i}",
            "window_id": all_window_ids[i] if i < len(all_window_ids) else str(i),
            "y_true": int(targets_np[i]),
            "y_pred": int(preds_np[i]),
            "true_class_name": class_names[int(targets_np[i])],
            "pred_class_name": class_names[int(preds_np[i])],
        }
        for c_idx, c_name in enumerate(class_names):
            row[f"prob_{c_name}"] = float(probs_np[i, c_idx])
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    return {
        "metrics": metrics,
        "summary_df": summary_df,
        "y_true": targets_np,
        "y_pred": preds_np,
        "probs": probs_np,
    }


# ---------------------------------------------------------------------
# progressive series evaluation
# ---------------------------------------------------------------------
def run_progressive_series_evaluation(
    model: torch.nn.Module,
    dataset_for_series: Any,
    device: torch.device,
    class_names: List[str],
    input_length: int,
    progressive_start_frac: float,
    progressive_end_frac: float,
    progressive_num_steps: int,
    min_prefix_len: int,
) -> pd.DataFrame:
    model.eval()

    series_records: List[Dict[str, Any]] = []
    total_series = len(dataset_for_series.series_items)

    with torch.no_grad():
        for idx, item in enumerate(dataset_for_series.series_items, start=1):
            series_id = str(item["series_id"])
            x_full = np.asarray(item["signal"], dtype=np.float32).reshape(-1)
            y_true = item.get("label", None)
            transition_index = item.get("transition_index", None)

            LOGGER.info(
                "Progressive inference [%d/%d] | series_id=%s | length=%d",
                idx,
                total_series,
                series_id,
                len(x_full),
            )

            recs = create_progressive_series_records(
                model=model,
                device=device,
                x_full=x_full,
                series_id=series_id,
                y_true=y_true,
                transition_index=transition_index,
                input_length=input_length,
                class_names=class_names,
                progressive_start_frac=progressive_start_frac,
                progressive_end_frac=progressive_end_frac,
                progressive_num_steps=progressive_num_steps,
                min_prefix_len=min_prefix_len,
            )

            if len(recs) == 0:
                LOGGER.warning("No progressive records created for series_id=%s", series_id)
                continue

            series_records.extend(recs)

    return pd.DataFrame(series_records)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # read checkpoint metadata first for run naming
    ckpt = load_checkpoint_safely(checkpoint_path)
    ckpt_meta = extract_checkpoint_metadata(ckpt)
    model_name = infer_model_name_from_checkpoint(checkpoint_path, ckpt_meta)

    train_dataset_token = ckpt_meta.get("dataset", "unknown_train_dataset")
    monitor_metric = ckpt_meta.get("metric", "unknown_metric")
    input_length = int(ckpt_meta.get("input_length", ckpt_meta.get("seq_len", 500)))
    num_classes = int(ckpt_meta.get("num_classes", 3))
    class_names = resolve_class_names(num_classes=num_classes, ckpt_meta=ckpt_meta)

    run_name = (
        args.run_name
        if args.run_name is not None
        else f"{model_name}_{train_dataset_token}_to_{args.dataset}_{monitor_metric}"
    )

    run_dir = Path(args.results_root) / "testing" / run_name
    ensure_dir(run_dir)

    # create dedicated folder structure immediately
    output_dirs = get_testing_output_dirs(run_dir)

    # log file inside dedicated run folder
    log_file = output_dirs["logs_dir"] / "test.log"
    setup_run_logging(log_file=log_file, verbose=args.verbose)

    LOGGER.info("=" * 80)
    LOGGER.info("Starting testing run")
    LOGGER.info("=" * 80)

    device = resolve_device(args.device)

    LOGGER.info("Run name            : %s", run_name)
    LOGGER.info("Checkpoint          : %s", checkpoint_path)
    LOGGER.info("Model               : %s", model_name)
    LOGGER.info("Train dataset token : %s", train_dataset_token)
    LOGGER.info("Test dataset        : %s", args.dataset)
    LOGGER.info("Metric              : %s", monitor_metric)
    LOGGER.info("Input length        : %d", input_length)
    LOGGER.info("Num classes         : %d", num_classes)
    LOGGER.info("Class names         : %s", class_names)
    LOGGER.info("Device              : %s", device)
    LOGGER.info("Run directory       : %s", run_dir)
    LOGGER.info("Log file            : %s", log_file)

    model = build_model_from_checkpoint(
        checkpoint=ckpt,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        num_classes=num_classes,
        input_length=input_length,
        device=device,
    )
    model.eval()

    LOGGER.info("Loading test dataset...")
    loaded = load_test_dataset_for_inference(
        dataset_name=args.dataset,
        split=args.split,
        input_length=input_length,
        num_classes=num_classes,
    )

    window_dataset = loaded["window_dataset"]
    series_dataset = loaded["series_dataset"]

    loader = DataLoader(
        window_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    config_summary = summarise_run_config(
        checkpoint_path=str(checkpoint_path),
        run_name=run_name,
        model_name=model_name,
        train_dataset=train_dataset_token,
        test_dataset=args.dataset,
        metric=monitor_metric,
        split=args.split,
        input_length=input_length,
        num_classes=num_classes,
        class_names=class_names,
        device=str(device),
        batch_size=args.batch_size,
        progressive_start_frac=args.progressive_start_frac,
        progressive_end_frac=args.progressive_end_frac,
        progressive_num_steps=args.progressive_num_steps,
        min_prefix_len=args.min_prefix_len,
    )
    save_json(config_summary, run_dir / "run_config.json")

    LOGGER.info("-" * 80)
    LOGGER.info("[1/3] Sample-level evaluation")
    LOGGER.info("-" * 80)

    sample_eval = run_sample_level_evaluation(
        model=model,
        loader=loader,
        device=device,
        class_names=class_names,
    )

    metrics_dict = sample_eval["metrics"]
    summary_df = sample_eval["summary_df"]
    y_true = sample_eval["y_true"]
    y_pred = sample_eval["y_pred"]

    LOGGER.info("Sample-level metrics:")
    LOGGER.info(json.dumps(metrics_dict, indent=2))

    save_json(metrics_dict, run_dir / "metrics.json")
    save_numpy_confusion_inputs(
        y_true=y_true,
        y_pred=y_pred,
        save_dir=run_dir / "confusion_inputs",
    )

    if args.save_summary_csv:
        summary_df.to_csv(output_dirs["tables_dir"] / "classification_summary.csv", index=False)
        LOGGER.info("Saved classification summary: %s", output_dirs["tables_dir"] / "classification_summary.csv")

    LOGGER.info("-" * 80)
    LOGGER.info("[2/3] Progressive per-series evaluation")
    LOGGER.info("-" * 80)

    progressive_df = run_progressive_series_evaluation(
        model=model,
        dataset_for_series=series_dataset,
        device=device,
        class_names=class_names,
        input_length=input_length,
        progressive_start_frac=args.progressive_start_frac,
        progressive_end_frac=args.progressive_end_frac,
        progressive_num_steps=args.progressive_num_steps,
        min_prefix_len=args.min_prefix_len,
    )

    if progressive_df.empty:
        LOGGER.warning("No progressive predictions were generated.")
    else:
        # save combined progressive CSV at run root
        progressive_csv_path = run_dir / "all_series_progressive_predictions.csv"
        progressive_df.to_csv(progressive_csv_path, index=False)
        LOGGER.info("Saved combined progressive predictions: %s", progressive_csv_path)

        # save copy into dedicated tables folder too
        progressive_df.to_csv(output_dirs["tables_dir"] / "all_series_progressive_predictions.csv", index=False)

        # save final step per series
        final_step_df = (
            progressive_df
            .sort_values(["series_id", "reveal_index"])
            .groupby("series_id", as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )

        final_csv_path = run_dir / "final_series_predictions.csv"
        final_step_df.to_csv(final_csv_path, index=False)
        LOGGER.info("Saved final series predictions: %s", final_csv_path)

        final_step_df.to_csv(output_dirs["tables_dir"] / "final_series_predictions.csv", index=False)

        # optional per-series csv files
        if args.save_series_csv:
            series_csv_dir = run_dir / "series_predictions"
            ensure_dir(series_csv_dir)

            for series_id, series_df in progressive_df.groupby("series_id", sort=True):
                safe_name = str(series_id).replace("/", "_").replace("\\", "_").replace(" ", "_")
                series_df.sort_values("reveal_index").to_csv(series_csv_dir / f"{safe_name}.csv", index=False)

            LOGGER.info("Saved per-series progressive CSV files: %s", series_csv_dir)

    LOGGER.info("-" * 80)
    LOGGER.info("[3/3] Plot generation")
    LOGGER.info("-" * 80)

    if args.make_plots and not progressive_df.empty:
        plot_summary = run_all_testing_plots(
            run_dir=run_dir,
            class_names=class_names,
            progressive_csv_name="all_series_progressive_predictions.csv",
            final_predictions_csv_name="final_series_predictions.csv",
        )
        save_json(plot_summary, run_dir / "plot_summary.json")
        LOGGER.info("Plot summary: %s", plot_summary)
    else:
        LOGGER.info("Plot generation skipped.")

    LOGGER.info("=" * 80)
    LOGGER.info("Testing completed successfully")
    LOGGER.info("All outputs saved in: %s", run_dir)
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
