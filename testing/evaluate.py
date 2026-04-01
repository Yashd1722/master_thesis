"""
    testing/evaluate.py
    ===================
    Universal inference script — runs one trained model on all PANGAEA cores.

    What one run does:
        For each core in [MS21, MS66, 64PE406E1]:
            For each test sapropel in that core:
                1. Load checkpoint(s) for this model
                2. Run rolling window → compute variance + lag-1 AC
                3. Run model inference → get p_transition at each step
                4. Save predictions CSV → results/
                5. Save test metrics  → metrics/

    Called by test_array.sh with --model argument.

    Usage:
        python testing/evaluate.py --model cnn_lstm
        python testing/evaluate.py --model lstm --dataset ts_500
        python testing/evaluate.py --model cnn --dataset ts_1500

    Output files:
        results/{model}_{core}_{sapropel}_predictions.csv
        metrics/{model}_{core}_test_metrics.json
        test_logs/{model}_{core}_inference.log
    """

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models import get_model, list_models
from src.dataset_loader import load_config, get_dataset_info
from src.rolling_window import run_all_sapropels, RollingWindowResult

# =============================================================================
#  Logging
# =============================================================================

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"evaluate_{log_path.stem}")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# =============================================================================
#  Load all checkpoints for one model + one dataset
# =============================================================================

def load_checkpoints(
    model_name:   str,
    dataset_name: str,
    ts_len:       int,
    num_classes:  int,
    cfg:          dict,
    device:       torch.device,
    logger:       logging.Logger,
) -> List[torch.nn.Module]:
    """
    Load all checkpoints for (model_name, dataset_name) — both pad variants.

    Returns list of loaded models in eval mode.
    During inference we average probabilities across all loaded models.

    Files expected:
        checkpoints/{model}_{dataset}_v1_best.ckpt
        checkpoints/{model}_{dataset}_v2_best.ckpt
    """
    ckpt_dir  = REPO_ROOT / cfg["paths"]["checkpoints"]
    # Determine pad variants based on the model's training configuration.  In the
    # updated configuration, each model (cnn_lstm, lstm, cnn, sdml) defines its
    # own list of pad_variants.  We attempt to read this list; if the model
    # isn't explicitly listed or the key is missing, we default to [1].
    try:
        pad_variants = cfg["training"][model_name]["pad_variants"]
    except Exception:
        pad_variants = [1]

    models = []
    for v in pad_variants:
        ckpt_name = f"{model_name}_{dataset_name}_v{v}_best.ckpt"
        ckpt_path = ckpt_dir / ckpt_name

        if not ckpt_path.exists():
            logger.warning(f"  Checkpoint not found: {ckpt_name} — skipping")
            continue

        m = get_model(model_name, ts_len=ts_len, num_classes=num_classes)
        m.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        m.to(device).eval()
        models.append(m)
        logger.info(f"  Loaded: {ckpt_name}")

    if not models:
        raise RuntimeError(
            f"No checkpoints found for {model_name}/{dataset_name}.\n"
            f"Run training first: python training/train.py --model {model_name}"
        )

    logger.info(f"  Using {len(models)} checkpoint(s) for ensemble average")
    return models


# =============================================================================
#  Inference — ensemble average over all loaded checkpoints
# =============================================================================

@torch.no_grad()
def run_inference(
    models:   List[torch.nn.Module],
    dl_inputs: List[np.ndarray],
    device:   torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Run all rolling-window inputs through all checkpoints and average.

    Parameters
    ----------
    models     : list of loaded nn.Module in eval mode
    dl_inputs  : list of np.ndarray each shape (ts_len,)
    device     : torch device
    batch_size : inference batch size

    Returns
    -------
    probs : np.ndarray shape (n_steps, num_classes)
            Ensemble-averaged softmax probabilities.
    """
    # Stack inputs: (n_steps, ts_len)
    X = np.stack(dl_inputs, axis=0)                    # (N, ts_len)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, ts_len)

    all_model_probs = []

    for model in models:
        model_probs = []
        for i in range(0, len(X_t), batch_size):
            batch = X_t[i: i + batch_size].to(device)
            probs = model.predict_proba(batch).cpu().numpy()
            model_probs.append(probs)
        all_model_probs.append(np.concatenate(model_probs, axis=0))

    # Ensemble average: (n_models, N, num_classes) → (N, num_classes)
    return np.mean(all_model_probs, axis=0)


# =============================================================================
#  Save predictions CSV
# =============================================================================

def save_predictions(
    result:      RollingWindowResult,
    probs:       np.ndarray,
    class_names: List[str],
    model_name:  str,
    cfg:         dict,
    logger:      logging.Logger,
) -> Path:
    """
    Save rolling window results + model probabilities to CSV.

    Output: results/{model}_{core}_{sapropel}_predictions.csv
    Columns: position, age_kyr_bp, variance, lag1_ac,
             ktau_variance, ktau_lag1_ac,
             p_{class} for each class, p_transition
    """
    results_dir = REPO_ROOT / cfg["paths"]["results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build filename from config naming template
    fname = cfg["naming"]["predictions"].format(
        model    = model_name,
        core     = result.core_name,
        sapropel = result.sapropel_id,
    )
    out_path = results_dir / fname

    # Fill probs into result and convert to DataFrame
    result.dl_probs    = probs
    # p_transition = 1 - p_null (index of "null" class)
    if "null" in class_names:
        null_idx           = class_names.index("null")
        result.p_transition = 1.0 - probs[:, null_idx]
    else:
        # SDML binary: p_transition = p_pre_transition
        pt_idx             = class_names.index("pre_transition")
        result.p_transition = probs[:, pt_idx]

    df = result.to_dataframe(class_names)
    df.to_csv(out_path, index=False)
    logger.info(f"  Saved predictions → {fname}")
    return out_path


# =============================================================================
#  Main — loops over all PANGAEA cores
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run inference on all PANGAEA cores for one model."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list_models(),
        help="Model to evaluate: cnn_lstm | lstm | cnn"
    )
    parser.add_argument(
        "--dataset", type=str, default="ts_500",
        choices=["ts_500", "ts_1500"],
        help="Which training dataset checkpoint to load (default: ts_500)"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml"
    )
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cores  = cfg["slurm"]["test_cores"]   # [MS21, MS66, 64PE406E1]

    # ── Dataset info (ts_len, num_classes, class_names) ───────────────────────
    ds_info     = get_dataset_info(args.dataset, cfg)
    ts_len      = ds_info["ts_length"]
    num_classes = ds_info["num_classes"]
    class_names = ds_info["class_names"]

    # ── Root log ──────────────────────────────────────────────────────────────
    log_dir      = REPO_ROOT / cfg["paths"]["test_logs"]
    log_dir.mkdir(parents=True, exist_ok=True)
    root_log     = log_dir / f"{args.model}_{args.dataset}_inference.log"
    logger       = setup_logging(root_log)

    logger.info(f"Model      : {args.model}")
    logger.info(f"Dataset    : {args.dataset}")
    logger.info(f"ts_len     : {ts_len}")
    logger.info(f"num_classes: {num_classes}  {class_names}")
    logger.info(f"Cores      : {cores}")
    logger.info(f"Device     : {device}")

    # ── Load checkpoints once (reused across all cores) ───────────────────────
    models = load_checkpoints(
        model_name   = args.model,
        dataset_name = args.dataset,
        ts_len       = ts_len,
        num_classes  = num_classes,
        cfg          = cfg,
        device       = device,
        logger       = logger,
    )

    # ── Per-core metrics summary ───────────────────────────────────────────────
    all_test_metrics = {}

    for core_name in cores:
        core_log = log_dir / f"{args.model}_{core_name}_inference.log"
        c_logger = setup_logging(core_log)
        c_logger.info(f"Core: {core_name}")

        t0 = time.time()

        # Run rolling window for all test sapropels in this core
        sapropel_results = run_all_sapropels(core_name, cfg, ts_len)

        if not sapropel_results:
            c_logger.warning(
                f"No test sapropels for {core_name} — "
                f"run pangea_cleaner.py first"
            )
            continue

        core_metrics = {}

        for sap_id, result in sapropel_results:
            c_logger.info(f"\n  Processing {core_name}/{sap_id}")

            # Run model inference
            probs = run_inference(models, result.dl_inputs, device)

            # Save predictions CSV
            save_predictions(
                result      = result,
                probs       = probs,
                class_names = class_names,
                model_name  = args.model,
                cfg         = cfg,
                logger      = c_logger,
            )

            # Per-sapropel metrics (for compute_metrics.py)
            core_metrics[sap_id] = {
                "n_series":     result.n_series,
                "n_steps":      len(result.positions),
                "ktau_variance": result.ktau_variance,
                "ktau_lag1_ac":  result.ktau_lag1_ac,
                "mean_p_transition": float(result.p_transition.mean())
                    if result.p_transition is not None else None,
            }

        elapsed = time.time() - t0
        c_logger.info(f"\n  {core_name} done in {elapsed:.1f}s")
        all_test_metrics[core_name] = core_metrics

        # Save per-core metrics to metrics/
        met_dir  = REPO_ROOT / cfg["paths"]["metrics"]
        met_dir.mkdir(parents=True, exist_ok=True)
        met_fname = cfg["naming"]["test_metrics"].format(
            model=args.model, core=core_name
        )
        met_path  = met_dir / met_fname
        with open(met_path, "w") as f:
            json.dump({
                "model":    args.model,
                "dataset":  args.dataset,
                "core":     core_name,
                "sapropels": core_metrics,
            }, f, indent=2)
        logger.info(f"  Metrics saved → {met_fname}")

    logger.info(f"\nAll cores done. Results in {REPO_ROOT / cfg['paths']['results']}")
    logger.info(
        "Next: python testing/compute_metrics.py "
        f"--model {args.model} --dataset {args.dataset}"
    )


if __name__ == "__main__":
    main()
