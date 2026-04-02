"""
testing/evaluate.py
===================
Universal inference: one model × all PANGAEA cores × all elements × both segments.

For each (core, sapropel, element, segment_type):
  1. Load checkpoint(s) for model + dataset
  2. Run rolling window → variance + lag-1 AC
  3. Run model inference → p_transition at each step
  4. Save: results/{model}_{core}_{sapropel}_{element}_{segment_type}.csv

Called by test_array.sh:
  python testing/evaluate.py --model cnn_lstm --dataset ts_500

Output per run:
  results/{model}_{core}_{sapropel}_{element}_forced.csv
  results/{model}_{core}_{sapropel}_{element}_neutral.csv
  metrics/{model}_{core}_test_metrics.json
  test_logs/{model}_{core}_inference.log
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models import get_model, list_models
from src.dataset_loader import load_config, get_dataset_info
from src.rolling_window import run_all_sapropels, ELEMENTS, PRIMARY_ELEMENT


# =============================================================================
#  Logging
# =============================================================================

def setup_log(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    name   = f"eval_{log_path.stem}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt    = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    fh     = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch     = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# =============================================================================
#  Load checkpoints
# =============================================================================

def load_checkpoints(model_name: str, dataset_name: str,
                     ts_len: int, num_classes: int,
                     cfg: dict, device: torch.device,
                     logger: logging.Logger) -> List[torch.nn.Module]:
    """
    Load all checkpoints for (model, dataset).
    For cnn_lstm: variants [1,2] → 2 checkpoints.
    For lstm/cnn: variant [1]   → 1 checkpoint.
    Returns list of models in eval mode — predictions are averaged.
    """
    ckpt_dir     = REPO_ROOT / cfg["paths"]["checkpoints"]
    pad_variants = cfg["training"][model_name]["pad_variants"]
    models       = []

    for v in pad_variants:
        name  = f"{model_name}_{dataset_name}_v{v}_best.ckpt"
        fpath = ckpt_dir / name
        if not fpath.exists():
            logger.warning(f"  Checkpoint not found: {name}")
            continue
        m = get_model(model_name, ts_len=ts_len, num_classes=num_classes)
        m.load_state_dict(torch.load(fpath, map_location=device, weights_only=True))
        m.to(device).eval()
        models.append(m)
        logger.info(f"  Loaded: {name}")

    if not models:
        raise RuntimeError(
            f"No checkpoints found for {model_name}/{dataset_name}.\n"
            f"Run training first: python training/train.py --model {model_name}"
        )
    return models


# =============================================================================
#  Inference — ensemble average
# =============================================================================

@torch.no_grad()
def run_inference(models: List, dl_inputs: List[np.ndarray],
                  device: torch.device, batch_size: int = 64) -> np.ndarray:
    """
    Run all dl_inputs through all checkpoints, return averaged softmax probs.
    Returns np.ndarray shape (n_steps, n_classes).
    """
    X   = np.stack(dl_inputs, axis=0)                        # (N, ts_len)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, ts_len)

    all_probs = []
    for model in models:
        probs_m = []
        for i in range(0, len(X_t), batch_size):
            batch = X_t[i: i + batch_size].to(device)
            probs_m.append(model.predict_proba(batch).cpu().numpy())
        all_probs.append(np.concatenate(probs_m, axis=0))

    return np.mean(all_probs, axis=0)   # (N, n_classes)


# =============================================================================
#  Save predictions CSV
# =============================================================================

def save_predictions(result, probs: np.ndarray, class_names: List[str],
                     model_name: str, cfg: dict,
                     logger: logging.Logger) -> Path:
    """
    Save rolling window results + model probs to CSV.
    Filename: {model}_{core}_{sapropel}_{element}_{segment_type}.csv
    """
    results_dir = REPO_ROOT / cfg["paths"]["results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    fname = (
        f"{model_name}_{result.core_name}_{result.sapropel_id}"
        f"_{result.element}_{result.segment_type}.csv"
    )

    result.dl_probs = probs

    # p_transition = 1 - p_null for 4-class, p_pre_transition for 2-class
    if "null" in class_names:
        result.p_transition = 1.0 - probs[:, class_names.index("null")]
    elif "pre_transition" in class_names:
        result.p_transition = probs[:, class_names.index("pre_transition")]
    else:
        result.p_transition = probs[:, -1]

    df = result.to_dataframe(class_names)
    df.to_csv(results_dir / fname, index=False)
    logger.info(f"  Saved → {fname}")
    return results_dir / fname


# =============================================================================
#  Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, choices=list_models())
    parser.add_argument("--dataset", default="ts_500",
                        choices=["ts_500", "ts_1500"])
    parser.add_argument("--config",  default="config.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cores  = cfg["slurm"]["test_cores"]

    ds_info     = get_dataset_info(args.dataset, cfg)
    ts_len      = ds_info["ts_length"]
    num_classes = ds_info["num_classes"]
    class_names = ds_info["class_names"]

    # Root log
    log_dir  = REPO_ROOT / cfg["paths"]["test_logs"]
    log_dir.mkdir(parents=True, exist_ok=True)
    root_log = log_dir / f"{args.model}_{args.dataset}_inference.log"
    logger   = setup_log(root_log)

    logger.info(f"Model   : {args.model}")
    logger.info(f"Dataset : {args.dataset}  ts_len={ts_len}  classes={class_names}")
    logger.info(f"Cores   : {cores}")
    logger.info(f"Elements: {ELEMENTS}")
    logger.info(f"Device  : {device}")

    # Load checkpoints once — reused across all cores
    models = load_checkpoints(
        args.model, args.dataset, ts_len, num_classes,
        cfg, device, logger
    )

    all_metrics = {}

    for core_name in cores:
        t0       = time.time()
        core_log = log_dir / f"{args.model}_{core_name}_inference.log"
        c_log    = setup_log(core_log)
        c_log.info(f"Core: {core_name}")

        # Run rolling window — all sapropels × all elements × forced + neutral
        sapropel_results = run_all_sapropels(core_name, cfg, ts_len)

        if not sapropel_results:
            c_log.warning(
                f"No results for {core_name}. "
                "Run: python src/pangea_cleaner.py first."
            )
            continue

        core_metrics = {}

        for sap_id, sap_res in sapropel_results.items():
            c_log.info(f"\n  Sapropel: {sap_id}")
            sap_metrics = {}

            for element, elem_res in sap_res.items():
                for seg_type, result in elem_res.items():
                    c_log.info(
                        f"    {element}/{seg_type}: "
                        f"N={result.n_series}  steps={len(result.dl_inputs)}"
                    )

                    # Run model inference
                    probs = run_inference(models, result.dl_inputs, device)

                    # Save predictions CSV
                    save_predictions(
                        result, probs, class_names,
                        args.model, cfg, c_log
                    )

                sap_metrics[element] = {
                    "n_forced":  len(sap_res[element].get("forced",
                                  type('', (), {'positions': []})()).positions
                                     if hasattr(sap_res[element].get("forced",
                                  None), "positions") else []),
                    "n_neutral": len(sap_res[element].get("neutral",
                                  type('', (), {'positions': []})()).positions
                                     if hasattr(sap_res[element].get("neutral",
                                  None), "positions") else []),
                }

            core_metrics[sap_id] = sap_metrics

        elapsed = time.time() - t0
        c_log.info(f"\n  {core_name} done in {elapsed:.1f}s")
        all_metrics[core_name] = core_metrics

        # Save per-core metrics
        met_dir  = REPO_ROOT / cfg["paths"]["metrics"]
        met_dir.mkdir(parents=True, exist_ok=True)
        met_fname = cfg["naming"]["test_metrics"].format(
            model=args.model, core=core_name
        )
        with open(met_dir / met_fname, "w") as f:
            json.dump({
                "model":    args.model,
                "dataset":  args.dataset,
                "core":     core_name,
                "elements": ELEMENTS,
                "sapropels": core_metrics,
            }, f, indent=2)
        logger.info(f"  Metrics → {met_fname}")

    logger.info(f"\nAll cores done.")
    logger.info(
        f"Next: python testing/compute_metrics.py "
        f"--model {args.model} --dataset {args.dataset}"
    )


if __name__ == "__main__":
    main()
