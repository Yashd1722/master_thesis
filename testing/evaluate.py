"""
testing/evaluate.py
===================
Universal evaluation script.

--dataset : which checkpoint to load (ts_500 | ts_1500)
--target  : what data to evaluate on
              pangaea  → PANGAEA cores (rolling window, all elements)
              ts_500   → Zenodo ts_500 test split (accuracy / F1 / confusion matrix)
              ts_1500  → Zenodo ts_1500 test split

Usage:
  # Thesis workflow: checkpoint trained on ts_500, tested on PANGAEA
  python testing/evaluate.py --model cnn_lstm --dataset ts_500 --target pangaea

  # Sanity check: test on the same Zenodo distribution as training
  python testing/evaluate.py --model cnn_lstm --dataset ts_500 --target ts_500

  # Cross-dataset: ts_500 checkpoint tested on ts_1500 distribution
  python testing/evaluate.py --model cnn_lstm --dataset ts_500 --target ts_1500

  # SVM: always tested on PANGAEA (trained on surrogates, no ts_500 target)
  python testing/evaluate.py --model svm --dataset ts_500 --target pangaea
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models import get_model, list_models, is_sklearn_model
from src.dataset_loader import load_config, get_dataset_info, get_dataloader
from src.rolling_window import run_all_sapropels, ELEMENTS


# =============================================================================
#  Logging
# =============================================================================

def setup_log(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"eval_{log_path.stem}")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for handler in [logging.FileHandler(log_path, mode="w"),
                    logging.StreamHandler(sys.stdout)]:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


# =============================================================================
#  Load checkpoints
# =============================================================================

def load_checkpoints(model_name: str, dataset_name: str,
                     ts_len: int, num_classes: int,
                     cfg: dict, device: torch.device,
                     logger: logging.Logger) -> List:
    """
    Load model checkpoints.
    PyTorch: one per pad variant → ensemble averaged at inference.
    SVM    : one .pkl per core  → dataset_name is the core name.
    """
    ckpt_dir = REPO_ROOT / cfg["paths"]["checkpoints"]

    if is_sklearn_model(model_name):
        from models.svm_ews import SVMClassifier
        pkl = ckpt_dir / f"{model_name}_{dataset_name}_best.pkl"
        if not pkl.exists():
            raise RuntimeError(
                f"SVM checkpoint not found: {pkl.name}\n"
                f"Run: python training/train.py --model svm"
            )
        m = SVMClassifier.load(pkl)
        logger.info(f"  Loaded: {pkl.name}")
        return [m]

    pad_variants = cfg["training"][model_name]["pad_variants"]
    models = []
    for v in pad_variants:
        name  = f"{model_name}_{dataset_name}_v{v}_best.ckpt"
        fpath = ckpt_dir / name
        if not fpath.exists():
            logger.warning(f"  Not found: {name}")
            continue
        m = get_model(model_name, ts_len=ts_len, num_classes=num_classes)
        m.load_state_dict(torch.load(fpath, map_location=device,
                                     weights_only=True))
        m.to(device).eval()
        models.append(m)
        logger.info(f"  Loaded: {name}")

    if not models:
        raise RuntimeError(
            f"No checkpoints for {model_name}/{dataset_name}.\n"
            f"Run: python training/train.py --model {model_name}"
        )
    return models


# =============================================================================
#  Inference helpers
# =============================================================================

@torch.no_grad()
def run_inference(models: List, dl_inputs: List[np.ndarray],
                  device: torch.device, batch_size: int = 64,
                  ews_features: np.ndarray = None) -> np.ndarray:
    """
    Ensemble-averaged inference for PyTorch or SVM models.
    Returns (n_steps, n_classes) softmax probabilities.
    """
    from models.svm_ews import SVMClassifier
    all_probs = []

    for model in models:
        if isinstance(model, SVMClassifier):
            if ews_features is None:
                raise RuntimeError("SVM inference requires ews_features")
            all_probs.append(model.predict_proba_numpy(ews_features))
        else:
            X   = np.stack(dl_inputs)
            X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
            out = []
            for i in range(0, len(X_t), batch_size):
                out.append(model.predict_proba(
                    X_t[i:i+batch_size].to(device)).cpu().numpy())
            all_probs.append(np.concatenate(out))

    return np.mean(all_probs, axis=0)


def build_ews_features(result) -> np.ndarray:
    """Build (n_steps, 4) EWS feature matrix for SVM from rolling window result."""
    from models.svm_ews import SVMClassifier
    n = len(result.positions)
    return np.stack([
        SVMClassifier.extract_features(
            result.variance[max(0, i-1):i+1],
            result.lag1_ac[max(0, i-1):i+1],
            result.ktau_variance,
            result.ktau_lag1_ac,
        )
        for i in range(n)
    ])


def save_predictions(result, probs: np.ndarray,
                     class_names: List[str],
                     model_name: str, cfg: dict,
                     logger: logging.Logger) -> Path:
    """Save prediction CSV to results/."""
    results_dir = REPO_ROOT / cfg["paths"]["results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    fname = (f"{model_name}_{result.core_name}_{result.sapropel_id}"
             f"_{result.element}_{result.segment_type}.csv")

    result.dl_probs = probs
    if "null" in class_names:
        result.p_transition = 1.0 - probs[:, class_names.index("null")]
    elif "pre_transition" in class_names:
        result.p_transition = probs[:, class_names.index("pre_transition")]
    else:
        result.p_transition = probs[:, -1]

    result.to_dataframe(class_names).to_csv(results_dir / fname, index=False)
    logger.info(f"  Saved → {fname}")
    return results_dir / fname


# =============================================================================
#  Target: Zenodo test split (ts_500 or ts_1500)
# =============================================================================

@torch.no_grad()
def evaluate_zenodo(model_name: str, dataset_name: str, target: str,
                    cfg: dict, device: torch.device,
                    logger: logging.Logger) -> dict:
    """
    Evaluate model on Zenodo (ts_500 or ts_1500) test split.
    Reports accuracy, macro F1, and confusion matrix per class.
    Useful for sanity check and cross-dataset generalisation.
    """
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    ds_info     = get_dataset_info(target, cfg)
    ts_len      = ds_info["ts_length"]
    num_classes = ds_info["num_classes"]
    class_names = ds_info["class_names"]

    logger.info(f"  Target : {target}  ts_len={ts_len}  classes={class_names}")

    models = load_checkpoints(model_name, dataset_name, ts_len,
                               num_classes, cfg, device, logger)

    loader = get_dataloader(target, "test", cfg, num_workers=0)
    logger.info(f"  Test batches: {len(loader)}")

    all_preds, all_labels = [], []

    for x, y in loader:
        x = x.to(device)
        probs = np.mean([
            m.predict_proba(x).cpu().numpy() for m in models
        ], axis=0)
        all_preds.extend(probs.argmax(axis=1).tolist())
        all_labels.extend(y.tolist())

    acc  = float(accuracy_score(all_labels, all_preds))
    f1   = float(f1_score(all_labels, all_preds, average="macro",
                           zero_division=0))
    cm   = confusion_matrix(all_labels, all_preds).tolist()

    logger.info(f"  Accuracy : {acc:.4f}")
    logger.info(f"  Macro F1 : {f1:.4f}")

    metrics = {
        "model":             model_name,
        "checkpoint":        dataset_name,
        "target":            target,
        "num_classes":       num_classes,
        "class_names":       class_names,
        "accuracy":          round(acc, 4),
        "macro_f1":          round(f1,  4),
        "confusion_matrix":  cm,
    }

    met_dir  = REPO_ROOT / cfg["paths"]["metrics"]
    met_dir.mkdir(parents=True, exist_ok=True)
    met_path = met_dir / f"{model_name}_{dataset_name}_on_{target}_metrics.json"
    with open(met_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Metrics → {met_path.name}")

    return metrics


# =============================================================================
#  Target: PANGAEA
# =============================================================================

def evaluate_pangaea(model_name: str, dataset_name: str,
                     cfg: dict, device: torch.device,
                     logger: logging.Logger) -> dict:
    """
    Evaluate model on all PANGAEA cores (the thesis main result).
    Runs rolling window on each core × sapropel × element × segment.
    Saves prediction CSVs to results/.
    """
    cores = cfg["slurm"]["test_cores"]

    ds_info     = get_dataset_info(dataset_name, cfg)
    ts_len      = ds_info["ts_length"]
    num_classes = ds_info["num_classes"]
    class_names = ds_info["class_names"]

    # PyTorch: load once, reuse across cores
    # SVM    : load per-core checkpoint inside the loop
    pytorch_models = None
    if not is_sklearn_model(model_name):
        pytorch_models = load_checkpoints(
            model_name, dataset_name, ts_len,
            num_classes, cfg, device, logger
        )

    all_metrics = {}

    for core_name in cores:
        t0      = time.time()
        log_dir = REPO_ROOT / cfg["paths"]["test_logs"]
        c_log   = setup_log(log_dir / f"{model_name}_{core_name}_inference.log")
        c_log.info(f"Core: {core_name}")

        # SVM: per-core checkpoint + binary class names
        if is_sklearn_model(model_name):
            try:
                core_models = load_checkpoints(
                    model_name, core_name, ts_len,
                    num_classes, cfg, device, c_log
                )
            except RuntimeError as e:
                c_log.warning(str(e))
                continue
            core_class_names = ["neutral", "pre_transition"]
        else:
            core_models      = pytorch_models
            core_class_names = class_names

        sap_results = run_all_sapropels(core_name, cfg, ts_len)
        if not sap_results:
            c_log.warning(f"No sapropel data. Run: python src/pangea_cleaner.py")
            continue

        core_metrics = {}
        for sap_id, sap_res in sap_results.items():
            c_log.info(f"\n  Sapropel: {sap_id}")
            for element, elem_res in sap_res.items():
                for seg_type, result in elem_res.items():
                    c_log.info(f"    {element}/{seg_type}: "
                               f"N={result.n_series}  "
                               f"steps={len(result.dl_inputs)}")

                    ews_features = (build_ews_features(result)
                                    if is_sklearn_model(model_name)
                                    else None)

                    probs = run_inference(core_models, result.dl_inputs,
                                         device, ews_features=ews_features)

                    save_predictions(result, probs, core_class_names,
                                     model_name, cfg, c_log)

            core_metrics[sap_id] = {"elements": list(sap_res.keys())}

        elapsed = time.time() - t0
        c_log.info(f"\n  {core_name} done in {elapsed:.1f}s")
        all_metrics[core_name] = core_metrics

        met_dir   = REPO_ROOT / cfg["paths"]["metrics"]
        met_fname = cfg["naming"]["test_metrics"].format(
            model=model_name, core=core_name
        )
        with open(met_dir / met_fname, "w") as f:
            json.dump({
                "model":     model_name,
                "dataset":   dataset_name,
                "core":      core_name,
                "sapropels": core_metrics,
            }, f, indent=2)
        logger.info(f"  Metrics → {met_fname}")

    return all_metrics


# =============================================================================
#  Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Universal evaluation — PANGAEA or Zenodo test split."
    )
    parser.add_argument("--model",   required=True, choices=list_models())
    parser.add_argument("--dataset", required=True,
                        choices=["ts_500", "ts_1500"],
                        help="Checkpoint to load (trained on this dataset)")
    parser.add_argument("--target",  default="pangaea",
                        choices=["pangaea", "ts_500", "ts_1500"],
                        help="Data to evaluate on (default: pangaea)")
    parser.add_argument("--config",  default="config.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = REPO_ROOT / cfg["paths"]["test_logs"]
    log_dir.mkdir(parents=True, exist_ok=True)
    logger  = setup_log(
        log_dir / f"{args.model}_{args.dataset}_on_{args.target}.log"
    )

    logger.info(f"Model   : {args.model}")
    logger.info(f"Dataset : {args.dataset}  (checkpoint source)")
    logger.info(f"Target  : {args.target}   (evaluation data)")
    logger.info(f"Device  : {device}")

    if args.target == "pangaea":
        if is_sklearn_model(args.model):
            logger.info("SVM → evaluating on PANGAEA (per-core checkpoints)")
        evaluate_pangaea(args.model, args.dataset, cfg, device, logger)

    else:
        # Zenodo test split — does not apply to SVM
        if is_sklearn_model(args.model):
            logger.error("SVM cannot be evaluated on Zenodo splits. "
                         "Use --target pangaea for SVM.")
            sys.exit(1)
        evaluate_zenodo(args.model, args.dataset, args.target,
                        cfg, device, logger)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
