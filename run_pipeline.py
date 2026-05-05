"""
run_pipeline.py — universal EWS pipeline.

Stages:
  cache       build numpy cache from Zenodo CSVs (run once)
  preprocess  clean PANGAEA XRF data + generate AAFT null series
  train       train models on ts_500/ts_1500
  evaluate    run inference on PANGAEA cores
  metrics     compute ROC/AUC
  figures     generate all figures
  test        evaluate + metrics + figures
  all         cache + preprocess + train + test

Usage:
  python run_pipeline.py --stage all    --model all  --dataset ts_500
  python run_pipeline.py --stage train  --model cnn  --dataset ts_1500
  python run_pipeline.py --stage test   --model lstm --dataset ts_500
  python run_pipeline.py --stage figures --fig3_only

Adding a new model:
  1. Create models/my_model.py with MODEL_NAME / MODEL_CLASS / IS_SKLEARN
  2. Add the name to pipeline.default_models in config.yaml
  3. Run this script — no other changes needed
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import yaml

REPO_ROOT = Path(__file__).resolve().parent


# =============================================================================
#  Config
# =============================================================================

def load_config(path: str = "config.yaml") -> dict:
    for p in [Path(path), REPO_ROOT / "config.yaml"]:
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("config.yaml not found")


# =============================================================================
#  SLURM detection
# =============================================================================

def slurm_available() -> bool:
    return shutil.which("sbatch") is not None


def submit_slurm(script: str, wait: bool = True,
                 poll: int = 30) -> int:
    result = subprocess.run(
        ["sbatch", script],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[ERROR] sbatch failed:\n{result.stderr}")
        return -1

    # Parse job ID from "Submitted batch job 12345"
    job_id = int(result.stdout.strip().split()[-1])
    print(f"  Submitted SLURM job {job_id}")

    if not wait:
        return job_id

    # Poll until job finishes
    print(f"  Waiting for job {job_id} ", end="", flush=True)
    while True:
        time.sleep(poll)
        r = subprocess.run(
            ["squeue", "-j", str(job_id), "-h"],
            capture_output=True, text=True
        )
        if not r.stdout.strip():
            print(" done.")
            break
        print(".", end="", flush=True)

    return job_id


def run_local(cmd: List[str], label: str = "") -> int:
    if label:
        print(f"\n  Running: {label}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


# =============================================================================
#  Skip-if-done checks
# =============================================================================

def checkpoint_exists(model: str, dataset: str, cfg: dict) -> bool:
    ckpt_dir     = REPO_ROOT / cfg["paths"]["checkpoints"]
    pad_variants = cfg["training"][model]["pad_variants"]
    return all(
        (ckpt_dir / f"{model}_{dataset}_v{v}_best.ckpt").exists()
        for v in pad_variants
    )


def predictions_exist(model: str, dataset: str, cfg: dict) -> bool:
    results_dir = REPO_ROOT / cfg["paths"]["results"]
    cores       = cfg["slurm"]["test_cores"]
    for core in cores:
        saps = [s for s in cfg["pangaea"]["cores"][core]["sapropels"]
                if s["role"] == "test"]
        for sap in saps:
            fname = f"{model}_{core}_{sap['id']}_Mo_forced.csv"
            if not (results_dir / fname).exists():
                return False
    return True


def cache_exists(dataset: str, cfg: dict) -> bool:
    ds_cfg   = cfg["datasets"][dataset]
    base_dir = REPO_ROOT / cfg["paths"][ds_cfg["path_key"]] / "combined"
    return (
        (base_dir / "cache_residuals.npy").exists() and
        (base_dir / "cache_labels.npy").exists()
    )


def pangaea_processed(cfg: dict) -> bool:
    clean_dir = REPO_ROOT / cfg["paths"]["pangaea_clean"]
    for core in cfg["slurm"]["test_cores"]:
        saps = [s for s in cfg["pangaea"]["cores"][core]["sapropels"]
                if s["role"] == "test"]
        for sap in saps:
            if not (clean_dir / core / f"{core}_{sap['id']}_forced.csv").exists():
                return False
    return True


# =============================================================================
#  Individual stages
# =============================================================================

def stage_cache(datasets: List[str], cfg: dict, skip: bool):
    print("\n=== STAGE: cache ===")
    for ds in datasets:
        if skip and cache_exists(ds, cfg):
            print(f"  [skip] cache exists: {ds}")
            continue
        rc = run_local(
            [sys.executable, "src/build_cache.py", "--dataset", ds],
            label=f"build cache {ds}"
        )
        if rc != 0:
            sys.exit(f"Cache build failed for {ds}")


def stage_preprocess(cfg: dict, skip: bool):
    print("\n=== STAGE: preprocess ===")
    if skip and pangaea_processed(cfg):
        print("  [skip] PANGAEA segments already exist")
        return
    rc = run_local(
        [sys.executable, "src/pangea_cleaner.py"],
        label="pangea_cleaner"
    )
    if rc != 0:
        sys.exit("PANGAEA preprocessing failed")


def stage_train(models: List[str], datasets: List[str],
                cfg: dict, skip: bool, parallel: bool):
    print("\n=== STAGE: train ===")

    from models import is_sklearn_model

    # Split into PyTorch and SVM models
    pytorch_models = [m for m in models if not is_sklearn_model(m)]
    sklearn_models = [m for m in models if     is_sklearn_model(m)]

    # ── SLURM: submit array job for PyTorch models ────────────────────────────
    if slurm_available() and pytorch_models:
        print("  SLURM detected — submitting train array job")
        slurm_script = REPO_ROOT / cfg["pipeline"]["slurm"]["train_script"]
        submit_slurm(str(slurm_script), wait=True,
                     poll=cfg["pipeline"]["slurm"]["poll_interval"])
    elif pytorch_models:
        # Local PyTorch training
        jobs = []
        for model in pytorch_models:
            for ds in datasets:
                if skip and checkpoint_exists(model, ds, cfg):
                    print(f"  [skip] checkpoint exists: {model}/{ds}")
                    continue
                jobs.append((model, ds))

        def _train_pytorch(model, ds):
            return run_local(
                [sys.executable, "training/train.py",
                 "--model", model, "--config", "config.yaml"],
                label=f"train {model}/{ds}"
            )

        if jobs:
            if parallel:
                try:
                    from joblib import Parallel, delayed
                    n_jobs  = min(cfg["pipeline"].get("n_jobs", 3), len(jobs))
                    results = Parallel(n_jobs=n_jobs)(
                        delayed(_train_pytorch)(m, d) for m, d in jobs
                    )
                    if any(r != 0 for r in results):
                        sys.exit("One or more PyTorch training jobs failed")
                except ImportError:
                    for m, d in jobs:
                        if _train_pytorch(m, d) != 0:
                            sys.exit(f"Training failed: {m}/{d}")
            else:
                for m, d in jobs:
                    if _train_pytorch(m, d) != 0:
                        sys.exit(f"Training failed: {m}/{d}")
        else:
            print("  All PyTorch checkpoints exist")

    # ── SVM: always runs locally (sklearn, fast) ──────────────────────────────
    if sklearn_models:
        print("\n  Training SVM on PANGAEA surrogates")
        rc = run_local(
            [sys.executable, "training/train_svm.py", "--config", "config.yaml"],
            label="train_svm"
        )
        if rc != 0:
            sys.exit("SVM training failed")


def stage_evaluate(models: List[str], dataset: str,
                   cfg: dict, skip: bool, parallel: bool,
                   target: str = "pangaea"):
    print("\n=== STAGE: evaluate ===")

    if slurm_available():
        print("  SLURM detected — submitting test array job")
        slurm_script = REPO_ROOT / cfg["pipeline"]["slurm"]["test_script"]
        submit_slurm(str(slurm_script), wait=True,
                     poll=cfg["pipeline"]["slurm"]["poll_interval"])
        return

    jobs = []
    for model in models:
        if skip and predictions_exist(model, dataset, cfg):
            print(f"  [skip] predictions exist: {model}/{dataset}")
            continue
        jobs.append(model)

    if not jobs:
        print("  All predictions exist — nothing to evaluate")
        return

    def _eval_one(model):
        print(f"  Evaluating: {model} / checkpoint={dataset} / target={target}")
        return run_local(
            [sys.executable, "testing/evaluate.py",
             "--model",   model,
             "--dataset", dataset,
             "--target",  target,
             "--config",  "config.yaml"]
        )

    if parallel:
        try:
            from joblib import Parallel, delayed
            n_jobs = min(cfg["pipeline"].get("n_jobs", 3), len(jobs))
            results = Parallel(n_jobs=n_jobs)(
                delayed(_eval_one)(m) for m in jobs
            )
            if any(r != 0 for r in results):
                sys.exit("One or more evaluation jobs failed")
        except ImportError:
            for m in jobs:
                if _eval_one(m) != 0:
                    sys.exit(f"Evaluation failed: {m}")
    else:
        for m in jobs:
            if _eval_one(m) != 0:
                sys.exit(f"Evaluation failed: {m}")


def stage_metrics(models: List[str], dataset: str, cfg: dict):
    print("\n=== STAGE: metrics ===")
    rc = run_local(
        [sys.executable, "testing/compute_metrics.py",
         "--all", "--dataset", dataset, "--config", "config.yaml"],
        label="compute_metrics"
    )
    if rc != 0:
        sys.exit("Metrics computation failed")


def stage_figures(models: List[str], dataset: str,
                  cfg: dict, extra_args: List[str]):
    print("\n=== STAGE: figures ===")
    model_arg = "all" if set(models) >= set(cfg["pipeline"]["default_models"]) \
                else models[0]
    cmd = [
        sys.executable, "testing/plot_figures.py",
        "--model",   model_arg,
        "--dataset", dataset,
        "--config",  "config.yaml",
    ] + extra_args
    rc = run_local(cmd, label="plot_figures")
    if rc != 0:
        sys.exit("Figure generation failed")


# =============================================================================
#  Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Universal EWS pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--stage", required=True,
        choices=["cache","preprocess","train","evaluate","metrics",
                 "figures","test","all"],
        help=(
            "cache       build numpy cache\n"
            "preprocess  clean PANGAEA + generate null\n"
            "train       train models\n"
            "evaluate    run inference on PANGAEA\n"
            "metrics     compute ROC/AUC\n"
            "figures     generate all figures\n"
            "test        evaluate + metrics + figures\n"
            "all         everything end-to-end"
        )
    )
    parser.add_argument(
        "--model", default="all",
        help="Model name or 'all' (default: all)"
    )
    parser.add_argument(
        "--dataset", default=None,
        choices=["ts_500", "ts_1500", "both"],
        help="Training dataset (default: from config)"
    )
    parser.add_argument(
        "--no_skip", action="store_true",
        help="Disable skip-if-done (re-run everything)"
    )
    parser.add_argument(
        "--no_parallel", action="store_true",
        help="Disable parallel model execution"
    )
    parser.add_argument(
        "--target", default="pangaea",
        choices=["pangaea", "ts_500", "ts_1500"],
        help="Evaluation target: pangaea (thesis) or Zenodo split (default: pangaea)"
    )
    parser.add_argument(
        "--fig3_only", action="store_true",
        help="Generate Figure 3 only (no model needed)"
    )
    parser.add_argument(
        "--fig2", action="store_true",
        help="Generate Figure 2 only"
    )
    parser.add_argument(
        "--fig4", action="store_true",
        help="Generate Figure 4 only"
    )
    parser.add_argument(
        "--fig5", action="store_true",
        help="Generate Figure 5 only"
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    pipe_cfg = cfg.get("pipeline", {})

    # Resolve models
    from models import list_models
    all_models = list_models()
    if args.model == "all":
        models = pipe_cfg.get("default_models", all_models)
        # Only include pytorch models for train/eval
        from models import is_sklearn_model
        models = [m for m in models if not is_sklearn_model(m)]
    else:
        if args.model not in all_models:
            sys.exit(f"Unknown model '{args.model}'. Available: {all_models}")
        models = [args.model]

    # Resolve datasets
    default_ds  = pipe_cfg.get("default_dataset", "ts_500")
    if args.dataset is None:
        datasets = [default_ds]
    elif args.dataset == "both":
        datasets = ["ts_500", "ts_1500"]
    else:
        datasets = [args.dataset]

    # Primary dataset for test/eval/figures (first in list)
    primary_ds = datasets[0]

    skip     = not args.no_skip and pipe_cfg.get("skip_if_done", True)
    parallel = not args.no_parallel and pipe_cfg.get("parallel", True)

    # Extra args for plot_figures.py
    fig_args = []
    if args.fig3_only: fig_args.append("--fig3_only")
    if args.fig2:      fig_args.append("--fig2")
    if args.fig4:      fig_args.append("--fig4")
    if args.fig5:      fig_args.append("--fig5")

    print(f"\nPipeline config:")
    print(f"  stage    : {args.stage}")
    print(f"  models   : {models}")
    print(f"  datasets : {datasets}")
    print(f"  skip     : {skip}")
    print(f"  parallel : {parallel}")
    print(f"  slurm    : {slurm_available()}")

    # Run stages
    s = args.stage

    if s in ("cache", "all"):
        stage_cache(datasets, cfg, skip)

    if s in ("preprocess", "all"):
        stage_preprocess(cfg, skip)

    if s in ("train", "all"):
        stage_train(models, datasets, cfg, skip, parallel)

    if s in ("evaluate", "test", "all"):
        stage_evaluate(models, primary_ds, cfg, skip, parallel,
                       target=args.target)

    if s in ("metrics", "test", "all"):
        stage_metrics(models, primary_ds, cfg)

    if s in ("figures", "test", "all"):
        stage_figures(models, primary_ds, cfg, fig_args)

    print(f"\n=== Pipeline complete: {args.stage} ===")


if __name__ == "__main__":
    main()
