"""
testing/compute_metrics.py
==========================
Correct ROC computation matching Bury 2021 / Ma 2025 exactly.

Key fix from previous version:
  BEFORE: split one 40-step sequence → 16 forced + 16 neutral (unreliable)
  NOW:    forced  = all 40 steps from pre-transition (forced.csv)
          neutral = all 40 steps from neutral segment (neutral.csv)
          → 40 vs 40 per sapropel, combined across sapropels for the core

Paper protocol:
  - N = total predictions used for ROC
  - MS21 (1 test event): N = 40 forced + 40 neutral = 80 → but paper shows N=800
    → they bootstrap 20 null series × 40 = 800 neutral, 1 forced × 40 = 40 forced
    → we use the simpler version: 40 forced + 40 neutral per sapropel
  - Each element has its own ROC curve (this is our extension beyond the paper)

Outputs:
  metrics/{model}_{core}_{sapropel}_{element}_roc.json   ← ROC curve data
  metrics/auc_comparison_all_models.csv                  ← final table

Usage:
  python testing/compute_metrics.py --model cnn_lstm --dataset ts_500
  python testing/compute_metrics.py --all --dataset ts_500
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc as sk_auc

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset_loader import load_config
from src.rolling_window import ELEMENTS, PRIMARY_ELEMENT
from models import list_models


# =============================================================================
#  ROC helpers
# =============================================================================

def compute_roc(scores_forced: np.ndarray,
                scores_neutral: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve and AUC.
    forced  → positive class (pre-transition)
    neutral → negative class
    """
    if len(scores_forced) == 0 or len(scores_neutral) == 0:
        return np.array([0, 1]), np.array([0, 1]), 0.5

    y_true  = np.concatenate([np.ones(len(scores_forced)),
                               np.zeros(len(scores_neutral))])
    y_score = np.concatenate([scores_forced, scores_neutral])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr, float(sk_auc(fpr, tpr))


# =============================================================================
#  Load prediction CSVs
# =============================================================================

def load_pred(model: str, core: str, sapropel: str,
              element: str, segment_type: str,
              cfg: dict) -> Optional[pd.DataFrame]:
    """Load one prediction CSV. Returns None if missing."""
    results_dir = REPO_ROOT / cfg["paths"]["results"]
    fname = f"{model}_{core}_{sapropel}_{element}_{segment_type}.csv"
    fpath = results_dir / fname
    if not fpath.exists():
        return None
    return pd.read_csv(fpath)


def get_test_sapropels(core: str, cfg: dict) -> List[str]:
    return [s["id"] for s in cfg["pangaea"]["cores"][core]["sapropels"]
            if s["role"] == "test"]


# =============================================================================
#  Compute ROC per (model, core, element) combining all test sapropels
# =============================================================================

def compute_core_roc(model_name: str, dataset_name: str,
                     core_name: str, cfg: dict) -> Dict:
    """
    Compute ROC for all elements on one core.
    Combines predictions across ALL test sapropels for that core
    to get more data points (matching paper approach of combining events).

    Returns:
        {element: {"fpr": [...], "tpr": [...], "auc": float,
                   "n_forced": int, "n_neutral": int}}
    """
    sap_ids = get_test_sapropels(core_name, cfg)
    roc_results = {}

    for element in ELEMENTS:
        all_forced_dl  = []
        all_neutral_dl = []
        all_forced_var = []
        all_neutral_var= []
        all_forced_ac  = []
        all_neutral_ac = []

        for sap_id in sap_ids:
            df_f = load_pred(model_name, core_name, sap_id,
                             element, "forced", cfg)
            df_n = load_pred(model_name, core_name, sap_id,
                             element, "neutral", cfg)

            if df_f is not None and "p_transition" in df_f.columns:
                all_forced_dl.extend(df_f["p_transition"].tolist())
                all_forced_var.extend(df_f["variance"].tolist())
                all_forced_ac.extend(df_f["lag1_ac"].tolist())

            if df_n is not None and "p_transition" in df_n.columns:
                all_neutral_dl.extend(df_n["p_transition"].tolist())
                all_neutral_var.extend(df_n["variance"].tolist())
                all_neutral_ac.extend(df_n["lag1_ac"].tolist())

        n_forced  = len(all_forced_dl)
        n_neutral = len(all_neutral_dl)

        if n_forced == 0 or n_neutral == 0:
            print(f"  [skip] {model_name}/{core_name}/{element}: "
                  f"no data (forced={n_forced}, neutral={n_neutral})")
            continue

        forced_dl  = np.array(all_forced_dl)
        neutral_dl = np.array(all_neutral_dl)
        forced_var = np.array(all_forced_var)
        neutral_var= np.array(all_neutral_var)
        forced_ac  = np.array(all_forced_ac)
        neutral_ac = np.array(all_neutral_ac)

        # DL model ROC
        fpr_dl, tpr_dl, auc_dl = compute_roc(forced_dl, neutral_dl)
        # Variance ROC
        fpr_var, tpr_var, auc_var = compute_roc(forced_var, neutral_var)
        # Lag-1 AC ROC
        fpr_ac, tpr_ac, auc_ac = compute_roc(forced_ac, neutral_ac)

        # Inset bar proportions (mean p_transition for each class)
        mean_p_forced  = float(np.mean(forced_dl))
        mean_p_neutral = float(np.mean(neutral_dl))

        roc_results[element] = {
            model_name: {
                "fpr": fpr_dl.tolist(), "tpr": tpr_dl.tolist(),
                "auc": round(auc_dl, 3),
            },
            "variance": {
                "fpr": fpr_var.tolist(), "tpr": tpr_var.tolist(),
                "auc": round(auc_var, 3),
            },
            "lag1_ac": {
                "fpr": fpr_ac.tolist(), "tpr": tpr_ac.tolist(),
                "auc": round(auc_ac, 3),
            },
            "n_forced":        n_forced,
            "n_neutral":       n_neutral,
            "N":               n_forced + n_neutral,
            "mean_p_forced":   round(mean_p_forced, 3),
            "mean_p_neutral":  round(mean_p_neutral, 3),
        }

        print(
            f"  {core_name}/{element}: "
            f"AUC {model_name}={auc_dl:.3f}  "
            f"var={auc_var:.3f}  ac={auc_ac:.3f}  "
            f"N_forced={n_forced}  N_neutral={n_neutral}"
        )

    return roc_results


# =============================================================================
#  Save ROC data
# =============================================================================

def save_roc_data(model_name: str, core_name: str,
                  roc_results: Dict, cfg: dict) -> None:
    """
    Save ROC data per element to metrics directory.
    One JSON per (model, core, element).
    """
    met_dir = REPO_ROOT / cfg["paths"]["metrics"]
    met_dir.mkdir(parents=True, exist_ok=True)

    for element, data in roc_results.items():
        fname = f"{model_name}_{core_name}_{element}_roc.json"
        with open(met_dir / fname, "w") as f:
            json.dump(data, f)


# =============================================================================
#  Build AUC comparison table — all models × all cores × all elements
# =============================================================================

def build_auc_table(cfg: dict) -> pd.DataFrame:
    """
    Read all ROC JSON files and build the final comparison table.
    Columns: core, element, auc_{model}, auc_variance, auc_lag1_ac, N
    """
    cores      = cfg["slurm"]["test_cores"]
    model_list = [m["name"] for m in cfg["models"]]
    met_dir    = REPO_ROOT / cfg["paths"]["metrics"]
    rows       = []

    for core_name in cores:
        for element in ELEMENTS:
            row = {"core": core_name, "element": element}

            for model_name in model_list:
                fname = f"{model_name}_{core_name}_{element}_roc.json"
                fpath = met_dir / fname
                if not fpath.exists():
                    row[f"auc_{model_name}"] = np.nan
                    continue
                with open(fpath) as f:
                    data = json.load(f)
                row[f"auc_{model_name}"] = data.get(model_name, {}).get("auc", np.nan)
                if "auc_variance" not in row:
                    row["auc_variance"] = data.get("variance", {}).get("auc", np.nan)
                    row["auc_lag1_ac"]  = data.get("lag1_ac",  {}).get("auc", np.nan)
                    row["N"]            = data.get("N", 0)

            rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
#  Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=None, choices=list_models())
    parser.add_argument("--dataset", default="ts_500",
                        choices=["ts_500", "ts_1500"])
    parser.add_argument("--all",     action="store_true")
    parser.add_argument("--config",  default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    models_to_run = (
        [m["name"] for m in cfg["models"]]
        if args.all or args.model is None
        else [args.model]
    )
    cores = cfg["slurm"]["test_cores"]

    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}  Dataset: {args.dataset}")
        print(f"{'='*60}")

        for core_name in cores:
            print(f"\n  Core: {core_name}")
            roc_results = compute_core_roc(
                model_name, args.dataset, core_name, cfg
            )
            save_roc_data(model_name, core_name, roc_results, cfg)

    # Build final AUC table
    print(f"\n{'='*60}")
    print("  Building AUC comparison table")
    print(f"{'='*60}")
    auc_table = build_auc_table(cfg)
    out_path  = (REPO_ROOT / cfg["paths"]["metrics"]
                 / cfg["naming"]["auc_table"])
    auc_table.to_csv(out_path, index=False)
    print(f"\nAUC table → {out_path.name}")
    print(auc_table.to_string(index=False))

    print("\nNext: python testing/plot_figures.py --model all --dataset ts_500")


if __name__ == "__main__":
    main()
