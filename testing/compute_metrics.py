"""
testing/compute_metrics.py
==========================
Reads prediction CSVs from results/ and computes all evaluation metrics:

    Per (model, core, sapropel):
        - AUC for DL model (p_transition as signal)
        - AUC for variance  (Kendall tau as threshold)
        - AUC for lag-1 AC  (Kendall tau as threshold)
        - Kendall tau trends

    Across all models:
        - AUC comparison table → metrics/auc_comparison_all_models.csv

ROC protocol matches Bury 2021 exactly:
    Pre-transition (forced) windows = positive class
    Neutral windows = negative class
    Threshold sweep → ROC curve → AUC

Usage:
    python testing/compute_metrics.py --model cnn_lstm --dataset ts_500
    python testing/compute_metrics.py --all            # all models
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc as sk_auc
from scipy.stats import kendalltau

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset_loader import load_config
from models import list_models


# =============================================================================
#  ROC helpers
# =============================================================================

def compute_auc(scores_forced: np.ndarray,
                scores_null:   np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve and AUC.

    Parameters
    ----------
    scores_forced : signal scores for pre-transition (positive) windows
    scores_null   : signal scores for neutral (negative) windows

    Returns (fpr, tpr, auc_score)
    """
    y_true  = np.concatenate([np.ones(len(scores_forced)),
                               np.zeros(len(scores_null))])
    y_score = np.concatenate([scores_forced, scores_null])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr, float(sk_auc(fpr, tpr))


def kendall_tau_to_auc(ktau_values_forced: np.ndarray,
                        ktau_values_null:   np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    For variance and lag-1 AC, the discrimination threshold is Kendall tau.
    Positive Kendall tau = increasing trend = warning signal.
    """
    return compute_auc(ktau_values_forced, ktau_values_null)


# =============================================================================
#  Load prediction CSVs
# =============================================================================

def load_predictions(model_name: str, core_name: str,
                     sapropel_id: str, cfg: dict) -> Optional[pd.DataFrame]:
    """
    Load the predictions CSV for one (model, core, sapropel).
    Returns None if file not found.
    """
    results_dir = REPO_ROOT / cfg["paths"]["results"]
    fname = cfg["naming"]["predictions"].format(
        model    = model_name,
        core     = core_name,
        sapropel = sapropel_id,
    )
    fpath = results_dir / fname

    if not fpath.exists():
        print(f"  [missing] {fname}")
        return None

    return pd.read_csv(fpath)


def get_test_sapropels(core_name: str, cfg: dict) -> List[str]:
    """Return list of test sapropel IDs for a core."""
    saps = cfg["pangaea"]["cores"][core_name]["sapropels"]
    return [s["id"] for s in saps if s["role"] == "test"]


# =============================================================================
#  Metrics computation for one model
# =============================================================================

def compute_model_metrics(
    model_name:   str,
    dataset_name: str,
    cfg:          dict,
) -> Dict:
    """
    Compute AUC for all signals across all cores and sapropels.

    Returns dict:
        {core_name: {sapropel_id: {signal: auc_score}}}
    """
    cores   = cfg["slurm"]["test_cores"]
    results = {}

    for core_name in cores:
        sap_ids = get_test_sapropels(core_name, cfg)
        results[core_name] = {}

        # Collect scores across all sapropels for this core
        # (AUC computed per-sapropel — each is an independent transition event)
        for sap_id in sap_ids:
            df = load_predictions(model_name, core_name, sap_id, cfg)
            if df is None:
                continue

            # ── DL model AUC ─────────────────────────────────────────────────
            # We need pre-transition vs neutral windows.
            # Pre-transition = last 40% of the prediction steps
            # Neutral = first 40% of the prediction steps
            # (approximation used when explicit null series aren't available)
            n     = len(df)
            split = max(1, int(n * 0.4))

            neutral_mask    = np.zeros(n, dtype=bool)
            pretrans_mask   = np.zeros(n, dtype=bool)
            neutral_mask[:split]  = True
            pretrans_mask[-split:] = True

            dl_forced = df.loc[pretrans_mask, "p_transition"].values
            dl_null   = df.loc[neutral_mask,  "p_transition"].values

            _, _, auc_dl = compute_auc(dl_forced, dl_null)

            # ── Variance AUC ─────────────────────────────────────────────────
            var_forced = df.loc[pretrans_mask, "variance"].values
            var_null   = df.loc[neutral_mask,  "variance"].values
            _, _, auc_var = compute_auc(var_forced, var_null)

            # ── Lag-1 AC AUC ─────────────────────────────────────────────────
            ac_forced = df.loc[pretrans_mask, "lag1_ac"].values
            ac_null   = df.loc[neutral_mask,  "lag1_ac"].values
            _, _, auc_ac = compute_auc(ac_forced, ac_null)

            # ── Kendall tau values ────────────────────────────────────────────
            ktau_var = float(df["ktau_variance"].iloc[0])
            ktau_ac  = float(df["ktau_lag1_ac"].iloc[0])

            results[core_name][sap_id] = {
                f"auc_{model_name}": round(auc_dl,  3),
                "auc_variance":       round(auc_var, 3),
                "auc_lag1_ac":        round(auc_ac,  3),
                "ktau_variance":      round(ktau_var, 4),
                "ktau_lag1_ac":       round(ktau_ac,  4),
            }

            print(
                f"  {core_name}/{sap_id}: "
                f"AUC {model_name}={auc_dl:.3f} "
                f"var={auc_var:.3f} ac={auc_ac:.3f}"
            )

    return results


# =============================================================================
#  Build AUC comparison table across all models
# =============================================================================

def build_auc_table(cfg: dict) -> pd.DataFrame:
    """
    Read all per-sapropel metrics and build the final comparison table.

    Columns: core, sapropel, auc_cnn_lstm, auc_lstm, auc_cnn,
             auc_variance, auc_lag1_ac
    """
    cores   = cfg["slurm"]["test_cores"]
    models  = [m["name"] for m in cfg["models"]]
    met_dir = REPO_ROOT / cfg["paths"]["metrics"]
    rows    = []

    for core_name in cores:
        sap_ids = get_test_sapropels(core_name, cfg)
        for sap_id in sap_ids:
            row = {"core": core_name, "sapropel": sap_id}

            for model_name in models:
                # Read from per-sapropel entry in test_metrics.json
                met_fname = cfg["naming"]["test_metrics"].format(
                    model=model_name, core=core_name
                )
                met_path = met_dir / met_fname
                if not met_path.exists():
                    row[f"auc_{model_name}"] = np.nan
                    continue

                with open(met_path) as f:
                    met = json.load(f)

                sap_data = met.get("sapropels", {}).get(sap_id, {})
                row[f"auc_{model_name}"] = sap_data.get(f"auc_{model_name}", np.nan)

            # Baselines (same regardless of model — use first model's file)
            first_model = models[0]
            met_fname = cfg["naming"]["test_metrics"].format(
                model=first_model, core=core_name
            )
            met_path = met_dir / met_fname
            if met_path.exists():
                with open(met_path) as f:
                    met = json.load(f)
                sap_data = met.get("sapropels", {}).get(sap_id, {})
                row["auc_variance"] = sap_data.get("auc_variance", np.nan)
                row["auc_lag1_ac"]  = sap_data.get("auc_lag1_ac",  np.nan)
            else:
                row["auc_variance"] = np.nan
                row["auc_lag1_ac"]  = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
#  Save ROC data for plotting
# =============================================================================

def save_roc_data(model_name: str, dataset_name: str,
                  cfg: dict) -> None:
    """
    For each (core, sapropel), compute and save full ROC curve arrays.
    Saved to metrics/{model}_{core}_{sapropel}_roc.json
    Used by plot_figures.py to draw Figure 5.
    """
    cores   = cfg["slurm"]["test_cores"]
    met_dir = REPO_ROOT / cfg["paths"]["metrics"]
    met_dir.mkdir(parents=True, exist_ok=True)

    for core_name in cores:
        sap_ids = get_test_sapropels(core_name, cfg)

        for sap_id in sap_ids:
            df = load_predictions(model_name, core_name, sap_id, cfg)
            if df is None:
                continue

            n     = len(df)
            split = max(1, int(n * 0.4))
            neutral_mask  = np.zeros(n, dtype=bool)
            pretrans_mask = np.zeros(n, dtype=bool)
            neutral_mask[:split]   = True
            pretrans_mask[-split:] = True

            roc_data = {}

            # DL model
            fpr, tpr, auc_val = compute_auc(
                df.loc[pretrans_mask, "p_transition"].values,
                df.loc[neutral_mask,  "p_transition"].values,
            )
            roc_data[model_name] = {
                "fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc_val
            }

            # Variance
            fpr, tpr, auc_val = compute_auc(
                df.loc[pretrans_mask, "variance"].values,
                df.loc[neutral_mask,  "variance"].values,
            )
            roc_data["variance"] = {
                "fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc_val
            }

            # Lag-1 AC
            fpr, tpr, auc_val = compute_auc(
                df.loc[pretrans_mask, "lag1_ac"].values,
                df.loc[neutral_mask,  "lag1_ac"].values,
            )
            roc_data["lag1_ac"] = {
                "fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc_val
            }

            roc_fname = f"{model_name}_{core_name}_{sap_id}_roc.json"
            with open(met_dir / roc_fname, "w") as f:
                json.dump(roc_data, f)

            print(f"  ROC data saved → {roc_fname}")


# =============================================================================
#  Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute ROC/AUC metrics from prediction CSVs."
    )
    parser.add_argument("--model",   type=str, default=None,
                        choices=list_models())
    parser.add_argument("--dataset", type=str, default="ts_500",
                        choices=["ts_500", "ts_1500"])
    parser.add_argument("--all",     action="store_true",
                        help="Process all models and build comparison table")
    parser.add_argument("--config",  type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.all or args.model is None:
        models_to_run = [m["name"] for m in cfg["models"]]
    else:
        models_to_run = [args.model]

    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"  Computing metrics: {model_name} / {args.dataset}")
        print(f"{'='*60}")

        # Compute per-sapropel metrics
        results = compute_model_metrics(model_name, args.dataset, cfg)

        # Save ROC curve data for plot_figures.py
        save_roc_data(model_name, args.dataset, cfg)

        # Update test_metrics.json with AUC values
        met_dir = REPO_ROOT / cfg["paths"]["metrics"]
        for core_name, sap_dict in results.items():
            met_fname = cfg["naming"]["test_metrics"].format(
                model=model_name, core=core_name
            )
            met_path = met_dir / met_fname
            if met_path.exists():
                with open(met_path) as f:
                    existing = json.load(f)
                # Merge AUC into sapropel entries
                for sap_id, auc_vals in sap_dict.items():
                    if sap_id in existing.get("sapropels", {}):
                        existing["sapropels"][sap_id].update(auc_vals)
                with open(met_path, "w") as f:
                    json.dump(existing, f, indent=2)

    # Build and save comparison table
    print(f"\n{'='*60}")
    print("  Building AUC comparison table (all models)")
    print(f"{'='*60}")

    auc_table = build_auc_table(cfg)
    out_path  = (REPO_ROOT / cfg["paths"]["metrics"]
                 / cfg["naming"]["auc_table"])
    auc_table.to_csv(out_path, index=False)
    print(f"\nAUC table saved → {out_path.name}")
    print(auc_table.to_string(index=False))

    print("\nNext: python testing/plot_figures.py --model all")


if __name__ == "__main__":
    main()
