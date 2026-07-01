"""
testing/evaluate.py

Evaluates trained models on:
  - Zenodo test set (Bury synthetic time series)
  - PANGAEA empirical XRF cores (rolling-window EWS inference)

AR(1) null surrogates follow Bury et al. (2021):
  Fit AR(1) to the FIRST 20% of the forced residuals (the neutral reference
  period before critical-slowing-down dominates). This prevents the CSD ramp
  near the transition from being baked into the null model, which would
  suppress the AUC signal.
"""
import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models import get_model, list_models, is_tsc_model
from src.constants import CLASS_NAMES, NULL_IDX
from src.dataset_loader import load_config, get_dataset_info, get_dataloader
from src.rolling_window import run_all_sapropels, compute_rolling_ews, ELEMENTS
from metric.auc         import compute_auc, ovr_macro_auc
from metric.roc         import compute_roc
from metric.accuracy    import compute_accuracy
from metric.kendall_tau import compute_kendall_tau, compute_tau_ci
from metric.multiclass  import macro_f1
from testing.plot_figures import plot_roc, plot_confusion_matrix, plot_pangaea_series


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic):  return obj.item()
        return super().default(obj)


def setup_log(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(log_path.stem)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for h in [logging.FileHandler(log_path, mode="w"),
              logging.StreamHandler(sys.stdout)]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


N_SURROGATES = 10
# Fraction of the forced series used to fit the AR(1) null model.
# Bury fits to the first 20% (the neutral period) so the CSD ramp near the
# transition does NOT inflate the AR(1) autocorrelation estimate.
AR1_FIT_FRACTION = 0.20


def _fit_ar1_neutral(forced_residuals: np.ndarray, fit_fraction: float = AR1_FIT_FRACTION):
    """
    Fit an AR(1) model to the first `fit_fraction` of the forced residuals.

    Using only the neutral (pre-CSD) segment ensures the surrogate reflects
    the background dynamics, not the slowing-down ramp near the transition.
    Mirrors Bury's generate_nulls_ar1.py exactly.
    """
    n_fit  = max(3, int(len(forced_residuals) * fit_fraction))
    x_ref  = forced_residuals[:n_fit]
    x_dm   = x_ref - x_ref.mean()

    if np.std(x_dm) < 1e-10:
        return 0.0, float(np.std(x_dm))

    # Lag-1 autocorrelation on the reference window.
    alpha = float(np.corrcoef(x_dm[:-1], x_dm[1:])[0, 1])
    alpha = np.clip(alpha, -0.99, 0.99)

    # Innovation standard deviation: sigma = sqrt(var * (1 - alpha^2))
    var   = float(np.var(x_ref, ddof=1))
    sigma = float(np.sqrt(max(var * (1.0 - alpha ** 2), 1e-12)))

    return alpha, sigma


def _generate_ar1_surrogates(forced_residuals: np.ndarray,
                              n: int = N_SURROGATES,
                              seed: int = 42) -> list:
    """
    Generate `n` AR(1) surrogate series the same length as forced_residuals.

    The AR(1) parameters are estimated from the first AR1_FIT_FRACTION of the
    series (neutral reference period). The first surrogate point starts from
    the stationary distribution N(0, sigma/sqrt(1-alpha^2)).
    """
    rng   = np.random.default_rng(seed)
    alpha, sigma = _fit_ar1_neutral(forced_residuals)
    mu    = forced_residuals.mean()

    out = []
    for _ in range(n):
        s    = np.zeros(len(forced_residuals))
        s[0] = rng.normal(0, sigma / max(np.sqrt(1.0 - alpha ** 2), 1e-6))
        for t in range(1, len(s)):
            s[t] = alpha * s[t - 1] + rng.normal(0, sigma)
        out.append(s + mu)
    return out

def _save_ar1_null_csv(forced_residuals: np.ndarray,
                        forced_ages: np.ndarray,
                        null_path: Path):
    import pandas as pd
    surrogates = _generate_ar1_surrogates(forced_residuals)
    data = {"age_kyr_bp": forced_ages}
    for i, s in enumerate(surrogates):
        data[f"null_{i}"] = s
    pd.DataFrame(data).to_csv(null_path, index=False)

def experiment_dir(model, dataset, metric, cfg) -> Path:
    name = cfg["naming"]["experiment_dir"].format(
        model=model, dataset=dataset, metric=metric)
    d = REPO_ROOT / cfg["paths"]["test_results"] / name
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_result(data: dict, out_dir: Path):
    path = out_dir / "result.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    return path

def load_dl_models(model_name, dataset_name, ts_len, num_classes, cfg, device, logger):
    ckpt_dir     = REPO_ROOT / cfg["paths"]["checkpoints"]
    pad_variants = cfg["training"][model_name]["pad_variants"]
    models = []
    for v in pad_variants:
        name  = cfg["naming"]["checkpoint_dl"].format(
            model=model_name, dataset=dataset_name, variant=v)
        fpath = ckpt_dir / name
        if not fpath.exists():
            logger.warning(f"  Not found: {name}")
            continue
        m = get_model(model_name, ts_len=ts_len, num_classes=num_classes)
        m.load_state_dict(torch.load(fpath, map_location=device, weights_only=True))
        m.to(device).eval()
        models.append(m)
        logger.info(f"  Loaded: {name}")
    if not models:
        raise RuntimeError(f"No checkpoints for {model_name}/{dataset_name}")
    return models

def load_tsc_model(model_name, dataset_name, cfg, logger):
    import joblib
    ckpt_dir = REPO_ROOT / cfg["paths"]["checkpoints"]
    name     = cfg["naming"]["checkpoint_tsc"].format(
        model=model_name, dataset=dataset_name)
    fpath = ckpt_dir / name
    if not fpath.exists():
        raise RuntimeError(f"TSC checkpoint not found: {name}")
    return joblib.load(fpath)

@torch.no_grad()
def dl_predict_proba(models, X_np, device, batch_size=256):
    X_t = torch.tensor(X_np, dtype=torch.float32).unsqueeze(1)
    all_probs = []
    for m in models:
        probs = []
        for i in range(0, len(X_t), batch_size):
            probs.append(m(X_t[i:i+batch_size].to(device)).softmax(dim=1).cpu().numpy())
        all_probs.append(np.concatenate(probs))
    return np.mean(all_probs, axis=0)

def _load_ch_stats(model_name, dataset_name, cfg) -> dict:
    """Load per-channel z-norm stats saved during 4-channel training, if present."""
    ckpt_dir  = REPO_ROOT / cfg["paths"]["checkpoints"]
    stem      = cfg["naming"]["checkpoint_tsc"].format(
        model=model_name, dataset=dataset_name).replace(".pkl", "")
    stats_path = ckpt_dir / f"{stem}_ch_stats.npz"
    if stats_path.exists():
        d = np.load(stats_path)
        return {"mean": d["mean"], "std": d["std"]}
    return None


def _prepare_tsc_input(X_2d, model_name, dataset_name, cfg):
    """
    Convert (N, L) residuals to the shape expected by a TSC model.
    Respects use_4channel config flag and loads saved channel stats.
    """
    use_4ch     = cfg.get("inference", {}).get("use_4channel", False)
    window_frac = cfg.get("inference", {}).get("rolling_window_frac_augment", 0.25)

    if use_4ch:
        from src.ews_augmenter import augment_ews_channels
        ch_stats = _load_ch_stats(model_name, dataset_name, cfg)
        X_aug, _ = augment_ews_channels(X_2d, window_frac=window_frac,
                                         channel_stats=ch_stats)
        return X_aug   # (N, 4, L)
    else:
        return X_2d[:, np.newaxis, :]   # (N, 1, L)


def evaluate_zenodo(model_name, dataset_name, cfg, device, logger, force=False):
    """
    Evaluate on the Bury synthetic test set.

    Reports (all saved to result.json):
      - Binary AUC: forced vs null, p_transition = 1 - P(null)  [Bury-comparable]
      - Per-class AUC: each bifurcation type vs null separately   [Bury Fig 2]
      - 4-class macro-F1 and OVR macro-AUC                       [thesis contribution]
      - Confusion matrix (rows=true, cols=predicted)
    """
    # Skip early if result already exists
    _rdir = experiment_dir(model_name, dataset_name, "zenodo", cfg)
    if (_rdir / "result.json").exists() and not force:
        logger.info(f"  Already evaluated — skipping (use --force to redo)")
        return json.loads((_rdir / "result.json").read_text())

    ds_info     = get_dataset_info(dataset_name, cfg)
    ts_len      = ds_info["ts_length"]
    num_classes = ds_info["num_classes"]
    class_names = CLASS_NAMES   # canonical ordering from src/constants.py
    null_idx    = NULL_IDX

    if is_tsc_model(model_name):
        model      = load_tsc_model(model_name, dataset_name, cfg, logger)
        predict_fn = lambda X: model.predict_proba(X)
    else:
        models     = load_dl_models(model_name, dataset_name, ts_len,
                                     num_classes, cfg, device, logger)
        predict_fn = lambda X: dl_predict_proba(models, X, device)

    # Load test split — single pass to keep X and y paired.
    test_dl  = get_dataloader(dataset_name, "test", cfg, pad_variant=1, batch_size=1024)
    X_parts, y_parts = [], []
    for x_b, y_b in test_dl:
        X_parts.append(x_b.squeeze(1).numpy())
        y_parts.append(y_b.numpy())
    X_all = np.concatenate(X_parts)
    y_all = np.concatenate(y_parts)

    if is_tsc_model(model_name):
        mask = X_all.std(axis=1) > 1e-7
        if (~mask).any():
            logger.warning(f"  Dropping {(~mask).sum()} flat test series")
        X_all, y_all = X_all[mask], y_all[mask]

    if len(X_all) == 0:
        logger.error("  No valid samples after filtering.")
        return

    # Convert to model input format (1-channel or 4-channel).
    if is_tsc_model(model_name):
        X_input = _prepare_tsc_input(X_all, model_name, dataset_name, cfg)
    else:
        X_input = X_all  # DL models receive (N, L) and handle shape internally

    t0       = time.time()
    probs    = predict_fn(X_input)   # (N, num_classes)
    inf_time = time.time() - t0

    preds         = probs.argmax(axis=1)
    # Binary: forced (any bifurcation class) vs null
    p_transition  = 1.0 - probs[:, null_idx]
    labels_binary = (y_all != null_idx).astype(int)

    # --- Metrics ---
    bin_auc        = compute_auc(labels_binary, p_transition)
    fpr, tpr, thr  = compute_roc(labels_binary, p_transition)
    mf1            = macro_f1(y_all, preds)
    mac_auc        = ovr_macro_auc(y_all, probs)
    acc_metrics    = compute_accuracy(y_all, preds)

    # Per-class AUC: each bifurcation type vs null only (Bury et al. Fig 2 metric).
    # For class c: positives = c-class samples, negatives = null samples,
    # score = 1 - P(null).  This lets us see which transition types are easiest.
    per_class_auc = {}
    for cls_idx, cls_name in enumerate(class_names):
        if cls_idx == null_idx:
            continue
        mask  = (y_all == cls_idx) | (y_all == null_idx)
        if mask.sum() < 2:
            continue
        y_bin = (y_all[mask] == cls_idx).astype(int)
        try:
            per_class_auc[cls_name] = round(float(compute_auc(y_bin, p_transition[mask])), 6)
        except Exception:
            per_class_auc[cls_name] = float("nan")

    logger.info(f"  binary AUC={bin_auc:.4f}  macro-F1={mf1:.4f}  "
                f"macro-AUC(OVR)={mac_auc:.4f}  acc={acc_metrics['accuracy']:.4f}")
    logger.info(f"  per-class AUC: " +
                "  ".join(f"{k}={v:.4f}" for k, v in per_class_auc.items()))

    base = {
        "model":   model_name,  "dataset": dataset_name,
        "num_classes": num_classes, "class_names": class_names,
        "null_idx": null_idx,
        "inference_time_sec": round(inf_time, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Save one result.json that contains ALL metrics for this model+dataset.
    result_dir  = experiment_dir(model_name, dataset_name, "zenodo", cfg)
    result_data = {
        **base,
        # Binary metrics (Bury-comparable)
        "binary_auc":    round(float(bin_auc), 6),
        "roc_fpr":       fpr,
        "roc_tpr":       tpr,
        "roc_thresh":    thr,
        # Per-class AUC (Bury Fig 2 metric)
        "per_class_auc": per_class_auc,
        # 4-class metrics (thesis contribution)
        "macro_f1":      round(float(mf1), 6),
        "macro_auc_ovr": round(float(mac_auc), 6),
        # Full accuracy breakdown
        **acc_metrics,
    }
    save_result(result_data, result_dir)
    logger.info(f"  Saved → {result_dir.name}/result.json")

    plot_roc(result_data, result_dir)
    plot_confusion_matrix(result_data, result_dir, class_names)
    return result_data

def evaluate_pangaea(model_name, dataset_name, cfg, device, logger, force=False):
    """
    Evaluate on PANGAEA empirical XRF cores using rolling-window EWS inference.

    Reports per (core, sapropel, element):
      - Binary AUC: forced rolling windows vs AR(1) null surrogates
      - Kendall tau: rising p_transition over the forced segment (mean ± CI)
    """
    import pandas as pd

    ds_info     = get_dataset_info(dataset_name, cfg)
    ts_len      = ds_info["ts_length"]
    num_classes = ds_info["num_classes"]
    class_names = CLASS_NAMES   # canonical ordering from src/constants.py
    null_idx    = NULL_IDX

    use_4ch     = cfg.get("inference", {}).get("use_4channel", False)
    window_frac = cfg.get("inference", {}).get("rolling_window_frac_augment", 0.25)

    if is_tsc_model(model_name):
        model = load_tsc_model(model_name, dataset_name, cfg, logger)
        if use_4ch:
            ch_stats = _load_ch_stats(model_name, dataset_name, cfg)
            from src.ews_augmenter import augment_ews_channels as _aug
            def predict_fn(X_2d):
                X_aug, _ = _aug(X_2d, window_frac=window_frac, channel_stats=ch_stats)
                return model.predict_proba(X_aug)
        else:
            predict_fn = lambda X_2d: model.predict_proba(X_2d[:, np.newaxis, :])
    else:
        models     = load_dl_models(model_name, dataset_name, ts_len,
                                     num_classes, cfg, device, logger)
        predict_fn = lambda X: dl_predict_proba(models, X, device)

    cores       = list(cfg["pangaea"]["cores"].keys())
    all_results = {}

    for core_name in cores:
        logger.info(f"\n  Core: {core_name}")
        clean_dir = REPO_ROOT / cfg["paths"]["pangaea_clean"] / core_name
        sapropels = cfg["pangaea"]["cores"][core_name]["sapropels"]
        test_saps = [s for s in sapropels if s["role"] == "test"]

        if not test_saps:
            logger.warning(f"  No test sapropels for {core_name}")
            continue

        for sap in test_saps:
            sap_id = sap["id"]

            for element in ELEMENTS:
                # ── Load forced segment ──────────────────────────────────────
                forced_path = clean_dir / f"{core_name}_{sap_id}_forced.csv"
                if not forced_path.exists():
                    logger.warning(f"  Missing: {forced_path.name} — skipping")
                    continue

                df_forced = pd.read_csv(forced_path)
                resid_col = f"{element}_residuals"
                if resid_col not in df_forced.columns:
                    continue

                forced_residuals = df_forced[resid_col].values.astype(np.float64)
                forced_ages      = df_forced["age_kyr_bp"].values.astype(np.float64)
                valid_f          = ~np.isnan(forced_residuals)
                if valid_f.sum() < 10:
                    continue

                forced_residuals = forced_residuals[valid_f]
                forced_ages      = forced_ages[valid_f]

                # ── FIX: sort oldest-first so rolling window approaches ───────
                # transition at the END (age_kyr_bp descending = old → young)
                sort_idx         = np.argsort(forced_ages)[::-1]
                forced_residuals = forced_residuals[sort_idx]
                forced_ages      = forced_ages[sort_idx]

                # ── Rolling-window EWS on forced ─────────────────────────────
                rw_forced = compute_rolling_ews(
                    residuals=forced_residuals, ages_kyr_bp=forced_ages,
                    element=element, core_name=core_name,
                    sapropel_id=sap_id, segment_type="forced",
                    cfg=cfg, ts_len=ts_len)

                X_forced     = np.stack(rw_forced.dl_inputs)
                probs_forced = predict_fn(X_forced)
                p_trans_f    = (1.0 - probs_forced[:, null_idx] if null_idx >= 0
                                else probs_forced[:, -1])

                # ── Load or generate AR(1) null series ───────────────────────
                null_path = clean_dir / f"{core_name}_{sap_id}_{element}_ar1_null.csv"

                if not null_path.exists():
                    logger.warning(
                        f"  AR(1) null missing for {sap_id}/{element} — "
                        f"generating {N_SURROGATES} surrogates on-the-fly")
                    null_path.parent.mkdir(parents=True, exist_ok=True)
                    _save_ar1_null_csv(forced_residuals, forced_ages, null_path)
                    logger.info(f"  Saved: {null_path.name}")

                df_null   = pd.read_csv(null_path)
                null_cols = [c for c in df_null.columns if c.startswith("null_")]
                null_ages = (df_null["age_kyr_bp"].values.astype(np.float64)
                             if "age_kyr_bp" in df_null.columns else forced_ages)

                # Build null rolling windows. Track how many windows each
                # surrogate contributes so we can split p_trans_n later for tau.
                all_null_dl      = []
                null_window_counts = []  # n_windows per surrogate
                for nc in null_cols:
                    null_resids = df_null[nc].values.astype(np.float64)
                    valid_n     = ~np.isnan(null_resids)
                    if valid_n.sum() < 10:
                        continue
                    null_sort      = np.argsort(null_ages[valid_n])[::-1]
                    null_r_sorted  = null_resids[valid_n][null_sort]
                    null_a_sorted  = null_ages[valid_n][null_sort]
                    rw_null = compute_rolling_ews(
                        residuals=null_r_sorted, ages_kyr_bp=null_a_sorted,
                        element=element, core_name=core_name,
                        sapropel_id=sap_id, segment_type="neutral",
                        cfg=cfg, ts_len=ts_len)
                    null_window_counts.append(len(rw_null.dl_inputs))
                    all_null_dl.extend(rw_null.dl_inputs)

                tau      = compute_kendall_tau(p_trans_f)
                null_taus = []

                if not all_null_dl:
                    logger.warning(f"  No valid null series for {sap_id}/{element} — AUC skipped")
                    auc, fpr, tpr = float("nan"), [0.0, 1.0], [0.0, 1.0]
                else:
                    X_null     = np.stack(all_null_dl)
                    probs_null = predict_fn(X_null)
                    p_trans_n  = 1.0 - probs_null[:, null_idx]

                    y_true  = np.concatenate([np.ones(len(p_trans_f)),
                                              np.zeros(len(p_trans_n))])
                    y_score = np.concatenate([p_trans_f, p_trans_n])

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            auc         = compute_auc(y_true, y_score)
                            fpr, tpr, _ = compute_roc(y_true, y_score)
                        except Exception as e:
                            logger.warning(f"  AUC failed: {e}")
                            auc, fpr, tpr = float("nan"), [0.0, 1.0], [0.0, 1.0]

                    # Split p_trans_n by surrogate to compute per-surrogate tau.
                    # Mean ± CI null tau is the baseline for significance testing.
                    offset = 0
                    for n_win in null_window_counts:
                        seg = p_trans_n[offset: offset + n_win]
                        null_taus.append(compute_kendall_tau(seg))
                        offset += n_win

                null_tau_ci = compute_tau_ci(null_taus) if null_taus else {}

                tag       = f"{core_name}_{sap_id}_{element}"
                _rdir_tag = experiment_dir(model_name, f"pangaea_{tag}", "pangaea", cfg)
                if (_rdir_tag / "result.json").exists() and not force:
                    logger.info(f"    {tag}: already evaluated — skipping")
                    all_results[tag] = json.loads(
                        (_rdir_tag / "result.json").read_text())
                    continue

                base = {
                    "model": model_name, "dataset": dataset_name,
                    "core": core_name, "sapropel": sap_id, "element": element,
                    "n_forced": len(p_trans_f),
                    "n_null": len(all_null_dl) if all_null_dl else 0,
                    "p_transition": p_trans_f.tolist(),
                    "ages_kyr_bp": rw_forced.ages_kyr_bp.tolist(),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }

                result_dir  = experiment_dir(model_name, f"pangaea_{tag}", "pangaea", cfg)
                result_data = {
                    **base,
                    # Binary AUC (Bury-comparable)
                    "binary_auc":   round(float(auc), 6),
                    "roc_fpr":      fpr,
                    "roc_tpr":      tpr,
                    # Kendall tau (rising p_transition toward transition)
                    "kendall_tau":         round(float(tau), 6),
                    "tau_null_mean":       round(float(null_tau_ci.get("mean", float("nan"))), 6),
                    "tau_null_ci_low":     round(float(null_tau_ci.get("ci_low", float("nan"))), 6),
                    "tau_null_ci_high":    round(float(null_tau_ci.get("ci_high", float("nan"))), 6),
                    # EWS rolling stats (for FIG1 panel c)
                    "variance": rw_forced.variance.tolist(),
                    "lag1_ac":  rw_forced.lag1_ac.tolist(),
                }
                save_result(result_data, result_dir)

                if not np.isnan(auc):
                    plot_roc(result_data, result_dir)
                plot_pangaea_series(result_data, None, result_dir)

                logger.info(f"    {tag}: AUC={auc:.3f}  tau_forced={tau:.3f}  "
                            f"tau_null_mean={null_tau_ci.get('mean', float('nan')):.3f}")
                all_results[tag] = {
                    "binary_auc":   float(auc),
                    "kendall_tau":  float(tau),
                    "null_tau_mean": null_tau_ci.get("mean", float("nan")),
                }

    return all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, choices=list_models())
    parser.add_argument("--dataset", required=True, choices=["ts_500", "ts_1500"])
    parser.add_argument("--target",  default="zenodo",
                        choices=["zenodo", "pangaea"])
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--force",   action="store_true",
                        help="Re-evaluate even if result.json already exists.")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = REPO_ROOT / cfg["paths"]["logs"]
    log_dir.mkdir(parents=True, exist_ok=True)
    logger  = setup_log(
        log_dir / f"{args.model}_{args.dataset}_{args.target}_eval.log")

    logger.info(f"Model  : {args.model}  IS_TSC={is_tsc_model(args.model)}")
    logger.info(f"Dataset: {args.dataset}  Target: {args.target}")
    logger.info(f"Device : {device}")
    logger.info(f"Results→ test_result/")

    try:
        if args.target == "zenodo":
            evaluate_zenodo(args.model, args.dataset, cfg, device, logger,
                            force=args.force)
        else:
            evaluate_pangaea(args.model, args.dataset, cfg, device, logger,
                             force=args.force)
    except RuntimeError as e:
        logger.error(f"Skipped — {e}")
        sys.exit(0)   # missing checkpoint is not a SLURM array failure

    logger.info("\nDone.")

if __name__ == "__main__":
    main()
