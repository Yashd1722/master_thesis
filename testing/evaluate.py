"""
testing/evaluate.py
FIX: AR(1) surrogate fallback + sort forced data oldest-first so rolling
     window approaches the transition at the END (fixes inverted AUC/tau).
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
from src.dataset_loader import load_config, get_dataset_info, get_dataloader
from src.rolling_window import run_all_sapropels, compute_rolling_ews, ELEMENTS
from metric.auc import compute_auc
from metric.roc import compute_roc
from metric.accuracy import compute_accuracy
from metric.kendall_tau import compute_kendall_tau
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

def _fit_ar1(x: np.ndarray):
    x_dm = x - x.mean()
    if len(x_dm) < 3 or np.std(x_dm) < 1e-10:
        return 0.0, float(np.std(x_dm))
    phi = float(np.corrcoef(x_dm[:-1], x_dm[1:])[0, 1])
    phi = np.clip(phi, -0.99, 0.99)
    sigma = float(np.std(x_dm[1:] - phi * x_dm[:-1]))
    return phi, sigma

def _generate_ar1_surrogates(forced_residuals: np.ndarray,
                              n: int = N_SURROGATES,
                              seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    phi, sigma = _fit_ar1(forced_residuals)
    mu  = forced_residuals.mean()
    out = []
    for _ in range(n):
        s    = np.zeros(len(forced_residuals))
        s[0] = rng.normal(0, sigma / max(np.sqrt(1 - phi**2), 1e-6))
        for t in range(1, len(s)):
            s[t] = phi * s[t-1] + rng.normal(0, sigma)
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
    d = REPO_ROOT / cfg["paths"]["results"] / name
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

def evaluate_zenodo(model_name, dataset_name, cfg, device, logger):
    ds_info     = get_dataset_info(dataset_name, cfg)
    ts_len      = ds_info["ts_length"]
    num_classes = ds_info["num_classes"]
    class_names = ds_info["class_names"]
    null_idx    = class_names.index("null") if "null" in class_names else -1

    if is_tsc_model(model_name):
        model      = load_tsc_model(model_name, dataset_name, cfg, logger)
        predict_fn = model.predict_proba
    else:
        models     = load_dl_models(model_name, dataset_name, ts_len,
                                     num_classes, cfg, device, logger)
        predict_fn = lambda X: dl_predict_proba(models, X, device)

    test_dl = get_dataloader(dataset_name, "test", cfg, pad_variant=1, batch_size=1024)
    X_all, y_all = [], []
    for x, y in test_dl:
        X_all.append(x.squeeze(1).numpy())
        y_all.append(y.numpy())
    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)

    if is_tsc_model(model_name):
        mask = X_all.std(axis=1) > 1e-7
        n_drop = (~mask).sum()
        if n_drop:
            logger.warning(f"  Dropping {n_drop} flat test series")
        X_all, y_all = X_all[mask], y_all[mask]

    if len(X_all) == 0:
        logger.error("  No valid samples after filtering.")
        return

    t0    = time.time()
    probs = predict_fn(X_all)
    inf_time = time.time() - t0

    p_transition  = 1.0 - probs[:, null_idx] if null_idx >= 0 else probs[:, -1]
    labels_binary = (y_all != null_idx).astype(int) if null_idx >= 0 else (y_all > 0).astype(int)

    auc           = compute_auc(labels_binary, p_transition)
    fpr, tpr, thr = compute_roc(labels_binary, p_transition)
    preds         = probs.argmax(axis=1)
    acc_metrics   = compute_accuracy(y_all, preds)

    base = {"model": model_name, "dataset": dataset_name,
             "num_classes": num_classes, "class_names": class_names,
             "inference_time_sec": round(inf_time, 2),
             "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    auc_dir  = experiment_dir(model_name, dataset_name, "auc", cfg)
    auc_data = {**base, "metric": "auc", "auc": round(auc, 6),
                 "roc_fpr": fpr, "roc_tpr": tpr, "roc_thresh": thr}
    save_result(auc_data, auc_dir)
    logger.info(f"  AUC={auc:.4f}  -> {auc_dir.name}/result.json")

    acc_dir  = experiment_dir(model_name, dataset_name, "accuracy", cfg)
    acc_data = {**base, "metric": "accuracy", **acc_metrics}
    save_result(acc_data, acc_dir)
    logger.info(f"  Acc={acc_metrics['accuracy']:.4f}  -> {acc_dir.name}/result.json")

    plot_roc(auc_data, auc_dir)
    plot_confusion_matrix(acc_data, acc_dir, class_names)
    return {"auc": auc_data, "accuracy": acc_data}

def evaluate_pangaea(model_name, dataset_name, cfg, device, logger):
    import pandas as pd

    ds_info     = get_dataset_info(dataset_name, cfg)
    ts_len      = ds_info["ts_length"]
    num_classes = ds_info["num_classes"]
    class_names = ds_info["class_names"]
    null_idx    = class_names.index("null") if "null" in class_names else -1

    if is_tsc_model(model_name):
        model      = load_tsc_model(model_name, dataset_name, cfg, logger)
        predict_fn = model.predict_proba
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

                all_null_dl = []
                for nc in null_cols:
                    null_resids = df_null[nc].values.astype(np.float64)
                    valid_n     = ~np.isnan(null_resids)
                    if valid_n.sum() < 10:
                        continue
                    # Sort null series oldest-first to match forced direction
                    null_sort    = np.argsort(null_ages[valid_n])[::-1]
                    null_r_sorted = null_resids[valid_n][null_sort]
                    null_a_sorted = null_ages[valid_n][null_sort]
                    rw_null = compute_rolling_ews(
                        residuals=null_r_sorted, ages_kyr_bp=null_a_sorted,
                        element=element, core_name=core_name,
                        sapropel_id=sap_id, segment_type="neutral",
                        cfg=cfg, ts_len=ts_len)
                    all_null_dl.extend(rw_null.dl_inputs)

                if not all_null_dl:
                    logger.warning(f"  No valid null series for {sap_id}/{element} — AUC skipped")
                    auc, fpr, tpr = float("nan"), [0.0, 1.0], [0.0, 1.0]
                else:
                    X_null     = np.stack(all_null_dl)
                    probs_null = predict_fn(X_null)
                    p_trans_n  = (1.0 - probs_null[:, null_idx] if null_idx >= 0
                                  else probs_null[:, -1])

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

                tau = compute_kendall_tau(p_trans_f)
                tag = f"{core_name}_{sap_id}_{element}"

                base = {"model": model_name, "dataset": dataset_name,
                         "core": core_name, "sapropel": sap_id,
                         "element": element,
                         "n_forced": len(p_trans_f),
                         "n_null": len(all_null_dl) if all_null_dl else 0,
                         "p_transition": p_trans_f.tolist(),
                         "ages_kyr_bp": rw_forced.ages_kyr_bp.tolist(),
                         "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

                auc_dir  = experiment_dir(model_name, f"pangaea_{tag}", "auc", cfg)
                auc_data = {**base, "metric": "auc", "segment": "forced_vs_ar1null",
                             "auc": round(float(auc), 6), "roc_fpr": fpr, "roc_tpr": tpr}
                save_result(auc_data, auc_dir)

                ktau_dir  = experiment_dir(model_name, f"pangaea_{tag}", "kendall_tau", cfg)
                ktau_data = {**base, "metric": "kendall_tau", "segment": "forced",
                              "kendall_tau": round(float(tau), 6),
                              "variance": rw_forced.variance.tolist(),
                              "lag1_ac":  rw_forced.lag1_ac.tolist()}
                save_result(ktau_data, ktau_dir)

                if not np.isnan(auc):
                    plot_roc(auc_data, auc_dir)
                plot_pangaea_series(ktau_data, None, ktau_dir)

                logger.info(f"    {tag}: AUC={auc:.3f}  tau={tau:.3f}")
                all_results[tag] = {"auc": float(auc), "kendall_tau": float(tau)}

    return all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, choices=list_models())
    parser.add_argument("--dataset", required=True, choices=["ts_500", "ts_1500"])
    parser.add_argument("--target",  default="zenodo",
                        choices=["zenodo", "pangaea"])
    parser.add_argument("--config",  default="config.yaml")
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

    if args.target == "zenodo":
        evaluate_zenodo(args.model, args.dataset, cfg, device, logger)
    else:
        evaluate_pangaea(args.model, args.dataset, cfg, device, logger)

    logger.info("\nDone.")

if __name__ == "__main__":
    main()
