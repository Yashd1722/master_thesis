"""Aggregate every train/eval result into flat CSVs under results/summary/.

    python testing/collect_results.py [--config config.yaml]

Outputs:
    results/summary/zenodo.csv      one row per (model, dataset)
    results/summary/pangaea.csv     one row per (model, dataset, core, sapropel, element)
    results/summary/train.csv       one row per (model, dataset[, pad_variant])
    results/summary/coverage.csv    24-model x 2-dataset x {train,zenodo,pangaea} status grid
    results/summary/zenodo_confusion.csv  fold/hopf/transcritical pairwise confusion rates
"""
import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.constants import load_config

DATASETS = ["ts_500", "ts_1500"]


def _load(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _iter_result_jsons(root, suffix):
    for d in sorted(Path(root).iterdir()):
        if d.is_dir() and d.name.endswith(suffix):
            r = _load(d / "result.json")
            if r:
                yield r


def collect_zenodo(test_root):
    rows = []
    for r in _iter_result_jsons(test_root, "_zenodo"):
        if "core" in r:
            continue
        rows.append({
            "model": r.get("model"),
            "dataset": r.get("dataset"),
            "binary_auc": r.get("binary_auc", r.get("auc")),
            "macro_f1": r.get("macro_f1"),
            "macro_auc_ovr": r.get("macro_auc_ovr"),
            "accuracy": r.get("accuracy"),
            "auc_fold": (r.get("per_class_auc") or {}).get("fold"),
            "auc_hopf": (r.get("per_class_auc") or {}).get("hopf"),
            "auc_transcritical": (r.get("per_class_auc") or {}).get("transcritical"),
            "inference_time_sec": r.get("inference_time_sec"),
        })
    return rows


def collect_zenodo_confusion(test_root):
    """Pairwise class-confusion RATE (not raw count) between each pair of
    non-null classes, one row per (model, dataset).

    Rates, not raw counts, because ts_500 and ts_1500 have differently-sized
    test splits (1% of differently-sized raw datasets — 500,000 vs 200,000
    examples), so raw confusion counts are not comparable across lengths.
    """
    rows = []
    for r in _iter_result_jsons(test_root, "_zenodo"):
        if "core" in r:
            continue
        cn = r.get("class_names")
        cm = r.get("confusion_matrix")
        if cn != ["fold", "hopf", "transcritical", "null"] or not cm:
            continue
        fold_row, hopf_row, trans_row, null_row = cm
        n_fold, n_hopf, n_trans = sum(fold_row), sum(hopf_row), sum(trans_row)
        rows.append({
            "model": r.get("model"),
            "dataset": r.get("dataset"),
            "accuracy": r.get("accuracy"),
            "n_test": n_fold + n_hopf + n_trans + sum(null_row),
            "fold_trans_rate": (fold_row[2] + trans_row[0]) / (n_fold + n_trans),
            "fold_hopf_rate": (fold_row[1] + hopf_row[0]) / (n_fold + n_hopf),
            "trans_hopf_rate": (trans_row[1] + hopf_row[2]) / (n_trans + n_hopf),
        })
    return rows


def collect_pangaea(test_root):
    rows, seen = [], set()
    for r in _iter_result_jsons(test_root, "_pangaea"):
        key = (r.get("model"), r.get("dataset"), r.get("core"),
               r.get("sapropel"), r.get("element"))
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "model": r.get("model"),
            "dataset": r.get("dataset"),
            "core": r.get("core"),
            "sapropel": r.get("sapropel"),
            "element": r.get("element"),
            "binary_auc": r.get("binary_auc", r.get("auc")),
            "kendall_tau": r.get("kendall_tau"),
            "tau_null_mean": r.get("tau_null_mean"),
            "tau_null_ci_low": r.get("tau_null_ci_low"),
            "tau_null_ci_high": r.get("tau_null_ci_high"),
            "n_forced": r.get("n_forced"),
            "n_null": r.get("n_null"),
        })
    return rows


def collect_pangaea_by_model(test_root):
    from statistics import mean, median
    by_key = {}
    for r in _iter_result_jsons(test_root, "_pangaea"):
        model, dataset, element = r.get("model"), r.get("dataset"), r.get("element")
        auc = r.get("binary_auc", r.get("auc"))
        if model is None or auc is None:
            continue
        by_key.setdefault((model, dataset), []).append((element, auc))

    rows = []
    for (model, dataset), vals in sorted(by_key.items()):
        all5 = [a for _, a in vals]
        mou = [a for e, a in vals if e in ("Mo", "U")]
        rows.append({
            "model": model, "dataset": dataset,
            "primary_mou_mean_auc": round(mean(mou), 4) if mou else None,
            "primary_mou_median_auc": round(median(mou), 4) if mou else None,
            "primary_mou_n": len(mou),
            "all5_mean_auc": round(mean(all5), 4) if all5 else None,
            "all5_n": len(all5),
        })
    return rows


TRUSTWORTHY_MODELS = {
    "cnn_lstm", "lstm", "inceptiontime", "patchtst", "resnet", "tcn", "rnn_fcn",
    "arsenal", "tsf", "st", "grsf", "rocket", "minirocket", "mrsqm", "catch22",
}


def collect_pangaea_ensemble(test_root):
    from metric.auc import compute_auc
    import numpy as np

    per_segment = {}  # (core, sap, element) -> list of (p_forced, p_null)
    for d in sorted(test_root.iterdir()):
        if not (d.is_dir() and d.name.endswith("_pangaea")):
            continue
        r = _load(d / "result.json")
        if not r or r.get("model") not in TRUSTWORTHY_MODELS:
            continue
        p_f, p_n = r.get("p_transition"), r.get("p_transition_null")
        if not p_f or not p_n:
            continue
        key = (r["core"], r["sapropel"], r["element"])
        per_segment.setdefault(key, []).append((p_f, p_n))

    rows = []
    for (core, sap, element), runs in sorted(per_segment.items()):
        if len(runs) < 3:
            continue
        n_f = min(len(p_f) for p_f, _ in runs)
        n_n = min(len(p_n) for _, p_n in runs)
        ens_f = np.mean([p_f[:n_f] for p_f, _ in runs], axis=0)
        ens_n = np.mean([p_n[:n_n] for _, p_n in runs], axis=0)
        y_true = np.concatenate([np.ones(n_f), np.zeros(n_n)])
        y_score = np.concatenate([ens_f, ens_n])
        try:
            auc = float(compute_auc(y_true, y_score))
        except Exception:
            continue
        rows.append({
            "core": core, "sapropel": sap, "element": element,
            "n_models_ensembled": len(runs), "ensemble_auc": round(auc, 4),
        })
    return rows


def collect_train(results_root):
    rows = []
    for f in sorted(Path(results_root).glob("*_train_metrics.json")):
        data = _load(f)
        if data is None:
            continue
        for m in (data if isinstance(data, list) else [data]):
            rows.append({
                "model": m.get("model"),
                "dataset": m.get("dataset"),
                "pad_variant": m.get("pad_variant"),
                "best_val_f1": m.get("best_val_f1"),
                "val_f1": m.get("val_f1"),
                "val_acc": m.get("val_acc"),
                "val_balanced_acc": m.get("val_balanced_acc"),
                "test_f1": m.get("test_f1"),
                "test_acc": m.get("test_acc"),
                "n_params": m.get("n_params"),
                "training_time_min": m.get("training_time_min"),
            })
    return rows


def build_coverage(models, zen_rows, pan_rows, tr_rows):
    zen = {(r["model"], r["dataset"]) for r in zen_rows if r["binary_auc"] is not None}
    tr = {(r["model"], r["dataset"]) for r in tr_rows}
    pan = {}
    for r in pan_rows:
        pan[(r["model"], r["dataset"])] = pan.get((r["model"], r["dataset"]), 0) + 1
    rows = []
    for model in models:
        for ds in DATASETS:
            rows.append({
                "model": model,
                "dataset": ds,
                "train": "yes" if (model, ds) in tr else "MISSING",
                "zenodo": "yes" if (model, ds) in zen else "MISSING",
                "pangaea_n": pan.get((model, ds), 0),
            })
    return rows


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    test_root = REPO_ROOT / cfg["paths"]["test_results"]
    results_root = REPO_ROOT / cfg["paths"]["results"]
    out = results_root / "summary"

    models = sorted(set(cfg["models"]["dl"]) | set(cfg["models"]["tsc"]))

    zen_rows = collect_zenodo(test_root)
    pan_rows = collect_pangaea(test_root)
    tr_rows = collect_train(results_root)
    cov_rows = build_coverage(models, zen_rows, pan_rows, tr_rows)
    by_model_rows = collect_pangaea_by_model(test_root)
    ensemble_rows = collect_pangaea_ensemble(test_root)
    confusion_rows = collect_zenodo_confusion(test_root)

    _write_csv(out / "zenodo.csv", sorted(zen_rows, key=lambda r: (r["model"] or "", r["dataset"] or "")))
    _write_csv(out / "pangaea.csv", sorted(pan_rows, key=lambda r: (r["model"] or "", r["dataset"] or "", r["core"] or "", r["sapropel"] or "", r["element"] or "")))
    _write_csv(out / "train.csv", sorted(tr_rows, key=lambda r: (r["model"] or "", r["dataset"] or "")))
    _write_csv(out / "coverage.csv", cov_rows)
    _write_csv(out / "pangaea_by_model.csv", by_model_rows)
    _write_csv(out / "pangaea_ensemble.csv", ensemble_rows)
    _write_csv(out / "zenodo_confusion.csv", sorted(confusion_rows, key=lambda r: (r["model"] or "", r["dataset"] or "")))

    missing = [f"{r['model']}/{r['dataset']}" for r in cov_rows
               if "MISSING" in (r["train"], r["zenodo"]) or r["pangaea_n"] == 0]
    print(f"zenodo rows : {len(zen_rows)}")
    print(f"pangaea rows: {len(pan_rows)}")
    print(f"train rows  : {len(tr_rows)}")
    print(f"pangaea_by_model rows: {len(by_model_rows)}")
    print(f"pangaea_ensemble rows: {len(ensemble_rows)}")
    print(f"zenodo_confusion rows: {len(confusion_rows)}")
    print(f"written to  : {out}")
    if missing:
        print(f"\nincomplete ({len(missing)}):")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()
