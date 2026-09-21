# Runbook — training, evaluation, results

Cluster: julia2 (University of Würzburg). Update paths/partitions for other clusters.
All SLURM scripts `cd` to `$HOME/Master_thesis/master_thesis` and
`source $HOME/Master_thesis/myenv/bin/activate`.

---

## 0. One-time environment setup

```bash
python3 -m venv ~/Master_thesis/myenv
source ~/Master_thesis/myenv/bin/activate
pip install --upgrade pip
pip install -r ~/Master_thesis/master_thesis/requirements.txt
```

`patchtst` additionally needs `tsai` (`pip install tsai`). If `tsai` is not
installed, every other model still works — only `patchtst` tasks fail.

Sanity check:

```bash
python -c "import torch, aeon, sklearn, pyts, tensorflow, tslearn; print('ok')"
python -c "from models import list_models; print(len(list_models()), 'models')"
```

---

## 1. Data (run once, already done if `dataset/processed/*.npz` exist)

```bash
python src/preprocess_bury_data.py
python src/pangea_cleaner.py
```

---

## 2. Run the whole matrix

```bash
bash submit_all.sh
```

This submits, in order:

| Stage | Script | Array | Partition |
|---|---|---|---|
| train DL | `training/train_dl_array.sh` | 0–13 (7 models × 2 datasets) | `h100` |
| train TSC | `training/train_tsc_array.sh` | 0–43 (22 models × 2 datasets) | `large_cpu` |
| eval synthetic | `testing/eval_zenodo_array.sh` | 0–57 (29 models × 2) | `large_cpu` |
| eval empirical | `testing/eval_pangaea_array.sh` | 0–57 (29 models × 2) | `large_cpu` |
| aggregate | `collect_results.py` + `plot_figures.py` | 1 job | `small_cpu` |

Eval and aggregate stages use `--dependency=afterany`, so they run even if some
training tasks fail. `evaluate.py` exits 0 when a checkpoint is missing.

Task index → model: `model = MODELS[task_id / 2]`, `dataset = ("ts_500","ts_1500")[task_id % 2]`.
The `MODELS` array is defined at the top of each script.

---

## 3. Re-run one model

```bash
# task_id = model_index_in_that_script * 2 + (0 for ts_500, 1 for ts_1500)
sbatch --array=6 training/train_tsc_array.sh          # one training task
sbatch --array=6,7 testing/eval_zenodo_array.sh       # both datasets, one model
EVAL_FORCE=--force sbatch --array=6 testing/eval_pangaea_array.sh   # overwrite existing result.json
```

`evaluate.py` skips a `(model, dataset, target[, core, sapropel, element])` that
already has a `result.json` unless `--force` is passed.

---

## 4. Collect results

```bash
python testing/collect_results.py
```

Writes:

| File | One row per |
|---|---|
| `results/summary/zenodo.csv` | (model, dataset) — binary AUC, per-class AUC, macro-F1, accuracy |
| `results/summary/pangaea.csv` | (model, dataset, core, sapropel, element) — AUC, Kendall τ, null-τ CI |
| `results/summary/train.csv` | (model, dataset[, pad_variant]) — val/test F1, params, time |
| `results/summary/coverage.csv` | (model, dataset) — train / zenodo / pangaea status |

The script prints an `incomplete (...)` list of `model/dataset` pairs still
missing a training, synthetic, or empirical result.

---

## 5. Heavy models

`boss`, `st`, `ls`, `drcif` are the ones that previously OOM'd or timed
out at full scale. They now train under explicit `max_train_samples` caps in
`config.yaml` (20 000 for `st`/`ls`, 5 000 for `boss`, 20 000 for `drcif`)
and the TSC array walltime is `2-00:00:00`. If one still fails:

- OOM → lower its `max_train_samples` in `config.yaml`, resubmit that `--array` index.
- Timeout → lower its `time_limit_in_minutes` (`boss`) or `n_estimators`
  (`drcif`, `tsf`), resubmit.

`cote` (HIVE-COTE v1) was dropped from the roster — it OOM'd on both datasets at
480 GB and is redundant with `tsf` + `st` + the standalone dictionary models.

`pf` (Proximity Forest) is capped at 5 000 in `models/tsc.py`; raise the
`config.yaml` `pf.max_train_samples` only on a node with headroom.

---

## 6. Monitoring

```bash
squeue -u $USER
tail -f logs/minirocket_ts_500_train.log
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS,ReqMem
```

SLURM stdout/stderr: `logs/slurm/<jobname>_<arrayjobid>_<taskid>.out|.err`.
Python training logs (tee'd): `logs/<model>_<dataset>_train.log`.

---

## 7. Config knobs

| Key | Effect |
|---|---|
| `inference.use_4channel` | `true` → TSC models get the 5-channel EWS suite. DL models always use 1 channel. |
| `inference.pad_mode` | `zero` (Bury default), `edge`, or `reflect`. Used by BOTH training and inference — keep them equal. |
| `augmentation.tsc_copies` | left-censored copies added per training series (TSC). |
| `augmentation.enabled` | master switch for left-censor augmentation. |
| `training.<model>.max_train_samples` | per-model cap; overrides the fallback in `models/tsc.py:TSC_SPECS`. |
| `datasets.<name>.{train,val,test}_frac` | split fractions (currently 0.95 / 0.04 / 0.01). |
