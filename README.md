# Early Warning Signals for Mediterranean Sapropel Transitions

Master thesis — Yashkumar Dhameliya  
Reproduces and extends **Bury et al. (2021)** using both deep learning and classical
time-series classification (TSC) to detect critical transitions in Mediterranean sediment records.

---

## Papers Reproduced

| Paper | DOI | What we reproduce |
|---|---|---|
| Bury et al. (2021) PNAS | [10.1073/pnas.2106140118](https://doi.org/10.1073/pnas.2106140118) | CNN-LSTM trained on Zenodo ts_500/ts_1500, AR(1) null, applied to PANGAEA Mo records |
| Hennekam et al. (2020) PANGAEA | [10.1594/PANGAEA.923197](https://doi.pangaea.de/10.1594/PANGAEA.923197) | Empirical XRF data for 3 cores × 5 elements |

---

## Model Roster

### Deep Learning (PyTorch) — 3 models

| Model | File | Description |
|---|---|---|
| `cnn_lstm` | `models/cnn_lstm.py` | Bury 2021 baseline: Conv1D → LSTM → LSTM → softmax |
| `lstm` | `models/lstm.py` | Stacked LSTM classifier |
| `inceptiontime` | `models/inceptiontime.py` | InceptionTime (3 Inception modules + GAP + residual) |

Trained on GPU partition (`h100`). Checkpoints: `checkpoints/{model}_{dataset}_v{variant}_best.ckpt`.

### Classical TSC (aeon v1.4.0) — 7 models

All TSC models share **one** wrapper (`models/tsc.py`); each is a row in `TSC_SPECS`.

| Model | Key hyperparams | Notes |
|---|---|---|
| `minirocket` | n_kernels=10000 | |
| `multirocket` | n_kernels=6250 | |
| `arsenal` | num_kernels=2000 | |
| `drcif` | n_estimators=100 | |
| `rocket` | n_kernels=10000 | |
| `rdst` | max_shapelets=10000 | float64 cast |
| `weasel2` | WEASEL v2 | univariate: uses channel 0 only |

Trained on CPU partition (`large_cpu`, 16 CPUs, 60 GB). Checkpoints: `checkpoints/{model}_{dataset}_best.pkl`.
(`hydra_multirocket` was dropped — aeon's Hydra conv1d OOMs regardless of sample count.)

`n_jobs` is NOT hardcoded in config — `train.py` reads `SLURM_CPUS_PER_TASK` at runtime.
Training is capped at `MAX_TRAIN_SAMPLES` (stratified subsample), defined per model in `TSC_SPECS`.

---

## Repository Structure

```
master_thesis/
│
├── config.yaml                         ← single source of truth (paths, hyperparams, cores)
├── requirements.txt
├── CHANGELOG.md                        ← per-phase change log
├── MIGRATION.md                        ← guide for adapting old checkpoints/results
├── README_HPC.md                       ← HPC-specific guide (SLURM, julia2)
├── REPORT_NOTES.md                     ← thesis-writing reference (deviations, caveats)
│
├── src/
│   ├── constants.py                    ← CLASS_NAMES, NULL_IDX (canonical Bury ordering)
│   ├── data_common.py                  ← shared transforms: normalise + left-censor + fixed window
│   ├── dataset_loader.py               ← DataLoader for ts_500/ts_1500 (with augmentation)
│   ├── preprocess_bury_data.py         ← NPZ cache builder from Zenodo CSVs
│   ├── ews_augmenter.py                ← 4-channel EWS feature augmenter
│   ├── pangea_cleaner.py               ← PANGAEA XRF loader, Gaussian detrend, AR(1) null
│   └── rolling_window.py               ← rolling window EWS engine (variance, AC, DL inference)
│
├── models/
│   ├── __init__.py                     ← explicit registry: get_model(), list_models()
│   ├── cnn_lstm.py / lstm.py / inceptiontime.py  ← DL models (PyTorch)
│   └── tsc.py                          ← all aeon TSC models: one wrapper + TSC_SPECS table
│
├── metric/
│   ├── __init__.py
│   ├── accuracy.py                     ← per-class and overall accuracy
│   ├── auc.py                          ← binary_auc (forced vs AR(1) null), ovr_macro_auc
│   ├── roc.py                          ← ROC curve (Bury 2021 protocol)
│   ├── kendall_tau.py                  ← Kendall τ trend test + CI
│   └── multiclass.py                   ← macro_f1 (4-class)
│
├── training/
│   ├── train.py                        ← universal training script (DL + TSC, --binary flag)
│   ├── train_tsc_array.sh              ← SLURM array: 8 TSC × 2 datasets = 16 tasks
│   └── train_dl_array.sh               ← SLURM array: 3 DL × 2 datasets = 6 tasks
│
├── testing/
│   ├── evaluate.py                     ← Zenodo test set + PANGAEA empirical evaluation
│   ├── plot_figures.py                 ← FIG1–FIG5 (inline + summary CLI)
│   └── run_all_evaluations.sh          ← convenience: evaluate all models, then plot
│
├── dataset/
│   ├── ts_500/combined/                ← Zenodo 500-step series (labels.csv + cache_*.npy)
│   ├── ts_1500/combined/               ← same, 1500-step series
│   └── pangaea_923197/datasets/clean_dataset/  ← PANGAEA XRF CSV files
│
├── checkpoints/                        ← trained model files (gitignored)
├── results/                            ← result.json files per (model, dataset/core/sap/elem)
│   └── comparison/                     ← FIG3–FIG5 summary plots
└── logs/                               ← SLURM .out/.err + Python training logs (gitignored)
```

---

## Quick Start — Full Pipeline

### 0. Install dependencies

```bash
pip install -r requirements.txt
pip install aeon==1.4.0    # TSC models — install separately (large dependency tree)
```

### 1. Build the NPZ cache (run once)

```bash
python src/preprocess_bury_data.py
```

Processes both datasets, writing `dataset/processed/{train,val,test}_{500,1500}.npz`
as `(N, 1, L)` arrays with labels `fold=0, hopf=1, transcritical=2, null=3`.

### 2. Preprocess PANGAEA data (run once)

```bash
python src/pangea_cleaner.py
```

Outputs per (core, sapropel, element): `{core}_{sap}_forced.csv` with columns
`age_kyr_bp`, `{element}`, `{element}_trend`, `{element}_residuals`.
AR(1) null series are generated on-the-fly during evaluation (not saved to disk).

### 3. Train all models

**On SLURM (recommended):**

```bash
sbatch training/train_tsc_array.sh   # 14 tasks: 7 TSC × 2 datasets
sbatch training/train_dl_array.sh    # 6 tasks: 3 DL × 2 datasets
```

SLURM task mapping for TSC array (`--array=0-13`, `task = model_idx*2 + dataset_idx`):

| Task | Model | Dataset |
|---|---|---|
| 0–1 | minirocket | ts_500, ts_1500 |
| 2–3 | multirocket | ts_500, ts_1500 |
| 4–5 | arsenal | ts_500, ts_1500 |
| 6–7 | drcif | ts_500, ts_1500 |
| 8–9 | rocket | ts_500, ts_1500 |
| 10–11 | rdst | ts_500, ts_1500 |
| 12–13 | weasel2 | ts_500, ts_1500 |

DL array (`--array=0-5`): tasks 0–1 = cnn_lstm, 2–3 = lstm, 4–5 = inceptiontime.

> **Note:** checkpoints are skipped if they already exist — pass `--force` or clear
> `checkpoints/` before retraining after a pipeline change.

**Locally (single model):**

```bash
python training/train.py --model minirocket --dataset ts_500
python training/train.py --model cnn_lstm   --dataset ts_500
```

Optional flags:
- `--binary` — 2-class (forced vs null) mode for DL models only (Bury replication)
- `--config config.yaml` — override default config path

### 4. Run inference (all models, all targets)

```bash
bash testing/run_all_evaluations.sh
```

Or run a single evaluation:

```bash
python testing/evaluate.py --model minirocket --dataset ts_500 --target zenodo
python testing/evaluate.py --model minirocket --dataset ts_500 --target pangaea
```

Results are saved to `results/{model}_{dataset}_zenodo/result.json` and
`results/{model}_pangaea_{core}_{sap}_{element}_pangaea/result.json`.

### 5. Generate all figures

```bash
python testing/plot_figures.py --config config.yaml
```

All comparison figures saved to `results/comparison/`.

---

## Metrics Reported

### Zenodo (synthetic test set)
| Metric | Key in result.json | Description |
|---|---|---|
| `binary_auc` | `binary_auc` | AUC (forced vs AR(1) null), Bury-comparable |
| `macro_f1` | `macro_f1` | 4-class macro-averaged F1 |
| `macro_auc_ovr` | `macro_auc_ovr` | OVR macro-averaged AUC (4-class) |
| Accuracy | `accuracy_*` | Per-class and overall |

### PANGAEA (empirical)
| Metric | Key in result.json | Description |
|---|---|---|
| `binary_auc` | `binary_auc` | AUC using Bury ROC protocol (AR(1) null surrogates) |
| `kendall_tau` | `kendall_tau` | Positive = rising p_transition toward transition |
| `tau_null_mean` | `tau_null_mean` | Mean τ of AR(1) null surrogates (reference baseline) |

---

## Figure Guide

| Figure | Function | Output path | Description |
|---|---|---|---|
| FIG1 | `plot_fig1_pangaea` | `results/comparison/{model}_{core}_{sap}_{elem}_fig1.png` | 4-panel: proxy, residuals, EWS indicators, p_transition |
| FIG2 | `plot_fig2_roc` | `results/comparison/{model}_{core}_{sap}_{elem}_fig2_roc.png` | ROC curve per combination |
| FIG3 | `plot_fig3_auc_heatmap` | `results/comparison/fig3_auc_heatmap.png` | AUC heatmap: models × elements |
| FIG4 | `plot_fig4_kendall_tau` | `results/comparison/fig4_kendall_tau.png` | Kendall τ per model (mean ± 95% CI) |
| FIG5 | `plot_fig5_roc_overlay` | `results/comparison/fig5_roc_overlay_{core}_{elem}.png` | All-model ROC overlay per core |

Inline figures are also generated automatically after each `evaluate.py` call.

---

## EWS Protocol

### Training signal: p_transition

```
p_transition = 1 - P(null)   where NULL_IDX = 3  (from src/constants.py)
```

The model predicts probabilities for [fold, hopf, transcritical, null]. The
EWS signal is the probability of belonging to ANY forced class (1 − P(null)).

### AR(1) null surrogates

AR(1) fitted to the **first 20%** of the forced residuals (neutral reference
period, before CSD ramp begins). 10 surrogates generated per forced series.
The ROC is computed with forced windows as positives, null windows as negatives,
following exactly the Bury (2021) procedure.

### Causal rolling windows

All rolling-window EWS features (variance, lag-1 AC, skewness) use only past
data — no `center=True`. This avoids look-ahead bias.

---

## Datasets

### Zenodo (Bury 2021 synthetic data)

Downloaded from [zenodo.org/record/5527154](https://zenodo.org/record/5527154)

| Dataset | Series | Length | Classes |
|---|---|---|---|
| ts_500 | ~500,000 | 500 steps | fold, hopf, transcritical, null |
| ts_1500 | ~500,000 | 1500 steps | fold, hopf, transcritical, null |

Label ordering: `fold=0, hopf=1, transcritical=2, null=3` (Bury canonical).

### PANGAEA (Hennekam et al. 2020)

Downloaded from [pangaea.de](https://doi.pangaea.de/10.1594/PANGAEA.923197)

| Core | Age range | Forced sapropels | Test sapropels |
|---|---|---|---|
| MS21PC | 0–95 ka BP | S3 | S1 |
| MS66PC | 0–150 ka BP | S5, S4, S3 | S1 |
| 64PE406-E1 | 50–340 ka BP | S9, S8, S7 | S3, S4, S5, S6 |

Elements: Al, Ba, Mo, Ti, U (Mo = primary anoxia proxy; Al, Ti = lithogenic negative controls).

Use `calibratedXRF` files (7000+ rows), **not** `calibrationICP-MS` (37–295 rows).

---

## Input Pipeline (train / inference parity)

Training and empirical inference share one set of transforms (`src/data_common.py`)
so a model never sees a different kind of input than it was trained on:

1. **Normalisation** — every series is divided by its mean absolute value
   (`mean|x| = 1`). Applied identically in training and at inference.
2. **Left-censoring augmentation (Bury-style)** — each *training* series gets
   random left zero-padding (`augmentation.pad_max_frac`), keeping the tail so
   the transition stays at the end. Empirical sediment cores (57–365 points) are
   much shorter than `ts_len` (500/1500) and are left-padded to length at
   inference; without this augmentation the model would never have seen that
   shape and collapses to chance. TSC models get `1 + tsc_copies` copies per
   series; DL models are censored on the fly each epoch.
3. **Fixed window** — `make_fixed_window` takes the last `ts_len` residuals
   before each rolling position, normalises, and left-pads. `prepare_dl_input`
   (empirical inference) calls the same function.

Controlled by the `augmentation:` block in `config.yaml`.

## 4-Channel EWS Features

`use_4channel: true` (default, **TSC only**) expands the normalised/censored
residual into 4 channels — the main lever for 4-class accuracy:

```
Channel 0: raw residual (normalised)
Channel 1: rolling variance    (window = 25% of series length)
Channel 2: rolling lag-1 AC    (causal Pearson, via pandas rolling.corr)
Channel 3: rolling skewness
```

Channels are computed on the same censored/normalised residual used at inference.
Per-channel z-normalisation uses train-set statistics, saved beside the
checkpoint as `{model}_{dataset}_best_ch_stats.npz`. Toggle `use_4channel` to
A/B 1-channel vs 4-channel. DL models remain 1-channel.

---

## References

- Bury, T.M. et al. (2021). Deep learning for early warning signals of tipping points. *PNAS*, 118(39). [DOI: 10.1073/pnas.2106140118](https://doi.org/10.1073/pnas.2106140118)
- Hennekam, R. et al. (2020). PANGAEA dataset 923197. [DOI: 10.1594/PANGAEA.923197](https://doi.pangaea.de/10.1594/PANGAEA.923197)
