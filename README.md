# Early Warning Signals for Mediterranean Sapropel Transitions

Master thesis — Yashkumar Dhameliya  
Reproduces and extends **Bury et al. (2021)** and **Ma et al. (2025)** using both deep learning and classical time-series classification to detect critical transitions in Mediterranean sediment records.

---

## Papers Reproduced

| Paper | DOI | What we reproduce |
|---|---|---|
| Bury et al. (2021) PNAS | [10.1073/pnas.2023610118](https://doi.org/10.1073/pnas.2023610118) | CNN-LSTM trained on Zenodo ts_500/ts_1500, applied to PANGAEA Mo records |
| Ma et al. (2025) Comms Physics | [10.1038/s42005-025-02172-4](https://doi.org/10.1038/s42005-025-02172-4) | LSTM + CNN classifiers, SDML surrogate method, multi-element extension |

---

## Models

### Deep Learning (PyTorch) — 4 models

| Model | File | Description |
|---|---|---|
| `cnn_lstm` | `models/cnn_lstm.py` | Bury 2021 baseline: Conv1D → LSTM → LSTM |
| `lstm` | `models/lstm.py` | Ma 2025 LSTM classifier |
| `cnn` | `models/cnn.py` | Ma 2025 CNN classifier |
| `multihead_cnn` | `models/multihead_cnn.py` | Multi-head CNN (Ma 2025 extension) |

Trained on GPU partition (`h100`). Checkpoints saved as `{model}_{dataset}_v{variant}_best.ckpt`.

### Classical TSC (aeon v1.4.0) — 13 models

| Model | File | MAX_TRAIN_SAMPLES | Key hyperparams |
|---|---|---|---|
| `rocket` | `models/rocket.py` | 200,000 | n_kernels=10000, n_jobs=16 |
| `minirocket` | `models/minirocket.py` | 100,000 | n_kernels=10000, n_jobs=16 |
| `multirocket` | `models/multirocket.py` | 80,000 | n_kernels=6250, n_jobs=16 |
| `arsenal` | `models/arsenal.py` | 100,000 | n_kernels=2000, n_jobs=4 |
| `knn_dtw` | `models/knn_dtw.py` | 10,000 | n_neighbors=1, n_jobs=8 |
| `boss` | `models/boss.py` | 100,000 | max_ensemble_size=50, n_jobs=4 |
| `weasel` | `models/weasel.py` | 50,000 | window_inc=4, n_jobs=8 |
| `shapelet` | `models/shapelet.py` | 5,000 | n_shapelet_samples=100, n_jobs=8 |
| `proximity_forest` | `models/proximity_forest.py` | 10,000 | n_trees=50, n_jobs=4 |
| `ts_chief` | `models/ts_chief.py` | 50,000 | n_jobs=8 |
| `drcif` | `models/drcif.py` | 50,000 | n_estimators=100, n_jobs=4 |
| `tde` | `models/tde.py` | unlimited | n_parameter_samples=250, n_jobs=8 |
| `hivecote` | `models/hivecote.py` | 10,000 | time_limit_in_minutes=60, n_jobs=4 |

> **`MAX_TRAIN_SAMPLES`** — each TSC model caps its training set via stratified subsampling. This prevents out-of-memory kills on the 64 GB HPC nodes when the full ts_500 dataset reaches 470,000 series. Set per-model in the model file; `None`/absent means no cap.

Trained on CPU partition (`large_cpu`, 32 CPUs, 64 GB). Checkpoints saved as `{model}_{dataset}_best.pkl`.

---

## Repository Structure

```
master_thesis/
│
├── config.yaml                       ← single source of truth for all paths and hyperparameters
├── requirements.txt
│
├── src/
│   ├── dataset_loader.py             ← DataLoader for ts_500/ts_1500 + SDML surrogates
│   ├── build_cache.py                ← pre-builds numpy cache from CSVs (run once)
│   ├── pangea_cleaner.py             ← loads PANGAEA XRF data, detrends, generates AAFT null
│   └── rolling_window.py             ← rolling window EWS (variance, lag-1 AC, DL inference)
│
├── models/
│   ├── __init__.py                   ← auto-discovering registry: get_model(), get_max_train_samples()
│   ├── cnn_lstm.py                   ← DL: Bury 2021 CNN-LSTM
│   ├── lstm.py                       ← DL: Ma 2025 LSTM
│   ├── cnn.py                        ← DL: Ma 2025 CNN
│   ├── multihead_cnn.py              ← DL: Multi-head CNN
│   ├── rocket.py                     ← TSC: ROCKET
│   ├── minirocket.py                 ← TSC: MiniRocket
│   ├── multirocket.py                ← TSC: MultiRocket
│   ├── arsenal.py                    ← TSC: Arsenal (probabilistic ROCKET)
│   ├── knn_dtw.py                    ← TSC: 1-NN DTW
│   ├── boss.py                       ← TSC: BOSS ensemble
│   ├── weasel.py                     ← TSC: WEASEL (word-based)
│   ├── shapelet.py                   ← TSC: Random shapelet transform
│   ├── proximity_forest.py           ← TSC: Proximity Forest
│   ├── ts_chief.py                   ← TSC: TS-CHIEF / RIST
│   ├── drcif.py                      ← TSC: DrCIF (diverse random channel forest)
│   ├── tde.py                        ← TSC: TDE (temporal dictionary ensemble)
│   └── hivecote.py                   ← TSC: HIVE-COTE v2 (meta-ensemble)
│
├── metric/
│   ├── __init__.py
│   ├── accuracy.py                   ← accuracy scoring
│   ├── auc.py                        ← AUC computation
│   ├── kendall_tau.py                ← Kendall τ trend test
│   └── roc.py                        ← ROC curve using Bury 2021 protocol
│
├── training/
│   ├── train.py                      ← universal training script (DL + TSC)
│   ├── train_tsc_array.sh            ← SLURM array: 13 TSC models × 2 datasets (26 tasks)
│   ├── train_dl_array.sh             ← SLURM array: 4 DL models × 2 datasets (8 tasks)
│   └── train_array.sh                ← legacy combined script
│
├── testing/
│   ├── evaluate.py                   ← inference on Zenodo test set + PANGAEA cores
│   ├── compute_metrics.py            ← ROC/AUC using Bury 2021 protocol
│   ├── plot_figures.py               ← generates all thesis figures
│   ├── test_tsc_array.sh             ← SLURM array: 13 TSC models × 2 datasets (26 tasks)
│   ├── test_dl_array.sh              ← SLURM array: 4 DL models × 2 datasets (8 tasks)
│   └── test_array.sh                 ← legacy combined script
│
├── dataset/
│   ├── ts_500/combined/              ← Zenodo 500-step series
│   │   ├── labels.csv
│   │   ├── output_resids/            ← 500,000 residual CSVs
│   │   ├── cache_residuals.npy       ← pre-built cache (run build_cache.py once)
│   │   └── cache_labels.npy
│   ├── ts_1500/combined/             ← same structure, 1500-step series
│   └── pangaea_923197/
│       └── datasets/
│           └── clean_dataset/        ← PANGAEA XRF files + processed segments
│               ├── MS21PC_calibratedXRF.csv
│               ├── MS66PC_calibratedXRF.csv
│               └── 64PE406-E1_calibratedXRF.csv
│
├── checkpoints/                      ← trained model files (excluded from git)
├── metrics/                          ← JSON ROC metrics + AUC CSV tables
├── results/                          ← prediction CSVs per (model, core, sapropel, element)
├── logs/                             ← SLURM .out / .err + Python training logs
└── test_results/                     ← all figure outputs (.png)
```

---

## Quick Start — Reproduce All Results

### Step 0 — Install dependencies
```bash
pip install -r requirements.txt
pip install aeon==1.4.0        # TSC models — install separately (large dependency)
```

### Step 1 — Build numpy cache (run once, ~10 minutes)
```bash
python src/build_cache.py --dataset both
```

### Step 2 — Preprocess PANGAEA data
```bash
python src/pangea_cleaner.py
```
Outputs per sapropel per element: `{core}_{sap}_forced.csv` and `{core}_{sap}_{element}_ar1_null.csv` (20 AAFT null series).

### Step 3 — Train all models

**On SLURM (HPC — recommended):**
```bash
# Deep Learning models (GPU partition)
sbatch training/train_dl_array.sh    # 8 tasks: 4 models × 2 datasets

# Classical TSC models (CPU partition, 64 GB)
sbatch training/train_tsc_array.sh   # 26 tasks: 13 models × 2 datasets
```

**SLURM task mapping for TSC array (task 0–25):**

| Task | Model | Dataset |
|---|---|---|
| 0 | rocket | ts_500 |
| 1 | rocket | ts_1500 |
| 2 | minirocket | ts_500 |
| 3 | minirocket | ts_1500 |
| 4 | multirocket | ts_500 |
| 5 | multirocket | ts_1500 |
| 6 | arsenal | ts_500 |
| 7 | arsenal | ts_1500 |
| 8 | knn_dtw | ts_500 |
| 9 | knn_dtw | ts_1500 |
| 10 | boss | ts_500 |
| 11 | boss | ts_1500 |
| 12 | weasel | ts_500 |
| 13 | weasel | ts_1500 |
| 14 | shapelet | ts_500 |
| 15 | shapelet | ts_1500 |
| 16 | proximity_forest | ts_500 |
| 17 | proximity_forest | ts_1500 |
| 18 | ts_chief | ts_500 |
| 19 | ts_chief | ts_1500 |
| 20 | drcif | ts_500 |
| 21 | drcif | ts_1500 |
| 22 | tde | ts_500 |
| 23 | tde | ts_1500 |
| 24 | hivecote | ts_500 |
| 25 | hivecote | ts_1500 |

**Locally (single model):**
```bash
python training/train.py --model rocket    --dataset ts_500
python training/train.py --model cnn_lstm  --dataset ts_1500
```

### Step 4 — Run inference
```bash
# On SLURM:
sbatch testing/test_dl_array.sh
sbatch testing/test_tsc_array.sh

# Locally:
python testing/evaluate.py --model cnn_lstm --dataset ts_500 --target zenodo
python testing/evaluate.py --model cnn_lstm --dataset ts_500 --target pangaea
```

### Step 5 — Compute ROC/AUC metrics
```bash
python testing/compute_metrics.py --all --dataset ts_500
```
Outputs:
- `metrics/auc_Mo_primary.csv` — Mo-only (Bury 2021 comparison)
- `metrics/auc_extension_elements.csv` — Al, Ba, Ti, U (thesis extension)
- `metrics/auc_comparison_all_models.csv` — full table across all 17 models

### Step 6 — Generate all figures
```bash
python testing/plot_figures.py --model all --dataset ts_500
```

---

## Figure Guide

| Figure | Filename pattern | Description |
|---|---|---|
| Fig 2 | `{model}_ts_500_fold_fig2.png` | Fold bifurcation test + confusion matrices |
| Fig 3 | `pangaea_overview_fig3.png` | PANGAEA Mo time series, colour-coded by sapropel role |
| Fig 4 | `{model}_{core}_{element}_fig4.png` | 4-panel EWS indicators per element per core |
| Fig 5 | `{model}_{core}_{element}_roc_fig5.png` | ROC curves per model |
| Fig 5 | `all_models_{core}_{element}_roc_fig5.png` | All-model ROC comparison |

---

## ROC Protocol — Bury 2021 Exact Method

1. **Positive class** — predictions from the last `(1 - roc_start_frac)` of the pre-transition window. Default `roc_start_frac=0.60` uses the 60–100% window.
2. **Negative class** — predictions from the same relative window of 20 AAFT null series generated from the first 20% of the pre-transition residuals.
3. **Combining events** — predictions from all test sapropels within a core are pooled before computing ROC.

Configurable in `config.yaml` under `inference.roc_start_frac`.

**Key deviation from Bury:** Bury combines 26 transitions (N≈800). We have 7 test transitions (N≈110–320). ROC curves are therefore coarser but directionally consistent.

---

## Results Summary — Mo Element (Bury 2021 Comparison)

| Core | Model | AUC | Bury 2021 AUC |
|---|---|---|---|
| MS21 | CNN-LSTM | **0.98** | 0.99 |
| MS21 | LSTM | **0.99** | n/a |
| MS21 | CNN | 0.58 | n/a |
| MS66 | CNN-LSTM | ~0.65–0.89 | ~0.97 |
| 64PE406E1 | CNN-LSTM | ~0.52 | ~0.85 |

DL models substantially outperform classical CSD indicators (variance, lag-1 AC) on MS21/Mo, consistent with Bury's main finding. TSC model results pending.

---

## Extension Beyond Bury — Five Elements

| Element | Role | EWS expected? |
|---|---|---|
| **Mo** | Primary anoxia proxy | Yes — main signal |
| Ba | Productivity proxy | Partial |
| U | Secondary redox proxy | Weak |
| Al | Lithogenic input | No — negative control |
| Ti | Lithogenic input | No — negative control |

Al and Ti serve as negative controls — they should show no EWS, validating the method.

---

## Training Data (Zenodo)

Downloaded from [zenodo.org/record/5527154](https://zenodo.org/record/5527154)

| Dataset | Series | Length | Classes |
|---|---|---|---|
| ts_500 | ~500,000 | 500 steps | fold, hopf, transcritical, null |
| ts_1500 | ~200,000 | 1500 steps | fold, hopf, transcritical, null |

Classes generated from randomly parameterised polynomial dynamical systems using AUTO-07P. Residuals are Lowess-detrended by Bury before release.

---

## PANGAEA Data

Downloaded from [pangaea.de](https://doi.pangaea.de/10.1594/PANGAEA.923197) (Hennekam et al. 2020)

| Core | Rows | Age range | Test sapropels |
|---|---|---|---|
| MS21PC | 7,460 | 0–95 ka BP | S1 |
| MS66PC | 5,489 | 0–150 ka BP | S1, S3 |
| 64PE406-E1 | 7,672 | 50–340 ka BP | S3, S4, S5, S6 |

> Use the `calibratedXRF` files (7000+ rows), **not** `calibrationICP-MS` (37–295 rows — calibration standards only).

---

## HPC Notes (University of Würzburg — julia2)

- **TSC jobs**: `large_cpu` partition, 32 CPUs, 64 GB RAM, up to 2 days wall time
- **DL jobs**: `h100` partition, 1 GPU, 4 CPUs, 32 GB RAM
- `NUMBA_NUM_THREADS=32` is set explicitly in the TSC SLURM script to prevent numba from reading `os.cpu_count()` (256 on large nodes) and exceeding the thread limit
- TSC models with large feature spaces (MultiRocket, WEASEL) use `MAX_TRAIN_SAMPLES` to keep the feature matrix within 64 GB

---

## Deviations from Papers

| Deviation | Reason |
|---|---|
| N=110–320 vs Bury's N=800 | Fewer test cores/sapropels available |
| AAFT null (20 series) vs AR(1) null | AAFT is more statistically rigorous |
| 5 elements vs Mo only | Extended scope for thesis |
| roc_start_frac=0.60 (configurable) | Matches Bury's "60–100%" protocol |
| CNN-LSTM DL probability saturates at 1.0 on PANGAEA | Distribution shift from synthetic training data |

---

## References

- Bury, T.M. et al. (2021). Deep learning for early warning signals of tipping points. *PNAS*, 118(39). [DOI: 10.1073/pnas.2023610118](https://doi.org/10.1073/pnas.2023610118)
- Ma, H. et al. (2025). Self-supervised deep learning for early warning signals. *Communications Physics*, 8(1). [DOI: 10.1038/s42005-025-02172-4](https://doi.org/10.1038/s42005-025-02172-4)
- Hennekam, R. et al. (2020). PANGAEA dataset 923197. [DOI: 10.1594/PANGAEA.923197](https://doi.pangaea.de/10.1594/PANGAEA.923197)
