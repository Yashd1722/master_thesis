# Early Warning Signals for Mediterranean Sapropel Transitions

Master thesis — Yashkumar Dhameliya  
Reproduces and extends **Bury et al. (2021)** and **Ma et al. (2025)** using deep learning to detect critical transitions in Mediterranean sediment records.

---

## Papers Reproduced

| Paper | DOI | What we reproduce |
|---|---|---|
| Bury et al. (2021) PNAS | [10.1073/pnas.2023610118](https://doi.org/10.1073/pnas.2023610118) | CNN-LSTM classifier trained on Zenodo ts_500, applied to PANGAEA Mo records |
| Ma et al. (2025) Comms Physics | [10.1038/s42005-025-02172-4](https://doi.org/10.1038/s42005-025-02172-4) | LSTM + CNN classifiers, SDML surrogate method, multi-element extension |

---

## Repository Structure

```
master_thesis/
│
├── config.yaml                    ← single source of truth for all paths and hyperparameters
│
├── src/
│   ├── dataset_loader.py          ← DataLoader for ts_500/ts_1500 + SDML surrogates
│   ├── build_cache.py             ← pre-builds numpy cache from 500k CSVs (run once)
│   ├── pangea_cleaner.py          ← loads PANGAEA XRF data, detrends, generates AAFT null
│   └── rolling_window.py          ← rolling window EWS (variance, lag-1 AC, DL inference)
│
├── models/
│   ├── __init__.py                ← get_model() registry
│   ├── cnn_lstm.py                ← Bury 2021 baseline (Conv1D → LSTM → LSTM)
│   ├── lstm.py                    ← Ma 2025 LSTM
│   └── cnn.py                     ← Ma 2025 CNN
│
├── training/
│   ├── train.py                   ← universal training loop (--model argument)
│   └── train_array.sh             ← SLURM array: 3 jobs × (ts_500 then ts_1500)
│
├── testing/
│   ├── evaluate.py                ← inference on PANGAEA cores × 5 elements × 2 segments
│   ├── compute_metrics.py         ← ROC/AUC using Bury 2021 protocol
│   ├── plot_figures.py            ← generates Fig 2, 3, 4, 5
│   └── test_array.sh              ← SLURM array for testing
│
├── dataset/
│   ├── ts_500/combined/           ← Zenodo training data (500-step series)
│   │   ├── labels.csv
│   │   ├── output_resids/         ← 500,000 residual CSVs
│   │   ├── cache_residuals.npy    ← pre-built cache (run build_cache.py once)
│   │   └── cache_labels.npy
│   ├── ts_1500/combined/          ← same structure, 1500 steps
│   └── pangaea_923197/
│       └── datasets/
│           └── clean_dataset/     ← PANGAEA XRF files + processed segments
│               ├── MS21PC_calibratedXRF.csv
│               ├── MS66PC_calibratedXRF.csv
│               ├── 64PE406-E1_calibratedXRF.csv
│               ├── MS21/
│               │   ├── MS21_S1_forced.csv        ← pre-transition segment
│               │   ├── MS21_S1_neutral_test.csv  ← inter-sapropel segment
│               │   └── MS21_S1_Mo_ar1_null.csv   ← 20 AAFT null series
│               ├── MS66/
│               └── 64PE406E1/
│
├── checkpoints/                   ← {model}_{dataset}_v{variant}_best.ckpt
├── metrics/                       ← JSON metrics + AUC CSV tables
├── results/                       ← prediction CSVs per (model, core, sapropel, element)
└── test_results/                  ← all figures (.png)
```

---

## Quick Start — Reproduce All Results

### Step 0 — Install dependencies
```bash
pip install torch numpy pandas scipy scikit-learn matplotlib pyyaml
```

### Step 1 — Build numpy cache (run once, ~10 minutes)
```bash
python src/build_cache.py --dataset both
```

### Step 2 — Preprocess PANGAEA data
```bash
python src/pangea_cleaner.py
```
Outputs per sapropel per element:
- `{core}_{sap}_forced.csv` — pre-transition segment
- `{core}_{sap}_{element}_ar1_null.csv` — 20 AAFT null series

### Step 3 — Train all models
```bash
# On SLURM (recommended):
sbatch training/train_array.sh

# Locally (one model at a time):
python training/train.py --model cnn_lstm
python training/train.py --model lstm
python training/train.py --model cnn
```

### Step 4 — Run inference on PANGAEA
```bash
python testing/evaluate.py --model cnn_lstm --dataset ts_500
python testing/evaluate.py --model lstm     --dataset ts_500
python testing/evaluate.py --model cnn      --dataset ts_500
```

### Step 5 — Compute ROC/AUC metrics
```bash
python testing/compute_metrics.py --all --dataset ts_500
```
Outputs:
- `metrics/auc_Mo_primary.csv` — Mo-only (Bury 2021 comparison)
- `metrics/auc_extension_elements.csv` — Al, Ba, Ti, U (original extension)
- `metrics/auc_comparison_all_models.csv` — full table

### Step 6 — Generate all figures
```bash
# Figure 3 only (no model needed):
python testing/plot_figures.py --fig3_only

# All figures:
python testing/plot_figures.py --model all --dataset ts_500
```

---

## Figure Guide

| Figure | Filename pattern | Description |
|---|---|---|
| Fig 2 | `{model}_ts_500_fold_fig2.png` | Fold bifurcation test on ts_500 + confusion matrices |
| Fig 3 | `pangaea_overview_fig3.png` | PANGAEA Mo time series, colour-coded by sapropel role |
| Fig 4 | `{model}_{core}_{element}_fig4.png` | 4-panel EWS indicators per element per core |
| Fig 5 | `{model}_{core}_{element}_roc_fig5.png` | ROC curves per model |
| Fig 5 | `all_models_{core}_{element}_roc_fig5.png` | All-model ROC comparison |

---

## ROC Protocol — Bury 2021 Exact Method

The ROC evaluation follows Bury et al. (2021) exactly:

1. **Positive class** — predictions from the last `(1 - roc_start_frac)` of the pre-transition window. Default `roc_start_frac=0.60` uses the 60–100% window.
2. **Negative class** — predictions from the same relative window of 20 AAFT null series. Each null series is generated from the first 20% of the pre-transition residuals (baseline before CSD builds up).
3. **Combining events** — predictions from all test sapropels within a core are pooled before computing ROC.

Configurable in `config.yaml` under `inference.roc_start_frac`.

**Key deviation from Bury:** Bury combines 26 transitions (N≈800). We have 7 test transitions (N≈110–320). ROC curves are therefore coarser (staircased) but directionally consistent.

---

## Results Summary — Mo Element (Bury 2021 Comparison)

| Core | Model | AUC | Bury 2021 AUC |
|---|---|---|---|
| MS21 | CNN-LSTM | **0.98** | 0.99 |
| MS21 | LSTM | **0.99** | n/a |
| MS21 | CNN | 0.58 | n/a |
| MS66 | CNN-LSTM | ~0.65–0.89 | ~0.97 |
| 64PE406E1 | CNN-LSTM | ~0.52 | ~0.85 |

DL models substantially outperform classical CSD indicators (variance, lag-1 AC) on MS21/Mo, consistent with Bury's main finding.

---

## Extension Beyond Bury — Five Elements

Unlike Bury (2021) who tested only Mo, this thesis evaluates all 5 geochemical proxies:

| Element | Role | EWS expected? |
|---|---|---|
| **Mo** | Primary anoxia proxy | **Yes** — main signal |
| Ba | Productivity proxy | Partial |
| U | Secondary redox proxy | Weak |
| Al | Lithogenic input | No — control |
| Ti | Lithogenic input | No — control |

Al and Ti serve as negative controls — they should show no EWS, which validates the method.

---

## Training Data (Zenodo)

Downloaded from [zenodo.org/record/5527154](https://zenodo.org/record/5527154)

| Dataset | Series | Length | Classes |
|---|---|---|---|
| ts_500 | 500,000 | 500 steps | fold, hopf, transcritical, null |
| ts_1500 | 200,000 | 1500 steps | fold, hopf, transcritical, null |

Classes generated from randomly parameterised polynomial dynamical systems using AUTO-07P. Residuals are Lowess-detrended by Bury before release.

---

## PANGAEA Data

Downloaded from [pangaea.de](https://doi.pangaea.de/10.1594/PANGAEA.923197) (Hennekam et al. 2020)

| Core | Rows | Age range | Test sapropels |
|---|---|---|---|
| MS21PC | 7460 | 0–95 ka BP | S1 |
| MS66PC | 5489 | 0–150 ka BP | S1, S3 |
| 64PE406-E1 | 7672 | 50–340 ka BP | S3, S4, S5, S6 |

**Important:** Use `calibratedXRF` files (7000+ rows), NOT `calibrationICP-MS` (37–295 rows, calibration standards only).

---

## Deviations from Papers

| Deviation | Reason |
|---|---|
| N=110–320 vs Bury's N=800 | Fewer test cores/sapropels |
| AAFT null (20 series) vs AR(1) null | AAFT more statistically rigorous |
| 5 elements vs Mo only | Extended scope for thesis |
| roc_start_frac=0.60 (configurable) | Matches Bury's "60-100%" protocol |
| CNN-LSTM DL probability saturates at 1.0 on PANGAEA | Distribution shift from synthetic training data |

---

## References

- Bury, T.M. et al. (2021). Deep learning for early warning signals of tipping points. *PNAS*, 118(39). [DOI: 10.1073/pnas.2023610118](https://doi.org/10.1073/pnas.2023610118)
- Ma, H. et al. (2025). Self-supervised deep learning for early warning signals. *Communications Physics*, 8(1). [DOI: 10.1038/s42005-025-02172-4](https://doi.org/10.1038/s42005-025-02172-4)
- Hennekam, R. et al. (2020). PANGAEA dataset 923197. [DOI: 10.1594/PANGAEA.923197](https://doi.pangaea.de/10.1594/PANGAEA.923197)
