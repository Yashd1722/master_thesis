# Early Warning Signals for Mediterranean Sapropel Transitions

Master thesis — Yashkumar Dhameliya  
Reproduces and extends **Bury et al. (2021)** using deep learning and classical time-series classification (TSC) algorithms from the *Great Time Series Classification Bake-Off* (Bagnall et al., 2017) to detect critical transitions in Mediterranean sediment records.

---

## Papers & Benchmarks Reproduced

| Paper | DOI | What We Reproduce / Extend |
|---|---|---|
| Bury et al. (2021) PNAS | [10.1073/pnas.2106140118](https://doi.org/10.1073/pnas.2106140118) | Deep learning for early warning signals of tipping points (Zenodo & empirical sediment core evaluation) |
| Bagnall et al. (2017) Data Mining & Knowl. Disc. | [10.1007/s10618-016-0483-9](https://doi.org/10.1007/s10618-016-0483-9) | *The Great Time Series Classification Bake-Off*: 22 classical TSC & Bake-Off classifiers evaluated on EWS signal detection |
| Hennekam et al. (2020) PANGAEA | [10.1594/PANGAEA.923197](https://doi.pangaea.de/10.1594/PANGAEA.923197) | High-resolution XRF sediment core geochemical data (3 Mediterranean cores × 5 elements) |

---

## Model Roster (29)

### 1. Deep Learning Models (PyTorch) — 7 Models
Trained on GPU partition (`h100`). Checkpoints saved to `checkpoints/{model}_{dataset}_v{variant}_best.ckpt`.
DL models consume the single normalised residual channel `(B, 1, L)`.

| Model | File | Description | Key Architectural Features |
|---|---|---|---|
| `cnn_lstm` | [`models/cnn_lstm.py`](file:///home/s466553/Master_thesis/master_thesis/models/cnn_lstm.py) | Bury 2021 baseline | Conv1D → MaxPool → LSTM(50) → LSTM(10) → Linear; classifier reads the hidden state at the final timestep, where the left-padded input holds the approach to the transition |
| `lstm` | [`models/lstm.py`](file:///home/s466553/Master_thesis/master_thesis/models/lstm.py) | Stacked Recurrent Network | Linear(1→128) → LSTM(128) → LSTM(64) → Linear; final-timestep readout |
| `inceptiontime` | [`models/inceptiontime.py`](file:///home/s466553/Master_thesis/master_thesis/models/inceptiontime.py) | InceptionTime Network | 3 Inception modules + Global Average Pooling + Residual connections |
| `patchtst` | [`models/patchtst.py`](file:///home/s466553/Master_thesis/master_thesis/models/patchtst.py) | Patch Transformer | `tsai` PatchTST, patch_len 16 / stride 8, 3 layers, 4 heads (requires `tsai`) |
| `resnet` | [`models/resnet.py`](file:///home/s466553/Master_thesis/master_thesis/models/resnet.py) | Wang 2017 ResNet baseline | 3 residual blocks {64,128,128}, kernels {8,5,3}, GAP → Linear; layer-for-layer match to Fawaz `dl-4-tsc` (see `ARCHITECTURE_COMPARISON.md`) |
| `tcn` | [`models/tcn.py`](file:///home/s466553/Master_thesis/master_thesis/models/tcn.py) | Bai 2018 Temporal ConvNet | 8 dilated causal residual blocks, `weight_norm`, dilation 2^i, last-timestep readout; block/init match to `locuslab/TCN` |
| `rnn_fcn` | [`models/rnn_fcn.py`](file:///home/s466553/Master_thesis/master_thesis/models/rnn_fcn.py) | Karim 2018 LSTM-FCN | dimension-shuffled LSTM(128) ∥ FCN {128,256,128}, concat → Linear; match to `titu1994/LSTM-FCN` and `tsai` |

### 2. Classical & Bake-Off TSC Models — 22 Models
Trained on CPU partition (`large_cpu`, 16 CPUs, 150 GB RAM). Checkpoints saved to `checkpoints/{model}_{dataset}_best.pkl`.
TSC models consume the 5-channel EWS feature suite (see Improvement 2) flattened to `(N, 5L)`.

| Model | File / Class | Category | Key Features / Scalability Knobs |
|---|---|---|---|
| `minirocket` | `models/tsc.py` | Random Convolution | 10,000 MiniRocket kernels + Softmax `decision_function` calibration |
| `multirocket` | `models/tsc.py` | Random Convolution | 6,250 MultiRocket kernels (first/second order differences) |
| `rocket` | `models/tsc.py` | Random Convolution | 10,000 original Rocket kernels |
| `arsenal` | `models/tsc.py` | Kernel Ensemble | 2,000 MiniRocket ensemble classifiers |
| `drcif` | `models/tsc.py` | Canonical Interval | Diverse Representation Canonical Interval Forest (100 trees) |
| `weasel2` | `models/tsc.py` | Dictionary (SFA) | WEASEL v2 multi-resolution word histograms |
| `rdst` | `models/tsc.py` | Shapelet | Random Dilation Shapelet Transform (1,000 shapelets) |
| `tsf` | `models/tsc.py` | Summary Interval | Time Series Forest (200 trees) |
| `st` | `models/tsc.py` | Shapelet Transform | Contractable Shapelet Transform Classifier |
| `ls` | `models/tsc.py` | Shapelet | Learned Shapelets |
| `boss` / `cboss` | `models/tsc.py` | Dictionary (SFA) | Contractable Bag of SFA Symbols |
| `bop` | [`models/bakeoff_dictionary.py`](file:///home/s466553/Master_thesis/master_thesis/models/bakeoff_dictionary.py) | Dictionary | Bag of Patterns (1-NN on SFA word histograms) |
| `saxvsm` | [`models/bakeoff_dictionary.py`](file:///home/s466553/Master_thesis/master_thesis/models/bakeoff_dictionary.py) | Dictionary | SAX-VSM (Vector Space Model with cosine similarity) |
| `tsbf` | [`models/bakeoff_tsbf.py`](file:///home/s466553/Master_thesis/master_thesis/models/bakeoff_tsbf.py) | Subseries Bagging | Time Series Bag of Features |
| `lps` | [`models/bakeoff_lps.py`](file:///home/s466553/Master_thesis/master_thesis/models/bakeoff_lps.py) | Tree Distance | Learned Pattern Similarity (randomized tree-based histograms) |
| `fastshapelet` | [`models/bakeoff_fastshapelet.py`](file:///home/s466553/Master_thesis/master_thesis/models/bakeoff_fastshapelet.py) | Shapelet | Fast Shapelet Discovery via SAX dimensionality reduction |
| `catch22` | `models/tsc.py` | Feature Based | 22 canonical time-series features + classifier |
| `tde` | `models/tsc.py` | Dictionary (SFA) | Temporal Dictionary Ensemble |
| `pf` | `models/tsc.py` | Distance Based | Proximity Forest (elastic-distance trees) |
| `cif` | `models/tsc.py` | Canonical Interval | Canonical Interval Forest (predecessor of `drcif`; single representation) |
| `mrsqm` | `models/tsc.py` | Symbolic Sequence | Multiple Representation Sequence Miner (SAX+SFA + linear SEQL; requires `mrsqm`) |
| `grsf` | `models/tsc.py` (wildboar) | Shapelet Forest | Generalized Random Shapelet Forest (shapelet split embedded in tree nodes; requires `wildboar`) |

---

## Key Pipeline & Architectural Improvements

1. **Continuous Probability Calibration for Rocket Family**:
   * Replaced hard step-function binary outputs from `RidgeClassifierCV` with continuous decision function probabilities (`scipy.special.softmax(decision_function(X))`), enabling meaningful ROC AUC scoring.
2. **Multivariate 5-Channel EWS Feature Suite (TSC models)**:
   * `src/ews_augmenter.py` expands the residual into $(N, 5, L)$ — raw residual, rolling variance, rolling lag-1 AC, rolling skewness, and the Variance Growth Ratio $\nabla V / (V + \epsilon)$ — flattened to $(N, 5L)$ so the univariate TSC classifiers ingest all five. Per-channel z-norm stats are fit on train and reused for val/test/empirical (`{model}_{dataset}_best_ch_stats.npz`).
3. **Geological Noise & Bioturbation Left-Censoring Augmentation**:
   * Enhanced `random_left_censor()` with AR(1) red noise ($\phi \in [0.3, 0.7]$) and 3-point Gaussian smoothing to bridge the synthetic-to-empirical (Sim-to-Real) domain gap.
4. **Right-Aligned Left-Padded Sequences**:
   * Training (`src/data_common.py`) and inference (`src/rolling_window.py`) share one transform: normalise the visible signal to mean$|x|=1$, then left-pad to `ts_len` (`pad_mode` from `config.yaml`, default `zero` per Bury 2021). The transition always sits at the final timestep, which the recurrent classifiers read out directly.
5. **Memory Buffer Pre-Allocation & Slurm Thread Pinning**:
   * Pre-allocated array memory buffers in chunked transformations (cutting peak RAM by 50%) and explicitly pinned `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and `BLIS_NUM_THREADS`.

---

## Clean Directory Map

```text
master_thesis/
├── config.yaml                    # Central configuration file (paths, hyperparameters, cores)
├── requirements.txt               # Environment dependencies
├── README.md                      # Primary documentation
├── README_HPC.md                  # HPC Slurm cluster guide
├── REPORT_NOTES.md                # Thesis notes & architectural comparisons
├── models/                        # Model architectures & Bake-Off classifiers
│   ├── tsc.py                     # Aeon & scikit-learn model wrapper (Rocket, Arsenal, DrCIF, etc.)
│   ├── cnn_lstm.py                # PyTorch CNN-LSTM architecture
│   ├── lstm.py                    # PyTorch LSTM architecture
│   ├── inceptiontime.py           # PyTorch InceptionTime architecture
│   ├── bakeoff_dictionary.py      # BOP & SAX-VSM scratch implementations
│   ├── bakeoff_fastshapelet.py    # FastShapelet scratch implementation
│   ├── bakeoff_lps.py             # LPS scratch implementation
│   └── bakeoff_tsbf.py            # TSBF scratch implementation
├── src/                           # Preprocessing & augmentation utilities
│   ├── constants.py               # Canonical class names & index definitions
│   ├── data_common.py             # Left-censoring augmentation (with red noise + bioturbation)
│   ├── dataset_loader.py          # PyTorch DataLoader & EWSDataset
│   ├── ews_augmenter.py           # Rolling EWS feature extraction
│   ├── pangea_cleaner.py          # PANGAEA sediment core preprocessing & AAFT null surrogates
│   ├── preprocess_bury_data.py    # Bury (2021) synthetic dataset preprocessor
│   └── rolling_window.py          # Causal rolling-window EWS inference engine
├── submit_all.sh                  # One command: train + eval + aggregate (Slurm, all 24 models)
├── RUNBOOK.md                     # Step-by-step run guide and troubleshooting
├── training/                      # Unified training pipeline & Slurm scripts
│   ├── train.py                   # Central training runner for DL & TSC models
│   ├── train_tsc_array.sh         # Slurm array: 20 TSC models x 2 datasets
│   └── train_dl_array.sh          # Slurm array: 4 DL models x 2 datasets
├── testing/                       # Evaluation pipeline & Slurm scripts
│   ├── evaluate.py                # Evaluation runner for Zenodo & PANGAEA test sets
│   ├── eval_zenodo_array.sh       # Slurm array: 24 models x 2 datasets (synthetic)
│   ├── eval_pangaea_array.sh      # Slurm array: 24 models x 2 datasets (empirical)
│   ├── collect_results.py         # Aggregate all result.json into results/summary/*.csv
│   └── plot_figures.py            # Publication figure & plot generation
├── metric/                        # Standard evaluation metrics
│   ├── auc.py                     # Binary & OVR macro-averaged ROC AUC metrics
│   ├── roc.py                     # ROC curve calculation utilities
│   └── kendall_tau.py             # Kendall tau rank correlation metrics
├── dataset/                       # Raw & processed data directories
├── checkpoints/                   # Saved model weights & feature scalers
└── logs/                          # Execution & Slurm job logs
```

---

## Quick Start — Running the Pipeline

### 0. Everything at once (Slurm)
```bash
bash submit_all.sh
```
Submits DL + TSC training, chains synthetic + empirical evaluation after them,
and runs `collect_results.py` + `plot_figures.py` at the end. See `RUNBOOK.md`
for the step-by-step version, per-model reruns, and troubleshooting.

### 1. Build Synthetic Data Cache (Bury et al. 2021)
```bash
python src/preprocess_bury_data.py
```

### 2. Preprocess PANGAEA Sediment Data & Build AAFT Surrogates
```bash
python src/pangea_cleaner.py
```

### 3. Model Training

**Train Deep Learning Models on GPU (Slurm):**
```bash
sbatch training/train_dl_array.sh
```

**Train Classical & Bake-Off TSC Models on CPU (Slurm):**
```bash
sbatch training/train_tsc_array.sh
```

**Run Locally (Single Model):**
```bash
python training/train.py --model minirocket --dataset ts_500 --force
python training/train.py --model cnn_lstm   --dataset ts_500 --force
```

### 4. Evaluation & Inference

**Run Evaluation Arrays on HPC (Slurm):**
```bash
sbatch testing/eval_zenodo_array.sh
sbatch testing/eval_pangaea_array.sh
```

**Run Single Evaluation Locally:**
```bash
python testing/evaluate.py --model minirocket --dataset ts_500 --target zenodo
python testing/evaluate.py --model minirocket --dataset ts_500 --target pangaea
```

### 5. Generate Figures
```bash
python testing/plot_figures.py --config config.yaml
```

---

## References

- Bury, T.M. et al. (2021). Deep learning for early warning signals of tipping points. *PNAS*, 118(39). [DOI: 10.1073/pnas.2106140118](https://doi.org/10.1073/pnas.2106140118)
- Bagnall, A. et al. (2017). The great time series classification bake off: a review and experimental evaluation of recent algorithmic advances. *Data Mining and Knowledge Discovery*, 31(3), 606–660. [DOI: 10.1007/s10618-016-0483-9](https://doi.org/10.1007/s10618-016-0483-9)
- Hennekam, R. et al. (2020). PANGAEA dataset 923197. [DOI: 10.1594/PANGAEA.923197](https://doi.pangaea.de/10.1594/PANGAEA.923197)
