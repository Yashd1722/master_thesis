# Thesis Report Notes

Quick-reference for writing Chapter 3 (Methods) and Chapter 4 (Results).
All deviations from Bury (2021) are flagged. All design choices are justified.

---

## Section: Data

### Zenodo training data

- Source: Bury et al. (2021), [zenodo.org/record/5527154](https://zenodo.org/record/5527154)
- 4-class labels: fold=0, hopf=1, transcritical=2, null=3 (canonical Bury ordering)
- Series are pre-detrended residuals (Lowess) by Bury before release
- ts_500: ~500k series × 500 steps; ts_1500: ~500k series × 1500 steps
- Split: 95% train / 4% val / 1% test (stratified, config-controlled)
- After flat-series filtering and optional degenerate-channel drop: effectively
  ~473k series after filtering for ts_500

### PANGAEA empirical data (Hennekam et al. 2020)

- 3 sediment cores: MS21PC, MS66PC, 64PE406-E1
- 5 elements: Al, Ba, Mo, Ti, U (Mo = primary; Al, Ti = lithogenic negative controls)
- Gaussian detrending: bandwidth = 20% of record length (matching Bury's `compute_ews.py`)
- Forced segments: full pre-transition record from `neutral_start` to `transition_kyr`
- Neutral reference period: first portion (neutral_start → neutral_end) used to fit AR(1)
- **Deviation from Bury:** Bury uses 26 transitions across 30 paleorecords; we have
  11 total (4 train + 7 test across 3 cores). ROC curves are coarser but comparable.

---

## Section: Methods

### EWS signal: p_transition

```
p_transition(t) = 1 − P(null | window ending at t)
```

The null class (index 3 in Bury's ordering) represents no-tipping-point dynamics.
The complement is the probability of an impending transition of any type.

### AR(1) null surrogates (Bury parity)

- AR(1) coefficient fitted to the **first 20%** of forced residuals (the neutral
  reference period before CSD signal begins). This fraction is configurable via
  `ar1_fit_fraction` in config.yaml (default 0.20, matching Bury).
- 10 surrogates generated per forced series (configurable: `n_null_series`).
- **Why not AAFT?** Bury uses AR(1) in their published code. AAFT preserves the
  full power spectrum of the original series, which would include the CSD ramp —
  defeating the purpose of a null model for CSD detection. AR(1) fitted to the
  neutral period is the correct choice.

### ROC protocol (exact Bury method)

1. Positive examples: rolling window predictions from the **last 40%** of the
   pre-transition segment (`roc_start_frac=0.60`, i.e., the 60–100% window).
2. Negative examples: predictions from the same relative window of AR(1) null series.
3. Labels: positive=1 (forced), negative=0 (AR(1) null).
4. AUC: standard binary AUC on `p_transition` scores.
5. Kendall τ: computed on `p_transition` across the 40 rolling window steps;
   positive τ indicates increasing EWS signal toward the transition.

### Rolling window

- Window size: 50% of series length (`rolling_window_frac: 0.50`, Bury default).
- Steps: 40 positions uniformly spaced (`prediction_steps: 40`).
- Causal: window looks only at past data (no `center=True`).
- Input to classifier: window normalised by mean absolute value, left-padded to
  `ts_len` (500 or 1500) with zeros if the window is shorter.

### Optional 4-channel augmentation (thesis extension)

- Expands each series from (N, 1, L) to (N, 4, L):
  - Ch0: raw residual
  - Ch1: rolling variance (window=25% of L, causal)
  - Ch2: rolling lag-1 AC (pandas `rolling.corr(df.shift(1))`, causal)
  - Ch3: rolling skewness (causal)
- z-normalised per channel using train-set statistics (saved as `*_ch_stats.npz`)
- Off by default (`use_4channel: false`); set to true to activate

---

## Section: Model Selection Rationale

### Why these 8 TSC models?

- **MiniRocket / MultiRocket / ROCKET**: random kernel baseline family; fast, strong
  Zenodo accuracy; MiniRocket is typically fastest with near-MultiRocket accuracy
- **Arsenal**: probabilistic ROCKET (returns calibrated class probabilities rather
  than hard votes) — better for ROC evaluation
- **DrCIF**: diversity via random channel × interval features; good on multivariate
  when `use_4channel=true`
- **HydraMultiRocket**: combines Hydra (dictionary-based) with MultiRocket; state-of-art
  speed/accuracy tradeoff on UCR benchmarks as of 2024
- **RDST**: modern shapelet-based method (random dilation, shapelet transform); retains
  shapelet interpretability at tractable cost
- **WEASEL v2**: word-based bag-of-patterns; captures periodic patterns; efficient

Deleted models (knn_dtw, boss, shapelet, proximity_forest, ts_chief, tde, hivecote)
were pruned for being too slow, superseded, or poorly maintained in aeon v1.4.0.

### Why InceptionTime as the DL baseline?

InceptionTime (Ismail Fawaz et al. 2020) is the canonical deep-learning TSC baseline
on the UCR archive. Adding it alongside CNN-LSTM and LSTM gives one Bury-style DL
model, one recurrent baseline, and one convolutional architecture from the TSC literature.

---

## Section: Results Notes

### Things to flag in discussion

1. **Distribution shift**: classifiers trained on synthetic (Zenodo) time series may
   not generalise to empirical (PANGAEA) data with geological noise and real
   non-stationarities. AUC on PANGAEA should be discussed with this caveat.

2. **Negative controls**: Al and Ti are lithogenic proxies with no expected EWS signal.
   A model that shows high AUC for Al/Ti is overfitting or detecting unrelated trends.
   Use these as a sanity check — expected AUC ≈ 0.5.

3. **Small test set**: Only 7 test transitions across 3 cores. ROC curves computed from
   ~280 windows (40 positions × 7 transitions) vs Bury's ~1000. CIs are wide.

4. **Tau CI**: Kendall τ CI is computed from the distribution over all (core, sapropel,
   element) combinations per model. Report mean ± 95% CI (t-distribution).

5. **4-channel vs 1-channel**: if `use_4channel=true` results are included, compare
   directly (same checkpoint, different input) — they are not independent experiments.

6. **MAX_TRAIN_SAMPLES cap**: TSC training is capped at 100k series (MiniRocket) or
   lower. Full ts_500 has ~473k series after filtering. This is a hardware constraint,
   not a methodological choice; state this explicitly.

---

## Key numbers for the thesis

| Quantity | Value |
|---|---|
| Zenodo train series (ts_500, after filter) | ~473k |
| Zenodo val series (ts_500, after filter) | ~19.9k |
| MAX_TRAIN_SAMPLES (MiniRocket, Rocket, Arsenal) | 100,000 |
| Rolling window positions per series | 40 |
| AR(1) surrogates per forced series | 10 |
| AR(1) fit fraction | 20% (neutral reference) |
| ROC positive window | 60–100% of pre-transition |
| Detrend bandwidth | 20% of record length |
| 4-channel augment window | 25% of series length |
| TSC SLURM: CPUs / RAM / walltime | 16 / 60 GB / 4 h |
| DL SLURM: GPU / CPUs / RAM / walltime | 1 H100 / 4 / 32 GB / 24 h |

---

## Correct DOI for Bury et al. (2021)

The Bury paper DOI is: **10.1073/pnas.2106140118**
(not 10.1073/pnas.2023610118 which appeared in older README drafts)

Full citation:
> Bury, T.M., Sujith, R.I., Pavithran, I., Scheffer, M., Lenton, T.M., Anand, M., &
> Bauch, C.T. (2021). Deep learning for early warning signals of tipping points.
> *Proceedings of the National Academy of Sciences*, 118(39), e2106140118.
> https://doi.org/10.1073/pnas.2106140118
