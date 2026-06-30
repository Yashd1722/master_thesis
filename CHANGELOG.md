# CHANGELOG

All changes are listed chronologically. One bullet per fix. Branch: `refactor-clean`.

---

## Phase 6 — Documentation

- **[6.1]** Rewrote `README.md`: accurate 11-model roster (8 TSC + 3 DL), current
  SLURM task maps (16 TSC tasks, 6 DL tasks), corrected Bury DOI
  (10.1073/pnas.2106140118), updated metric table (binary_auc/macro_f1/macro_auc_ovr/
  kendall_tau), updated figure guide (FIG1–FIG5), AR(1) null protocol, 4-channel
  augmentation section. Removed all deleted model references.
- **[6.2]** Created `MIGRATION.md`: documents both root causes of invalid old checkpoints
  (wrong label ordering + DataLoader label mismatch), deleted vs added models, result
  directory naming changes (old `_auc`/`_kendall_tau` → new `_zenodo`/`_pangaea`),
  metric field rename (`auc` → `binary_auc`), null method change (AAFT → AR(1)).
- **[6.3]** Created `REPORT_NOTES.md`: Methods section notes (EWS signal formula, AR(1)
  null rationale, ROC protocol, rolling window spec, 4-channel design), model selection
  rationale for all 11 models, results caveats (distribution shift, negative controls,
  small N, tau CI, MAX_TRAIN_SAMPLES), key numbers table, correct Bury DOI with full
  citation.
- **[6.4]** Created `README_HPC.md`: environment setup, partition summary (large_cpu /
  h100), full task-to-model mapping for both SLURM arrays, monitoring commands, 4 common
  failure modes with fixes, useful SLURM snippets, checkpoint backup guidance.
- **[6.5]** Updated `testing/run_all_evaluations.sh`: replaced stale 9-model list
  (included deleted cnn, multihead_cnn, shapelet, etc.) with correct 11-model roster.
  Removed emoji (shell portability). Calls `plot_figures.py --config config.yaml`
  after evaluation.

## Phase 5 — Standardised Figures

- **[5.1]** Rewrote `testing/plot_figures.py` with fixed style constants at module level:
  `MODEL_ORDER` (11 models, TSC then DL), `MODEL_COLORS` (fixed palette per model),
  `CLASS_COLORS` (fold/hopf/transcritical/null), `ELEMENT_ORDER` = ["Al","Ba","Mo","Ti","U"].
- **[5.2]** Inline plots (called immediately from `evaluate.py`):
  - `plot_roc`: ROC with `binary_auc` field (backwards-compatible with old `auc` key).
  - `plot_confusion_matrix`: heatmap with white-on-dark text for readability.
  - `plot_pangaea_series`: quick 3-panel preview (variance + lag-1 AC + p_transition).
- **[5.3]** FIG1 (`plot_fig1_pangaea`): 4-panel stacked — raw proxy + trend (panel a),
  residuals (panel b), variance + lag-1 AC dual-y (panel c), p_transition (panel d).
  Panels a/b loaded from pangaea CSV if available; falls back to 2-panel if not.
  x-axis: Age (kyr BP), oldest at left, transition at right edge. AUC and τ in title.
- **[5.4]** FIG2 (`plot_fig2_roc`): ROC for each (model, core, sapropel, element) with
  AUC in legend. Saved as `{model}_{core}_{sap}_{element}_fig2_roc.png`.
- **[5.5]** FIG3 (`plot_fig3_auc_heatmap`): mean AUC matrix (rows=models, cols=elements),
  averaged over all (core, sapropel) combinations. RdYlGn colormap, vmin=0.5/vmax=1.0.
- **[5.6]** FIG4 (`plot_fig4_kendall_tau`): bar chart of mean Kendall τ ± 95% CI
  (t-distribution) per model. TSC bars in orange, DL bars in blue. Null τ = 0 line shown.
- **[5.7]** FIG5 (`plot_fig5_roc_overlay`): all models overlaid on one ROC axes for one
  (core, element). TPR interpolated to common FPR grid then averaged over sapropels.
  One file per core × element: `fig5_roc_overlay_{core}_{element}.png`.
- **[5.8]** `load_pangaea_results` handles both new (`*_pangaea`) and old (`*_auc`/`*_kendall_tau`)
  directory formats, merging old split dirs into unified records. No duplicates via `seen` set.
- **[5.9]** Summary CLI: `python testing/plot_figures.py [--config config.yaml]` drives
  FIG1+FIG2 per (model,core,sap,element), then FIG3/FIG4 globally, then FIG5 per (core,element).
  All saved to `results/comparison/`.

Smoke (Phase 5): all 8 figure functions pass with mock data; correct filename convention;
  x-axis direction verified (oldest-left, transition-right) ✓

## Critical Bug Fix (discovered during Phase 3 smoke test)

- **[BUG]** Fixed label mismatch in `train_tsc()`: the shuffled training DataLoader was
  iterated twice in separate list comprehensions — once for X, once for y — producing
  different random permutations and completely mismatched label/data pairs. This caused
  near-random val accuracy (~0.24 for a 4-class problem). Fixed by collecting both X and y
  inside a single `for x_batch, y_batch in train_dl` loop. All previous TSC checkpoints
  are invalid; retrain required.

## Phase 4 — Metrics & Parity

- **[4.1]** `metric/auc.py`: removed the fragile `probs[:, -1]` branch (which returned
  P(null), not 1-P(null)). `compute_auc` now accepts only 1D `p_transition` — callers
  must compute `1 - probs[:, NULL_IDX]` explicitly. Added `ovr_macro_auc()` (OVR
  macro-averaged AUC for 4-class evaluation).
- **[4.2]** `metric/kendall_tau.py`: added `compute_tau_ci()` for mean ± CI across
  surrogates. Added module-level unit checks: rising p_transition → tau > 0; constant
  p_transition → tau = 0. Both fire at import time (fail fast).
- **[4.3]** `metric/multiclass.py` (NEW): `macro_f1()` (4-class macro-averaged F1) and
  re-exports `ovr_macro_auc`. `metric/__init__.py` updated to export all new functions.
- **[4.4]** `testing/evaluate.py` — `evaluate_zenodo()`: now reports binary_auc, macro_f1,
  macro_auc_ovr, and accuracy together in a single result.json per model+dataset.
  `evaluate_pangaea()`: adds per-surrogate tau via `compute_tau_ci(null_taus)`, reports
  tau_null_mean/ci alongside forced tau; uses `null_window_counts` to split p_trans_n
  by surrogate cleanly (no duplicate predict_fn calls). 4-channel augmentation wired
  up: loads `*_ch_stats.npz` and calls `augment_ews_channels` with saved stats.
- **[4.5]** `training/train.py`: added `--binary` flag. `train_dl_binary()` remaps labels
  to forced(1)/null(0), trains DL model with 2-class head. For Bury-comparable replication.
  TSC models raise an error if `--binary` is passed (not supported).

Smoke (Phase 4, minirocket/ts_500, 1-channel):
  binary_auc=0.8309  macro-F1=0.6606  macro-AUC(OVR)=0.7766  acc=0.6646
  All fields verified in result.json ✓

## Phase 3 — Optional 4-Channel EWS Augmenter

- **[3.1]** Rewrote `src/ews_augmenter.py`: causal rolling windows (no `center=True`),
  correct lag-1 Pearson AC via `rolling.corr(df.shift(1))`, removed random jitter hack.
  Channels: [raw_residual, rolling_variance, rolling_lag1_ac, rolling_skewness].
  Vectorised over (L, N) pandas DataFrame; chunked in batches of 10k series for memory safety.
- **[3.2]** z-normalisation: each channel normalised with TRAIN-set mean/std computed in
  `fit_channel_stats()`. Stats saved as `{model}_{dataset}_ch_stats.npz` alongside the
  checkpoint. Val/test/empirical must pass saved stats to avoid distribution leakage.
- **[3.3]** `config.yaml`: added `rolling_window_frac_augment: 0.25` (separate from
  `rolling_window_frac: 0.50` used for sliding-window inference).
- **[3.4]** `training/train.py`: augmentation path saves `ch_stats.npz`, applies
  post-augmentation per-channel variance check (drops rare all-zero channels), and reuses
  train stats for val. `use_4channel: false` default preserved — 1-channel runs are valid.
  1-channel val acc ~0.60; 4-channel val acc ~0.64 on ts_500 with MiniRocket.

## Phase 2 — Model Roster Pruned + Modern Models Added

- **[2.1]** Deleted 10 retired model files: `knn_dtw`, `boss`, `weasel`, `shapelet`, `proximity_forest`, `ts_chief`, `tde`, `hivecote` (slow/obsolete TSC), `multihead_cnn` (~0.59 acc), `cnn` (redundant with cnn_lstm). Reason: per spec — prune dead weight.
- **[2.2]** Rewrote existing TSC model wrappers (`rocket`, `minirocket`, `multirocket`, `drcif`, `arsenal`): removed `supported_hyperparameters()` indirection, pass kwargs directly, IS_TSC=True pattern throughout.
- **[2.3]** Added `models/hydra_multirocket.py` — `MultiRocketHydraClassifier` (best speed/accuracy TSC add, multivariate).
- **[2.4]** Added `models/rdst.py` — `RDSTClassifier` (replaces shapelet; casts to float64 for numba kernel).
- **[2.5]** Added `models/weasel2.py` — `WEASEL_V2` (replaces boss/weasel/tde; univariate — takes channel 0).
- **[2.6]** Added `models/inceptiontime.py` — PyTorch InceptionTime DL baseline (3 Inception modules + GAP + residual shortcut).
- **[2.7]** Rewrote `models/__init__.py`: removed monkey-patching of arsenal/knn_dtw/weasel; clean auto-discovery via IS_TSC pattern.
- **[2.8]** Updated `config.yaml` model lists and training hyperparameters: removed deleted model configs, added new models. `n_jobs` is NOT hardcoded — injected at runtime from `SLURM_CPUS_PER_TASK`.
- **[2.9]** Updated `training/train_tsc_array.sh`: new 8-model roster (16 array tasks), NUMBA/OMP/MKL all set to `$SLURM_CPUS_PER_TASK`, walltime lowered to 4h, portable `$HOME` paths.
- **[2.10]** Updated `training/train_dl_array.sh`: new 3-model DL roster (6 array tasks), portable paths.

## Phase 1 — Correctness Fixes

- **[1.1]** Fixed label bug in `src/preprocess_bury_data.py`: old map was `{null:0,fold:1,...}`; corrected to `{"fold":0,"hopf":1,"transcritical":2,"null":3}` matching Bury's canonical ordering. Removed `.fillna(0)` — now raises `ValueError` on unknown labels. Added `_verify_null_index()` to assert null is at index 3 after saving. => All old checkpoints are invalid; NPZ regeneration started.
- **[1.2]** Fixed AR(1) null fitting in `testing/evaluate.py`: replaced full-series fit with Bury's first-20% method (`_fit_ar1_neutral`). The CSD ramp near the transition is now excluded from the null model. Added `AR1_FIT_FRACTION = 0.20` constant. Added `_generate_ar1_surrogates` docstrings explaining the science.
- **[1.3]** `evaluate.py`: replaced `class_names.index("null")` with `NULL_IDX` imported from `src.constants`. Both `evaluate_zenodo` and `evaluate_pangaea` use canonical constants.
- **[1.4]** Config: added `detrend_bandwidth_frac: 0.20` to pangaea section. Expanded `inference` section: `null_method: "AR1"`, `ar1_fit_fraction: 0.20`, `use_4channel: false` (default OFF).
- **[1.5]** Fixed HPC thread env in `training/train.py`: now reads `SLURM_CPUS_PER_TASK` for NUMBA, OMP, AND MKL. Added `N_JOBS` module constant read from `SLURM_CPUS_PER_TASK` (fallback=4); injected into every TSC model call.
- **[1.6]** `training/train.py`: removed triple-duplicate memory-safety blocks; single explicit `MAX_TRAIN_SAMPLES` gate with stratified subsampling.
- **[1.7]** `training/train.py`: EWS augmentation is now config-gated (`use_4channel`, default OFF). When OFF, data stays as (N,1,L) univariate. When ON, expands to (N,4,L).
- **[1.8]** `training/train.py`: added tqdm progress bars for both DL epoch loop and batch loop. Confusion matrix printed after every training run.

## Phase 0 — Repo Hygiene

- **[0.1]** Created branch `refactor-clean` from `main`.
- **[0.2]** Deleted stray files: `testing/evaluate.py#` (Emacs backup), `training/.gitignorgit` (typo duplicate).
- **[0.3]** Updated `.gitignore`: added `old_results_backup/`, `logs/`, `dataset/processed/`, `data/**/*.npz`, `*.keras`, `__MACOSX/`; consolidated checkpoint rules; added rule for nested pangaea CSVs.
- **[0.4]** Removed `old_results_backup/` from git tracking (`git rm --cached -r`) — files kept on disk, now gitignored. Reason: 247 binary/result files have no place in version control.
- **[0.5]** Created `src/constants.py` with `CLASS_NAMES = ["fold","hopf","transcritical","null"]` and `NULL_IDX = 3`. This is the canonical Bury class order; all files that hard-code `null_idx` should import from here.
