# Code Review and Bury Reference Comparison (Phase 2)

We have mapped the codebase (`src/`, `training/`, `testing/`, `config.yaml`) against Bury's reference repo. Below are the confirmed gaps and implementation choices:

## 1. Missing / Mismatched Training-Time Padding Augmentation
* **Gap**: While `augmentation.enabled` exists in `config.yaml` and is implemented in `dataset_loader.py` for PyTorch DL models, the TSC models (Rocket/MultiRocket/Arsenal, etc.) fit in `train_tsc` only use a static `tsc_copies = 2` amount of random left censoring. The data loader and training script must ensure that **every** model is trained with adequate random left-pad censoring matching Bury's proportion (up to 90% or 95% of the series length zero-padded).

## 2. Train-vs-Inference Normalization Mismatch
* **Gap**: Standard TSC model pipelines (e.g. Rocket/MiniRocket) compute rolling EWS channels and then apply training-set z-score stats. However, the zero-padded regions introduce a large number of exact zeros, making the rolling features (especially variance) heavily skewed. Z-score stats computed on the training set (which has a lower average padding fraction) do not align with empirical inputs (which have higher padding fractions).
* **Fix**: To achieve train-inference parity, we should use a single shared normalisation function `normalize_mean_abs` on raw residuals before computing EWS channels or feeding them to the classifiers.

## 3. Growing History vs. Fixed-Length Window at Inference
* **Gap**: Currently, `make_fixed_window` uses `residuals[:end_pos]`, which represents a growing history from the start of the core. Bury's reference implementation uses a sliding window (unveiled in steps of 10 data points). We should verify if a fixed-length sliding window or growing history provides better generalization and align them across training and testing.

## 4. Per-Model Memory and Threading Gaps
* **Gap**: Per-model memory is not capped, leading to OOM errors when fitting MultiRocket or Arsenal on the full dataset.
* **Fix**: Implement `max_train_samples` limits (e.g., 30,000 for multirocket/arsenal/hydra, 80,000 for others) and transform features in chunks (e.g., batches of 5,000 to 10,000) followed by garbage collection.
* **Gap**: Threading environment variables (`NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`) are set to 32 instead of 16 in some array scripts.
* **Fix**: Ensure they are synchronized to `$SLURM_CPUS_PER_TASK` (16) at the top of `train.py`.

## 5. Label and Orientation Parity
* **Status**: **Matched**. The label mapping in `src/preprocess_bury_data.py` maps `"null" -> 3`, which is correct. The empirical residuals are oriented oldest-first (transition at the end), which matches Bury's setup.

---
