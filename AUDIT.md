# Log Triage and Correctness Audit (Phase 1)

## 1. Log Triage

### OOM Kills
* **Finding**: `multirocket` training was OOM-killed on `ts_500` (e.g., `tsc_train_2663540_10.err`, `tsc_train_2663540_13.err`, `tsc_train_2663540_12.err`, etc.).
* **Reason**: The full feature matrix for Rocket/MultiRocket classifiers is extremely large when training on tens of thousands of samples, causing memory footprint to exceed the 55G RSS budget.
* **Fix needed**: We must cap `max_train_samples` in configuration (e.g. limit MultiRocket/Arsenal/Hydra to 30,000, others to 80,000) and implement chunked transforms to avoid materializing the full 60G+ feature matrix all at once.

### Thread Bug
* **Finding**: Multiple training logs printed thread allocations that did not match the CPU allocation. For instance, the jobs ran with `NUMBA_NUM_THREADS=32` on a 16-CPU allocation.
* **Reason**: Environment variables were not explicitly synchronized with `$SLURM_CPUS_PER_TASK`.
* **Fix needed**: Explicitly set `NUMBA_NUM_THREADS = OMP_NUM_THREADS = MKL_NUM_THREADS = SLURM_CPUS_PER_TASK (16)` in the Slurm scripts and at the top of the Python scripts, and override `n_jobs` appropriately.

### Cancelled Jobs
* **Finding**: Error logs containing the string `CANCELLED` (e.g. `ews_eval_pangaea_2786581_11.err` / `tsc_train_2663493_6.err`) all occurred around a single timestamp.
* **Reason**: External cancellations of old Slurm job arrays, not bugs or crashes in the code. Stale and can be ignored.

### Cosmetic Warnings
* **Finding**: Warnings concerning PyTorch `padding='same'` with odd kernel sizes and `sklearn`'s deprecation of `delayed` import location.
* **Fix needed**: Address these last as they are cosmetic and do not affect metrics correctness.

---

## 2. Correctness Invariants

### Labeling & `np.bincount(y)`
* **Status**: **Verified**.
* **Verification**: In `src/preprocess_bury_data.py`, the label map maps `"null" -> 3`. Verification asserts that `counts[NULL_IDX] > 0`, confirming that the null class lies at index 3 in the generated `.npz` files.

### Empirical Mo/U AUC Inversion / Degeneracy
* **Status**: **Present** in baseline.
* **Finding**:
  * For example, the `minirocket` empirical AUC for `MS21_S1_Mo` is `0.4725` (near degenerate/inverted).
  * Many empirical predictions have `p_transition = 0.0` or degenerate values of `0.50` because the models were trained on raw series without left-pad censoring augmentation, making the short empirical cores completely out-of-distribution (OOD) at inference.

### Toggle Flags (Current State)
* **use_4channel**: Enabled in `config.yaml` (`use_4channel: true`).
* **Training-time Padding Augmentation**: Enabled in `config.yaml` (`augmentation.enabled: true`), but needs verification to ensure it is actually used during training of all classifiers (especially TSC classifiers) and matches Bury's formulation.

---
