# Migration Guide — Old Codebase → Refactored Codebase

This document covers what changed between the original codebase and the
Phase 0–5 refactor. If you have old checkpoints or result files, read this
before re-running anything.

---

## ALL old checkpoints are invalid

Two separate bugs corrupted every checkpoint trained before the refactor:

### Bug 1 — Wrong label ordering (Phase 1)

**Old mapping:** `{null: 0, fold: 1, hopf: 2, transcritical: 3}`  
**Correct Bury ordering:** `{fold: 0, hopf: 1, transcritical: 2, null: 3}`

The `null` class was at index 0, not 3. This means `p_transition = 1 − probs[:, 0]`
computed in the old code was `1 − P(fold)`, not `1 − P(null)`. All AUC and
Kendall τ values from old checkpoints are meaningless.

### Bug 2 — Label/data mismatch in training (Phase 3)

The `train_tsc()` function iterated the shuffled DataLoader **twice** in
separate list comprehensions — once for X, once for y — producing different
random permutations each time. Every TSC checkpoint was trained on completely
mismatched label/data pairs. This explains the ~0.24 val accuracy (expected
for random labels on a 4-class problem: 1/4 = 0.25).

**Action required:** delete all `.pkl` and `.ckpt` files in `checkpoints/`
and retrain from scratch.

---

## Deleted models

The following models were removed in Phase 2. Any checkpoints or result
directories referencing them can be archived or deleted.

| Deleted model | Reason |
|---|---|
| `knn_dtw` | Too slow for 470k training series |
| `boss` | Replaced by WEASEL v2 |
| `weasel` (old) | Replaced by WEASEL v2 |
| `shapelet` | Replaced by RDST |
| `proximity_forest` | Slow and rarely competitive |
| `ts_chief` | Slow, unmaintained in aeon |
| `tde` | Replaced by WEASEL v2 |
| `hivecote` | Meta-ensemble; too slow for this dataset |
| `cnn` | Redundant with cnn_lstm |
| `multihead_cnn` | Only ~0.59 val acc; not justified |

---

## New models added

| New model | Phase | Notes |
|---|---|---|
| `hydra_multirocket` | 2 | Best speed/accuracy TSC; supports multivariate |
| `rdst` | 2 | Random dilated shapelet transform; replaces shapelet |
| `weasel2` | 2 | WEASEL v2; replaces boss/weasel/tde; univariate only |
| `inceptiontime` | 2 | PyTorch InceptionTime DL baseline |

---

## Result directory naming

Old directories used split suffix conventions. New directories use a single
unified result per (model, dataset or core/sap/element):

| Old format | New format |
|---|---|
| `{model}_{dataset}_auc/result.json` | `{model}_{dataset}_zenodo/result.json` |
| `{model}_{dataset}_accuracy/result.json` | merged into `_zenodo/result.json` |
| `{model}_pangaea_{core}_{sap}_{elem}_auc/result.json` | `{model}_pangaea_{core}_{sap}_{elem}_pangaea/result.json` |
| `{model}_pangaea_{core}_{sap}_{elem}_kendall_tau/result.json` | merged into `_pangaea/result.json` |

`load_pangaea_results()` and `load_zenodo_results()` in `testing/plot_figures.py`
handle both old and new formats transparently (old dirs are not deleted).

---

## Metric field name changes

| Old field | New field | Notes |
|---|---|---|
| `"auc"` | `"binary_auc"` | Renamed for clarity |
| `"auc"` (old 2D probs path) | N/A | Old code returned `P(null)` instead of `1-P(null)` — wrong |
| separate `_auc` and `_kendall_tau` dirs | single `result.json` | All metrics in one file |

`plot_roc` and other inline plots accept both `"auc"` and `"binary_auc"` for
backward compatibility, but new result.json files always use `"binary_auc"`.

---

## AR(1) null (was AAFT null)

Old result files computed the null distribution using AAFT surrogates.
The new code uses **AR(1) surrogates fitted to the first 20% of the series**,
matching Bury (2021) exactly. Old AUC/ROC values using AAFT are not comparable
to the new values.

---

## 4-channel augmentation (`use_4channel`)

Old code: `use_4channel` was always ON (EWS features were always added).
New code: `use_4channel: false` by default (1-channel = raw residual).

Old 4-channel checkpoints used `center=True` rolling windows (data leakage)
and a random jitter hack. These are invalid — retrain with the new augmenter.

If you set `use_4channel: true`, the train-set channel stats are saved as
`{model}_{dataset}_best_ch_stats.npz` alongside the checkpoint. This file
**must be present** for evaluation to work in 4-channel mode.

---

## `run_all_evaluations.sh` updated

The old script listed 9 models including deleted ones (cnn, multihead_cnn,
shapelet, etc.). The updated script uses the correct 11-model roster.
