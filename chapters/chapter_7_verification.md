# Chapter 7 ("Project Structure & Implementation") — Verification Log, Full Rewrite

## 🔴 The whole chapter previously described code that doesn't exist

This was the deepest problem found across all five chapters. The previous
version described an object-oriented architecture — an abstract
`BaseDataLoader` class with `ZenodoSDELoader`/`PangaeaEmpiricalLoader`
subclasses, a `FeatureEngineer` class, a centralized `Trainer` class
orchestrating both training paths, a standalone `inference.py` script — and
none of it exists in the actual repository. I checked every specific claim
directly against the real source files rather than assume the original
description was a reasonable simplification:

| Chapter claimed | Actual code (checked directly) |
|---|---|
| `BaseDataLoader` abstract class with `ZenodoSDELoader`/`PangaeaEmpiricalLoader` subclasses | `src/dataset_loader.py`: one `EWSDataset` class (a direct `torch.utils.data.Dataset` subclass, not a custom hierarchy) plus free functions. `src/pangea_cleaner.py`: free functions (`process_core()` etc.), no loader class at all. |
| `FeatureEngineer` class | `src/ews_augmenter.py`: one function, `augment_ews_channels()`. No class. |
| Centralized `Trainer` class | `training/train.py`: two top-level functions, `train_dl_variant()` and `train_tsc()`, dispatched by a command-line argument in `main()`. No `Trainer` class anywhere in the repo. |
| Standalone `inference.py` script | Does not exist. Empirical inference is `testing/evaluate.py --target pangaea`, the same file and mechanism as synthetic evaluation, not a separate script. |
| `ModelDefinition` module / class | `models/tsc.py` has a config dictionary (`TSC_SPECS`) and a function that builds the requested classifier. No `ModelDefinition` class. |
| Data-bridge function "within the `Trainer` class" for GPU→CPU/NumPy conversion | No such abstraction exists; `.cpu().numpy()` conversion happens inline where actually needed in `evaluate.py`/`train.py`, not through a dedicated bridge class. |

**Fix applied:** the whole chapter is rewritten to describe the real,
function-based architecture, file by file, with the actual file names and
function names quoted directly rather than invented class names. I did not
try to "improve" the fictional design into something plausible — I checked
what's actually there and described that.

## Other factual corrections, each checked directly

| Claim | Checked against | Fix |
|---|---|---|
| K-fold cross-validation orchestrated by the Trainer | Same as Chapter 6: `config.yaml` shows a single fixed 95/4/1 split, no fold rotation anywhere | Removed; this chapter now just points to Chapter 6 for the split, rather than repeating (and previously mis-stating) it here too |
| Zero-shot linear interpolation inside `PangaeaEmpiricalLoader` | Same as Chapters 5/6: no interpolation exists anywhere in the pipeline | Removed |
| Checkpoint files saved as `.pt` | `ls checkpoints/*.ckpt` — actual extension used by `torch.save` calls in this repo is `.ckpt`, not `.pt` | Fixed |
| "sktime" as the second TSC library, alongside `aeon` | Checked all imports in `models/tsc.py` and `models/bakeoff_dictionary.py`: no `sktime` import anywhere. The actual second library is `pyts` (backs `BOP` and `SAXVSM` specifically), plus `wildboar` (backs `gRSF`, added later this session, not part of the previous chapter's description at all since gRSF didn't exist in the roster when it was last written) | Fixed to name the three libraries actually imported, with which models each backs |
| "requesting 32 or 64 distinct CPU cores" for TSC jobs | `training/train_tsc_array.sh`: `--cpus-per-task=16`. `training/train_dl_array.sh`: `--cpus-per-task=4` | Fixed to the real numbers for both job types |
| Empirical inference evaluates "at every single available temporal step" | `config.yaml`: `inference.prediction_steps: 40` — a fixed, evenly-spaced count, not literally every step. `src/rolling_window.py` confirms: `np.linspace(start_pos, end_pos, n_steps)` | Fixed to state the real, fixed number of windows |
| "we implemented a Ridge Regression classifier" as this thesis's own contribution for the ROCKET backend | `aeon`'s own `MiniRocketClassifier`/`RocketClassifier`/etc. use `RidgeClassifierCV` internally, as part of the library, not as something built for this thesis | Reworded to say this thesis instantiates `aeon`'s classifier directly and does not implement the backend itself — avoids claiming credit for a design choice made inside a third-party library |

## Figure added

This chapter had zero figures and was the clearest candidate in the whole
document for the pipeline-flow diagram you asked for, since it is entirely
architecture description. Added one TikZ diagram (Figure 7.1) showing the
real flow: Zenodo/PANGAEA data → the actual preprocessing files →
`ews_augmenter.py` → `training/train.py` → `.ckpt`/`.pkl` checkpoints →
`testing/evaluate.py` → `result.json`. Every node in the diagram is named
after an actual file or function in the repository, not a generic label.

**A caution on this diagram, since I cannot compile LaTeX on this machine**:
I deliberately used only plain `tikzpicture` syntax with absolute
coordinates (`\node ... at (x,y)`) and plain `->` arrows, avoiding the
`positioning` and `calc` TikZ library features I had used in an earlier
draft of this same diagram, specifically because those are harder for me to
hand-verify without a compiler and more likely to silently fail. I checked
node/edge consistency by script (every node referenced in an arrow is
defined, brace-balanced, environment-matched) but this is not the same
guarantee as an actual `pdflatex` run. **You will need `\usepackage{tikz}`
in your document preamble** if it isn't already there — same caveat as
`graphicx` for the other chapters' figures. Please compile this one and
check the layout before treating it as final; a diagram is exactly the kind
of content where a manual syntax check can miss something a compiler would
catch immediately.

## Carried forward, not independently re-checked this pass

- The SLURM array-job mechanism description (bash indexing into a model
  list via `$SLURM_ARRAY_TASK_ID`) — this part of the previous chapter was
  broadly accurate in spirit even though it attributed the logic to a
  `Trainer` class; the underlying mechanism (arrays, GPU/CPU partition
  split) is real and is kept, just with the class-based framing removed and
  the actual bash indexing shown directly.
- PyTorch `Dataset`/`DataLoader` mechanics (`__len__`/`__getitem__`,
  multi-worker loading) — this description was already accurate to how
  `EWSDataset` actually works; not rewritten, only shortened.
- `joblib`/`torch.save` as the two serialization mechanisms — correct in
  the original, kept, only the file extension (`.pt` → `.ckpt`) was wrong
  and is now fixed.

## Second pass: real comparison against Bury's own repo structure

Added one paragraph checking this thesis's function-based (not
class-hierarchy) design choice against \citet{bury2021}'s own accompanying
repository, rather than presenting it as an isolated design decision.
Confirmed directly (this repo's structure was fetched and read in full
earlier this session): Bury's own code is organized the same way — standalone
scripts (`training_data/`, `dl_train/DL_training.py`, `test_empirical/anoxia/`)
communicating through files on disk, no shared class hierarchy. This also
means the earlier, fictional OOP description this chapter had before this
rewrite (already flagged and corrected) wasn't just inaccurate to this
thesis's own code — it didn't match the pattern used by the paper this
thesis is built on either.

**Shorter than the 8-9 page target** (1,532 words currently, likely 3-4
pages) — implementation chapters naturally have less room for the
comparison-heavy treatment applied to the other chapters, since there
isn't much competing "prior implementation work" to compare against beyond
Bury's repo (now done). If more length is wanted here specifically, tell me
and I'll go deeper on the SLURM/HPC mechanics or add a walkthrough of one
full example job, rather than guess at what to expand.

## Ninth pass (2026-09-14): §7.1 re-verified this session, two real bugs found and fixed

Fresh re-check of the whole chapter against current code, since this file
last had a full pass on 2026-09-05, before this session's `use_4channel`/
5-channel scoping fixes were established in Chapters 3-6.

| Claim | Source | Result |
|---|---|---|
| `train_dl_variant()` never calls `augment_ews_channels()` | `training/train.py:96-125`, read in full | Confirmed |
| `evaluate_zenodo()`/`evaluate_pangaea()` gate the 5-channel expansion behind `is_tsc_model()` | `testing/evaluate.py`, 6 occurrences across both functions | Confirmed |
| Old §7.3 "Input Shape": DL input becomes `(batch_size, 5, L)`, "every deep-learning model... built to accept" | Contradicted by the above | **Fixed** — DL input is `(batch_size, 1, L)`; only the 22 TSC models see 5 channels |
| Old Figure 7.1: `rolling` (PANGAEA) -> `feat` -> `train` | Contradicts the chapter's own caption ("training/train.py only ever runs on synthetic data") and `grep -i pangaea training/train.py` (zero matches) | **Fixed** — figure redrawn: `train` only receives from the zenodo/`prep` side; PANGAEA only ever reaches `evaluate.py`; DL path (dashed) bypasses `feat` entirely, TSC path (solid) goes through it |
| Bury repo structure (`training_data/`, `dl_train/DL_training.py`, `DL_test.py`, `test_empirical/anoxia/`) | Checked directly against `/home/s466553/abc/deep-early-warnings-pnas/` | Confirmed, unchanged, no fix needed |
| aeon backs the majority of the roster; pyts backs BOP/SAX-VSM; wildboar backs gRSF | `models/tsc.py` (13 `from aeon...` imports), `models/bakeoff_dictionary.py` (`_pyts_compat`), `models/tsc.py:172` (`from wildboar.ensemble import ShapeletForestClassifier`) | Confirmed, unchanged |
| §7.6: 40 evenly-spaced prediction steps by default | `config.yaml:36`, `prediction_steps: 40` | Confirmed, unchanged |

**Fix applied:** Figure 7.1 redrawn (dashed DL path bypassing `ews_augmenter.py`,
solid TSC path through it, `train` fed only from the zenodo side) and the
"Input Shape" paragraph rewritten to state the correct `(batch_size, 1, L)`
shape for DL models, cross-referencing Chapter 6 Section 6.2 for the fork
and Section 7.4 for the TSC-side 5-channel detail.

Braces re-checked: 162 open / 162 close, balanced. All `\ref`/`\label`
pairs resolve, including the two cross-chapter refs into `chapter_6.tex`
(`ch:methodology`, `sec:dl_architectures`), both confirmed defined there.

### Plagiarism check
No source wording used; restates this codebase's own structure in this
thesis's own words. Clean.

## Tenth pass (fresh independent audit, this session): one real error found and fixed

Full re-check of every factual claim against current code, independent of prior passes' conclusions.

| Claim | Source | Result |
|---|---|---|
| `training/train.py` never references PANGAEA | `grep -in pangaea training/train.py` | Confirmed — zero matches |
| `is_tsc_model()` gates the channel fork in `evaluate.py` | `grep -n is_tsc_model testing/evaluate.py` | Confirmed — 6 occurrences (import + 4 call sites + 1 log line) |
| `EWSDataset(Dataset)`, `__len__`, `__getitem__`, `get_dataloader()` all in `src/dataset_loader.py` | Read directly | Confirmed |
| `load_config()` is a free function of `src/dataset_loader.py` | Read `src/dataset_loader.py` and `src/constants.py` | **Wrong** — `load_config()` is defined in `src/constants.py` (`def load_config(config_path...)` at line 33) and merely imported into `dataset_loader.py` (`from src.constants import load_config`). Fixed: attributed correctly to `src/constants.py`, kept `get_dataloader()` as the dataset_loader.py free function. |
| `process_core()` in `src/pangea_cleaner.py`, no class | `grep -n "^def process_core\|^class "` | Confirmed — function only, no class |
| `augment_ews_channels()` in `src/ews_augmenter.py` | Read directly | Confirmed |
| `normalize_mean_abs()`, `left_pad_to()`, `random_left_censor()` in `src/data_common.py` | Read directly | Confirmed |
| SLURM array formula (`MODEL=.../2`, `DATASET=...%2`) | `train_dl_array.sh`, `train_tsc_array.sh` | Confirmed verbatim in both |
| DL partition `h100`+`gpu:1`+4 CPUs; TSC partition `large_cpu`+16 CPUs, no GPU | Both array scripts | Confirmed |
| aeon backs MiniRocket/MultiRocket/Rocket/Arsenal/RDST/WEASEL2/TDE/DrCIF/catch22/ProximityForest/TSF/ShapeletTransform/LearningShapelets/BOSS/CIF/MrSQM | `grep -n "from aeon" models/tsc.py` | Confirmed — all 16 present (Arsenal and CIF specifically re-checked after initial grep missed them due to multi-line import) |
| pyts backs BOP and SAX-VSM | `models/bakeoff_dictionary.py` | Confirmed |
| wildboar backs gRSF via `ShapeletForestClassifier` | `models/tsc.py:172` | Confirmed |
| Bury repo structure (`training_data/`, `dl_train/{DL_training.py,DL_test.py}`, `test_empirical/anoxia/`) | Listed `/home/s466553/abc/deep-early-warnings-pnas/` directly | Confirmed |
| Dataset size 500,000 at L=500 | `dataset/ts_500/combined/cache_labels.npy`, shape check | Confirmed |
| `prediction_steps: 40` | `config.yaml:36` | Confirmed |
| Pipeline figure (TikZ): DL dashed 1ch bypass, TSC solid 5ch through `feat`, PANGAEA only reaching `eval`, never `train` | Cross-checked edges against `is_tsc_model` gating and the `pangaea`-grep-empty result on `train.py` | Confirmed accurate, no changes needed |
| All citation keys (`bury2021`, `burygithub`, `yoo2003slurm`, `paszke2019pytorch`, `aeon2024`, `pyts2020`, `pedregosa2011sklearn`) | `references.bib` | All present |

**Fix applied:** one sentence in §7.1 corrected to attribute `load_config()` to `src/constants.py` rather than `src/dataset_loader.py`. Braces re-checked: 163/163 balanced. All `\ref`/`\label` pairs resolve (4 local labels: `ch:implementation`, `fig:pipeline_flow`, `sec:hpc`, `sec:tsc_libraries`; 2 legitimate external refs into Chapter 6: `ch:methodology`, `sec:dl_architectures`).

### Plagiarism check
No source wording used in the fix; restates the codebase's own structure in this thesis's own words. Clean.

## Eleventh pass (2026-09-20, this session): genuinely independent re-derivation, two real errors found

User asked to check "once again," independent of this file's prior conclusions, specifically flagging that `src/rolling_window.py` is currently uncommitted and was refactored this session (`run_all_sapropels`, `to_dataframe`, and the `__main__` block removed; `pad_mode` added to `prepare_dl_input`/`compute_rolling_ews`). Re-derived every claim from the current working tree rather than trusting this log's prior entries.

| Claim | Source checked | Result |
|---|---|---|
| Figure 7.1 `pangaea` node: "PANGAEA cores / Mo, Al, Ba, Ti, U" | `src/rolling_window.py` line 18: `ELEMENTS = ["Mo", "U"]` (currently uncommitted; committed HEAD still has the 5-element list, but the working tree — what the pipeline actually runs — has the 2-element restriction) | **Wrong.** The figure still showed the pre-restriction 5-element list. Fixed to "Mo, U". |
| "The 22 classical TSC models are not reimplemented from scratch... Three libraries back the roster" (§7.4 opening) | `models/tsc.py` `TSC_SPECS` (22 keys, counted programmatically) cross-referenced against `_build_classifier()`'s import statements, plus the three new untracked files `models/bakeoff_tsbf.py`, `models/bakeoff_lps.py`, `models/bakeoff_fastshapelet.py` (`git status` shows these as `??`, i.e. brand new, not yet committed) | **Wrong.** 19 of 22 come from a library (16 aeon, 2 pyts, 1 wildboar — counts and names re-confirmed against the actual `from aeon...`/`from pyts...`/`from wildboar...` import lines). The other 3 — TSBF, LPS, Fast Shapelets — have no maintained library implementation and are built from scratch for this thesis, using only `sklearn.RandomForestClassifier` as a building block; each file's own docstring says so explicitly ("built from scratch... No maintained library ships this") and each is validated only against UCR GunPoint (`models/test_bakeoff_scratch.py`), not against a reference implementation. This was missed by every prior pass in this log (the tenth pass's own accounting only reaches 16+2+1=19 of 22, silently). Fixed: rewrote the §7.4 opening and added a paragraph naming the three from-scratch models, their files, and their validation method; softened the corresponding sentence in §7.1's "Model definitions" bullet, which had implied every TSC model comes from a library. |
| `training/train.py` never references PANGAEA | `grep -in pangaea training/train.py` | Confirmed — zero matches, unchanged |
| `is_tsc_model()` gates the 5-channel fork | `models/__init__.py:50-51` (also currently uncommitted, `M`): `def is_tsc_model(name): return name in TSC_SPECS` | Confirmed — matches the figure and Input Shape section |
| DL input shape `(batch_size, 1, L)`; TSC input shape `(N, 5, L)` | `src/dataset_loader.py` `EWSDataset.__getitem__` (`x = torch.from_numpy(out).unsqueeze(0)`, i.e. `(1, length)` per item) and `src/ews_augmenter.py` `_rolling_channels_chunk`/`_compute_rolling_channels` (both explicitly build `(B, 5, L)` / `(N, 5, L)`, stacking raw + variance + lag-1 AC + skewness + variance-growth-ratio) | Confirmed — note `ews_augmenter.py`'s own module docstring and `apply_channel_norm()` docstring say "(N, 4, L)", which is stale/wrong *inside the source file itself* (the actual array-construction code is 5-channel); this doesn't affect the chapter, which was already checking the real computation, not the stale docstring |
| `load_config()` in `src/constants.py`, imported by `src/dataset_loader.py` and `testing/evaluate.py` | `src/constants.py:33`, `src/dataset_loader.py:11` (`from src.constants import load_config`), `testing/evaluate.py:29` (`from src.dataset_loader import load_config, ...`) | Confirmed — `evaluate.py` gets it via `dataset_loader.py`'s re-export, consistent with the chapter's description |
| `evaluate.py --target pangaea` calls `evaluate_pangaea()`, which imports `compute_rolling_ews`/`ELEMENTS` directly from `src/rolling_window.py` and no longer goes through the removed `run_all_sapropels` | `testing/evaluate.py:30` (`from src.rolling_window import compute_rolling_ews, ELEMENTS`), `:434`/`:477` (direct calls) | Confirmed — the chapter's PANGAEA section never named `run_all_sapropels` or any other removed function, so nothing there was stale |
| `model.eval()` called before DL inference | `testing/evaluate.py:138` (`m.to(device).eval()`) | Confirmed |
| `p_transition = 1 - P(null)` written to `result.json` alongside `variance`/`lag1_ac` | `testing/evaluate.py:287` (`p_transition = 1.0 - probs[:, null_idx]`), `:535-560` (`result_data` dict includes `p_transition`, `p_fold`/`p_hopf`/`p_transcritical`/`p_null`, `variance`, `lag1_ac`) | Confirmed |
| `torch.save(model.state_dict(), ...)` as `.ckpt`; `joblib.dump`/`load` as `.pkl` | `training/train.py:160,436`, `testing/evaluate.py:137,146-153`; `config.yaml` `naming.checkpoint_dl`/`checkpoint_tsc` | Confirmed |
| `checkpoints/` actually contains `.ckpt`, `.pkl`, `.npz` (channel stats) and nothing else | `ls checkpoints/ \| sed -E 's/.*(\.[a-z]+)$/\1/' \| sort -u` | Confirmed — exactly `.ckpt .npz .pkl` |
| SLURM array formula, GPU/CPU partition split (h100+gpu:1+4 CPUs for DL, large_cpu+16 CPUs for TSC) | `training/train_dl_array.sh`, `training/train_tsc_array.sh`, read in full | Confirmed verbatim, unchanged |
| `prediction_steps: 40`, `pad_mode: "zero"`, `rolling_window_frac: 0.25` | `config.yaml:36-38` (also currently uncommitted, `M`, but these three values are unchanged from what the chapter states) | Confirmed |
| 475,000 training examples at L=500 ("hundreds of thousands... per model") | `dataset/processed/train_500.npz`, `X.shape` | Confirmed — `(475000, 1, 500)` |
| No `BaseDataLoader`/`Trainer`/`FeatureEngineer` class hierarchy anywhere in `src/`, `models/`, `training/`, `testing/` | Re-scanned all DL model files (`cnn_lstm.py`, `lstm.py`, `inceptiontime.py`, `resnet.py`, `tcn.py`, `rnn_fcn.py`, `patchtst.py`) — each is a direct `torch.nn.Module` subclass, no shared custom base class | Confirmed |
| Citation keys `bury2021`, `burygithub`, `yoo2003slurm`, `paszke2019pytorch`, `aeon2024`, `pyts2020`, `pedregosa2011sklearn` | `references.bib` | All present, re-confirmed fresh |
| `\ref{}` targets (`ch:methodology`, `fig:pipeline_flow`, `sec:dl_architectures`, `sec:hpc`, `sec:tsc_libraries`) | Grepped `\label{}` across all `chapters/chapter_*.tex` | All 5 resolve |

**Fixes applied:**
1. Figure 7.1's `pangaea` node element list corrected from "Mo, Al, Ba, Ti, U" to "Mo, U", matching the current (uncommitted) `ELEMENTS` restriction actually used by the pipeline.
2. §7.4 rewritten: 19 (not 22) of the TSC models come from a library; added a new paragraph naming TSBF, LPS, and Fast Shapelets as this thesis's own from-scratch implementations, their files, what they simplify from their original papers, and their GunPoint-only validation.
3. §7.1's "Model definitions" bullet softened to not imply every TSC model is built from a library.

**Caveat caught while writing fix #2 and corrected before finalizing:** the three new source files' own docstrings say "Validated only against UCR GunPoint (see `models/test_bakeoff_scratch.py`)". Checked directly: `models/test_bakeoff_scratch.py` does not exist anywhere in the repository — not on disk, not untracked, not anywhere in `git log --all` history for that path. The GunPoint validation itself may well have happened (the three docstrings independently claim it, and the claim is plausible), but the specific file the docstrings point to as evidence cannot be found or inspected this session. Per WRITING_RULES.md's rule against asserting what cannot be checked, the chapter does not cite that file path as a source; it now attributes the GunPoint-validation claim to "each file's own documentation" rather than presenting the (unfindable) test file as something this session verified.

Post-edit integrity check: braces 171/171 (was 163/163 before this pass — net +8/+8 from the new paragraph, balanced). All `\ref{}` in the chapter resolve, unchanged set.

### Plagiarism check
No source wording used; the new paragraph on the from-scratch models paraphrases their own docstrings in this thesis's own words, cites no external phrasing. Clean.
