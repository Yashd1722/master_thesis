# Chapter 6 ("Methodology") — Verification Log, Full Rewrite

## 🔴 Two factual mismatches with the real pipeline, both corrected

### K-fold cross-validation was never used

§6.1.3 previously claimed "Stratified K-Fold Cross-Validation" with $K=5$
rotating folds. Checked directly against `config.yaml`:
```
train_frac: 0.95
val_frac:   0.04
test_frac:  0.01
```
This is one fixed split into three parts, applied once — there is no fold
rotation anywhere in `training/train.py` or `src/dataset_loader.py`. Fixed
to state the real split plainly.

### Preprocessing description repeated Chapter 5's original errors

§6.1.1 said the data "immediately undergoes strict linear interpolation" and
"Z-score normalization." Both are wrong for the same reasons established in
`chapter_5_verification.md` this session (no interpolation exists anywhere
in the pipeline; normalization is by mean absolute value, verified against
both `src/data_common.py` and Bury et al.'s own training code). Rather than
re-derive this a second time, the chapter now states the conclusion and
points to Chapter 5's fuller explanation, since re-deriving the same
verification twice in two chapters would just be more repetition of the
kind this rewrite is trying to remove.

## 🔴 Model roster: 12 → 29

The previous chapter described roughly 12 models in depth (LSTM, CNN-LSTM,
InceptionTime, MiniRocket, MultiRocket, Arsenal, WEASEL2, SAX-VSM, TSF,
TSBF, LPS) and did not mention the other 17 at all. Checked the actual
roster against `config.yaml`'s `models.dl` and `models.tsc` lists directly:
7 deep-learning models, 22 classical TSC models, 29 total.

**Fix applied:** kept full-depth explanations for the three DL models that
represent genuinely distinct design families (pure recurrence, hybrid,
multi-scale convolution), added a new subsection for the four newer DL
architectures (ResNet, TCN, RNN-FCN, PatchTST) citing the specific reference
paper each one reproduces, and added coverage of every remaining classical
model, grouped by family, plus a summary table (Table 6.1) listing all 29
by name so nothing is silently missing from the chapter even where the
prose doesn't go into individual depth.

The ResNet/TCN/RNN-FCN descriptions are not new claims — they restate,
in shorter form, what `ARCHITECTURE_COMPARISON.md` (in the project repo)
already verified layer-by-layer against each paper's own reference
implementation earlier this session: ResNet against `hfawaz/dl-4-tsc`, TCN
against `locuslab/TCN`, RNN-FCN against both the original Keras
implementation and `tsai`'s PyTorch port (specifically the "dimension
shuffle" detail, which is easy to get wrong and was flagged in that document
as the defining, non-obvious feature of the architecture). PatchTST is used
via the `tsai` library directly, not reimplemented, and the chapter says so
rather than implying it was rebuilt from scratch like the other three.

## Citations added

Five architecture papers were previously uncited even though their models
were used (ResNet, TCN, RNN-FCN, PatchTST, catch22) — checked and added:

| Model | Citation added |
|---|---|
| ResNet | `\citet{wang2017resnet}`, IJCNN 2017 |
| TCN | `\citet{bai2018tcn}`, arXiv:1803.01271 |
| RNN-FCN / LSTM-FCN | `\citet{karim2018lstmfcn}`, IEEE Access 2018 |
| PatchTST | `\citet{nie2023patchtst}`, ICLR 2023 |
| catch22 | `\citet{lubba2019catch22}`, DMKD 2019 |

All five are the actual original paper for each named architecture —
cross-checked against the paper titles/venues already recorded correctly in
`ARCHITECTURE_COMPARISON.md` for the first three, and against my own
knowledge of the other two (PatchTST, catch22), not guessed or invented.

## Other corrections

- **Hardcoded "Chapter 9" references** (AUC being "the primary metric used in
  the Results presented in Chapter 9") — there is no Chapter 9 in this
  five-chapter set, and I don't know what your full document's actual
  chapter numbering is once assembled. Removed the specific chapter number
  rather than guess one; the AUC/metric definitions here stand on their own
  without needing to forward-reference a specific results chapter.
- **Batch size claim** — previously stated "e.g., using a batch size of 64
  or 128." Checked against every `training.<model>` block in `config.yaml`:
  actual values are 128 (most models) or 256 (CNN-LSTM, InceptionTime
  specifically) — 64 does not appear anywhere in the current config. Fixed
  to state the real values.
- **Class index order** — the chapter's prose lists classes as Fold,
  Transcritical, Hopf, Null for readability, but `src/constants.py` (the
  actual canonical ordering used by every script that computes a class
  index) is Fold, Hopf, Transcritical, Null. Added an explicit note about
  this discrepancy so a reader going from the chapter to the code isn't
  confused by the mismatched order — this is worth keeping as a written-out
  note rather than "fixing" the prose order, since Fold/Transcritical/Hopf
  is the more natural order to explain in text (grouping the two hardest-
  to-distinguish classes together, as Chapter 3 does), while the code's
  order is fixed by an early implementation choice that would be
  disruptive to change now.

## Carried forward, not independently re-checked this pass

- MiniRocket's 84-kernel description (already corrected in an earlier pass,
  per the `%% FLAGGED` comment in the previous version) — restated, not
  re-derived from scratch.
- LPS naming (Learned Pattern Similarity, already corrected in an earlier
  pass) — restated.
- ROC/AUC textbook definitions — standard, unchanged content, only shortened.
- Adam optimizer citation and CCE loss formula — unchanged, standard.

## Second pass: comparison depth + double-checked the new factual claim

Added two comparison-focused pieces per your request for "why is this
different/better" content, each checked before writing:

1. **"Why Benchmark Both" subsection** — states the concrete, checkable
   reason for testing both DL and classical TSC (not just "for breadth"):
   the general TSC literature (Bagnall bake-off successors) finds classical
   kernel/interval methods often competitive with DL, but whether that
   holds specifically for bifurcation-type classification was never tested
   by \citet{bury2021}. Framed as an open question this thesis answers
   (Chapter 8), not an assumption.
2. **MiniRocket speed claim, verified twice.** First stated it loosely
   ("substantially faster"); you asked me to re-check and verify twice, so
   I went back to the paper's own abstract via search rather than trust the
   first pass. Confirmed exact figure: "up to 75 times faster... while
   maintaining essentially the same accuracy" — updated the chapter text to
   use this precise, directly-quoted number instead of the vaguer wording.
3. **AUC vs.\ F1 methodological comparison** — explains why this thesis
   uses AUC where \citet{bury2021} used F1 (threshold-independence needed
   to fairly compare 29 heterogeneous models, not all of which produce
   comparably-calibrated probability outputs at a shared cutoff). This is
   methodological reasoning, not an external factual claim, so it doesn't
   need a citation beyond the standard classifier-evaluation argument for
   AUC's rank-invariance already implicit in the ROC/AUC definitions.

Re-checked the whole document's structure after these edits (braces,
labels, citations, cross-references) — clean, no regressions.

## Third pass (2026-09-13): §6.1 — the 5-channel input is TSC-only, not universal

Same finding as the cross-chapter fix already made in Chapters 3, 4, and
5 this session (logged in each of their own verification files):
Stage 2 ("Feature engineering") and Stage 3 ("Model inference") of the
pipeline, and the $\mathcal{X}$ definition in "Supervised Learning
Framing," all implied the 5-channel expansion applies to every model.
It doesn't.

| Claim | Source |
|---|---|
| `train_tsc()` calls `augment_ews_channels()`; `train_dl_variant()` never does | `training/train.py`, re-grepped this turn: the channel-expansion call only appears inside `train_tsc()` |
| Same split at evaluation time | `testing/evaluate.py`: the function is literally named `_prepare_tsc_input()`, TSC-only |
| Confirms `ARCHITECTURE_COMPARISON.md`'s own statement | *"Input to every DL model in this repo is the single-channel... residual... The 5-channel EWS feature stack is applied to TSC models only, not to DL models."* |

**Fixed:** Stage 2 now says "(classical models only)" and states the DL
models skip the step entirely; Stage 3 states 5-channel vs. 1-channel
explicitly per model type; $\mathcal{X}$'s definition now says "5 channels
wide for the classical models, 1 channel wide for the deep-learning
models" instead of asserting one fixed channel count.

Also fixed in the same pass: Stage 1 listed the PANGAEA proxies as
"(Mo, Al, Ba, Ti, U)" — stale. Current pipeline restricts PANGAEA
evaluation to Mo and U only, confirmed fresh this turn against
`src/rolling_window.py`: `ELEMENTS = ["Mo", "U"]`.

### Plagiarism check

No source wording involved. Clean.

## Fourth pass (2026-09-13): pipeline flowchart added to §6.1

Per user request, added a boxes-and-arrows pipeline diagram (matching the
convention of a typical ML-paper figure they pointed to), placed right
before the "Pipeline, Stage by Stage" enumerate list in §6.1. Generated
with matplotlib (`/tmp/claude-216236/figscript/make_fig6_pipeline.py`),
matching this thesis's existing convention of pre-rendered PDF figures
(no inline TikZ, since the other five chapters' figures are all
matplotlib PDFs and there's no visible root `.tex` file to confirm `tikz`
is even loaded in the preamble).

Every box and arrow traces to a claim already verified this session —
no new claims introduced by the figure itself:

| Box | Verified against |
|---|---|
| Synthetic SDE corpus, Zenodo, 500k/200k | Chapter 5 §`sec:sde_corpus` (Bury README, already verified) |
| PANGAEA cores, Mo/U, 3 named cores | `src/rolling_window.py` `ELEMENTS = ["Mo","U"]`; Chapter 5 §`sec:pangaea_data` |
| Preprocessing: windowed by position, normalized by mean(\|x\|) | Chapter 5 §`sec:preprocessing` (already verified against `src/data_common.py` and Bury's own training code) |
| TSC = 5-channel, DL = 1-channel split | this chapter's own third-pass fix (`training/train.py`, `testing/evaluate.py`, `ARCHITECTURE_COMPARISON.md`) |
| 29 models, L=500/L=1500 | `config.yaml` model counts, already verified in Chapter 4/5 |
| 4-class output order | `src/constants.py` `CLASS_NAMES`, already verified this chapter, first pass |
| OVR macro AUC (synthetic) / binary AUC vs. AR(1) null (PANGAEA) | `metric/auc.py` `ovr_macro_auc()` / `compute_auc()`, re-read this session |

Saved to `figures/fig6_pipeline.pdf` (and `.png` for review), referenced
as `\label{fig:pipeline}`.

### Plagiarism check

Diagram, not prose — no source text involved. Clean.

## Fifth pass (2026-09-13): two new figure generators in `testing/plot_figures.py`

Per user request, added two new plotting functions matching the style of
Ma et al. (2025, *Communications Physics*) Figs. 2/3 and 4 (viewed
directly — extracted the embedded raster images from the cached arXiv PDF
with PyMuPDF, since Nature's site is paywalled and no PDF-page-render tool
was available):

- `plot_fig_stack_ma()` — 4-row stacked panel (raw proxy, rolling
  variance, rolling lag-1 AC, model p(transition)) vs. age, one per
  (model, dataset, core, sapropel, element). Pure plotting: every row
  either comes straight from `result.json` (variance, lag1_ac,
  p_transition — already written by `evaluate.py`) or the raw forced.csv
  (the proxy series) — no new computation.
- `plot_fig_roc3()` — 3-curve ROC comparison (this model vs. classical
  variance vs. classical lag-1 AC, all against the same AR(1) null
  surrogates). The model's ROC is already in `result.json`
  (`roc_fpr`/`roc_tpr`/`binary_auc`). The two classical curves are the one
  new computation: the null surrogates' own variance/lag-1 AC aren't
  saved anywhere (only the model's prediction on them is), so they're
  recomputed from the already-generated `*_ar1_null.csv` file, reusing
  `src/rolling_window.py`'s own `_variance`/`_lag1_ac` functions directly
  (not reimplemented) so the numbers are guaranteed identical to what the
  rest of the pipeline computes.

Both wired into `main()` alongside the existing `plot_fig2_bury()`, same
Mo/U gating, same per-record iteration. Ran on the full current (Mo/U-only,
649-record) result set: 649/649 written, 0 errors.

**Sanity-checked the ROC3 output isn't trivially degenerate**: picked a
segment with a known near-chance model AUC (`bop`, MS21/S1/Mo, binary_auc
≈ 0.51 in `results/summary/pangaea.csv`) and confirmed the generated
figure shows the model curve sitting on the diagonal (A=0.51) while the
classical Variance and Lag-1 AC curves diverge from it independently
(A=1.00 and A=0.64 respectively) — the three curves are not just
copies of each other, confirming the classical recomputation is doing
real, differentiated work, not silently mirroring the model's own score.

### Plagiarism check

Figure style (stacked panel layout, ROC-comparison-with-AUC-in-legend
layout) is a standard visualization convention, not copyrightable
expression; no source code or text copied. Clean.

## Sixth pass (2026-09-14): LSTM fix finally applied, §6.1 pipeline narrative expanded

The §6.2 LSTM fix (proposed several turns earlier in the session, pending
approval) is applied now. User pasted an excerpt of \citet{ma2025sdml}'s
own published Methods section, which **independently re-confirms** the
architecture and the F1=0.99/64PE406E1 number already verified earlier
this session against the arXiv preprint — now confirmed against two
separate readings of the source, satisfying "verify twice" a second time
over.

**One discrepancy caught between the arXiv preprint and the published
text**, noted for the record (not asserted in the thesis, since it isn't
about anything the thesis claims): the preprint said train/val split
0.8/0.2 with F1 reported on validation; the user's pasted (published)
text says train/val/test 0.6/0.2/0.2 with F1 on a held-out test set. Not
otherwise relevant to what Chapter 6 claims about the architecture itself.

**Explicitly did not copy the user's pasted text into the thesis** — it is
Ma et al.'s own prose (their Methods section, verbatim). Copying it would
be exactly the plagiarism this whole session has been checking for. Used
it only as a second, independent confirmation of facts already verified,
same as every other citation in this project.

### §6.1 pipeline narrative — expanded to properly describe Figure 6.1

Per user request: rewrote the paragraph introducing Figure~\ref{fig:pipeline}
from a one-sentence summary into a full walkthrough of the figure's own
structure (two sources → shared preprocessing → model-family split →
inference → evaluation split by data source), and added a fifth item to
the stage-by-stage enumerate list covering the evaluation split (the
figure's last two boxes), which the list previously stopped short of —
it ended at "probabilistic output" without ever describing what happens
to that output. No new factual claims — every sentence restates something
already verified elsewhere in this chapter or Chapter 5, now connected
explicitly to the figure.

### Plagiarism check

No source wording used; LSTM fix already independently verified. Clean.

## Seventh pass (2026-09-14): CNN-LSTM and InceptionTime given real architecture detail; every model given its full name

Per user request: (1) write the remaining DL model-family subsections with
the same rigor as the LSTM fix, (2) use real/full names instead of bare
abbreviations throughout, everywhere in the chapter.

### CNN-LSTM — real layer sizes, verified against the actual code

| Claim | Source |
|---|---|
| Architecture is Bury et al.'s own, not original | already established (this chapter, LSTM subsection contrasts the two) |
| Conv1d(1→50, k=12, same-pad) → ReLU → Dropout(0.1) → MaxPool(2) → LSTM(50→50) → Dropout(0.1) → LSTM(50→10) → Dropout(0.1) → Linear(10→4) | `models/cnn_lstm.py`, read directly this turn |
| Docstring: "reproduction of Bury et al. (PNAS 2021)"; "Dropout is 10% (Bury: higher hurt F1)" | same file, quoted almost verbatim (code comment, not prose to paraphrase around) |

### InceptionTime — real module structure, verified against the actual code

| Claim | Source |
|---|---|
| 3 stacked Inception modules + global average pooling + linear | `models/inceptiontime.py` docstring and class structure, read directly this turn |
| Each module: 1×1 bottleneck (32 ch) → three parallel convs (kernel 9/19/39) + maxpool-passthrough branch → concatenate → batch-norm | same file, `_InceptionModule.__init__`/`forward` |
| Named for resemblance to Inception-style (GoogLeNet) image classifiers | standard, well-known naming rationale for the "Inception" family of architectures (InceptionTime paper's own title references it: *"Finding AlexNet for Time Series Classification"*) |

### Full names added, each verified this turn (web search where not already established)

| Abbreviation | Full name | Source |
|---|---|---|
| ROCKET | RandOm Convolutional KErnel Transform | already established, Chapter 4 |
| WEASEL | Word ExtrAction for time SEries cLassification | verified via `timeseriesclassification.com` algorithm page, matches Schäfer & Leser's own naming |
| SAX-VSM | Symbolic Aggregate approXimation -- Vector Space Model | standard, well-established terms (SAX: Lin et al.; VSM: classic IR term) |
| BOSS | Bag of SFA Symbols | verified — Schäfer 2015, *"The BOSS is concerned with time series classification in the presence of noise"* |
| catch22 | CAnonical Time-series CHaracteristics | verified — Lubba et al. 2019 paper's own title; also confirmed the "22 from 4791" and "~1000x speedup / ~7% accuracy cost" figures from the same source |
| MrSQM | Multiple Representations Sequence Miner | verified — Nguyen & Ifrim, arXiv:2109.01036 abstract |
| DrCIF | Diverse Representation Canonical Interval Forest | verified — adds periodogram + first-order-difference representations to CIF's interval search, confirmed via the HIVE-COTE 2.0 paper and the `tsml-java` source |
| PatchTST | Patch Time Series Transformer | verified — Nie et al. 2023, *"A Time Series is Worth 64 Words"* (ICLR 2023), matches the already-cited `nie2023patchtst` |
| ResNet / TCN / RNN-FCN | Residual Network / Temporal Convolutional Network / Recurrent Neural Network -- Fully Convolutional Network | standard expansions of already-cited architectures, no new claim |

### Checked and deliberately NOT added: Fourier/AAFT/IAAFT surrogate methods

User asked whether this thesis uses Fourier Transform, AAFT, or IAAFT
surrogates (from a pasted excerpt of \citet{ma2025sdml}'s own Methods
section) and, if so, to write about them. Checked directly:

| Claim | Source |
|---|---|
| AAFT/IAAFT1/IAAFT2 surrogate generation *is* implemented | `src/pangea_cleaner.py`: `_aaft_surrogate()`, `_iaaft_surrogate()`, `SURROGATE_METHODS = {"AAFT":..., "IAAFT1":..., "IAAFT2":...}`, citing *"Bury 2021 / Ma 2025: use AAFT surrogates as null"* |
| ...but is **not** the null method used for any reported result | `config.yaml`: `null_method: AR1` (not AAFT); and directly on disk: every null file under `dataset/.../clean_dataset/` is named `*_ar1_null.csv` — zero `*_aaft_null.csv` files exist anywhere |

Per the user's own instruction ("if we have used then only otherwise you
have to ignore it") and the same precedent already set for edge/reflect
padding (Chapter 4 §`sec:padding_gap`: implemented but unused, future
work, not claimed as a contribution) — **not added to the chapter**. AR(1)
is the only null-surrogate method this thesis's results actually rest on,
and that is already documented correctly (Chapter 3 §`sec:ar1_surrogates`).

### Plagiarism check

All facts attributed; no source prose copied. The one near-verbatim
element (the `cnn_lstm.py` docstring line) is a code comment being quoted
as a technical citation, not text being passed off as original writing.
Clean.

## Eighth pass (2026-09-14): §6.1.1 items 4–5 — predict_proba() and a real PANGAEA frequency number

Per user request: name the exact function that produces each model's
probability output, confirm it's used identically for TSC models on
PANGAEA data (not a different mechanism there), and add a real number
describing the PANGAEA output's class-frequency shape.

| Claim | Source |
|---|---|
| Every model exposes `predict_proba()` | grepped all `models/*.py` this session — present in every file, same name throughout |
| DL: `torch.softmax(self.forward(x), dim=-1)` | `models/lstm.py`, `models/cnn_lstm.py`, read directly |
| TSC: delegates to `self._clf.predict_proba(X)`, or for ridge-regression kernel classifiers (ROCKET family) computes `decision_function()` then softmaxes the result by hand | `models/tsc.py` lines 337–362, read in full this session — exact code: `exp_df = np.exp(df - np.max(df, axis=1, keepdims=True)); probs = exp_df / np.sum(exp_df, axis=1, keepdims=True)` |
| Same `predict_proba()` call feeds both the synthetic-AUC and PANGAEA-detection scoring — no separate mechanism for real data | direct consequence of the above: nothing in `testing/evaluate.py`'s PANGAEA path calls anything other than the model's own `predict_proba()` |
| Pooled PANGAEA favoured-class frequency: 28.5% fold / 37.4% hopf / 14.6% transcritical / 19.5% null, 25,960 forced-window predictions across 649 records | computed fresh this session directly from `test_result/*_pangaea/result.json`, Mo/U only, all models unfiltered |

**Explicitly scoped as a methodology-section description, not a Chapter 8
finding**: the pooled number includes every model regardless of quality
(including the ones excluded from type-consensus claims per memory:
weasel2, ls, multirocket, minirocket, mrsqm), and the chapter text says so
directly, pointing to Chapter 8 for the filtered, weighted version.

### Plagiarism check

No source wording used; all facts computed or read directly this session.
Clean.

## Eighth pass (2026-09-14): §6.4/§6.5 re-verified against code, one fix applied

Re-checked every claim in §6.4 (Evaluation Metrics) and §6.5 (Training
Details) directly against code, since these two subsections had not had a
fresh verification pass this session.

| Claim | Source | Result |
|---|---|---|
| Binary AUC = `roc_auc_score(labels_binary, p_transition)`, `p_transition = 1 - P(null)` | `metric/auc.py`, `compute_auc()` | Confirmed exact |
| One-vs-rest macro AUC = `roc_auc_score(y_true, probs, multi_class="ovr", average="macro")` | `metric/auc.py`, `ovr_macro_auc()` | Confirmed exact |
| Bury et al. F1: 88.2% at L=1500, 84.2% at L=500 | `bury2021.txt` lines 283-286, re-read this pass | Confirmed exact, correct pairing |
| Loss = categorical cross-entropy | `training/train.py:128`, `criterion = nn.CrossEntropyLoss()` | Confirmed |
| Adam optimizer | `training/train.py:129` | Confirmed |
| "All deep-learning models use a ReduceLROnPlateau scheduler" | `config.yaml` training block, all 7 DL models | **False as written** — `lstm: scheduler: none`; `train.py:131-135` shows `scheduler` stays `None` unless `config == "reduce_on_plateau"`. 6/7 models use it, LSTM does not. |
| Scheduler factor 0.5 throughout, patience 10-20 (for the 6 that have one) | `config.yaml`: cnn_lstm 20, inceptiontime 10, patchtst 10, resnet 15, tcn 12, rnn_fcn 15, all factor 0.5 | Confirmed |
| Batch size 128 default, 256 only for CNN-LSTM/InceptionTime | `config.yaml` | Confirmed, all 7 checked individually |
| Early-stop patience 20-50 | `config.yaml`: cnn_lstm 50, inceptiontime 30, lstm 20, patchtst 20, resnet 40, tcn 30, rnn_fcn 40 | Confirmed |
| Model freeze: no gradient/training on PANGAEA data | `testing/evaluate.py`: `m.eval()` (line 138), `@torch.no_grad()` (line 180) | Confirmed |

**Fix applied:** §6.5.2 rewritten to state the LSTM is trained at a fixed
learning rate (`scheduler: none`), rather than claiming all seven models use
`ReduceLROnPlateau`. One sentence changed, nothing else touched.

Braces re-checked after the edit: 184 open / 184 close, balanced. All
`\ref`/`\label` pairs intact.

### Plagiarism check
No source wording used for the fix; restates config/code facts in this
thesis's own words. Clean.

## Chapter 6 status: all five sections (6.1-6.5) now verified this session and approved by the user. Chapter considered final pending any later cross-chapter consistency pass.

## Ninth pass (fresh audit, independent of prior log): full re-verification

Re-checked every factual claim in chapter_6.tex against current code, independent of trusting this log's earlier entries.

**Verified exactly correct (no changes needed):**
- All 7 DL model architectures (models/lstm.py, cnn_lstm.py, inceptiontime.py, resnet.py, tcn.py, rnn_fcn.py, patchtst.py) — every layer size, kernel size, dropout rate, and the RNN-FCN "dimension shuffle" (`nn.LSTM(ts_len, lstm_hidden)` confirmed: LSTM's input-size dim is ts_len, i.e. whole series as one step) checked directly against current code.
- TCN receptive field: recomputed from first principles (1 + 2*(k-1)*sum(dilations), k=7, 8 levels, dilations 1..128) = 3061, matches exactly.
- models/tsc.py predict_proba(): decision_function + numerically-stable manual softmax, exact code match.
- src/constants.py CLASS_NAMES = ["fold","hopf","transcritical","null"], matches.
- config.yaml: train/val/test fracs (0.95/0.04/0.01), scheduler (6 reduce_on_plateau + lstm's "none"), batch sizes (256 for cnn_lstm/inceptiontime, 128 for rest) — all confirmed.
- Model roster: 7 DL + 22 TSC = 29, confirmed via config.yaml directly.
- All 20 citation keys used in this chapter confirmed present in references.bib.
- Bury et al. F1 84.2%(L=500)/88.2%(L=1500): re-confirmed against bury2021.txt lines 284-285.
- MiniRocket 84-kernel combinatorics: C(9,3)=84 recomputed directly, correct.
- **The pooled favoured-class statistic in §6.1 (649 records, 25,960 forced-window predictions, 28.5%/37.4%/14.6%/19.5%) was suspected stale (computed early in the session, before several models finished retraining later) — recomputed fresh from current test_result/*_pangaea/result.json files this pass. Result: IDENTICAL to what's in the text. Not stale — confirmed current.**

**Fixed:**
- InceptionTime paragraph omitted the residual shortcut connection that models/inceptiontime.py actually has (a 1×1-conv+BN projection of the raw input, added back after the 3 stacked Inception modules, before the final ReLU/GAP). Added a sentence describing it, attributed to the code's own comment citing Fawaz 2020 Fig. 2 (not independently re-derived from the paper itself, stated honestly as such).

Braces: 186 open / 186 close, balanced. All \ref/\label pairs resolve (remaining unresolved names are legitimate cross-chapter references into chapters 4/5/8, not local errors).

### Plagiarism check
Fix uses this thesis's own wording, sourced from the code's own comment (quoted appropriately). Clean.

## Tenth pass (2026-09-20): fresh, independent full re-verification (not a confirmation of the ninth pass)

Re-derived every claim in the chapter from current code/config from scratch,
per the same standard as the chapter 8 audit this session (recompute
numbers independently, check logs/timestamps for any before/after
comparison, don't trust the prior log's conclusions).

**Verified exactly correct, independently re-derived (no changes needed):**
- InceptionTime residual shortcut: `models/inceptiontime.py`'s
  `_ResidualBlock` docstring reads verbatim "Shortcut connection every 3
  Inception modules (Fawaz 2020 Fig. 2)" — matches the chapter's claim and
  attribution exactly.
- Scheduler per DL model, read fresh from every `training.<model>` block in
  `config.yaml`: `cnn_lstm`, `inceptiontime`, `patchtst`, `resnet`, `tcn`,
  `rnn_fcn` all `scheduler: reduce_on_plateau`; `lstm` alone is
  `scheduler: none`. Scheduler patience 10–20, factor 0.5 throughout, batch
  size 256 (cnn_lstm, inceptiontime) / 128 (rest), early-stop patience
  20–50 — all individually re-checked against the current file, all correct.
- All 7 DL architectures re-read directly from current `models/*.py` and
  matched line-for-line against the chapter's descriptions: `lstm.py`
  (Linear(1→128)→LSTM(128)→Dropout(.5)→LSTM(64)→Dropout(.5)→Linear(64→128)
  +ReLU→Linear(128→4)), `cnn_lstm.py` (Conv1d(1→50,k=12)→ReLU→Dropout(.1)
  →MaxPool(2)→LSTM(50→50)→Dropout(.1)→LSTM(50→10)→Dropout(.1)→Linear(10→4)),
  `resnet.py` (3 blocks, widths 64/128/128, kernels 8/5/3, shortcut is a
  plain BN when channels match and 1×1-conv+BN otherwise), `tcn.py`
  (receptive field recomputed from scratch: 8 levels, kernel 7, dilations
  1..128, `1+2*(k-1)*sum(dilations) = 1+2*6*255 = 3061`, exact match),
  `rnn_fcn.py` (`nn.LSTM(ts_len, lstm_hidden, batch_first=True)` fed `x`
  directly with shape `(B,1,ts_len)` — confirms the "dimension shuffle": one
  time step of length `ts_len`, not `ts_len` steps of length 1),
  `patchtst.py` (`from tsai.models.PatchTST import PatchTST` — used
  directly, not reimplemented, confirmed).
- Bury et al. F1 (88.2% L=1500 / 84.2% L=500): re-read directly from
  `/tmp/claude-216236/bury2021.txt` lines 284–286 this pass, exact match.
- `metric/auc.py`'s `compute_auc()` (binary, `p_transition` vs. AR1 null)
  and `ovr_macro_auc()` (`roc_auc_score(..., multi_class="ovr",
  average="macro")`) re-read directly, match the chapter's description
  exactly.
- `training/train.py:128-135`: `nn.CrossEntropyLoss()`, `optim.Adam(...)`,
  `ReduceLROnPlateau` gated on `tr_cfg.get("scheduler") == "reduce_on_plateau"`
  — matches.
- `testing/evaluate.py`: `m.eval()` (line 138) and `@torch.no_grad()`
  (line 180) confirm the model-freeze claim.
- `models/tsc.py:337-362`: `predict_proba()` — delegates to
  `self._clf.predict_proba(X)` for most models, or for the ridge/decision-
  function classifiers computes `decision_function()` then a manual,
  numerically-stable softmax (`exp_df = np.exp(df - np.max(...)); probs =
  exp_df / np.sum(...)`) — exact code match, unchanged since the last pass.
- Pooled PANGAEA favoured-class frequency: **recomputed completely from
  scratch this pass** (not reusing the prior pass's number) by loading all
  `test_result/*_pangaea/result.json` files, stacking each record's
  per-window `p_fold`/`p_hopf`/`p_transcritical`/`p_null` arrays, and
  taking the argmax per window. Result: 649 records, 25,960 windows,
  fold 28.5% / hopf 37.4% / transcritical 14.6% / null 19.5% — identical to
  the chapter's stated numbers.
- All 20 `\citet`/`\citep`/`\citealp` keys used in this chapter confirmed
  present in `references.bib` (re-grepped fresh, not reused from the ninth
  pass's list).
- 5-channel (classical) vs. 1-channel (DL) split: re-grepped
  `augment_ews_channels` and `is_tsc_model` across `training/train.py` and
  `testing/evaluate.py` — the channel expansion is still gated by
  `is_tsc_model()` in both files. No regression.

**Errors found and fixed (new this pass, not caught in the ninth pass):**

1. **Train/val/test split misattributed to `config.yaml`.** The chapter
   said the dataset "is split once... 95%/4%/1%... (`config.yaml`:
   `train_frac: 0.95, val_frac: 0.04, test_frac: 0.01`)," implying those
   config fields define the split. They don't: grepping every `.py` file in
   the repo for `train_frac`/`val_frac`/`test_frac` returns zero hits —
   nothing reads them. The actual split is Bury et al.'s own, inherited
   from their Zenodo release: `groups.csv` assigns each sequence a fixed
   `dataset_ID` (1/2/3), and `src/preprocess_bury_data.py` reads that
   column directly (`map_split()`) to sort series into train/val/test —
   it never computes a split itself. Counted the actual saved `.npz` files
   to confirm the resulting proportions are still 95/4/1 at both lengths
   (475,000/20,000/5,000 of 500,000 at $L=500$; 190,000/8,000/2,000 of
   200,000 at $L=1500$), so the *numbers* in the chapter were right, but
   the *mechanism* attributed to them was wrong. Fixed to correctly
   attribute the split to `groups.csv`/Bury's release, note that
   `config.yaml`'s fields are descriptive only and unused by any split
   code, and keep the confirmed 95/4/1 proportions.

2. **Model roster table and prose were missing Shapelet Transform (ST)
   entirely** — 21 classical models named, not the 22 claimed everywhere
   else in the chapter (section intro, table caption). `config.yaml`'s
   `models.tsc` list has 22 entries including `st`; `models/tsc.py`
   instantiates it as aeon's `ShapeletTransformClassifier`; chapter 7 lists
   "Shapelet Transform" among the aeon-backed roster; chapter 8 discusses
   `st` extensively as a real, complete model (synthetic AUC, PANGAEA AUC,
   roster-completeness accounting) — so this was a real omission, not a
   deliberate exclusion. Fixed: added a sentence describing ST (searches
   for discriminative subsequences, transforms series into per-shapelet
   distances, trains a standard classifier on the result) to the "Interval
   and Feature-Based Methods" subsection, citing `\citet{bagnall2017bakeoff}`
   (which specifically names Shapelet Transform as the second-best
   individual classifier in its comparison, already cited elsewhere in this
   chapter), and added "ST" to Table 6.1.

3. **"BOSS" in the dictionary-based subsection doesn't name which BOSS
   variant is actually trained.** `models/tsc.py`'s own code comment states
   plainly: "BOSSEnsemble itself has no time contract and does not scale to
   full-dataset training. ContractableBOSS (cBOSS) is aeon's own scalable
   replacement for exactly this case" — and the code instantiates
   `ContractableBOSS`, not `BOSSEnsemble`. The chapter previously just said
   "BOSS (Bag of SFA Symbols)" with no indication a substitution was made.
   Fixed: added one sentence naming the actual estimator (cBOSS) and why.
   Left the short-form "BOSS" naming used elsewhere in the chapter and in
   chapters 4/7/8 unchanged for consistency, since cBOSS is still the same
   dictionary-based family the chapter already groups it with — only the
   scalability substitution was undocumented, not the family assignment.

**Checked and confirmed NOT an error:** `checkpoints/` contains
`cote_ts_500_best_ch_stats.npz` / `cote_ts_1500_best_ch_stats.npz`
(channel-stat files only, no trained `.pkl`), and `cote` is not in
`config.yaml`'s `models.tsc` list and not referenced in `models/tsc.py`.
This is an abandoned HIVE-COTE2 attempt, not part of the 22-model roster —
the chapter only cites HIVE-COTE 2.0 as Arsenal's source suite
(`\citealp{middlehurst2021hivecote2}`), never claims HIVE-COTE2 itself was
trained, so no fix needed.

Post-edit integrity check: brace balance 203/203 (was 186/186 before this
pass's 3 fixes — the increase is from the added ST/cBOSS/split sentences
and the new table cell, not an imbalance). All `\ref{}` in the chapter
resolve against the full cross-chapter label set (`chapters/chapter_*.tex`
`\label{}` grep), confirmed fresh this pass.

### Plagiarism check
All three fixes are this thesis's own wording, sourced from code comments
(quoted/attributed, not copied as prose) and `config.yaml`/checkpoint
facts. Clean.
