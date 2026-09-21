# Chapter 5 ("Data") — Verification Log, Full Rewrite

This chapter was fully rewritten this session (previous version's own verification
log is superseded by this one — the claims changed, so the verification had to
be redone against those new claims, not carried forward). Every technical claim
below was checked against either the actual pipeline source code in this repo,
Bury et al.'s own public repository (`ThomasMBury/deep-early-warnings-pnas`),
or a result computed directly this session and shown reproducible. Nothing is
carried over from memory or an earlier session's notes without being re-checked
here.

## 🔴 Correction made during writing — most significant item in this pass

### The "generating SDE" is not the canonical normal form

The chapter originally stated (matching how Chapter 3 already describes it)
that each training series is produced by directly integrating the canonical
normal form, e.g. `dx = (λ − x²)dt + σdW` for the fold class. I checked this
against Bury et al.'s actual data-generation code before finalizing the
chapter, rather than assume the pre-existing framing was correct:

- `training_data/gen_model.py` (fetched from their repo): generates a random
  2D cubic polynomial system, `dx/dt = Σ aₖφₖ(x,y)`, `dy/dt = Σ bₖφₖ(x,y)`,
  with 20 randomly-drawn coefficients (about half set to zero for sparsity).
- `training_data/sim_model.py`: confirmed the actual Euler-Maruyama loop
  (`s[i+1] = s[i] + de_fun(s[i],pars)*dt + dW[i]`), the 100-time-unit burn-in
  (`tburn = 100`), and that the bifurcation parameter is moved **linearly**
  in time from its start value to the value AUTO located as the bifurcation
  point (`b = pd.Series(np.linspace(bl,bh,len(t)))`).
- `training_data/stoch_sims.py`: confirmed the noise amplitude formula,
  `sigma = sqrt(2*rrate) * sigma_tilde * rv_tri` (`sigma_tilde = 0.01` fixed,
  `rrate` = local recovery rate, `rv_tri` = random triangular variate) — not
  a flat random draw from a range, as I had first written.

So: burn-in, Euler-Maruyama, and linear parameter forcing were all correct as
originally stated and are unchanged. But the *equation being integrated* is
not the bare normal form — it is a random polynomial system, verified via
AUTO continuation to actually pass through a bifurcation of the target type.
The normal form is what that system is provably equivalent to *near* the
bifurcation point (Normal Form Theorem), not what is literally simulated.
This is exactly the "library of random dynamical systems" approach that
Babazadeh Maghsoodlo et al. (2025) explicitly describe Bury et al. as using,
when contrasting it with their own from-normal-forms approach — an
independent, external cross-check that this reading is correct, not just
something I inferred from the code alone.

**Fix applied:** rewrote §5.2.1 ("Four Classes, and What Actually Generates
Them") to describe the random-polynomial-plus-AUTO-continuation process
accurately, cite `burygithub` at the specific claim, and explain the Normal
Form Theorem's role in why the canonical forms in Chapter 3 are still the
right thing to derive even though they are not literally what was simulated.

**Open item for you:** Chapter 3 (already treated as "verified" in an earlier
pass) presents the same canonical-normal-form equations without this caveat.
That earlier verification pass only checked citations for Chapter 3, not this
specific claim against Bury's generation code — so the gap was not caught
before. I have not touched Chapter 3 yet, per the agreed pacing (Chapter 5
first, checkpoint, then continue). Flagging now so you can decide whether
Chapter 3 needs the same caveat added when we get to it.

## Claims checked against the real codebase (not literature) — all confirmed

| Claim in the chapter | Verified against | Result |
|---|---|---|
| Training/testing are fully separated; weights frozen before empirical testing | `training/train.py` (writes checkpoints) vs `testing/evaluate.py --target pangaea` (loads checkpoint, no gradient updates) | Confirmed |
| No interpolation to a uniform time grid anywhere in the pipeline | `grep -n "interp\|resample"` across `src/data_common.py`, `src/pangea_cleaner.py` — zero hits | Confirmed |
| Windowing is by position/index count, not elapsed time | `src/rolling_window.py`: `win = int(win_frac * n)` where `n = len(residuals)`, a point count | Confirmed |
| Normalization is by mean absolute value, not Z-score | `src/data_common.py: normalize_mean_abs()` — `x / mean(abs(x))`, no mean-subtraction, no std-division | Confirmed (chapter previously claimed Z-score; this was wrong and is now fixed) |
| Left-censoring augmentation exists and was tightened this session | `src/data_common.py: random_left_censor()`; `config.yaml` diff this session: `min_visible: 30→15`, `pad_max_frac: 0.9→0.97` | Confirmed |
| 5-channel feature suite: raw, variance, lag-1 AC, skewness, variance-growth-ratio | `src/ews_augmenter.py` docstring: "(N,L) residuals -> (N,4,L)... 0 raw residual \| 1 rolling variance \| 2 rolling lag-1 AC \| 3 rolling skewness" plus the 5th channel (variance ratio) added in the same function | Confirmed — matches exactly, including channel order |
| Channels are z-normalized with train-set statistics, reused (never refit) at test time | `src/ews_augmenter.py` docstring: "Each channel is z-normalised with TRAIN-set stats that must be saved and reused for val/test/empirical (never refit on held-out data)" | Confirmed, quoted near-verbatim |
| This thesis does not itself run the SDE simulation, only repackages Bury's released files | `src/preprocess_bury_data.py` docstring + code: reads `labels.csv`, `groups.csv`, `output_resids/*.csv` from a `combined/` directory — these are Bury's own released output files, not generated locally | Confirmed |
| Interpolation-before-evaluation was tested and reduced AUC | This session's own experiment: `cnn_lstm` AUC 0.919→0.805 (ts_500) and 0.962→0.857 (ts_1500) under interpolation; `catch22` AUC 0.774→0.783 (ts_500), a small gain | Confirmed — corrected the chapter text, which had over-generalized this to "the deep-learning models" plural when only one DL model was actually tested |

## Claims checked against literature/external sources

| Claim | Source checked | Result |
|---|---|---|
| Bury et al.'s own normalization is by mean absolute value | `dl_train/DL_training.py` in their repo: computes `values_avg` as the mean of `abs()` over non-zero (unpadded) entries, then divides each sequence by it | Confirmed directly against their code — the previous verification log asserted this from the *paper text*, described as "verified"; I re-checked it against their actual *training script* this time, which is a stronger, code-level confirmation, not just a paper-level one |
| Bury et al. also use random padding as training-time augmentation | Same file: `pad_left`/`pad_right` random zero-padding applied to both ends of each sequence before normalization | Confirmed, and added as a citation supporting the left-censoring paragraph, which previously had no citation |
| Sapropel formation mechanism (orbital forcing → monsoon intensification → freshwater capping → anoxia), and the open question of whether onset is linear-response or bifurcation | Unchanged from previous (already-verified) chapter text, re-read against `hennekam2020` and `dakos2012` citations already in place | Not re-verified from scratch this pass (no wording changed here); carried forward from the prior verification |
| Proxy interpretations (Mo/U = redox, Ba = productivity, Ti/Al = terrigenous) | Unchanged from previous chapter text (`tribovillard2006`) | Not re-verified from scratch this pass; carried forward, no wording changed |
| Three cores (MS21, MS66, 64PE406E1) match Bury et al.'s own empirical test case | Unchanged from previous chapter text (`burygithub`) | Not re-verified from scratch this pass; carried forward, no wording changed |

The three rows above were not re-derived this pass because the sentences
containing them were not rewritten — only reformatted/shortened. I did not
independently re-fetch Hennekam et al. or Tribovillard et al. this session to
re-confirm them; that check was done in the original verification pass before
this rewrite, and the underlying facts did not change.

## Numbers computed fresh this session, with exact source

| Number in the chapter | How it was computed | Reproducible via |
|---|---|---|
| "Moving from L=500 to L=1500 raises AUC by 3.6 percentage points at the median" (19 models) | Computed directly from `results/summary/zenodo.csv`, mean/median of `(ts_1500 AUC − ts_500 AUC)` across every model with both | `python3` one-liner over `results/summary/zenodo.csv`, shown in-session; exact per-model deltas listed from +6.2 (mrsqm) to −12.6 (multirocket) |
| Table 5.1 (old vs. corrected sapropel ages, Mo enrichment factors, core 64PE406E1) | Computed directly from the raw `64PE406-E1_calibratedXRF.csv` file: mean Mo/Ba in a ±2 kyr window around each claimed age, divided by the core's own background median Mo | Script shown in-session; same computation is what originally caught the label bug two sessions ago |
| "mean AUC on the corrected empirical data rose from roughly 0.65 to the 0.77–0.96 range" | Taken from `results/summary/pangaea.csv`, restricted to the 12 models that also perform well on the labeled synthetic data (the "trustworthy" set used throughout `testing/bif_*.py`) | `results/summary/pangaea_by_model.csv` |
| Segment lengths, Figure 5.2 (before/after) | Row counts of the regenerated `*_forced.csv` files under `dataset/pangaea_923197/datasets/clean_dataset/`, compared against the row counts before `src/pangea_cleaner.py` was re-run with corrected ages | Directly observable via `wc -l` on the current vs. a backed-up copy of those files from earlier in the session |

## Claims softened or removed because they could not be verified

- "S1 was already close to correct... it is the youngest and best-preserved
  sapropel... the easiest one to date accurately by eye" — the second half of
  this sentence was an unverified guess at *why* S1's original age was
  already accurate. I did not check sediment preservation quality or dating
  precision for S1 specifically against any source. Reworded to state only
  the observation (S1's original age was close, the others were not) without
  the unverified causal explanation.
- "reduced accuracy by roughly 10 AUC points for the deep-learning models"
  (plural) — only one deep-learning model (`cnn_lstm`) and one classical
  model (`catch22`) were actually tested with interpolation, not a
  representative sample of "the deep-learning models" as a category.
  Reworded to name the two specific models tested and be explicit that this
  was not tested on the full 29-model roster.

## Second rewrite pass (2026-09-12): subsection-by-subsection re-verification

Per the same process now used for Chapter 4: each subsection re-verified
against a primary source or repo file, presented for approval, written
only after approval. Rules file: `chapters/WRITING_RULES.md`.

### §5.1 (Dual-Dataset Strategy) — verified, no changes

General ML reasoning + Bury-strategy claims, both already independently
verified in Chapter 4 §`sec:bury2021`. No edit needed.

### §5.2.1–§5.2.2 (SDE generation) — independently re-verified against the repo

The previous pass (above) checked these against the cloned Bury repo.
This pass re-checked them again, independently, directly against the
same files:

| Claim | File / line |
|---|---|
| 20 coefficients drawn N(0,1) | `training_data/gen_model.py`: `pars = np.random.normal(loc=0,scale=1,size=20)` |
| Sparsity = exactly half set to zero | `training_data/gen_model.py`: `sparsity=0.5; index_zero = np.random.choice(range(20),int(20*sparsity),...)` |
| 100-time-unit burn-in | `training_data/sim_model.py`: `tburn = 100` |
| Bifurcation parameter moves linearly | `training_data/sim_model.py`: `b = pd.Series(np.linspace(bl,bh,len(t)),index=t)` |
| Euler-Maruyama step | `training_data/sim_model.py`: `s[i+1] = s[i] + de_fun(s[i],pars)*dt + dW[i]` |
| Noise = $\sqrt{2\gamma}\tilde\sigma\xi$, $\tilde\sigma=0.01$, $\xi\sim$triangular(0.75,1,1.25) | `training_data/stoch_sims.py`: `sigma_tilde = 0.01`; `rv_tri = np.random.triangular(0.75,1,1.25)`; `sigma = np.sqrt(2*rrate)*sigma_tilde*rv_tri` |

All confirmed exact matches. No changes.

### §5.2.3 (Two Sequence Lengths) — two fixes, approved and written 2026-09-12

**Fix 1**: "Bury et al. released classifiers trained on sequences of
length L=500" was wrong — they released classifiers at **both** L=500 and
L=1500 (already corrected in Chapter 4 §`sec:bury2021`; this section had
not been updated to match).

**Fix 2**: the AUC-length-gain numbers were stale. The underlying
`results/summary/zenodo.csv` was evidently regenerated after this section
was last written — none of the old numbers ("−12.6" for MultiRocket,
"5–6 points (mrsqm, rocket, catch22)" as the strongest gainers) match the
current file on any column checked (binary_auc, macro_auc_ovr, accuracy,
macro_f1, auc_fold, auc_hopf, auc_transcritical).

Recomputed fresh, directly from the current file:

| Claim | Recomputed value |
|---|---|
| 19 models with results at both lengths | confirmed, unchanged |
| Median gain | +3.61 → still "3.6 points," unchanged |
| Strongest gains | rdst +9.31, lps +8.95, tsbf +7.65 (previously unnamed; mrsqm/rocket/catch22 are real gainers at +6.0 to +6.7 but not the top 3) |
| Weakest gains | cnn_lstm +1.30, tcn +1.54, lstm +1.55 (previous text said "the weakest DL models," unnamed) |
| These three's AUC at L=500 | cnn_lstm 0.9688, lstm 0.9657, tcn 0.9643 — supports "already close to their ceiling" |
| MultiRocket | **−6.44**, not −12.6 |

This is the same current-data recomputation already used and cross-checked
in Chapter 4 §4.2.4 — the two chapters now report identical numbers for
the same underlying comparison.

### Plagiarism check

No external-source wording involved in this fix (pure data description).
Clean.

### §5.3.1 (Sapropels as Real Tipping Points) — fixed and written 2026-09-12

The previous text said Hennekam "found signals consistent with an
approaching transition" as a blanket statement across all cores. Chapter 4
§`sec:hennekam_ews` (verified against Hennekam 2020 full text this
session) established this is depth-dependent: strong in the two deep
cores, largely absent in shallow MS21. Reworded to match, with a
cross-reference instead of repeating the full result.

Also swapped the parenthetical PANGAEA-DOI citation for a direct quote
from Bury's own repo README, re-checked directly this session:

| Claim | Source |
|---|---|
| Hennekam signal was depth-dependent (deep cores strong, MS21 weak) | Hennekam 2020 (already verified in Ch.4): *"Increasing variance is observed prior to all analyzed anoxic events in the relatively deep cores… the absence of rising autocorrelation and variance in the record of shallow core MS21"* |
| Bury used the same PANGAEA.923197 dataset via Hennekam 2020 | `README.md` line 89 in the cloned Bury repo, re-checked directly this session: *"Sedimentary archive data from the Mediterranean Sea are available at the PANGAEA (https://doi.pangaea.de/10.1594/PANGAEA.923197) data repository. Data were preprocessed according to… Hennekam, Rick, et al."* |

### §5.3.2 (The Five Geochemical Proxies) — verified, no changes

Mo/U as redox proxies already cross-checked against Bury 2021's own text
in Chapter 4 (*"molybdenum (Mo) and uranium (U), proxies for anoxic"*).
Ba/Ti/Al proxy roles are attributed to `tribovillard2006`, which was not
independently fetched full-text this session — flagged per the
source-coverage table in `WRITING_RULES.md` §9, no absence claim made
about it.

## §5.5.1 (Uneven Sampling: A Real Limitation) — approved and written 2026-09-12

Per user request: make the "same method as Bury, cited" comparison
explicit wherever this thesis's real-data preprocessing matches his,
rather than presenting an inherited choice as original.

Re-read Bury 2021's methods section (p.744, anoxia-specific paragraph,
not previously grepped closely) and found this thesis's real-data
preprocessing matches his almost exactly:

| Claim | Source |
|---|---|
| Bury does not interpolate the anoxia data; same reasoning (aliasing) | Bury 2021 p.744: *"Interpolation is not done, as most data points are equidistant, and it can give rise to aliasing effects that strongly affect variance and AC"* |
| Bury detrends the anoxia data with a Gaussian kernel, bandwidth 900 y | Bury 2021 p.744: *"smoothing the data with a Gaussian kernel with a bandwidth of 900 y"* — exact match to `config.yaml` `bandwidth_years: 900` |
| Bury uses rolling window = 0.5 | Bury 2021 p.744: *"EWS are computed using a rolling window of 0.5"* — exact match to `rolling_window_frac: 0.5` |
| cnn_lstm AUC drop (~10 pts) / catch22 small gain (2–3 pts) under interpolation | already verified in the first pass of this chapter's rewrite (this session), unchanged |

Rewrote §5.5.1 to state plainly that the no-interpolation decision is
"not new" and cite Bury's own stated reason, and reframed this thesis's
own empirical AUC-drop test as *confirming* Bury's reasoning rather than
being an independent discovery.

**Consequence for Chapter 3**: the same 900-year bandwidth / 0.5 window
match, discovered here, showed Chapter 3's methodology-comparison table
had wrongly attributed the 900-year bandwidth to "this thesis's own"
choice. Fixed there too — see Chapter 3 verification log, eighth pass.

### Plagiarism check

The one Bury quote used ("as most data points are equidistant...") is
short, attributed, and marked as a quotation. Clean.

## What was not re-verified in this pass

Anything carried over unchanged from the previous (already-verified) version
of the chapter — the sapropel formation mechanism, the proxy interpretations,
and the three-cores/Bury-et-al match — was not re-derived from primary
sources again here, since the sentences themselves were not rewritten. If you
want a from-scratch re-verification of those too (not just a check that the
wording didn't change), say so and I'll redo them the same way as the items
above, citing sources directly rather than trusting the earlier log.

## §5.1 (Dual-Dataset Strategy) — real error caught and fixed, 2026-09-12

User asked to double-check the current folder rather than trust a first
pass. Re-reading §5.1 against §5.2.2 (already correctly verified in this
chapter) turned up a direct self-contradiction: §5.1 said *"Because we
generate this data ourselves, we know the exact label of every single
example"*, while §5.2.2 correctly states *"This thesis does not run any
of the simulation... it downloads Bury et al.'s already-generated dataset
from Zenodo."* §5.1 was wrong. Fixed to state plainly that Bury et al.
generated the data and this thesis does not generate any of it.

Also reframed the whole subsection: opens with "This thesis solves the
bottleneck the same way \citet{bury2021} did" instead of presenting the
train/test split as this thesis's own idea, and closes with an explicit
statement that the dual-dataset strategy itself is Bury's, not original.

### Verified claims (re-checked directly against the current repo and
### Bury's cloned GitHub repo — not the paper alone, per user request)

| Claim | Source |
|---|---|
| "we generate this data ourselves" was false | contradicted §5.2.2; confirmed via `REPORT_NOTES.md` line 12 *"Source: Bury et al. (2021), zenodo.org/record/5527154"* |
| No training happens during PANGAEA evaluation | `testing/evaluate.py`: re-grepped this turn, zero hits for `backward`/`optimizer`/`loss.`/`zero_grad` |
| `train.py` writes checkpoints (`torch.save`) | `training/train.py` lines 160, 436, re-grepped this turn |
| 500,000 / 200,000 series, Zenodo record 5527154 | Bury repo `README.md` line 40, 56 — re-grepped fresh this turn (not just recalled from earlier in the session) |

## §5.5.2 (Fixed-Length Input and Left-Censoring) — approved and written 2026-09-12

Per user instruction, dropped the "tightened from an earlier min_visible
30 / pad_max_frac 0.9" claim carried over from the *first* chapter-5 pass
(an earlier session's git-diff-based check) — this session has no way to
confirm that historical claim from the current folder alone, and the user
has said not to use git history as a trusted source for "before" states.
Kept only the current, directly-checkable values.

Every Bury-repo claim in the new text re-verified directly against the
cloned GitHub repo this turn (not the paper, not memory):

| Claim | Exact repo evidence |
|---|---|
| Two padding variants: both-ends vs. left-only | `dl_train/DL_training.py` lines 44–52, comment: *"1: both left and right sides of time series are padded / 2: only left side of time series is padded"* |
| Both-ends pad: 225/225 (L=500), 725/725 (L=1500) | same file: `pad_left = 225 if ts_len==500 else 725; pad_right = 225 if ts_len==500 else 725` |
| Left-only pad: 450/0 (L=500), 1450/0 (L=1500) | same file: `pad_left = 450 if ts_len==500 else 1450; pad_right = 0` |
| Min visible = 50, both variants, both lengths | derived arithmetically from the constants above: `500−450=50`, `1500−1450=50` (left-only); `500−225−225=50`, `1500−725−725=50` (both-ends) — not a coded floor variable, a consequence of the chosen pad constants (noted here for the record only, not stated as a coding detail in the thesis text since the thesis only claims the outcome) |
| Ensemble of 20 networks | `dl_train/DL_apply.py`: **exact code comments** *"# Compute DL predictions from all 20 trained models"*, *"# Compute average prediction among all 20 DL classifiers"*, loop `for model_type in [1,2]: for kk in np.arange(1,11)` |
| This thesis: `both_sided: false`, `min_visible: 15`, `pad_max_frac: 0.97` | `config.yaml`, current values, re-read this turn |
| Shortest real segment (103 pts) needs 93% padding at L=1500 | (1500−103)/1500 = 0.931, from already-verified segment length |

## §5.5.3 (Normalization) — approved and written 2026-09-12

| Claim | Exact repo evidence |
|---|---|
| Bury normalizes by mean(abs()) over non-zero (unpadded) entries, divides those entries by it | `dl_train/DL_training.py` lines 97–110: averages `abs(sequences[i,j])` only where `!= 0`, then divides those same entries by `values_avg` — re-read directly this turn |

### Plagiarism check (all three subsections)

No source wording tracked; all facts/numbers attributed with `\citet{}`.
Clean.

## §5.6 (Feature Engineering: The 5-Channel Input) — fixed 2026-09-12

Same bug already fixed in Chapters 3 and 4: the variance growth ratio
channel was described as "the rolling variance's own rate of change,
divided by its current level," producing "a roughly flat growth-ratio
channel" for linear growth — not what the code computes.

Re-confirmed fresh this turn, directly against `src/ews_augmenter.py`:
`var_ratio = np.log1p(np.maximum(0.0, var_ch / (init_var + 1e-6)))` — a
log-scaled ratio of current windowed variance to the windowed variance at
the start of the record. Not a derivative.

**Fix applied:** reworded to `\log(1 + \text{Var}(t)/\text{Var}(0))`,
matching Chapters 3 and 4's wording exactly. Also added the explicit
Bury-comparison the user asked for throughout this chapter: \citet{bury2021}
feeds only the raw normalized residual, no other channel — stated
directly, with a pointer to Chapter 3 §`sec:fold_transcritical_challenge`
for the fuller originality argument (not restated here, to avoid
repetition per `WRITING_RULES.md` §4).

This resolves the "still to fix" item both `WRITING_RULES.md` §5 and
Chapter 3's seventh-pass verification note had flagged as outstanding for
this chapter (Chapter 6 still has the same wording to fix, separately).

### Plagiarism check

No source wording tracked. Clean.

## Post-review fix (2026-09-13): the 5-channel input is TSC-only, not universal

Surfaced while working through Chapter 6. §5.6 said "every input series
here is expanded... into five channels before being read by a model" —
unscoped, implying all models. False: verified directly against the
current pipeline this turn.

| Claim | Source |
|---|---|
| `train_tsc()` calls `augment_ews_channels()`; `train_dl_variant()` never does | `training/train.py`: `use_4ch`/`augment_ews_channels` only appears inside `train_tsc()` (line ~254–312); grepped the whole file for `ews_augmenter`/`use_4ch` — zero hits inside `train_dl_variant` or `train_dl_binary` |
| Same split at evaluation time | `testing/evaluate.py`: the channel-expansion function is literally named `_prepare_tsc_input()`, gated on `use_4channel`; the DL prediction path never calls it |
| `ARCHITECTURE_COMPARISON.md` already documents this | *"Input to every DL model in this repo is the single-channel right-aligned, left-padded residual x... The 5-channel EWS feature stack is applied to TSC models only, not to DL models."* |

**Fixed:** §5.6 rewritten to state the split explicitly — 22 classical
models get the 5-channel expansion, 7 deep-learning models get the raw
residual only, matching Bury et al.'s own single-channel input. Figure
5.4's caption updated to say the same. Same fix made in Chapter 3
(§`sec:fold_transcritical_challenge` and the chapter's closing paragraph)
and Chapter 4 (§4.5 item 2), logged in their own verification files.

### Plagiarism check

No source wording involved. Clean.

## Chapter 5 rewrite pass — complete (2026-09-12)

All sections (§5.1–§5.6) re-verified subsection-by-subsection, each
against a primary source or repo file (Bury's paper text, Bury's cloned
GitHub repo code, or this thesis's own current files), each approved by
the user before being written. One real self-contradiction was caught and
fixed (§5.1's false "we generate this data ourselves"). One stale numeric
claim was caught and fixed (§5.2.3's AUC-gain figures, regenerated
results). One inherited-not-original misattribution was caught and fixed,
with a consequence for Chapter 3 (the 900-year bandwidth). One
unverifiable historical claim was dropped rather than asserted (§5.5.2's
"tightened from 30/0.9"). The chapter-wide framing now makes explicit,
wherever true, that this thesis's real-data preprocessing deliberately
matches Bury et al.'s own — and names the few places (feature channels,
left-censoring parameters, model roster) where it doesn't.

## Fresh independent re-verification pass (2026-09-14, full-thesis audit)

Full re-check of every factual/numeric claim against current repo state and Bury's cloned GitHub repo, independent of the passes above. Found and fixed 3 real drift issues (results had changed since earlier passes) and confirmed everything else correct:

| Claim | Check | Result |
|---|---|---|
| L=500→1500 gain: median 3.6pts, top gainers, MultiRocket exception | Recomputed fresh from `results/summary/zenodo.csv` | **STALE — fixed.** Old text said rdst+9.3/lps+9.0/tsbf+7.7 top, cnn_lstm/tcn/lstm+1.3-1.5 weakest, MultiRocket −6.4. Current data: rdst+7.3/mrsqm+6.2/rocket+6.1 top, inceptiontime/weasel2+0.8 weakest, MultiRocket −9.7. Median (3.6) unchanged. Now matches Chapter 8's already-fixed numbers. |
| Table 5.1 (sapropel age/enrichment table) | Recomputed directly from `64PE406-E1_calibratedXRF.csv`, ±2kyr window mean / background median, matching the documented original methodology | Confirmed accurate, within rounding, no change needed |
| Random polynomial system (10 monomials, 20 coefficients, sparsity 0.5), AUTO continuation, "branch point"=transcritical, 100-unit burn-in, noise formula σ=√(2γ)·σ̃·ξ (σ̃=0.01, triangular ξ) | Read directly: `training_data/gen_model.py`, `run_cont.py`, `convert_bifdata.py`, `sim_model.py`, `stoch_sims.py` in Bury's cloned repo | All confirmed exact, verbatim match to code |
| Bury's padding: 225/725 both-ends, 450/1450 left-only, ensemble of 20 networks | `dl_train/DL_training.py` (`pad_left = 225 if ts_len==500 else 725`, etc.), `dl_train/DL_apply.py` ("Compute DL predictions from all 20 trained models") | Confirmed exact |
| "Minimum 50 visible points" | Not a coded constant — confirmed as correct arithmetic (500−450=50, 1500−1450=50) already documented in an earlier pass; re-confirmed the arithmetic holds | Confirmed, no change |
| This thesis's config: `min_visible: 15`, `pad_max_frac: 0.97`, `both_sided: false`; shortest real segment 103 points | Re-read `config.yaml` directly; recounted `*_forced.csv` row counts | Confirmed exact, unchanged |
| Bury normalization = mean(abs()) over non-zero entries | Re-read `dl_train/DL_training.py` lines 97-110 independently | Confirmed exact |
| Interpolation experiment: CNN-LSTM ~10pt drop both lengths, catch22 "2-3 point" gain "held across both" | Cross-checked against the exact numbers recorded earlier in this same log (line ~73, ~219) | **Inaccurate — fixed.** Only one catch22 number was ever recorded (0.774→0.783, ts_500, +0.9pts, not "2-3 points"), and catch22 was never tested at ts_1500. Text now states the real recorded numbers only, removes the unsupported "held across both" claim for catch22 specifically. |
| "Mean AUC on corrected empirical data rose from ~0.65 to 0.77-0.96 range" | Recomputed from `results/summary/pangaea_by_model.csv`, current 15-model trustworthy list, `primary_mou_n>=14` | **Minor drift — fixed.** Actual current range is 0.75-0.96, not 0.77-0.96 (trustworthy list composition changed since the original claim). |
| Cross-reference "discussed fully... in Chapter~\ref{ch:methodology}" (referring to PANGAEA evaluation results) | Checked which chapter actually reports these results | **Wrong reference — fixed.** Results are in Chapter 8 (`ch:results`), not Chapter 6 (`ch:methodology`, which is methodology only). Now points to `ch:results`. |
| PANGAEA DOI, three cores, proxy interpretations, Hennekam findings | Unchanged text, not re-derived (no wording changed) | Carried forward from prior passes, consistent with above findings |

Brace balance after fixes: 144 open / 144 close. All `\ref`/`\label` pairs resolve (including cross-chapter refs into `ch:theoretical_background`, `ch:related_work`, `ch:methodology`, `ch:results`). All 4 citation keys (`bury2021`, `burygithub`, `hennekam2020`, `tribovillard2006`) confirmed present in `references.bib`.

### Note on WRITING_RULES.md drift (not fixed — out of scope for this chapter's own file)
`chapters/WRITING_RULES.md` §5 still says "All trained models use 5 input channels" without the DL/TSC split — this is stale relative to the now-established 5ch-TSC-only / 1ch-DL fact (which chapter_5.tex itself already states correctly, line 177). WRITING_RULES.md itself was not edited as part of this pass since the directive was scoped to chapter_5.tex; flagging for whoever next touches that rules file.

## Full section-by-section fact-verification pass (2026-09-20)

Same standard as the Chapter 8 audit two turns earlier this session: every
number independently recomputed from the current working-tree file or a
directly-readable git object, never copied from the chapter's own existing
text. `src/rolling_window.py` is currently uncommitted (`ELEMENTS = ["Mo",
"U"]`, working tree) vs. the committed `HEAD` version (all five elements) —
checked the working-tree file, per instruction.

**Scope checked, confirmed correct, no change:**
- Model-roster counts: `config.yaml` `models.dl` = 7, `models.tsc` = 22 (both
  counted directly) — matches "7 deep-learning / 22 classical" throughout.
- `train_tsc()` (training/train.py:198) is the only function that calls
  `augment_ews_channels`; `train_dl_variant`/`train_dl_binary` never do.
  `testing/evaluate.py`'s DL path is gated the same way by `is_tsc_model()`.
  Re-grepped fresh, not carried over from an earlier pass.
- 5-channel order and variance-growth-ratio formula: `src/ews_augmenter.py`
  — raw, var, lag1_ac, skew, `log1p(var/(init_var+eps))` — exact match to
  the chapter's equation and to `WRITING_RULES.md` §5.
- `src/data_common.py`'s `left_pad_to`/`make_fixed_window`: right-aligned,
  left (start-of-window) zero padding by default — matches "Fixed-Length
  Input and Left-Censoring" exactly.
- `config.yaml`: `min_visible: 15`, `pad_max_frac: 0.97`, `both_sided:
  false`, `pad_mode: "zero"` (all under `augmentation:`/`inference:`) —
  exact match to the chapter's stated values.
- Table 5.1 (sapropel onset ages, Mo enrichment): every one of the 7
  corrected ages recomputed by reading `config.yaml`'s current
  `transition_kyr` values for core `64PE406E1` directly — exact match. Every
  one of the 7 original (wrong) ages recomputed by reading the last-committed
  `config.yaml` (`git show <commit>:config.yaml`, several commits back,
  before the correction, which is itself still uncommitted) — exact match.
  Every enrichment ratio (both columns) independently recomputed from the raw
  `64PE406-E1_calibratedXRF.csv` file (mean Mo in a $\pm2$ ka window around
  each claimed age, divided by the core's whole-record median Mo = 2.345
  mg/kg) — all 14 values matched the table to within rounding.
- "Two Sequence Lengths" AUC-gain figures (median 3.6pp; rdst +7.3, mrsqm
  +6.2, rocket +6.1 strongest; inceptiontime/weasel2 +0.8, lstm/tcn +1.7
  weakest; MultiRocket $-$9.7): recomputed fresh from `results/summary/
  zenodo.csv`. These do **not** match the `binary_auc` column (which gives a
  different top/bottom ranking and $-$6.44 for MultiRocket) but match the
  `macro_auc_ovr` column exactly. The chapter's own wording didn't specify
  which AUC, so this pass added "(one-vs-rest, ...)" to remove the ambiguity
  rather than treating it as an error — the numbers themselves are correct.
- Every `\ref{}`/`\label{}` pair (28 refs in this chapter, checked against
  every `\label{}` in all six chapter files) resolves. All 4 citation keys
  (`bury2021`, `burygithub`, `hennekam2020`, `tribovillard2006`) exist in
  `references.bib`. Brace balance 150/150 after the edits below (was 144/144
  before).

**Errors found and fixed:**

1. **"Every onset age... was wrong by 60 to 120 thousand years."** Recomputed
   the age error for all 7 sapropels affected in all 3 cores (not just the
   64PE406E1 rows shown in Table 5.1), reading both the current and the
   last-committed `config.yaml`: 64PE406E1 alone already reaches a maximum
   error of 137.6 ka (S8: 226.6 vs. 89.0), and MS66/MS21 add errors of
   63–74 ka. The true range across all three cores is 62.9–137.6 ka, not
   60–120. Fixed to "60 to 140 thousand years."

2. **"Mean AUC on the corrected empirical data rose from roughly 0.65 to the
   0.75–0.96 range."** The 0.75–0.96 half is correct (recomputed from
   `results/summary/pangaea_by_model.csv`, all 15 models with a complete
   primary-Mo/U evaluation, `primary_mou_n==14`: true range 0.7534–0.9625).
   The "0.65" figure is not reproducible for that 15-model set: only 6 of the
   15 trustworthy models (`arsenal`, `cnn_lstm`, `inceptiontime`, `lstm`,
   `minirocket`, `rocket`) have any committed result under the original,
   wrong ages at all — the rest were only ever evaluated after the
   correction. Recovered the pre-correction evaluation for those 6 directly
   from the last commit (`git show HEAD:test_result/<model>_pangaea_<core>_
   <sap>_<elem>_pangaea/result.json`, the old naming convention, still
   committed even though the working tree has since moved to per-length
   directory names) — confirmed by checking the recovered files' own
   `ages_kyr_bp` values fall on the *original* (wrong) sapropel ages, not the
   corrected ones. Mean AUC for those exact 6 models: **0.50** before
   (chance level — consistent with the "near-chance" argument the chapter
   already makes one paragraph earlier) and **0.95** after, on the same
   primary Mo/U segments. `chapter_8_verification.md`'s own first-pass log
   (line 17) separately records a "0.65 mean AUC... under the original,
   incorrect ages" figure sourced to a single-model (`catch22`) before/after
   re-eval, not a roster-wide mean — this is almost certainly where the
   chapter 5 "0.65" figure originated, generalized from one model to "the
   models that also perform well on the synthetic data" as a group, which it
   was never shown to represent. Fixed to report the honest, reproducible
   6-model comparison (0.50→0.95) alongside the already-correct full-roster
   corrected-data range (0.75–0.96).

3. **"Real segment length roughly doubled to quadrupled" (Figure 5.2, the
   seven most-affected segments).** Recomputed old vs. new row counts for
   the 7 segments this almost certainly refers to (the 7 primary Mo/U test
   sapropels used everywhere else in the thesis: 64PE406E1 S3/S4/S5/S6,
   MS21 S1, MS66 S1/S3) by diffing the currently-uncommitted `*_forced.csv`
   files against their last-committed (pre-correction) version
   (`git show HEAD:<path>` vs. the working-tree file, row count minus
   header). Actual growth ratios: 1.36$\times$ (64PE406E1 S6, 76→103) up to
   2.93$\times$ (64PE406E1 S5, 57→167) — "doubled to quadrupled" overstates
   both ends. Fixed to "grew by roughly 40\% to nearly 3$\times$."

4. **The Five Geochemical Proxies section never stated the Mo/U evaluation
   restriction.** Per this session's standing instruction (also the subject
   of a major Chapter 8 fix two turns ago): `src/pangea_cleaner.py` cleans
   all five elements (`ELEMENTS = ["Al","Ba","Mo","Ti","U"]`, unchanged,
   committed), but `src/rolling_window.py`'s current *uncommitted*
   working-tree version restricts `ELEMENTS = ["Mo","U"]` for the actual
   PANGAEA evaluation (a `ponytail:` comment at that line gives the
   redox-physical reasoning: Al/Ba/Ti are detrital/productivity proxies that
   only added noise to the type-classification consensus). Chapter 5 never
   said this anywhere, even though Chapter 8's own §`sec:pangaea_provenance`
   (`"Which proxies are used, and which are not"`) explicitly cites
   `Chapter~\ref{ch:data}` as having already established the redox-proxy
   reasoning — a forward reference to content that did not exist. Added one
   sentence to "The Five Geochemical Proxies" stating the restriction and
   pointing to Chapter 8's fuller discussion, so the cross-reference chain
   is no longer broken.

**Flagged, not corrected — no reproducible source found:**

- The interpolation-ablation numbers (CNN-LSTM 0.919→0.805 at $L=500$,
  0.962→0.857 at $L=1500$; catch22 0.774→0.783 at $L=500$) have no script or
  saved result file anywhere in the repository or its git history (checked
  `find . -iname "*interp*"` across the working tree and `git log --all
  --diff-filter=A --name-only | grep -i interp` across every commit — zero
  hits either way). These numbers were carried forward from earlier in this
  session's own ad hoc testing, per this chapter's prior verification-log
  entries, but nothing in the current repository can reproduce them. Left
  the numbers in place (removing them would delete a real, previously
  load-bearing finding on weak grounds) but added a sentence stating plainly
  that the script was not kept and the figures cannot be independently
  rerun from the current codebase.

Post-edit integrity check: brace balance 150/150. All 28 `\ref{}`s in the
chapter resolve against the full cross-chapter label set, including the two
newly-added references (`ch:results`/`sec:pangaea_provenance` and
`ch:methodology`/`sec:evaluation_metrics`). All 4 citation keys confirmed
present in `references.bib`.
