# Chapter 3 ("Theoretical Background") — Verification Log, Second Pass

This supersedes nothing from the first pass (citations, below) but adds a
second layer: checking the chapter's claims against the actual data-
generation code and this session's own results, not just against textbook
sources. The first pass only checked citations; it did not check whether
the "generating SDE" claim was literally true, which is the main finding
of this pass.

## 🔴 Same correction as Chapter 5 — required here too

§3.2 (now §3.2, "Bifurcation Theory and Normal Forms") previously read: "In
this thesis, aligning with the synthetic SDE datasets generated in
\citet{bury2021}, we focus on three primary types of bifurcations..." — this
implies the normal forms shown are literally what was numerically simulated
to produce the training data.

Checked directly against Bury et al.'s public data-generation code
(`training_data/gen_model.py`, `sim_model.py` in `ThomasMBury/deep-early-warnings-pnas`,
fetched and read this session — see `chapter_5_verification.md` for the full
code excerpts): the actual generator is a randomly-parameterized cubic
polynomial system verified via AUTO bifurcation-continuation software to pass
through the target bifurcation type, not the bare normal form equation.

**Fix applied:** reworded the introduction to §3.2 to state plainly that
these are the canonical forms (guaranteed correct near any bifurcation of
that type, by the Center Manifold Theorem — already correctly cited to
`kuznetsov2004, strogatz2015` in the first pass) and point to Chapter 5,
Section "Synthetic Training Data: The SDE Corpus" for what was actually
simulated. No new claim is made here that needs a new citation — the fix is
removing an implied claim, not adding one.

## Claims fixed this pass, with what was checked

| Claim | What I checked | Fix |
|---|---|---|
| "as demonstrated in our experiments, a window that is too small lacks the statistical degrees of freedom..." (§3.5.1) | This is now directly true and citable: this session's own work found rolling windows of 14–19 points on real PANGAEA segments had a standard error on the lag-1 autocorrelation estimate (≈1/√(window−1)) comparable to or larger than the ≈0.2–0.4 rise in AC1 the method is trying to detect, and corrected the window-fraction wiring in `src/rolling_window.py` in response | Reworded to explicitly point to Chapter 5's label-correction section, where this is discussed with the real numbers, instead of leaving it as an unsupported "as demonstrated" |
| "as we observed in our empirical analysis of left-censored Mediterranean time series, improper data padding... causes severe spectral leakage" (§3.5.1) | I could not find or reproduce a specific test of this claim (spectral leakage / edge artifacts from zero-padding *before classical detrending specifically*) anywhere in this project's history or this session's work. What **was** tested this session is a different thing: zero-padding a short real segment before it enters a *trained classifier* (not before a classical detrending filter) — that test is already correctly described in Chapter 5. | This specific sentence's claim was not verifiable, so it was softened from "as we observed in our empirical analysis" (a claim to a specific finding) to a general, standard signal-processing caution with no claimed empirical source. If you have a specific result behind the original wording, tell me and I will restore a precise, cited version of it instead. |
| "Chapter 6" (hardcoded, end of §3.5.2) | Not a factual claim, but hardcoded chapter numbers break if the document is restructured (e.g. a chapter is added or removed elsewhere) | Changed to `\ref{ch:methodology}`, consistent with how Chapter 5 now cross-references other chapters |
| "Section 3.3" (×2) and "Sections 3.2 and 3.3" (hardcoded, §3.4 and §3.5.1) | Same reasoning as above | Changed to `\ref{sec:linear_stability}` / `\ref{sec:normal_forms}`, and added the corresponding `\label{}`s that didn't exist before this pass, so Chapters 4 and 5 can now cite these subsections precisely instead of just the whole chapter |

## Resolved (third pass): variance growth ratio channel — searched online as requested

The first pass flagged this as unresolved; the second pass (above) removed
"novel" without checking further. This pass did the actual search, in three
steps:

1. **General web search** for prior work distinguishing Fold from
   Transcritical bifurcations via variance behaviour. Result: the
   literature confirms the premise this thesis states (variance and
   autocorrelation rise similarly for both, and generically cannot tell
   them apart) — e.g. Kéfi et al. 2013 and related EWS-robustness papers.
   No paper found proposing an engineered "variance growth ratio" channel
   specifically.
2. **Search for the underlying scaling-law concept** (whether different
   bifurcation types have theoretically distinguishable *rates* of change
   in their CSD statistics, not just whether the statistics rise). This
   surfaced a real body of work on power-law scaling of recovery
   rate/variance with distance from a bifurcation, with different
   characteristic exponents reported for different bifurcation types in
   search summaries. I was **not** able to pin this down to one specific,
   confidently-citable source within the scope of this pass — I checked one
   candidate (Prettyman, Kuna & Livina 2022, *Environmental Research
   Letters* 17:035004) directly and it turned out to only cover the AR(1)
   case, not a cross-bifurcation-type comparison, so I did not cite it. The
   chapter now states this general point without a specific citation,
   flagged as such, rather than attach a citation I couldn't confirm
   actually supports it.
3. **Direct check against the two papers most likely to already contain
   this exact technique**: \citet{bury2021} (already established, raw
   series input) and its closer, more relevant follow-up,
   \citet{bury2023discretetime} ("Predicting discrete-time bifurcations
   with deep learning," *Nature Communications* 14:6331, 2023 — same
   author lineage, also specifically about classifying bifurcation type).
   Fetched this paper's methods section directly: it feeds only the raw,
   normalized time series into its classifier ("normalised by their mean
   absolute value and prepended with zeros"), with no variance-derivative
   or growth-rate input channel of any kind.

**Conclusion, stated in the chapter now**: this thesis's construction —
feeding a variance-growth-rate ratio directly to a classifier specifically
to address the Fold/Transcritical ambiguity — does not appear in Bury et
al.'s original or follow-up work, and a general (non-exhaustive) search
found no other paper doing exactly this either. This is reported as this
thesis's own construction, with the caveat, stated explicitly in the
chapter text, that this rests on a real but bounded search rather than a
systematic literature review, and added `\citet{bury2023discretetime}` to
`references.bib` for the specific claim it's cited against.

## Language changes (no factual content changed)

Per your instruction to write in plain, simple English: removed or toned
down phrases that added no information but added drama — "devastating false
positive alarms," "desperately wish to predict," "perhaps most critically,"
"powerful paradigm shift." These were stylistic only; nothing measurable or
factual was cut or changed by removing them. Also shortened a few sentences
that repeated the same point twice in a row.

## Figures added

Both previously-empty placeholder figures (`\rule{0.8\textwidth}{6cm}`) are
now real, generated figures:

- **Figure 3.1** (bifurcation landscapes): the actual potential-well function
  $V(x)$ for the Fold ($V = -\lambda x + x^3/3$, since $dx/dt = -dV/dx =
  \lambda - x^2$) and Transcritical ($V = -\lambda x^2/2 + x^3/3$) normal
  forms, plotted at three values of $\lambda$ each, plus the Hopf phase
  portrait (fixed point vs. limit cycle). Generated directly from the normal
  form equations already in this chapter — not illustrative, the potential
  functions are the literal integral of $-f(x)$ for each equation shown.
- **Figure 3.2** (AR(1) comparison): two real simulated AR(1) series at
  $\alpha=0.2$ and $\alpha=0.9$, with their $x(t)$-vs-$x(t-1)$ scatter plots,
  generated with a fixed random seed (reproducible) rather than describing
  what such a plot would look like without showing one.

## Carried forward unchanged from the first pass (not re-verified this pass)

- All four citation fixes (Bury 2021, Dakos 2012's journal correction,
  Scheffer 2009, Bury 2021 again) — no reason to re-check citations that
  didn't change.
- The Fold/Transcritical/Hopf normal form equations themselves, and the
  AR(1)/OU discretization math (`α = e^{μΔt}`, `Var(y) = σ_ε²/(1−α²)`) — this
  is standard, checkable textbook mathematics, already confirmed correct
  against Kuznetsov and Strogatz in the first pass, and no wording in these
  derivations changed in this pass.
- Center Manifold Theorem and Ornstein-Uhlenbeck citations — unchanged.

## What I did not check in this pass

I did not re-verify the Dakos et al. (2012) methodology description
(sliding-window + Gaussian detrend + Kendall's τ) against the paper a second
time — the first pass already did this and the sentence describing it was
not rewritten here.

## Fourth pass: depth expansion (4,328 → 5,749 words)

Per your request for a much fuller thesis (~70 pages total across all
chapters, not a shallow pass), added five substantive new pieces of content,
each checked as described below — no padding, every addition is either a
derivation I verified by hand, a fact checked against the actual pipeline
code, or a citation checked before being added.

1. **What a center manifold actually is** (§3.2 intro) — previously the
   Center Manifold Theorem was named but never explained. Added a real
   explanation (eigenvalue directions with non-zero real part decay fast and
   drop out; the manifold is what's left) and connected it explicitly to why
   the normal forms are still the right theoretical tool even though the
   actual training data is generated from a wider family of systems, not the
   bare equations (Chapter 5's finding).
2. **A worked numerical example for the Fold bifurcation** (Table 3.1) —
   computed $\tau = 1/(2\sqrt{\lambda})$ at four $\lambda$ values by hand
   (checked: $\lambda=1\to\tau=0.5$, $0.25\to1.0$, $0.04\to2.5$,
   $0.01\to5.0$, all follow directly from the recovery-rate formula already
   derived in the chapter). Makes Critical Slowing Down concrete with actual
   numbers instead of only the abstract limit.
3. **A new subsection deriving 2D linear stability (the Jacobian)** before
   the Hopf bifurcation, since the existing text jumped straight to the
   Hopf normal form in polar coordinates without ever explaining where the
   two-dimensional eigenvalue analysis it relies on comes from. Standard,
   checkable textbook material (Strogatz *Nonlinear Dynamics and Chaos*,
   already cited elsewhere in this chapter) — trace/determinant eigenvalue
   formula, complex-conjugate-pair classification, and the direct
   observation that the Hopf normal form's $\omega$ is literally the
   imaginary part of the Jacobian eigenvalues at criticality.
4. **Two real-world examples with citations**: the SIR epidemic threshold
   ($R_0=1$) as a Transcritical bifurcation, citing `miry2025transcritical`
   (found and verified this session, flagged honestly as an unreviewed
   preprint, not a peer-reviewed source); predator-prey (Rosenzweig-MacArthur)
   population cycles as a Hopf bifurcation example, citing the already-used
   `strogatz2015` textbook (a standard example covered in that book, not a
   new claim needing a new citation).
5. **The OU process's stationary distribution and autocorrelation function**,
   derived and connected explicitly to the AR(1) discretization already in
   the chapter — $\text{Var}(y)=\sigma^2/(2|\mu|)$ and $\rho(s)=e^{\mu s}$,
   the latter shown to be exactly where $\alpha=e^{\mu\Delta t}$ comes from
   when $s=\Delta t$. Standard stochastic-process results, not independently
   sourced to one paper (same treatment as the existing AR(1) derivation,
   which the original pass already noted was "too elementary to trace to
   one paper").
6. **A new subsection on AR(1) surrogate significance testing** — this is
   the most consequential addition. I checked `testing/evaluate.py`
   directly (functions `_fit_ar1_neutral` and `_generate_ar1_surrogates`)
   rather than describe classical surrogate-testing generically, and
   confirmed the real numbers used: $\alpha$ and $\sigma$ are fit on only
   the first 20% of each forced segment (docstring: "Mirrors Bury's
   `generate_nulls_ar1.py` exactly"), 10 surrogates are generated per
   segment, and $\sigma$ is computed as $\sqrt{\text{Var}(y)(1-\alpha^2)}$
   — which is exactly the stationary-variance formula already derived
   earlier in this same chapter, rearranged. This ties the chapter's own
   math directly to the real evaluation pipeline used for every PANGAEA
   result in this thesis, rather than leaving classical significance
   testing as a disconnected literature description.

All new labels (`sec:jacobian`, `sec:ar1_model`, `sec:ar1_surrogates`) and
the one new cross-chapter label added to Chapter 6
(`sec:evaluation_metrics`) were checked for resolution across the whole
document after this pass — no dangling references, no duplicate labels, no
unresolved citations.

## Fifth pass: cut to match your actual page target and "only what's used"

You asked for 8-9 pages per chapter (not ~70 total) and specifically asked
whether the fourth-pass additions were things this thesis's pipeline
actually uses. Answered honestly (see table below, also given directly in
chat) and cut accordingly:

| Addition from pass 4 | Actually used by the pipeline? | Kept? |
|---|---|---|
| Center manifold explanation | No — theoretical justification only | Kept, shortened (it directly explains why the normal forms are valid despite Chapter 5's random-polynomial finding, which is a real, load-bearing point) |
| Worked numerical Fold example (table) | No — illustrative, made-up λ values | **Cut** |
| Jacobian / 2D linear stability subsection | No — code never computes a Jacobian | **Cut entirely** |
| Epidemic threshold (SIR) example | No — external domain, not this thesis's data | **Cut**, and the now-unused `miry2025transcritical` citation removed from `references.bib` too |
| Predator-prey (Hopf) example | No — external domain, not this thesis's data | **Cut** |
| OU stationary variance/ACF | Indirectly — feeds the σ formula the real code uses | Kept, short |
| AR(1) surrogate testing subsection | **Yes** — checked directly against `testing/evaluate.py` | Kept in full, this is the one section that's actually load-bearing |

Chapter is now 2,011 words (down from the 5,749-word fourth-pass version,
and below the original 4,328 too) — likely 5-6 pages by the same rough
estimate used earlier in this session, under your 8-9 page target rather
than over it. I did not pad it back up with more external examples to hit
the page count, since that would reintroduce exactly what you just asked me
to remove. If more length is wanted, the honest way to add it is more
depth on content that \emph{is} used (e.g. more on Kendall's τ mechanics,
since that's actually computed in this thesis's evaluation), not more
illustrative material from outside this thesis's own work — flagging this
rather than guessing which you'd prefer.

## Sixth pass: exact computation detail + three-way methodology comparison

Added the depth you asked for specifically: how variance/autocorrelation are
actually computed in this thesis's own code, and an honest comparison
against Dakos's classical method and Bury's DL method — including where
this thesis's choices are original and where they are directly inherited,
rather than a blanket "our approach is better" framing.

- **Exact computation** (new subsection): pulled the literal formulas from
  `src/rolling_window.py` (`_variance`: `np.var(series, ddof=1)`; `_lag1_ac`:
  Pearson correlation between the series and itself shifted by one step) and
  `src/pangea_cleaner.py` (Gaussian-kernel detrending, bandwidth 900 years,
  confirmed against the `bandwidth_years` config key and the `norm.pdf`
  weighting in `smooth_all_elements()`).
- **Three-way comparison table** (new): Dakos (classical) vs. Bury et al.
  2021 vs. this thesis, on detection mechanism, window convention,
  detrending method, bifurcation-type capability, and significance testing.
- **Explicit honesty about what's inherited vs. original**: the AR(1)
  surrogate procedure is stated plainly as copied from Bury's own code; the
  half-length rolling window is stated as matching Dakos's convention, not
  a departure. The one thing genuinely different and justified on its own
  terms: this thesis uses a fixed 900-year detrending bandwidth rather than
  a fraction-of-segment-length bandwidth (Bury's convention), because
  segment lengths in this thesis's corrected data vary from ~100 to ~650
  points, and a fraction-based bandwidth would apply a different physical
  smoothing scale to each one — no justification for that, given the
  underlying process (orbital forcing) operates on a fixed real-time scale.
- **Deliberately avoided a "which is better" verdict** in the abstract — the
  chapter states specifically what can and cannot be claimed with the
  evidence gathered (better-than-chance separation on binary detection; not
  reliable on the 4-class type question, a limitation neither prior work
  directly measured either).

**New cross-chapter dependency**: this section forward-references a Results
chapter (`Chapter~\ref{ch:results}`). Since that chapter doesn't exist yet
in the agreed one-at-a-time order, created a minimal placeholder
`chapter_8.tex` (just `\chapter{}` and `\label{ch:results}`, no content) so
the reference resolves correctly now rather than dangling or being
hardcoded as a literal chapter number. To be filled in properly when we
reach Chapter 8.

Word count after this pass: 2,673 (up from 2,011).

## Seventh pass: variance growth ratio channel re-described to match the code

Triggered while verifying Chapter 3 against the current pipeline before
starting Chapter 4. The "variance growth ratio" channel was described in
§`sec:fold_transcritical_challenge` as "computes the rate directly" / "the
rolling variance's own rate of change, divided by its current level".

Checked directly against `src/ews_augmenter.py` (`_rolling_channels_chunk`,
the 5th channel), which is what every trained checkpoint actually saw — all
`checkpoints/*_ch_stats.npz` have shape `(5,)`, confirming 5-channel
training. The channel is:

```python
init_var  = var_ch.replace(0.0, np.nan).bfill().iloc[0].fillna(1e-6)
var_ratio = np.log1p(np.maximum(0.0, var_ch / (init_var + 1e-6)))
```

i.e. `log(1 + Var_window(t) / Var_window(start))` — a log-scaled ratio of
the current windowed variance to the baseline windowed variance. It is
**not** a derivative and has no "divide by current level" term. For a
linearly rising variance this channel rises (log-of-linear), it is not
"roughly flat" as the old wording implied.

**Fix applied:** reworded the sentence to describe the actual feature (ratio
to baseline windowed variance, then `log1p`) and to explain that the
Fold/Transcritical rate difference shows up in it as a difference in the
*shape* of the climb (faster, more sharply upward-curving for a Fold), not
as an explicit rate the model is handed. The originality claim is
unaffected — Bury 2021 and Bury 2023 still feed only the raw series, with no
engineered variance-growth channel of any kind — so that paragraph stands;
only "no rate-of-change channel" was softened to "no engineered
variance-growth channel".

**Same wording, status by chapter:**
- `chapter_5.tex` §`sec:feature_engineering` — **fixed 2026-09-12**
  (Chapter 5's own verification log, "§5.6" entry), now reads
  `\log(1 + \text{Var}(t)/\text{Var}(0))`, matching this chapter.
- `chapter_4.tex` §4.5 item 2 — **fixed 2026-09-11** (Chapter 4's own
  verification log), same wording.
- `chapter_6.tex` (line ~12): lists "the variance growth ratio" in the
  channel enumeration — the name is fine, just check the cross-reference
  once Chapter 6 gets its pass; still outstanding.

**Not changed (checked, still correct against current code):** AR(1)
surrogate description (`AR1_FIT_FRACTION = 0.20`, `N_SURROGATES = 10`,
`sigma = sqrt(var*(1-alpha^2))` in `testing/evaluate.py`); detrending
bandwidth 900 yr / `bandwidth_years` in `src/pangea_cleaner.py`;
half-segment classical window (`pangaea.rolling_window_frac: 0.5`);
`_variance` = `np.var(ddof=1)` and `_lag1_ac` = shift-1 Pearson in
`src/rolling_window.py`.

**Left as-is deliberately (cosmetic, not factual errors):** §3.1 uses
"molybdenum or barium" as an example proxy though the PANGAEA eval now
restricts to Mo+U; §3.5.4 names MiniRocket and WEASEL2 as example models
though both are excluded from the type-consensus roster. Both are generic
illustrations in a theory chapter, not claims about the pipeline.

## Eighth pass (2026-09-12): the 900-year bandwidth is not this thesis's own choice

Surfaced while working through Chapter 5's data-processing comparison to
Bury et al. — checked directly against Bury 2021's own methods text
(p.744): *"We perform the same data preprocessing as Hennekam et al. (31)
... Residuals are obtained from smoothing the data with a Gaussian kernel
with a bandwidth of 900 y, and EWS are computed using a rolling window of
0.5."* — an exact match to this thesis's own `config.yaml`
(`bandwidth_years: 900`, `rolling_window_frac: 0.5`) for the same anoxia
cores.

The seventh pass (above) explicitly listed the 900-year bandwidth as
"not changed... still correct" — that check only confirmed the number
matches the current code, not that the *attribution* ("this thesis's own
choice") was correct. It wasn't: Bury used the same bandwidth for the same
real cores, having taken it from Hennekam et al. in turn. Three choices in
Table~\ref{tab:methodology_comparison}'s "This thesis" column were
originally-claimed but are actually all inherited (AR(1) surrogate test,
half-length window, and now the 900-year bandwidth) — only one column
entry (keeping classical variance/AC1 as a surrogate-test support rather
than the detector) is genuinely this thesis's own.

**Fix applied:**
- Table row "Detrending": Bury's column now shows both of his actual
  choices — Lowess span 0.2 for the synthetic training data, Gaussian
  900 y for his own anoxia evaluation — and "This thesis" column now
  states plainly that it matches Bury's anoxia-evaluation choice exactly,
  rather than presenting the number as original.
- Prose: "Two choices are this thesis's own" → "Three of the choices...
  taken directly from earlier work" (adding the 900-year bandwidth to the
  inherited list, with the Bury quote), and "Two choices are this thesis's
  own" → "One choice in the table is this thesis's own" (only the
  surrogate-support framing remains as an original choice).
- Kept, reframed: the reasoning for *why* a fixed-in-years bandwidth
  (rather than fraction-of-segment-length) makes sense given this
  thesis's wide segment-length spread (100–650 points) is still stated —
  that reasoning is a genuine point even though the specific number (900)
  isn't original.

### Plagiarism check

The Bury quote is short (13 words), directly attributed, and marked as a
quotation. No paraphrase drift. Clean.

## Ninth pass (2026-09-13): "every model" gets the variance growth ratio — false

Surfaced while working through Chapter 6. Traced directly against the
current pipeline: `training/train.py` — `train_tsc()` (classical models)
calls `augment_ews_channels()` (the 5-channel expansion); `train_dl_variant()`
(deep-learning models) never does. Confirmed identically in
`testing/evaluate.py`'s `_prepare_tsc_input()` (TSC-only) vs. the DL
prediction path. `ARCHITECTURE_COMPARISON.md` already stated this
plainly: *"Input to every DL model in this repo is the single-channel...
residual... The 5-channel EWS feature stack is applied to TSC models
only, not to DL models."*

Two sentences in this chapter (§3.5.1 line ~43, §3.5.4 line ~127) said
"every model" / "every model is handed an input" — both false. The
variance growth ratio channel (and the other 4 channels) goes only to the
22 classical models; the 7 deep-learning models read the raw residual
directly. Fixed both sentences to state this split explicitly. Same fix
made in Chapter 4 §4.5 item 2 and Chapter 5 §`sec:feature_engineering`,
logged in their own verification files.

### Plagiarism check

No source wording involved. Clean.

## Tenth pass (2026-09-14, fresh independent audit): window-fraction misattribution found and fixed

Full re-audit of every checkable claim in the chapter, independent of prior
passes' conclusions. Checked directly against source, not against the
verification log:

| Claim | Source checked | Result |
|---|---|---|
| `_variance`/`_lag1_ac` formulas (n-1 sample variance; shifted-copy correlation) | `src/rolling_window.py:40-50`, read directly | Confirmed exact match |
| Gaussian detrending = weighted local mean subtracted, bandwidth in years converted to ka | `src/pangea_cleaner.py:109-136`, read directly | Confirmed exact match |
| `pangaea.rolling_window_frac: 0.5` actually drives the classical variance/AC window (not some other window) | `src/rolling_window.py:84` (`win_frac = cfg["pangaea"]["rolling_window_frac"]`), read directly | Confirmed |
| AR(1) surrogate: first 20%, `sigma=sqrt(var*(1-alpha^2))`, 10 surrogates | `testing/evaluate.py:46-101`, read directly | Confirmed exact match on all three |
| "Mirrors Bury's generate\_nulls\_ar1.py exactly" (first-20% claim) | `/home/s466553/abc/deep-early-warnings-pnas/test_empirical/anoxia/generate_nulls_ar1.py`, independently grepped (not just trusting this thesis's own docstring) | Confirmed: file's own comment says "first 20% of data points" |
| Segment lengths "about 100 to 650" | Counted rows directly in all 7 `*_forced.csv` files | Confirmed: min 103, max 648 |
| Bury quote "smoothing the data with a Gaussian kernel with a bandwidth of 900 y" | `bury2021.txt` lines 745-746 | Confirmed exact |
| §sec:fold_transcritical_challenge: variance-growth-ratio channel absent from both `bury2021` and `bury2023discretetime` | Re-checked `bury2021.txt` (no hits for variance/growth/channel terms) and `/tmp/claude-216236/bury2023_discrete.txt` (confirms only "normalised... prepending it with zeroes... feeding it into the classifier", no engineered channel) | Confirmed, independently, this pass |
| Citation keys kuznetsov2004, strogatz2015, scheffer2009, bury2021, bury2023discretetime, uhlenbeck1930, dakos2012 | `references.bib` | All present |
| All `\ref`/`\label` pairs | Brace-balance + label scan | 114/114 braces; 4 refs are legitimate cross-chapter (into ch5/ch6), all confirmed to resolve |

**Error found and fixed:** the same class of mistake as the eighth pass's
bandwidth fix, but for the *window fraction*, missed at the time. Bury et
al.'s own anoxia methods sentence gives both numbers together — "smoothing
the data with a Gaussian kernel with a bandwidth of 900 y, **and EWS are
computed using a rolling window of 0.5**" — but the chapter attributed the
0.5 window only to \citet{dakos2012}, never mentioning that \citet{bury2021}
independently states the identical value for this same anoxia evaluation.
Fixed in three places: the §sec:actual_computation sentence, the
methodology-comparison table's "Window" row (Bury column now separates his
classifier's fixed-length input from his classical-EWS comparison's 0.5
window; "This thesis" column now credits both Dakos and Bury), and the
§sec:methodology_comparison prose paragraph (now quotes both numbers from
the same source sentence together).

No other errors found. Every other checked claim held exactly.

### Plagiarism check
The added Bury quote (17 words, "EWS are computed using a rolling window of
0.5") is short, directly attributed, and quoted verbatim inside quotation
marks — not paraphrased as original text. Clean.

## Eleventh pass (2026-09-20): full section-by-section re-audit, same rigor as Chapter 8

Independent re-verification of every checkable claim in the chapter against
current files on disk, not against this log's prior conclusions. Sources
checked directly: `src/ews_augmenter.py`, `src/rolling_window.py`,
`src/pangea_cleaner.py`, `config.yaml`, `testing/evaluate.py`,
`training/train.py`, `models/__init__.py`, all seven `*_forced.csv`
segment files, `/home/s466553/abc/deep-early-warnings-pnas/` (Bury's own
repo: `generate_nulls_ar1.py`, `compute_resids.py`), `bury2021.txt`
(full text), and Dakos et al. 2012's own methods/results text (fetched
fresh this pass, presence claim only, not used for any absence claim per
WRITING_RULES §9).

**Error found and fixed:**

1. **§"The Deep Learning Response" (closing paragraph) named MiniRocket and
   WEASEL2 as examples of models that "build their own way of reading the
   raw record," in the same breath as CNN-LSTM and InceptionTime.** MiniRocket
   and WEASEL2 are classical TSC models, not deep-learning models
   (`config.yaml` `models.tsc` lists both; `models.dl` lists neither).
   Checked `training/train.py`'s `train_tsc()` (calls
   `src/ews_augmenter.augment_ews_channels()`) and `models/__init__.py`'s
   `is_tsc_model()` directly: MiniRocket and WEASEL2 both receive the full
   5-channel input (raw residual + variance + lag-1 AC + skewness + variance
   growth ratio), not the raw series alone. This directly contradicted the
   very next sentence in the same paragraph, which correctly states that
   "classical models" get the variance-growth-ratio channel and
   "deep-learning models" are left with the raw series only — MiniRocket/
   WEASEL2 were introduced one sentence earlier as if they were in the
   raw-series-only group. Reworded to split the example list correctly:
   CNN-LSTM/InceptionTime (deep learning, raw series) vs. MiniRocket/WEASEL2
   (classical TSC, raw series plus four engineered channels), matching
   WRITING_RULES §5's model-family split exactly.

**Independent re-derivations and re-checks (all confirmed correct, no
change needed):**

- Variance growth ratio channel: `src/ews_augmenter.py`'s
  `_rolling_channels_chunk()` computes exactly
  `log1p(max(0, Var_window(t) / (Var_window(start) + eps)))` — matches the
  chapter's description word for word, not described as a derivative
  anywhere.
- Fold-vs-Transcritical acceleration direction: re-ran an independent check
  (different metric, different random sample of 60 Fold + 59/60
  Transcritical series from `dataset/ts_500/combined/cache_residuals.npy`
  and `cache_labels.npy`, using `_rolling_channels_chunk` from the actual
  pipeline code) — mean second-half-growth/first-half-growth ratio of the
  variance-growth channel was 0.844 for Fold vs. 0.720 for Transcritical,
  same direction as this session's earlier 2.64x-vs-1.95x result (Fold
  accelerates more than Transcritical). The chapter itself states this only
  qualitatively (no specific ratio numbers appear in the text), so there
  was nothing numeric to correct — the qualitative claim is confirmed
  correct by two independent measurements now, not one.
- Classical variance/AC1 formulas: `src/rolling_window.py` `_variance()`
  (`np.var(ddof=1)`) and `_lag1_ac()` (Pearson correlation of `series[:-1]`
  vs. `series[1:]`) match the chapter's description exactly.
- Gaussian detrending, 900-year bandwidth: `src/pangea_cleaner.py`
  `smooth_all_elements()` — weighted local average via `norm.pdf` kernel,
  bandwidth converted from years to ka, subtracted from raw value — matches.
  `config.yaml` confirms `pangaea.bandwidth_years: 900`.
- Half-segment classical window: `config.yaml` `pangaea.rolling_window_frac:
  0.5`, and `src/rolling_window.py:84` confirms `cfg["pangaea"]
  ["rolling_window_frac"]` (not the separate `inference.rolling_window_frac:
  0.25`, which governs a different, unrelated window) is what actually
  drives the real-data classical EWS window.
- Bury quote ("smoothing the data with a Gaussian kernel with a bandwidth
  of 900 y, and EWS are computed using a rolling window of 0.5") re-checked
  against `bury2021.txt` lines 745–746: exact match.
- Dakos et al. 2012's own "half the series length" window convention:
  re-fetched the paper's own text this pass ("We estimated autocorrelation,
  variance..., and skewness within rolling windows half the size of the
  datasets") — confirms the chapter's attribution to Dakos independently
  of Bury's quote, as a presence claim from the fetched-summary source
  (not an absence claim, consistent with WRITING_RULES §9).
- Bury's own synthetic-training-data detrending: independently found and
  read `training_data/compute_resids.py` in Bury's repo directly (not
  relying on the earlier passes' citation) — `span = 0.2` Lowess, matching
  the table's "Lowess, span 0.2 (synthetic training data)" cell.
- Bury's 4-class output: `bury2021.txt` and Bury's own `dl_apply.py`
  script (`/tmp/bury_dl_apply.py`) both confirm the four classes are fold /
  Hopf / transcritical / null — matches the table's "In principle, via
  4-class output" cell.
- AR(1) surrogate test: `testing/evaluate.py` `_fit_ar1_neutral()` /
  `_generate_ar1_surrogates()` — first 20% fit fraction
  (`AR1_FIT_FRACTION = 0.20`), `sigma = sqrt(var*(1-alpha**2))`, 10
  surrogates (`N_SURROGATES = 10`) — all match the chapter's description
  exactly. Independently re-grepped Bury's own
  `test_empirical/anoxia/generate_nulls_ar1.py` (not just re-trusting this
  log's earlier transcription): confirms "first 20% of data points" and the
  same sigma formula.
- Segment length range "about 100 to 650": recounted rows in all seven
  current `*_forced.csv` files on disk (these files show as modified in
  `git status` at the start of this session, so recomputed fresh rather
  than trusted) — min 103 (`64PE406E1_S6`), max 648 (`MS21_S1`), unchanged
  from the prior pass's count despite the working-tree modifications.
- Fold/Transcritical/Hopf normal-form equilibria and stability (via
  $f'(x^*)$ sign) re-derived by hand for all three — all match the
  equations and stability claims in the chapter exactly.
- AR(1) stationary variance formula $\text{Var}(y)=\sigma_\epsilon^2/(1-\alpha^2)$
  — standard result, re-derived, correct.
- All 7 `\citet`/`\citep` keys (`bury2021`, `bury2023discretetime`,
  `dakos2012`, `kuznetsov2004`, `scheffer2009`, `strogatz2015`,
  `uhlenbeck1930`) confirmed present in `references.bib`.
- All 11 `\ref{}` targets in the chapter resolve against the full label set
  across all `chapters/chapter_*.tex` files.
- Bury 2023 quote about the closest prior work not using an engineered
  variance-growth channel — not re-fetched this pass (already independently
  confirmed in the tenth pass against `/tmp/claude-216236/bury2023_discrete.txt`
  directly); no new claim added here that changes this.

**Flagged, not changed (a scope question, not a fact error):** the
Fold/Transcritical/Hopf subsections each still carry a one-line real-world
illustrative example (shallow lake, species colonisation, predator-prey
cycle) with no citation on two of the three. This log's own "Fifth pass"
explicitly cut a predator-prey Hopf example and other illustrative
material as against this chapter's own stated rule ("no illustrative
external-domain examples as filler" — `WRITING_RULES.md` §4, which still
quotes this cut as precedent). The three examples now in the chapter are
shorter one-liners than what was cut, and are not factually wrong, but
their presence is not accounted for in this log after the fifth pass cut
them — flagging this discrepancy rather than silently removing content
that may have been deliberately restored later in the session for other
reasons (e.g. readability) without being logged.

Post-edit integrity check: brace balance 114/114 (unchanged from before
this pass — the only fix was a same-length reword). All `\ref{}` in the
chapter resolve (11 targets, all found in the cross-chapter label set).
All 7 citation keys confirmed in `references.bib`.
