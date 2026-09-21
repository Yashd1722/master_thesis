# Chapter 4 ("Related Work") — Verification Log, Full Rewrite

## 🔴 Resolved: the fabricated/unverifiable "Ma et al. 2025" citation

The previous version had an explicit `%% UNRESOLVED` flag on a claim about
"Ma et al. 2025" describing Multi-Head CNN and CNN-LSTM/SVM hybrids for
bifurcation diagnosis. The earlier verification pass could not find this
paper and suspected it was fabricated by an earlier AI-assisted drafting
pass. I did not find it either. **The claim has been removed, not replaced
with a paraphrase of something unverifiable.**

In its place, I added a real, checked citation to a paper I have direct,
full-text knowledge of from earlier this session: \citet{babazadeh2025}
("Deep Learning for Bifurcation Detection: Extending Early Warning Signals
to Dynamical Systems with Coloured Noise," Research Square preprint, DOI
10.21203/rs.3.rs-5975924/v1). I fetched and read this paper's full text
directly this session (not from memory or a secondhand description) and can
confirm what it actually contains:
- A CNN-LSTM trained on synthetic fold/transcritical/Hopf/null data, built
  directly from normal forms (a narrower approach than Bury et al.'s wider
  random-system library).
- Tested on real anoxia data from the *same* Mediterranean cores
  (\citet{hennekam2020}) used in this thesis.
- Reports the classifier's prediction on the real anoxia data favours a
  Hopf-type bifurcation, in tension with the fold-type onset expected on
  domain grounds.
- Also tests robustness to coloured (autocorrelated) noise vs. white noise
  in training, finding broadly similar accuracy either way.

This is a stronger, more directly relevant citation than the removed one —
it is the closest published paper to this thesis's own method and dataset
I am aware of, and every claim attributed to it above I can point to a
specific passage of the paper for. Marked in the chapter text as **not
independently re-derived or re-confirmed** by this thesis; it is presented
as a related finding, not something this thesis proves.

I flagged this paper as **preprint, not peer-reviewed** in the `.bib` entry
(`howpublished = {Research Square preprint}`, explicit note) since Research
Square is a preprint server, not a journal — this should be represented
honestly rather than cited as if it were a peer-reviewed publication.

## 🔴 Resolved: overclaimed "solved" padding contribution

The previous version's §4.7–4.8 stated that edge-constant and reflection
padding were "rigorously identified and resolved" as "a major, novel
contribution." I checked this against the actual pipeline before leaving it
in:
```
config.yaml:  pad_mode: "zero"
```
grepped across every place `pad_mode` is read (`src/data_common.py`,
`testing/evaluate.py`, `training/train.py`) — it is `"zero"` everywhere,
with no evidence any result in this thesis was produced with `"edge"` or
`"reflect"` padding, even though both are implemented as options in
`src/data_common.py: left_pad_to()`.

**This means the previous claim was false as a description of what this
thesis actually did.** The code exists; it was never used to produce a
reported result. Fixed to state this precisely: edge/reflection padding are
implemented and available, comparing them against zero-padding on the real
cores is a natural next step given the existing code, but it is future work,
not a completed contribution. This also removes the associated "severe
spectral leakage" / "catastrophic collapse of AUC" claims about zero-padding,
which (as in Chapter 3) I could not find or reproduce evidence of being
directly tested — same fix as Chapter 3's equivalent claim.

## Repetition removed

The "$L=1500$ is a novel extension" claim appeared 3 times (§4.2, §4.5,
§4.8) and the padding-artifacts claim appeared 2–3 times (§4.2, §4.7, §4.8)
in the previous version. Both are now stated once each, with the accurate
number (median +3.6 AUC points, not "~10%," computed fresh this session from
`results/summary/zenodo.csv` — same number and source as used in Chapter 5,
not re-derived independently here since it's the same underlying data).

## Citations added

| Claim | Citation added | Why |
|---|---|---|
| Hennekam et al. already applied classical EWS to the same real cores | `\citet{hennekam2020}` | Previously this chapter's "Applications in Paleoclimatology" subsection described this generically with no citation, even though Chapter 5's verification log (previous pass) already identified this exact paper and flagged that Chapter 4 should probably cite it too. Done now. |
| The TSC bake-off / model-family framing | `\citet{bagnall2017bakeoff}` | Your own README names Bagnall et al. 2017 as one of three papers this thesis reproduces/extends, but no chapter previously cited it. Added at the point the TSC model families (ROCKET, dictionary, interval methods) are introduced, since that survey paper is the standard reference for exactly this landscape. |

## Claims softened because they could not be verified

- "their supplementary materials and subsequent community replication
  studies indicate the neural network still heavily struggled to cleanly
  separate [Fold and Transcritical]" — I do not have access to Bury et al.'s
  supplementary materials with a specific fold-vs-transcritical confusion
  number, and "subsequent community replication studies" named no specific
  study. Removed the claim rather than repeat it unverified. Chapter now
  states plainly that this thesis did not verify how well Bury et al.'s own
  classifier separates these two classes, and does not claim a comparison
  point that isn't backed by something checkable.
- "the first massive empirical benchmarking of algorithms like MiniRocket...
  on tipping point data" — softened to "the first... to my knowledge, though
  I have not exhaustively surveyed the field to confirm no one else has done
  this specific comparison." This is a claim about the absence of prior work,
  which requires a literature search to actually support; I did not do an
  exhaustive one, so the chapter no longer asserts it as fact.

## Second pass: real numbers from Bury 2021 and Bagnall 2017, fetched directly

Per your request for deeper, comparison-focused engagement with prior work
rather than generic paraphrase, fetched and read the actual results
sections of two papers rather than describe them abstractly:

- **Bury et al. (2021)**, via the PMC free-access mirror: their reported
  F1 scores are 84.2\% ($L=500$, ensemble of 10 classifiers) and 88.2\%
  ($L=1500$) — added as the first real precedent, from the source this
  thesis builds on, that longer sequences help. Also added: their DL
  classifier beat classical lag-1 AC/variance in 6 of 8 ROC comparisons
  tested, and their empirical anoxia test case was 26 time series across 8
  anoxic episodes from the same Hennekam et al. cores this thesis uses
  (not previously stated precisely — the chapter now makes clear this
  thesis's own evaluation uses the full 3-core, corrected-onset-age set,
  not directly comparable event-for-event to their subset without more
  information I don't have).
- **Bagnall et al. (2017)**, via web search of the paper's own description:
  18 classifiers plus 2 baselines, 85 UCR datasets, only 9 algorithms
  significantly beat the baselines. Added as real scope context for what
  "the bake-off" actually benchmarked — and stated explicitly that this is
  a different task (general sequence classification, not bifurcation-type
  classification), so it's not a like-for-like performance comparison, only
  a scope and practice comparison.
- **New comparison table** (Table 4.1): Bury et al. 2021 vs.\ Babazadeh et
  al. 2025 vs.\ this thesis, on architecture(s), training data, sequence
  lengths, noise robustness, real test data, and cross-model type-diagnosis
  reliability. Includes one explicit limitation of this thesis stated
  plainly: colored-noise robustness (which Babazadeh's paper does test) was
  not tested here.

## Carried forward, not independently re-checked this pass

- ROCKET/MiniRocket/MultiRocket, WEASEL2 (SFA, not SAX — already corrected in
  the first pass), and TSF descriptions and citations — these were already
  checked once and the sentences describing them were only lightly edited
  for length here, not substantively rewritten.
- Dakos et al. (2012) and Scheffer et al. (2009) citations and journal
  corrections — unchanged from the first pass.

---

# Rewrite pass (2026-09-10): subsection-by-subsection expansion + double verification

Per user instruction: expand each subsection to Chapter-3 depth, in plain
English, no repetition, no "this is a review paper" framing; verify every
claim twice against a paper line or a code line before it goes in; the user
approves each subsection before it is written to `chapter_4.tex`. Rules
file: `chapters/WRITING_RULES.md`.

Primary sources fetched and read in full this pass:
- Scheffer et al. 2009 — free PDF mirror `pdodds.w3.uvm.edu/.../scheffer2009a.pdf`
- Dakos et al. 2012 — PLoS ONE open access
- Hennekam et al. 2020 — WUR open copy `edepot.wur.nl/535071`
- Bury et al. 2021, Babazadeh et al. 2025 — full-text PDFs cached locally
- Dablander & Bury 2022 — PMC9477405
- Bagnall et al. 2017 — arXiv 1602.01711 abstract

## §4.1 — approved and written 2026-09-10

Old §4.1: ~230 words, 2 subsections. New §4.1: ~900 words, 4 subsections
(§4.1.1 Scheffer, §4.1.2 Dakos + `\label{sec:classical_method}`,
§4.1.3 Hennekam + `\label{sec:hennekam_ews}`, §4.1.4 Limitations +
`\label{sec:classical_limitations}`). No new `.bib` entries.

### Verified claims (claim → exact source)

| Claim | Source line | 
|---|---|
| Scheffer 2009: generic principle across fields | *"work in different fields of science is now suggesting the existence of generic early warning signals"* (abstract) |
| Critical slowing down = slower recovery near a bifurcation | Scheffer 2009 §Theory *"Critical slowing down and its symptoms"*; Box 3 |
| Same fingerprint regardless of system | Scheffer 2009 *"generic properties, regardless of differences in the details of each system"* |
| Examples: epileptic seizure, asthma attack, market crash, end of ice age, shallow lake turning murky | Scheffer 2009 *"asthma attacks or epileptic seizures"*, *"systemic market crashes"*, *"glacial cycles"* / *"eight examples of abrupt climate change"*, *"models of lake eutrophication"* |
| Variance ↑ and lag-1 autocorrelation ↑ near the tipping point | Scheffer 2009 Box 3 *"an increase in the autocorrelation and variance of the fluctuations"* |
| Mechanism: exponential recovery at speed λ, AR(1) form | Scheffer 2009 Box 3 (AR(1) equations shown) |
| Caveat: rising perturbation amplitude can mimic the signal | Scheffer 2009 *"also if the system is driven closer to the basin boundary by an increasing amplitude of perturbation"* |
| Dakos 2012 = R package `earlywarnings` | Dakos 2012 *"http://earlywarnings.r-forge.r-project.org/"* |
| Detrend: Gaussian default; linear + first-differencing alternatives | Dakos 2012 *"Gaussian smoothing (autocorrelation, variance, skewness)"*, *"linear detrending"*, *"first-differencing"* (LOESS NOT confirmed by fetch — removed from draft) |
| Window = half the series | Dakos 2012 *"window size = half the size of the datasets … 50% of series length"* |
| Kendall τ compared to a null distribution | Dakos 2012 *"observed Kendall τ is compared against this null distribution"* |
| Surrogates = AR(1)/ARMA fit to residuals, 1,000 sims | Dakos 2012 *"generating data from the simplest fitted linear first-order autoregressive model … generated 1,000 simulated datasets"* |
| Thesis uses same surrogate test in Bury's form | repo `testing/evaluate.py`: `AR1_FIT_FRACTION = 0.20`, `N_SURROGATES = 10`, `_fit_ar1_neutral`, `_generate_ar1_surrogates` |
| Hennekam cores + depths: 64PE406E1 (1,760 m), MS66 (1,630 m), MS21 (1,022 m) | Hennekam 2020 *"MS66 (1,630 m …), 64PE406E1 (1,760 m …), and MS21 (1,022 m)"* |
| Hennekam method: Gaussian detrend, half-record window, 1,000 surrogates, Kendall τ | Hennekam 2020 *"detrended by subtracting a Gaussian kernel smoothing function"*, *"rolling window size was set at half the size of the records"*, *"tested using 1,000 surrogate time series"*, *"Kendall's τ values"* |
| Deep cores: variance rose before every event, autocorrelation before most | Hennekam 2020 *"Increasing variance is observed prior to all analyzed anoxic events in the relatively deep cores … rising temporal autocorrelation occurs prior to most"* |
| Significance: variance p < 10⁻¹³, autocorrelation p < 0.003 | Hennekam 2020 *"variance (Fisher's combined probability test p < 10⁻¹³) and autocorrelation (p < 0.003)"* |
| Shallow core MS21: signals largely absent | Hennekam 2020 *"the absence of rising autocorrelation and variance in the record of shallow core MS21 (~1,000 m)"* |
| Uranium: same pattern; variance the more consistent indicator | Hennekam 2020 *"U are in line with results from Mo"*, *"increasing variance is the more consistent"* |
| Bury 2021 used these cores, restricted to Mo + U | Bury 2021 p.735 *"three cores that … span eight anoxic events. Variables include molybdenum (Mo) and uranium (U), proxies for anoxic"* |
| Thesis makes the same Mo + U restriction | repo `src/rolling_window.py` `ELEMENTS = ["Mo", "U"]` |
| Corrected PANGAEA pre-transition segments ~100–650 points | repo: 7 `*_forced.csv` files, lengths 103, 153, 167, 184, 218, 311, 648 |
| SE of lag-1 autocorrelation from n points ≈ 1/√n; 50 pts → 0.14 | standard sampling result; matches Chapter 3 verification log (*"≈1/√(window−1)"*, *"≈0.2–0.4 rise"*) |
| Dablander 2022: classifier learned the detrending filter, misclassified under a different one | Dablander 2022 (PMC) *"the method may have learned features specific to a Lowess filter rather than (only) generic features of a system approaching a bifurcation"* |
| Fold vs Transcritical indistinguishable on variance/AC1 | cross-ref `chapter_3.tex` §`sec:fold_transcritical_challenge` |

### Corrections made vs. the pre-rewrite §4.1

- Removed the vague "established … in the late 2000s and early 2010s" and
  the genre framing.
- "LOESS" dropped from the detrending alternatives (fetch confirmed only
  Gaussian / linear / first-diff).
- Hennekam paragraph was previously one sentence ("preceding several of the
  events"); now carries the real depth-dependent result with the p-values.
- Old text attributed Hennekam's method to "Dakos et al." generally;
  Hennekam actually cites Dakos et al. 2008, so the new text points to the
  in-thesis §`sec:classical_method` instead of a specific Dakos citation
  the thesis does not carry.

### Plagiarism check

Re-read §4.1 against all four source texts side by side. No sentence
tracks a source's wording or structure. The Dakos four-step list is a
standard procedure written in original wording and attributed. Clean.

## §4.2 — approved and written 2026-09-11

Restructured from 3 subsections (Core Idea / Their Results / Open
Questions) to 4 (Core Idea / The Architecture and What It Scored / The
Three Real-World Tests / Where This Thesis Departs), splitting the old
"Their Results" paragraph, which crammed the synthetic scores and the
three empirical tests together. New label `\label{sec:bury2021}` on the
section head. Source: `bury2021.txt` (full text, read this session).

### Verified claims (claim → exact source)

| Claim | Source |
|---|---|
| Training systems = random 2-D systems, polynomials to 3rd order, $a_i,b_i\sim N(0,1)$, half zeroed, cubic terms forced negative | Bury p.527–533 *"randomly generated, two-dimensional dynamical systems"*, *"all polynomials in x and y up to third order"*, *"drawing each aᵢ and bᵢ from a normal distribution with zero mean and unit variance … half … set to zero … cubic terms are set to the negative of their absolute value"* |
| Simulate → check equilibrium → AUTO-07P finds fold/Hopf/transcritical on the equilibrium branch | Bury p.558–564 *"test for convergence to an equilibrium point. Convergence is required … to search for bifurcations"*; *"we use AUTO-07P (43) to identify bifurcations along the equi[librium]"*; p.549 *"codimension-one bifurcations … fold, Hopf, and transcritical"* |
| Generate until enough of each type; pre-bifurcation portion = training example | Bury p.546–548 *"continue to generate models until a desired number of each type … has been found"*; p.280 *"prebifurcation portion of the simulation time series"* |
| 500,000 × 500 and 200,000 × 1,500 | Bury p.545–547 *"500,000 time series of length 500 … 200,000 time series of length 1,500"* |
| Lowess span 0.2 detrend; normalise by mean absolute value | Bury p.604–608 *"detrended using Lowess smoothing with a span of 0.2"*, *"dividing each … data point by the average absolute value of the residuals"* |
| 500-classifier for short series, 1,500-classifier for long | Bury p.549–551 *"The 500-classifier was used on shorter time series, while the 1,500-classifier was used on the longer"* |
| Four classes: fold, transcritical, Hopf, null/neutral | Bury p.278–279, p.322 *"(fold, transcritical, Hopf, and neutral)"* |
| CNN reads subsequences → LSTM interprets in order, loops on itself | Bury p.213–217 *"The CNN layer reads in subsequences … The LSTM layer then reads in the output of the CNN and interprets those features. The LSTM layer loops back on itself"* |
| Also tried resnet / conv / recurrent; CNN-LSTM highest | Bury p.594–597 *"experimented with a residual network, functional convolutional network, and recurrent neural network but found … CNN-LSTM … highest precision and recall"* (draft says "convolutional network", dropping Bury's unusual "functional" qualifier) |
| F1 84.2/84.4/84.2 (L=500); 88.2/88.3/88.3 (L=1500), ensemble of ten | Bury p.612–616 *"ensemble of ten 500-classifier models were 84.2%, 84.4%, and 84.2% … ten 1,500-classifier models were 88.2%, 88.3%, and 88.3%"* |
| 8 comparisons, 6 systems (3 ecological + 3 empirical), DL wins 6/8 on AUC | Bury p.289–290, p.374–380 *"eight comparisons across all six study systems … outperforms lag-1 AC and variance in six"* |
| Ecological models: type named correctly in all three cases | Bury p.328–330 *"correctly predicts the type of bifurcation in each of the three cases"* |
| Classifier splits probability early, commits later | Bury p.335–339 *"initially assigns similar probabilities to all three … after a specific time point … becomes highly confident"* |
| Paleoclimate = 7 of 8 shifts from Dakos et al. | Bury p.776–778 *"seven out of the eight climate transitions … previously analyzed for EWS by Dakos et al."* |
| Thermoacoustic Rijke tube: 19 forced, 10 steady-state | Bury p.748–765 |
| Anoxia: 26 series, 8 events, 3 Hennekam cores | Bury p.734–736 |
| Each length applied to whichever real series matched (not "L=1500 for empirical tests") | Bury p.550–551, p.555–560 *"For the ROC curves in Fig. 2, we used the 500-classifier for the paleoclimate data and the ecological models, and used the 1,500-classifier for the thermoacoustic data, anoxia data, and disease model"* |
| Median +3.6 AUC L=1500 vs L=500 across 19 models; MultiRocket negative; ≥+6 for several | recomputed from `results/summary/zenodo.csv` (19 models w/ both lengths; deltas incl. rdst +9.3, tsbf +7.7, mrsqm +6.7, rocket +6.0, catch22 +6.2, lps +9.0; multirocket −6.4) |
| Variance growth ratio channel is a direct input for Fold/Transcritical | repo `src/ews_augmenter.py` (5th channel), cross-ref `sec:fold_transcritical_challenge` |

### Corrections made vs. the pre-rewrite §4.2

- "the $L=1500$ classifier being the one they used for their empirical
  tests" (a bad edit from earlier this session) → corrected to "each
  applied to whichever real series matched its length", per Bury p.555–560.
- "as much as +6.2 for some" → "above +6 points for several models" (true
  max is ≈+9, so +6.2 understated the spread).
- Added the actual training-data construction (Bury Eqs. 4–5, AUTO-07P,
  Lowess span 0.2, dataset sizes) — previously only summarised as "a
  wider family of random systems," now given in full with the source
  lines above.
- Added the CNN-LSTM mechanism, the alternative architectures tried, and
  the "splits probability early, commits late" behaviour — none of this
  was in the chapter before.

### Plagiarism check

Re-read §4.2 against `bury2021.txt` side by side. No sentence tracks
Bury's wording or structure. The one display equation reproduces Bury's
Eqs. 4–5 in standard notation, attributed to the paper. Clean.

## §4.3 — approved and written 2026-09-11

Source: full PDF of \citet{bagnall2017bakeoff} fetched and saved this
session (`bagnall2017.txt`), plus web-verified facts for MiniRocket and
MultiRocket (2 independent sources each, converging), plus the
already-cached Babazadeh PDF.

### Verified claims (claim → exact source)

| Claim | Source |
|---|---|
| 18 classifiers + 2 baselines (1-NN DTW, RotF), 85 datasets, only 9 sig. better than both | `bagnall2017.txt` p.45 *"only 9 of these algorithms are signiﬁcantly more"* — confirmed twice: arXiv abstract fetch (earlier pass) + full PDF (this pass) |
| 6 families: distance, differential-distance, dictionary, shapelet, interval, ensemble | `bagnall2017.txt` section headers 2.1–2.6 |
| COTE = overall winner, ensemble | `bagnall2017.txt` p.166 *"by far the best classiﬁer is COTE"* |
| **COTE ≈8% more accurate than either baseline — verified 3× independently within the source, per user request** | (1) p.167 *"on average over 8% more accurate than either benchmark"*; (2) Table 4 p.1424 *"COTE 96.47% 8.12% COTE 84.71% 8.14%"*; (3) Conclusion p.1489–90 *"on average 8% more accurate than DTW"* |
| ST (shapelet) = 2nd overall | `bagnall2017.txt` *"ST … is the second most accurate classiﬁer overall"* |
| BOSS (dictionary) = 3rd overall | `bagnall2017.txt` *"the BOSS ensemble is one the most accurate classiﬁers we tested (ranked 3rd)"* |
| TSF, TSBF, LPS (interval) all sig. better than baselines, no sig. difference, TSF favoured for simplicity | `bagnall2017.txt` *"TSF, TSBF and LPS, are all signiﬁcantly better than both the benchmarks … no signiﬁcant diﬀerence between them … would favour TSF for its simplicity"* |
| No kernel-based family in the 2017 bake-off (correction — see below) | confirmed absent from section list 2.1–2.6; ROCKET published 2020 |
| ROCKET: 10,000 kernels, random length/weight/bias/dilation, max+PPV, 20,000 features, no gradient descent | established/well-documented (Dempster 2020); not re-fetched full text — unchanged from the description already accepted earlier in this session |
| MiniRocket: 84 fixed length-9 kernels, weights 3×(+2)/6×(−1), PPV only, biases from data quantiles, "almost" deterministic, >10× faster | 2 independent sources converged: GitHub `angus924/minirocket` + aeon/sktime docs (via web search) |
| MultiRocket: +MPV, +MIPV, +LSPV pooling beyond PPV; raw + first-difference; 50k features from 10k kernels | aeon docs + arXiv 2102.00457 abstract, both converge (via web search) |
| TSF: √m random intervals/tree, mean/std/slope, 3√m features, majority vote | `bagnall2017.txt` p.815–840 *"each member of the ensemble is given √m intervals"*, *"mean, standard deviation and slope"*, *"3√m features"* |
| WEASEL 2.0: SFA words, random window length + dilation per word | `references.bib` title itself: *"WEASEL 2.0 — A Random Dilated Dictionary Transform"* |
| Babazadeh: 0.83–0.85 accuracy across redness range; AUC 0.9–1 | extracted earlier this session from cached full-text PDF: *"minimal sensitivity to redness (ranging from 0.83 to 0.85)"*, *"AUC scores (ranging from 0.9 to 1)"* |

### Correction made vs. the pre-rewrite §4.3

- Old text: *"the specific classifier families (kernel-based, dictionary-based,
  interval-based) that came out on top there [Bagnall 2017]"* — **false
  attribution**, since kernel-based methods (ROCKET, 2020) postdate the
  2017 bake-off and were never tested in it. New text credits
  dictionary/shapelet/interval as what the bake-off actually found on top,
  and separately notes kernel-based methods as a later development this
  thesis draws on.

### Plagiarism check

Re-read §4.3 against `bagnall2017.txt`, the Babazadeh PDF, and the
MiniRocket/MultiRocket search sources. No sentence tracks source wording.
Table-4 numbers are data, cited. Clean.

## §4.4 — approved and written 2026-09-11

Rewritten in plain, short sentences per user request (dropped
"C⁰/C¹ continuity" language, split dense clauses). Carries forward the
Dablander misattribution fix already made earlier this session, now
re-verified and folded into the full-subsection rewrite.

### Verified claims (claim → exact source)

| Claim | Source |
|---|---|
| 500 or 1,500-point fixed input | repo: `ts_500`/`ts_1500` dataset naming throughout |
| 7 forced segments, ~100–650 points, all shorter than the 500-point model | repo: the 7 `*_forced.csv` files, lengths 103, 153, 167, 184, 218, 311, 648 — all < 500 |
| Bury pads both ends (variant 1) or left only (variant 2) before normalising | Bury GitHub README: *"10 networks use time series that are padded on both the left and the right with zeros (model_type=1). 10 networks use time series that are padded only on the left with zeros (model_type=2)"* |
| `pad_mode: "zero"` used for every result | `config.yaml` line 37 |
| Repeat-first-value and reflect are the two alternative padding modes implemented | `src/data_common.py` line 45 `left_pad_to()`, zero/edge/reflect branches confirmed |
| Dablander 2022 = detrending-filter sensitivity, not padding | Dablander 2022 (PMC9477405), fetched earlier this session — *"the method may have learned features specific to a Lowess filter rather than (only) generic features of a system approaching a bifurcation"* |
| Training-time padding amount was capped, randomised, and tightened to match real segments | `chapter_5.tex` §`sec:preprocessing`: *"both of which were tightened during this project specifically so that training-time padding better matches the amount of padding real short segments actually need at test time"* |

### Note on scope

The "sharp jump" / zero-padding-as-discontinuity paragraph is stated as a
general caution, explicitly flagged as not a measured effect — same
treatment Chapter 3's verification log used for an equivalent unverifiable
claim (softened from "as observed" to a general caution). No new claim
needing a source was added here.

### Plagiarism check

Re-read against the Bury README, Dablander PMC summary, and
`src/data_common.py`. No sentence tracks source wording; code behaviour is
described, not quoted as prose. Clean.

## §4.5 — approved and written 2026-09-12

Final section of the chapter: section intro, Table 4.1, and the
contributions list. Per user request, the contributions list ends at 2
items (the "corrected onset ages" item was cut — that contribution is
already covered in depth in Chapter 5 §`sec:label_correction`, and is
cross-referenced from §4.1.4 and §4.4 elsewhere in this chapter, so cutting
the third bullet here loses no unique content).

### Verified claims (claim → exact source)

| Claim | Source |
|---|---|
| **Fix 1**: intro previously said Bury "demonstrated on one architecture and one real dataset" | Wrong — §4.2 (verified earlier this pass) established Bury tested **three** real-data settings. New text: "each of its three real-world tests was small" |
| **Fix 2**: Table 4.1 said Babazadeh's sequence length was "not stated in the paper" | Wrong — `babazadeh.txt` line 259 *"σ was set to 0.1 times the length of the time series (= 100)"* → training series length is 100 points. New cell: "Training series length 100" |
| **Fix 3**: Table 4.1 said Bury's real test data was "Ice cores…" | Inconsistent with the §4.2 fix (Bury's own term is "paleoclimate transitions," not specifically ice cores). New cell: "Paleoclimate transitions…" |
| 7 DL + 22 classical TSC models (thesis roster) | `config.yaml` `models.dl` — 7 entries (cnn_lstm, lstm, inceptiontime, patchtst, resnet, tcn, rnn_fcn); `models.tsc` — 22 entries, counted directly |
| Training data = Bury's released Zenodo library, not resimulated | `REPORT_NOTES.md` line 12: *"Source: Bury et al. (2021), zenodo.org/record/5527154"* |
| Babazadeh tested thermoacoustic data too, same 19/10 split as Bury | `babazadeh.txt` line 344 *"19 forced trajectories with varying voltage ramp rates and 10 steady-state trajectories"* — confirms that table cell was already correct |
| Variance growth ratio channel description | consistent with the fix already verified for Chapter 3 and §4.5 item 2 earlier this session |

### Contributions list — user edits

- Item 1's hedge rewritten: dropped the first-person "I have not exhaustively surveyed the field" aside per user instruction ("you do not have to write something like this"); kept the substance as a standard scoped academic claim: "As far as this thesis is aware, no earlier study has run a comparison at this scale on this specific task."
- Item 3 ("A corrected, verified set of real transition dates") removed entirely per user instruction. Not lost: the same finding is covered in Chapter 5 §`sec:label_correction` and referenced from §4.1.4 and §4.4.

### Plagiarism check

Re-read against `babazadeh.txt`, `REPORT_NOTES.md`, and `config.yaml`. No
sentence tracks source wording. Clean.

---

# Chapter 4 — rewrite pass complete (2026-09-12)

All five sections (§4.1–§4.5) redrafted subsection-by-subsection, each
verified against a primary source or repo file, each approved by the user
before being written. Summary of corrections made across the whole
chapter during this pass:

1. §4.1 Hennekam paragraph — from one vague sentence to the actual
   depth-dependent result with p-values, sourced to the paper directly.
2. §4.2 — "Bury used L=1500 for empirical tests" → corrected to "each
   length applied to whichever real series matched it"; "+6.2 max AUC
   gain" → "+6 or more for several" (true max ≈+9).
3. §4.3 — removed false attribution of "kernel-based" as a family that
   "came out on top" in the 2017 Bagnall bake-off (kernel methods didn't
   exist yet); added verified mechanism detail for ROCKET/MiniRocket/
   MultiRocket and TSF.
4. §4.4 — Dablander 2022 misattribution (detrending, not padding) fixed;
   added exact segment-length numbers.
5. §4.5 — "one real dataset" → "three real-world tests, each small";
   Babazadeh sequence length corrected from "not stated" to "100"; "Ice
   cores" → "Paleoclimate transitions" for internal consistency; dropped
   the third contribution item and the first-person hedge per user
   request.

All cross-references and labels checked and resolve after every write.
No new `.bib` entries were needed. Rules followed throughout:
`chapters/WRITING_RULES.md`.

## Post-review fix (2026-09-13): "every model in the roster" gets the variance growth ratio — false

Surfaced while working through Chapter 6. §4.5 item 2 said the variance
growth ratio channel "gives every model in the roster" this input.
Traced directly against the current pipeline: `training/train.py` —
`train_tsc()` (the 22 classical models) calls `augment_ews_channels()`;
`train_dl_variant()` (the 7 deep-learning models) never does. Confirmed
identically in `testing/evaluate.py` (`_prepare_tsc_input()` is TSC-only).
`ARCHITECTURE_COMPARISON.md` already documents this: DL models get only
the single-channel raw residual.

**Fixed:** scoped the claim to "the classical (non-deep-learning) models
in the roster," with an explicit sentence that the deep-learning models
receive only the raw residual. Same fix made in Chapter 3
(§`sec:fold_transcritical_challenge` and the chapter's closing paragraph)
and Chapter 5 (§`sec:feature_engineering`), logged in their own
verification files.

## Post-review fix (2026-09-12): two overclaimed absence statements

User flagged, correctly: two sentences asserted that a source "says
nothing about X" / "reports Y, not Z," when in fact only a partial view of
that source had been checked (a summarised WebFetch answer for Dablander,
and the main text only — not the SI Appendix — for Bury). Asserting what a
paper does *not* contain requires having read the whole thing; neither
condition was met.

**Fixed:**
- §4.4: *"Their paper says nothing about padding"* → *"This thesis did not
  find a discussion of padding in their paper"* (a claim about what was
  looked for and found, not about the totality of Dablander & Bury 2022).
- §4.2.4: *"their paper reports overall accuracy, not a per-class
  confusion rate"* → *"Their main text reports only overall accuracy
  figures; this thesis did not check their supplementary materials for a
  per-class breakdown"* (Bury 2021 references an SI Appendix multiple
  times — confirmed present at p.344, p.425, p.575 of the extracted main
  text — which was never fetched or read this session).

**New standing rule added**: `WRITING_RULES.md` §6, "Never assert that a
source 'doesn't' do something" — applies to every remaining chapter, not
just this one. The source-coverage table in that file (§9) now marks
which sources were fully read vs. only summary-fetched, specifically so
this class of error is caught before writing rather than after.

## Fresh independent re-verification pass (this session, fork audit)

Full re-check of every factual claim in chapter_4.tex against primary sources, independent of earlier logged passes.

### Confirmed correct (no changes needed)
- Bury et al. F1/precision/recall: 84.2%/84.4%/84.2% (L=500), 88.2%/88.3%/88.3% (L=1500) — `bury2021.txt:613-615`, exact match.
- Bury training set sizes: 500,000 (L=500) / 200,000 (L=1500) — `bury2021.txt:545-548`, exact match.
- Bury equation construction: polynomials "up to third order", cubic terms "set to the negative of their absolute value", AUTO-07P — `bury2021.txt:527,532,564`, exact match.
- Bury real-world test counts: 19 forced/10 steady-state thermoacoustic, 26 time series/8 anoxic events, 7-of-8 paleoclimate transitions — `bury2021.txt:763,767,734,736,776`, exact match.
- Hennekam et al.: core depths (MS66 1,630m, 64PE406E1 1,760m, MS21 1,022m), Gaussian kernel detrending, window = half record length, 1,000 surrogates, p<10⁻¹³ (variance)/p<0.003 (AC), "variance rose before every event"/"AC rose before most" — `hennekam2020.txt:137-138,178-181,228,209-214`, exact match on every figure.
- Babazadeh et al.: Hopf-favoured prediction on anoxia data, 83–85% validation-accuracy band, AUC 0.9–1 — `babazadeh.txt:160,39-42`, exact match.
- Bagnall et al. bake-off: 85 datasets, 18+2 baseline classifiers, 9 significantly beating both benchmarks, COTE "over 8%" more accurate, ranking COTE(1st)/ST(2nd)/BOSS(3rd) — `bagnall2017.txt:42,38-46,167,1305-1376`, exact match.
- ROCKET/MiniRocket/MultiRocket mechanism descriptions: consistent with established facts used elsewhere in this thesis.

### Errors found and fixed
1. **Bury type-accuracy mischaracterization** (§4.2.2, "The Architecture and What It Scored"). Chapter claimed the classifier "named the type correctly in all three cases" on "the ecological models." The paper's actual statement (`bury2021.txt:398-403`) is that it predicted the correct type in *all eight comparisons but one* — the thermoacoustic system, where the Hopf-vs-fold favoured-frequency gap was narrow — not a claim scoped to "three ecological models." Fixed to state this precisely: "seven of the eight comparisons... the one exception was the thermoacoustic system, where the favoured-probability frequency for Hopf was only slightly higher than for fold" (directly reflecting the source wording, paraphrased).
2. **Ma et al. 2025 (ma2025sdml) missing entirely from Related Work.** This paper is cited elsewhere in the thesis (Chapter 6's LSTM architecture, Chapter 8's results comparisons) and is directly relevant — same three PANGAEA cores, an LSTM architecture this thesis's own `models/lstm.py` reproduces, and a shared lead author with Bury et al. 2021 — yet it was entirely absent from Chapter 4. Fixed by renaming §4.6.4 "A Closely Related Recent Study" → "Closely Related Recent Studies" (plural) and adding a full paragraph on Ma et al. 2025: their SDML (surrogate-data, not simulated-dynamics) training approach, per-core best-architecture F1 scores (SVM/MS66=1.0, LSTM/64PE=0.99, CNN/MS21=0.99, `ma2025.txt:408-412`), the explicit caveat that these F1 figures are a synthetic-validation metric not a real-data accuracy figure (`ma2025.txt:456-459`), and the LSTM-architecture match to this thesis's own model.

### Not independently re-derivable, flagged not fixed
- "TSF, TSBF, LPS... no significant difference between them" (§4.4): the critical-difference diagram in `bagnall2017.txt` places them close together (ranks 5.73/5.91/6.42, all in the 9-classifier "significantly better than both benchmarks" clique) but no explicit sentence stating "no significant difference among these three specifically" was found in the extracted text. Left as-is (plausible, not disconfirmed) rather than edited without a directly checkable source.

### Post-fix checks
Brace balance: 178 open / 178 close. All 13 `\citet`/`\citep` keys confirmed present in `references.bib`. All 15 cross-chapter `\ref` targets confirmed defined somewhere in `chapters/*.tex` (verified by grep across the full chapter set, not assumed).

## Full section-by-section re-verification pass, 2026-09-20 (same standard as chapter_8's audit)

Independent, from-scratch re-check of every section against primary sources and current repo state — numbers recomputed, not copied from the chapter's existing text or from the log entries above. Re-fetched Dakos 2012 (PLoS ONE page, presence claims only) and Dablander & Bury 2022 (PMC, presence + one absence-of-mention check) fresh this pass rather than relying on prior fetch results.

### Confirmed correct (spot-checked/recomputed fresh, no changes needed)
- Dakos et al. 2012 routine (detrend → slide window → measure → Kendall's τ), R package `earlywarnings`, surrogate-based significance test, half-record-length default window, bandwidth as a user choice, first-differencing listed as an alternative filter — all confirmed by fresh WebFetch of the PLoS ONE page; all are presence claims the summary directly supports (per WRITING_RULES §9, Dakos remains fetched-summary-only, so no absence claim rests on it — checked none exists in the chapter).
- Hennekam et al. 2020: core depths, Gaussian-kernel detrending, half-record window, 1,000 surrogates, "variance rose before every event / AC before most," combined p<10⁻¹³ (variance)/p<0.003 (AC), MS21 signal largely absent, variance the more consistent indicator — all re-confirmed line-by-line against `hennekam2020.txt`.
- Bury et al. 2021 F1/precision/recall (84.2/84.4/84.2 at L=500; 88.2/88.3/88.3 at L=1500) — re-verified fresh against `bury2021.txt:613-615` independently of the prior log entry above, exact match.
- Bury et al. 2021 real-world test counts (7/8 paleoclimate, 19 forced + 10 steady thermoacoustic, 26 series / 8 anoxic events) and Mo/U-only proxy restriction — re-confirmed against `bury2021.txt`.
- Bury et al. 2021 zero-padding description ("both ends or left only") — confirmed against `bury2021.txt:616-632`.
- Dablander & Bury 2022 ("learned features specific to a Lowess filter") — re-confirmed by fresh WebFetch of the PMC article; the article also contains no mention of padding anywhere in the main text (checked fresh this pass), so the chapter's bounded phrasing ("this thesis did not find a discussion of padding in their paper") remains correctly scoped and is not an overclaim.
- Bagnall et al. 2017 bake-off: 85 datasets, 18 classifiers + 2 baselines (1-NN DTW, Rotation Forest), 9 significantly beating both, COTE "over 8%" more accurate, six families in the exact order given (distance-based `sec2.1`/difference-based `sec2.2`/dictionary-based `sec2.3`/shapelet-based `sec2.4`/interval-based `sec2.5`/ensemble `sec2.6`), ST ranked 2nd overall, BOSS ranked 3rd — all re-confirmed against `bagnall2017.txt`.
- Time Series Forest mechanism (√m random intervals, mean/sd/slope, 3√m features, majority vote) — re-confirmed word-for-word against `bagnall2017.txt:815-843`.
- **TSF/TSBF/LPS "no significant difference between them," previously flagged as not independently re-derivable (see log entry above) — now resolved as confirmed.** `bagnall2017.txt:1414-1416` states this explicitly: "TSF, TSBF and LPS, are all significantly better than both the benchmarks... There is no significant difference between them." The earlier flag can be dropped.
- Babazadeh et al. 2025: Hopf-favoured prediction vs. fold-type expectation, 83-85% validation-accuracy band across redness values, AUC 0.9-1 on coloured test series for either training noise — re-confirmed against `babazadeh.txt`.
- Ma et al. 2025 (SDML): per-core best-architecture F1 (SVM/MS66=1.0, LSTM/64PE406E1=0.99, CNN/MS21=0.99), F1 reported on the surrogate validation set with the real ROC-curve test data explicitly excluded from training, LSTM layer sequence (dense→LSTM(128)→dropout(.5)→LSTM→dropout(.5)→dense(128,ReLU)→dense(out)) matching `models/lstm.py` directly — re-confirmed against `ma2025.txt:408-459` and the current `models/lstm.py` source.
- Model roster counts (7 deep-learning + 22 classical = 29) — re-confirmed against `chapters/chapter_6.tex`'s `tab:model_roster`.
- Padding implementation: `config.yaml` has `pad_mode: "zero"` (line 37); `src/data_common.py` implements `zero`/`edge`/`reflect` modes (lines 45-79, 156-164), with `edge` repeating the first value and `reflect` mirroring the signal — all three modes exist in code, only `zero` was used for reported results. Re-confirmed directly against current working-tree files.
- Variance growth ratio formula: `np.log1p(var_ch / init_var)` in `src/ews_augmenter.py:36` — matches "log-scaled ratio... not just the variance level," re-confirmed directly.
- AUC gain from L=500 to L=1500: recomputed fresh from `results/summary/zenodo.csv` (`macro_auc_ovr` column) independently of the chapter's own text. 19 models ran at both lengths (matches). Median gain = 3.58 points (chapter says "about 3.6," matches). MultiRocket is the only negative case, at -9.69 points (matches "negative for one... does worse at the longer length"). Three models (rocket +6.06, mrsqm +6.23, rdst +7.34) exceed +6 points (matches "above +6 points for several models").
- 7-sapropel segment-length range "about 100 to 650 points": recomputed by counting rows directly in the current working-tree `dataset/pangaea_923197/datasets/clean_dataset/*/*_forced.csv` files (7 files = 7 sapropels). Actual range: 103 (64PE406E1/S6) to 648 (MS21/S1) points — matches the chapter's "about 100 to 650" almost exactly.

### Errors found and fixed (this pass)
1. **Scheffer et al. 2009 example list overclaimed what the source documents for each case** (§4.1.1). The chapter listed five examples ("an epileptic seizure beginning, an asthma attack, a market crash, the end of an ice age, a clear shallow lake turning murky") and asserted the paper shows *both* variance and autocorrelation rising in each. Checking `scheffer2009.txt` directly: the asthma discussion (lines 541-546) is entirely about a self-organized spatial bronchoconstriction pattern, with no variance/autocorrelation claim attached; and the paper's one worked climate example with a rise shown in a figure (Fig. 4, `scheffer2009.txt:635-641`) is the *greenhouse-to-icehouse* transition 34 million years ago — a shift into colder conditions, not "the end of an ice age" (a warming transition, which the paper only asserts in passing with no figure). Fixed by replacing the example list with three cases the paper explicitly backs with data (epileptic-seizure variance, Fig. 5; financial-market variance-and-autocorrelation, line 574; greenhouse-icehouse autocorrelation, Fig. 4) and softening "two things happen... in each" to "a rise in variance, in autocorrelation, or in both," which is what the source actually supports per case.
2. **"Three ecological models" mischaracterized Bury et al. 2021's model-system test set** (§4.2.2). One of the three ("a system of five equations representing the coupled dynamics of infection transmission and vaccine opinion propagation," i.e. the SEIRx behaviour-disease model, `bury2021.txt:294-296`) is epidemiological/social, not ecological — the paper's own abstract groups "ecology... and epidemiology" as separate domains. Fixed to "three model systems (a harvesting model, a predator-prey model, and a coupled disease-behaviour model)."
3. **"Their main text reports only overall accuracy figures" was factually wrong about which metric Bury et al. 2021 report** (§4.2.4, "Where This Thesis Departs"). Checked `bury2021.txt` for the word "accuracy": zero occurrences anywhere in the main text. The paper reports F1 score, precision, and recall (`bury2021.txt:612-615`), never "accuracy." Fixed "overall accuracy figures" → "overall F1, precision, and recall figures." (The bounded-absence-claim phrasing around it, established as correct in the earlier 2026-09-12 pass, was left unchanged — only the wrong metric name was fixed.)
4. **"Every one shorter than even the smaller model size" was false for one of the seven segments** (§4.5.1, padding section). Counted rows directly in the current working-tree `*_forced.csv` files: six of the seven pre-transition segments are under 500 points, but `MS21_S1_forced.csv` currently has 648 rows — longer than the smaller model's L=500 (though still shorter than L=1500). This file was recently regenerated in the working tree (`git diff` shows 365→648 rows, uncommitted) — a real, current discrepancy, not a stale reading. Fixed to "all but one shorter than even the smaller model size, and every one shorter than the larger model size of 1,500," which is accurate against the current data and preserves the paragraph's point (padding is still needed at both lengths for six of seven segments, and at L=1500 for all seven).

### Scope of this pass
Every section of chapter_4.tex checked: the Foundation (Scheffer, Dakos, Hennekam, classical limitations), the Bury et al. 2021 section (core idea, architecture/scores, three real-world tests, where this thesis departs), Advancements in TSC (bake-off summary, recurrence rationale, ROCKET family, dictionary/interval methods, closely related studies), Left-Censored Data and Padding, and the Gap-in-the-Literature/Contributions section including the comparison table. Every `\citet`/`\citep` key confirmed present in `references.bib`; every `\ref` confirmed to resolve against the full label set across all six chapter files; brace balance confirmed even (179 open / 179 close, after the four edits above, each of which added one balanced `{,}` thousands-separator pair).

### Left as honestly uncertain (not fixed, not disconfirmed)
- The general mechanism descriptions of ROCKET, MiniRocket, MultiRocket, WEASEL 2.0, and Time Series Forest's role as later work are consistent with established facts used elsewhere in this thesis (chapter 6) and with domain knowledge, but dempster2020, dempster2021minirocket, tan2022multirocket, schafer2023weasel2, and deng2013tsf are not in this thesis's local-source table (`WRITING_RULES.md` §9) and their full texts were not fetched this pass. Left as-is; flagged here rather than independently re-verified against a primary source.
