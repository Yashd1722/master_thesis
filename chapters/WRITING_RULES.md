# Thesis Writing & Verification Rules

Single source of truth for how chapters 4–8 get expanded and checked.
Re-read this file at the start of every writing pass. Chapter 3 is already
complete and serves as the style/depth reference.

---

## 1. Workflow (per subsection, no exceptions)

1. Draft the expanded/rephrased subsection.
2. Verify **every** factual claim twice — once against the primary source
   (paper text or repo code), once re-checked independently.
3. Present the draft **and** a verification table (claim → exact source
   line) to the user.
4. Wait for explicit approval.
   - Approved → write it into the chapter `.tex` file, then append the
     verified claims to `chapter_N_verification.md`.
   - Not approved → rephrase and repeat from step 1.
5. Never write to a `.tex` file before approval.

## 2. Verification standard

- Every claim about **another paper's method, result, or weakness** must
  cite a specific line/passage of that paper. Record it as
  `paper_key p.N / "quoted phrase"` or line number in the local text dump.
- Every claim about **this thesis's own method** must cite a specific
  file + line in the repo (`src/…`, `testing/…`, `config.yaml`,
  `results/summary/…`).
- Numbers (F1, AUC, counts, ranges) are copied from the source, never
  recalled from memory. Re-derive from data files where the claim is a
  computed summary.
- If a claim cannot be tied to a checkable source this session, it is
  **flagged in red in the draft** (`\textcolor{red}{[FLAG: …]}`) and NOT
  asserted as fact. The user decides whether to chase the source or cut
  the claim.
- "Verify twice" = check the source, then re-open it and confirm the
  wording still supports the exact sentence written (not a looser version).

## 3. Plagiarism — zero tolerance

- No sentence may track the wording or sentence structure of a source.
  Paraphrase fully, in this thesis's own plain voice.
- Borrowed **facts** (numbers, counts, dataset descriptions) are fine when
  attributed with `\citet{}` / `\citep{}`; borrowed **phrasing** is not.
- Standard procedures (e.g. the Dakos 4-step routine, ROCKET's mechanism)
  may be described step by step, but in original wording and attributed.
- Final pass over all chapters at the end: read every paragraph against
  its cited sources; target is zero flagged overlap.

## 4. Style (matches Chapter 3, per user)

- Plain, simple English. Short sentences. One idea per sentence.
- Concrete before abstract. Give the mechanism, not just the label.
- No meta-commentary about a paper's genre — do **not** write "this is a
  review paper", "the paper does not report an experiment", etc. State
  what the paper says or found, directly.
- No repetition: each point is made once, in the place it belongs. Later
  mentions are a short pointer + `\ref{}`, not a restatement.
- No drama words ("devastating", "catastrophic", "crucially", "paradigm
  shift"). Removed from Chapter 3; keep them out.
- Bigger = more real detail (mechanism, exact numbers, honest scoping),
  not padding. Depth only on content the pipeline actually uses — no
  illustrative external-domain examples as filler (Chapter 3 pass 5 cut
  these: SIR, predator-prey, worked toy tables).

## 5. Accuracy of "our method" claims — describe the code as it is

- The variance-growth-ratio channel is `log(1 + Var_window(t) /
  Var_window(start))` — a log-scaled ratio of current windowed variance to
  baseline windowed variance (`src/ews_augmenter.py`). It is **not** a
  derivative / rate of change. Do not describe it as one. (Fixed in
  Chapters 3, 4, 5, and 6 — all confirmed correct as of the
  session-wide re-verification pass, 2026-09-14.)
- Only the **22 classical TSC models** use 5 input channels (raw residual,
  rolling variance, rolling lag-1 AC, rolling skewness, variance growth
  ratio; `checkpoints/*_ch_stats.npz` for TSC models all shape `(5,)`). The
  **7 deep-learning models** read the single raw residual channel only,
  shape `(batch, 1, L)` — the same shape Bury et al.'s own classifier
  reads. Never write "all models use 5 channels" — this was a real,
  repeated bug found and fixed in Chapters 3, 4, 5, 6, and 7 earlier this
  session; the split is by model family (`is_tsc_model()`), not universal.
- PANGAEA evaluation is restricted to elements **Mo and U**
  (`src/rolling_window.py` `ELEMENTS = ["Mo", "U"]`).
- Every reported result used `pad_mode: "zero"` (`config.yaml`).
  Edge-constant and reflection padding are implemented but unused —
  future work, not a contribution.
- No retraining: the 26 trained models stay as-is. Weak/degenerate models
  (`weasel2`, `ls`, `multirocket`, `minirocket`, `mrsqm`) are excluded
  from bifurcation-type claims, not retrained.

## 6. Never assert that a source "doesn't" do something

A claim that a paper "says nothing about X," "does not address X," or
"reports Y, not Z" is a claim about the *entire* paper — including
supplementary material you may not have read. Do not write it that way
unless you have verified the whole document (including any SI/appendix)
and can point to that verification.

Default phrasing instead: **"This thesis did not find X in \[source\]"**
or **"\[source\]'s main text reports Y; its supplementary material was not
checked."** This is a claim about what you looked for and found, not an
assertion about what does or doesn't exist in the source overall — it
stays true even if the SI turns out to contain something you didn't see.

Applies to every chapter, not just the one where it was first raised
(caught in Chapter 4 §4.2/§4.4 on 2026-09-12: "Their paper says nothing
about padding" and "their paper reports overall accuracy, not a per-class
confusion rate" — both fixed to the bounded phrasing above, since neither
source's supplementary material had actually been checked).

## 7. Cross-references & labels

- Every `\ref{}` must resolve. Check after each write.
- Add a `\label{}` wherever a later chapter needs to point back
  (e.g. `sec:classical_method`, `sec:hennekam_ews`).
- Use `\ref{ch:…}` / `\ref{sec:…}`, never hardcoded chapter/section
  numbers.

## 8. Chapter → label map (verified 2026-09-10)

| Chapter | file | `\label{ch:…}` |
|---|---|---|
| 3 Theoretical Background | `chapter_3.tex` | `ch:theoretical_background` |
| 4 Related Work | `chapter_4.tex` | `ch:related_work` |
| 5 Data | `chapter_5.tex` | `ch:data` |
| 6 Methodology | `chapter_6.tex` | `ch:methodology` |
| 7 Project Structure & Implementation | `chapter_7.tex` | `ch:implementation` |
| 8 Results | `chapter_8.tex` | `ch:results` |

## 9. Primary sources available locally (updated 2026-09-12)

Coverage level matters for Rule 6 — "full text (own extraction)" means a
PDF was fetched and grepped directly by this session (can support absence
claims about the main text, never about an SI/appendix not read); "fetched
summary" means a WebFetch answered a targeted question and was not
independently re-read (do not support absence claims at all, only
presence claims for what the summary quoted).

| Source | Location | Coverage |
|---|---|---|
| Bury et al. 2021 (PNAS) | `/tmp/claude-216236/bury2021.txt` | full text (own extraction), main text only — SI Appendix not read |
| Babazadeh et al. 2025 (Research Square) | `/tmp/claude-216236/babazadeh.txt` | full text (own extraction) |
| Scheffer et al. 2009 (Nature) | `/tmp/claude-216236/scheffer2009.txt` | full text (own extraction) |
| Dakos et al. 2012 (PLoS ONE) | fetched summary only (PLoS ONE page) | fetched summary — do not use for absence claims |
| Hennekam et al. 2020 (GRL) | `/tmp/claude-216236/hennekam2020.txt` | full text (own extraction) |
| Bagnall et al. 2017 | `/tmp/claude-216236/bagnall2017.txt` | full text (own extraction) |
| Bury 2021 code | `/home/s466553/abc/deep-early-warnings-pnas/` | full repo |
| Dablander & Bury 2022 | fetched summary only (PMC9477405) — full text not obtained despite retry | fetched summary — do not use for absence claims (caught and fixed 2026-09-12) |
| This thesis | `src/`, `testing/`, `config.yaml`, `results/summary/` | full |

## 10. Attribution (git, per session reminder)

- Commit trailer: `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`
  + `Claude-Session:` line.
- Only commit/push when the user asks.
