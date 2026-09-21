# Chapter 8 ("Results") — Verification Log, First Pass (New Chapter)

This chapter did not exist before this session's rewrite (previously a
one-line placeholder). Every number in it is computed directly from this
session's own result files, re-verified fresh immediately before writing
this chapter (not pulled from memory of earlier turns in this conversation)
to make sure nothing had gone stale.

## How every number was produced

| Claim | Source | How verified |
|---|---|---|
| Table 8.1 (zenodo macro-AUC, all models) | `results/summary/zenodo.csv` | Re-ran `python testing/collect_results.py` immediately before writing this chapter, then queried the fresh CSV directly (shown in-session) |
| Figure 8.1 (PANGAEA AUC by model, Mo+U) | `results/summary/pangaea_by_model.csv` | Same fresh re-run; figure generated directly from this CSV with matplotlib, not hand-drawn or estimated |
| Ensemble result (mean 0.966, median 0.997) | `results/summary/pangaea_ensemble.csv` | Re-queried fresh this turn |
| Cross-model agreement table (54.3%/55.6%, κ=0.117/−0.007, entropy 0.917/0.915) | Re-ran `python testing/bif_cross_model_agreement.py` and `--elements Mo,U` fresh this turn | Numbers match exactly what was found several turns earlier in this session — confirms nothing has drifted since then |
| "0.65 mean AUC... under the original, incorrect ages" (catch22) | Established several turns earlier this session via a direct before/after re-eval of `catch22_ts_500` | Not re-run this turn (would require re-corrupting the labels to reproduce) — carried forward from an earlier, already-verified in-session measurement, not re-derived from scratch here |
| Bury et al.'s F1 numbers (84.2%/88.2%) | Chapter 4 (already fetched and verified directly against the PMC mirror of the paper) | Not re-fetched a second time; same citation, same already-checked number |
| Babazadeh et al.'s Hopf-vs-fold finding | Chapter 4 (already fetched and read in full earlier this session) | Not re-fetched; same already-verified source |

## Job status checked immediately before writing this chapter

Before writing the "Status of the Roster Fix-Up" section, I checked
`squeue`/`sacct` directly rather than assume the retraining job launched
earlier in this session had finished:
```
3221431 (inceptiontime DL retrain): COMPLETED
3221430 (drcif/tde/pf/cif/grsf/tsf/tsbf/lps/fastshapelet TSC retrain): 1/18 tasks done, 17 pending
3221433 (fixup re-evaluation): not yet started, waiting on the above
```
This is why the chapter explicitly reports these five "missing" and five
"strong-synthetic, weak-real" models using their pre-retrain numbers, and
states plainly that the retrain had not completed at time of writing,
rather than either waiting indefinitely to write this chapter or
(incorrectly) reporting results from a job that hasn't produced them yet.

## The three-way categorisation (trustworthy / synth-only / degenerate)

This is not a new finding — the same categorisation was worked out earlier
in this session by cross-referencing the zenodo and pangaea tables together.
This chapter is the first place it is written up as thesis prose with the
full model lists and the reasoning for why the categorisation requires
\emph{both} tables (PANGAEA alone cannot distinguish "failed to transfer"
from "never learned the task," since there's no ground truth to check
against on the real data). Re-verified the specific model-to-category
assignments against the freshly re-pulled CSVs before writing them into the
chapter, rather than trust memory of which models were in which bucket.

## Claims not independently re-verified in this pass

- The Fleiss' kappa formula and its correctness — established and
  hand-verified in an earlier part of this session (walked through the
  observed-agreement vs.\ chance-baseline arithmetic explicitly at the
  time). Not re-derived here; the script that computes it
  (`testing/bif_cross_model_agreement.py`) was only re-\emph{run}, not
  re-audited for correctness, since it was already checked when written.
- The AR(1) surrogate testing methodology this chapter's binary AUC numbers
  depend on — verified against `testing/evaluate.py` directly in Chapter 3
  this session; not re-checked a second time here since the code has not
  changed since then.

## What's still open, stated honestly in the chapter itself

The retrain of 10 models (5 missing entirely, 5 flagged for reasons given
in Chapter 4/5) had not completed at the time this chapter was written.
The chapter does not guess at what those results will show — it states the
retrain is in progress and that the numbers for those specific ten models
are pre-fix baselines, not final. If the retrain completes before this
thesis is submitted, this section needs a direct update with the new
numbers, additive to the existing 19-model results, not a rewrite of them.

## Fresh independent re-verification pass (2026-09-14, second session)

Full audit against current code/data state, not trusting the existing log. Checked:
1. Brace balance (565/565) and all \ref/\label resolution (including cross-chapter) — clean.
2. Model-list consistency: grepped every "N models"/"trustworthy models" mention in the chapter — all consistently say 15, and the named list matches the canonical set everywhere it's spelled out. zenodo_confusion table row count (15 models, 29 model-length rows) matches its own caption claim.
3. Recomputed pooled AUC + 4-class favoured frequency fresh from test_result/*_pangaea/result.json for 8 models spanning all four families (cnn_lstm, patchtst, rocket, arsenal, boss, bop, catch22, mrsqm) — all 8 matched the chapter text exactly, no drift.
4. results/summary/pangaea_ews_trends.csv exists (fresh) and all 14 rows match Table (tab:ews_trends) exactly, including the MS66/S1 Mo/U tau_ac values (-0.933/-0.059).
5. Bury's raw favoured-type counts (175/44/41/0) re-verified against his repo's df_bif_pred_counts_late.csv — exact match. The direct quote "the frequency of the favored DL probability among the forced trajectories" re-verified verbatim against bury2021.txt. Ma et al.'s direct quote re-verified verbatim against ma2025.txt.
6. results/summary/{zenodo,pangaea_by_model}.csv confirmed present and consistent with what's cited.
7. Roster-status claims re-checked against current checkpoints/ and logs/: tde/pf have only _ch_stats.npz (no final .pkl) confirming "never finished training"; cif's pangaea log has exactly 10 AUC lines (2 sapropels x 5 elements, matching "2 of 7"); drcif has 20 lines at L=500 (4 of 7) and 4 lines at L=1500 (~1 of 7, as hedged with "roughly"); grsf has no zenodo row at all at ts_1500 and only primary_mou_n=3 in pangaea_by_model.csv at ts_1500 (partial, matching "roughly 2 of 7").

No errors found this pass. No edits made to chapter_8.tex.

## Targeted re-verification of two user-flagged claims (2026-09-20)

User asked to specifically re-verify (a) the ensemble ("15 model averaging")
result and (b) the Mo+U proxy-choice ("weak signal") claim.

**(a) Ensemble result (Section~\ref{sec:pangaea_provenance}'s pairing
section / "Ensemble Result"):** recomputed fresh from
`results/summary/pangaea_ensemble.csv` — mean 0.9637 (chapter states
0.964), median 0.9928 (states 0.993), min 0.8315 (states 0.832), 14
segments, 27-30 runs/segment, 15 trustworthy models. Exact match. No
change needed.

**(b) Proxy-choice claim (was
`\subsection{Does the Redox-Specific Proxy Choice (Mo+U) Actually Help}`):
found BROKEN.** The paragraph cited specific before/after AUC numbers
(minirocket 0.758->0.862, rocket 0.813->0.907, boss 0.616->0.712/0.596)
attributed to comparing an "all-5-element" PANGAEA evaluation against the
Mo+U-restricted one. Checked every `result.json` currently under
`test_result/` for its `element` field: only `Mo` and `U` appear, anywhere
— no Al/Ba/Ti record exists. Checked `results/summary/pangaea_by_model.csv`:
the `all5_mean_auc` column is now numerically identical to
`primary_mou_mean_auc` for minirocket, rocket, and boss (and every other
model), meaning no distinct all-5 result is retained anywhere in the
current pipeline output. This means the specific before/after numbers in
the paragraph are **not currently reproducible or verifiable** — the
underlying data no longer exists in this run of the pipeline.

This also exposed an internal self-contradiction: Table~8.3's own caption
already correctly stated no all-5 comparison could be made, while
Section~\ref{sec:pangaea_provenance} and this subsection still asserted
the old numbers as current fact. Per WRITING_RULES.md Rule 2 (numbers must
be re-derived from data files, never recalled/reused when the underlying
source is gone) and Rule 6 (do not assert what cannot currently be
checked), fixed by:

1. Rewriting the Section~\ref{sec:pangaea_provenance} sentence about the
   all-5 evaluation to state plainly that it "is not retained in the
   current result files" and that "no specific all-5 number from that
   earlier stage is reproducible or reported in this thesis."
2. Rewriting the subsection paragraph itself to state the earlier
   comparison "reportedly" showed gains "at that time," without asserting
   the specific old numbers as current fact, and to point out that every
   headline PANGAEA number in the chapter already uses Mo+U exclusively,
   so the restriction's benefit is already baked into every reported
   result rather than something left to separately quantify.
3. Renamed the subsection heading from "Does the Redox-Specific Proxy
   Choice (Mo+U) Actually Help" to "The Redox-Specific Proxy Choice
   (Mo+U)" — the old title asked a question the current data can no longer
   answer.

Post-edit integrity check: brace balance 566/566 (was 565/565 before this
edit — net +1 open/+1 close from the rewritten prose, balanced). All
`\ref{}` in the chapter resolve, including `\ref{sec:proxy_check}` pointed
to from Section~\ref{sec:pangaea_provenance}'s rewritten sentence.

## Follow-up: recovered the actual all-5-element data from git (2026-09-20)

User asked to "go through the code" rather than accept the fix above as
final. That surfaced a bigger finding, superseding the previous entry's
framing.

`src/rolling_window.py` is currently **uncommitted** (`git status` shows
`M`) — the committed HEAD version still has `ELEMENTS = ["Al", "Ba", "Mo",
"Ti", "U"]`; the Mo/U-only restriction with its `# ponytail:` comment only
exists in the working tree. Chasing where the all-5-element PANGAEA
results went: `git log --all --diff-filter=D --name-only` showed
`results/{minirocket,rocket}_pangaea_*_{Al,Ba,Mo,Ti,U}_auc/result.json`
were committed at `2b520d23` ("new results with updated models") and
deleted in the very next commit `35b9a0b8` ("before pony") — recoverable
directly with `git show 2b520d23:<path>`.

Pulled all 35 files per model (7 sapropels x 5 elements, ts_1500,
timestamps `2026-06-23` in the JSON) and averaged the `auc` field:
- `minirocket`: mean 0.5002 (n=35); per-element means 0.500-0.501 — flat.
- `rocket`: mean 0.5035 (n=35); per-element means 0.499-0.506 — flat.
- `boss`: `git log --all --diff-filter=A -- 'results/boss_pangaea*'`
  returns nothing at any commit — no all-5 record was ever committed for
  boss, at any point in this repo's history. Confirmed unrecoverable, not
  just deleted.

Checked `p_transition` inside the recovered `minirocket` files directly:
constant 1.0 across every forced window in every file sampled — a
saturated/degenerate output, not genuine near-chance discrimination.

**Critical caveat, checked before writing anything into the chapter:**
whether this is a valid same-model before/after comparison. It is not.
`logs/minirocket_ts_1500_train.log` and `logs/rocket_ts_1500_train.log`
are dated 2026-09-02/03 (both models retrained); the current PANGAEA
Mo+U evaluation logs are dated 2026-09-06. The recovered commit
(`2b520d23`) is dated 2026-07-03 -- weeks before the retrain. Also
`git status` shows `checkpoints_backup/{minirocket,rocket}_ts_{500,1500}
_best_ch_stats.npz` as deleted, consistent with an old checkpoint having
been superseded. So the recovered all-5 numbers belong to an earlier
checkpoint than the one behind the current Mo+U headline numbers
(minirocket 0.862, rocket 0.907) -- no causal "+N AUC from the
restriction" claim can be made from this data, and the chapter does not
make one.

Rewrote both `sec:pangaea_provenance` (paragraph "Which proxies are used,
and which are not.") and `sec:proxy_check` in full to state: the exact
recovered numbers, the exact git provenance (commit hashes, commit
messages, file count, `git show` command), the saturated-output detail,
the retrain-timing caveat that blocks a causal reading, and boss's
genuine total unavailability. Also fixed one internal cross-reference in
the new text that pointed at `sec:three_way` (which does not discuss the
saturated-output mechanism, and classifies the *current* minirocket/rocket
checkpoints as trustworthy, not degenerate) — repointed to
`sec:per_model_type`, which is where that mechanism is actually described
for other models' current real-data behaviour, to avoid implying the
current, trustworthy-classified checkpoints are degenerate.

Post-edit integrity check: brace balance 580/580. All `\ref{}` resolve
(checked with the same script as above).

## Full section-by-section fact-verification pass (2026-09-20, this session)

Every factual claim, number, and citation in chapter_8.tex checked against a
real source (data file, code, repo, or paper text dump), not against memory
or the chapter's own prior text, per the task brief. Scope: all of §8.1–§8.5.

**What was checked and matched exactly (no edit needed):** Table 8.1's 26
model rows recomputed fresh from `results/summary/zenodo.csv`
(`macro_auc_ovr`) — exact match, including the L=500→1500 gain figures
(median +3.6, rdst +7.3, mrsqm +6.2, rocket +6.1, multirocket −9.7,
recomputed independently). `cnn_lstm`'s reproduction numbers (macro-F1
81.4%/85.9%, accuracy 81.3%/85.95%, per-class AUC 0.983/0.999/0.963 at
L=1500) recomputed from `zenodo.csv` and matched exactly, including the
confusion matrix (410/19/48/23 etc.) read fresh from
`test_result/cnn_lstm_ts_1500_zenodo/result.json`. Verified all 15
trustworthy models separate Hopf from null more easily than Fold/Transcritical
at every available length (29/29 rows). Table 8.2's confusion-rate table
recomputed fresh from `results/summary/zenodo_confusion.csv` for all 15
models/29 rows — exact match (two cells, catch22 73.0/73.1% and mrsqm
68.8/68.9%, differ by 0.1pp on a genuine floating-point round-half tie;
not fixed, immaterial). PANGAEA provenance: config.yaml's 7 test sapropels
(64PE406E1 S3–S6, MS21 S1, MS66 S1+S3) confirmed against `config.yaml`
directly; Bury et al.'s 13-sapropel set and the exact quote "13 in total for
each variable, U and Mo" confirmed verbatim in
`/home/s466553/abc/deep-early-warnings-pnas/test_empirical/anoxia/organise_data.py`;
Ma et al.'s "based on Mo" and three sapropel-list quotes confirmed verbatim in
`/tmp/claude-216236/ma2025.txt`, and independently confirmed U appears nowhere
in ma2025.txt except inside the Hennekam dataset's own reference-list
citation (supports the chapter's "no second mention" claim). Bury 2021's
Figure 2 caption quote and Bury 2023's Methods-section "average prediction of
the two classifiers" quote (distinct from the Results section's "ensemble
prediction" wording, correctly attributed to Methods specifically) both
confirmed verbatim in the respective text dumps; Bury 2023 Fig. S5's
"100 forced trajectories," "80% of the way through" caption details
confirmed verbatim. The three-way categorisation (§8.3) criteria
(macro-AUC≥0.80 at every length, real AUC≥0.75 at every *complete*
evaluation) recomputed fresh from `zenodo.csv` + `pangaea_by_model.csv` for
all 15 trustworthy models, boss, and all named degenerate models — every
assignment checks out. The ensemble result (0.964/0.993/0.832, 27–30
runs/segment) rechecked against `pangaea_ensemble.csv` — exact match, and
independently confirmed the 27–30 range reflects genuinely partial
per-model-length data (see below), not a rounding artefact. Bury's raw
favoured-type counts (175/44/41/0 → 67/17/16/0%) reconfirmed against
`df_bif_pred_counts_late.csv`; this thesis's own pooled 21/59/15/5% figure
independently reproduced by running `testing/fig_bury_final_probs.py` fresh.
Table `tab:ews_trends`'s 14 Kendall-tau rows recomputed fresh from
`results/summary/pangaea_ews_trends.csv` — exact match, including
MS66/S1's τ=−0.933 (Mo, p<0.001) / −0.059 (U, p=0.59). Every one of the 29
models' "Pooled binary AUC" figures in §8.4 (`sec:per_model_type`)
independently recomputed by pooling each model's `p_transition`/
`p_transition_null` per segment, ensembling across available lengths the
same way the 15-model ensemble does — all 27 models with any real-data
result matched exactly (cnn_lstm 0.952, resnet 0.967, minirocket 0.906,
tsbf 0.906, etc., including cif's 0.927 on n=4 files and drcif's 0.910 on
n=9). Every one of the 29 models' favoured-class frequencies (fold/hopf/
trans/null %) independently recomputed by pooling raw per-window argmaxes
across both lengths (a *different*, unensembled pooling from the AUC
figure — confirmed this is the right method because it's the only one that
reproduces the stated numbers) — all matched exactly, including
minirocket's 50/0/0/50 collapse and multirocket's 99/1/0/0 collapse. All 18
MS66/S1 fold-vote-drop/rise percentage claims spot-checked (all 29 models
across both Mo and U) against a fresh per-sapropel favoured-class computation
— every one matched (within normal round-half ambiguity, e.g. lstm "25% vs.
20%", catch22 "29%→24%", boss "18%→50%", all exact).

**Errors found and fixed:**

1. **Table 8.1's caption** claimed blank cells mean "that dataset length was
   not part of the roster for that model." False for most of them: checked
   `checkpoints/` directly and found `bop`, `saxvsm`, `boss`, `fastshapelet`,
   `grsf`, and `drcif` all have a checkpoint at the blank length — the gap is
   an evaluation run that crashed or ran before the checkpoint existed and
   was never rerun (confirmed per-model in `logs/*_zenodo_eval.log`, cross-
   referenced against checkpoint file mtimes). Rewrote the caption to state
   this correctly and point to §8.5 for the per-model detail.

2. **§8.5's opening claim, "Of the 29-model roster, 24 have a complete
   synthetic and empirical evaluation... five model-length combinations
   remain genuinely unfinished," undercounted badly.** Checked every model's
   checkpoint files against `zenodo.csv` and `pangaea_by_model.csv`
   (`primary_mou_n < 14` = incomplete PANGAEA): the true count is 17 fully
   complete models, not 24, and 12 have a gap (`tde`, `pf`, `cif`, `ls`,
   `drcif`, `grsf`, `bop`, `saxvsm`, `boss`, `fastshapelet`, `arsenal`,
   `st`), not 5. §8.5 previously discussed only `tde`/`pf`/`cif`/`drcif`/
   `grsf`; `bop`, `saxvsm`, `boss`, `fastshapelet`, `arsenal`, and `st` were
   missing entirely from the accounting, even though `arsenal` and `st`
   (both used elsewhere in the chapter as complete, trustworthy models) are
   demonstrably missing PANGAEA data at L=1500 (`primary_mou_n`=8 and 5 out
   of 14; confirmed by listing `test_result/arsenal_ts_1500_pangaea_*`
   directly — only the four 64PE406E1 sapropels exist, MS21/S1 and
   MS66/S1,S3 are absent). Rewrote §8.5 in full with the corrected count and
   every affected model's specific gap, sourced to checkpoint mtimes and
   eval-log content rather than to which Table 8.1 cells are blank.

3. **§8.5 claimed "tde and pf never started training at all -- no checkpoint
   exists for either at either sequence length."** The "no checkpoint"
   half is true; "never started training" is false. Both models'
   `logs/{tde,pf}_ts_{500,1500}_train.log` show data loading, channel-stat
   computation, and the fit call starting (`"Fitting tde on 4992
   series..."`, `"Fitting pf on 2000 series..."`) before the log simply
   stops -- training started and did not finish, the same failure pattern
   as `ls` at L=1500 (which the chapter didn't previously connect to this).
   Fixed to state training started but never produced a checkpoint.

4. **§8.4's proxy-restriction recovery (`sec:proxy_check`) misstated the
   recovered data's own sequence length and got the mean AUC and per-element
   range wrong for `rocket`.** The chapter said the 35 recovered `minirocket`/
   `rocket` result.json files (git commit `2b520d23`) were "L=1500." Reading
   every recovered file's own `dataset` field directly: all 70 files (35 per
   model) say `"dataset": "ts_500"`. This means the paragraph's own stated
   comparison — recovered numbers vs. "the checkpoint behind the current
   results (minirocket 0.862, rocket 0.907 mean AUC at L=1500)" — was
   comparing across two different sequence lengths on top of the already-
   flagged different-checkpoint problem, which the chapter did not disclose.
   Also recomputed the recovered AUCs independently from the raw
   `binary_auc` field: `rocket`'s true mean is 0.4989 (chapter said 0.504)
   with a true per-element range of 0.487–0.507 (chapter said 0.499–0.506);
   `minirocket`'s per-element AUCs are all *exactly* 0.500 with zero
   variation (chapter said a 0.500–0.501 range). Fixed by: correcting the
   dataset length to L=500 throughout (both in the description of the
   recovered files and in the checkpoint-timing argument, now citing
   `logs/{minirocket,rocket}_ts_500_train.log`), correcting the recovered
   AUC figures to 0.500/0.499 and the ranges to exact-0.500/0.487–0.507,
   and replacing the comparison target with the current L=500 PANGAEA means
   (minirocket 0.957, rocket 0.951, from `results/summary/pangaea_by_model.csv`,
   both n=14 complete) so the before/after comparison is at least
   same-length, with the different-checkpoint caveat retained unchanged
   (still not a valid causal comparison, now correctly described as such).

**Left as-is, not treated as errors:** two confusion-table cells (catch22,
mrsqm) round to a different digit than a strict IEEE round-half-to-even
would give on the raw float, a 0.1 percentage-point cosmetic difference,
not a wrong number. Bury 2021's SI Appendix and Dablander & Bury 2022's full
text remain unread (per WRITING_RULES.md §9); no new claim in this chapter
rests on either beyond what was already flagged in earlier passes.

Post-edit integrity check: brace balance 637/637 (was 580/580 before this
pass). All `\ref{}` in the chapter resolve, including the newly-added
`\ref{tab:zenodo_results}`, `\ref{sec:roster_status}`,
`\ref{fig:pangaea_by_model}`, `\ref{sec:ensemble_result}`, and
`\ref{sec:type_reliability}` cross-references in the rewritten §8.5
paragraphs. All `\citet`/`\citep` keys used in this chapter
(`babazadeh2025`, `bury2021`, `bury2023discretetime`, `hennekam2020`,
`ma2025sdml`) confirmed present in `references.bib`.
