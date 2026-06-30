# CHANGELOG

All changes are listed chronologically. One bullet per fix. Branch: `refactor-clean`.

---

## Phase 0 — Repo Hygiene

- **[0.1]** Created branch `refactor-clean` from `main`.
- **[0.2]** Deleted stray files: `testing/evaluate.py#` (Emacs backup), `training/.gitignorgit` (typo duplicate).
- **[0.3]** Updated `.gitignore`: added `old_results_backup/`, `logs/`, `dataset/processed/`, `data/**/*.npz`, `*.keras`, `__MACOSX/`; consolidated checkpoint rules; added rule for nested pangaea CSVs.
- **[0.4]** Removed `old_results_backup/` from git tracking (`git rm --cached -r`) — files kept on disk, now gitignored. Reason: 247 binary/result files have no place in version control.
- **[0.5]** Created `src/constants.py` with `CLASS_NAMES = ["fold","hopf","transcritical","null"]` and `NULL_IDX = 3`. This is the canonical Bury class order; all files that hard-code `null_idx` should import from here.
