# Stability variant: public preview vs the 50-refit audit

This note records the exact relationship between the shipped `experiments/`
Sprint-B code and the restricted variant that produced
`results/framework_pilot/stability/`.

**Bottom line.** The public tree is an **L0–L2 artifact**: it reads, verifies
and re-renders the frozen aggregates. `experiments/` is a **five-seed preview**
of the audit method. It is **not** the code that produced the shipped stability
tables and it is **not byte-reproducible** as the 50-refit, holdout-excluded
audit. No raw holdout labels and no row-level predictions are shipped or read by
the preview or by this note. The shipped aggregates and their `SHA256SUMS` are
the authoritative record.

## Scope of the shipped audit

The shipped stability result is:

- status `provisional_holdout_not_integrated`;
- `audit_design = frozen_during_confirmatory_run_holdout_rows_excluded`: the
  grid was frozen at `2026-10-23T06:31:40Z`, i.e. **after** the confirmatory
  read began at `2026-10-23T05:26:55.031577Z` and **before** it completed at
  `2026-10-23T07:45:19Z`; the 50-refit rerun ran afterwards, `14:30`–`16:07Z`
  (`generated_utc 16:07:57Z`). The holdout was not used. This is not a
  pre-holdout design: the design freeze is mid-run, so the audit is
  non-confirmatory and post-holdout in execution;
- **50** dyad-cluster bootstrap refits with **C fixed at 1.0**;
- computed after excluding all rows of the eventual holdout dyads from fitting
  and aggregation: **5,282 directed dyads / 2,790 unordered pairs**, taken from
  `results/framework_pilot/holdout/preregistration.json`;
- a **50-replicate** month-block bootstrap in `summary_stability.json`.

The distributed actor tables are **aggregate attribution/exposure** tables only.
Predicted probabilities (`mean_p`, `p_sum`) and observed counts/rates
(`y_sum`, `mean_y`) are not shipped; see `DATA_EXCLUSIONS.md` and
`results/README.md`.

The exclusion is a feature-use exclusion based on the dyad ids in the public
plan (`preregistration.json`). It is not a label read.

## The five deltas

| # | Public preview (`experiments/`) | Restricted audit variant | Anchor |
|---|---|---|---|
| 1 | `N_SEEDS = 5` | `N_SEEDS = 50` | `sprint_b_common.py` |
| 2 | No holdout-dyad exclusion | `_holdout_dids()` reads the public plan and removes all rows of the 5,282 holdout dyads (2,790 unordered pairs) from train/select/test/panel before fitting and aggregation | `sprint_b_common.py` (`_holdout_dids`, `_subset_did`, `load_splits`) |
| 3 | `C` selected automatically on the 2015 selection window (`fit_logistic` default `C_grid`) | `C_grid=[1.0]` in the point fit; seed refits inherit the point-fit `C` | `sprint_b_common.py` (`fit_point`) |
| 4 | `N_BOOT = 1000` in the stability bootstrap | `N_BOOT = 50` in the stability bootstrap | `sprint_b_stability.py` |
| 5 | Loader hard-codes `("point", "rep0", …, "rep4")` | Loader enumerates `["point"] + [f"rep{s}" for s in range(C.N_SEEDS)]` | `sprint_b_stability.py` (`main`) |

Notes on the deltas:

- Delta 3 changes the estimand: the preview can select a different penalty than
  the confirmatory primary. The audit fixes the penalty at the frozen primary
  value rather than re-selecting after the holdout exclusion.
- Delta 5 is an enumeration limitation, not a statistical choice. Changing
  `N_SEEDS` alone would still fail when the loader asks for `rep5`; all five
  deltas are required for method parity.
- The HGB post-processing script keeps `N_BOOT = 1000` in both trees. The
  50-replicate limit applies to the stability bootstrap behind
  `summary_stability.json`, not to `sprint_b_hgb_post.py`.
- The output paths (`stability/`, `stability/tables/`, `stability/cache/`) and
  the current status literals are identical in both trees. They are not
  additional deltas.
- Status comments, the HGB `holdout` metadata block and docstring anchors also
  differ between the trees. Those are metadata/prose edits, not method deltas;
  the five rows above are the complete statistical difference.

## What cannot be reproduced from the public tree

Even with all five deltas applied, a licensed replicator cannot produce a
byte-identical copy of the shipped stability tree from the public artifact
alone:

- The restricted derived feature and contribution caches are not redistributed,
  and the public tree has **no cache-build entry point**. Licensed ICEWS access
  is necessary but not sufficient.
- `generated_utc`, runtime fields and last-bit floating-point differences would
  change on a rerun even with the same inputs and code.
- Row-level predictions and run logs are excluded.
- The confirmatory holdout is closed by a one-shot lock: `run_once.json` records
  completion and `test_access_log.json` records every post-freeze 2018–22
  holdout read (the three earlier exploratory passes G2, D2 and B predate the
  log and are out of its scope; this audit's own burned-retrospective read is
  logged in `summary_decompose.json`). Re-running
  it would violate the one-shot lock and the plan's split commitment and is
  outside this artifact's contract.

The frozen aggregates, not a regenerated run, are what the public verifier
checks.

## Write protection

The public preview defaults to **refusing** to write into
`results/framework_pilot/stability/`, the reserved authoritative tree. Two
explicit opt-ins exist:

```bash
# Recommended: write preview output to a separate tree.
NEUROSIM_STABILITY_OUT=results/framework_pilot/stability_preview \
  python3 experiments/sprint_b_stability.py

# Only when intentionally replacing the shipped aggregates.
NEUROSIM_ALLOW_OVERWRITE=1 python3 experiments/sprint_b_stability.py
# or: python3 experiments/sprint_b_report.py --allow-overwrite
```

The guard is checked before any directory is created, so a refused preview run
does not modify the shipped tree. `make test`, `make verify`,
`make stability-check`, `make worldline-check` and `make notebooks` do not run
these scripts and do not write results.

## Verification

From the repository root:

```bash
sha256sum -c results/framework_pilot/stability/SHA256SUMS
make stability-check
make verify
```

`make stability-check` asserts the status, the 50-refit / C=1.0 /
holdout-excluded scope fields, the 50-replicate bootstrap and the stability
checksums. `make verify` additionally checks that this note is present and that
the public documentation does not claim a five-seed audit or the legacy
`provisional_no_fresh_holdout` status.
