# Results — provenance, status and terms

This directory contains the aggregate artifacts for the paper. It contains
**no ICEWS event rows, no row-level predictions or labels, no actor-level
predicted probabilities or observed outcome rates, no caches and no run logs**;
the aggregate results remain subject to the ICEWS terms. The illustrative
world-line preview ships a rendered figure and aggregate summaries only — no
row-level trace; see [`../DATA_EXCLUSIONS.md`](../DATA_EXCLUSIONS.md). The
confirmatory holdout was frozen on 2026-10-23; the stability audit is
provisional. Read the status table before quoting a number.

## Status at a glance

| Artifact | Status | What it supports |
|---|---|---|
| `framework_pilot/holdout/` | **Frozen, one-shot, completed** | The pre-registered 49-feature primary, pre-registered 29-feature secondary, baselines, within/between-dyad checks, feature-group LOO and sensitivity tables. |
| `framework_pilot/stability/` | **Provisional; holdout-excluded 50-refit rerun** | Exactness, source/target aggregation, seed/regime/window stability, top-k and decision proxy, HGB occlusion. `audit_design = frozen_during_confirmatory_run_holdout_rows_excluded`: the grid was frozen at 06:31:40Z, after the confirmatory read began at 05:26:55Z and before it completed at 07:45:19Z; the rerun is non-confirmatory and every eventual holdout dyad is excluded at row level. The public `experiments/` tree is a five-seed preview, not this audit; see `experiments/STABILITY_VARIANT.md`. |
| `worldline_preview/` | **Illustrative only** | A two-step communication preview; no multi-step accuracy claim. Ships the vector 150 mm A1 print version — byte-identical to the paper's Figure A1 — and a 300 dpi PNG plus aggregate JSON; no row-level trace file. |

The stability status is recorded machine-readably in `stability/summary.json`:

```json
"holdout": {
  "available_at_audit": false,
  "completed_later": true,
  "used": false,
  "confirmatory_run_start_utc": "2026-10-23T05:26:55.031577Z",
  "completed_utc": "2026-10-23T07:45:19Z",
  "audit_design": "frozen_during_confirmatory_run_holdout_rows_excluded",
  "design_frozen_utc": "2026-10-23T06:31:40Z"
}
```

The audit is **not a holdout-side stability test**. It was recomputed with all
eventual holdout dyads excluded from fitting and aggregation; every stability
number remains provisional and must not be written as a confirmatory headline.

## Artifact map

### Confirmatory holdout

| File | Content |
|---|---|
| `holdout/report.md` | Human-readable one-shot report, protocol, headline metrics, primary comparisons, within/between checks, feature-group anchors/LOO/permutation, calibration, MDE and exact sentences. |
| `holdout/summary.json` | Machine-readable headline aggregates, comparisons and verdict flags. |
| `holdout/parts/holdout_core.json` | Core one-shot arrays and point metrics. |
| `holdout/parts/holdout_extras.json` | Post-holdout diagnostics computed from the saved predictions (bootstrap/within-dyad/LOO). |
| `holdout/tables/*.csv` | The 18 pre-registered tables behind Table 2 and the supporting checks (`holdout_metrics_main`, `primary_comparisons`, `auc_ap_deltas`, `within_between_*`, `feature_group_*`, `calibration`, `hgb_grid`, `logistic_selection`, `mde`, `seed_spread`, `bh_families`, `dyad_fe`), plus the post-hoc pair-cluster sensitivity table. |
| `holdout/preregistration.json` | Frozen pre-registration: definition, endpoints, feature sets, stopping rules and claim boundary. |
| `holdout/preregistration.json.sha256` | Repo-relative SHA-256 of the pre-registration file. |
| `holdout/run_once.json` | One-shot lock record: status, attempts, counts, timestamps. |
| `holdout/test_access_log.json` | Every 2018–22 read, including the pre-registration aggregate support count and the single holdout evaluation. |
| `holdout/manifest.json` | Freeze manifest: host, packages, inputs, code and output hashes, plus post-freeze release annotations. |
| `holdout/SHA256SUMS` | `sha256sum -c` compatible checksums for every distributed file in this directory. |
| `holdout/CORRECTIONS.md` | Errata notes: the 29-feature sign-convention and pair-weighted-label clarification, a misattributed BH q in the 29-feature sentence, and the BH family D `p`/`q` recomputation from the frozen `p_normal` column (all nine still `<1e-12`). |

**Raw support versus filtered rows (manuscript Table 1).** The manuscript reports
raw-panel support of 6,344,160 training, 317,208 selection and 311,638 holdout
dyad-months. The frozen fits and the public aggregates correspond to the filtered
rows: 5,055,328 training, 232,672 selection and 311,638 holdout. The raw-panel
counts — including the 67,057 inner-period positives — are **manuscript-only
context**: no public table, aggregate or test is derived from them, they are not
redistributed, and Table 1 is neither reproduced nor changed here. `run_once.json`
and `manifest.json#/counts` carry the filtered counts.

### Stability audit (provisional)

| File | Content |
|---|---|
| `stability/report.md` | Audit report with a release banner: grid frozen mid-run, holdout not integrated. |
| `stability/summary.json` | Machine-readable summary, status and holdout timing flags. |
| `stability/summary_decompose.json` | Exactness, bucket contract, actor headlines and collinearity. |
| `stability/summary_stability.json` | Seed/period/window stability, variance decomposition, top-k and decision proxy. |
| `stability/summary_hgb.json` | HGB capacity control and block-occlusion agreement. |
| `stability/tables/*.csv` | The tables behind the audit. Actor tables carry aggregate attribution, exposure, bucket shares and ranks only; predicted probabilities (`mean_p`, `p_sum`) and observed counts/rates (`y_sum`, `mean_y`) are not released. |
| `stability/SHA256SUMS` | `sha256sum -c` compatible checksums for all 47 distributed files in this directory (4 summaries, `report.md`, 42 tables). |

### World-line preview

| File | Content |
|---|---|
| `worldline_preview/README.md` | Scope, sample rule, figure status and deletion note. |
| `worldline_preview/worldline_trace.json` | Per-anchor counts, state-path and transition tallies, per-anchor `p1_summary`/`p2_summary` distributions (mean, median, min and max of the step-1/step-2 predicted probability over the 20 sampled, unlabeled dyads; no per-dyad values), disclaimers, model provenance and sample rule. No per-row records. |
| `worldline_preview/run_manifest.json` | Command, host/packages, input/output hashes, preview scope and the no-row-level rule. |
| `worldline_preview/worldline_preview.{pdf,png}` | Illustrative preview figure; vector 150 mm print version, byte-identical to paper Figure A1, Arimo/TrueType, no baked key, no accuracy claim. |

The 240-row `worldline_trace.csv` and 249-row `node_trace.csv` are **not
shipped**: the per-row exports carry `state`/`p_model` and were removed.
The generator writes them only with the explicit `--emit-row-traces`
debug flag.

## Exclusions

These files are recorded in the freeze manifest but are deliberately **not
redistributed**. The manifest marks each as `redistributed: false` with an
`exclusion_reason`; `summary.json` carries the same annotations under
`release_annotations.artifact_redistribution`.

| File | Reason |
|---|---|
| `holdout/preds/holdout_preds.npz` | Row-level predictions and labels; licensing and one-shot exclusion. |
| `holdout/preds/holdout_fe_scores.npz` | Row-level dyad fixed-effects scores derived from the same predictions. |
| `holdout/run.log` | Run log; may contain local paths and operational detail. |
| `holdout/run_blind_quick_aborted.log` | Aborted blind/quick smoke log; operational detail. |

No `preds/` directory, `*.npz`, `*.parquet`, cache, log or `data/` tree is part
of the release. `make verify` fails if one appears.

## Sign convention

- **Prose, README, `summary.json:paper_sentences` and notebook 01:**
  `Δ = model − comparator`. For log-loss, Brier and ECE a **negative** Δ is
  better; for AUC and AP a **positive** Δ is better.
- **Frozen CSVs:** `primary_comparisons.csv` stores `d_log_loss`, `d_ece` and
  `d_brier` as **comparator − model** (positive = primary better). Its
  `multiway_ci_*` columns use the same direction. The prose flips the sign when
  it reports a signed Δ, so `d_log_loss = +0.047999` becomes
  `Δ = −0.047999` (primary better).
- **AUC/AP tables:** `auc_ap_deltas.csv` stores model − comparator, so a
  positive value already means the primary is better.
- **Errata and the 29-feature contrast:** `holdout/CORRECTIONS.md` states the
  same 29-feature versus warm-history interval under both conventions:
  `+0.004510` (comparator − model; positive = secondary better, as frozen in
  `primary_comparisons.csv`) and `−0.004510` (model − comparator; 95% CI
  `−0.00760` to `−0.00142`, as printed in the manuscript).
- **Feature-group LOO:** `delta_vs_full` is “log-loss without the group minus
  the full model”; positive means the removed group carried held-out signal.
  This is a different estimand from the primary contrast and is labelled in the
  notebook.

## Derived-data terms (RESULTS_NOTICE)

The source code is MIT (`LICENSE`). **The aggregate results are not MIT.** They
are derived from ICEWS v37 (doi:10.7910/DVN/28075), which is made available for
research and teaching only, and they remain subject to the ICEWS terms.

Included derived quantities:

- aggregate metrics, confidence intervals, p-values, calibration and stability
  summaries at the model/period/feature-group level;
- actor-month and actor-period **aggregate attribution** tables (196 actors):
  attribution, bucket shares, exposure counts, top features and ranks, with no
  predicted probability and no observed outcome count or rate;
- model-agreement and occlusion diagnostics at the actor/role/period level;
- a deliberately selected illustrative world-line preview (20 sampled dyads
  per anchor, inner period) distributed only as a rendered figure plus
  per-anchor aggregate counts; no row-level trace or per-row probability file;
- top-actor lists and transition counts in the stability and worldline
  metadata.

These are aggregate or small-sample research outputs. They do not permit
reconstruction of the ICEWS event panel, and no attempt to re-identify
individuals or events should be made. The exact removed paths and columns are
listed in [`../DATA_EXCLUSIONS.md`](../DATA_EXCLUSIONS.md). Redistribution of the aggregates outside
the terms of the ICEWS licence is not permitted. Cite ICEWS v37 alongside this
artifact.

Not included: the raw or derived event panel, row-level predictions, labels,
actor-level predicted probabilities or observed outcome rates, feature caches
and run logs.

## Verification

From the repository root:

```bash
# every distributed holdout file
sha256sum -c results/framework_pilot/holdout/SHA256SUMS

# every distributed stability file
sha256sum -c results/framework_pilot/stability/SHA256SUMS

# full public-release verifier (strict JSON, notebooks, paths, secrets)
make verify

# shipped stability aggregates, scope metadata and hashes only
make stability-check
```

Each `SHA256SUMS` lists every distributed file under its directory except
itself. The holdout manifest's `outputs` hashes track the **shipped bytes** for
redistributed files. The stability `SHA256SUMS` pins the four summaries,
`report.md` and 42 tables; the 50-refit scope block is asserted separately by
`make stability-check`, and the public preview scripts cannot silently replace
the files.

Report/summary/extras text and JSON received post-freeze release annotations
(banners, exclusion flags, `NaN` → `null`, and the final wording sync) without
changing any metric, prediction or model output. For those files,
`post_run_corrections.run_sha256` records the freeze-time hash and
`manifest.json:release_annotations.current_sha256` plus `SHA256SUMS` record the
shipped hash.

`holdout/manifest.json` also records the six code files that produced the
holdout aggregates. `manifest.code` is the freeze-time digest of the bytes
actually run; `release_annotations.code_annotations.current_sha256` is the
digest of the bytes shipped at this tag for every entry, and
`changed_since_freeze` names each file that moved with the reason. `make verify`
and `make stability-check` recompute both sides and fail on a missing
annotation or a mismatched digest.

The freeze bundle's `hashes.figures_manifest` is a **normalized** digest, not
the SHA-256 of the `figures.sha256` wrapper file. It is the SHA-256 of the 16
sorted lines produced by removing the leading `repo/` from each wrapper path
(so each line is `<digest>  paper/figures/<name>`), newline-terminated:

```bash
sed 's|  repo/|  |' figures.sha256 | sha256sum
```

Both forms describe the same 16 figure files, and
`sha256sum -c figures.sha256` verifies the wrapper. The published aggregate tree
does not ship either file; this note only explains the freeze identity.

All four stability summaries — including the nested `summary_stability.json`
and `summary_hgb.json` — carry `provisional_holdout_not_integrated`. The
superseded pre-holdout status label appears in no shipped file; the artifact
uses only the current status and the holdout-excluded audit literal.

## Glossary

| Internal label | Public meaning |
|---|---|
| “Sprint A” | The pre-registered confirmatory holdout phase. |
| “Sprint B” | The exact-attribution and stability-audit phase. |
| “P6” | An internal stage identifier; no public claim depends on it. |

## Claim boundary

- Binary dyad-month event prediction; no node-level prediction.
- Unit-fresh, not time-fresh: 2018–22 is retrospective and burned.
- The 49-feature logistic is the pre-registered primary; the 29-feature logistic
  is the pre-registered secondary. Dropping the graph block is a post-hoc
  candidate for dyad-level probability ranking, not a new confirmatory test and
  not a primary swap.
- No causal, actor-lifecycle, state-of-the-art, sim-to-real, human-decision or
  validated multi-step claim.
- Stability is provisional and not holdout-side. The public `experiments/`
  scripts are a five-seed preview and do not reproduce the 50-refit,
  C=1.0, holdout-excluded aggregates.
