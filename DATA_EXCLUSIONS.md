# Data exclusions and privacy boundary

**Applies to:** the public artifact `lags1-dyad-explanations`, release candidate
0.1.1 (`review-r2c6` tree). **Status:** the earlier row-level leak is closed (the
contaminated history is quarantined in a private repository and is not reachable
from this tree).

This file is the authoritative inventory of what the public tree removes or
redacts, and why. It records file paths, schemas, row counts and reasons only.
It does not reproduce any removed value, label, probability, actor-month
observation or dyad identifier. No raw ICEWS data and no holdout label is read
or redistributed. The public `experiments/` tree remains a **five-seed preview**
of the stability method, not the restricted 50-refit, holdout-excluded audit.

## Rule

A distributed file may contain **aggregate attribution, exposure, bucket shares,
ranks and model-level metrics**. It must not contain:

1. a row-level label (`y`) together with a row-level prediction (`p`,
   `p_model`) for the same unit;
2. an actor-level predicted probability or observed outcome rate
   (`mean_p`, `p_sum`, `y_sum`, `mean_y`) attached to a named or id-addressable
   actor;
3. a dyad-month contribution vector (per-feature `c_*`, per-bucket `b_*`,
   `eta`, `intercept`, `total_contrib`) attached to a dyad identifier;
4. any row derived from the 2018–2022 confirmatory holdout.

The only declared exception is the bounded world-line preview described below.

## 1. Removed file

| File (removed) | Was | Why |
|---|---|---|
| `results/framework_pilot/stability/tables/dyad_contributions_inner_sample.csv` | 3,450 dyad-month rows × 68 columns; `did/sid/tid/source/target/month`, `y`, `p`, `eta`, `intercept`, 49 per-feature contributions, 6 bucket sums and a check column; inner period 2016-01…2017-11 | Row-level labels plus fitted probabilities plus the full per-feature contribution vector for 3,179 directed dyads. It is exactly the derived row-level panel the release says it withholds. No replacement row sample is shipped, and no schema-only copy is retained. |

No holdout row was present in the removed file (inner period only), so no
confirmatory number changes. The file was also removed from git history in the
release history rewrite (`review-r2c2`); `git ls-files` and
`git rev-list --objects --all` contain no entry for it.

## 2. Redacted actor tables

Thirteen shipped actor tables remain as **aggregate attribution and exposure**
tables. Four distinct column names were removed in total — three from the six
actor-month tables, two from the six actor tables and two from the
all-periods table — and no other column was changed. No value in a retained
column was recomputed.

| Files | Rows | Removed columns | Retained content |
|---|---|---|---|
| `stability/tables/actor_month_{source,target}_{inner_2016_17,retro_2018_22,covid_2020_21}.csv` (6) | 4,508–11,564 each | `p_sum`, `y_sum`, `mean_p` | period, role, actor id and ISO3 label, month index, month, `n_dyads`, attribution, total contribution |
| `stability/tables/actor_{source,target}_{inner_2016_17,retro_2018_22,covid_2020_21}.csv` (6) | 196 each | `mean_p`, `mean_y` | period, role, actor id and ISO3 label, region, exposure (`n_rows`, `n_months`), attribution and bucket shares, top-3 features/contributions, ego share, attribution rank and percent |
| `stability/tables/actor_attribution_all_periods.csv` (1) | 1,176 | `mean_p`, `mean_y` | same aggregate attribution/exposure schema across the three periods |

Why: `mean_p`/`p_sum` are predicted probabilities, `y_sum`/`mean_y` are observed
event counts/rates. Combined with an actor id or ISO3 label they reconstructed
an actor-month risk ranking, which contradicted the manuscript's "no actor risk
scores" statement. Attribution is a bookkeeping decomposition of a dyad
forecast, not a probability, an intent or an operational risk score.

## 3. Removed embedded JSON keys

`results/framework_pilot/stability/summary.json` embedded the same quantities
inside `source_target.top_actors` (60 records). The `mean_p` and `mean_y` keys
were removed from every record; `period`, `role`, `side`, `actor`,
`attribution` and `ego_abs_share` remain. The `files` list no longer names the
removed dyad table. JSON remains strict (no `NaN`/`Infinity`).

No other shipped JSON contains `mean_p`, `p_sum`, `y_sum` or `mean_y`.

## 4. Removed row-level exports — world-line preview

`results/worldline_preview/` no longer ships any row-level file. The 240-row
`worldline_trace.csv` and the 249-row `node_trace.csv` were removed from the
public tree: they carried per-row `state` and `p_model` (dyads) and per-row
induced activity (actors). The rendered illustration and the per-anchor /
aggregate summaries remain.

| Property | Value |
|---|---|
| Removed machine-readable rows | `worldline_trace.csv` 240 dyad-step rows (`p_model`, `state`); `node_trace.csv` 249 induced-actor rows (`state_prev`, `active`) |
| Retained | `worldline_preview.pdf` / `worldline_preview.png` (rendered illustration), `worldline_trace.json` (per-anchor counts, transition tallies, per-anchor `p1_summary`/`p2_summary` distribution statistics — `mean`, `median`, `min`, `max` of the predicted probability over the 20 sampled, unlabeled dyads at steps 1 and 2, with no per-dyad value and no identifier — and provenance), `README.md`, `run_manifest.json` |
| Period | inner validation 2016-01…2017-11; maximum `t_idx` 274 (the 2017-12 embargo row is absent) |
| Holdout rows | 0 |
| Sample | 20 sampled dyads per anchor over four anchors (71 distinct across anchors); deliberately selected for visibility, not random |
| Labels | the PNG/PDF keeps ISO3 labels for readability; those labels are not join keys and no probability is exported. Local debug runs (`--emit-row-traces`) may still write anonymized `dyad_XX` / `actor_XX` CSVs, but they must not be committed |
| Rationale | the frozen paper's Appendix A1 figure is rendered from this preview; the figure is marked `ILLUSTRATIVE PREVIEW - NOT VALIDATED`, no accuracy is computed, `labels_used_for_scoring=false`, and deleting the preview changes no main result |
| Machine enforcement | `scripts/verify_release.py` fails if either per-row CSV reappears under `results/worldline_preview/`; `make worldline-check` asserts the directory ships no CSV, the aggregate JSON is present, the inner-period bound holds and the manifest covers only the rendered/aggregate outputs |

The boundary is bounded and enforced: restoring a per-row export, adding
anchors, or adding holdout rows fails `make verify` / `make worldline-check`.

## 5. Reviewed and retained (no probability, label or outcome rate)

| File(s) | Content | Why it is retained |
|---|---|---|
| `decision_impact_actor_month.csv`, `decision_impact_dyad.csv` | aggregate alert precision/lift/Jaccard by period, role and `k` | no actor/dyad id, no probability, no `y`; Table 3B is reproducible from these aggregates |
| `bucket_shares.csv`, `feature_attribution_periods.csv`, `feature_bucket_map.csv`, `feature_stability.csv` | bucket/feature attribution and rank stability | feature-level or numeric-actor-id aggregates; no probability, label or outcome rate |
| `stability_*.csv`, `variance_decomposition.csv`, `topk_churn_*.csv`, `sign_flip_summary.csv` | stability, variance, churn and sign-flip summaries | model/period/role aggregates; no row-level unit |
| `actor_regime_shift.csv`, `actor_seed_variance.csv` | actor-level attribution shift and seed variance keyed by numeric actor id | attribution only; no probability, label or outcome rate |
| `hgb_occlusion_*.csv` | HGB/logistic occlusion agreement and per-actor failure diagnostics | model-agreement diagnostics (delta log-loss), not predicted event probabilities or observed event rates |
| `holdout/tables/*.csv`, `holdout/*.json` | one-shot holdout aggregates | no row-level prediction or label; `preds/` remains excluded |

Retained actor-level attribution is explicitly framed as **not** intent, not a
threat score and not a node-level forecast. See `NOTICE` and
`results/README.md`.

## 6. Never released

- ICEWS raw or derived event panels, parquet files and feature/graph caches.
- Row-level holdout predictions and labels (`results/framework_pilot/holdout/preds/*.npz`).
- Run logs, experiment configuration and the manuscript.
- Any 2018–2022 confirmatory label or dyad-month row.

## 7. Enforcement

From the repository root:

```bash
make verify            # includes the row-level/named-rate scan below
make stability-check   # scope, hashes and aggregate-only tables
make worldline-check   # no row-level file, period bound, manifest hashes
make test              # contract + negative tests
sha256sum -c results/framework_pilot/stability/SHA256SUMS
```

`scripts/verify_release.py` fails if any distributed CSV:

- contains both `y` and `p`, or a `p_model` column;
- contains `mean_p`, `p_sum`, `y_sum` or `mean_y`;
- combines a prediction/label column with a dyad/actor/month identifier;
- is named `dyad_contributions_*`.

It also fails if any `results/**/*.json` contains a `mean_p`, `p_sum`, `y_sum`
or `mean_y` key, and if either removed world-line export
(`worldline_trace.csv`, `node_trace.csv`) reappears. `tests/test_results_contract.py`
repeats the positive checks and adds synthetic negative tests for the CSV rule
and for a reappearing world-line trace.

## 8. What did not change

No model, fit, holdout evaluation, stability refit, bootstrap or aggregate
metric was recomputed. No frozen headline metric changed; BH family D
`p`/`q` columns were re-expressed from the frozen normal-tail `p_normal`
values (Benjamini–Hochberg over nine groups) and every group is still rejected
at `q < 1e-12`. The 50-refit stability audit remains provisional and
holdout-excluded; the public experiments remain a five-seed preview. The
L0–L2 public scope and the "no node-level, causal, lifecycle or SOTA" boundary
are unchanged.
