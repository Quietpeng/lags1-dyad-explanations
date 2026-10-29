# Code map

The public package contains the paper path only: the ownership decomposition,
the V2 matched-vintage audit and the within-panel stability audit, not a
reusable framework and not a new estimator. The
directory and import name `framework/` are retained for compatibility (tests and
the installed package import `framework`); they do not imply general-purpose
abstractions. Earlier pilot, synthetic/T1, graph-encoder, operator and reporting
modules were removed from the shipped tree; the quarantined history that
contains them is not part of this public release, and they are not importable
or tested here.

## Paper path (`framework/`)

| Module | Role |
|---|---|
| `__init__.py` | Release version; no node-level or GraphSAGE claim. |
| `config.py` | JSON defaults, recursive overrides, split invariants. No config file is required to import or test. |
| `data.py` | DuckDB connection, ICEWS panel build, cache and tail mask. Restricted inputs are caller-supplied. |
| `features.py` | Frozen 29-feature dyad contract (`27 causal + br + seen_tr`), 20-feature graph core, 35-feature graph bank, actor-month features; no labels. The 29 dyad features are **27 causal features plus two train-only statistics** (`br`, `seen_tr`); the emitted manifest tag `available_at=t` is **inaccurate for training-row encoding** (manuscript §4.2; Data availability). |
| `splits.py` | Temporal train/select/eval/test masks, actor/dyad/region holdouts, cold-start imputation and D2 windows. |
| `models.py` | Standardised L2 logistic, HistGradientBoosting capacity control, temperature/isotonic calibration, persistence/global-transition/per-dyad Markov baselines, exact linear contributions and occlusion. |
| `metrics.py` | Binary log-loss/Brier/ECE/accuracy/F1/AUC/AP, reliability slope, ICC/design effect and MDE. |
| `bootstrap.py` | Cluster bootstrap, paired gains, multiway sufficient statistics and AUC/AP intervals. |
| `sprint_a.py` | Pre-registration hashing, one-shot holdout bundle, split assertions, warm-history baselines, model specs and metrics blocks. |
| `sprint_a_inference.py` | Multiway/dyad/source/target paired inference, BH-FDR and paired MDE. |
| `aggregation.py` | Exact additive attribution, six ownership buckets, actor-month views, top-k/Jaccard stability and window helpers. |
| `regions.py` | ISO3 macro-region mapping used by the panel build. |
| `utils.py` | Logging/tee, JSON, hashing, host/package metadata, timing. |

The contract suite exercises these modules with synthetic arrays and shipped
aggregates only. `framework/data.py` needs the restricted ICEWS panel, so only
its `tail_mask` helper is unit-tested; `bootstrap.py`, `sprint_a.py`,
`sprint_a_inference.py` and `regions.py` have focused unit tests that read no
data, configuration or predictions.

## Entry points

| Command | Purpose | Restricted inputs |
|---|---|---|
| `make doctor` | Python/dependency/forbidden-path preflight | no |
| `make test` | Unit and contract suite | no |
| `make verify` | Release verifier (hashes, strict JSON, notebooks, scope, paths) | no |
| `make notebooks` | Execute the three read-only notebooks | no |
| `make stability-check` | Verify shipped stability aggregates, 50-refit scope and hashes | no |
| `make worldline-check` | Verify the worldline illustration/aggregate boundary (no CSV) | no |
| `make dist` | Build and verify a `git archive HEAD` release tarball; untracked files never enter the archive and a modified tracked file aborts the build | no |
| `experiments/sprint_a_prereg.py` | Freeze the holdout plan (`preregistration.json`) | ICEWS + cache |
| `experiments/sprint_a_run.py` | One-shot holdout evaluation (refuses a second completed run) | ICEWS + cache |
| `experiments/sprint_a_holdout_extras.py` | Post-holdout diagnostics from saved predictions | row-level predictions |
| `experiments/sprint_b_*.py` | Five-seed preview of exactness, stability, HGB occlusion and report generation; not the 50-refit audit runner | ICEWS + cache + (post-hoc) row-level predictions |
| `experiments/worldline_preview.py` | Illustrative two-step preview | ICEWS + cache |

The experiment scripts are audit material. They are not invoked by any public
`make` target, and the public tree does not contain their inputs or local
configuration. The code that produced the restricted 50-refit audit is not
shipped. The shipped Sprint-B scripts are a **five-seed preview**: five
seed replicates, no holdout-dyad exclusion, `C` re-selected on the selection
window, 1,000 stability bootstrap replicates and a fixed `rep0`–`rep4` loader.
They do not regenerate the 50-refit, C=1.0, holdout-excluded aggregates behind
`results/framework_pilot/stability/`; `experiments/STABILITY_VARIANT.md`
records the exact five differences. The preview refuses to write into that
reserved tree unless `NEUROSIM_ALLOW_OVERWRITE=1` (or `--allow-overwrite`) is
set, and `NEUROSIM_STABILITY_OUT` redirects output to a preview directory. The
shipped aggregates and `stability/SHA256SUMS`, not these scripts, are the
authoritative audit record.

## Data flow (paper path)

```text
ICEWS v37 event panel (restricted)
        │  data.py
        ▼
dyad-month / actor-month panels + feature cache
        │  features.py
        ▼
29 dyad features (27 causal + train-only br/seen_tr) + 20 graph features (= 49)
        │  splits.py (unit-fresh 20% dyad holdout)
        ▼
logistic49 (primary) / logistic29 (secondary) / HGB49 (capacity control)
        │  models.py + metrics.py + sprint_a_inference.py
        ▼
frozen holdout aggregates (summary.json, tables/*.csv, report.md)
        │  aggregation.py
        ▼
exact source/target actor contributions + within-panel stability audit (provisional)
        │
        ▼
read-only notebooks 01–03
```

## Results tree

| Path | Status | Notes |
|---|---|---|
| `results/framework_pilot/holdout/` | Frozen, one-shot completed | Aggregates only; `preds/`, `run.log` and the aborted smoke log are **not redistributed**. `manifest.json` records freeze hashes and release annotations; `SHA256SUMS` covers every distributed file. |
| `results/framework_pilot/stability/` | Provisional, holdout not integrated | Frozen 2026-10-23T06:31:40Z; holdout completed 07:45:19Z; `available_at_audit: false`, `completed_later: true`, `used: false`. The shipped files are the holdout-excluded 50-refit rerun; `SHA256SUMS` covers all 47 distributed files. Actor tables are aggregate attribution/exposure only. The public scripts are a five-seed preview. |
| `results/worldline_preview/` | Illustrative only | No multi-step accuracy claim; the shipped PDF is a standalone repo-only 150 mm vector print version (425.197 × 330 pt; no paper figure or appendix is derived from it) and the PNG is a 300 dpi preview. Only rendered/aggregate outputs ship: the per-row trace CSVs were removed, so no row-level probability file is distributed (`DATA_EXCLUSIONS.md`). |

## Removed legacy modules

`audit.py`, `graph.py`, `interpret.py`, `operators.py`, `pipeline.py`,
`reporting.py`, `run_pilot.py`, `synth.py`, `synth_arms.py`, `targets.py` and
`windows.py` were part of an earlier pilot (synthetic/T1 arms, a hand-written
graph encoder, operator heads and a synthetic runner). They are not used by the
paper path, tests or public results. `scripts/freeze_manifest.py`,
`scripts/leakage_audit.py` and `scripts/verify_snapshot.py` were likewise
removed because they required restricted inputs and could not run from the
public tree. The active verifier is `scripts/verify_release.py`.

## What is deliberately absent

`paper/`, `docs/`, `configs/`, `data/`, caches, logs, row-level predictions and
actor-level probabilities/outcome rates. `results/framework_pilot/stability_preview/`
is a local preview output path and is not shipped. Root-level `README.md`,
`REPRODUCING.md`, `CODE_MAP.md`, `DATA_EXCLUSIONS.md` and `results/README.md`
carry the documentation that would otherwise live in `docs/`.
