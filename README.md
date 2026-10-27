# Actor-role components of dyadic event forecasts: exact attribution, near-collinearity, and a within-panel stability audit

**Exact attribution, near-collinearity, and a within-panel stability audit.**

Code, tests and aggregate results for the **actor-role attribution contract**
and **within-panel stability audit** of the manuscript: an ex ante ownership
rule that fixes unit and sign before results, exact additive source- and
target-role components for binary dyad-month political event forecasts, and a
separate stability audit. The deliverable is a contract plus audit, not a
general-purpose framework and not a new estimator: the decomposition is the
logistic identity and coincides with linear SHAP under a training-mean
background.

The public artifact is an **L0–L2 verification release**: reading the frozen
aggregates, verifying their hashes and contracts, and re-rendering the
notebooks. It does not regenerate the restricted experiment pipeline: the
50-refit audit code is restricted and not redistributed, and row-level
predictions and labels are withheld. The aggregate results do not redistribute
event rows and remain subject to the ICEWS terms. The manuscript is not part of
this repository.

**Release status:** v0.1.1 release candidate under review; the metadata version
is `0.1.1`, but no `v0.1.1` git tag has been created and no DOI has been
minted; until the tag and DOI exist, cite the repository URL and this candidate.
Production replaces the review-candidate label with the `v0.1.1` tag and DOI; do
not cite this tree as an archival release. The confirmatory
holdout is frozen. The stability audit is **provisional**: it was recomputed
after excluding all eventual holdout dyads from fitting and aggregation, with C
fixed at the primary value and 50 dyad-cluster bootstrap refits. The
confirmatory holdout completed later and was not used. The shipped
`experiments/` Sprint-B scripts are a five-seed preview, not the restricted
50-refit audit; see `experiments/STABILITY_VARIANT.md`.

## What is here

- `framework/` — CPU-only Python library implementing the paper path: the
  dyad-month feature contract, the primary/secondary/capacity model set (a flat
  comparator list, not a ladder), calibration, exact additive decomposition,
  actor views, metrics and cluster-bootstrap inference. The package contains
  the paper path only; the directory and import name are retained for
  compatibility and do not imply a reusable framework.
- `experiments/` — scripts for the pre-registered confirmatory holdout, a
  five-seed preview of the stability audit and the illustrative world-line
  preview. They require the licensed ICEWS panel and local caches; post-hoc
  holdout diagnostics additionally need row-level predictions. None of those
  inputs is redistributed. The Sprint-B preview does not regenerate the shipped
  50-refit, holdout-excluded aggregates: `experiments/STABILITY_VARIANT.md`
  lists the exact five differences. The preview refuses to write into
  `results/framework_pilot/stability/` unless `NEUROSIM_ALLOW_OVERWRITE=1` (or
  `--allow-overwrite`) is set; `NEUROSIM_STABILITY_OUT` redirects output to a
  preview directory.
- `tests/` — unit and contract tests for features, splits, models, attribution,
  frozen results and notebooks. They run without data, configuration or cache.
- `notebooks/` — three executed, read-only notebooks that display the frozen
  aggregate artifacts. They do not fit models, read the ICEWS event panel or
  load row-level predictions.
- `results/framework_pilot/holdout/` — one-shot unit-fresh 20% dyad-holdout
  report, summary, tables, pre-registration, manifest and access log. Row-level
  predictions and run logs are not redistributed.
- `results/framework_pilot/stability/` — **provisional** stability audit on
  inner-validation and burned-retrospective periods, with a `SHA256SUMS`
  manifest covering every distributed stability file. Actor tables are
  aggregate attribution/exposure only; no predicted probabilities or observed
  outcome rates are shipped.
- `results/worldline_preview/` — illustrative two-step preview; not a validated
  multi-step forecast. The shipped PDF is the vector Appendix A1 print version;
  the PNG is a 300 dpi preview. Only the rendered figure and aggregate JSON
  summaries ship: the per-row trace CSVs were removed, so the directory carries
  no row-level probability file (see `DATA_EXCLUSIONS.md`).
- `scripts/` — release verifier, aggregate checker, notebook executor and
  preflight doctor. `CODE_MAP.md` maps modules to the paper path;
  `REPRODUCING.md` gives the L0–L4 reproduction matrix.

## What is not included

- ICEWS raw or derived event data (research-and-teaching terms; not
  redistributable), row-level predictions/labels, actor-level predicted
  probabilities or observed outcome rates, caches and run logs.
- The manuscript, internal project documents and experiment configuration.
- Earlier synthetic and hierarchy pilot code; it is not part of this paper.

`DATA_EXCLUSIONS.md` lists every file removed or column redacted for this
boundary, including the removed world-line trace exports, and the machine
checks that enforce it.

## Quick start

Requires **Python 3.12+**: the pinned `numpy==2.5.4` and `scipy==1.18.1`
releases publish no 3.11 wheels. This artifact was tested on Python 3.13.5.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
make doctor
make test
```

`make test` runs the public suite. `make verify` checks hashes, strict JSON,
notebook outputs, forbidden paths, absolute paths and credential-like strings.
No restricted input is required by either target.

To read the frozen results without installing Jupyter:

```bash
python3 - <<'PY'
import json
s = json.load(open("results/framework_pilot/holdout/summary.json"))
print(s["status"])
print(s["primary"]["metrics"])
PY
```

To execute the notebooks headlessly from the repository root (optional
notebook stack):

```bash
python3 -m pip install "nbclient>=0.11" "nbformat>=5.11" "ipykernel>=7.0" "jupyterlab>=4.4"
make notebooks
```

The notebooks are read-only with respect to the frozen artifacts: execution
re-renders the shipped aggregates and does not re-fit or re-score anything.

## Reproduction boundary

`make doctor`, `make test`, `make verify`, `make stability-check`,
`make worldline-check` and `make notebooks` run from this tree without
restricted inputs. Regenerating the aggregate tables (L3) is not executable
here: it needs licensed ICEWS v37, the restricted derived feature and
contribution caches, the local experiment configuration, and for post-holdout
diagnostics the excluded row-level predictions. The public tree has **no
cache-build entry point**, so licensed access alone is not sufficient. The
confirmatory holdout (L4) is closed by a one-shot lock (`run_once.json` plus
`test_access_log.json`); re-running it would invalidate the pre-registration.
See `REPRODUCING.md` for the L0–L4 matrix.

## Sign convention

**Δ = model − comparator.** For log-loss, Brier and ECE a **negative** Δ is
better; for AUC and AP a **positive** Δ is better. The frozen tables under
`results/` store `d_log_loss`, `d_ece` and `d_brier` as **comparator − model**
(positive = primary better); the prose below and `report.md` §9 flip the sign
explicitly where a signed Δ is reported. See `results/README.md`.

## Frozen evidence in one paragraph

The pre-registered primary is a 49-feature logistic model evaluated on a
one-shot, unit-fresh 20% dyad holdout (311,638 dyad-months; 32,614 positives;
5,282 directed dyads; 2018-01–2022-11; not time-fresh). It achieved log-loss
0.217875, AUC 0.910152 and AP 0.671863. Against the train-only per-dyad Markov
baseline (log-loss 0.265874), log-loss was lower by 0.047999 (Δ = −0.047999;
18.05%; 95% multiway CI −0.052665 to −0.043333; negative = better). Against the
stronger warm-history Markov baseline (log-loss 0.209729; ECE 0.010228), the
primary improved AUC by 0.015979 and AP by 0.035016, but its log-loss was worse
by 0.008146 (Δ = +0.008146; 95% CI +0.002334 to +0.013958; positive = worse) and
its ECE was worse by 0.031883 (0.042112 vs 0.010228). That contrast is
**unequal in two directions**: warm history has 35 additional months of outcomes
(through 2017-11) that the model does not see, while the model carries the
49-feature actor-role and graph block that the baseline does not see; neither
side is dominated. The pre-registered
29-feature secondary (log-loss 0.205219, ECE 0.027463, AUC 0.908290, AP 0.671747)
beat the warm-history baseline on log-loss by 0.004510 (Δ = −0.004510; 95% CI
−0.007597 to −0.001423; secondary, not in a BH family); its use for dyad-level
probability ranking is **post-hoc**, not a new confirmatory test and not a
primary swap. In the pre-registered 49-feature contract, under the
**holdout-zeroed** graph cache, the 20 graph features do not improve held-out
log-loss. The confirmatory evaluation and the stability audit use distinct cache
policies: the confirmatory numbers are **holdout-zeroed** (held-out edges zeroed
for every month, including the training snapshots), whereas the audit is
**holdout-row-excluded** (eventual holdout rows removed from fitting and
aggregation). These are different cache policies for different estimands, not
interchangeable runs. The train-only
statistics block is constant on the holdout; removing it is an intercept/
recalibration effect, not a transfer result. The graph effect reverses when the
train-statistics block is absent, so this is not a general claim about graph
features; the warm-history comparison is likewise not a general claim about
persistence beyond this panel. The stability audit is provisional: its grid was
frozen during the confirmatory run (after the read began, before it completed),
it excludes all eventual holdout dyads at row level (holdout-row-excluded) and
uses 50 refits; it is
not a holdout-side stability test and not a pre-holdout design.
The public `experiments/` scripts are a five-seed preview and do not reproduce
those 50-refit estimates. We make no node-level, causal, lifecycle or
state-of-the-art claim.

## Data and license

ICEWS v37 is available from the Harvard Dataverse
(doi:10.7910/DVN/28075) under research-and-teaching terms. This repository does
not redistribute the raw or derived event panel, row-level predictions or
labels, actor-level predicted probabilities or observed outcome rates. Source
code is released under the MIT license, **code only** (see `LICENSE`). The
aggregate results
include ICEWS-derived quantities (for example actor-level aggregate attribution
and a rendered, aggregate-only world-line preview) and are **not** MIT: they
remain subject to the ICEWS terms and are provided for research and teaching
only. See `NOTICE`, `DATA_EXCLUSIONS.md` and the “Derived-data terms” section of
`results/README.md`.

## Citation and status

Citation metadata is in `CITATION.cff` and `.zenodo.json`. This is a v0.1.1
release candidate under review: the `v0.1.1` git tag and the DOI are pending, so
cite the repository URL and this candidate until they exist; production replaces
the review-candidate label with the `v0.1.1` tag and DOI. Author metadata
(individual names, affiliations, ORCIDs) is **pending** and is not invented
here. The stability audit is
provisional (holdout-row-excluded, 50 refits); the public `experiments/` tree is a
five-seed preview. The manuscript is under review and is not part of this
repository.
