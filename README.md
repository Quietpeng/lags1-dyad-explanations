# Actor-side components of dyadic event forecasts: an ownership decomposition, specification dependence, and a matched-vintage audit in a 196-actor ICEWS panel

**an ownership decomposition, specification dependence, and a matched-vintage audit in a 196-actor ICEWS panel.**

Code, tests and aggregate results for the **ownership decomposition**,
**specification-dependence analysis** and **matched-vintage audit** of the V2
manuscript: a post-hoc six-class ownership convention over a plan-declared
feature list that fixes one owner per feature, unit and sign; exact additive
source- and target-role components for binary dyad-month political event
forecasts; a post-hoc matched-vintage audit of the three asymmetries in the
pre-specified comparison; and a separate within-panel stability audit. The
deliverable is a contract plus audit, not a general-purpose framework and not a
new estimator: the decomposition is the
logistic identity and coincides with linear SHAP under a training-mean
background.

The ownership rule is an analysis convention applied after the fact, not a
preregistered object: no pre-run record of it exists. The feature list is
plan-declared and internally hash-timestamped, but only the split carries an
independently written pre-run timestamp. The V2 recomputation on the
5.06 M-row training window gives 45.2% actor-owned and 54.8% not actor-owned
(the maximum of a 36.6–54.8% allocation range; 63.2% if the dyad-specific
neighbour-activity terms leave the actor class). The V1 6.34 M-row fit gave
44.9/55.1 and 62.3; the difference is at most 1.2 percentage points. These
shares are descriptive outputs of the convention, not properties of the
actors. Only 4 of the 49 features are role-specific: six of the seven
source/target column pairs are one function for the same actor (five
undirected graph terms plus the log in-plus-out ego count).

The public artifact is an **L0–L2 verification release**: reading the frozen
aggregates, verifying their hashes and contracts, and re-rendering the
notebooks. It does not regenerate the restricted experiment pipeline: the
50-refit audit code is restricted and not redistributed, and row-level
predictions and labels are withheld. The aggregate results do not redistribute
event rows and remain subject to the ICEWS terms. The manuscript is not part of
this repository.

**Release status:** this is the **`v0.1.1` archival release** of the V2
artifact. The annotated `v0.1.1` tag points at the commit that contains this README;
`main` is fast-forwarded to the same commit when the release is pushed. The metadata version is `0.1.1`; **no DOI has been
minted**, so cite the repository URL and the `v0.1.1` tag until a DOI exists.
`v0.1.1` carries the V2 aggregate tree
(`results/framework_pilot_v2/`, aggregates only) alongside the V1 confirmatory
and stability aggregates.

The earlier review snapshot `review-r2c13` at commit `1c3c8ae` remains
**published** and fetchable; `review-r2c12` and `review-r2c10` (commit
`95aa07b`) are earlier published snapshots, so `review-r2c10`, `review-r2c12`
and `review-r2c13` are public. `review-r2c14` at commit `2d27421`
is a **local text-only revision that has not been pushed or published**: no
remote `review-r2c14` tag, branch or commit exists, and its aggregate tree is
byte-identical to `review-r2c13`. It is retained in local history below the
`v0.1.1` commit, not advertised as a public release. `review-r2c9` and
`review-r2c11` are local only. The public review-tag inventory is `review-r2c2`
… `review-r2c8` plus `review-r2c10`, `review-r2c12` and `review-r2c13`; none of
those review tags is the archival release, and `v0.1.1` supersedes them as the
version of record. The confirmatory holdout is frozen. The stability audit is **provisional**: it was recomputed
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
- `experiments/` — scripts for the plan-declared, internally hash-timestamped
  one-shot confirmatory holdout, a five-seed preview of the stability audit and
  the illustrative world-line preview. They require the licensed ICEWS panel and
  local caches; post-hoc holdout diagnostics additionally need row-level
  predictions. None of those inputs is redistributed. The Sprint-B preview does
  not regenerate the shipped
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
  report, summary, tables, the frozen plan (`preregistration.json`), manifest
  and access log. Row-level
  predictions and run logs are not redistributed.
- `results/framework_pilot_v2/` — **V2 aggregate tree**: the post-hoc
  matched-vintage audit (`tables/arms_contrasts.csv`, `arms_main.csv`,
  `a7_cache_matrix.csv`, `a8_*.csv`, `a9_*.csv`, `mde.csv`), the machine-readable
  `summary.json` / `v2_manifest.json`, the provenance note and a `SHA256SUMS`
  manifest. Row-level predictions, fitted coefficients and caches are withheld
  (see `DATA_EXCLUSIONS.md`). V2 is a post-hoc secondary analysis of a burned
  holdout; the V1 49-feature result remains the plan-declared primary of record.
- `results/framework_pilot/stability/` — **provisional** stability audit on
  inner-validation and burned-retrospective periods, with a `SHA256SUMS`
  manifest covering every distributed stability file. Actor tables are
  aggregate attribution/exposure only; no predicted probabilities or observed
  outcome rates are shipped.
- `results/worldline_preview/` — illustrative two-step preview; not a validated
  multi-step forecast. It is a standalone, repo-only illustration, not a
  manuscript exhibit: no paper figure or appendix is derived from it. The
  sample is four anchors × 20 directed dyads (71 distinct), non-random, from
  the inner-validation period 2016-01–2017-11; the manuscript cites this
  directory once as a repository-only illustration and analyses nothing here,
  so no paper result or number derives from it. The shipped PDF is a 150 mm
  vector render (Arimo/TrueType); the PNG is a 300 dpi
  preview. Only the rendered figure and aggregate JSON summaries ship: the
  per-row trace CSVs were removed, so the directory carries no row-level
  probability file (see `DATA_EXCLUSIONS.md`).
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
The absolute-path scan matches four prefixes — `/home`, `/tmp`, `/Users` and
`C:` user directories — so a copied local path fails the release even when the
rest of the text is clean; the credential scan covers AWS key ids, GitHub and
OpenAI token shapes, private-key blocks and inline secret assignments. No
restricted input is required by either target. The CFF is schema-validated in
CI with `cffconvert --validate -i CITATION.cff`; it carries the MIT code licence
and keeps the ICEWS aggregate carve-out.

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
`make notebooks` also leaves the tracked notebook JSON unchanged — the executor
warms the matplotlib font cache first — so `make notebooks && make dist` works
in one checkout on a cold machine.

## Reproduction boundary

`make doctor`, `make test`, `make verify`, `make stability-check`,
`make worldline-check` and `make notebooks` run from this tree without
restricted inputs. Regenerating the aggregate tables (L3) is not executable
here: it needs licensed ICEWS v37, the restricted derived feature and
contribution caches, the local experiment configuration, and for post-holdout
diagnostics the excluded row-level predictions. The public tree has **no
cache-build entry point**, so licensed access alone is not sufficient. The
confirmatory holdout (L4) is closed by a one-shot lock (`run_once.json` plus
`test_access_log.json`); re-running it would violate the one-shot lock and the
plan's split commitment.
See `REPRODUCING.md` for the L0–L4 matrix.

## Sign convention

**Δ = model − comparator.** For log-loss, Brier and ECE a **negative** Δ is
better; for AUC and AP a **positive** Δ is better. The frozen tables under
`results/` store `d_log_loss`, `d_ece` and `d_brier` as **comparator − model**
(positive = primary better); the prose below and `report.md` §9 flip the sign
explicitly where a signed Δ is reported. See `results/README.md`.

## Frozen evidence in one paragraph

The plan-declared primary is a 49-feature logistic model evaluated on a
one-shot, unit-fresh 20% dyad holdout (311,638 dyad-months; 32,614 positives;
5,282 directed dyads; 2018-01–2022-11; not time-fresh). The plan is internally
hash-timestamped; only the split carries an independently written pre-run
timestamp. It achieved log-loss
0.217875, AUC 0.910152 and AP 0.671863. Against the train-only per-dyad Markov
baseline (log-loss 0.265874), log-loss was lower by 0.047999 (Δ = −0.047999;
18.05%; 95% multiway CI −0.052665 to −0.043333; negative = better). Against the
stronger warm-history Markov baseline (log-loss 0.209729; ECE 0.010228), the
primary improved AUC by 0.015979 and AP by 0.035016, but its log-loss was worse
by 0.008146 (Δ = +0.008146; 95% CI +0.002334 to +0.013958; positive = worse) and
its ECE was worse by 0.031884 (0.042112 vs 0.010228). That contrast is
**unequal in two directions**: warm history has 35 additional months of outcomes
(through 2017-11) that the model does not see, while the model carries the
49-feature actor-role and graph block that the baseline does not see; neither
side is dominated. The plan-declared
29-feature secondary (log-loss 0.205219, ECE 0.027463, AUC 0.908290, AP 0.671747)
beat the warm-history baseline on log-loss by 0.004510 (Δ = −0.004510; 95% CI
−0.007597 to −0.001423; secondary, not in a BH family); its use for dyad-level
probability ranking is **post-hoc**, not a new confirmatory test and not a
primary swap. The 29-vs-warm contrast itself is **post hoc, nominal and
unadjusted**: the plan declared the 29-feature model, not that comparison. In
the plan-declared 49-feature contract, under the
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

## V2 matched-vintage audit

V2 is a **post-hoc secondary analysis on the same burned holdout**; it cannot
upgrade the plan-declared V1 result and did not re-open the holdout. It removes
the three asymmetries one at a time: the two future-inclusive train-only columns
(`br`, `seen_tr`), the 35-month vintage advantage of the warm comparator, and the
H0 cache. All arms fix `C = T = 1.0`; the matched warm comparator uses a 2014-11
cutoff (`t < 239`).

- **A0 replication:** 49@H0 vs the original 2017-vintage warm baseline loses
  **−0.008235** (95% CI −0.014198 to −0.002272; p = 6.8e−3), 8.9e−5 from the V1
  −0.008146.
- **A2 primary (post hoc):** 47@P1 beats matched persistence by **+0.020954**
  (95% CI 0.015443 to 0.026465; q = 1.4e−13).
- **A5 key secondary:** the 25-feature model with no actor-owned column beats
  matched persistence by **+0.017743** (95% CI 0.013118 to 0.022368;
  q = 1.1e−13), 84.7% of the A2 gain.
- **Nested 47 vs 25:** the 22 added columns contribute **+0.003211**
  (95% CI 0.001772 to 0.004650; q = 1.6e−5), 15.3% of the gain. They are 12
  actor-owned graph terms, 2 ego counts and 8 symmetric graph terms; the
  contrast cannot separate actor from graph content.
- **Cache factor:** material only for the 49-feature fit (+0.014783,
  p = 1.3e−16); null for 47 (−0.000017, p = 0.96) and mechanically zero for 25.
- **Claim boundary:** the as-specified 49 loses only under H0 against the
  2017-vintage comparator. The supported statement is that the 47-feature
  contract beats matched persistence and the nested contrast excludes zero; it
  is not a claim that actor-role content drives the gain.

`results/framework_pilot_v2/` ships the aggregate tables, summaries, manifests
and provenance note behind these numbers. The V2 fit-stage certification is
behavioural (the frozen current script reproduces all 10 fits bit-for-bit), not
archival; Appendix B of the manuscript states this.

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

Citation metadata is in `CITATION.cff` and `.zenodo.json`. This is the
**`v0.1.1` archival release**: the annotated `v0.1.1` tag points at the commit
that contains this README. **No DOI has been minted**, so cite the repository
URL and the `v0.1.1` tag until one exists; a DOI, when minted, will be added to
`CITATION.cff`, `.zenodo.json` and the manuscript's data-availability
statement. The single author is **Jingzhe Peng**, Independent researcher,
ORCID `0009-0001-7037-6244`, corresponding e-mail
`pengjingzhe_quiet@foxmail.com`; the same values are in `CITATION.cff`,
`.zenodo.json` and the manuscript front matter. The stability audit is
provisional (holdout-row-excluded, 50 refits); the public `experiments/` tree is
a five-seed preview. The manuscript is a separate V2 candidate and is not part
of this repository. The public ref state is recorded under **Release status**
above: `main` and `v0.1.1` are the current release; `review-r2c13` is the
preceding published review snapshot; `review-r2c12` and `review-r2c10` are
earlier published snapshots; `review-r2c14`, `review-r2c9` and `review-r2c11`
remain local and unpublished; there is no DOI.
