# Actor-side components of dyadic event forecasts: an ownership decomposition, specification dependence, and a matched-vintage audit in a 196-actor ICEWS panel

Code, tests and aggregate results for the ownership decomposition,
specification-dependence analysis and matched-vintage audit of the V2
manuscript. The analysis applies a post-hoc six-class ownership convention to a
plan-declared feature list, fixing one owner per feature and the unit and sign.
It yields exact additive source- and target-role components for binary
dyad-month political event forecasts, a post-hoc matched-vintage audit of the
three asymmetries in the pre-specified comparison, and a separate within-panel
stability audit. What ships is a contract plus audit. It is not a
general-purpose framework or a new estimator: the decomposition is the logistic
identity and coincides with linear SHAP under a training-mean background.

The ownership rule is an analysis convention applied after the fact; no pre-run
record of it exists. The feature list is plan-declared and internally
hash-timestamped, but only the split carries an independently written pre-run
timestamp. On the 5.06 M-row V2 training window the recomputation gives 45.2%
actor-owned and 54.8% not actor-owned (the maximum of a 36.6–54.8% allocation
range; 63.2% if the dyad-specific neighbour-activity terms leave the actor
class). The V1 6.34 M-row fit gave 44.9/55.1 and 62.3; the difference is at most
1.2 percentage points. These shares describe the output of the convention; they
are not properties of the actors. Only 4 of the 49 features are role-specific:
six of the seven source/target column pairs are one function for the same actor
(five undirected graph terms plus the log in-plus-out ego count).

The public artifact is an L0–L2 verification release: it supports reading the
frozen aggregates, verifying their hashes and contracts, and re-rendering the
notebooks. It does not regenerate the restricted experiment pipeline. The
50-refit audit code is restricted and not redistributed, and row-level
predictions and labels are withheld. The aggregate results do not redistribute
event rows and remain subject to the ICEWS terms.

**Paper:** arXiv link (to be added).

**Paper and code are separate.** This repository contains code and aggregate
results only. It does not contain the paper's PDF, LaTeX/Markdown source,
figures, bibliography, raw or derived ICEWS rows, row-level predictions, a
`docs/`, `configs/` or `data/` tree, or any `paper/` directory. The paper is
cited by its arXiv link (to be added); no arXiv identifier exists yet, there is
no DOI and no Zenodo DOI, and the single author is an independent researcher
with no institutional affiliation. The code is the `v0.1.2` release at commit
`70d6d78` (annotated tag object `af229d3`). No Zenodo DOI has been minted for
the code or the paper.

**Release status:** `v0.1.2` at `70d6d78` is the published code release and the
snapshot the paper cites. The annotated tag is at that commit; `main` carries
this docs-only follow-up, which changes no code or aggregate bytes. It carries the V2 aggregate tree (`results/framework_pilot_v2/`,
aggregates only) alongside the V1 confirmatory and stability aggregates, and
adds the V2 code (`experiments/sprint_c_v2*.py`) and notebooks 04–05. The
previous release, `v0.1.1` (annotated tag `69f5400` → commit `13b96db`), is
retained as history and is not the current release. The packaged metadata
version (`pyproject.toml`, the `CITATION.cff` `version` field, `.zenodo.json`)
stays `0.1.1` because no packaged byte changed between `13b96db` and `70d6d78`;
the git tag is the release identifier. No DOI is claimed, and the paper's arXiv
identifier will be added to this README, `CITATION.cff` and `.zenodo.json` when
one is assigned.

The earlier review snapshot `review-r2c13` at commit `1c3c8ae` remains
published and fetchable; `review-r2c12` and `review-r2c10` (commit `95aa07b`)
are earlier published snapshots, so `review-r2c10`, `review-r2c12` and
`review-r2c13` are public. None of those review tags is the archival release.
`review-r2c9`, `review-r2c11` and the `review-r2c14` review tag are local only:
the `review-r2c14` tag object at `2d27421` has not been pushed or published,
and no remote `review-r2c14` ref exists. Commit `2d27421` is an ancestor of the
published `v0.1.2` release, so its text is contained in `70d6d78`; only the
review tag name remains local.

The confirmatory holdout is frozen. The stability audit is provisional: it was
recomputed after excluding all eventual holdout dyads from fitting and
aggregation, with C fixed at the primary value and 50 dyad-cluster bootstrap
refits. The confirmatory holdout completed later and was not used. The shipped
`experiments/` Sprint-B scripts are a five-seed preview, not the restricted
50-refit audit; see `experiments/STABILITY_VARIANT.md`.

## What is here

- `framework/`: CPU-only Python library implementing the paper path: the
  dyad-month feature contract, the primary/secondary/capacity model set (a flat
  comparator list, not a ladder), calibration, exact additive decomposition,
  actor views, metrics and cluster-bootstrap inference. The package contains
  the paper path only; the directory and import name are retained for
  compatibility and do not imply a reusable framework.
- `experiments/`: the historical V1 scripts (`sprint_a_*`: the plan-declared,
  internally hash-timestamped one-shot confirmatory holdout; `sprint_b_*`: the
  five-seed stability preview; `worldline_preview.py`: the illustrative
  preview) and the V2 matched-vintage runners `sprint_c_v2.py` /
  `sprint_c_v2_fit_repro.py`. All of them require the licensed ICEWS panel and
  local caches; post-hoc holdout diagnostics additionally need row-level
  predictions. None of those inputs is redistributed, and the V2 runner also
  needs the frozen protocol and fails closed without it. The Sprint-B preview
  does not regenerate the shipped 50-refit, holdout-excluded aggregates;
  `experiments/STABILITY_VARIANT.md` lists the exact five differences. The
  preview refuses to write into `results/framework_pilot/stability/` unless
  `NEUROSIM_ALLOW_OVERWRITE=1` (or `--allow-overwrite`) is set;
  `NEUROSIM_STABILITY_OUT` redirects output to a preview directory. See the
  "V2 code and notebooks" section below for what the V2 ports can and cannot
  do.
- `tests/`: unit and contract tests for features, splits, models, attribution,
  frozen results and notebooks. They run without data, configuration or cache.
- `notebooks/`: five executed, read-only notebooks that display the frozen
  aggregate artifacts. Notebooks 01–03 cover the V1 holdout, stability audit and
  world-line preview; 04–05 cover the V2 ownership decomposition and the
  training pipeline. They do not fit models on real data, read the ICEWS event
  panel or load row-level predictions; notebook 05 runs the shipped estimator
  on synthetic arrays only.
- `results/framework_pilot/holdout/`: one-shot unit-fresh 20% dyad-holdout
  report, summary, tables, the frozen plan (`preregistration.json`), manifest
  and access log. Row-level predictions and run logs are not redistributed.
- `results/framework_pilot_v2/`: the V2 aggregate tree. It holds the post-hoc
  matched-vintage audit (`tables/arms_contrasts.csv`, `arms_main.csv`,
  `a7_cache_matrix.csv`, `a8_*.csv`, `a9_*.csv`, `mde.csv`), the
  machine-readable `summary.json` / `v2_manifest.json`, the provenance note and
  a `SHA256SUMS` manifest. Row-level predictions, fitted coefficients and
  caches are withheld (see `DATA_EXCLUSIONS.md`). V2 is a post-hoc secondary
  analysis of a burned holdout; the V1 49-feature result remains the
  plan-declared primary of record.
- `results/framework_pilot/stability/`: the provisional stability audit on
  inner-validation and burned-retrospective periods, with a `SHA256SUMS`
  manifest covering every distributed stability file. Actor tables are
  aggregate attribution/exposure only; no predicted probabilities or observed
  outcome rates are shipped.
- `results/worldline_preview/`: illustrative two-step preview, not a validated
  multi-step forecast. It is a standalone, repo-only illustration and not a
  manuscript exhibit; no paper figure or appendix is derived from it. The
  sample is four anchors × 20 directed dyads (71 distinct), non-random, from
  the inner-validation period 2016-01–2017-11. The manuscript cites this
  directory once as a repository-only illustration and analyses nothing here,
  so no paper result or number derives from it. The shipped PDF is a 150 mm
  vector render (Arimo/TrueType); the PNG is a 300 dpi preview. Only the
  rendered figure and aggregate JSON summaries ship. The per-row trace CSVs
  were removed, so the directory carries no row-level probability file (see
  `DATA_EXCLUSIONS.md`).
- `scripts/`: release verifier, aggregate checker, notebook executor and
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
The absolute-path scan matches four prefixes (`/home`, `/tmp`, `/Users` and
`C:` user directories), so a copied local path fails the release even when the
rest of the text is clean. The credential scan covers AWS key ids, GitHub and
OpenAI token shapes, private-key blocks and inline secret assignments. Neither
target requires a restricted input. The CFF is schema-validated in CI with
`cffconvert --validate -i CITATION.cff`; it carries the MIT code licence and
keeps the ICEWS aggregate carve-out.

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
`make notebooks` also leaves the tracked notebook JSON unchanged, because the
executor warms the matplotlib font cache first, so `make notebooks && make dist`
works in one checkout on a cold machine.

## Reproduction boundary

`make doctor`, `make test`, `make verify`, `make stability-check`,
`make worldline-check` and `make notebooks` run from this tree without
restricted inputs. Regenerating the aggregate tables (L3) is not executable
here: it needs licensed ICEWS v37, the restricted derived feature and
contribution caches, the local experiment configuration, and for post-holdout
diagnostics the excluded row-level predictions. The public tree has no
cache-build entry point, so licensed access alone is not sufficient. The
confirmatory holdout (L4) is closed by a one-shot lock (`run_once.json` plus
`test_access_log.json`); re-running it would violate the one-shot lock and the
plan's split commitment. See `REPRODUCING.md` for the L0–L4 matrix.

## Sign convention

**Δ = model − comparator.** For log-loss, Brier and ECE a negative Δ is better;
for AUC and AP a positive Δ is better. The frozen tables under `results/` store
`d_log_loss`, `d_ece` and `d_brier` as comparator − model (positive = primary
better); the prose below and `report.md` §9 flip the sign explicitly where a
signed Δ is reported. See `results/README.md`.

## Frozen evidence in one paragraph

The plan-declared primary is a 49-feature logistic model evaluated on a
one-shot, unit-fresh 20% dyad holdout (311,638 dyad-months; 32,614 positives;
5,282 directed dyads; 2018-01–2022-11; not time-fresh). The plan is internally
hash-timestamped; only the split carries an independently written pre-run
timestamp. The model achieved log-loss 0.217875, AUC 0.910152 and AP 0.671863.
Against the train-only per-dyad Markov baseline (log-loss 0.265874), log-loss
was lower by 0.047999 (Δ = −0.047999; 18.05%; 95% multiway CI −0.052665 to
−0.043333; negative = better). Against the stronger warm-history Markov
baseline (log-loss 0.209729; ECE 0.010228), the primary improved AUC by 0.015979
and AP by 0.035016, but its log-loss was worse by 0.008146 (Δ = +0.008146; 95%
CI +0.002334 to +0.013958; positive = worse), and its ECE was worse by 0.031884
(0.042112 vs 0.010228). The contrast is unequal in two directions: warm history
has 35 additional months of outcomes (through 2017-11) that the model does not
see, while the model carries the 49-feature actor-role and graph block that the
baseline does not see. Neither side is dominated.

The plan-declared 29-feature secondary (log-loss 0.205219, ECE 0.027463, AUC
0.908290, AP 0.671747) beat the warm-history baseline on log-loss by 0.004510
(Δ = −0.004510; 95% CI −0.007597 to −0.001423; secondary, not in a BH family).
Its use for dyad-level probability ranking is post-hoc; it is not a new
confirmatory test or a primary swap. The 29-vs-warm contrast is itself post hoc,
nominal and unadjusted: the plan declared the 29-feature model, not that
comparison.

In the plan-declared 49-feature contract, under the holdout-zeroed graph cache,
the 20 graph features do not improve held-out log-loss. The confirmatory
evaluation and the stability audit use distinct cache policies. The
confirmatory numbers are holdout-zeroed (held-out edges zeroed for every month,
including the training snapshots), whereas the audit is holdout-row-excluded
(eventual holdout rows removed from fitting and aggregation). The two policies
serve different estimands, and the runs are not interchangeable. The train-only
statistics block is constant on the holdout; removing it is an intercept and
recalibration effect, not a transfer result. The graph effect reverses when the
train-statistics block is absent, so this is not a general claim about graph
features. The warm-history comparison is likewise not a general claim about
persistence beyond this panel.

The stability audit is provisional. Its grid was frozen during the confirmatory
run (after the read began, before it completed), it excludes all eventual
holdout dyads at row level (holdout-row-excluded) and uses 50 refits. It is not
a holdout-side stability test or a pre-holdout design. The public
`experiments/` scripts are a five-seed preview and do not reproduce those
50-refit estimates. No node-level, causal, lifecycle or state-of-the-art claim
is made.

## V2 matched-vintage audit

V2 is a post-hoc secondary analysis on the same burned holdout. It cannot
upgrade the plan-declared V1 result and did not re-open the holdout. It removes
the three asymmetries one at a time: the two future-inclusive train-only columns
(`br`, `seen_tr`), the 35-month vintage advantage of the warm comparator, and
the H0 cache. All arms fix `C = T = 1.0`; the matched warm comparator uses a
2014-11 cutoff (`t < 239`).

- `A0` replication: 49@H0 vs the original 2017-vintage warm baseline loses
  −0.008235 (95% CI −0.014198 to −0.002272; p = 6.8e−3), 8.9e−5 from the V1
  −0.008146.
- `A2` primary (post hoc): 47@P1 beats matched persistence by +0.020954
  (95% CI 0.015443 to 0.026465; q = 1.4e−13).
- `A5` key secondary: the 25-feature model with no actor-owned column beats
  matched persistence by +0.017743 (95% CI 0.013118 to 0.022368;
  q = 1.1e−13), 84.7% of the A2 gain.
- Nested 47 vs 25: the 22 added columns contribute +0.003211
  (95% CI 0.001772 to 0.004650; q = 1.6e−5), 15.3% of the gain. They are 12
  actor-owned graph terms, 2 ego counts and 8 symmetric graph terms; the
  contrast cannot separate actor from graph content.
- Cache factor: material only for the 49-feature fit (+0.014783,
  p = 1.3e−16); null for 47 (−0.000017, p = 0.96) and mechanically zero for 25.
- Claim boundary: the as-specified 49 loses only under H0 against the
  2017-vintage comparator. The supported statement is that the 47-feature
  contract beats matched persistence and the nested contrast excludes zero; it
  is not a claim that actor-role content drives the gain.

`results/framework_pilot_v2/` ships the aggregate tables, summaries, manifests
and provenance note behind these numbers. The V2 fit-stage certification is
behavioural (the frozen current script reproduces all 10 fits bit-for-bit), not
archival; Appendix B of the manuscript states this.

## V2 code and notebooks

The V2 code sits in the public tree next to the V1 scripts:

| File | What it is | Runnable here? |
|---|---|---|
| `experiments/sprint_c_v2.py` | The V2 matched-vintage runner behind `results/framework_pilot_v2/`: stages `selftest`, `fits_p1`, `fits_h0`, `compare`, `a8`, `a9`, `report`; 10 fits (F1–F9, D1), 14 canonical contrasts, the A8 ownership recomputation and the A9 50-refit stability pass. | No. |
| `experiments/sprint_c_v2_fit_repro.py` | The archival code copy the fit stage deposited beside the frozen results. Its body is byte-identical to `sprint_c_v2.py`; only the header differs, and `tests/test_smoke.py` enforces that. | No. |

They cannot run here. The aggregate-only release does not redistribute the
licensed ICEWS v37 panel, the restricted feature/graph caches, the local
experiment configuration, the frozen protocol (`reviews/round2/v2_protocol.md`)
or the row-level predictions read by the `compare`, `a8` and `a9` stages.
`verify_protocol()` is therefore the first gate in every stage and fails closed
with an explicit message instead of building a partial run. No runnable path is
faked: point `NEUROSIM_V2_PROTOCOL_DIR` at a licensed copy of the frozen
protocol and supply the licensed caches to re-enable it. The `v0.1.2` port
changes the header and two input paths only. The protocol directory and the
scratch directory are environment variables (`NEUROSIM_V2_PROTOCOL_DIR`,
`NEUROSIM_V2_SCRATCH`) rather than host paths. No fit, contrast, table or
number changed, so the shipped file necessarily hashes differently from the
pre-port digest recorded in `results/framework_pilot_v2/manifest.json`
(`a8f83417dfc7670740571ff19b3df8a65031e92dfd3459b6b7faaedd22bc0843`); the
aggregate tree and its `SHA256SUMS` remain the authoritative record.

The V2 notebooks are read-only and aggregate-only:

- `notebooks/04_exact_attribution.ipynb`: the exact additive decomposition and
  its 1.15e-14 round-trip, the six ownership classes and the seven-class view
  (55.1 / 62.3 in V1, 54.8 / 63.2 recomputed), the 36.6–54.8% allocation range,
  source/target near-collinearity (5 duplicated graph pairs, 4 role-unique
  columns, 12 of 14 partner-blind), top-k churn and the decision-lift panel.
- `notebooks/05_training_pipeline.ipynb`: the feature contract (27 + 20 + 2 →
  49 / 47 / 29 / 25), the split and embargo arithmetic, matched-vintage versus
  original-vintage warm comparators, `C = 1.0` logistic training and
  calibration, the P1/H0 cache factor and the A0–A7 arms. The mechanics are
  re-run on synthetic arrays only and labelled as such; the real panel is
  restricted.

The V1 results remain the record. The V1 scripts are labelled historical in the
sense that they produced the V1 confirmatory and stability artifacts and are
not the V2 audit runner. The V1 results remain the plan-declared primary of
record, and V2 cannot upgrade them. Both notebook sets read frozen aggregates,
and neither regenerates a paper figure. `paper/` is not part of this repository,
and the paper is cited by its arXiv link (to be added).

## Data and license

ICEWS v37 is available from the Harvard Dataverse
(doi:10.7910/DVN/28075) under research-and-teaching terms. This repository does
not redistribute the raw or derived event panel, row-level predictions or
labels, actor-level predicted probabilities or observed outcome rates. The
source code is released under the MIT license, code only (see `LICENSE`). The
aggregate results include ICEWS-derived quantities (for example actor-level
aggregate attribution and a rendered, aggregate-only world-line preview) and
are not MIT-licensed. They remain subject to the ICEWS terms and are provided
for research and teaching only. See `NOTICE`, `DATA_EXCLUSIONS.md` and the
"Derived-data terms" section of `results/README.md`.

## Citation and status

Citation metadata is in `CITATION.cff` and `.zenodo.json`. This is the `v0.1.2`
code release: the annotated `v0.1.2` tag (object `af229d3`) points at commit
`70d6d78`; this README is the docs-only follow-up on `main`; the tagged code commit is unchanged. No
DOI has been minted for the code or the paper, so cite the repository URL and
the `v0.1.2` tag; a DOI, when minted, will be added to `CITATION.cff`,
`.zenodo.json` and the manuscript's data-availability statement. The `v0.1.1`
archival release (tag `v0.1.1` → `13b96db`) is the previous release and is
superseded by `v0.1.2`. The single author is Jingzhe Peng, Independent
researcher, ORCID `0009-0001-7037-6244`, corresponding e-mail
`pengjingzhe_quiet@foxmail.com`; the same values are in `CITATION.cff`,
`.zenodo.json` and the manuscript front matter. The stability audit is
provisional (holdout-row-excluded, 50 refits); the public `experiments/` tree is
a five-seed preview. The paper is a separate document: no paper PDF, source,
figures, raw data or row-level predictions are included here. Cite the paper by
its arXiv link (to be added) and the code as `v0.1.2` at commit `70d6d78`.
`CITATION.cff` carries a `preferred-citation` entry for the paper (title,
author, year, arXiv URL placeholder, no DOI, no arXiv identifier) and
`.zenodo.json` a `related_identifiers` entry. The public ref state is recorded
under Release status above: `v0.1.2` is the current code release at
`70d6d78`; `main` carries the docs-only follow-up. `review-r2c13` is the preceding published review snapshot;
`review-r2c12` and `review-r2c10` are earlier published snapshots; the
`review-r2c14`, `review-r2c9` and `review-r2c11` review tags remain local and
unpublished; there is no DOI.
