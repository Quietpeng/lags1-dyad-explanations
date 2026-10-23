# Actor-level explanations of dyadic event forecasts

Code, tests, and non-restricted aggregate results for a reproducible framework
for binary dyad-month political event prediction, exact additive source- and
target-actor explanations, and a stability audit. The accompanying manuscript is
not included in this repository.

## What is here

- `framework/` — CPU-only Python library for the panel, feature contract, model
  ladder, exact additive decomposition, actor views, metrics and bootstrap
  inference.
- `experiments/` — scripts for the pre-registered confirmatory holdout, the
  stability audit and the illustrative world-line preview.
- `tests/` — 14 unit and smoke tests (`python -m unittest discover -s tests -v`).
- `notebooks/` — read-only notebooks that load the frozen aggregate artifacts;
  no ICEWS data, model fitting or row-level predictions.
- `results/framework_pilot/holdout/` — one-shot unit-fresh 20% dyad-holdout
  report, summary, tables, pre-registration and access log. Row-level
  predictions are not redistributed.
- `results/framework_pilot/stability/` — provisional stability audit on
  inner-validation and burned-retrospective periods.
- `results/worldline_preview/` — illustrative two-step preview; not a validated
  multi-step forecast.
- `scripts/` — leakage-audit and manifest-verification utilities.

## What is not included

- ICEWS raw or derived event data (research-and-teaching terms; not
  redistributable) and row-level predictions/labels.
- The manuscript, internal project documents, and internal configuration files.
- The old hierarchy/Laya notebooks and the synthetic/T1 arms; they are not part
  of this paper.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

To read the frozen results without installing Jupyter:

```bash
python - <<'PY'
import json
s = json.load(open("results/framework_pilot/holdout/summary.json"))
print(s["status"])
print(s["primary"]["metrics"])
PY
```

If Jupyter is available, open `notebooks/` and run the notebooks from the
repository root. They are read-only and CPU-only.

## Frozen evidence in one paragraph

The pre-registered primary is a 49-feature logistic model evaluated on a
one-shot, unit-fresh 20% dyad holdout (311,638 dyad-months; 32,614 positives;
5,282 directed dyads; 2018-01–2022-11; not time-fresh). It achieved log-loss
0.217875, AUC 0.910152 and AP 0.671863. Against the per-dyad Markov baseline it
reduced log-loss by 0.047999 (18.05%; 95% multiway CI 0.043333–0.052665), but
against a stronger warm-history Markov baseline it improved AUC by 0.015979
while losing on log-loss (−0.008146) and calibration (ECE 0.042112 vs 0.010228).
The pre-registered 29-feature secondary improves log-loss over the warm-history
baseline by 0.004510 (95% CI 0.001423–0.007597); it is a post-hoc candidate for
dyad-level probability ranking, not a deployment recommendation for actor-level
explanation or month-level timing, and it has not been given a new confirmatory
test. The 20 graph features and the train-only statistics block do not transfer
to the holdout. The stability audit is provisional and was not re-run on the
holdout. We make no node-level, causal or state-of-the-art claim.

## Data and license

ICEWS v37 is available from the Harvard Dataverse
(doi:10.7910/DVN/28075) under research-and-teaching terms. This repository does
not redistribute the raw or derived event panel. Source code is released under
the MIT license; see `LICENSE` and `NOTICE`.

## Citation and status

Citation metadata is in `CITATION.cff`. The repository is a v0.1.0 release
candidate; a DOI will be added after archiving. The manuscript is under
review and is not part of this repository.
