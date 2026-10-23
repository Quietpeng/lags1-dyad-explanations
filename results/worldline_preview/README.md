# World-line preview — illustrative only

**Status:** `ILLUSTRATIVE PREVIEW - NOT VALIDATED; NO MULTI-STEP ACCURACY CLAIM.`

This directory is a **communication and debugging artifact** for the paper's
Appendix A1, not a main result and not a validated multi-step forecast. It uses
only the frozen inner-validation period (2016-01..2017-11); no 2018–22 test
label is read. No accuracy is computed.

## What is shown

- Four anchors: **2016-06, 2016-12, 2017-06, 2017-09** (`t = 257, 263, 269,
  272`; the last step is 2017-11, before the 2017-12 embargo row).
- Per anchor, a small **visibility sample** of 20 directed dyads: five source
  actors at evenly spaced ranks of the active-source distribution; per source,
  up to three currently active dyads (mostly close to p = 0.5, plus one
  persistent high-p dyad) and one inactive high-p dyad. The sample is **not
  random** and supports no prevalence or accuracy inference.
- **Step 0:** observed anchor state.
- **Step 1:** state sampled from the frozen one-step logistic model
  (`p(t+1)`); features are observed at the anchor.
- **Step 2:** state sampled from a **state-lag-only approximation**
  (`p(t+2)`). Active/lag/rolling/history/calendar terms are updated from the
  sampled state. Graph, reverse-dyad, actor-ego and `br`/`seen_tr` terms are
  frozen at the anchor; event magnitude is fixed at the anchor value (or the
  train mean positive event count when the anchor is zero). This is **not** a
  generative feature model.
- **Node views:** source- and target-actor states induced by the sampled dyad
  subset only. The number in each node cell is the count of active sampled
  dyads, not full-network activity. Actor states are explanation-side
  projections, not node-level predictions.

## Files

| File | Content |
|---|---|
| `worldline_preview.png` / `.pdf` | Figure: two representative anchors (dyad ribbons + induced actor ribbons) and per-window transition counts aggregated over four anchors. |
| `worldline_trace.csv` | One row per anchor × sampled dyad × step: state, state origin, p, sample role, anchor/simulated event count. |
| `node_trace.csv` | One row per anchor × role × actor × step: active state, active-dyad count, transition. |
| `worldline_trace.json` | Metadata, disclaimers, anchor summaries, per-anchor and aggregate transition matrices, model provenance, sample rule. |
| `run_manifest.json` | Command, host/package versions, input/output SHA256, runtime. |

## Reproduce

The script uses the project venv and `matplotlib` (installed for this preview;
`Agg` backend, CPU only).

```bash
cd <repo>
PY=python
$PY experiments/worldline_preview.py --threads 1
```

Quick smoke (one anchor, two sources):

```bash
$PY experiments/worldline_preview.py \
  --only-anchor 2016-06 --n-sources 2 --n-active 2 --n-inactive 1 \
  --threads 1 --out-dir /tmp/opencodeprogram/worldline_smoke
```

## Figure caption (use in Appendix A1)

> **Illustrative world-line preview, not a validated forecast.** One sampled
> two-step path for a small, deliberately selected set of dyads at two
> representative anchors (2016-06, 2017-06) from the 2016–17 inner-validation
> period; the trace covers all four anchors. Step 1 uses the frozen one-step
> model. Step 2 uses a state-lag-only approximation with graph, reverse-dyad,
> actor-ego and train-only features frozen at the anchor. Cell values are
> p(t+1)/p(t+2) for dyads and active-dyad counts for induced actor states.
> Actor states are projections of the sampled dyad subset, not node-level
> predictions. The sample is selected for visibility, not random. No test
> labels were read and no multi-step accuracy is claimed.

## What would be needed to validate a rolling claim

1. An intermediate-month feature-generating model (or a joint state/feature
   model) that produces the full feature vector at t+1.
2. Out-of-sample multi-step scoring at h = 1, 2, ... against persistence,
   global-transition and per-dyad Markov baselines, with paired cluster
   bootstrap intervals.
3. A pre-registered error-accumulation analysis and a stopping rule for when
   the rollout is no longer informative.
4. A separate pre-registration for the multi-step endpoint; the current
   confirmatory endpoint is one-step dyad activity on the fresh 20% dyad
   holdout.
5. Multiple sampled paths and path-level uncertainty, not one seed.

## Deletion

This artifact is deletable. Remove this directory, the Appendix A1 entry, the
§6.5 discussion paragraph and `experiments/worldline_preview.py`; no main
result, table, figure or abstract sentence depends on it.
