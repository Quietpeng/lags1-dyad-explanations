# World-line preview — illustrative only

**Status:** `ILLUSTRATIVE PREVIEW - NOT VALIDATED; NO MULTI-STEP ACCURACY CLAIM.`

> **Print version (2026-10-26).** `worldline_preview.pdf` is **byte-identical
> to the paper's Figure A1** (`paper/figures/figA1_worldline.pdf`): vector,
> 425.2 × 330 pt (150 mm), Arimo, TrueType fonts, no baked-in key. The PNG is
> the same render at 300 dpi. It was rendered from the restricted per-row trace
> exports and is **not regenerable from this tree**; `experiments/worldline_preview.py`
> ships for inspection only. The underlying experiment was not rerun. The
> superseded DejaVu print version with a baked key is gone; see `CORRECTIONS.md`.

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
| `worldline_preview.png` / `.pdf` | Three-panel print figure: sampled dyad paths for two anchors, induced actor states for 2016-06, and transition tallies over four anchors. |
| `worldline_trace.json` | Metadata, disclaimers, per-anchor counts, state-path and transition tallies, per-anchor `p1_summary`/`p2_summary` distributions (`mean`, `median`, `min`, `max` of the step-1 and step-2 predicted probabilities over the 20 sampled, unlabeled dyads; no per-dyad value), model provenance, sample rule and the no-row-level rule. No per-row records. |
| `run_manifest.json` | Command, host/package versions, input/output SHA256, runtime, figure revision, preview scope and the withheld-export list. |

**No CSV is shipped.** The 240-row `worldline_trace.csv` (per-row `state`,
`p_model`) and the 249-row `node_trace.csv` (per-row induced actor state) were
removed from the public tree. `experiments/worldline_preview.py` writes them
only when `--emit-row-traces` is passed explicitly for a local debug run; such
files must never be committed.

## Release boundary

The public release ships **no row-level prediction or per-row probability
file**. Only the rendered illustration (PNG/PDF) and the per-anchor/aggregate
summary blocks in `worldline_trace.json` are distributed; see
`DATA_EXCLUSIONS.md`. The preview is an inner-period (2016-01..2017-11)
illustrative sample only: 20 sampled dyads per anchor, 71 distinct dyads over
four anchors, no 2018–22 holdout row, no accuracy computed. The figure keeps
ISO3 labels for readability; those labels are not join keys and no probability
value is exported. `scripts/verify_release.py` fails if either removed CSV
reappears, and `make worldline-check` asserts that this directory contains no
CSV, that the manifest covers only the rendered/aggregate outputs, and that the
anchor summaries stay inside the inner period. Deleting this directory changes
no main result.

## Reproduce

The script uses the project venv and `matplotlib` (installed for this preview;
`Agg` backend, CPU only). It requires the licensed ICEWS panel and local feature
caches; those inputs are not redistributed.

```bash
cd <repo>
PY=python3
$PY experiments/worldline_preview.py --threads 1
```

Quick smoke (one anchor, two sources):

```bash
$PY experiments/worldline_preview.py \
  --only-anchor 2016-06 --n-sources 2 --n-active 2 --n-inactive 1 \
  --threads 1 --out-dir build/worldline_smoke
```

## What would be needed to validate a rolling claim

1. An intermediate-month feature-generating model (or a joint state/feature
   model), not a state-lag-only approximation.
2. A rolling-origin evaluation with a prospective period and a pre-registered
   multi-step metric.
3. Uncertainty intervals over sampled paths and a comparison with persistence
   and Markov baselines.

None of the three is present here. The preview is deletable without affecting
any main result.
