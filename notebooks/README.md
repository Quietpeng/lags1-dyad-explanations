# Notebooks — actor-side components of dyadic event forecasts

Read-only, CPU-only companion notebooks for:

> **Actor-side components of dyadic event forecasts: an ownership decomposition, specification dependence, and a matched-vintage audit in a 196-actor ICEWS panel**

The notebooks display frozen aggregate artifacts. They do **not** fit models on the event panel, read the ICEWS panel, or load row-level predictions; notebook 05 fits the shipped estimator on **synthetic arrays** only, as a labelled mechanics demo. Some shipped aggregates contain actor-level **aggregate attribution and exposure** (not predicted probabilities or observed outcome rates); those are ICEWS-derived and remain under the ICEWS terms. The world-line preview ships no row-level trace: only the rendered figure and aggregate summaries. See `NOTICE`, `DATA_EXCLUSIONS.md` and `results/README.md`.

All five notebooks are committed **with executed outputs** (aggregate-only). `make notebooks` re-executes them headlessly from the repository root.

## Contents

| Notebook | Purpose | Evidential status |
|---|---|---|
| `01_confirmatory_holdout.ipynb` | Rebuild Table 2 Panels A/C, the strict-mean vs pair-weighted within-dyad contrast, the graph/train-statistics LOO result and the baseline forest from the one-shot unit-fresh 20% dyad holdout (2018-01–2022-11). | Confirmatory, but retrospective: unit-fresh, **not** time-fresh. |
| `02_stability_audit.ipynb` | Display the exactness check, actor-bucket shares, source/target redundancy, variance decomposition, top-k Jaccard, period/window contrasts, decision proxy, and HGB occlusion. | **Provisional**: holdout-excluded 50-refit audit on inner 2016–17 plus burned 2018–22/COVID views; grid frozen mid-run (after the confirmatory read began, before it completed); the holdout is not integrated with it. The public `experiments/` tree is a five-seed preview and does not regenerate these aggregates. |
| `03_worldline_preview.ipynb` | Display the two-step illustrative world-line preview, per-anchor and aggregate transition counts, and induced actor states (no row-level trace). | **Illustrative only**; not validated; no multi-step accuracy claim; deletable. |
| `04_exact_attribution.ipynb` | The exact additive decomposition (`b + Σ c = logit p`, max abs error 1.15e-14 over 50,000 rows), the six ownership classes and the seven-class `g_nra` view (55.1/62.3 in V1, 54.8/63.2 recomputed), the allocation range, source/target near-collinearity, top-k churn and the decision-lift panel. Reads only V2 aggregate tables plus the provisional stability tables. | **Post-hoc secondary** on a burned holdout; the shares are outputs of a convention, not properties of the actors; A9 is provisional. |
| `05_training_pipeline.ipynb` | The feature contract (27 + 20 + 2 → 49/47/29/25), the split and embargo arithmetic, matched-vintage versus original-vintage warm comparators, `C = 1.0` logistic training and calibration, the P1/H0 cache factor and the A0–A7 arms. Re-runs the mechanics on **synthetic arrays** only. | **Method display**, not evidence: the real ICEWS v37 panel is restricted and is not read here. |

## Quick start

Python 3.12+ is supported. The release was tested with Python 3.13.5 and:

| Package | Tested version |
|---|---|
| numpy | 2.5.4 |
| pandas | 2.3.3 |
| scikit-learn | 1.9.1 |
| scipy | 1.18.1 |
| matplotlib | 3.11.2 |
| duckdb | 1.5.6 |
| nbclient | 0.11.0 |
| nbformat | 5.11.1 |

```bash
cd <repo-root>
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install "nbclient>=0.11" "nbformat>=5.11" "ipykernel>=7.0" "jupyterlab>=4.4"
make notebooks          # headless execution from the repository root
jupyter lab notebooks/  # optional interactive viewing
```

`make notebooks` executes all five notebooks with `MPLBACKEND=Agg`, writes the outputs in place, and fails if an execution error, an absolute path or a credential-like string appears.

### Working directory

Launch Jupyter from the repository root (recommended), or from `notebooks/`. The first code cell of each notebook walks up from the current working directory until it finds `results/framework_pilot/`. If it cannot find that directory, the notebook prints `[missing] ...` messages instead of raising; notebook 01 also guards the empty comparison-table case.

All paths in notebook code are relative to the detected repository root. No absolute home paths are used, and the first cell prints the root as a relative path.

## Data and release boundaries

The public artifact release intentionally excludes:

- `data/` — ICEWS and other local inputs; their licences forbid redistribution.
- `configs/`, `docs/`, `paper/`, and internal reviews.
- Row-level predictions, actor-level predicted probabilities/outcome rates, and caches (`results/**/preds/`, `results/**/cache/`, `results/**/*.npz`), logs, and the raw event panel.

The notebooks read only:

- `results/framework_pilot/holdout/summary.json` and `holdout/tables/*.csv`
- `results/framework_pilot/stability/summary.json` and `stability/tables/*.csv`
- `results/framework_pilot_v2/summary.json`, `v2/tables/*.csv` and `v2/parts/a8_ownership.json` (the ownership block; aggregate only)
- `results/worldline_preview/worldline_trace.json` (per-anchor/aggregate summaries) and `worldline_preview.png`; no row-level trace file ships

They do not read the ICEWS event panel and do not load row-level predictions. The aggregates they do read contain ICEWS-derived aggregate attribution; see the derived-data terms in `results/README.md`. No node-level prediction is made or claimed.

## Relation to the manuscript

- **Notebook 01** corresponds to the confirmatory holdout section, Table 2 Panels A/C, and Figure 2 (`paper/figures/fig2_holdout.pdf`). The manuscript figure is **not** redistributed because `paper/` is excluded. Panel (a)'s primary contrasts are reconstructed as a forest from `tables/primary_comparisons.csv`; the graph/train-statistics part of panel (b) comes from the LOO table; panel (c) from the within-dyad table. The forest uses the shared Morandi palette (slate `#35525F`, terracotta `#A2664F`, sage `#84957A`, ochre `#C2A05F`) with distinct markers.
- **Notebook 02** corresponds to the stability audit, Table 3, and Figure 3 (`paper/figures/fig3_stability.pdf`). Figure 3 is not redistributed; the notebook displays the underlying variance, churn, decision, and occlusion tables instead.
- **Notebooks 04 and 05** correspond to the V2 ownership decomposition and the
  matched-vintage audit code path. Neither regenerates a paper figure or a
  manuscript table; they display the frozen aggregates and, for 05, the pipeline
  mechanics on synthetic arrays. The V2 runner itself
  (`experiments/sprint_c_v2.py`) is not runnable from this tree — it needs the
  licensed panel, the restricted caches and the frozen protocol, and fails
  closed without them (see `REPRODUCING.md` L3 and `DATA_EXCLUSIONS.md` §6–§7).
- **Notebook 03** re-displays the standalone, repo-only world-line preview. Its PDF and PNG live under `results/worldline_preview/`, so they are included in the public release. The shipped PDF is a vector 425.197 × 330 pt (150 mm) print version with TrueType fonts; the PNG is a 300 dpi preview. The preview is an illustration only: four anchors × 20 sampled dyads (71 distinct), non-random, inner-validation 2016-01–2017-11. The manuscript cites this directory once as a repository-only illustration and analyses nothing here; it is not a manuscript exhibit, not a paper result, and no paper figure or appendix is derived from it.

The notebooks do not regenerate the manuscript PDF or any LaTeX artifact.

## Explicitly not included

- Earlier hierarchy/synthetic pilot notebooks. They are not copied or adapted here.
- Model fitting on real data, hyper-parameter tuning, calibration fitting, or
  bootstrap refits. Notebook 05 fits the shipped estimator on synthetic arrays
  only, as a mechanics demo, and says so in the cell.
- New confirmatory tests. Notebook 01 re-displays a one-shot holdout; the labels were read once by the experiment script.
- Claims beyond the frozen artifacts: no causal claim, no actor-lifecycle claim, no node-level forecast claim, and no validated multi-step timing claim.

## Validation

The notebooks are plain JSON (`nbformat` 4.x) and can be validated without Jupyter:

```bash
python3 - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("notebooks").glob("*.ipynb")):
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(c["execution_count"] is not None for c in code)
    print(path, "OK", len(nb["cells"]), "cells")
PY
```

Run `make notebooks` for a full headless execution. `make verify` additionally checks that the committed outputs are present, error-free and path-clean (executed code cells, no error output, no absolute home/temporary path, embedded PNGs where expected). It does not assert an aggregate-only property; that boundary is enforced by the shipped file list, the row-level verifier checks and `DATA_EXCLUSIONS.md`.
