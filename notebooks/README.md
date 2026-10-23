# Notebooks — Q2 dyadic-forecast framework

Read-only, CPU-only companion notebooks for:

> **Actor-level explanations of dyadic event forecasts: an open-source framework with stability auditing.**

The notebooks display frozen aggregate artifacts. They do **not** fit models, read ICEWS data, or load row-level predictions.

## Contents

| Notebook | Purpose | Evidential status |
|---|---|---|
| `01_confirmatory_holdout.ipynb` | Rebuild Table 2 Panels A/C, the strict-mean vs pair-weighted within-dyad contrast, and the graph/train-statistics LOO result from the one-shot unit-fresh 20% dyad holdout (2018-01–2022-11). | Confirmatory, but retrospective: unit-fresh, **not** time-fresh. |
| `02_stability_audit.ipynb` | Display the exactness check, actor-bucket shares, source/target redundancy, variance decomposition, top-k Jaccard, period/window contrasts, decision proxy, and HGB occlusion. | **Provisional**: inner 2016–17 plus burned 2018–22/COVID views; not re-run on the holdout. |
| `03_worldline_preview.ipynb` | Display the two-step illustrative world-line preview, its trace excerpt, transition counts, and induced actor states. | **Illustrative only**; not validated; no multi-step accuracy claim; deletable. |

## Quick start

Python 3.11+ is assumed.

```bash
cd <repo-root>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install jupyterlab
jupyter lab notebooks/
```

The analysis stack is in `requirements.txt`; `jupyterlab` is an optional viewing dependency and is not pinned there.

### Working directory

Launch Jupyter from the repository root (recommended), or from `notebooks/`. The first code cell of each notebook walks up from the current working directory until it finds `results/framework_pilot/`. If it cannot find that directory, the notebook prints `[missing] ...` messages instead of raising.

All paths in notebook code are relative to the detected repository root. No absolute home paths are used.

## Data and release boundaries

The public artifact release intentionally excludes:

- `data/` — ICEWS and other local inputs; their licenses forbid redistribution.
- `configs/`, `docs/`, `paper/`, and internal reviews.
- Row-level predictions and caches (`results/**/preds/`, `results/**/cache/`, `results/**/*.npz`).

The notebooks read only:

- `results/framework_pilot/holdout/summary.json` and `results/framework_pilot/holdout/tables/*.csv`
- `results/framework_pilot/stability/summary.json` and `results/framework_pilot/stability/tables/*.csv`
- `results/worldline_preview/worldline_trace.json`, `worldline_trace.csv`, `node_trace.csv`, and `worldline_preview.png`

No ICEWS rows are read or redistributed. No node-level prediction is made or claimed.

## Relation to the manuscript

- **Notebook 01** corresponds to the confirmatory holdout section, Table 2 Panels A/C, and Figure 3. The manuscript figure is **not** redistributed because `paper/` is excluded. Panel (a)'s primary contrasts are reconstructed as a forest from `tables/primary_comparisons.csv`; the graph/train-statistics part of panel (b) comes from the LOO table; panel (c) from the within-dyad table.
- **Notebook 02** corresponds to the stability audit, Table 3, and Figure 2. Figure 2 is not redistributed; the notebook displays the underlying variance, churn, decision, and occlusion tables instead.
- **Notebook 03** corresponds to Appendix A1 / Figure A1. Its PNG lives under `results/worldline_preview/`, so it is included in the public release.

The notebooks do not regenerate the manuscript PDF or any LaTeX artifact.

## Explicitly not included

- The old `neurosim/notebooks/hierarchy_*` notebooks from the stopped hierarchy project. They are not copied or adapted here.
- Model fitting, hyper-parameter tuning, calibration fitting, or bootstrap refits.
- New confirmatory tests. Notebook 01 re-displays a one-shot holdout; the labels were read once by the experiment script.
- Claims beyond the frozen artifacts: no causal claim, no actor-lifecycle claim, no node-level forecast claim, and no validated multi-step timing claim.

## Validation

The notebooks are plain JSON (`nbformat` 4.5) and can be validated without Jupyter:

```bash
python - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("notebooks").glob("*.ipynb")):
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    print(path, "OK", len(nb["cells"]), "cells")
PY
```

To execute them, install the viewing stack and run Jupyter from the repository root. A headless check can use `MPLBACKEND=Agg`; notebook 01 renders its forest from a CSV, and notebook 03 displays the world-line PNG.
