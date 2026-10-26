# Notebooks — actor-level explanations of dyadic event forecasts

Read-only, CPU-only companion notebooks for:

> **Actor-role components of dyadic event forecasts: exact attribution and a portability audit.**

The notebooks display frozen aggregate artifacts. They do **not** fit models, read the ICEWS event panel, or load row-level predictions. Some shipped aggregates contain actor-level **aggregate attribution and exposure** (not predicted probabilities or observed outcome rates); those are ICEWS-derived and remain under the ICEWS terms. The world-line preview ships no row-level trace: only the rendered figure and aggregate summaries. See `NOTICE`, `DATA_EXCLUSIONS.md` and `results/README.md`.

All three notebooks are committed **with executed outputs** (aggregate-only). `make notebooks` re-executes them headlessly from the repository root.

## Contents

| Notebook | Purpose | Evidential status |
|---|---|---|
| `01_confirmatory_holdout.ipynb` | Rebuild Table 2 Panels A/C, the strict-mean vs pair-weighted within-dyad contrast, the graph/train-statistics LOO result and the baseline forest from the one-shot unit-fresh 20% dyad holdout (2018-01–2022-11). | Confirmatory, but retrospective: unit-fresh, **not** time-fresh. |
| `02_stability_audit.ipynb` | Display the exactness check, actor-bucket shares, source/target redundancy, variance decomposition, top-k Jaccard, period/window contrasts, decision proxy, and HGB occlusion. | **Provisional**: holdout-excluded 50-refit audit on inner 2016–17 plus burned 2018–22/COVID views; grid frozen mid-run (after the confirmatory read began, before it completed); the holdout is not integrated with it. The public `experiments/` tree is a five-seed preview and does not regenerate these aggregates. |
| `03_worldline_preview.ipynb` | Display the two-step illustrative world-line preview, per-anchor and aggregate transition counts, and induced actor states (no row-level trace). | **Illustrative only**; not validated; no multi-step accuracy claim; deletable. |

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

`make notebooks` executes all three notebooks with `MPLBACKEND=Agg`, writes the outputs in place, and fails if an execution error, an absolute path or a credential-like string appears.

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
- `results/worldline_preview/worldline_trace.json` (per-anchor/aggregate summaries) and `worldline_preview.png`; no row-level trace file ships

They do not read the ICEWS event panel and do not load row-level predictions. The aggregates they do read contain ICEWS-derived aggregate attribution; see the derived-data terms in `results/README.md`. No node-level prediction is made or claimed.

## Relation to the manuscript

- **Notebook 01** corresponds to the confirmatory holdout section, Table 2 Panels A/C, and Figure 3. The manuscript figure is **not** redistributed because `paper/` is excluded. Panel (a)'s primary contrasts are reconstructed as a forest from `tables/primary_comparisons.csv`; the graph/train-statistics part of panel (b) comes from the LOO table; panel (c) from the within-dyad table. The forest uses the shared Morandi palette (slate `#35525F`, terracotta `#A2664F`, sage `#84957A`, ochre `#C2A05F`) with distinct markers.
- **Notebook 02** corresponds to the stability audit, Table 3, and Figure 2. Figure 2 is not redistributed; the notebook displays the underlying variance, churn, decision, and occlusion tables instead.
- **Notebook 03** corresponds to Appendix A1 / Figure A1. Its PDF and PNG live under `results/worldline_preview/`, so they are included in the public release. The shipped PDF is the vector, 425.197 × 330 pt (150 mm) print version with TrueType fonts; the PNG is a 300 dpi preview. The paper's Appendix A1 uses the same data and caption.

The notebooks do not regenerate the manuscript PDF or any LaTeX artifact.

## Explicitly not included

- Earlier hierarchy/synthetic pilot notebooks. They are not copied or adapted here.
- Model fitting, hyper-parameter tuning, calibration fitting, or bootstrap refits.
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

Run `make notebooks` for a full headless execution. `make verify` additionally checks that the committed outputs are present, error-free, aggregate-only, and free of absolute paths.
