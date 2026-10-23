# Notebooks

Read-only, CPU-only notebooks for the frozen aggregate artifacts. They do not
read ICEWS data, refit models, or use row-level predictions.

| Notebook | Content |
|---|---|
| `01_confirmatory_holdout.ipynb` | One-shot unit-fresh 20% dyad holdout: primary/secondary metrics, warm-history comparison, within-dyad definitions and feature-block transfer. |
| `02_stability_audit.ipynb` | Provisional stability audit: variance decomposition, top-k churn, decision proxy and HGB occlusion agreement. |
| `03_worldline_preview.ipynb` | Illustrative two-step world-line preview; not validated and no multi-step accuracy claim. |

Run Jupyter from the repository root so the relative `results/` paths resolve.
If Jupyter is not installed, the notebooks are still readable as JSON and the
same tables can be loaded with pandas from `results/`.
