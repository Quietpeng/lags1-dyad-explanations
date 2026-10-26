# Release notes

## Metadata, docs and CI fix (2026-10-27; no data change)

Documentation, metadata, tests and CI only: no result file, figure, notebook
blob, `SHA256SUMS` entry or frozen number changed.

- **Python floor is now 3.12.** `numpy==2.5.4` and `scipy==1.18.1` publish no
  cp311 wheels, so `requires-python` and the classifier, `scripts/doctor.py`
  (`MIN_PYTHON`), `README.md`, `REPRODUCING.md`, `notebooks/README.md` and the
  CI matrix (`3.12`, `3.13`) agree on `>= 3.12`; the failing 3.11 CI leg is
  removed.
- **Stale figure geometry corrected.** `CODE_MAP.md` and `notebooks/README.md`
  no longer advertise a 164 mm / 465x380 pt preview; they state the shipped
  150 mm (425.197 x 330 pt) paper-A1 render. `make verify` and the contract
  tests now guard both documents.
- **World-line inventory completed.** `results/README.md`,
  `DATA_EXCLUSIONS.md`, `results/worldline_preview/README.md` and
  `run_manifest.json` name the retained per-anchor `p1_summary`/`p2_summary`
  distributions (mean, median, min and max over the 20 sampled dyads; no
  per-dyad value).
- **`make dist` archives HEAD only.** The temporary-index `read-tree HEAD` +
  `add -A` recipe is replaced by `git archive HEAD`, whose file list must equal
  `git ls-tree -r --name-only HEAD`; untracked files never ship and a modified
  tracked file aborts the build. The extract-and-verify step is unchanged.
  `tests/test_metadata_contract.py` pins these contracts.

## v0.1.1 — public-artifact cleanup (release candidate)

- Install path uses `python3` throughout; `make doctor` checks Python,
  dependencies and forbidden paths.
- `make test`, `make verify`, `make notebooks`, `make stability-check`,
  `make worldline-check`, `make dist` and `make clean` replace the old
  `paper`/`worldline`/`repro` targets. `make dist` uses `git archive` and
  verifies the archive.
- Notebooks are executed headlessly and ship aggregate-only outputs. Notebook
  01 no longer raises on an empty results tree and uses the shared Morandi
  palette.
- The stability report/summaries carry a release banner: the grid was frozen
  after the confirmatory read began and before it completed, the holdout was
  not integrated, and the numbers are provisional. The shipped numbers are the
  holdout-excluded 50-refit rerun, not a holdout-side test. No stability number
  was changed.
- `experiments/` is the earlier five-refit audit code (five seed replicates),
  not the restricted 50-refit variant. `experiments/STABILITY_VARIANT.md`
  publishes the exact five deltas: seed count, holdout-dyad exclusion, fixed C=1.0, bootstrap
  replicates and cache enumeration. The public scripts cannot reproduce the
  shipped aggregates exactly and do not claim to. They refuse to write into
  `results/framework_pilot/stability/` unless `NEUROSIM_ALLOW_OVERWRITE=1` (or
  `--allow-overwrite`) is set; `NEUROSIM_STABILITY_OUT` redirects output to a
  preview directory.
- Added `results/framework_pilot/stability/SHA256SUMS` (47 files) and hardened
  `make stability-check` / `make verify`: status, 50 refits, C=1.0,
  `frozen_during_confirmatory_run_holdout_rows_excluded` (grid frozen
  06:31:40Z, after the confirmatory read began at 05:26:55Z and before it
  completed at 07:45:19Z), 5,282/2,790 exclusions, 50-replicate bootstrap,
  stability hashes, `STABILITY_VARIANT.md` presence and rejection of the
  superseded pre-holdout status label and five-seed audit claims.
- Public data exclusion (release gate): removed the 3,450-row
  `dyad_contributions_inner_sample.csv`; dropped `mean_p`/`p_sum`/`y_sum`/
  `mean_y` from the thirteen actor tables and from `summary.json`; de-identified
  the world-line CSV exports. `DATA_EXCLUSIONS.md` records the removed paths,
  columns and reasons; `make verify` / `make stability-check` /
  `make worldline-check` and the contract tests now fail if a row-level
  prediction/label file or a named actor probability/outcome rate reappears.
  No frozen number changed.
- Results provenance: `results/README.md`, `holdout/SHA256SUMS`, a repo-relative
  pre-registration checksum, `redistributed: false` annotations for the four
  excluded prediction/log files, and `scripts/verify_release.py`.
- The shipped package contains the paper path only. Legacy audit/graph/
  interpret/operators/pipeline/reporting/run_pilot/synth/synth_arms/targets/
  windows modules and the restricted snapshot scripts were removed from the
  public tree (git history retains them).
- Added `pyproject.toml`, `requirements.lock`, `CODE_MAP.md`,
  `REPRODUCING.md`, `.gitattributes`, `.github/workflows/ci.yml` and stricter
  ignore rules.
- Metadata: v0.1.1 release candidate; no `v0.1.1` git tag and no DOI yet,
  provisional stability audit, code MIT but ICEWS-derived aggregates not MIT.
- JSON results are strict (`NaN`/`Infinity` replaced by `null` where a value is
  undefined). No metric, prediction or model output changed.
- The world-line preview is retained as an illustrative artifact and is now a
  **byte-identical copy of the paper's Figure A1** (vector, 425.2 × 330 pt /
  150 mm, Arimo, no baked key); the PNG is the same render at 300 dpi. It is
  rendered from restricted trace exports and is not regenerable from this tree.
  The underlying experiment was not rerun.
- R2C3 artifact-integrity pass (no frozen number changed): the stability
  `audit_design` literal is
  `frozen_during_confirmatory_run_holdout_rows_excluded` with the confirmatory
  start time recorded; family-B / within-between p-values below numerical
  resolution are reported as bounds (`<1e-300`, `<0.001`) with an explicit
  `censoring` field instead of `0.0`; the absolute-attribution top-10 overlap
  range in the report generator is 0.00–0.33 (log-occlusion stays 0.05–0.43).
  See `results/framework_pilot/holdout/CORRECTIONS.md`.
- Title, abstract, stability status and world-line figure wording were aligned
  with the final manuscript before this release candidate was frozen. Only
  post-freeze annotations and prose changed; no frozen metric, table, prediction
  or model output was changed.
- BH family D `p`/`q` were recomputed from the frozen normal-tail `p_normal`
  column of `feature_group_loo.csv` (Benjamini–Hochberg over nine groups). The
  old rows were bootstrap tails floored at `p=0.0`; the new rows are the
  normal-approximation values, all `q < 1e-12`, all rejected. Families A–C and
  every ΔLL/CI are unchanged.
- The world-line per-row exports (`worldline_trace.csv`, `node_trace.csv`) were
  removed. Only the rendered PNG/PDF and the per-anchor/aggregate JSON ship, so
  the release contains no row-level probability file; `make verify` and
  `make worldline-check` fail if either CSV reappears.

## v0.1.0 — first curated release

This was a curated public release of the code and non-restricted aggregate
artifacts. Local absolute paths in copied reports/scripts were replaced with
`<repo>`; no metric, prediction or model output was changed.

Not included: ICEWS raw/derived data, row-level predictions/labels, the
manuscript, internal project documents, internal configuration files, and the
old hierarchy pilot notebooks.
