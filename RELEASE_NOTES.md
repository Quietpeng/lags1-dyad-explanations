# Release notes

## Public-text revision (R2C7; metadata, tests and documentation)

Text, metadata and tests only; no analysis, refit, holdout rerun or number
change. Freeze hashes stay in `manifest.json#/post_run_corrections`; the shipped
bytes are tracked by `outputs`/`current_sha256` and `SHA256SUMS`.

- `holdout/report.md` labels the strict within-dyad column **pair-weighted**
  (0.5616) and states the strict equal-dyad mean (0.5139); 0.5266 remains the
  distinct P1-centred value. No value changed.
- `holdout/CORRECTIONS.md` adds the table convention (comparator − model;
  positive = secondary better) and the manuscript-equivalent 29-feature
  contrast (`−0.004510 [−0.00760, −0.00142]`). The frozen `+0.004510` is
  unchanged.
- Comparator and cache wording: the previous one-sided comparator label is
  replaced by “unequal in two directions”, with the **holdout-zeroed**
  confirmatory cache named separately from the **holdout-row-excluded** audit.
- Year-fixed-effects status: the anchor is **plan-declared**; its promotion and
  warm-history contrast are post-hoc, C-confounded and uncorrected; it remains
  “not a prospective model” with the linear-extrapolation limitation.
- Deposit metadata: `CITATION.cff` licence is `other-open`, matching
  `.zenodo.json`; `pyproject.toml` and `LICENSE` keep MIT **code only**, and the
  aggregates are explicitly not MIT. Author metadata stays pending; no DOI, tag
  or ORCID is invented.
- `tables/dyad_fe.csv` marks the non-converged conditional-logit row with
  `converged=false` and `used_as_evidence=false`; values are unchanged.
- `results/README.md` maps raw-panel support (6,344,160 / 317,208 / 311,638) to
  the filtered rows (5,055,328 / 232,672 / 311,638), marks the raw counts and the
  67,057 inner positives manuscript-only, and records the dual sign convention.
  Table 1 is not reproduced or changed.
- The metadata contract is folded into the existing 15-check verifier
  (CFF/Zenodo licence equality, code-only MIT carve-out, pending author
  metadata, title equality, nine required test modules and a 123-test floor).
  New assertions extend `tests/test_metadata_contract.py` and
  `tests/test_results_contract.py`; the three git-only dist checks skip in
  archive mode.

## Manifest provenance repair (2026-10-27; metadata, tests and documentation)

No result, figure, notebook output, data file or row-level policy changed.

- `holdout/manifest.json:release_annotations.code_annotations` now records the
  shipped SHA-256 of all six `manifest.code` files (the freeze-time digests stay
  in `code`), names the three post-freeze files under `changed_since_freeze`,
  and adds the `p_display` entry for the censored-p inference code.
- `scripts/verify_release.py` gained a `manifest code hashes` check (14 → 15
  checks), and `scripts/check_aggregates.py` applies the same rule in both
  `stability-check` and `worldline-check`. Contract tests cover a tampered code
  hash and a tampered row-level column.
- Documentation corrections: notebook 01 cites Figure 4
  (`fig4_holdout.pdf`), `DATA_EXCLUSIONS.md` names the current review tree,
  `CODE_MAP.md` and these notes no longer claim the quarantined history is
  public, the nested stability-summary wording is explicit, and the freeze
  `figures_manifest` normalization is documented in `results/README.md`.
- CI runs `make stability-check` and `make worldline-check`; the notebook
  metadata test asserts the declared 3.12 floor instead of a fixed interpreter.

## Public title alignment (2026-10-27; documentation and metadata only)

No result, figure, notebook, manifest hash, `SHA256SUMS` entry, data file or
test semantics changed.

- **Public title follows the manuscript.** `CITATION.cff`, `.zenodo.json`,
  `README.md`, `notebooks/README.md`, `REPRODUCING.md` and the
  `pyproject.toml` description now carry *Actor-role components of dyadic event
  forecasts: exact attribution, near-collinearity, and a within-panel stability
  audit*.
- **Audit label corrected in public prose.** Public text now names only the
  within-panel stability audit; `CODE_MAP.md` uses the same label. No numerical
  claim, scope statement or evidential status changed.
- Version `0.1.1`, the L0–L2 executable boundary, the five-seed preview
  disclosure, the restricted 50-refit audit and the withheld row-level data are
  unchanged. No `v0.1.1` tag and no DOI exist.

## Public-label alignment (2026-10-27; documentation and metadata only)

Documentation, citation metadata and one module docstring only: no result,
figure, notebook, manifest hash, `SHA256SUMS` entry or test changed, and no
frozen number moved.

- **The deliverable is labelled as the manuscript labels it.** `CITATION.cff`,
  `.zenodo.json` and `README.md` now describe an actor-role **attribution
  contract plus audit** and an **L0–L2 verification release**; neither they nor
  the package docstring advertise a framework. All three state that the
  decomposition is the logistic identity, so it is not a new estimator, and
  that the release is one paper path rather than reusable abstractions.
- **Claim boundaries are explicit in the public metadata.** The citation
  abstract, the Zenodo description and the README state that no general claim
  is made about graph features, persistence or any corpus beyond this panel;
  that the restricted 50-refit audit code is not redistributed (the public
  `experiments/` scripts remain a five-seed preview); and that row-level
  predictions and labels are withheld.
- **Package labels follow.** `CODE_MAP.md` and `REPRODUCING.md` state that
  `framework/` is a retained import path for the one paper path, not a
  general-purpose framework. The `framework/__init__.py` docstring says the
  same. The directory name is unchanged, so imports, tests and archives are
  unaffected.
- Title, version `0.1.1`, the L0–L2 boundary, the five-seed preview
  disclosure, and the absence of a `v0.1.1` tag and DOI are unchanged.

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
  public tree; the quarantined history that contains them is not part of this
  public release.
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

## v0.1.0 — first curated release (no git tag)

This milestone is recorded in these notes only: no `v0.1.0` tag exists in the
public repository, and the tag inventory is `review-r2c2` … `review-r2c6`.

This was a curated public release of the code and non-restricted aggregate
artifacts. Local absolute paths in copied reports/scripts were replaced with
`<repo>`; no metric, prediction or model output was changed.

Not included: ICEWS raw/derived data, row-level predictions/labels, the
manuscript, internal project documents, internal configuration files, and the
old hierarchy pilot notebooks.
