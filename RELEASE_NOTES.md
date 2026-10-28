# Release notes

## Public-artifact revision (R2C13; provenance, world-line disclosure and doc truth)

Documentation, metadata, verifier and tests only; no analysis, refit, label
read, plan edit, data change or aggregate edit. The frozen plan
`preregistration.json` stays byte-identical (`41bda0e3…`); no metric, interval,
sign, timestamp, split, class share or scope literal changed. The title is
unchanged and is the T1 string:

> *Actor-role components of dyadic event forecasts: shared graph columns,
> near-collinearity in a 196-actor panel, and a within-panel stability audit*

- **Published snapshot.** `main` and annotated tag `review-r2c13` carry the
  commit that contains these notes as the **published review snapshot**: a
  frozen, fetchable record of the reviewed aggregate tree, not an archival
  release and not a DOI-bearing version. `review-r2c12` is the previous
  published snapshot and `review-r2c10` remains fetchable at commit `95aa07b`,
  so `review-r2c10`, `review-r2c12` and `review-r2c13` are public.
  `review-r2c9` and `review-r2c11` were superseded and remain local only: no
  remote tag or branch carries either. No `v0.1.0` or `v0.1.1` archival tag is
  public and no DOI has been minted. The CFF keeps `license: MIT` (code only)
  and Zenodo keeps `other-open` (mixed deposit); the version stays `0.1.1`.
- **Ownership provenance corrected (MJ01).** `README.md`, `REPRODUCING.md`,
  `CITATION.cff`, `.zenodo.json` and `framework/__init__.py` no longer present
  the six-class ownership rule as a pre-run object that fixed ownership in
  advance. They state the defensible provenance: the feature list is
  plan-declared and internally hash-timestamped, the six-class ownership rule
  is a **post-hoc analysis convention** with no pre-run record, and only the
  split carries an independently written pre-run timestamp. The 44.9%
  actor-owned / 55.1% not-actor-owned shares are unchanged and remain
  descriptive outputs of the convention, not a property of the actors.
- **World-line preview disclosed (MJ02).** `README.md`,
  `results/README.md`, `results/worldline_preview/README.md`,
  `notebooks/README.md`, notebook 03 and the Zenodo description now state the
  same scope as the manuscript's §6.4 disclosure: a repository-only
  illustrative preview, four anchors × 20 sampled directed dyads (71 distinct),
  **non-random**, inner period 2016-01–2017-11, step 2 a state-lag-only
  approximation, no accuracy computed. The manuscript cites the directory once
  as a repository-only illustration and analyses nothing here; it is not a
  paper result, exhibit or number. The previous README claim that §6.4
  "discusses it" is removed; no preview file, digest or manifest entry changed.
- **Doc truth repaired.** Notebook 01 and `notebooks/README.md` cite the
  holdout figure as Figure 2 (`fig2_holdout.pdf`) and the audit as Figure 3
  (`fig3_stability.pdf`); the non-existent `fig4_holdout.pdf` reference is
  retired (the R2C8 note is historical and superseded here).
  `results/README.md` now states that `hashes.figures_manifest` **is** the
  SHA-256 of the shipped `figures.sha256` wrapper in this freeze, because its
  paths already carry `paper/figures/`; the false "normalized digest, not the
  wrapper" clause is gone. The regeneration boundary names the missing figure
  generators and shared palette, and `DATA_EXCLUSIONS.md` resolves the
  four-names/seven-slots wording.
- **Verifier and tests.** The notebook check now scans `text/plain` and
  `text/html` outputs for row-level prediction/rate tokens, not just execution
  errors; the existing negative-output contract is exercised inside the
  notebook suite rather than as a new test. The release-state checks carry the
  R2C13 inventory, and the metadata contract rejects any reintroduced pre-run
  ownership claim. The suite stays at 144 tests and the verifier at 15 checks.
- **Unchanged contract.** `cffconvert`, cold-cache notebook execution and
  deterministic `make dist` pass; holdout 30/30 and stability 47/47 digests are
  unchanged. No `results/**` byte, `DATA_EXCLUSIONS` boundary, licence split or
  plan hash moved.

## Public-artifact revision (R2C12; publishing the reviewed snapshot)

Documentation, metadata and tests only; no analysis, refit, label read, plan
edit, code or aggregate change. The frozen plan `preregistration.json` stays
byte-identical (`41bda0e3…`) and no metric, interval, sign, timestamp, split or
scope literal changed. The title is unchanged and is the T1 string:

> *Actor-role components of dyadic event forecasts: shared graph columns,
> near-collinearity in a 196-actor panel, and a within-panel stability audit*

- **Published snapshot.** `main` and annotated tag `review-r2c12` carry the
  commit that contains these notes as the **published review snapshot**: a
  frozen, fetchable record of the reviewed aggregate tree, not an archival
  release and not a DOI-bearing version. The tag and `main` point at exactly
  this commit; no other revision is published by this pass.
- **Tag inventory.** Public review refs: `review-r2c2` … `review-r2c8`,
  `review-r2c10` (the prior published snapshot, commit `95aa07b`) and
  `review-r2c12`. `review-r2c9` and `review-r2c11` were superseded and remain
  local only: no remote tag or branch carries either. No `v0.1.0` or `v0.1.1`
  tag is public and no DOI has been minted.
- **Availability statements scoped.** `README.md` (release status and citation
  sections), `DATA_EXCLUSIONS.md` and `results/README.md` now name the published
  `review-r2c12` tag as the fetchable ref, keep `review-r2c10` as the prior
  published snapshot, and restrict the "on request" wording to the two local
  revisions (`review-r2c9`, `review-r2c11`). No reader-facing surface claims an
  external registration: the plan is plan-declared and internally
  hash-timestamped, and only the split carries an independently written pre-run
  timestamp.
- **Unchanged contract.** CFF `license: MIT` (code only) and Zenodo
  `other-open` (mixed deposit) are unchanged; the version stays `0.1.1`; the
  full suite (144 tests), the verifier (15 checks), `cffconvert`, cold-cache
  notebook execution and deterministic `make dist` all pass.

## Public-artifact revision (R2C11; release state, availability and title checks)

Documentation and metadata prose only; no analysis, refit, label read, plan
edit, code or aggregate change. The frozen plan `preregistration.json` stays
byte-identical (`41bda0e3…`) and no metric, interval, sign, timestamp, split or
scope literal changed. The title is unchanged and is the T1 string:

> *Actor-role components of dyadic event forecasts: shared graph columns,
> near-collinearity in a 196-actor panel, and a within-panel stability audit*

- **Release state corrected.** The R2C10 notes said the `review-r2c10` branch
  and tag were local and that `origin/main` still pointed at the earlier R2C7
  commit. That is no longer true: publication happened outside the revision
  agent, after the R2C10 freeze. The public remote now has `main = 95aa07b` and
  annotated tag
  `review-r2c10` (tag object `13f654a3…`, peeled `95aa07b`), i.e. a **published
  review snapshot** of the reviewed aggregates — frozen and fetchable, but not
  an archival release and not a DOI-bearing version. At the R2C11 freeze the tag
  inventory was
  `review-r2c2` … `review-r2c8` plus `review-r2c10`; branch `review-r2c8` also
  remains public; `review-r2c9` was
  superseded and remains local. The R2C12 entry above gives the current
  inventory and publishes this wording. No `v0.1.1` tag and
  no DOI exist, and no `v0.1.0` tag is public.
- **R2C11 is local and superseded by R2C12.** At the R2C11 freeze this text
  revision was a local commit on branch `review-r2c11`; no `review-r2c11` tag
  was created and none was pushed. The R2C12 revision publishes that wording
  plus these notes, so the branch is superseded; its text remains available from
  the authors on request, and anyone needing a fetchable ref should cite the
  published `review-r2c12` snapshot.
- **Availability statements scoped.** `README.md` (release status and citation
  sections) and `DATA_EXCLUSIONS.md` now name the fetchable tag and commit
  instead of an unscoped "on request", and state that only the R2C11 prose is
  request-only. Claim wording is unchanged: the plan is
  plan-declared and internally hash-timestamped, the split is pre-specified and
  independently timestamped, and no reader-facing surface claims an external
  registration.
- **Superseded status note.** `results/README.md` keeps the single surviving
  superseded stability status literal annotated as superseded and unused, and
  now also annotates `preregistration.json#/status` and the `train_rows`
  denominator (the plan counts the excluded held-out-dyad training rows; the
  manifest counts the filtered training set). Both annotations are
  documentation-only; no `results/**` JSON is edited.
- **Title checks.** The T1 title is carried unchanged by `README.md`,
  `CITATION.cff`, `.zenodo.json`, `pyproject.toml`, `notebooks/README.md`, the
  verifier and the metadata tests; the metadata test suite still rejects the
  superseded R2C9 title framing, and `cffconvert --validate` passes.
- **Unchanged contract.** CFF `license: MIT` (code only) and Zenodo
  `other-open` (mixed deposit) are unchanged; the version stays `0.1.1`; the
  full suite (144 tests), the verifier (15 checks), `cffconvert`, cold-cache
  notebook execution and deterministic `make dist` all pass.

## Public-artifact revision (R2C10; construction statement, status annotations and title)

Text, public documentation, metadata and tests only; no analysis, refit, label
read or aggregate value changed. The frozen plan `preregistration.json` stays
byte-identical to R2C8 (`41bda0e3…`); no metric, interval, sign, timestamp,
split or scope literal changed. The only shipped-artifact edit is a prose status
note in `results/framework_pilot/holdout/report.md`, whose digest is updated in
`results/framework_pilot/holdout/SHA256SUMS` and
`manifest.json#/release_annotations/current_sha256`; `summary.json` bytes are
unchanged.

- **Title corrected (unchanged at R2C11).** The R2C9 title made an empirical
  single-actor-signal claim; that framing is superseded, not current, and is
  never used as a description of this release. The public title
  now reads *Actor-role components of dyadic event forecasts: shared graph
  columns, near-collinearity in a 196-actor panel, and a within-panel stability
  audit*. The manuscript-side replacement is the construction statement: five
  of the seven actor-owned column pairs are the same undirected graph vector, so
  near-collinearity is largely a property of the feature contract, scoped to
  this 196-actor panel; the empirical residual is the ego/neighbour-activity
  asymmetry and the ridge split. `README.md`, `CITATION.cff`, `.zenodo.json`,
  `pyproject.toml`, `notebooks/README.md`, the verifier and the metadata tests
  carry the new string, and the old phrase is a banned framing in both the
  verifier and the test suite.
- **Artifact status annotations.** `results/README.md` names the single
  surviving superseded status literal
  (`stability/summary_decompose.json#/provisional`,
  `fresh_dyad_holdout: not available in this sprint`) and records that it is
  superseded and unused by the manuscript. `holdout/report.md` §9 now states
  that "internally hash-timestamped" is superseded by the paper's Appendix B:
  only the split carries an independently written pre-run timestamp, while the
  analysis plan is internally hash-timestamped. `summary.json` and
  `summary_decompose.json` bytes are unchanged; the annotation is
  documentation-only.
- **Public ref (superseded by external publication).** At R2C10 freeze the
  `review-r2c10` branch and tag were still local, `origin/main` was still the
  earlier R2C7 commit, and remote review refs stopped at `review-r2c8`; that
  state is superseded.
  After the freeze, `main` was moved to `95aa07b` and annotated tag
  `review-r2c10` was published, so that reviewed snapshot became fetchable;
  `review-r2c9` remained local and unpublished. R2C12 later published this
  wording as `review-r2c12`, which is now the current snapshot (R2C10 stays
  fetchable as the prior one); no `v0.1.1` tag and no DOI exist. See the R2C12
  entry above for the current statement.
- **Unchanged contract.** CFF `license: MIT` (code only) and Zenodo
  `other-open` (mixed deposit) are unchanged; the version stays `0.1.1`; the
  full suite (144 tests), the verifier (15 checks), `cffconvert`, cold-cache
  notebook execution and deterministic `make dist` all pass.

## Public-artifact revision (R2C9; claim wording, licence split and title)

Text, metadata, documentation, tests and CI only; no analysis, refit, label
read or aggregate value changed. The frozen holdout JSON changed in two
`paper_sentences` strings (and the `holdout_extras` mirror) and the `report.md`
BH-family label; the plan file `preregistration.json` is **byte-identical** to
R2C8 (`41bda0e3…`), and no numeric, timestamp, split or scope field changed.
`CORRECTIONS.md` records the errata and the shipped hashes are updated.

- **Claim status qualified.** Public status adjectives that implied an external
  registration are replaced by the paper's vocabulary: the plan is
  plan-declared and internally hash-timestamped, the split is pre-specified, the
  holdout is one-shot, and only the split carries an independently written
  pre-run timestamp. Numbers, timestamps and frozen scope literals are
  unchanged; the access log is untouched.
- **Licence split.** `CITATION.cff` now carries `license: MIT` (valid SPDX,
  source code only); `.zenodo.json` keeps `other-open` for the mixed deposit;
  `pyproject.toml`, `LICENSE` and `NOTICE` keep MIT for code only and state that
  the ICEWS-derived aggregates are not MIT. `cffconvert --validate` runs in CI
  and in the metadata tests when the tool is installed.
- **Title cascade (superseded at R2C10).** `README.md`, `CITATION.cff`,
  `.zenodo.json`, `pyproject.toml`, `notebooks/README.md`, the verifier, the
  metadata tests and the frozen title strings carried the R2C9 title, which is
  superseded at R2C10 (see above).
- **Public-boundary notes.** `DATA_EXCLUSIONS.md` names the `review-r2c9` tree
  (local then, superseded now); the tag inventory at R2C9 was `review-r2c2` …
  `review-r2c8`, with `review-r2c9` local only (the R2C12 entry above gives the
  current inventory);
  `results/README.md` and
  `DATA_EXCLUSIONS.md` document the 48-name `files[]` inventory versus the 47
  redistributed files plus `SHA256SUMS`; the audit/figure regeneration boundary
  (private 50-refit tree versus the public five-seed preview) is stated; and the
  world-line preview remains a standalone, repo-only illustration.
- **Cache gloss.** `experiments/sprint_b_common.py`, `results/README.md` and
  notebook 02 state that `provisional_holdout_not_integrated` is literal about
  holdout *rows*: the P1 graph cache retains edges incident to holdout dyads, so
  burned-period attributions are conditional on them.

## Public-artifact revision (R2C8; documentation, code-reproducibility and tests)

Documentation, verifier, notebook-executor, CI and lock-note changes only; no
analysis, no refit, no label read and no aggregate value changed. Freeze hashes
stay in `manifest.json#/post_run_corrections`; shipped bytes are tracked by
`outputs`/`current_sha256` and both `SHA256SUMS` trees.

- **World-line preview is repo-only.** Every public mention that presented
  `results/worldline_preview/` as a manuscript exhibit is retired: the preview
  is a standalone, illustrative repository illustration, not a manuscript
  exhibit. Its PDF/PNG bytes and digests are unchanged, and `make verify` /
  `make worldline-check` still pin them; those checks now fail if a document
  re-asserts paper-exhibit membership. No preview file was added or deleted.
- **Access log scoped.** `holdout/test_access_log.json#/policy` now claims
  coverage only for post-freeze holdout reads, names the three earlier
  exploratory passes (G2, D2, B) as out of scope and disclosed in the
  manuscript, points at the stability audit's separate ledger, and annotates
  the post-process `labels_read` flag (no raw label read; labels restored from
  the saved predictions). No timestamp, count, entry order or
  `definition_sha256` changed.
- **Cold-cache notebooks.** `scripts/execute_notebooks.py` warms the matplotlib
  font cache before a kernel starts and absorbs the known one-time font-cache
  stream, so `make notebooks` leaves the tracked notebooks byte-identical from
  an empty `MPLCONFIGDIR`. Notebook 01 keeps its nine outputs and its figure.
  CI runs `make notebooks` and `make dist` in the same checkout.
- **Numeric and inventory corrections.** `README.md` prints the correctly
  rounded ECE difference 0.031884; `results/README.md` gives the 14-entry figure
  manifest and marks the `repo/` normalisation as historical.
- **Verifier honesty.** An absent `.git` now prints
  `[SKIP] git index (not a git tree)` instead of `[OK]`.
- **Lock drift documented.** `requirements.lock` lists every transitive drift a
  clean install produces (python-dateutil, platformdirs, wcwidth, narwhals,
  fonttools); no package is re-pinned.

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
- Deposit metadata: `CITATION.cff` licence was `other-open` at R2C7, matching
  `.zenodo.json`; R2C9 splits the fields (CFF MIT, code only; Zenodo keeps
  `other-open`). `pyproject.toml` and `LICENSE` keep MIT **code only**, and the
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
  `pyproject.toml` description carried the manuscript title verbatim at this
  revision; the final title is set at R2C9 (see above).
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
  no longer advertise a 164 mm / 465x380 pt preview; they state the pinned
  150 mm (425.197 x 330 pt) preview geometry. `make verify` and the contract
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
  checksum of the holdout plan, `redistributed: false` annotations for the four
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
- The world-line preview is retained as an illustrative, repo-only artifact and
  is now a **standalone 150 mm vector render** (425.2 × 330 pt, Arimo, no baked
  key); the PNG is the same render at 300 dpi. It is not a manuscript exhibit:
  no paper figure or appendix is derived from it. It is rendered from restricted
  trace exports and is not regenerable from this tree. The underlying experiment
  was not rerun.
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
public repository, and the tag inventory is `review-r2c2` … `review-r2c8` plus
`review-r2c10` and `review-r2c12` (`review-r2c9` and `review-r2c11` local
only). Neither a `v0.1.0` nor a `v0.1.1`
tag has been pushed to the public remote.

This was a curated public release of the code and non-restricted aggregate
artifacts. Local absolute paths in copied reports/scripts were replaced with
`<repo>`; no metric, prediction or model output was changed.

Not included: ICEWS raw/derived data, row-level predictions/labels, the
manuscript, internal project documents, internal configuration files, and the
old hierarchy pilot notebooks.
