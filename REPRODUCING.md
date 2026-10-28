# Reproducing the public artifact

This release separates four levels of reproduction. Only L0–L2 are executable
from the public tree. L3 needs licensed ICEWS v37, the restricted derived
feature and contribution caches, the local experiment configuration, and — for
post-holdout diagnostics — row-level predictions. The public tree has no
cache-build entry point, so licensed access alone is not sufficient. L4 is the
one-shot confirmatory holdout and **cannot** be re-run without violating the
plan's split commitment and the one-shot lock.

All commands assume the repository root and `python3` (>= 3.12; the pinned
`numpy==2.5.4` and `scipy==1.18.1` have no cp311 wheels). CPU only.

**What this artifact is.** What ships is the manuscript's actor-role
attribution contract and its within-panel stability audit, released for
verification: a post-hoc six-class ownership convention applied to a
plan-declared feature list, fixing one owner per feature and the unit and sign,
an exact group-additive decomposition (the logistic identity, not a new
estimator) and a separate stability audit. The ownership convention has no
pre-run record and is not a preregistered object; only the split carries an
independently written pre-run timestamp. It is one paper path, not a
reusable framework, and it makes no general claim about graph features,
persistence or any corpus beyond this panel. The executable surface is L0–L2;
L3–L4 are documented for context only. The 50-refit audit code is restricted
and is not in this tree — the public `experiments/` scripts are a five-seed
preview — and row-level predictions and labels are withheld.

## Levels

| Level | Question | Needs | Command | Status in this release |
|---|---|---|---|---|
| **L0 — Read** | What do the frozen aggregates say? | Nothing beyond the files | open `results/README.md`, `notebooks/*.ipynb` on the repository page | Available |
| **L1 — Verify** | Do the shipped bytes, JSON and tests hold? | Python 3.12+, `requirements.txt` | `make doctor && make test && make verify` | Available; no data/GPU |
| **L2 — Re-render** | Can the notebooks be executed from the aggregates? | L1 + `nbclient`/`ipykernel` | `make notebooks` | Available; rewrites notebook outputs in place, byte-stable from a cold cache |
| **L3 — Restricted regeneration** | Can the aggregate tables be rebuilt? | licensed ICEWS v37, restricted feature and contribution caches, local experiment configuration, row-level holdout predictions for post-processing; no public cache-build entry point | `experiments/*.py` with local inputs | **Not runnable here**; the public Sprint-B scripts are a five-seed preview and do not regenerate the 50-refit tables |
| **L4 — Confirmatory holdout** | Can the one-shot holdout be re-run? | the burned 2018–22 labels and a new pre-specified split/plan (there is no external registration; the existing plan is internally hash-timestamped) | — | **Not reproducible by design**; labels were read once, `run_once.json` locks the run and row-level predictions are excluded |

`make test` and `make verify` pass without `data/`, `configs/`, `paper/`,
`docs/`, `preds/` or any cache.

## Level 1 — clean install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
make doctor
make test
make verify
```

`requirements.txt` pins the direct analysis stack. `requirements.lock` records
the full tested environment (including the optional notebook stack) generated
from the Python 3.13.5 virtual environment used for this release. The package is
also installable:

```bash
python3 -m pip install -e .
# or, with the notebook tools
python3 -m pip install -e ".[notebooks]"
```

`make dist` builds the release archive **from the committed tree**, not from the
working directory: it runs `git archive HEAD`, so untracked files are never
packaged, and it aborts if a tracked file is modified or staged. The archive
file list is then compared with `git ls-tree -r --name-only HEAD`, scanned for
forbidden paths, extracted to `build/dist/verify/` and passed through
`scripts/verify_release.py`.

## Level 2 — notebooks

```bash
make notebooks
```

The executor starts each kernel in the repository root, runs all cells
top-to-bottom with `MPLBACKEND=Agg`, writes the outputs back into the notebook
JSON, and fails on an execution error, an absolute path or a credential-like
string. It warms the matplotlib font cache before the kernel starts (honouring
`MPLCONFIGDIR` if set, otherwise `build/mplconfig`) and absorbs the known
one-time font-cache stream, so a run from an empty cache leaves the tracked
notebooks unchanged and `git status` clean. Notebook 01 degrades gracefully
(prints `[missing] ...`) when the `results/` tree is absent; it does not raise a
`KeyError`.

## Level 3 — restricted regeneration (documented, not executable)

The experiment scripts are shipped for auditability, not as a turnkey pipeline.
They read:

- the ICEWS v37 event panel (doi:10.7910/DVN/28075), which is not
  redistributable;
- `framework/cache/` feature panels and Sprint-B contribution caches;
- `results/framework_pilot/holdout/preds/*.npz` for the post-holdout
  diagnostics and for the HGB holdout block. The stability audit itself does
  not read row-level predictions or holdout labels.

None of these inputs is in the public tree. The exact local configuration is
also excluded, and there is **no public cache-build driver**. Running the
scripts therefore requires a separately licensed working copy and is outside
the public artifact's contract.

The shipped Sprint-B scripts are the **pre-rerun five-seed preview**, not the
code that produced `results/framework_pilot/stability/`. The shipped
aggregates come from a restricted holdout-excluded 50-refit variant. The exact
five differences are published in `experiments/STABILITY_VARIANT.md`:

1. five versus 50 seed refits;
2. no holdout-dyad exclusion versus the 5,282 directed / 2,790 unordered
   holdout dyads removed from fitting and aggregation;
3. `C` re-selected on the 2015 selection window versus `C_grid=[1.0]`;
4. 1,000 versus 50 stability bootstrap replicates;
5. a hard-coded `rep0`–`rep4` loader versus enumeration over `N_SEEDS`.

The variant is not shipped as a runner. Applying the diff to the public code
still cannot produce a byte-identical copy of the frozen tree: the restricted
caches are absent, `generated_utc` would change and last-bit floating-point
differences are possible. L3 therefore documents the method; the frozen
aggregates and their `SHA256SUMS` are the authoritative record.

The preview defaults to refusing writes into
`results/framework_pilot/stability/`. Use
`NEUROSIM_STABILITY_OUT=results/framework_pilot/stability_preview` to redirect
output, or `NEUROSIM_ALLOW_OVERWRITE=1` only when intentionally replacing the
shipped aggregates.

## Level 4 — why the holdout is not reproducible

The confirmatory holdout is a **one-shot** design: `experiments/sprint_a_run.py`
refuses a second confirmatory run once `run_once.json` says `completed`, and
`test_access_log.json` records every post-freeze 2018–22 holdout read; the
post-freeze stability audit logs its own burned-retrospective read in
`results/framework_pilot/stability/summary_decompose.json`. Reads that predate
the freeze, and the three earlier exploratory passes (G2, D2, B), are outside
its scope and are disclosed in the manuscript. The lock file is part of the
shipped holdout tree and there is no public unlock path. The labels were read
once on 2026-10-23. Re-running the holdout, re-reading labels, or fitting to
the holdout would violate the one-shot lock and the plan's split commitment.
The public release therefore verifies the frozen aggregates and their hashes,
not a fresh fit.

## Artifact map

| Claim / display | Frozen file | Public view |
|---|---|---|
| Headline holdout metrics | `results/framework_pilot/holdout/summary.json`, `report.md` | README, notebook 01 |
| Table 2 Panel A (accuracy/calibration) | `holdout/tables/holdout_metrics_main.csv` | notebook 01 |
| Table 2 Panel C (warm-history contrast) | `holdout/tables/primary_comparisons.csv`, `auc_ap_deltas.csv` | notebook 01 |
| Within- vs between-dyad AUC | `holdout/tables/within_between_auc.csv`, `within_between_deltas.csv` | notebook 01 |
| Graph / train-statistics LOO | `holdout/tables/feature_group_loo.csv` | notebook 01 |
| Stability audit (provisional) | `stability/report.md`, `stability/summary*.json`, `stability/tables/*.csv` | notebook 02 |
| Stability scope and preview deltas | `stability/summary.json` scope block, `stability/SHA256SUMS`, `experiments/STABILITY_VARIANT.md` | README, `make stability-check` |
| World-line preview (illustrative) | `worldline_preview/worldline_trace.json`, `worldline_preview.pdf`, `worldline_preview.png` (no row-level CSV ships) | notebook 03 |
| Data exclusions and privacy boundary | `DATA_EXCLUSIONS.md`, `results/README.md` | `make verify`, `make stability-check`, `make worldline-check`, contract tests |
| One-shot integrity | `holdout/run_once.json`, `test_access_log.json`, `preregistration.json` | README, `results/README.md` |
| Release integrity | `holdout/SHA256SUMS`, `holdout/manifest.json` | `make verify` |

The manuscript is not redistributed, so table/figure numbers are mapped by name
in `results/README.md` and the notebooks rather than by a `paper/` build.

## What cannot be reproduced

- The confirmatory holdout fit or any new 2018–22 label read (L4).
- The restricted feature caches and exact experiment configuration (L3).
- Row-level predictions, actor-level predicted probabilities or observed outcome
  rates, run logs and the ICEWS event panel; these are deliberately excluded.
- The provisional stability numbers on the holdout: the audit was frozen before
  the holdout completed and was not integrated. The shipped stability files are
  the holdout-excluded 50-refit rerun; they are not a holdout-side test.
- A byte-identical copy of the stability tree, even with the five-delta variant
  applied: restricted caches, timestamps and floating-point last bits differ.
  Verify the shipped bytes with `stability/SHA256SUMS` instead.
- Manuscript Figures 3–4 or a byte-identical stability table from the public
  tree: the default path is the private 50-refit tree, while the public
  five-seed preview fails the 50-replicate guard by design and no
  figure-generation script ships here.

## Verification details

`scripts/verify_release.py` checks only distributed files:

- required public files and directories, with a split code/deposit licence
  contract (CITATION.cff `MIT`, code only; `.zenodo.json` `other-open`),
  matching title metadata, pending author metadata, the nine required test
  modules and the 123-test floor;
- absence of `paper/`, `docs/`, `configs/`, `data/`, `preds/`, caches and logs;
- strict JSON (`NaN`/`Infinity` rejected) for every `results/**/*.json`;
- `sha256sum -c results/framework_pilot/holdout/SHA256SUMS` and
  `sha256sum -c results/framework_pilot/stability/SHA256SUMS` compatibility;
- the repo-relative checksum of the plan file (`preregistration.json`);
- `redistributed: false` annotations for the four excluded prediction/log files;
- executed, error-free, path-clean notebook outputs;
- the frozen headline numbers, the stability status banner, the 50-refit /
  C=1.0 / holdout-excluded scope fields and the 50-replicate stability
  bootstrap;
- the presence of `experiments/STABILITY_VARIANT.md` and the absence of the
  legacy pre-holdout status label or five-seed audit claims in public docs;
- no absolute home/temporary paths and no credential-like strings;
- no forbidden path tracked by git (when run in a git tree).

The path scan names four absolute-path prefixes — `/home`, `/tmp`, `/Users` and
`C:` user directories — and the credential scan covers AWS key ids, GitHub and
OpenAI token shapes, private-key blocks and inline secret assignments. Both are
plain text scans over the shipped tree, so a copied local path or a pasted token
in a document fails `make verify` exactly as it would in code.

`cffconvert --validate -i CITATION.cff` is part of CI and runs in the metadata
tests when the tool is installed; it must report schema 1.2.0 validity.

The suite has no coverage threshold: `make test` runs the full public suite
(144 tests at this revision) and `make verify` enforces a 123-test floor, so a
shrinking suite fails the release even though no line-coverage percentage is
reported.

Use `python3 scripts/verify_release.py --root <extracted-release>` to verify an
archive or DOI deposit.
