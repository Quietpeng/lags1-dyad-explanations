# V2 matched-vintage audit — aggregate release

**Status:** post-hoc secondary analysis on a burned holdout, shipped with the
`v0.1.1` archival release. It is **not** a new confirmatory test, does not
re-open the holdout, and does not change the V1 49-feature primary. The V1
confirmatory result (`../framework_pilot/holdout/`) remains the
pre-registered primary of record. V2 was executed after the holdout was burned;
the paper labels every V2 arm "post hoc secondary".

**Protocol:** `184ae726ead6412c2515317e9d6169e61eb85308427849c9aa5e7427cbe57676`
(`FROZEN_BEFORE_V2_FIT`). **Script:** `a8f83417dfc7670740571ff19b3df8a65031e92dfd3459b6b7faaedd22bc0843`
(the frozen current script; the fit stage is certified behaviourally, not
archivally, by a bit-identical re-execution of all 10 fits).

## What ships

| File | Content |
|---|---|
| `summary.json` | Machine-readable V2 fits, contrasts, BH families, row counts and status. |
| `v2_manifest.json` | Execution manifest: protocol/script hashes, inputs, packages, arm results, checks, runtimes. |
| `manifest.json` | First-pass manifest with the same protocol hash, package set and arm table. |
| `once/run_once_v2.json` | One-shot record for the V2 post-hoc pass. |
| `tables/arms_main.csv` | One row per fit: features, cache, C/T, log-loss, Brier, ECE, AUC, AP, calibration. |
| `tables/arms_contrasts.csv` | All 14 contrasts: dLL, 95% multiway CI, p, dECE, dBrier, interval views. |
| `tables/a7_cache_matrix.csv` | Paired P1−H0 cache factor for 49/47/25. |
| `tables/a8_ownership.csv`, `a8_allocations.csv`, `a8_structural.csv` | Ownership shares, allocation range, role-specificity and exactness. |
| `tables/a9_stability.csv`, `a9_crosscache.csv`, `a9_coef_flip.csv` | Provisional 50-refit stability and cross-cache agreement. |
| `tables/bh_families.csv`, `mde.csv`, `auc_ap.csv`, `comparator_matrix.csv`, `warm_support.csv`, `within_dyad.csv`, `reconciliation.csv` | BH families, power, AUC/AP, comparator matrix, warm support, within-dyad checks and V1 reconciliation. |
| `parts/*.json` | A6/A8/A9 summaries, fit table, comparisons, runtimes, self-test and warm-count hash. |
| `v2_report.md` | Human-readable execution report. |
| `v2_fit_provenance.md` | Re-execution, F9 anchor and thread-sensitivity provenance. |
| `SHA256SUMS` | `sha256sum -c` checksums for every file in this directory except itself. |

## Headline numbers

- **A0 replication:** 49@H0 vs original 2017-vintage warm = **−0.008235**
  [−0.014198, −0.002272], p = 6.8e−3; 8.9e−5 from the V1 −0.008146.
- **A2 primary (post hoc):** 47@P1 vs matched (2014-vintage) warm =
  **+0.020954** [0.015443, 0.026465], q = 1.38e−13.
- **A5 key secondary:** 25@P1 (no actor-owned column) vs matched warm =
  **+0.017743** [0.013118, 0.022368], q = 1.10e−13 (84.7% of A2).
- **Nested 47 vs 25:** **+0.003211** [0.001772, 0.004650], q = 1.64e−5
  (15.3% of A2). The 22 added columns are 12 actor-owned graph terms, 2 ego
  counts and 8 symmetric graph terms; the contrast cannot separate actor from
  graph content.
- **Cache:** 49 +0.014783 (p = 1.33e−16); 47 −0.000017 (p = 0.959);
  25 mechanically zero.
- **Ownership (V2 training window):** 45.21% actor-owned / 54.79% not;
  63.17% not when the neighbour-activity terms leave the actor class; 4 of 49
  role-specific.

## What is withheld

`preds/v2_preds.npz`, `parts/preds_p1.npz`, `parts/preds_h0.npz`,
`parts/a9_attributions.npz`, `parts/warm_counts.npz`, `parts/fit_mu_sd.npz` and
`run.log` are **not** redistributed. They carry row-level predictions/labels,
fitted state or operational paths. The aggregate tables and JSONs above are the
public unit; see `../../DATA_EXCLUSIONS.md` §6.

## Verification

From the repository root:

```bash
sha256sum -c results/framework_pilot_v2/SHA256SUMS
python3 - <<'PY'
import json
s = json.load(open("results/framework_pilot_v2/summary.json"))
print(s["status"])
print(s["contrasts"]["A2"]["d_log_loss"])
PY
```

`make verify` checks the checksum manifest, strict JSON and the absence of
row-level or forbidden files. The V2 fit stage cannot be re-executed from this
tree: it needs the licensed ICEWS panel, the restricted caches and the full
experiment script; the aggregate is the reproducible unit.
