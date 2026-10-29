# V2 fit-stage provenance note

**Status:** fit stage re-executed from the frozen current script and reproduced
**bit-for-bit** against the stored predictions. F9/25 anchor explained. F1/49
residual narrowed to BLAS thread configuration and left unresolved. No stored
number was changed; A9 was not re-run.

**Author:** WP-EXEC. **Date:** 2026-10-29. **Scope:** provenance only — no
paper/public/candidate edit, no push, no new data, no protocol change.

---

## 1. Trigger

The V2 verifier returned CONDITIONAL:

- numeric core PASS;
- the script hash does not certify the **fit** stage, because
  `experiments/sprint_c_v2.py` has mtime `2026-10-29 04:11:32`, i.e. after the
  compare stage (`04:04`) and before A9, and no copy older than the fits
  exists. The `a9probe` N=1 attempt at `04:08` (a missing-`parts/` crash, then
  a successful retry) was read as evidence of an A9-only edit.

The gap is real: the manifest's `code_sha256` certifies the script that wrote
the summary, not the script that produced the fits. This note closes the gap
**empirically** (deterministic refit with the frozen current script), and
records what cannot be closed cryptographically (there is no pre-fit copy).

## 2. Frozen script

| item | path | sha256 |
|---|---|---|
| current script | `experiments/sprint_c_v2.py` | `a8f83417dfc7670740571ff19b3df8a65031e92dfd3459b6b7faaedd22bc0843` |
| frozen copy (named deliverable) | `experiments/sprint_c_v2_fit_repro.py` | same |
| frozen copy (inside the artifact bundle, SUMS-covered) | `results/framework_pilot_v2/parts/sprint_c_v2_fit_repro.py` | same |

`cmp` is byte-identical for all three. **This is the current script, not a
recovered original fit script.** No pre-04:11 copy exists; the certification
below is behavioural (bit-identical predictions), not archival.

Environment: Python 3.13.5, numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1,
duckdb 1.5.6; 4 CPU cores; BLAS/OMP default thread count = 4.

## 3. Re-run (fits only)

```bash
python experiments/sprint_c_v2_fit_repro.py --stage fits_p1 \
  --out <scratch>/repro_fit
python experiments/sprint_c_v2_fit_repro.py --stage fits_h0 \
  --out <scratch>/repro_fit
```

- Executed the 10 frozen fits (F1–F9, D1); **no A9**, no new data.
- Output went only to the scratch dir `<scratch>/repro_fit/`.
- The canonical `results/framework_pilot_v2/parts/` and `preds/` were not
  written by this re-run (verified by unchanged SUMS hashes, §7).

## 4. Fit-stage comparison (stored precision)

Canonical `preds/v2_preds.npz` vs the repro `preds_p1.npz` / `preds_h0.npz`,
same row order (meta keys `y,did,sid,tid,t_idx,active` asserted equal).

| fit | bit-identical (float32) | max abs Δp | n rows Δp>1e-7 | Δ log-loss | Δ Brier | Δ AUC | Δ AP | verdict |
|---|---|---|---|---|---|---|---|---|
| F1 | yes | 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | PASS |
| F2 | yes | 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | PASS |
| F3 | yes | 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | PASS |
| F4 | yes | 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | PASS |
| F5 | yes | 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | PASS |
| F6 | yes | 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | PASS |
| F7 | yes | 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | PASS |
| F8 | yes | 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | PASS |
| F9 | yes | 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | PASS |
| D1 | yes | 0.0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | PASS |

- Pass criterion: stored predictions are float32, so `max|Δp| ≤ 1e-6`;
  observed maximum is **exactly 0.0** for every arm.
- Coefficients: `max|Δcoef| = 0.0` for all 10 fits; `select_log_loss` identical
  to the stored value for all 10 fits.
- Conclusion: under the canonical run's environment, the frozen current script
  reproduces the entire fit stage bit-for-bit. The fit-stage numbers are
  certified by this script.

## 5. F9 / 25-feature anchor — root cause identified

Anchor: V2 `F9` = `25@H0` = **0.197078**, V1 ladder row = **0.197023**
(Δ = +5.5e-5). The 25-feature arm contains neither `seen_tr` nor graph
columns, so the `seen_tr` diagnosis cannot apply; F9 is also bit-identical to
F5 (`25@P1`), so the cache is not involved.

Diagnostic re-fits of the same 25 columns (same rows, `C=1.0`,
`max_iter=300`):

| variant | column order | tol | iters | holdout log-loss |
|---|---|---|---|---|
| V2 as-run | `DYAD_CAUSAL` order (calendar **last**) | 1e-4 | 26 | 0.197078485 |
| V1 ladder order | group order (calendar **first**) | 1e-4 | 30 | **0.197022505** |
| V2 order, converged | calendar last | 1e-6 | 110 | **0.197023599** |
| V2 order, float64 input | calendar last | 1e-4 | 26 | 0.197082251 |

**Finding.** V1's cumulative ladder built the design matrix by concatenating
feature **groups in ladder order** (`calendar` first); V2 passes the same 25
columns in `DYAD_CAUSAL` order (`calendar` last). L2 logistic regression is
permutation-equivariant in exact arithmetic, but LBFGS is not
permutation-invariant in its iterates: at the frozen `tol=1e-4` the two orders
stop after 26 vs 30 iterations at slightly different points. V1's
calendar-first early stop lands on 0.1970225, which rounds to the stored
0.197023; tightening the tolerance in either order converges to ≈0.1970236.

Therefore the F9 difference is an **optimizer-stopping × column-order
artifact**, not `seen_tr`, not the graph cache, and not float precision
(float64 input does not remove it). The V2 number is left unchanged. The same
mechanism plausibly explains why the V1 cumulative ladder's 25-feature value
sits 5.5e-5 from the converged optimum.

## 6. F1 / 49-feature anchor — thread sensitivity (unresolved exact value)

F1 (`49@H0`) = 0.217964 vs V1 = 0.217875 (Δ = +8.9e-5, PASS_WITH_NOTE).
The F1 column order is identical to V1's (`FEATURES_49`), so §5's order
mechanism does not apply. The fit is, however, **BLAS-thread dependent**:

| BLAS/OMP threads | F1 holdout log-loss | LBFGS iters | select log-loss |
|---|---|---|---|
| 1 | 0.218178316 | 43 | 0.233994 |
| 2 | 0.217999062 | 46 | 0.233978 |
| 4 (canonical) | 0.217964280 | 51 | 0.233969 |
| V1 stored | 0.217875298 | — | 0.233956 |

The 49-column matrix is ill-conditioned (20 graph columns with SD as small as
0.0024, plus the train-only `seen_tr` direction), so the BLAS reduction order
changes the LBFGS iterates and the stopping point at `tol=1e-4`; the select
log-loss moves in the same direction, i.e. the difference is in the fit
itself, not in the evaluation. V1's exact BLAS/thread configuration is not
recorded, so the residual +8.9e-5 cannot be reproduced on demand.

**Classification:** unresolved, environment-level (thread/BLAS configuration).
It is not a data, split, feature-contract or model-specification difference:
the frozen current script reproduces the stored F1 exactly at 4 threads, the
47-feature H0 anchor (0.193850) matches V1 exactly, and D1 is bit-exact.

## 7. Robustness of the primary to the thread-induced fit variation

The same thread sensitivity applies to the primary arm. One-thread re-fits
(same rows, `C=1.0`, `tol=1e-4`):

| quantity | 1 thread | 4 threads (canonical) | Δ |
|---|---|---|---|
| F3 `47@P1` log-loss | 0.193434198 (57 iters) | 0.193867190 (102) | −4.33e-4 |
| F5 `25@P1` log-loss | 0.197079873 (26) | 0.197078486 (26) | +1.39e-6 |
| A2 dLL vs matched warm | +0.021386354 | +0.020953362 | +4.33e-4 |
| nested dLL (25→47) | +0.003645675 | +0.003211296 | +4.34e-4 |

`warm_match_markov` is fit-free and deterministic (0.2148205519). The
thread-induced shift (≤ 4.4e-4) is an order of magnitude smaller than the A2
effect (2.1e-2) and its multiway CI half-width (5.5e-3), and smaller than the
nested gain (3.2e-3–3.6e-3). The sign, the CI exclusion of 0 and the §12
claim-rule verdict are unchanged. No number was adjusted; the sensitivity is
recorded, not corrected.

## 8. SUMS update

Two lines were appended to `SUMS.sha256` (the pre-existing 37 lines are
untouched):

- `v2_fit_provenance.md`
- `parts/sprint_c_v2_fit_repro.py`

Verification (from `results/framework_pilot_v2/`):

```bash
sha256sum -c SUMS.sha256
```

## 9. Unresolved

1. **No pre-fit script copy exists.** The original fit-stage script cannot be
   certified cryptographically; certification rests on the bit-identical
   refit in §4 under the same environment.
2. **F1 exact V1 value.** Residual +8.9e-5; V1's BLAS/thread configuration is
   unrecorded. Reproduced values span 0.21796–0.21818 across 1/2/4 threads.
3. **Thread dependence of F3.** ±4.3e-4 on the primary point estimate.
   Documented; correcting it would require a new protocol (thread pinning) and
   is out of scope here.
4. **A9 not re-run** (per instruction). A9 remains provisional and its
   provenance rests on the same frozen script (A9-only code paths, unchanged
   after the fits).
