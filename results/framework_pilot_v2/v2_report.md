# V2 execution report (WP-EXEC)

**Status:** completed. **Post hoc secondary analysis on a burned, unit-fresh holdout** — not confirmatory, not pre-registered.

- Protocol: `184ae726ead6412c2515317e9d6169e61eb85308427849c9aa5e7427cbe57676` (`FROZEN_BEFORE_V2_FIT`, verified before any fit)
- Script: `experiments/sprint_c_v2.py` sha256 `a8f83417dfc7670740571ff19b3df8a65031e92dfd3459b6b7faaedd22bc0843`
- Outputs: `results/framework_pilot_v2/`; `SUMS.sha256` covers every file; `suppressed = []`
- Row counts: tr 5,055,328, sel 232,672, ev 311,638 (32,614 positives, 5,282 dyads), ext 5,816,800, inn 607,982

V2 is a post hoc secondary analysis of a burned holdout; the pre-registered primary remains the 49-feature, H0-cache, 2017-vintage-warm comparison (V1).

## 1. Fits (eval 2018-01..2022-11, raw probabilities, C=1.0, T=1.0)

| fit | features@cache | window | log-loss | Brier | ECE | AUC | AP |
|---|---|---|---|---|---|---|---|
| F1 | 49@H0 | tr (t≤238) | 0.217964 | 0.064074 | 0.042177 | 0.910164 | 0.671712 |
| F2 | 49@P1 | tr (t≤238) | 0.203181 | 0.058625 | 0.024682 | 0.909710 | 0.670469 |
| F3 | 47@P1 | tr (t≤238) | 0.193867 | 0.055197 | 0.008945 | 0.910298 | 0.675498 |
| F4 | 29@P1 | tr (t≤238) | 0.204964 | 0.059283 | 0.024081 | 0.908290 | 0.671748 |
| F5 | 25@P1 | tr (t≤238) | 0.197078 | 0.055788 | 0.012484 | 0.906670 | 0.670934 |
| F6 | 49@P1 | ext (t≤274) | 0.202924 | 0.058659 | 0.023971 | 0.910004 | 0.671787 |
| F7 | 47@P1 | ext (t≤274) | 0.193204 | 0.055010 | 0.004002 | 0.910420 | 0.675771 |
| F8 | 47@H0 | tr (t≤238) | 0.193850 | 0.055169 | 0.008609 | 0.910302 | 0.675701 |
| F9 | 25@H0 | tr (t≤238) | 0.197078 | 0.055788 | 0.012484 | 0.906670 | 0.670934 |
| D1 | 29@H0 | tr (t≤238) | 0.205219 | 0.058994 | 0.027463 | 0.908290 | 0.671747 |
| warm_orig_markov | warm Markov | t<275 | 0.209729 | 0.059039 | 0.010228 | 0.894173 | 0.636848 |
| warm_orig_rate | warm base rate | t<275 | 0.214682 | 0.060598 | 0.011853 | 0.887348 | 0.618604 |
| warm_match_markov | warm Markov | t<239 | 0.214821 | 0.059910 | 0.011332 | 0.888477 | 0.627384 |
| warm_match_rate | warm base rate | t<239 | 0.222656 | 0.061451 | 0.013121 | 0.875851 | 0.608949 |

D1 is the V1 29-feature reconciliation diagnostic (C=0.3, T=1.1); it is not an arm. F5 and F9 are bit-identical predictions (25 features do not use the graph cache).

## 2. Contrasts (dLL = LL_baseline − LL_model; positive = model better)

| arm | contrast | dLL | 95% multiway CI | p | q (BH family) | dECE | dBrier |
|---|---|---|---|---|---|---|---|
| A0 replication | F1 vs warm_orig_markov | -0.008235 | [-0.014198, -0.002272] | 6.797e-03 | — (—) | -0.031949 | -0.005035 |
| A1 as-specified | F2 vs warm_match_markov | 0.011640 | [0.005716, 0.017564] | 1.175e-04 | 1.763e-04 (F3_sensitivity) | -0.013350 | 0.001285 |
| A2 PRIMARY | F3 vs warm_match_markov | 0.020954 | [0.015443, 0.026465] | 9.188e-14 | 1.378e-13 (F1_primary) | 0.002387 | 0.004713 |
| A1 both comparators | F2 vs warm_orig_markov | 0.006548 | [0.002899, 0.010197] | 4.368e-04 | — (—) | -0.014454 | 0.000414 |
| A3 as-specified | F3 vs warm_orig_markov | 0.015862 | [0.013082, 0.018642] | 4.933e-29 | 2.220e-28 (F3_sensitivity) | 0.001283 | 0.003842 |
| A2 rate comparator | F3 vs warm_match_rate | 0.028789 | [0.017195, 0.040383] | 1.134e-06 | 2.041e-06 (F3_sensitivity) | 0.004176 | 0.006254 |
| A4 sensitivity | F4 vs warm_match_markov | 0.009857 | [0.004628, 0.015086] | 2.199e-04 | 2.473e-04 (F3_sensitivity) | -0.012749 | 0.000627 |
| A5 key secondary | F5 vs warm_match_markov | 0.017743 | [0.013118, 0.022368] | 5.515e-14 | 1.103e-13 (F2_key_secondary) | -0.001152 | 0.004122 |
| A6 49x vintage | F6 vs warm_orig_markov | 0.006805 | [0.003205, 0.010405] | 2.117e-04 | 2.473e-04 (F3_sensitivity) | -0.013743 | 0.000380 |
| A6 47x vintage | F7 vs warm_orig_markov | 0.016525 | [0.013770, 0.019280] | 6.687e-32 | 6.018e-31 (F3_sensitivity) | 0.006226 | 0.004029 |
| A7 49@H0 cache | F1 vs warm_match_markov | -0.003143 | [-0.010431, 0.004145] | 3.979e-01 | 3.979e-01 (F3_sensitivity) | -0.030845 | -0.004164 |
| A7 47@H0 cache | F8 vs warm_match_markov | 0.020971 | [0.015716, 0.026226] | 5.236e-15 | 1.571e-14 (F3_sensitivity) | 0.002723 | 0.004741 |
| A7 25@H0 cache | F9 vs warm_match_markov | 0.017743 | [0.013118, 0.022368] | 5.515e-14 | 1.241e-13 (F3_sensitivity) | -0.001152 | 0.004122 |
| nested 47 vs 25 | F3 vs F5 | 0.003211 | [0.001772, 0.004650] | 1.228e-05 | 1.638e-05 (F2_key_secondary) | 0.003539 | 0.000591 |

## 3. Primary (A2) and key secondary (A5)

- **A2** `47@P1` vs `warm_match_markov`: dLL = **+0.020954**, 95% CI **[0.015443, 0.026465]**, p = 9.188e-14, F1 q = 1.378e-13.
- **Nested** `25@P1` (base) vs `47@P1` (model): dLL = **+0.003211**, CI [0.001772, 0.004650], p = 1.228e-05, q = 1.638e-05.
- **A5** `25@P1` vs `warm_match_markov`: dLL = **+0.017743**, CI [0.013118, 0.022368], p = 5.515e-14, q = 1.103e-13. A5 does **not** beat A2 (nested dLL > 0), so the actor-owned+graph block does not hurt.
- AUC/AP (multiway): A2 dAUC +0.021821 [0.014445, 0.029196], dAP +0.048114 [0.037071, 0.059158].
- **Claim-rule conditions** (§12): {'d_gt_0': True, 'ci_excludes_0': True, 'q_F1_le_0.05': True, 'nested_ci_excludes_0': True}. Verdict: **support: the 47-feature contract beats matched persistence and the nested 47-vs-25 contrast excludes 0; the supported claim is bounded by the 22 added columns (20 graph + 2 ego)**.
- Bound on the claim: The 22 columns added over the 25-feature actor-free model are 20 graph terms plus 2 ego counts (12 of the 22 are actor-owned); they contribute only +0.003211 of the +0.020954 gain over matched persistence.

## 4. A0 / A7 reconciliation

| arm | quantity | observed | expected | Δ | status |
|---|---|---|---|---|---|
| A0 | F1 log_loss | 0.217964 | 0.217875 | 8.900e-05 | PASS_WITH_NOTE |
| A0 | warm_orig_markov log_loss | 0.209729 | 0.209729 | 0.000e+00 | PASS |
| A7 | F8 log_loss | 0.193850 | 0.193850 | 0.000e+00 | PASS |
| A7 | F9 log_loss | 0.197078 | 0.197023 | 5.500e-05 | PASS_WITH_NOTE |
| A0 | warm_orig_rate log_loss | 0.214682 | 0.214682 | 0.000e+00 | PASS |
| D1 | D1 log_loss (C=0.3,T=1.1) | 0.205219 | 0.205219 | 0.000e+00 | PASS |
| A7 | warm_orig_markov - F8 (gain) | 0.015879 | 0.015879 | 3.469e-18 | PASS |

F8, D1, both warm comparators and the +0.015879 gain are exact. A0's F1 is a near-replication: +8.9e-5 (PASS_WITH_NOTE, ≤1e-4). Diagnosis: the 49@H0 fit is optimizer-path sensitive because `seen_tr` is constant 1 on train and 0 on eval; V1 and V2 F1 predictions correlate 0.99991 in logit space, the 47@H0 fit reproduces exactly, and D1 is bit-exact. No STOP was triggered.

## 5. A7 cache factor (paired LL_H0 − LL_P1; positive = P1 better)

| features | P1 LL | H0 LL | Δ (H0−P1) | 95% CI | p |
|---|---|---|---|---|---|
| 49 | 0.203181 | 0.217964 | +0.014783 | — | 1.330e-16 |
| 47 | 0.193867 | 0.193850 | -0.000017 | — | 9.592e-01 |
| 25 | 0.197078 | 0.197078 | +0.000000 | — | 1.000e+00 |

The cache handicap matters only for the 49-feature arm (p=1.3e-16); for 47 it is null (p=0.96) and for 25 it is mechanically zero.

## 6. A8 ownership (descriptive; no p-values)

F2@inn six-class shares (source / target / dyad / graph / calendar / train_stats), %:

| class | V2 | V1 (audit order) | Δ |
|---|---|---|---|
| source_actor | 21.69 | 21.58 | +0.11 |
| target_actor | 23.52 | 23.35 | +0.17 |
| dyad | 29.05 | 29.74 | -0.69 |
| graph | 18.17 | 16.96 | +1.21 |
| calendar | 3.23 | 2.80 | +0.43 |
| train_stats | 4.35 | 5.57 | -1.22 |

- Actor-owned **45.21%**, not-actor **54.79%** (V1 44.93 / 55.07).
- `g_nra` reclassified to `actor_dyad_interaction`: actor-owned 12 columns 36.83%, not-actor **63.17%** (V1 target 62.27; Δ +0.90).
- Allocation range: graph-unowned 54.79, graph-all-actor 36.63, graph-half 45.71 (V1 55.07 / 38.11 / 46.59); span 18.17 pp.
- Role-specificity: 14 actor-owned of 49; 5 duplicate graph pairs (deg/wdeg/deg12/eig/pr); 4 role-unique (`src_ev`, `tgt_ev`, `g_nra_s`, `g_nra_t`); 12 of 14 partner-blind (only `g_nra_*` varies with partner).
- Structural: 6/7 source features constant across partners at t=200; `g_nra_s` varies in 195/195 actor cells (max 137 distinct); reversal invariance 7/7 at 3,798/3,798; mean|g_nra_s−g_nra_t| = 0.9659 (V1 0.966).
- Exactness identity on 50,000 eval rows: max|ε| = 1.15e-14 (< 1e-5).
- V1 order note: V1's 16.96 is the graph class and 2.80 the calendar class; the protocol lists the six numbers without labelling that pair.

## 7. A9 stability (provisional)

- Rule: t_rep = **50.0 s**; 100×t_rep = 1.39 h < 6 h → **N = 50 per cache** (no 50→20 downgrade), symmetric, 1225 pairs.
- P1: Spearman source 0.99904 (min 0.99210) / target 0.99896 (min 0.99361); J@10 0.818; J@50 0.923/0.887; replicate SD 0.0274/0.0289; coef sign-flip median 0.000; not-actor share 54.37/61.48/70.12 (point 65.18).
- H0: Spearman source 0.99541 (min 0.97056) / target 0.99586 (min 0.96531); J@10 0.818; J@50 0.887/0.961; replicate SD 0.0326/0.0334; coef sign-flip median 0.000; not-actor share 60.05/64.98/67.71 (point 66.36).
- Cross-cache: Spearman 0.9335 (source) / 0.9315 (target); J@10 0.538; 2/47 coefficients flip sign; Δ not-actor share -1.18 pp.
- Status: provisional; it cannot upgrade or refute the primary.

## 8. Runtime

| stage | seconds |
|---|---|
| selftest | 2.2 |
| fits_p1 | 79.9 |
| fits_h0 | 94.3 |
| compare | 235.7 |
| a8 | 14.0 |
| a9 | 3291.4 |
| report | 10.0 |
| **total** | **3727.5** |

Measured pipeline runtime ≈ 62.1 min; wall clock including one aborted compare pass and retries ≈ 2 h 15 min. Output size 31 MB.

## 9. Unresolved

- V2 is post hoc on a burned holdout; it cannot be confirmatory.
- A8 shares differ from V1 by construction (5.06M vs 6.34M train rows).
- A9 is provisional and cannot upgrade or refute the primary.
- A0 F1 is a near-replication (+8.9e-5, PASS_WITH_NOTE), not a bit-level reproduction: the 49@H0 fit is optimizer-path sensitive because seen_tr is train-only; F8 is exact and D1 is bit-exact.
- A9 per-replicate attribution matrices are stored as 196x49 (the 47 arm columns plus br/seen_tr appended as zeros for the accumulator); the first 47 columns are the arm.
- V1's six-class list gives 16.96 for graph and 2.80 for calendar; the protocol text does not label that pair. V2 recomputes 18.17/3.23.

## 10. No selective reporting

All 10 fits and 4 warm predictors are printed; all 14 canonical contrasts are printed, plus a 40-row fit × comparator matrix in `tables/comparator_matrix.csv`. Every p and q is printed, including non-rejections. `summary.json.suppressed = []`.

