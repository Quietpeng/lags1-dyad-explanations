> **Release annotation (v0.1.1).** This public copy excludes the four
> row-level prediction/log files; the manifest marks each as
> `redistributed: false`. The metric and model numbers are unchanged.
> The within-dyad/calendar wording was corrected to match the manuscript, and
> the warm-history comparator is now described as unequal in two directions
> (with the graph-cache limitation); no frozen number changed.

# Sprint-A confirmatory unit-fresh dyad holdout

Generated: 2026-10-23T08:33:12.652846+00:00  
Status: **one-shot full holdout completed** (non-blind, non-quick).  
Pre-registration file SHA-256: `41bda0e3eaf8bb32e398c260169b85715e8c82e04569a50a185ae49d0783704c`  
Definition SHA-256: `776a763e663b845003cd628d0656692d8ff2652aaa5cec18820e078baec62c8e`  
Period: 2018-01–2022-11 (retrospective; unit-fresh, not time-fresh).

## 1. Protocol and one-shot integrity

- Holdout: 311,638 dyad-months, 32,614 positives (10.47%), 5,282 directed dyads (2,790 unordered pairs).
- Embargo rows dropped at t_idx = 239, 251, 275, 335; train < 239, select 240–250, eval 276–334.
- The holdout labels were read once, by `experiments/sprint_a_run.py --part holdout`; this report only post-processes the saved predictions.
- No 2018–22 labels were used for fitting, model selection, calibration, or hyper-parameter tuning.
- Fixed-effects and leave-one-group-out models below are fitted on train/select only; eval labels are used for scoring only.

## 2. Headline metrics (full holdout)

| model | LL | Brier | ECE | AUC | AP | within-dyad AUC (P1) | within-dyad AUC (strict, pair-weighted) | between-dyad AUC (P1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_global_transition` | 0.265874 | 0.073497 | 0.003470 | 0.7323 | 0.3211 | 0.5315 | 0.5376 | 1.0000 |
| `base_per_dyad_markov` | 0.265874 | 0.073497 | 0.003470 | 0.7323 | 0.3211 | 0.5315 | 0.5376 | 1.0000 |
| `base_dyad_base_rate` | 0.335382 | 0.093738 | 0.006109 | 0.5000 | 0.1047 | 0.5000 | 0.5000 | 0.5000 |
| `base_warm_base_rate` | 0.214682 | 0.060598 | 0.011853 | 0.8873 | 0.6186 | 0.5000 | 0.5000 | 0.9860 |
| `base_warm_markov` | 0.209729 | 0.059039 | 0.010228 | 0.8942 | 0.6368 | 0.5503 | 0.5364 | 0.9947 |
| `base_markov_recency` | 0.233599 | 0.068849 | 0.066095 | 0.9027 | 0.6400 | 0.5518 | 0.5066 | 0.9973 |
| `logistic49_S1_none` | 0.217875 | 0.064010 | 0.042112 | 0.9102 | 0.6719 | 0.5266 | 0.5616 | 0.9983 |
| `logistic49_S1_temperature` | 0.217875 | 0.064010 | 0.042112 | 0.9102 | 0.6719 | 0.5266 | 0.5616 | 0.9983 |
| `logistic49_S1_isotonic` | 0.216895 | 0.063761 | 0.040307 | 0.9092 | 0.6648 | 0.5284 | 0.5620 | 0.9984 |
| `logistic29_S1_temperature` | 0.205219 | 0.058994 | 0.027463 | 0.9083 | 0.6717 | 0.5273 | 0.5416 | 0.9982 |
| `logistic49_S2_temperature` | 0.218494 | 0.064199 | 0.042743 | 0.9101 | 0.6718 | 0.5264 | 0.5616 | 0.9983 |
| `hgb49_none` | 0.225823 | 0.061506 | 0.035326 | 0.8805 | 0.6293 | 0.5332 | 0.5610 | 0.9980 |

Within-dyad columns: `P1` is the P1-compatible centred score. The strict column is **pair-weighted** over the 3,310 dyads with both classes; the strict **equal-dyad mean** is **0.5139** for the primary (`summary.json#/within_between/primary`, mean 0.513858), not 0.5616. Sections 4 and 5 report both definitions; they are not interchangeable, and no number in this table changed.

Primary vs per-dyad Markov (table convention: comparator − model, positive = primary better): ΔLL = **+0.047999** (+18.05% relative), 95% multiway CI [0.043333, 0.052665]; ΔECE = **-0.038642** (95% multiway CI [-0.053028, -0.024256]). Dyad-only CI [0.046280, 0.049732]; source CI [0.044708, 0.052009]; target CI [0.044731, 0.051589].

**Baseline caveat.** The train-only per-dyad Markov baseline collapses to the global transition for unit-fresh dyads (identical LL 0.265874); the warm-history Markov baseline (pre-2018 observed history only) is the strong comparator and is reported as a secondary row below. That contrast is **unequal in two directions**: warm history has 35 additional months of outcomes (through 2017-11) that the model does not see, while the primary carries the 49-feature actor-role and graph set and observes within-period history through t. Separately, the confirmatory evaluation uses the **holdout-zeroed** graph cache (held-out edges zeroed for every month, including the training snapshots) whereas the stability audit is **holdout-row-excluded** (eventual holdout rows removed from fitting and aggregation), so that cache difference limits the primary's evaluation and is not a baseline advantage. Against the warm-history baseline the primary model **loses on LL and ECE** but **wins on AUC/AP**; see Section 3 and the exact sentences in Section 9.

## 3. Primary comparisons and multiplicity

| comparison | ΔLL | rel. gain | multiway 95% CI | p (multiway) | BH q |
|---|---:|---:|---|---:|---:|
| `logistic49_temperature__vs__per_dyad_markov` | 0.047999 | 18.05% | [0.043333, 0.052665] | 2.1e-90 | 4.199e-90 |
| `logistic49_temperature__vs__dyad_base_rate` | 0.117507 | 35.04% | [0.094600, 0.140414] | 8.79e-24 | 1.172e-23 |
| `logistic49_temperature__vs__markov_recency` | 0.015724 | 6.73% | [0.010948, 0.020500] | 1.095e-10 | 1.095e-10 |
| `logistic49_temperature__vs__global_transition` | 0.047999 | 18.05% | [0.043333, 0.052665] | 2.1e-90 | 4.199e-90 |
| `logistic49_temperature__vs__warm_markov` | -0.008146 | -3.88% | [-0.013958, -0.002334] | 0.006011 | NA (secondary) |
| `logistic49_temperature__vs__warm_base_rate` | -0.003193 | -1.49% | [-0.009742, 0.003356] | 0.3392 | NA (secondary) |
| `logistic29_temperature__vs__per_dyad_markov` | 0.060655 | 22.81% | [0.052706, 0.068604] | 1.432e-50 | 4.295e-50 |
| `logistic29_temperature__vs__warm_markov` | 0.004510 | 2.15% | [0.001423, 0.007597] | 0.004189 | NA (secondary) |
| `hgb49__vs__per_dyad_markov` | 0.040051 | 15.06% | [0.031092, 0.049010] | 1.923e-18 | 2.884e-18 |
| `hgb49__vs__logistic49_temperature` | -0.007948 | -3.65% | [-0.012708, -0.003188] | 0.001066 | 0.001066 |

BH families (plan-declared): A primary LL (4), B primary ECE/AUC/AP (3), C capacity/feature (3), D group attribution (9 LOO). See `tables/bh_families.csv`.

Full dyad, source-actor, target-actor, and multiway intervals for ECE, Brier, AUC, and AP are in `tables/primary_comparisons.csv` and `tables/auc_ap_deltas.csv`.

## 4. Within- vs between-dyad discrimination (key verdict)

- Primary within-dyad AUC (P1-compatible centered score): **0.5266**.
- Primary strict per-dyad **equal-dyad mean** AUC: **0.5139**; strict **pair-weighted** AUC: **0.5616** (3,310 dyads with both classes).
- Per-dyad Markov: within-dyad AUC 0.5185 (strict pair-weighted 0.5376; P1-compatible 0.5315).
- HGB49: within-dyad AUC 0.5169 (P1-compatible 0.5332).
- Paired strict within-dyad ΔAUC (primary − Markov): 0.0240 (95% dyad bootstrap CI [0.020156, 0.028349]).
- P1-compatible centered within-dyad ΔAUC: -0.0049 (95% CI [-0.019078, 0.008487]).

Within/between intervals use the dyad cluster bootstrap because the estimand is dyad-level. The primary LL/ECE/Brier/AUC/AP comparisons in Section 3 report dyad, source-actor, target-actor, and multiway intervals.

**Interpretation.** The conditional-logit check did not converge and provides no within-dyad evidence; the converged linear probability model gives P1 0.5386, strict equal-dyad mean 0.5382 and strict pair-weighted 0.5559, which we report descriptively. The global AUC cannot be interpreted as pure between-dyad propensity.

## 5. Dyad fixed-effects / conditional logit

Exact Chamberlain conditional logit on 16,559 of 21,152 training dyads (dyads with 1–60 positives; k median 5); 3,957,601 rows; 48 within-varying features. Optimizer success=False, nit=120.

| model | within-dyad AUC (P1) | strict mean | strict pair-weighted |
|---|---:|---:|---:|
| `primary logistic49` | 0.5266 | 0.5139 | 0.5616 |
| `per-dyad Markov` | 0.5315 | 0.5185 | 0.5376 |
| `HGB49` | 0.5332 | 0.5169 | 0.5610 |
| dyad FE conditional logit (did not converge) | 0.5561 | 0.5597 | 0.5759 |
| within-dyad OLS (LPM) | 0.5386 | 0.5382 | 0.5559 |

Strict mean = equal-dyad mean over the 3,310 eligible dyads; strict pair-weighted = pooled over all within-dyad pairs from those dyads. The conditional-logit row is shown for completeness only and is not used as evidence.

**Verdict.** The conditional logit did not converge; no within-dyad signal is claimed. The converged LPM is descriptive only.

## 6. Feature-group anchors and attribution

| anchor | LL | Brier | ECE | AUC | AP |
|---|---:|---:|---:|---:|---:|
| `intercept_only` | 0.335382 | 0.093738 | 0.006109 | 0.5000 | 0.1047 |
| `global_transition` | 0.265874 | 0.073497 | 0.003470 | 0.7323 | 0.3211 |
| `calendar_linear` | 0.348501 | 0.096936 | 0.056787 | 0.5054 | 0.1092 |
| `calendar_clipped` | 0.341058 | 0.095014 | 0.036169 | 0.4990 | 0.1045 |
| `calendar_year_fe_29` | 0.200633 | 0.057171 | 0.020368 | 0.9084 | 0.6724 |

| group | ΔLL without group | 95% dyad bootstrap CI | p |
|---|---:|---|---:|
| `calendar` | +0.000454 | [0.000405, 0.000502] | 0 |
| `dyad_current` | +0.002080 | [0.001719, 0.002479] | 0 |
| `dyad_lags` | +0.003768 | [0.003288, 0.004308] | 0 |
| `dyad_rolling` | +0.001196 | [0.000912, 0.001486] | 0 |
| `dyad_history` | +0.042582 | [0.039870, 0.045538] | 0 |
| `reverse_dyad` | +0.000805 | [0.000678, 0.000930] | 0 |
| `actor_ego` | +0.010241 | [0.009640, 0.010847] | 0 |
| `train_stats` | -0.024025 | [-0.026073, -0.022171] | 0 |
| `graph` | -0.012911 | [-0.014016, -0.011839] | 0 |

The `p` column is the floored bootstrap tail (1,000 replicates). Pre-registered BH family D uses the normal-approximation `p_normal` column of `tables/feature_group_loo.csv`: all nine `q < 1e-12` and all nine are rejected (see `tables/bh_families.csv` and `CORRECTIONS.md`).

**Cache policies.** The confirmatory holdout and this feature-group LOO use the **holdout-zeroed** graph cache: held-out edges are zeroed for every month, including the training snapshots, so the graph coefficients are estimated on the reduced network. The provisional stability audit uses the distinct **holdout-row-excluded** policy (eventual holdout rows removed from fitting and aggregation). These are different estimands and are not interchangeable.

**Grouped permutation** (all columns of a group permuted jointly; 10 permutations; dyad bootstrap CI):

| group | ΔLL permutation | 95% CI | SE |
|---|---:|---|---:|
| `calendar` | +0.000012 | [-0.000004, 0.000028] | 0.000009 |
| `dyad_current` | +0.012732 | [0.011555, 0.014016] | 0.000634 |
| `dyad_lags` | +0.018501 | [0.016705, 0.020489] | 0.000965 |
| `dyad_rolling` | +0.002124 | [0.001901, 0.002355] | 0.000116 |
| `dyad_history` | +0.102429 | [0.096401, 0.108897] | 0.003190 |
| `reverse_dyad` | +0.005498 | [0.005043, 0.006014] | 0.000245 |
| `actor_ego` | +0.026150 | [0.024392, 0.028076] | 0.000941 |
| `train_stats` | +0.000000 | [0.000000, 0.000000] | 0.000000 |
| `graph` | -0.001697 | [-0.002434, -0.000988] | 0.000375 |

Positive ΔLL means the group carries held-out signal. Full table: `tables/feature_group_permutation.csv`.

## 7. Calibration, capacity, and sensitivity

| sensitivity | LL | ECE | AUC | AP | note |
|---|---:|---:|---:|---:|---|
| `logistic49_S1_none` | 0.217875 | 0.042112 | 0.9102 | 0.6719 | raw log-odds |
| `logistic49_S1_temperature` | 0.217875 | 0.042112 | 0.9102 | 0.6719 | temperature (primary) |
| `logistic49_S1_isotonic` | 0.216895 | 0.040307 | 0.9092 | 0.6648 | isotonic (sensitivity only) |
| `logistic29_S1_temperature` | 0.205219 | 0.027463 | 0.9083 | 0.6717 | 29-feature logistic, temperature |
| `logistic49_S2_temperature` | 0.218494 | 0.042743 | 0.9101 | 0.6718 | S2 clipped calendar |
| `hgb49_none` | 0.225823 | 0.035326 | 0.8805 | 0.6293 | HGB49 capacity control |
| `hgb49_temperature` | 0.229387 | 0.042747 | 0.8805 | 0.6293 | HGB49 temperature |

HGB tuning parity: 49 features, 2×3 learning-rate × leaves grid plus max-iter refinement; selected params `{'learning_rate': 0.1, 'max_leaf_nodes': 63, 'max_iter': 120}`. Relative HGB vs logistic LL improvement = -3.648%.

## 8. MDE and stopping rules

- `logistic49_temperature__vs__per_dyad_markov`: paired MDE 0.002370 (0.89% of base LL); baseline-variance MDE 0.010773.
- `logistic49_temperature__vs__warm_markov`: paired MDE 0.003072 (1.46% of base LL); baseline-variance MDE 0.008806.

- **SR-F7** (multiway CI includes 0): **NO**.
- **Gain < 2% relative**: **NO** (18.05%).
- **HGB beats logistic > 0.5%**: **NO** (-3.648%).
- **Seed spread > 0.5%**: **NO** (point-estimate spread 0.000000; CI endpoint range 0.009582).
The five seeds vary only the bootstrap weights; the fitted models are deterministic, so the point estimates are identical. The CI endpoint range is reported for completeness, but it is not a model-seed spread.

## 9. Exact sentences the paper can use

**Status note (R2C10).** These sentences mirror
`summary.json#/paper_sentences`; the frozen JSON bytes are unchanged. The phrase
"internally hash-timestamped" predates the paper's Appendix B and is superseded
there: only the split carries an independently written pre-run timestamp, while
the analysis plan is internally hash-timestamped through the post-run manifest.
No number in these sentences changed.

> In a plan-declared, internally hash-timestamped, one-shot unit-fresh 20% unordered-dyad holdout (311,638 dyad-months; 32,614 positives; 5,282 directed dyads; 2018-01-–2022-11), the frozen temperature-calibrated 49-feature logistic model achieved log-loss 0.21788, Brier 0.06401, ECE 0.04211, AUC 0.9102, and AP 0.6719.

> Against the train-only per-dyad Markov baseline (log-loss 0.26587, which equals the global transition on unit-fresh dyads), the primary model reduced log-loss by 0.04800 (18.05% relative; 95% multiway CI [0.04333, 0.05267]; BH q<1e-12 (normal-approximation tail; 1,000 reps)), increased AUC by 0.17788 and AP by 0.35072, but its ECE was worse by 0.03864 (95% multiway CI [-0.05303, -0.02426]).

> Against the warm-history Markov baseline with 35 extra months of dyad history (log-loss 0.20973, ECE 0.01023), the primary model log-loss was worse by 0.00815 (model − comparator; 95% multiway CI [0.00233, 0.01396]) and its ECE was worse by 0.03188 (95% CI [0.02293, 0.04084]); its AUC was higher by 0.01598 and AP by 0.03502. The comparison is unequal in two directions: warm history has 35 additional months of outcomes, the model carries the 49-feature set; neither side is dominated.

> The plan-declared secondary 29-feature model (log-loss 0.20522, ECE 0.02746, AUC 0.9083, AP 0.6717) had a nominal log-loss improvement over warm history (Δ = −0.00451, model − comparator; 95% multiway CI [−0.00760, −0.00142]; Brier null; exploratory, not multiplicity-controlled). Its strict-mean within-dyad AUC is 0.4716, below chance. It is a dyad-level ranking candidate, not a new primary.

> A 29-feature year-fixed-effects anchor achieved log-loss 0.20063 and Brier 0.05717, better than the primary; it is a plan-declared descriptive anchor; its promotion and warm-history contrast are post-hoc, C-confounded and uncorrected; it is not a prospective model, and linear calendar extrapolation is a limitation.

> Within-dyad AUC was 0.5266 (P1-compatible), 0.5139 (strict mean) and 0.5616 (strict pair-weighted) for the primary; the warm-history levels were 0.5503, 0.5179 and 0.5364. Only the pair-weighted definition favors the primary; there is no robust within-dyad advantage. The conditional logit did not converge; the converged LPM gave 0.5386/0.5382/0.5559 descriptively.

> The confirmatory claim is limited to dyad-level event propensity/ranking; the within-dyad result is definition-dependent and exploratory, and we make no month-level timing, node-level, causal or lifecycle claim.

> Because the holdout is unit-fresh rather than time-fresh and the 2018-22 period was retrospectively burned by earlier runs, these results are retrospective and should not be presented as a prospective forecast.

## 10. Scope and limitations

- Binary dyadic event prediction only; no node-level prediction.
- Unit-fresh 20% unordered-dyad holdout, not a time-fresh forecast; 2018–22 was retrospectively burned by earlier G2/D2/B runs.
- The primary model is a transparent logistic model; HGB is a capacity control, not the interpretable model.
- The exact additive source/target actor explanations and stability audit are reported in the companion Sprint-B artifacts; this report covers prediction, calibration, and the within-dyad timing check.

## 11. Artifacts

- `parts/holdout_core.json` — one-shot core outputs.
- `preds/holdout_preds.npz` — row-level predictions and labels.
- `preds/holdout_fe_scores.npz` — dyad FE scores.
- `tables/*.csv` — metrics, comparisons, AUC/AP, BH, within/between, FE, LOO, anchors, permutation, calibration, HGB, seeds, MDE.
- `summary.json`, `manifest.json`, `run.log`, `test_access_log.json`.
