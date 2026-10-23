# Sprint B — exact additive actor explanations and stability audit

**Status: `provisional_no_fresh_holdout`.** Generated 2026-10-23T06:31:40Z. CPU only; no new data; no synthetic or T1 run; no test tuning. 2016-17 is inner validation; 2018-22 and COVID are burned retrospective secondary views.

> The only confirmatory set is a fresh 20% dyad-holdout. It was not available to Sprint B. Every number below is therefore provisional and must not be written as a confirmatory headline.

| question | result | status |
|---|---|---|
| Exact additive identity | decision-function max error 1.4e-6; raw-vs-logit max 4.1e-5 (float32 p) | PASS |
| Source/target shares | 22.0% / 23.7% of mean absolute contribution; actor views correlate r=0.998 | descriptive |
| Seed stability | Spearman ≥0.996; signed top-10 Jaccard 0.82-1.00 | stable |
| Regime stability | Spearman 0.95-0.99; top-10 Jaccard 0.54-0.82; seed noise 4-5% of within-actor variance | conditional |
| Top-k / decision | risk lift 2.5-4.3 (actor-month), 7.9-9.1 (dyad); attribution lift 0.10-0.20 / 0.06-0.56; post-hoc explanation Spearman ≥0.996 | negative for alerting, positive for post-hoc |
| HGB occlusion | occlusion-vs-occlusion Spearman 0.69-0.82; top-10 absolute overlap 0.05-0.43 | broad rank agreement |
| Fresh holdout | no prediction artifact; Sprint A smoke run blind/quick | P6 blocked |

## 1. Exactness verdict

**Verdict: PASS.** For every evaluated row, the intercept plus the 49 feature contributions equals the model's own `decision_function` to numerical precision. The larger `raw` error against `logit(p)` is a float32 `predict_proba` round-trip, not an additive-decomposition failure. The selected temperature is T=1.0, so the temperature track coincides with the raw track; affine scaling C/T is exact by construction (the unit test at T=2 passes at 1e-12). Isotonic calibration is monotone but not affine and is excluded from the exact track.

| period | n | raw max | raw mean | raw >1e-4 | decision max | temperature max | stream max | stream >1e-6 |
|---|---|---|---|---|---|---|---|---|
| inner_2016_17 | 50000 | 4.06e-05 | 1.46e-07 | 0 | 1.20e-06 | 8.17e-14 | 6.62e-05 | 4331 |
| retro_2018_22 | 50000 | 3.54e-05 | 1.44e-07 | 0 | 1.18e-06 | 7.64e-14 | 1.67e-04 | 9642 |
| covid_2020_21 | 50000 | 3.79e-05 | 1.45e-07 | 0 | 1.42e-06 | 7.82e-14 | 4.41e-05 | 3194 |

_Table note: with the selected T=1.0, the `temperature max` column is the sigmoid/logit round-trip of the raw track, not an independent temperature fit; the affine identity itself is tested at T=2 in `tests/test_attribution.py`._

Reading the accumulator warning: the streaming accumulator checks `b + Σ C = logit(p)` at `1e-6`. `predict_proba` returns float32, so `logit(p)` carries a quantisation error of up to ~1.7e-4 in raw log-odds; the decomposition itself matches `decision_function` to ≤1.42e-6 (mean ≤1.24e-7). The strict `1e-6` stream flag should be read as a storage-precision diagnostic, not as a failed identity.

| period | LL raw | LL isotonic | mean abs logit distortion |
|---|---|---|---|
| inner_2016_17 | 0.204040 | 0.203378 | 0.214833 |
| retro_2018_22 | 0.197285 | 0.196786 | 0.211732 |
| covid_2020_21 | 0.182687 | 0.182361 | 0.225650 |

Isotonic gains 0.0003-0.0007 LL but moves the log-odds by ~0.21 on average; it cannot be decomposed into the same additive contributions and is reported only as a sensitivity check.

## 2. Source/target aggregation

The bucket contract assigns each feature to exactly one owner: source ego, target ego, dyad history, calendar, symmetric graph, or train-only statistics. The two actor buckets carry about 46% of the mean absolute contribution; the dyad block carries ~29%, graph ~16%.

| period | source actor | target actor | dyad | calendar | graph | train stats |
|---|---|---|---|---|---|---|
| inner_2016_17 | 0.2201 | 0.2374 | 0.2967 | 0.0315 | 0.1580 | 0.0563 |
| retro_2018_22 | 0.2196 | 0.2372 | 0.2921 | 0.0390 | 0.1565 | 0.0556 |
| covid_2020_21 | 0.2208 | 0.2387 | 0.2869 | 0.0403 | 0.1575 | 0.0558 |

Top positive and top negative attribution actors (top 5 per view; `mean_p` is the actor's mean forecast in the same period):

| view | side | actor | attribution | mean p | mean y | ego abs share |
|---|---|---|---|---|---|---|
| source | top positive | SMR | 0.673 | 0.009 | 0.002 | 0.623 |
| source | top positive | ATG | 0.507 | 0.008 | 0.003 | 0.606 |
| source | top positive | MCO | 0.385 | 0.011 | 0.007 | 0.593 |
| source | top positive | KNA | 0.333 | 0.010 | 0.008 | 0.615 |
| source | top positive | LIE | 0.321 | 0.014 | 0.009 | 0.615 |
| source | top negative | USA | -0.750 | 0.616 | 0.605 | 0.387 |
| source | top negative | CHN | -0.687 | 0.425 | 0.433 | 0.425 |
| source | top negative | GBR | -0.624 | 0.401 | 0.343 | 0.421 |
| source | top negative | JPN | -0.585 | 0.239 | 0.225 | 0.471 |
| source | top negative | RUS | -0.584 | 0.403 | 0.413 | 0.433 |
| target | top positive | SMR | 0.670 | 0.009 | 0.001 | 0.623 |
| target | top positive | ATG | 0.489 | 0.008 | 0.006 | 0.613 |
| target | top positive | MCO | 0.360 | 0.011 | 0.010 | 0.594 |
| target | top positive | ZAN | 0.304 | 0.015 | 0.016 | 0.608 |
| target | top positive | MHL | 0.291 | 0.013 | 0.012 | 0.597 |
| target | top negative | USA | -0.776 | 0.592 | 0.585 | 0.405 |
| target | top negative | CHN | -0.724 | 0.423 | 0.435 | 0.434 |
| target | top negative | GBR | -0.669 | 0.397 | 0.367 | 0.427 |
| target | top negative | FRA | -0.634 | 0.366 | 0.322 | 0.434 |
| target | top negative | JPN | -0.612 | 0.228 | 0.208 | 0.484 |

The same small-state set (SMR, ATG, MCO, KNA, LIE, DMA) dominates positive source and target attribution in every period; the major powers (USA, CHN, RUS, GBR, JPN, FRA) dominate the negative tail. This is an activity-level contrast, **not** a risk ranking: the top positive actors have mean predicted probabilities of 0.008-0.015, far below the ~0.11 period base rate, while USA/CHN have mean p of 0.4-0.6. Positive attribution means 'less negative than the training mean', not 'likely to act'.

Top contributors by period and view (full tables: `tables/actor_source_{period}.csv`, `tables/actor_target_{period}.csv`):

| period | view | top positive (attribution) | top negative (attribution) |
|---|---|---|---|
| inner_2016_17 | source | SMR (+0.673), ATG (+0.507), MCO (+0.385) | USA (-0.750), CHN (-0.687), GBR (-0.624) |
| inner_2016_17 | target | SMR (+0.670), ATG (+0.489), MCO (+0.360) | USA (-0.776), CHN (-0.724), GBR (-0.669) |
| retro_2018_22 | source | SMR (+0.603), KNA (+0.478), LIE (+0.427) | USA (-0.822), CHN (-0.763), RUS (-0.698) |
| retro_2018_22 | target | SMR (+0.601), KNA (+0.440), LIE (+0.396) | USA (-0.855), CHN (-0.806), JPN (-0.738) |
| covid_2020_21 | source | SMR (+0.582), KNA (+0.453), DMA (+0.438) | USA (-0.889), CHN (-0.818), JPN (-0.785) |
| covid_2020_21 | target | SMR (+0.581), DMA (+0.412), KNA (+0.411) | USA (-0.931), CHN (-0.867), JPN (-0.841) |

Source and target views are highly redundant at the actor level:

| period | n | Pearson | Spearman | mean abs diff |
|---|---|---|---|---|
| inner_2016_17 | 196 | 0.9980 | 0.9973 | 0.0376 |
| retro_2018_22 | 196 | 0.9986 | 0.9983 | 0.0405 |
| covid_2020_21 | 196 | 0.9984 | 0.9982 | 0.0460 |

The target view adds little ranking information beyond the source view for these 196 actors. Per-dyad contributions can still differ; the paper should report both views but must not present them as independent evidence.

## 3. Stability audit

Five replicates are **dyad-cluster bootstrap refits with C fixed**, not optimizer seeds. The lbfgs point fit is deterministic, so optimizer noise is exactly zero; the reported seed spread is cluster-resampling uncertainty.

| period | view | Spearman | min | J@10 | min | J@20 | sign | min |
|---|---|---|---|---|---|---|---|---|
| inner_2016_17 | source | 0.9994 | 0.9968 | 0.8182 | 0.8182 | 1.0000 | 0.9694 | 0.9541 |
| inner_2016_17 | target | 0.9998 | 0.9993 | 0.8182 | 0.8182 | 1.0000 | 0.9923 | 0.9796 |
| retro_2018_22 | source | 0.9996 | 0.9974 | 0.8182 | 0.8182 | 1.0000 | 0.9821 | 0.9643 |
| retro_2018_22 | target | 0.9998 | 0.9993 | 1.0000 | 0.8182 | 0.9048 | 0.9770 | 0.9643 |
| covid_2020_21 | source | 0.9996 | 0.9967 | 1.0000 | 0.8182 | 1.0000 | 0.9796 | 0.9337 |
| covid_2020_21 | target | 0.9998 | 0.9994 | 1.0000 | 1.0000 | 1.0000 | 0.9898 | 0.9745 |

Signed top-k Jaccard is more decision-relevant than absolute top-k. At k=10 the median seed-pair Jaccard is 0.82-1.00; k=5 is 0.67-1.00. The churn is concentrated at near-ties: the boundary gap at k=10 is 0.67-1.29 seed SD for actor-month attribution, versus 16-42 seed SD for risk scores. Rank correlation is ≥0.996 everywhere. Sign flips are absent for actors with |attribution| ≥0.10; at the 0.05 threshold one COVID source actor flips in one replicate, and the pairwise sign agreement is 0.93-1.00 (typically 0.97-0.99), with disagreement confined to near-zero actors.

Per-actor seed SD illustrates the same point: SMR's inner source attribution is 0.673 ± 0.044 and ATG's is 0.507 ± 0.054 (mean ± seed SD), so the top-two order is separated by ~3 seed SD; the KNA/LIE pair (0.333 vs 0.321, seed SD 0.038/0.027) is inside one seed SD. The churn is therefore concentrated in the middle of the top-10, not at the top.

Across periods (regime shift):

| view | period A | period B | Spearman | J@10 | sign |
|---|---|---|---|---|---|
| source | inner_2016_17 | retro_2018_22 | 0.9727 | 0.8182 | 0.9337 |
| source | inner_2016_17 | covid_2020_21 | 0.9643 | 0.6667 | 0.9235 |
| source | retro_2018_22 | covid_2020_21 | 0.9916 | 0.8182 | 0.9490 |
| target | inner_2016_17 | retro_2018_22 | 0.9620 | 0.6667 | 0.9286 |
| target | inner_2016_17 | covid_2020_21 | 0.9500 | 0.5385 | 0.9235 |
| target | retro_2018_22 | covid_2020_21 | 0.9889 | 0.8182 | 0.9541 |

Rank correlation stays high (0.95-0.99) but signed top-10 Jaccard falls to 0.54-0.82; sign agreement is 0.92-0.95. Regime change, not seed noise, is the dominant source of top-k churn.

D2 windows (median over seeds):

| period | view | window A | window B | Spearman | J@10 | sign |
|---|---|---|---|---|---|---|
| inner_2016_17 | source | W_b_dyad_onset | W_c_dormancy_reentry | 0.937 | 0.538 | 0.853 |
| inner_2016_17 | source | W_b_dyad_onset | W_d_stable | 0.933 | 0.667 | 0.921 |
| inner_2016_17 | source | W_b_dyad_onset | full_period | 0.943 | 0.667 | 0.916 |
| inner_2016_17 | source | W_c_dormancy_reentry | W_d_stable | 0.986 | 0.429 | 0.913 |
| inner_2016_17 | source | W_c_dormancy_reentry | full_period | 0.991 | 0.429 | 0.913 |
| inner_2016_17 | source | W_d_stable | full_period | 0.998 | 1.000 | 0.985 |
| inner_2016_17 | target | W_b_dyad_onset | W_c_dormancy_reentry | 0.911 | 0.333 | 0.818 |
| inner_2016_17 | target | W_b_dyad_onset | W_d_stable | 0.931 | 0.667 | 0.915 |
| inner_2016_17 | target | W_b_dyad_onset | full_period | 0.939 | 0.818 | 0.931 |
| inner_2016_17 | target | W_c_dormancy_reentry | W_d_stable | 0.966 | 0.333 | 0.887 |
| inner_2016_17 | target | W_c_dormancy_reentry | full_period | 0.972 | 0.333 | 0.876 |
| inner_2016_17 | target | W_d_stable | full_period | 0.998 | 1.000 | 0.990 |
| retro_2018_22 | source | W_b_dyad_onset | W_c_dormancy_reentry | 0.919 | 0.538 | 0.919 |
| retro_2018_22 | source | W_b_dyad_onset | W_d_stable | 0.924 | 0.429 | 0.952 |
| retro_2018_22 | source | W_b_dyad_onset | full_period | 0.931 | 0.429 | 0.946 |
| retro_2018_22 | source | W_c_dormancy_reentry | W_d_stable | 0.981 | 0.818 | 0.913 |
| retro_2018_22 | source | W_c_dormancy_reentry | full_period | 0.990 | 0.818 | 0.934 |
| retro_2018_22 | source | W_d_stable | full_period | 0.997 | 0.818 | 0.985 |
| retro_2018_22 | target | W_b_dyad_onset | W_c_dormancy_reentry | 0.911 | 0.538 | 0.908 |
| retro_2018_22 | target | W_b_dyad_onset | W_d_stable | 0.913 | 0.429 | 0.957 |
| retro_2018_22 | target | W_b_dyad_onset | full_period | 0.925 | 0.429 | 0.940 |
| retro_2018_22 | target | W_c_dormancy_reentry | W_d_stable | 0.977 | 0.667 | 0.923 |
| retro_2018_22 | target | W_c_dormancy_reentry | full_period | 0.988 | 0.818 | 0.939 |
| retro_2018_22 | target | W_d_stable | full_period | 0.997 | 0.818 | 0.974 |

Window stability is conditional: `W_d_stable` vs the full period is essentially identical (J@10 ~1.0), but onset and dormancy-reentry windows disagree with each other and with the full period (J@10 0.33-0.82). Explanations for cold-start/onset cases should not be pooled with stable-history cases.

Variance decomposition within actor (period × seed):

| view | component | share |
|---|---|---|
| source | period_regime | 0.3548 |
| source | seed_noise | 0.0510 |
| source | actor_x_time_resid | 0.5942 |
| source | regime_var_over_seed_var_median | 9.3981 |
| source | share_actors_ratio_gt_2 | 0.7857 |
| target | period_regime | 0.3677 |
| target | seed_noise | 0.0406 |
| target | actor_x_time_resid | 0.5917 |
| target | regime_var_over_seed_var_median | 12.7570 |
| target | share_actors_ratio_gt_2 | 0.8776 |

Seed noise is 4.1-5.1% of within-actor variance; period/regime is 35.5-36.8%; the actor×time residual is 59-60%. The median regime/seed variance ratio is 9.4 (source) and 12.8 (target); 79-88% of actors have a ratio above 2. This is the central stability finding: the audit is not seed-limited, it is regime-limited.

Coefficient-space check: the exact linearisation `Var_seed(a_i) = z̄_i' Σ_β z̄_i` reproduces the empirical seed SD with ratio 1.000 and correlation 1.000 in every period/view. Collinearity does not create the seed spread; the covariance ratio is 0.39-0.80 at the median and 0.84-1.41 at the 90th percentile, so for most actors coefficient covariance damps seed variance, while the top decile is amplified by up to ~1.4×. The first principal component of Σ_β explains 21-62% of the linear seed variance.

Collinearity is severe in the design matrix (condition number 270,893; max VIF 8,329; 32 features with VIF>10), but the coefficient-space check shows it does not translate into unstable actor rankings: the covariance among coefficients is negative-signed for most actors and the empirical seed SD is reproduced exactly by the linearisation. The instability that remains is regime-driven, not collinearity-driven.

Bootstrap CIs (month-block for period comparisons, actor-cluster for seed pairs): the period-pair Spearman CIs are narrow and exclude 0.90 in most cells; top-10 Jaccard CIs are wide (e.g. inner→COVID source [0.33,0.82]). The rank ordering is stable; the exact membership of a 10-actor review set is not.

## 4. Top-k churn and decision impact

We distinguish two alert rules: `risk` ranks actor-months by mean predicted probability; `attribution` ranks them by the actor's own bucket contribution. The latter is an explanation score, not a risk score.

| period | view | score | k | seed Jaccard | churn | boundary gap / seed SD |
|---|---|---|---|---|---|---|
| inner_2016_17 | source | attribution | 10 | 0.906 | 0.094 | 0.839 |
| inner_2016_17 | source | risk | 10 | 1.000 | 0.000 | 35.294 |
| inner_2016_17 | target | attribution | 10 | 0.954 | 0.046 | 1.293 |
| inner_2016_17 | target | risk | 10 | 0.995 | 0.005 | 41.748 |
| retro_2018_22 | source | attribution | 10 | 0.908 | 0.092 | 1.003 |
| retro_2018_22 | source | risk | 10 | 0.998 | 0.002 | 24.583 |
| retro_2018_22 | target | attribution | 10 | 0.955 | 0.045 | 1.082 |
| retro_2018_22 | target | risk | 10 | 1.000 | 0.000 | 34.517 |
| covid_2020_21 | source | attribution | 10 | 0.927 | 0.073 | 1.205 |
| covid_2020_21 | source | risk | 10 | 1.000 | 0.000 | 15.860 |
| covid_2020_21 | target | attribution | 10 | 0.967 | 0.033 | 0.667 |
| covid_2020_21 | target | risk | 10 | 1.000 | 0.000 | 33.205 |

| period | view | k | alert | mean lift | CI low | CI high | Jaccard with risk |
|---|---|---|---|---|---|---|---|
| inner_2016_17 | source | 10 | attribution | 0.144 | 0.117 | 0.177 | 0.000 |
| inner_2016_17 | source | 10 | risk | 3.235 | 3.196 | 3.275 | 1.000 |
| inner_2016_17 | target | 10 | attribution | 0.156 | 0.123 | 0.193 | 0.000 |
| inner_2016_17 | target | 10 | risk | 3.231 | 3.176 | 3.280 | 1.000 |
| retro_2018_22 | source | 10 | attribution | 0.153 | 0.131 | 0.175 | 0.000 |
| retro_2018_22 | source | 10 | risk | 3.319 | 3.267 | 3.371 | 1.000 |
| retro_2018_22 | target | 10 | attribution | 0.161 | 0.142 | 0.181 | 0.000 |
| retro_2018_22 | target | 10 | risk | 3.256 | 3.209 | 3.303 | 1.000 |
| covid_2020_21 | source | 10 | attribution | 0.116 | 0.098 | 0.136 | 0.000 |
| covid_2020_21 | source | 10 | risk | 3.429 | 3.339 | 3.530 | 1.000 |
| covid_2020_21 | target | 10 | attribution | 0.139 | 0.112 | 0.168 | 0.000 |
| covid_2020_21 | target | 10 | risk | 3.330 | 3.239 | 3.424 | 1.000 |

The risk alert is essentially seed-invariant (actor-month Jaccard 0.995-1.000; dyad Jaccard 0.98-1.00; boundary gap 16-42 seed SD for actor-month scores) and yields lift 2.5-4.3 for actor-month review and 7.9-9.1 for dyad review. The attribution alert is stable enough to be reproducible but is **anti-useful as an alert rule**: lift 0.10-0.20 at actor-month level, 0.06-0.56 for source/target dyad buckets, and Jaccard overlap with the risk alert is 0.000-0.003. This is not a bug: top positive attribution selects low-activity microstates whose probabilities are below the base rate.

Dyad-level churn (strict inner window; unique dyads):

| score | k | seed Jaccard | point vs seed | boundary gap / seed SD |
|---|---|---|---|---|
| total | 10 | 0.927 | 0.964 | 0.501 |
| total | 50 | 0.984 | 0.961 | 0.184 |
| total | 100 | 1.000 | 1.000 | 1.600 |
| source | 10 | 0.824 | 0.891 | 0.060 |
| source | 50 | 0.913 | 0.946 | 0.027 |
| source | 100 | 0.892 | 0.921 | 0.149 |
| target | 10 | 0.891 | 0.927 | 0.050 |
| target | 50 | 0.931 | 0.938 | 0.185 |
| target | 100 | 0.928 | 0.946 | 0.032 |
| risk | 10 | 1.000 | 1.000 | 1.007 |
| risk | 50 | 0.977 | 0.984 | 0.656 |
| risk | 100 | 1.000 | 1.000 | 0.578 |

Dyad-level decision impact (top-k dyads; all rows of a selected dyad are reviewed):

| alert | k | coverage | precision | recall | lift | Jaccard with risk |
|---|---|---|---|---|---|---|
| risk | 10 | 0.000 | 1.000 | 0.003 | 9.067 | 1.000 |
| risk | 50 | 0.002 | 1.000 | 0.017 | 9.067 | 1.000 |
| risk | 100 | 0.004 | 1.000 | 0.034 | 9.067 | 1.000 |
| risk | 500 | 0.019 | 0.964 | 0.165 | 8.736 | 1.000 |
| risk | 1000 | 0.038 | 0.874 | 0.300 | 7.921 | 1.000 |
| total | 10 | 0.000 | 1.000 | 0.003 | 9.067 | 1.000 |
| total | 50 | 0.002 | 1.000 | 0.017 | 9.067 | 0.961 |
| total | 100 | 0.004 | 1.000 | 0.034 | 9.067 | 0.980 |
| total | 500 | 0.019 | 0.964 | 0.165 | 8.738 | 0.988 |
| total | 1000 | 0.038 | 0.873 | 0.300 | 7.920 | 0.990 |
| source | 10 | 0.000 | 0.013 | 0.000 | 0.118 | 0.000 |
| source | 50 | 0.002 | 0.007 | 0.000 | 0.063 | 0.000 |
| source | 100 | 0.004 | 0.020 | 0.001 | 0.181 | 0.000 |
| source | 500 | 0.019 | 0.039 | 0.007 | 0.353 | 0.000 |
| source | 1000 | 0.038 | 0.054 | 0.019 | 0.493 | 0.002 |
| target | 10 | 0.000 | 0.009 | 0.000 | 0.079 | 0.000 |
| target | 50 | 0.002 | 0.007 | 0.000 | 0.063 | 0.000 |
| target | 100 | 0.004 | 0.013 | 0.000 | 0.122 | 0.000 |
| target | 500 | 0.019 | 0.042 | 0.007 | 0.383 | 0.000 |
| target | 1000 | 0.038 | 0.062 | 0.021 | 0.564 | 0.003 |

At the dyad level, the `total` attribution is a monotone transform of mean risk (mean total contribution = mean log-odds minus the intercept), so its top-k is trivially the risk top-k. The non-trivial explanation alerts are the source/target bucket views; they achieve lift 0.06-0.56, again below random review.

Decision verdict: use the risk score to choose what to review; attach the exact actor decomposition afterwards to explain why. Do not use the attribution top-k as a thresholded alert rule. This is a negative decision-impact result, not a failed stability result.

Conditional explanation stability inside the risk alert set:

| period | view | k | cells | seed Spearman | min | sign |
|---|---|---|---|---|---|---|
| inner_2016_17 | source | 5 | 115 | 0.9987 | 0.9972 | 1.0000 |
| inner_2016_17 | source | 10 | 230 | 0.9994 | 0.9984 | 1.0000 |
| inner_2016_17 | source | 20 | 460 | 0.9996 | 0.9984 | 1.0000 |
| inner_2016_17 | target | 5 | 115 | 0.9996 | 0.9978 | 1.0000 |
| inner_2016_17 | target | 10 | 230 | 0.9999 | 0.9992 | 1.0000 |
| inner_2016_17 | target | 20 | 460 | 0.9999 | 0.9995 | 0.9978 |
| retro_2018_22 | source | 5 | 295 | 0.9993 | 0.9981 | 1.0000 |
| retro_2018_22 | source | 10 | 590 | 0.9996 | 0.9979 | 1.0000 |
| retro_2018_22 | source | 20 | 1180 | 0.9997 | 0.9984 | 1.0000 |
| retro_2018_22 | target | 5 | 295 | 0.9998 | 0.9991 | 1.0000 |
| retro_2018_22 | target | 10 | 590 | 0.9999 | 0.9996 | 1.0000 |
| retro_2018_22 | target | 20 | 1180 | 0.9999 | 0.9997 | 1.0000 |
| covid_2020_21 | source | 5 | 120 | 0.9986 | 0.9963 | 1.0000 |
| covid_2020_21 | source | 10 | 240 | 0.9994 | 0.9977 | 1.0000 |
| covid_2020_21 | source | 20 | 480 | 0.9997 | 0.9986 | 1.0000 |
| covid_2020_21 | target | 5 | 120 | 0.9992 | 0.9985 | 1.0000 |
| covid_2020_21 | target | 10 | 240 | 0.9998 | 0.9995 | 1.0000 |
| covid_2020_21 | target | 20 | 480 | 0.9999 | 0.9997 | 1.0000 |

Once the risk score has selected the review set, the source/target attribution rankings **within that set** are essentially seed-invariant (Spearman 0.996-1.000; sign agreement 0.998-1.000). This is the constructive decision result: explanations are stable enough for post-hoc attribution of a risk alert, but not for choosing the alert set.

The same check on the dyad risk alert set:

| k | view | dyads | seed Spearman | min | sign |
|---|---|---|---|---|---|
| 10 | source | 10 | 1.0000 | 1.0000 | 1.0000 |
| 10 | target | 10 | 1.0000 | 1.0000 | 1.0000 |
| 50 | source | 50 | 0.9989 | 0.9973 | 0.9800 |
| 50 | target | 50 | 0.9991 | 0.9985 | 1.0000 |
| 100 | source | 100 | 0.9993 | 0.9977 | 0.9900 |
| 100 | target | 100 | 0.9996 | 0.9993 | 0.9850 |
| 500 | source | 500 | 0.9996 | 0.9981 | 0.9790 |
| 500 | target | 500 | 0.9997 | 0.9995 | 0.9840 |

Within the top-k risk dyads, source/target attribution rankings are also seed-invariant (Spearman ≥0.997; sign agreement 0.98-1.00). The post-hoc use therefore survives the finer-grained decision unit.

## 5. HGB occlusion agreement

HGB is refit on the current graph cache and is **not** the P1 HGB artifact (the P1 predictions were produced before the current graph cache; see `summary_decompose.json`). Its accuracy is descriptive context for the occlusion check; no HGB-vs-logistic superiority or parity claim is made. The selected 150 iterations are the grid edge.

| period | n | pos | iters | select LL | LL | ECE | AUC | AP |
|---|---|---|---|---|---|---|---|---|
| inner_2016_17 | 607982 | 67057 | 150 | 0.2306 | 0.2021 | 0.0056 | 0.9106 | 0.6707 |
| retro_2018_22 | 1559606 | 161539 | 150 | 0.2306 | 0.1968 | 0.0040 | 0.9078 | 0.6535 |
| covid_2020_21 | 634416 | 59253 | 150 | 0.2306 | 0.1829 | 0.0040 | 0.9115 | 0.6377 |

Actor-level HGB block occlusion (`ΔNLL` when the source/target feature block is replaced by its training mean). The signed attribution–ΔNLL column is **negative by construction of the loss scale** in a zero-dominated sample; use the like-for-like occlusion–occlusion column, not the signed attribution column, to judge model agreement.

| period | view | n | occl–occl Spearman | CI low | CI high | sign | J@10 (abs occl) | abs attr vs abs HGB | J@10 (abs attr) | signed attr–HGB |
|---|---|---|---|---|---|---|---|---|---|---|
| inner_2016_17 | source | 196 | 0.707 | 0.611 | 0.780 | 0.827 | 0.176 | 0.435 | 0.053 | -0.532 |
| inner_2016_17 | target | 196 | 0.811 | 0.729 | 0.871 | 0.821 | 0.333 | 0.662 | 0.176 | -0.688 |
| retro_2018_22 | source | 196 | 0.750 | 0.657 | 0.821 | 0.816 | 0.176 | 0.501 | 0.111 | -0.563 |
| retro_2018_22 | target | 196 | 0.694 | 0.584 | 0.783 | 0.796 | 0.429 | 0.666 | 0.250 | -0.586 |
| covid_2020_21 | source | 196 | 0.825 | 0.759 | 0.872 | 0.811 | 0.176 | 0.652 | 0.111 | -0.665 |
| covid_2020_21 | target | 196 | 0.781 | 0.686 | 0.851 | 0.781 | 0.176 | 0.705 | 0.250 | -0.707 |

Actor-month agreement (pooled cells; per-cell mean ΔNLL):

| period | view | n | occl–occl Spearman | CI low | CI high | sign | J@10 (abs occl) | abs attr vs abs HGB |
|---|---|---|---|---|---|---|---|---|
| inner_2016_17 | source | 4508 | 0.669 | 0.635 | 0.703 | 0.764 | 0.111 | 0.284 |
| inner_2016_17 | target | 4508 | 0.692 | 0.665 | 0.718 | 0.769 | 0.176 | 0.435 |
| retro_2018_22 | source | 11564 | 0.682 | 0.654 | 0.712 | 0.756 | 0.111 | 0.369 |
| retro_2018_22 | target | 11564 | 0.670 | 0.644 | 0.696 | 0.751 | 0.053 | 0.438 |
| covid_2020_21 | source | 4704 | 0.742 | 0.695 | 0.786 | 0.774 | 0.111 | 0.455 |
| covid_2020_21 | target | 4704 | 0.714 | 0.663 | 0.763 | 0.753 | 0.053 | 0.482 |

Worst like-for-like rank-divergence cases (logistic occlusion vs HGB occlusion; absolute-rank gap; not causal):

| period | view | actor | rows | logistic ΔNLL | HGB ΔNLL | rank gap | sign flip |
|---|---|---|---|---|---|---|---|
| inner_2016_17 | source | IRQ | 3818 | 0.01338 | -0.00001 | 181 | True |
| inner_2016_17 | source | ITA | 4117 | 0.00964 | -0.00007 | 159 | True |
| inner_2016_17 | source | MDA | 1679 | -0.00049 | -0.00308 | 155 | False |
| inner_2016_17 | source | SAU | 3657 | 0.00760 | 0.00019 | 133 | False |
| inner_2016_17 | source | NOR | 3772 | 0.00404 | 0.00001 | 131 | False |
| inner_2016_17 | source | GRC | 3726 | 0.00548 | -0.00019 | 117 | True |
| inner_2016_17 | source | RUS | 4439 | 0.00791 | 0.00035 | 114 | False |
| inner_2016_17 | source | SMR | 1012 | -0.00262 | 0.00000 | 107 | True |
| inner_2016_17 | source | LKA | 3450 | 0.00092 | 0.00188 | 106 | False |
| inner_2016_17 | source | QAT | 3749 | 0.00328 | 0.00009 | 106 | False |
| inner_2016_17 | target | IRN | 4048 | 0.01653 | 0.00024 | 160 | False |
| inner_2016_17 | target | PRT | 3956 | 0.00302 | 0.00011 | 103 | False |

Interpretation: HGB block occlusion reproduces the logistic occlusion ranking at Spearman 0.69-0.82 (actor level; month-block/actor-bootstrap CIs exclude 0), and at 0.67-0.74 for actor-month cells. Top-10 absolute-importance overlap is only 0.05-0.43, so the agreement is broad-ordering, not set-level. HGB selected 150 iterations at the grid edge (select LL still improving); it is a capacity control, not a tuned competitor. Signed attribution and ΔNLL are not the same scale: because most rows are zeros, a positive attribution actor can have a negative ΔNLL when the block is removed. HGB is a capacity control, not an exact decomposer; block replacement can be absorbed by correlated graph features, and the worst disagreements are concentrated in actors (e.g. IRQ, ITA, MDA, SMR) whose graph and actor blocks are plausibly redundant.

## 6. Fresh holdout integration

No fresh-holdout prediction artifact exists. The Sprint A run in `results/framework_pilot/holdout/run.log` was a `--blind --quick` smoke run and stopped on a broken pipe after point metrics; `holdout/preds/` is empty. Sprint B therefore makes **no confirmatory claim**, and P6 remains blocked. The stability results above are inner-validation + burned-retrospective evidence only.

## 7. Sentences the paper can use

Each sentence is tagged with its evidence status. Do not upgrade a `provisional` sentence to confirmatory without the fresh holdout.

- **Exactness (supported).** The logistic forecast is exactly additive: the intercept plus 49 feature contributions reproduces the model's decision function to at most 1.4e-6 in log-odds; temperature scaling is affine and therefore preserves the identity, and isotonic calibration is reported only as a sensitivity check because it is not affine. `[supported (inner + burned secondary)]`

- **Source/target aggregation (supported, with a redundancy caveat).** Source-ego and target-ego features each carry about 22-24% of the mean absolute contribution; the two actor views rank actors almost identically (r=0.998), so they should be presented as two bookkeeping views of the same forecast, not as independent evidence. `[supported (inner + burned secondary)]`

- **Activity, not risk (supported).** The positive attribution tail is dominated by low-activity microstates (SMR, ATG, MCO, KNA, LIE) whose mean predicted probabilities are 0.008-0.015, while the negative tail is dominated by high-activity major powers (USA, CHN, RUS, GBR, JPN); positive attribution means 'above the training mean', not 'likely to act'. `[supported (inner + burned secondary)]`

- **Seed stability (supported).** Across five dyad-cluster bootstrap refits, actor-attribution rank correlation is at least 0.996 and signed top-10 Jaccard is 0.82-1.00; no actor with |attribution| ≥0.10 changes sign, and pairwise sign agreement is 0.93-1.00, with disagreement confined to near-zero actors. `[supported (inner + burned secondary)]`

- **Regime-limited, not seed-limited (supported).** Seed noise accounts for only 4-5% of within-actor variance, versus 35-37% for period and 59-60% for actor×time residual; the median regime/seed variance ratio is 9.4-12.8, and 79-88% of actors exceed a ratio of 2. `[supported (inner + burned secondary)]`

- **Top-k churn (supported, conditional).** Signed top-10 actor sets have a seed-pair Jaccard of 0.82-1.00 but a cross-period Jaccard of 0.54-0.82; the churn is concentrated at near-ties, with a k=10 boundary gap of 0.67-1.29 seed SD for actor-month attribution versus 16-42 seed SD for risk scores. `[supported (inner + burned secondary)]`

- **Decision impact (supported negative result).** Risk-based top-k alerts are seed-invariant (actor-month Jaccard 0.995-1.000; dyad Jaccard 0.98-1.00) and achieve lift 2.5-9.1 over the base rate; attribution-based alerts achieve lift 0.10-0.20 at actor-month level and 0.06-0.56 for source/target dyad buckets, and overlap the risk alert almost never. Use explanations after the alert, not as the alert rule. `[supported (inner + burned secondary)]`

- **Post-hoc explanation stability (supported).** Conditional on the risk-based alert set, source/target attribution rankings within that set are essentially seed-invariant (actor-month Spearman 0.996-1.000; dyad Spearman ≥0.997; sign agreement 0.98-1.00), so the exact decomposition is stable enough to attach to an alert after the fact. `[supported (inner + burned secondary)]`

- **HGB occlusion (supported as a rank-agreement diagnostic, not as exactness).** HGB block occlusion reproduces the logistic occlusion ranking at Spearman 0.69-0.82 across actors (0.67-0.74 across actor-month cells), with top-10 absolute-importance overlap of only 0.05-0.43; signed attribution and occlusion ΔNLL are not the same scale because the evaluation set is zero-dominated. `[supported as rank-agreement diagnostic]`

- **Confirmatory status (provisional).** All stability results are inner-validation and burned-retrospective evidence; the fresh 20% dyad-holdout was not available, so no confirmatory claim about explanation stability or decision impact is made. `[provisional; P6 blocked]`

## 8. What not to say

- Do not call the actor attribution a node-level prediction or a risk score; it is a conditional decomposition of a dyad forecast.
- Do not claim that source and target views are independent; they are nearly collinear at the actor level.
- Do not claim that explanations improve decisions; the decision-impact proxy points the other way.
- Do not claim 2018-22 or COVID as confirmatory; they are burned.
- Do not claim HGB occlusion is an exact decomposition or a causal effect; it is a block-replacement rank check, and signed attribution vs ΔNLL is not a like-for-like comparison.
- Do not claim HGB superiority or parity from the Sprint B refit; it uses a changed graph cache and is a capacity control, not a tuned competitor.
- Do not claim stability generalizes beyond ICEWS + this 196-actor roster.

## 9. Files

`summary.json`, `summary_decompose.json`, `summary_stability.json`, `summary_hgb.json`, `run.log`, `report.md`, and the `tables/*.csv` listed in `summary.json`. The full contribution cache for the inner window is `framework/cache/sprint_b/contrib_inner_point.npz` (119 MB); it is a cache, not a reported table.

## 10. Reproduction

From `repo/`, with the CPU-only virtual environment:

```bash
PY=python
$PY experiments/sprint_b_decompose.py      # exactness + actor tables (~8 min)
$PY experiments/sprint_b_stability.py      # stability + top-k (~1 min)
$PY experiments/sprint_b_hgb.py            # HGB fit + occlusion (~12 min)
$PY experiments/sprint_b_hgb_post.py       # per-row normalization + dyad (~1 min)
$PY experiments/sprint_b_report.py         # report.md + summary.json
```

The HGB post-process is separate because the first HGB run wrote the occlusion caches before a dyad-shape bug stopped the script; the bug is fixed in `sprint_b_hgb.py`, and the post-process recomputes the exposure-neutral agreement without refitting. `run.log` preserves both the crash and the recovery for audit.
