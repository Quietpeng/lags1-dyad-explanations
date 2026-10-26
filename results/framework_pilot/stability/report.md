# Sprint B — exact additive actor explanations and stability audit

> **Audit status.** The audit design was frozen at 2026-10-23T06:31Z. The confirmatory holdout completed after this audit at 07:45:19Z. The audit was recomputed after excluding all eventual holdout dyads from fitting and aggregation, with C fixed at the primary value 1.0 and 50 dyad-cluster bootstrap refits. The holdout was available_at_audit=false, completed later, and was not integrated or used. This is a pre-holdout design, not a holdout-side stability test.

**Status: `provisional_holdout_not_integrated`.** CPU only; no new data; no synthetic or T1 run; no test tuning. The month-block bootstrap uses 50 replicates; the seed axis uses 50 refits.

## 1. Exactness

- Decision-function max absolute error: 1.28864e-06 log-odds; raw max 3.12e-05.
- Temperature T=1.0; isotonic is outside the exact track.

- Inner shares: source 0.2158, target 0.2335, dyad 0.2974, graph 0.1696, calendar 0.0280, train 0.0557; actor-owned 0.4493.

- Source/target Pearson 0.997931, Spearman 0.997555, mean abs diff 0.036420.

## 2. Seed stability (50 refits; 1,225 pairs per period and view; inner-period values shown)

- source: Spearman median 0.9989 [0.9891, 1.0000]; J@10 median 1.000 [0.818, 1.000]; sign median 0.9796 [0.8878, 1.0000].

- target: Spearman median 0.9990 [0.9894, 1.0000]; J@10 median 1.000 [1.000, 1.000]; sign median 0.9745 [0.9082, 1.0000].

## 3. Period and window

- source inner→retro_2018_22: Spearman 0.9739, sign 0.9592, J@10 0.6667.

- source inner→covid_2020_21: Spearman 0.9650, sign 0.9439, J@10 0.5385.

- target inner→retro_2018_22: Spearman 0.9644, sign 0.9388, J@10 0.6667.

- target inner→covid_2020_21: Spearman 0.9522, sign 0.9286, J@10 0.5385.

- Periods are nested (COVID 2020–21 is inside burned 2018–22); the period component is a regime-window contrast, not a clean additive period effect.

## 4. Within-actor variance

- source: period/regime 0.323, seed 0.099, residual 0.577.

- target: period/regime 0.335, seed 0.053, residual 0.613.

- Random top-k null overlap: k=10 0.0262, k=20 0.0538 (descriptive, not a test).

## 5. Decision proxy (inner/burned, not holdout)

- Actor-month source attribution: lift 0.139 [0.117, 0.164].

- Actor-month source risk: lift 3.220 [3.175, 3.257].

- Actor-month target attribution: lift 0.151 [0.116, 0.184].

- Actor-month target risk: lift 3.203 [3.163, 3.253].

- Dyad risk k=10: lift 9.053, precision 1.000.

- Dyad risk k=1000: lift 7.439, precision 0.822.

- Dyad source k=10: lift 0.118, precision 0.013.

- Dyad source k=1000: lift 0.495, precision 0.055.

- Dyad target k=10: lift 0.039, precision 0.004.

- Dyad target k=1000: lift 0.559, precision 0.062.

## 6. HGB occlusion

- source: rank Spearman 0.542, top-10 log-occ Jaccard 0.176, top-10 absolute Jaccard 0.053.

- target: rank Spearman 0.862, top-10 log-occ Jaccard 0.429, top-10 absolute Jaccard 0.333.

Target agreement is broad; source agreement is moderate; top-10 set overlap is low. This is a rank check, not an exact decomposition.

## 7. Holdout and limits

The confirmatory holdout completed later and was not used. The audit is holdout-excluded and provisional; it makes no confirmatory stability or decision-benefit claim. The period factor is nested; the residual is an unexplained period×seed/actor×time interaction, not clean noise.
