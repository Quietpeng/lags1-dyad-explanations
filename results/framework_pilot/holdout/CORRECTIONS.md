# Corrections / errata

## 2026-10-26 — BH family D recomputed from the frozen normal-tail p-values

`tables/bh_families.csv` and the `bh` blocks of `summary.json` /
`parts/holdout_extras.json` carried family D rows with `p = q = 0.0` and
`p_reported = q_reported = "<0.001"`. Those were bootstrap tails floored at
zero, which contradicted the stated method (normal-approximation z-tails over
the nine group leave-one-out fits). The nine D rows now carry the frozen
`p_normal` value from `tables/feature_group_loo.csv` and its
Benjamini–Hochberg step-up `q` over m = 9. Arithmetic only: no refit, bootstrap,
label read or new data.

All nine groups remain rejected at 0.05, and the largest D `q` is 8.53e-16, so
every row still displays `<1e-12`. Family D therefore remains fully rejected;
families A–C, every ΔLL, every confidence interval and every metric are
unchanged (the CSV rows are byte-identical).

## 2026-10-23 — misattributed BH q in the 29-feature vs warm-history sentence

`report.md` and `summary.json` (`paper_sentences`) stated that the pre-registered
secondary 29-feature model beat the warm-history Markov baseline with
`BH q=4.295e-50`. That q value belongs to the **29-feature vs per-dyad Markov**
comparison (BH family C). The **29-feature vs warm-history Markov** comparison is
secondary and is not in a BH family.

Corrected to: **p=0.0042**, 95% multiway CI **[0.00142, 0.00760]**. The point
estimate (ΔLL +0.00451) and all other numbers are unchanged. No data,
predictions, core metrics, or model fits were changed.

## 2026-10-23 — within-dyad/calendar wording correction

The public summary/report strings were corrected to match the manuscript (no robust within-dyad advantage; calendar not claimed to drive the holdout differences). No frozen metric changed. Pair-cluster sensitivity and censored p/q display columns were added as post-hoc reporting aids.

## 2026-10-24 — final public-release wording sync

The release copies of `summary.json`, `parts/holdout_extras.json` and
`report.md` were synchronized with the corrected paper sentences: within-dyad
claim strings, calendar wording, and the model−comparator sign label for the
29-feature versus warm-history contrast. The sibling `summary` block in
`holdout_extras.json` now matches `summary.json`. No table, metric,
prediction, label, model output or
frozen number changed; `SHA256SUMS` and the manifest `current_sha256` fields
were regenerated for the annotated files only.
