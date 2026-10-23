# Corrections / errata

## 2026-10-23 — misattributed BH q in the 29-feature vs warm-history sentence

`report.md` and `summary.json` (`paper_sentences`) stated that the pre-registered
secondary 29-feature model beat the warm-history Markov baseline with
`BH q=4.295e-50`. That q value belongs to the **29-feature vs per-dyad Markov**
comparison (BH family C). The **29-feature vs warm-history Markov** comparison is
secondary and is not in a BH family.

Corrected to: **p=0.0042**, 95% multiway CI **[0.00142, 0.00760]**. The point
estimate (ΔLL +0.00451) and all other numbers are unchanged. No data,
predictions, core metrics, or model fits were changed.
