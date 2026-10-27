# Corrections / errata

## 2026-10-28 — reader-facing status wording qualified (no number or plan-hash change)

Reader-facing surfaces (the two `paper_sentences`, their mirrors in `report.md`,
and the public README/Zenodo/CFF/notebook/reproduction documents) used a status
adjective that the design cannot establish. They now use the paper's
vocabulary: the plan is plan-declared and internally hash-timestamped, the split
is pre-specified, the holdout is one-shot, and only the split carries an
independently written pre-run timestamp.

The frozen plan file `preregistration.json` is **byte-identical** to its R2C8
state, SHA-256
`41bda0e3eaf8bb32e398c260169b85715e8c82e04569a50a185ae49d0783704c`. Its
internal `S1_primary`/`S2_sensitivity` self-descriptions were not edited, and
`preregistration.json.sha256` keeps the same digest and repo-relative path.
There is **no plan-hash drift**: `summary.json#/holdout/prereg_file_sha256`,
the `holdout_extras.json` summary mirror, `report.md`
(`Pre-registration file SHA-256`), `manifest.json#/prereg.file_sha256`,
`manifest.json#/inputs` and `SHA256SUMS` all still carry `41bda0e3…`.

No definition, split, endpoint, feature set, number, count or timestamp
changed. Only the reader-facing prose files (`report.md`, `summary.json`,
`holdout_extras.json` and the public documents) moved; their shipped hashes and
the corresponding `current_sha256` entries were reconciled as usual.

## 2026-10-27 — public wording and sign-convention clarification (no number changed)

The public holdout report prints the 29-feature versus warm-history contrast as
`+0.00451` under the frozen table convention **comparator − model; positive =
secondary better**. The manuscript prints the same contrast under the opposite
convention, **model − comparator**: **−0.004510, 95% CI [−0.00760, −0.00142]**.
The two statements are the same interval with the sign flipped; no point
estimate, interval endpoint or table value changed, and the CSV keeps
`+0.004510` with CI `[0.001423, 0.007597]`.

The same revision relabels the strict within-dyad column as **pair-weighted**
(0.5616) and states the strict equal-dyad mean (0.5139) separately; marks the
non-converged conditional-logit row in `tables/dyad_fe.csv` with
`converged=false` and `used_as_evidence=false`; replaces the previous one-sided
comparator wording with “unequal in two directions”; and
adds the raw-panel versus filtered-row support footnote to `results/README.md`.
Text and metadata only: no metric, prediction, label, model output or frozen
number changed. `SHA256SUMS` and the manifest `current_sha256` annotations were
regenerated for the annotated files only.

## 2026-10-26 — stability audit status relabelled (no number changed)

`stability/summary.json` (`#/scope/audit_design`, `#/holdout/audit_design`) and
`stability/summary_hgb.json` (`#/holdout/audit_design`) carried
`pre_holdout_holdout_excluded`. That literal is false. The grid was frozen at
`2026-10-23T06:31:40Z`, i.e. **after** the confirmatory label read began
(`2026-10-23T05:26:55.031577Z`) and **before** it completed
(`2026-10-23T07:45:19Z`); the 50-refit rerun then ran `14:30`–`16:07Z`.

Corrected literal, in both JSONs and every public prose site:
`frozen_during_confirmatory_run_holdout_rows_excluded`. `#/holdout` now also
carries `confirmatory_run_start_utc` and a note naming start / freeze /
completion / rerun. The audit remains non-confirmatory, the holdout remains
`used=false`, and all 5,282 directed dyads (2,790 unordered pairs) are still
excluded at row level before fitting and aggregation. No stability estimate,
interval, CI, refit count, C, seed, CSV cell or table row changed;
`SHA256SUMS` and the manifest `current_sha256` entries were regenerated for the
edited files. `make verify` and `make stability-check` now fail if the
superseded literal reappears. Paper text (`manuscript.md:101`, `:364`) is
unchanged and remains correct.

## 2026-10-26 — BH family-B (and residual within/between) p-values: zero floor

`tables/bh_families.csv` row `B_primary_other_metrics, d_ap vs per_dyad_markov`
and `tables/auc_ap_deltas.csv` column `d_ap_multiway_p` carried `0.0`.
Both cells are two-sided normal tails `2*Phi(-z)` of the stored multiway
standard errors: `z = 0.35072182017821874 / 0.004128971364976906 = 84.9417`
(primary) and `z = 0.35060597902021906 / 0.004254887234907397 = 82.4008`
(29-feature). The true tails are ~1.5e-1569 and ~4e-1477, i.e. below the
smallest float64 (~4.9e-324), so the stored literal was a numerical artefact,
not a p-value; a p-value is never exactly zero.

The cells now carry the strict upper bound `<1e-300` and an explicit
`censoring` column (`p_normal_tail_below_float64` /
`d_ap_multiway_p_normal_tail_below_float64`). Same treatment for the two
`summary.json` / `parts/holdout_extras.json` `bh` mirrors.

`tables/within_between_deltas.csv` (and the `summary.json` /
`holdout_extras.json` `within_between.paired` / `within_deltas` mirrors)
carried `p_mean`/`p_pooled = 0.0`. Those are two-sided **paired bootstrap**
tails over 1,000 draws: zero hits on one side means `p < 1/1001`, not
`p = 0`. They now carry `<0.001` plus the censoring token
`two_sided_paired_bootstrap_tail_zero_of_1000`.

No refit, bootstrap rerun, label read or new data. Every repaired p-value was
already the smallest in its family; every family still rejects at 0.05, no
`d_ap`, `d_pooled`, CI, `se`, ordering or decision changes, and the
`reject_0.05` column is unchanged. `framework/sprint_a_inference.py` gained
`p_display` / `boot_p_display` so a re-run reproduces the convention instead of
reintroducing a literal zero.

## 2026-10-26 — §5.2 top-10 overlap range: absolute attribution is 0.00–0.33

Two public generator/doc sentences quoted "top-10 absolute-importance overlap
0.05–0.43" — a range that belongs to **log-occlusion**, not to absolute
attribution. Recomputed from `stability/tables/hgb_occlusion_agreement.csv`
(`level = actor`, 3 periods × 2 roles):

| metric | min | max |
|---|---:|---:|
| `jaccard_logocc_pos@10` | 0.0526 | 0.4286 |
| `jaccard_abs@10` | **0.0000** | **0.3333** |
| `jaccard_pos@10` (signed) | 0.0000 | 0.0000 |

At actor-month level `jaccard_abs@10` is 0.0 everywhere and log-occlusion is not
computed (blank). `experiments/sprint_b_report.py` now prints "0.05-0.43
(log-occlusion) and 0.00-0.33 (absolute attribution, actor level)". The
shipped `stability/report.md` already reported the per-cell values (source
0.053, target 0.333) and needed no change. Arithmetic only: no table cell, no
conclusion and no caption changed.

## 2026-10-26 — public world-line preview re-rendered at 150 mm

`results/worldline_preview/worldline_preview.{pdf,png}` were the superseded
465 × 380 pt (164 mm) DejaVu print version with a baked-in key. They were
replaced by a standalone 150 mm vector render (425.197 × 330 pt, Arimo, no baked
key; sha256 `77081960…8bb3b1`) and its 300 dpi raster (sha256 `3b6e0f…e02d4`).
The preview is an illustration only: it is not a manuscript exhibit, and no
manuscript figure or appendix is derived from it. `run_manifest.json` gained a
`figure_provenance` block and new `outputs` digests; `results/README.md` and the
preview README say "standalone repo-only illustration; rendered from restricted
trace exports; not regenerable from this tree". `make worldline-check` and
`make verify` now assert 425.197 × 330 pt plus both digests and reject the old
465 × 380 pt DejaVu file. No CSV was added, no experiment was rerun and no main
result depends on this directory.

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

`report.md` and `summary.json` (`paper_sentences`) attributed the secondary
29-feature model's win over the warm-history Markov baseline to
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
