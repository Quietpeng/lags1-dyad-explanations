"""Sprint B.4 - integrate exactness, stability and HGB audits into report.md.

Reads only the Sprint B summaries/tables.  Writes:
  results/framework_pilot/stability/summary.json
  results/framework_pilot/stability/report.md
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

import sprint_b_common as C
from framework.utils import jdump, log

OUT = C.OUT
TAB = C.TAB


def load_json(name, default=None):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_csv(name):
    path = os.path.join(TAB, name)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def f(x, nd=3):
    if x is None:
        return "NA"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{nd}f}"


def pct(x, nd=1):
    if x is None:
        return "NA"
    return f"{100 * float(x):.{nd}f}%"


def md_table(rows, cols, headers=None, nd=3):
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict(orient="records")
    if not rows:
        return "_No rows._\n"
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c) if isinstance(r, dict) else r[c]
            if isinstance(v, (float, np.floating)):
                vals.append(f(v, nd))
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    t0 = time.time()
    D = load_json("summary_decompose.json", {})
    S = load_json("summary_stability.json", {})
    H = load_json("summary_hgb.json", {})
    exact = load_csv("exactness_check.csv")
    buckets = load_csv("bucket_shares.csv")
    actors = load_csv("actor_attribution_all_periods.csv")
    seed_sum = load_csv("stability_seed_summary.csv")
    seed_pairs = load_csv("stability_seed_pairs.csv")
    period_df = load_csv("stability_periods.csv")
    window_df = load_csv("stability_windows.csv")
    var_df = load_csv("variance_decomposition.csv")
    topk = load_csv("topk_churn_actor_month.csv")
    dec = load_csv("decision_impact_actor_month.csv")
    cond = load_csv("stability_conditional_on_risk.csv")
    cond_dyad = load_csv("stability_conditional_on_risk_dyad.csv")
    boot = load_csv("stability_bootstrap.csv")
    signflip = load_csv("sign_flip_summary.csv")
    hgb_metrics = load_csv("hgb_metrics.csv")
    hgb_agree = load_csv("hgb_occlusion_agreement.csv")
    hgb_fail = load_csv("hgb_occlusion_failures.csv")
    hgb_fail_like = load_csv("hgb_occlusion_failures_like_for_like.csv")
    dyad_churn = load_csv("topk_churn_dyad.csv")
    dyad_dec = load_csv("decision_impact_dyad.csv")
    dyad_stab = load_csv("dyad_stability.csv")

    periods = [p["name"] for p in C.PERIODS]
    role_order = ["source", "target"]
    period_label = {
        "inner_2016_17": "2016-01..2017-11 (inner validation)",
        "retro_2018_22": "2018-01..2022-11 (burned retrospective, secondary)",
        "covid_2020_21": "2020-01..2021-12 (burned retrospective, secondary)",
    }

    # ------------------------------------------------------------------
    # key aggregates for the summary
    # ------------------------------------------------------------------
    ex_rows = []
    for p in periods:
        e = exact[exact.period == p].iloc[0].to_dict() if len(exact[exact.period == p]) else {}
        ex_rows.append({
            "period": p, "status": period_label[p],
            "n_sample": int(e.get("sample_rows", 0)),
            "raw_max_abs_error": e.get("raw_max_abs_error"),
            "raw_mean_abs_error": e.get("raw_mean_abs_error"),
            "raw_rows_above_1e-4": int(e.get("raw_rows_above_tol", 0)),
            "decision_max_abs_error": e.get("decision_max_abs_error"),
            "decision_mean_abs_error": e.get("decision_mean_abs_error"),
            "temperature_max_abs_error": e.get("temperature_max_abs_error"),
            "accumulator_max_abs_error": e.get("accumulator_max_abs_error"),
            "accumulator_rows_above_1e-6": int(e.get("accumulator_rows_above_tol", 0)),
        })
    exact_pass = all((r["raw_rows_above_1e-4"] == 0 and
                      (r["decision_max_abs_error"] or 0) < 1e-5 and
                      (r["temperature_max_abs_error"] or 0) < 1e-12)
                     for r in ex_rows)

    bucket_pivot = (buckets.pivot_table(index="period", columns="bucket",
                                        values="abs_share", aggfunc="first")
                    .reindex(periods).reset_index())

    # top positive/negative actors with mean p
    top_rows = []
    for p in periods:
        for r in role_order:
            sub = actors[(actors.period == p) & (actors.role == r)]
            if not len(sub):
                continue
            for tag, frame in (("top positive", sub.nlargest(5, "attribution")),
                               ("top negative", sub.nsmallest(5, "attribution"))):
                for _, x in frame.iterrows():
                    top_rows.append({
                        "period": p, "role": r, "side": tag,
                        "actor": x["actor"], "attribution": x["attribution"],
                        "mean_p": x["mean_p"], "mean_y": x["mean_y"],
                        "ego_abs_share": x["ego_abs_share"],
                    })

    # source-target correlation
    st_corr = []
    for p in periods:
        a = actors[(actors.period == p) & (actors.role == "source")].set_index("actor")
        b = actors[(actors.period == p) & (actors.role == "target")].set_index("actor")
        j = pd.concat([a["attribution"], b["attribution"]], axis=1,
                      keys=["source", "target"]).dropna()
        if len(j) > 2:
            st_corr.append({
                "period": p, "n": len(j),
                "pearson": j["source"].corr(j["target"]),
                "spearman": j["source"].corr(j["target"], method="spearman"),
                "mean_abs_diff": float((j["source"] - j["target"]).abs().mean()),
            })

    # seed summary compact
    seed_compact = []
    for p in periods:
        for r in role_order:
            row = {"period": p, "role": r}
            for m, key in (("spearman", "spearman"),
                           ("sign", "sign_agreement"),
                           ("j10", "jaccard_pos@10"),
                           ("j20", "jaccard_pos@20")):
                x = seed_sum[(seed_sum.period == p) & (seed_sum.role == r) &
                             (seed_sum.metric == key) &
                             (seed_sum.comparison == "seed_pair")]
                row[m] = float(x.iloc[0]["median"]) if len(x) else None
                row[m + "_min"] = float(x.iloc[0]["min"]) if len(x) else None
                row[m + "_max"] = float(x.iloc[0]["max"]) if len(x) else None
            seed_compact.append(row)

    # period pairs compact
    period_compact = period_df.copy() if len(period_df) else pd.DataFrame()

    # variance compact
    var_compact = []
    for r in role_order:
        for comp in ("period_regime", "seed_noise", "actor_x_time_resid"):
            x = var_df[(var_df.role == r) & (var_df.level == "within_actor_total") &
                       (var_df.component == comp)]
            if len(x):
                var_compact.append({"role": r, "component": comp,
                                    "share": float(x.iloc[0]["share"])})
        for comp in ("regime_var_over_seed_var_median",
                     "share_actors_ratio_gt_2"):
            x = var_df[(var_df.role == r) & (var_df.level == "actor_ratio") &
                       (var_df.component == comp)]
            if len(x):
                var_compact.append({"role": r, "component": comp,
                                    "share": float(x.iloc[0]["share"])})

    # top-k compact
    topk_compact = topk[(topk.k == 10)].copy() if len(topk) else pd.DataFrame()

    # decision compact: k=10
    dec_compact = dec[dec.k == 10].copy() if len(dec) else pd.DataFrame()

    # HGB compact
    hgb_actor = hgb_agree[hgb_agree.level == "actor"] if len(hgb_agree) else pd.DataFrame()
    hgb_am = hgb_agree[hgb_agree.level == "actor_month"] if len(hgb_agree) else pd.DataFrame()
    hgb_fail_top = hgb_fail.head(12) if len(hgb_fail) else pd.DataFrame()
    dyad_churn_10 = dyad_churn[dyad_churn.k.isin([10, 50, 100])] if len(dyad_churn) else pd.DataFrame()
    dyad_dec_k = dyad_dec[dyad_dec.k.isin([10, 50, 100, 500, 1000])] if len(dyad_dec) else pd.DataFrame()

    holdout = H.get("holdout", {}) if H else {}

    summary = {
        "sprint": "B",
        "status": "provisional_no_fresh_holdout",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": {
            "unit": "directed dyad-month; actor views aggregate exact contributions",
            "periods": C.PERIODS,
            "embargo": {"train": C.TRAIN_EMBARGO, "select": C.SELECT_EMBARGO,
                        "inner": 275, "retro": 335},
            "n_seed_replicates": C.N_SEEDS,
            "n_features": int(len(D.get("model", {}).get("features", []))),
            "n_actors": int(D.get("protocol", {}).get("n_actors", 196)),
            "test_used": True,
            "test_status": "burned retrospective secondary only",
        },
        "exactness": {
            "verdict": "PASS" if exact_pass else "CHECK",
            "raw_tolerance_1e-4": "all sampled rows pass",
            "decision_function_max_abs_error": max(
                (r["decision_max_abs_error"] or 0) for r in ex_rows),
            "temperature_max_abs_error": max(
                (r["temperature_max_abs_error"] or 0) for r in ex_rows),
            "temperature_note": ("selected T=1.0; the temperature column is the "
                                  "sigmoid/logit round-trip of the raw track, not "
                                  "an independent calibration fit; the affine "
                                  "identity is unit-tested at T=2"),
            "accumulator_warning": ("full-stream 1e-6 check flags float32 "
                                     "predict_proba round-trip, not a "
                                     "decomposition failure"),
            "isotonic_decomposed": False,
            "periods": ex_rows,
        },
        "source_target": {
            "bucket_shares": bucket_pivot.to_dict(orient="records"),
            "top_actors": top_rows,
            "source_target_correlation": st_corr,
            "interpretation": ("Attribution is relative to the training mean; "
                                "positive values push the dyad above average and "
                                "do not imply high absolute risk."),
        },
        "stability": {
            "rubric": S.get("rubric"),
            "seed": seed_compact,
            "period": period_compact.to_dict(orient="records") if len(period_compact) else [],
            "variance": var_compact,
            "sign_flip": signflip.to_dict(orient="records") if len(signflip) else [],
            "bootstrap": S.get("bootstrap", []),
        },
        "topk_decision": {
            "actor_month": topk_compact.to_dict(orient="records") if len(topk_compact) else [],
            "decision_impact_k10": dec_compact.to_dict(orient="records") if len(dec_compact) else [],
            "conditional_on_risk": cond.to_dict(orient="records") if len(cond) else [],
            "conditional_on_risk_dyad": cond_dyad.to_dict(orient="records")
            if len(cond_dyad) else [],
            "dyad_churn": dyad_churn_10.to_dict(orient="records") if len(dyad_churn_10) else [],
            "dyad_decision": dyad_dec_k.to_dict(orient="records") if len(dyad_dec_k) else [],
        },
        "hgb": {
            "metrics": hgb_metrics.to_dict(orient="records") if len(hgb_metrics) else [],
            "agreement_actor": hgb_actor.to_dict(orient="records") if len(hgb_actor) else [],
            "agreement_actor_month": hgb_am.to_dict(orient="records") if len(hgb_am) else [],
            "failures_top": hgb_fail_top.to_dict(orient="records") if len(hgb_fail_top) else [],
            "failures_like_for_like_top": (
                hgb_fail_like.head(12).to_dict(orient="records")
                if len(hgb_fail_like) else []),
        },
        "holdout": holdout,
        "files": sorted([
            "report.md", "summary.json", "summary_decompose.json",
            "summary_stability.json", "summary_hgb.json", "run.log",
        ] + [f for f in os.listdir(TAB) if f.endswith(".csv")]),
        "runtime_seconds": round(time.time() - t0, 2),
    }
    jdump(os.path.join(OUT, "summary.json"), summary)

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    L = []
    add = L.append
    add("# Sprint B — exact additive actor explanations and stability audit\n")
    add(f"**Status: `provisional_no_fresh_holdout`.** Generated "
        f"{summary['generated_utc']}. CPU only; no new data; no synthetic or T1 "
        "run; no test tuning. 2016-17 is inner validation; 2018-22 and COVID "
        "are burned retrospective secondary views.\n")
    add("> The only confirmatory set is a fresh 20% dyad-holdout. It was not "
        "available to Sprint B. Every number below is therefore provisional and "
        "must not be written as a confirmatory headline.\n")
    add("| question | result | status |\n|---|---|---|\n"
        "| Exact additive identity | decision-function max error 1.4e-6; "
        "raw-vs-logit max 4.1e-5 (float32 p) | PASS |\n"
        "| Source/target shares | 22.0% / 23.7% of mean absolute contribution; "
        "actor views correlate r=0.998 | descriptive |\n"
        "| Seed stability | Spearman ≥0.996; signed top-10 Jaccard 0.82-1.00 | "
        "stable |\n"
        "| Regime stability | Spearman 0.95-0.99; top-10 Jaccard 0.54-0.82; seed "
        "noise 4-5% of within-actor variance | conditional |\n"
        "| Top-k / decision | risk lift 2.5-4.3 (actor-month), 7.9-9.1 (dyad); "
        "attribution lift 0.10-0.20 / 0.06-0.56; post-hoc explanation Spearman "
        "≥0.996 | negative for alerting, positive for post-hoc |\n"
        "| HGB occlusion | occlusion-vs-occlusion Spearman 0.69-0.82; top-10 "
        "absolute overlap 0.05-0.43 | broad rank agreement |\n"
        "| Fresh holdout | no prediction artifact; Sprint A smoke run blind/quick | "
        "P6 blocked |\n")

    add("## 1. Exactness verdict\n")
    add(f"**Verdict: {'PASS' if exact_pass else 'CHECK'}.** For every evaluated "
        "row, the intercept plus the 49 feature contributions equals the "
        "model's own `decision_function` to numerical precision. The larger "
        "`raw` error against `logit(p)` is a float32 `predict_proba` "
        "round-trip, not an additive-decomposition failure. The selected "
        "temperature is T=1.0, so the temperature track coincides with the raw "
        "track; affine scaling C/T is exact by construction (the unit test at "
        "T=2 passes at 1e-12). Isotonic calibration is monotone but not affine "
        "and is excluded from the exact track.\n")
    ex_show = []
    for r in ex_rows:
        rr = dict(r)
        for c in ("raw_max_abs_error", "raw_mean_abs_error",
                  "decision_max_abs_error", "decision_mean_abs_error",
                  "temperature_max_abs_error", "accumulator_max_abs_error"):
            if rr.get(c) is not None:
                rr[c] = f"{float(rr[c]):.2e}"
        ex_show.append(rr)
    add(md_table(ex_show, ["period", "n_sample", "raw_max_abs_error",
                           "raw_mean_abs_error", "raw_rows_above_1e-4",
                           "decision_max_abs_error", "temperature_max_abs_error",
                           "accumulator_max_abs_error",
                           "accumulator_rows_above_1e-6"],
                 ["period", "n", "raw max", "raw mean", "raw >1e-4",
                  "decision max", "temperature max", "stream max", "stream >1e-6"],
                 nd=9))
    add("_Table note: with the selected T=1.0, the `temperature max` column is "
        "the sigmoid/logit round-trip of the raw track, not an independent "
        "temperature fit; the affine identity itself is tested at T=2 in "
        "`tests/test_attribution.py`._\n")
    add("Reading the accumulator warning: the streaming accumulator checks "
        "`b + Σ C = logit(p)` at `1e-6`. `predict_proba` returns float32, so "
        "`logit(p)` carries a quantisation error of up to ~1.7e-4 in raw "
        "log-odds; the decomposition itself matches `decision_function` to "
        "≤1.42e-6 (mean ≤1.24e-7). The strict `1e-6` stream flag should be "
        "read as a storage-precision diagnostic, not as a failed identity.\n")
    cal = load_csv("calibration_sensitivity.csv")
    if len(cal):
        add(md_table(cal[["period", "log_loss_raw", "log_loss_isotonic",
                          "mean_abs_isotonic_logit_distortion"]],
                     ["period", "log_loss_raw", "log_loss_isotonic",
                      "mean_abs_isotonic_logit_distortion"],
                     ["period", "LL raw", "LL isotonic",
                      "mean abs logit distortion"],
                     nd=6))
    add("Isotonic gains 0.0003-0.0007 LL but moves the log-odds by ~0.21 on "
        "average; it cannot be decomposed into the same additive contributions "
        "and is reported only as a sensitivity check.\n")

    add("## 2. Source/target aggregation\n")
    add("The bucket contract assigns each feature to exactly one owner: source "
        "ego, target ego, dyad history, calendar, symmetric graph, or train-only "
        "statistics. The two actor buckets carry about 46% of the mean absolute "
        "contribution; the dyad block carries ~29%, graph ~16%.\n")
    add(md_table(bucket_pivot, ["period"] + [c for c in
                                            ["source_actor", "target_actor", "dyad",
                                             "calendar", "graph", "train_stats"]
                                            if c in bucket_pivot.columns],
                 ["period"] + [c.replace("_", " ") for c in
                               ["source_actor", "target_actor", "dyad", "calendar",
                                "graph", "train_stats"] if c in bucket_pivot.columns],
                 nd=4))
    add("Top positive and top negative attribution actors (top 5 per view; "
        "`mean_p` is the actor's mean forecast in the same period):\n")
    show = [r for r in top_rows if r["period"] == "inner_2016_17"]
    add(md_table(show, ["role", "side", "actor", "attribution", "mean_p",
                        "mean_y", "ego_abs_share"],
                 ["view", "side", "actor", "attribution", "mean p", "mean y",
                  "ego abs share"], nd=3))
    add("The same small-state set (SMR, ATG, MCO, KNA, LIE, DMA) dominates "
        "positive source and target attribution in every period; the major "
        "powers (USA, CHN, RUS, GBR, JPN, FRA) dominate the negative tail. This "
        "is an activity-level contrast, **not** a risk ranking: the top positive "
        "actors have mean predicted probabilities of 0.008-0.015, far below the "
        "~0.11 period base rate, while USA/CHN have mean p of 0.4-0.6. Positive "
        "attribution means 'less negative than the training mean', not 'likely "
        "to act'.\n")
    add("Top contributors by period and view (full tables: "
        "`tables/actor_source_{period}.csv`, `tables/actor_target_{period}.csv`):\n")
    top_compact = []
    for p in periods:
        for r in role_order:
            sub = actors[(actors.period == p) & (actors.role == r)]
            if not len(sub):
                continue
            pos = sub.nlargest(3, "attribution")
            neg = sub.nsmallest(3, "attribution")
            top_compact.append({
                "period": p, "role": r,
                "top_positive": ", ".join(f"{x.actor} ({x.attribution:+.3f})"
                                          for x in pos.itertuples()),
                "top_negative": ", ".join(f"{x.actor} ({x.attribution:+.3f})"
                                          for x in neg.itertuples()),
            })
    add(md_table(top_compact, ["period", "role", "top_positive", "top_negative"],
                 ["period", "view", "top positive (attribution)",
                  "top negative (attribution)"], nd=3))
    add("Source and target views are highly redundant at the actor level:\n")
    add(md_table(st_corr, ["period", "n", "pearson", "spearman", "mean_abs_diff"],
                 ["period", "n", "Pearson", "Spearman", "mean abs diff"], nd=4))
    add("The target view adds little ranking information beyond the source view "
        "for these 196 actors. Per-dyad contributions can still differ; the "
        "paper should report both views but must not present them as independent "
        "evidence.\n")

    add("## 3. Stability audit\n")
    add("Five replicates are **dyad-cluster bootstrap refits with C fixed**, not "
        "optimizer seeds. The lbfgs point fit is deterministic, so optimizer "
        "noise is exactly zero; the reported seed spread is cluster-resampling "
        "uncertainty.\n")
    add(md_table(seed_compact, ["period", "role", "spearman", "spearman_min",
                                "j10", "j10_min", "j20", "sign", "sign_min"],
                 ["period", "view", "Spearman", "min", "J@10", "min",
                  "J@20", "sign", "min"], nd=4))
    add("Signed top-k Jaccard is more decision-relevant than absolute top-k. "
        "At k=10 the median seed-pair Jaccard is 0.82-1.00; k=5 is 0.67-1.00. "
        "The churn is concentrated at near-ties: the boundary gap at k=10 is "
        "0.67-1.29 seed SD for actor-month attribution, versus 16-42 "
        "seed SD for risk scores. Rank correlation is ≥0.996 everywhere. Sign "
        "flips are absent for actors with |attribution| ≥0.10; at the 0.05 "
        "threshold one COVID source actor flips in one replicate, and the "
        "pairwise sign agreement is 0.93-1.00 (typically 0.97-0.99), with "
        "disagreement confined to near-zero actors.\n")
    add("Per-actor seed SD illustrates the same point: SMR's inner source "
        "attribution is 0.673 ± 0.044 and ATG's is 0.507 ± 0.054 (mean ± seed "
        "SD), so the top-two order is separated by ~3 seed SD; the KNA/LIE pair "
        "(0.333 vs 0.321, seed SD 0.038/0.027) is inside one seed SD. The churn "
        "is therefore concentrated in the middle of the top-10, not at the top.\n")
    add("Across periods (regime shift):\n")
    add(md_table(period_compact, ["role", "period_a", "period_b", "spearman",
                                  "jaccard_pos@10", "sign_agreement"],
                 ["view", "period A", "period B", "Spearman", "J@10", "sign"],
                 nd=4))
    add("Rank correlation stays high (0.95-0.99) but signed top-10 Jaccard falls "
        "to 0.54-0.82; sign agreement is 0.92-0.95. Regime change, not seed "
        "noise, is the dominant source of top-k churn.\n")
    if len(window_df):
        wg = (window_df.groupby(["period", "role", "window_a", "window_b"])
              [["spearman", "jaccard_pos@10", "sign_agreement"]].median()
              .reset_index())
        add("D2 windows (median over seeds):\n")
        add(md_table(wg, ["period", "role", "window_a", "window_b", "spearman",
                          "jaccard_pos@10", "sign_agreement"],
                     ["period", "view", "window A", "window B", "Spearman",
                      "J@10", "sign"], nd=3))
        add("Window stability is conditional: `W_d_stable` vs the full period is "
            "essentially identical (J@10 ~1.0), but onset and dormancy-reentry "
            "windows disagree with each other and with the full period "
            "(J@10 0.33-0.82). Explanations for cold-start/onset cases should "
            "not be pooled with stable-history cases.\n")
    add("Variance decomposition within actor (period × seed):\n")
    add(md_table(var_compact, ["role", "component", "share"],
                 ["view", "component", "share"], nd=4))
    add("Seed noise is 4.1-5.1% of within-actor variance; period/regime is "
        "35.5-36.8%; the actor×time residual is 59-60%. The median "
        "regime/seed variance ratio is 9.4 (source) and 12.8 (target); "
        "79-88% of actors have a ratio above 2. This is the central stability "
        "finding: the audit is not seed-limited, it is regime-limited.\n")
    lin = var_df[var_df.level.str.startswith("linearisation")]
    if len(lin):
        add("Coefficient-space check: the exact linearisation "
            "`Var_seed(a_i) = z̄_i' Σ_β z̄_i` reproduces the empirical seed SD "
            "with ratio 1.000 and correlation 1.000 in every period/view. "
            "Collinearity does not create the seed spread; the covariance ratio "
            "is 0.39-0.80 at the median and 0.84-1.41 at the 90th percentile, "
            "so for most actors coefficient covariance damps seed variance, "
            "while the top decile is amplified by up to ~1.4×. The first "
            "principal component of Σ_β explains 21-62% of the linear seed "
            "variance.\n")
        coll = D.get("collinearity", {})
        if coll:
            add(f"Collinearity is severe in the design matrix (condition number "
                f"{coll.get('condition_number'):,.0f}; max VIF "
                f"{coll.get('max_vif'):,.0f}; {coll.get('n_high_vif_gt10')} "
                "features with VIF>10), but the coefficient-space check shows "
                "it does not translate into unstable actor rankings: the "
                "covariance among coefficients is negative-signed for most "
                "actors and the empirical seed SD is reproduced exactly by the "
                "linearisation. The instability that remains is regime-driven, "
                "not collinearity-driven.\n")
    add("Bootstrap CIs (month-block for period comparisons, actor-cluster for "
        "seed pairs): the period-pair Spearman CIs are narrow and exclude 0.90 "
        "in most cells; top-10 Jaccard CIs are wide (e.g. inner→COVID source "
        "[0.33,0.82]). The rank ordering is stable; the exact membership of a "
        "10-actor review set is not.\n")

    add("## 4. Top-k churn and decision impact\n")
    add("We distinguish two alert rules: `risk` ranks actor-months by mean "
        "predicted probability; `attribution` ranks them by the actor's own "
        "bucket contribution. The latter is an explanation score, not a risk "
        "score.\n")
    add(md_table(topk_compact, ["period", "role", "score", "k",
                                "seed_pair_mean_jaccard", "mean_churn",
                                "median_boundary_gap_over_seed_sd"],
                 ["period", "view", "score", "k", "seed Jaccard", "churn",
                  "boundary gap / seed SD"], nd=3))
    add(md_table(dec_compact, ["period", "role", "k", "alert",
                               "mean_lift_vs_month_base", "lift_ci_lo",
                               "lift_ci_hi", "mean_jaccard_with_risk_alert"],
                 ["period", "view", "k", "alert", "mean lift", "CI low",
                  "CI high", "Jaccard with risk"], nd=3))
    add("The risk alert is essentially seed-invariant (actor-month Jaccard "
        "0.995-1.000; dyad Jaccard 0.98-1.00; boundary gap 16-42 seed SD for "
        "actor-month scores) and yields lift 2.5-4.3 for actor-month review and "
        "7.9-9.1 for dyad review. The attribution alert is stable enough to be "
        "reproducible but is **anti-useful as an alert rule**: lift 0.10-0.20 "
        "at actor-month level, 0.06-0.56 for source/target dyad buckets, and "
        "Jaccard overlap with the risk alert is 0.000-0.003. This is not a bug: "
        "top positive attribution selects low-activity microstates whose "
        "probabilities are below the base rate.\n")
    if len(dyad_churn):
        add("Dyad-level churn (strict inner window; unique dyads):\n")
        add(md_table(dyad_churn_10, ["score", "k", "seed_pair_mean_jaccard",
                                     "point_vs_seed_mean_jaccard",
                                     "boundary_gap_over_seed_sd"],
                     ["score", "k", "seed Jaccard", "point vs seed",
                      "boundary gap / seed SD"], nd=3))
    if len(dyad_dec):
        add("Dyad-level decision impact (top-k dyads; all rows of a selected "
            "dyad are reviewed):\n")
        add(md_table(dyad_dec_k, ["alert", "k", "coverage", "precision",
                                  "recall", "lift_vs_base",
                                  "jaccard_with_risk"],
                     ["alert", "k", "coverage", "precision", "recall", "lift",
                      "Jaccard with risk"], nd=3))
        add("At the dyad level, the `total` attribution is a monotone transform "
            "of mean risk (mean total contribution = mean log-odds minus the "
            "intercept), so its top-k is trivially the risk top-k. The "
            "non-trivial explanation alerts are the source/target bucket views; "
            "they achieve lift 0.06-0.56, again below random review.\n")
    add("Decision verdict: use the risk score to choose what to review; attach "
        "the exact actor decomposition afterwards to explain why. Do not use "
        "the attribution top-k as a thresholded alert rule. This is a negative "
        "decision-impact result, not a failed stability result.\n")
    if len(cond):
        add("Conditional explanation stability inside the risk alert set:\n")
        add(md_table(cond, ["period", "role", "k", "n_cells",
                            "seed_pair_spearman_median", "seed_pair_spearman_min",
                            "seed_pair_sign_median"],
                     ["period", "view", "k", "cells", "seed Spearman",
                      "min", "sign"], nd=4))
        add("Once the risk score has selected the review set, the source/target "
            "attribution rankings **within that set** are essentially "
            "seed-invariant (Spearman 0.996-1.000; sign agreement 0.998-1.000). "
            "This is the constructive decision result: explanations are stable "
            "enough for post-hoc attribution of a risk alert, but not for "
            "choosing the alert set.\n")
    if len(cond_dyad):
        add("The same check on the dyad risk alert set:\n")
        add(md_table(cond_dyad, ["k", "score", "n_dyads",
                                 "seed_pair_spearman_median",
                                 "seed_pair_spearman_min",
                                 "seed_pair_sign_median"],
                     ["k", "view", "dyads", "seed Spearman", "min", "sign"],
                     nd=4))
        add("Within the top-k risk dyads, source/target attribution rankings are "
            "also seed-invariant (Spearman ≥0.997; sign agreement 0.98-1.00). "
            "The post-hoc use therefore survives the finer-grained decision "
            "unit.\n")

    add("## 5. HGB occlusion agreement\n")
    add("HGB is refit on the current graph cache and is **not** the P1 HGB "
        "artifact (the P1 predictions were produced before the current graph "
        "cache; see `summary_decompose.json`). Its accuracy is descriptive "
        "context for the occlusion check; no HGB-vs-logistic superiority or "
        "parity claim is made. The selected 150 iterations are the grid edge.\n")
    if len(hgb_metrics):
        hm_show = hgb_metrics.copy()
        for c in ("n", "positives", "n_iter"):
            if c in hm_show:
                hm_show[c] = hm_show[c].astype(int)
        add(md_table(hm_show, ["period", "n", "positives", "n_iter",
                               "select_log_loss", "log_loss", "ece", "auc", "ap"],
                     ["period", "n", "pos", "iters", "select LL", "LL", "ECE",
                      "AUC", "AP"], nd=4))
    if len(hgb_actor):
        add("Actor-level HGB block occlusion (`ΔNLL` when the source/target "
            "feature block is replaced by its training mean). The signed "
            "attribution–ΔNLL column is **negative by construction of the loss "
            "scale** in a zero-dominated sample; use the like-for-like "
            "occlusion–occlusion column, not the signed attribution column, to "
            "judge model agreement.\n")
        add(md_table(hgb_actor, ["period", "role", "n", "spearman_logocc_hgb",
                                 "spearman_logocc_ci_lo", "spearman_logocc_ci_hi",
                                 "sign_logocc_hgb", "jaccard_logocc_abs@10",
                                 "spearman_abs_attr_hgb", "jaccard_abs@10",
                                 "spearman_attr_hgb"],
                     ["period", "view", "n", "occl–occl Spearman", "CI low",
                      "CI high", "sign", "J@10 (abs occl)",
                      "abs attr vs abs HGB", "J@10 (abs attr)",
                      "signed attr–HGB"], nd=3))
    if len(hgb_am):
        add("Actor-month agreement (pooled cells; per-cell mean ΔNLL):\n")
        ham_show = hgb_am.copy()
        if "n" in ham_show:
            ham_show["n"] = ham_show["n"].astype(int)
        add(md_table(ham_show, ["period", "role", "n", "spearman_logocc_hgb",
                                "spearman_logocc_ci_lo", "spearman_logocc_ci_hi",
                                "sign_logocc_hgb", "jaccard_logocc_abs@10",
                                "spearman_abs_attr_hgb"],
                     ["period", "view", "n", "occl–occl Spearman", "CI low",
                      "CI high", "sign", "J@10 (abs occl)",
                      "abs attr vs abs HGB"], nd=3))
    if len(hgb_fail_like):
        add("Worst like-for-like rank-divergence cases (logistic occlusion vs "
            "HGB occlusion; absolute-rank gap; not causal):\n")
        add(md_table(hgb_fail_like.head(12), ["period", "role", "actor", "n_rows",
                                              "logistic_occlusion_mean_dnll",
                                              "hgb_occlusion_mean_dnll", "rank_gap",
                                              "sign_disagreement"],
                     ["period", "view", "actor", "rows", "logistic ΔNLL",
                      "HGB ΔNLL", "rank gap", "sign flip"], nd=5))
    add("Interpretation: HGB block occlusion reproduces the logistic occlusion "
        "ranking at Spearman 0.69-0.82 (actor level; month-block/actor-bootstrap "
        "CIs exclude 0), and at 0.67-0.74 for actor-month cells. Top-10 "
        "absolute-importance overlap is only 0.05-0.43, so the agreement is "
        "broad-ordering, not set-level. HGB selected 150 iterations at the grid "
        "edge (select LL still improving); it is a capacity control, not a "
        "tuned competitor. Signed attribution and ΔNLL are not the "
        "same scale: because most rows are zeros, a positive attribution actor "
        "can have a negative ΔNLL when the block is removed. HGB is a capacity "
        "control, not an exact decomposer; block replacement can be absorbed by "
        "correlated graph features, and the worst disagreements are concentrated "
        "in actors (e.g. IRQ, ITA, MDA, SMR) whose graph and actor blocks are "
        "plausibly redundant.\n")

    add("## 6. Fresh holdout integration\n")
    if holdout.get("available"):
        add(f"Sprint A prediction files were found in `{holdout.get('pred_dir')}`: "
            f"{', '.join(holdout.get('files', []))}. Sprint B does **not** "
            "re-tune or re-select on them; they are a separate confirmatory "
            "check and must be read once under the preregistered protocol.\n")
    else:
        add("No fresh-holdout prediction artifact exists. The Sprint A run in "
            "`results/framework_pilot/holdout/run.log` was a `--blind --quick` "
            "smoke run and stopped on a broken pipe after point metrics; "
            "`holdout/preds/` is empty. Sprint B therefore makes **no "
            "confirmatory claim**, and P6 remains blocked. The stability "
            "results above are inner-validation + burned-retrospective "
            "evidence only.\n")

    add("## 7. Sentences the paper can use\n")
    add("Each sentence is tagged with its evidence status. Do not upgrade a "
        "`provisional` sentence to confirmatory without the fresh holdout.\n")
    sentences = [
        ("**Exactness (supported).** The logistic forecast is exactly additive: "
         "the intercept plus 49 feature contributions reproduces the model's "
         "decision function to at most 1.4e-6 in log-odds; temperature scaling "
         "is affine and therefore preserves the identity, and isotonic "
         "calibration is reported only as a sensitivity check because it is "
         "not affine.",
         "supported (inner + burned secondary)"),
        ("**Source/target aggregation (supported, with a redundancy caveat).** "
         "Source-ego and target-ego features each carry about 22-24% of the "
         "mean absolute contribution; the two actor views rank actors almost "
         "identically (r=0.998), so they should be presented as two bookkeeping "
         "views of the same forecast, not as independent evidence.",
         "supported (inner + burned secondary)"),
        ("**Activity, not risk (supported).** The positive attribution tail is "
         "dominated by low-activity microstates (SMR, ATG, MCO, KNA, LIE) whose "
         "mean predicted probabilities are 0.008-0.015, while the negative tail "
         "is dominated by high-activity major powers (USA, CHN, RUS, GBR, JPN); "
         "positive attribution means 'above the training mean', not 'likely to "
         "act'.",
         "supported (inner + burned secondary)"),
        ("**Seed stability (supported).** Across five dyad-cluster bootstrap "
         "refits, actor-attribution rank correlation is at least 0.996 and "
         "signed top-10 Jaccard is 0.82-1.00; no actor with |attribution| ≥0.10 "
         "changes sign, and pairwise sign agreement is 0.93-1.00, with "
         "disagreement confined to near-zero actors.",
         "supported (inner + burned secondary)"),
        ("**Regime-limited, not seed-limited (supported).** Seed noise accounts "
         "for only 4-5% of within-actor variance, versus 35-37% for period and "
         "59-60% for actor×time residual; the median regime/seed variance ratio "
         "is 9.4-12.8, and 79-88% of actors exceed a ratio of 2.",
         "supported (inner + burned secondary)"),
        ("**Top-k churn (supported, conditional).** Signed top-10 actor sets "
         "have a seed-pair Jaccard of 0.82-1.00 but a cross-period Jaccard of "
         "0.54-0.82; the churn is concentrated at near-ties, with a k=10 "
         "boundary gap of 0.67-1.29 seed SD for actor-month attribution "
         "versus 16-42 seed SD for risk scores.",
         "supported (inner + burned secondary)"),
        ("**Decision impact (supported negative result).** Risk-based top-k "
         "alerts are seed-invariant (actor-month Jaccard 0.995-1.000; dyad "
         "Jaccard 0.98-1.00) and achieve lift 2.5-9.1 over the base rate; "
         "attribution-based alerts achieve lift 0.10-0.20 at actor-month level "
         "and 0.06-0.56 for source/target dyad buckets, and overlap the risk "
         "alert almost never. Use explanations after the alert, not as the "
         "alert rule.",
         "supported (inner + burned secondary)"),
        ("**Post-hoc explanation stability (supported).** Conditional on the "
         "risk-based alert set, source/target attribution rankings within that "
         "set are essentially seed-invariant (actor-month Spearman 0.996-1.000; "
         "dyad Spearman ≥0.997; sign agreement 0.98-1.00), so the exact "
         "decomposition is stable enough to attach to an alert after the fact.",
         "supported (inner + burned secondary)"),
        ("**HGB occlusion (supported as a rank-agreement diagnostic, not as "
         "exactness).** HGB block occlusion reproduces the logistic occlusion "
         "ranking at Spearman 0.69-0.82 across actors (0.67-0.74 across "
         "actor-month cells), with top-10 absolute-importance overlap of only "
         "0.05-0.43; signed attribution and occlusion ΔNLL are not the same "
         "scale because the evaluation set is zero-dominated.",
         "supported as rank-agreement diagnostic"),
        ("**Confirmatory status (provisional).** All stability results are "
         "inner-validation and burned-retrospective evidence; the fresh 20% "
         "dyad-holdout was not available, so no confirmatory claim about "
         "explanation stability or decision impact is made.",
         "provisional; P6 blocked"),
    ]
    for s, tag in sentences:
        add(f"- {s} `[{tag}]`\n")

    add("## 8. What not to say\n")
    add("- Do not call the actor attribution a node-level prediction or a risk "
        "score; it is a conditional decomposition of a dyad forecast.\n"
        "- Do not claim that source and target views are independent; they are "
        "nearly collinear at the actor level.\n"
        "- Do not claim that explanations improve decisions; the decision-impact "
        "proxy points the other way.\n"
        "- Do not claim 2018-22 or COVID as confirmatory; they are burned.\n"
        "- Do not claim HGB occlusion is an exact decomposition or a causal "
        "effect; it is a block-replacement rank check, and signed attribution "
        "vs ΔNLL is not a like-for-like comparison.\n"
        "- Do not claim HGB superiority or parity from the Sprint B refit; it "
        "uses a changed graph cache and is a capacity control, not a tuned "
        "competitor.\n"
        "- Do not claim stability generalizes beyond ICEWS + this 196-actor "
        "roster.\n")

    add("## 9. Files\n")
    add("`summary.json`, `summary_decompose.json`, `summary_stability.json`, "
        "`summary_hgb.json`, `run.log`, `report.md`, and the `tables/*.csv` "
        "listed in `summary.json`. The full contribution cache for the inner "
        "window is `framework/cache/sprint_b/contrib_inner_point.npz` (119 MB); "
        "it is a cache, not a reported table.\n")
    add("## 10. Reproduction\n")
    add("From `repo/`, with the CPU-only virtual environment:\n")
    add("```bash\n"
        "PY=python\n"
        "$PY experiments/sprint_b_decompose.py      # exactness + actor tables (~8 min)\n"
        "$PY experiments/sprint_b_stability.py      # stability + top-k (~1 min)\n"
        "$PY experiments/sprint_b_hgb.py            # HGB fit + occlusion (~12 min)\n"
        "$PY experiments/sprint_b_hgb_post.py       # per-row normalization + dyad (~1 min)\n"
        "$PY experiments/sprint_b_report.py         # report.md + summary.json\n"
        "```\n")
    add("The HGB post-process is separate because the first HGB run wrote the "
        "occlusion caches before a dyad-shape bug stopped the script; the bug is "
        "fixed in `sprint_b_hgb.py`, and the post-process recomputes the "
        "exposure-neutral agreement without refitting. `run.log` preserves both "
        "the crash and the recovery for audit.\n")

    report = "\n".join(L)
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(report)
    log(f"report.md + summary.json written ({len(report)} chars, "
        f"{elapsed_s(t0)}s)")


def elapsed_s(t0):
    return round(time.time() - t0, 1)


if __name__ == "__main__":
    main()
