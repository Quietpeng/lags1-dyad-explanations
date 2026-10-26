"""Sprint B.2 - stability audit, top-k churn and decision impact.

Reads the streaming aggregates produced by ``sprint_b_decompose.py`` only; it
does not refit models and does not touch the frozen test split.  All numbers are
either inner validation (2016-17) or burned retrospective secondary views.

Outputs (results/framework_pilot/stability):
  tables/stability_seed_pairs.csv
  tables/stability_seed_summary.csv
  tables/stability_periods.csv
  tables/stability_windows.csv
  tables/stability_bootstrap.csv
  tables/variance_decomposition.csv
  tables/actor_seed_variance.csv
  tables/actor_regime_shift.csv
  tables/feature_stability.csv
  tables/topk_churn_actor_month.csv
  tables/topk_churn_window.csv
  tables/decision_impact_actor_month.csv
  summary_stability.json
"""

from __future__ import annotations

import itertools
import json
import os
import time

import numpy as np
import pandas as pd

import sprint_b_common as C
from framework import aggregation as A
from framework.utils import elapsed, jdump, log

ROLES = ("source", "target")
KS = (5, 10, 20, 50)
N_BOOT = 1000
BOOT_SEED = C.SEED_BASE + 17
# Audit rubric fixed before looking at the stability numbers.  It is a
# reporting convention, not a model-selection criterion.
STABLE = {"spearman": 0.90, "jaccard10": 0.80, "sign": 0.90}
CONDITIONAL = {"spearman": 0.70, "jaccard10": 0.60, "sign": 0.80}


def _cache(period: str, model: str) -> dict:
    path = os.path.join(C.AGG, f"agg_{model}_{period}.npz")
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _role_index(role: str) -> int:
    return 0 if role == "source" else 1


def actor_attribution(z: dict, role: str) -> np.ndarray:
    """Mean owned-bucket contribution per actor over the period."""
    j = _role_index(role)
    b = z[f"bucket_sum_{role}"][:, :, j].sum(axis=1)
    n = z[f"count_{role}"].sum(axis=1)
    return np.where(n > 0, b / np.maximum(n, 1), np.nan)


def actor_month_scores(z: dict, role: str, kind: str = "attr") -> np.ndarray:
    """(n_actors, n_months) mean cell score; NaN for empty cells."""
    j = _role_index(role)
    n = z[f"count_{role}"]
    if kind == "attr":
        num = z[f"bucket_sum_{role}"][:, :, j]
    elif kind == "risk":
        num = z[f"p_sum_{role}"]
    else:
        raise ValueError(kind)
    return np.where(n > 0, num / np.maximum(n, 1), np.nan)


def window_attribution(z: dict, role: str, window_names) -> tuple[np.ndarray, list]:
    """(n_windows, n_actors) mean owned-bucket contribution inside each window."""
    key = f"win_count_{role}"
    if key not in z:
        # Old cache: only the source denominator was stored.
        return np.full((len(window_names), z[f"win_sum_{role}"].shape[1]), np.nan), []
    j = _role_index(role)
    ws = z[f"win_sum_{role}"][:, :, j]
    wc = z[key]
    out = np.where(wc > 0, ws / np.maximum(wc, 1), np.nan)
    return out, list(window_names)


def topk_set(scores: np.ndarray, k: int, mode: str = "pos") -> set:
    x = np.asarray(scores, dtype=np.float64)
    ok = np.isfinite(x)
    if ok.sum() < k:
        return set()
    score = np.where(ok, x if mode == "pos" else np.abs(x), -np.inf)
    return set(np.argsort(-score, kind="stable")[:k].tolist())


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return float("nan")
    return len(a & b) / len(a | b)


def overlap(a: set, b: set, k: int) -> float:
    if not a or not b:
        return float("nan")
    return len(a & b) / max(k, 1)


def _summ(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "median": None, "min": None, "max": None,
                "p10": None, "p90": None}
    return {"n": int(x.size), "median": round(float(np.median(x)), 4),
            "min": round(float(x.min()), 4), "max": round(float(x.max()), 4),
            "p10": round(float(np.percentile(x, 10)), 4),
            "p90": round(float(np.percentile(x, 90)), 4)}


def bootstrap_actor_ci(a, b, reps: int, seed: int) -> dict:
    """Cluster bootstrap over actors for a rank correlation."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    idx = np.where(ok)[0]
    if len(idx) < 3:
        return {"ci": None, "values": np.array([])}
    rng = np.random.default_rng(seed)
    vals = np.empty(reps)
    for r in range(reps):
        take = rng.choice(idx, size=len(idx), replace=True)
        vals[r] = A.spearman(a[take], b[take])
    vals = vals[np.isfinite(vals)]
    return {"ci": A.ci(vals), "values": vals}


def bootstrap_months(z: dict, role: str, reps: int, seed: int) -> dict:
    """Month-block bootstrap of actor attribution (temporal sampling CI)."""
    j = _role_index(role)
    b = z[f"bucket_sum_{role}"][:, :, j]
    n = z[f"count_{role}"]
    M = b.shape[1]
    rng = np.random.default_rng(seed)
    attrs = np.empty((reps, b.shape[0]))
    for r in range(reps):
        take = rng.integers(0, M, size=M)
        bb = b[:, take].sum(axis=1)
        nn = n[:, take].sum(axis=1)
        attrs[r] = np.where(nn > 0, bb / np.maximum(nn, 1), np.nan)
    return {"attrs": attrs, "M": M}


def main() -> None:
    rl = C.RunLog("sprint_b_stability")
    t_start = time.time()
    C.ensure_dirs()
    periods = [p["name"] for p in C.PERIODS]
    names = [n for n in np.load(os.path.join(C.AGG, "coefs.npz"))["names"]]
    n_actors = len(np.load(os.path.join(C.AGG, "zbar_inner_2016_17.npz"))["n_source"])

    # ------------------------------------------------------------------
    # Load caches and build actor-level views
    # ------------------------------------------------------------------
    caches = {p: {m: _cache(p, m) for m in ("point", "rep0", "rep1", "rep2",
                                            "rep3", "rep4")} for p in periods}
    attr = {p: {r: {"point": actor_attribution(caches[p]["point"], r),
                    "seeds": [actor_attribution(caches[p][f"rep{s}"], r)
                              for s in range(C.N_SEEDS)]}
                for r in ROLES} for p in periods}
    for p in periods:
        for r in ROLES:
            A_ = np.vstack(attr[p][r]["seeds"])
            assert A_.shape == (C.N_SEEDS, n_actors)
            log(f"loaded {p} {r}: n_seeds={A_.shape[0]} "
                f"median seed SD={np.nanmedian(np.nanstd(A_, axis=0, ddof=1)):.4f}")

    # ------------------------------------------------------------------
    # 1. Seed-pair stability
    # ------------------------------------------------------------------
    pair_rows = []
    for p in periods:
        for r in ROLES:
            point = attr[p][r]["point"]
            seeds = attr[p][r]["seeds"]
            for i, j in itertools.combinations(range(C.N_SEEDS), 2):
                row = {"period": p, "role": r, "seed_a": i, "seed_b": j,
                       "n_actors": int((np.isfinite(seeds[i]) &
                                        np.isfinite(seeds[j])).sum()),
                       "spearman": round(A.spearman(seeds[i], seeds[j]), 6),
                       "sign_agreement": round(A.sign_agreement(seeds[i], seeds[j]), 6)}
                for k in KS:
                    row[f"jaccard_pos@{k}"] = round(A.topk_jaccard(seeds[i], seeds[j], k, "pos"), 6)
                    row[f"jaccard_abs@{k}"] = round(A.topk_jaccard(seeds[i], seeds[j], k, "abs"), 6)
                pair_rows.append(row)
    seed_pairs = pd.DataFrame(pair_rows)
    seed_pairs.to_csv(os.path.join(C.TAB, "stability_seed_pairs.csv"), index=False)

    summ_rows = []
    for p in periods:
        for r in ROLES:
            d = seed_pairs[(seed_pairs.period == p) & (seed_pairs.role == r)]
            point = attr[p][r]["point"]
            for metric in ["spearman", "sign_agreement", "jaccard_pos@5",
                           "jaccard_pos@10", "jaccard_pos@20", "jaccard_pos@50",
                           "jaccard_abs@10"]:
                s = _summ(d[metric])
                summ_rows.append({"period": p, "role": r, "metric": metric,
                                  "comparison": "seed_pair", **s})
            for k in KS:
                vals = [A.topk_jaccard(point, s, k, "pos") for s in attr[p][r]["seeds"]]
                s = _summ(vals)
                summ_rows.append({"period": p, "role": r,
                                  "metric": f"jaccard_pos@{k}",
                                  "comparison": "point_vs_seed", **s})
            vals = [A.spearman(point, s) for s in attr[p][r]["seeds"]]
            s = _summ(vals)
            summ_rows.append({"period": p, "role": r, "metric": "spearman",
                              "comparison": "point_vs_seed", **s})
    seed_summary = pd.DataFrame(summ_rows)
    seed_summary.to_csv(os.path.join(C.TAB, "stability_seed_summary.csv"), index=False)

    # Sign-flip diagnostics: overall agreement can hide flips near zero.
    sign_rows = []
    for p in periods:
        for r in ROLES:
            point = attr[p][r]["point"]
            stack = np.stack(attr[p][r]["seeds"])
            flip = (np.sign(stack) != np.sign(point)[None, :]).mean(axis=0)
            for thr in (0.0, 0.05, 0.10, 0.20):
                m = np.isfinite(point) & (np.abs(point) >= thr)
                sign_rows.append({
                    "period": p, "role": r, "abs_attr_threshold": thr,
                    "n_actors": int(m.sum()),
                    "mean_seed_sign_flip_rate": round(float(np.mean(flip[m])), 4)
                    if m.any() else None,
                    "max_seed_sign_flip_rate": round(float(np.max(flip[m])), 4)
                    if m.any() else None,
                })
    pd.DataFrame(sign_rows).to_csv(
        os.path.join(C.TAB, "sign_flip_summary.csv"), index=False)

    # ------------------------------------------------------------------
    # 2. Across-period (regime) stability
    # ------------------------------------------------------------------
    period_rows = []
    for r in ROLES:
        for p1, p2 in itertools.combinations(periods, 2):
            a, b = attr[p1][r]["point"], attr[p2][r]["point"]
            row = {"role": r, "period_a": p1, "period_b": p2,
                   "n_common": int((np.isfinite(a) & np.isfinite(b)).sum()),
                   "spearman": round(A.spearman(a, b), 6),
                   "sign_agreement": round(A.sign_agreement(a, b), 6),
                   "mean_abs_diff": round(float(np.nanmean(np.abs(a - b))), 6),
                   "max_abs_diff": round(float(np.nanmax(np.abs(a - b))), 6)}
            for k in KS:
                row[f"jaccard_pos@{k}"] = round(A.topk_jaccard(a, b, k, "pos"), 6)
                row[f"jaccard_abs@{k}"] = round(A.topk_jaccard(a, b, k, "abs"), 6)
            # median over seed pairs (each seed uses the same seed id in both periods)
            reps = []
            for s in range(C.N_SEEDS):
                aa, bb = attr[p1][r]["seeds"][s], attr[p2][r]["seeds"][s]
                reps.append((A.spearman(aa, bb),
                             A.topk_jaccard(aa, bb, 10, "pos"),
                             A.sign_agreement(aa, bb)))
            row["seed_spearman_median"] = round(float(np.median([x[0] for x in reps])), 6)
            row["seed_jaccard10_median"] = round(float(np.median([x[1] for x in reps])), 6)
            row["seed_sign_median"] = round(float(np.median([x[2] for x in reps])), 6)
            period_rows.append(row)
    period_df = pd.DataFrame(period_rows)
    period_df.to_csv(os.path.join(C.TAB, "stability_periods.csv"), index=False)

    # ------------------------------------------------------------------
    # 3. D2 window stability (inner / retro)
    # ------------------------------------------------------------------
    win_rows = []
    for p in periods:
        wnames = json.loads(str(caches[p]["point"]["meta"])).get("window_names", [])
        if not wnames:
            continue
        for r in ROLES:
            for s in range(C.N_SEEDS):
                z = caches[p][f"rep{s}"]
                W, _ = window_attribution(z, r, wnames)
                for i, j in itertools.combinations(range(len(wnames)), 2):
                    a, b = W[i], W[j]
                    win_rows.append({
                        "period": p, "role": r, "seed": s,
                        "window_a": wnames[i], "window_b": wnames[j],
                        "n_common": int((np.isfinite(a) & np.isfinite(b)).sum()),
                        "spearman": round(A.spearman(a, b), 6),
                        "sign_agreement": round(A.sign_agreement(a, b), 6),
                        "jaccard_pos@10": round(A.topk_jaccard(a, b, 10, "pos"), 6),
                        "jaccard_abs@10": round(A.topk_jaccard(a, b, 10, "abs"), 6),
                        "mean_abs_diff": round(float(np.nanmean(np.abs(a - b))), 6),
                    })
                full = actor_attribution(z, r)
                for i, w in enumerate(wnames):
                    a = W[i]
                    win_rows.append({
                        "period": p, "role": r, "seed": s,
                        "window_a": w, "window_b": "full_period",
                        "n_common": int((np.isfinite(a) & np.isfinite(full)).sum()),
                        "spearman": round(A.spearman(a, full), 6),
                        "sign_agreement": round(A.sign_agreement(a, full), 6),
                        "jaccard_pos@10": round(A.topk_jaccard(a, full, 10, "pos"), 6),
                        "jaccard_abs@10": round(A.topk_jaccard(a, full, 10, "abs"), 6),
                        "mean_abs_diff": round(float(np.nanmean(np.abs(a - full))), 6),
                    })
    window_pairs = pd.DataFrame(win_rows)
    window_pairs.to_csv(os.path.join(C.TAB, "stability_windows.csv"), index=False)

    # ------------------------------------------------------------------
    # 4. Bootstrap CIs
    # ------------------------------------------------------------------
    # (a) actor-cluster CI for seed-pair rank correlation
    boot_rows = []
    for p in periods:
        for r in ROLES:
            for i, j in itertools.combinations(range(C.N_SEEDS), 2):
                out = bootstrap_actor_ci(attr[p][r]["seeds"][i],
                                         attr[p][r]["seeds"][j],
                                         reps=N_BOOT, seed=BOOT_SEED + 31 * i + j)
                boot_rows.append({
                    "comparison": "seed_pair", "period_a": p, "period_b": p,
                    "role": r, "seed_a": i, "seed_b": j,
                    "metric": "spearman", "point": round(A.spearman(
                        attr[p][r]["seeds"][i], attr[p][r]["seeds"][j]), 6),
                    "ci_lo": out["ci"][0] if out["ci"] else None,
                    "ci_hi": out["ci"][1] if out["ci"] else None,
                    "n_boot": int(len(out["values"])), "method": "actor_cluster",
                })
    # (b) month-block CI for point-vs-seed and period-pair metrics
    for p in periods:
        for r in ROLES:
            out = bootstrap_months(caches[p]["point"], r, reps=N_BOOT,
                                   seed=BOOT_SEED + 101 + periods.index(p) * 7)
            point = attr[p][r]["point"]
            for s in range(C.N_SEEDS):
                vals_sp, vals_j, vals_sg = [], [], []
                seed = attr[p][r]["seeds"][s]
                for a in out["attrs"]:
                    vals_sp.append(A.spearman(a, seed))
                    vals_j.append(A.topk_jaccard(a, seed, 10, "pos"))
                    vals_sg.append(A.sign_agreement(a, seed))
                boot_rows.append({
                    "comparison": "point_vs_seed", "period_a": p, "period_b": p,
                    "role": r, "seed_a": "bootstrap", "seed_b": s,
                    "metric": "spearman", "point": round(A.spearman(point, seed), 6),
                    "ci_lo": A.ci(vals_sp)[0], "ci_hi": A.ci(vals_sp)[1],
                    "n_boot": int(np.isfinite(vals_sp).sum()), "method": "month_block",
                })
                boot_rows.append({
                    "comparison": "point_vs_seed", "period_a": p, "period_b": p,
                    "role": r, "seed_a": "bootstrap", "seed_b": s,
                    "metric": "jaccard_pos@10", "point": round(A.topk_jaccard(
                        point, seed, 10, "pos"), 6),
                    "ci_lo": A.ci(vals_j)[0], "ci_hi": A.ci(vals_j)[1],
                    "n_boot": int(np.isfinite(vals_j).sum()), "method": "month_block",
                })
    for r in ROLES:
        for p1, p2 in itertools.combinations(periods, 2):
            o1 = bootstrap_months(caches[p1]["point"], r, reps=N_BOOT,
                                  seed=BOOT_SEED + 201 + periods.index(p1) * 11)
            o2 = bootstrap_months(caches[p2]["point"], r, reps=N_BOOT,
                                  seed=BOOT_SEED + 202 + periods.index(p2) * 13)
            vals_sp, vals_j = [], []
            for a, b in zip(o1["attrs"], o2["attrs"]):
                vals_sp.append(A.spearman(a, b))
                vals_j.append(A.topk_jaccard(a, b, 10, "pos"))
            boot_rows.append({
                "comparison": "period_pair", "period_a": p1, "period_b": p2,
                "role": r, "seed_a": "bootstrap", "seed_b": "bootstrap",
                "metric": "spearman", "point": round(A.spearman(
                    attr[p1][r]["point"], attr[p2][r]["point"]), 6),
                "ci_lo": A.ci(vals_sp)[0], "ci_hi": A.ci(vals_sp)[1],
                "n_boot": int(np.isfinite(vals_sp).sum()), "method": "month_block",
            })
            boot_rows.append({
                "comparison": "period_pair", "period_a": p1, "period_b": p2,
                "role": r, "seed_a": "bootstrap", "seed_b": "bootstrap",
                "metric": "jaccard_pos@10", "point": round(A.topk_jaccard(
                    attr[p1][r]["point"], attr[p2][r]["point"], 10, "pos"), 6),
                "ci_lo": A.ci(vals_j)[0], "ci_hi": A.ci(vals_j)[1],
                "n_boot": int(np.isfinite(vals_j).sum()), "method": "month_block",
            })
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(os.path.join(C.TAB, "stability_bootstrap.csv"), index=False)
    log(f"stability tables: seed pairs={len(seed_pairs)}, periods={len(period_df)}, "
        f"windows={len(window_pairs)}, bootstrap={len(boot_df)}")

    # ------------------------------------------------------------------
    # 5. Variance decomposition: seed noise vs regime shift
    # ------------------------------------------------------------------
    var_rows = []
    for r in ROLES:
        A3 = np.stack([np.stack(attr[p][r]["seeds"], axis=1) for p in periods], axis=1)
        # A3: (actors, periods, seeds)
        B = A3 - np.nanmean(A3, axis=(1, 2), keepdims=True)
        pm = np.nanmean(B, axis=(0, 2))            # (periods,)
        sm = np.nanmean(B, axis=0)                 # (periods, seeds)
        P, S, na = len(periods), C.N_SEEDS, A3.shape[0]
        ss_period = na * S * float(np.nansum(pm ** 2))
        ss_seed = na * float(np.nansum((sm - pm[:, None]) ** 2))
        ss_resid = float(np.nansum((B - sm[None, :, :]) ** 2))
        ss_total = float(np.nansum(B ** 2))
        shares = {"period_regime": ss_period / ss_total if ss_total else np.nan,
                  "seed_noise": ss_seed / ss_total if ss_total else np.nan,
                  "actor_x_time_resid": ss_resid / ss_total if ss_total else np.nan}
        for k, v in shares.items():
            var_rows.append({"role": r, "level": "within_actor_total",
                             "component": k, "share": round(float(v), 4),
                             "ss": round(float({"period_regime": ss_period,
                                                "seed_noise": ss_seed,
                                                "actor_x_time_resid": ss_resid}[k]), 3)})
        seed_var_i = np.nanmean(np.nanvar(A3, axis=2, ddof=1), axis=1)
        regime_var_i = np.nanvar(np.nanmean(A3, axis=2), axis=1, ddof=1)
        ratio = regime_var_i / np.maximum(seed_var_i, 1e-12)
        var_rows.append({"role": r, "level": "actor_ratio",
                         "component": "regime_var_over_seed_var_median",
                         "share": round(float(np.nanmedian(ratio)), 4),
                         "ss": None})
        var_rows.append({"role": r, "level": "actor_ratio",
                         "component": "regime_var_over_seed_var_p90",
                         "share": round(float(np.nanpercentile(ratio, 90)), 4),
                         "ss": None})
        var_rows.append({"role": r, "level": "actor_ratio",
                         "component": "share_actors_ratio_gt_2",
                         "share": round(float(np.nanmean(ratio > 2)), 4),
                         "ss": None})

        # Linearisation of seed variance: zbar_bucket' Sigma_beta zbar_bucket.
        coefs = np.load(os.path.join(C.AGG, "coefs.npz"))
        beta = coefs["coefs"][1:]
        Sigma = np.cov(beta, rowvar=False, ddof=1)
        bucket = A.SOURCE_OWN if r == "source" else A.TARGET_OWN
        bidx = [names.index(f) for f in bucket]
        Sb = Sigma[np.ix_(bidx, bidx)]
        diag_sigma = np.diag(Sb)
        for pi, p in enumerate(periods):
            zz = np.load(os.path.join(C.AGG, f"zbar_{p}.npz"))
            Z = zz[f"z_{r}"][:, bidx]
            n = zz[f"n_{r}"]
            pred = np.einsum("ij,jk,ik->i", Z, Sb, Z)
            diag = (Z ** 2) @ diag_sigma
            emp = np.nanvar(A3[:, pi, :], axis=1, ddof=1)
            ok = (n > 0) & np.isfinite(pred) & (diag > 0)
            amp = pred[ok] / diag[ok]
            evals, evecs = np.linalg.eigh(Sb)
            v1 = evecs[:, -1]
            pc1 = ((Z @ v1) ** 2) * evals[-1]
            pc1_share = pc1[ok] / np.maximum(pred[ok], 1e-12)
            var_rows.append({
                "role": r, "level": f"linearisation_{p}",
                "component": "median_seed_SD_ratio_empirical_over_linear",
                "share": round(float(np.nanmedian(np.sqrt(emp[ok]) /
                                                   np.maximum(np.sqrt(pred[ok]), 1e-12))), 4),
                "ss": None})
            var_rows.append({
                "role": r, "level": f"linearisation_{p}",
                "component": "median_collinearity_amplification",
                "share": round(float(np.nanmedian(amp)), 4), "ss": None})
            var_rows.append({
                "role": r, "level": f"linearisation_{p}",
                "component": "p90_collinearity_amplification",
                "share": round(float(np.nanpercentile(amp, 90)), 4), "ss": None})
            var_rows.append({
                "role": r, "level": f"linearisation_{p}",
                "component": "median_PC1_share_of_linear_variance",
                "share": round(float(np.nanmedian(pc1_share)), 4), "ss": None})
            var_rows.append({
                "role": r, "level": f"linearisation_{p}",
                "component": "corr_empirical_SD_vs_linear_SD",
                "share": round(float(np.corrcoef(np.sqrt(emp[ok]),
                                                 np.sqrt(pred[ok]))[0, 1]), 4),
                "ss": None})
    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(os.path.join(C.TAB, "variance_decomposition.csv"), index=False)

    # Per-actor seed variance / regime shift
    actor_rows, regime_rows = [], []
    for r in ROLES:
        A3 = np.stack([np.stack(attr[p][r]["seeds"], axis=1) for p in periods], axis=1)
        point = np.stack([attr[p][r]["point"] for p in periods], axis=1)  # A×P
        seed_sd = np.nanstd(A3, axis=2, ddof=1)                          # A×P
        for i in range(n_actors):
            rec = {"role": r, "aid": i,
                   "seed_sd_inner": round(float(seed_sd[i, 0]), 6),
                   "seed_sd_retro": round(float(seed_sd[i, 1]), 6),
                   "seed_sd_covid": round(float(seed_sd[i, 2]), 6),
                   "attr_inner": round(float(point[i, 0]), 6),
                   "attr_retro": round(float(point[i, 1]), 6),
                   "attr_covid": round(float(point[i, 2]), 6)}
            rec["regime_range"] = round(float(np.nanmax(point[i]) - np.nanmin(point[i])), 6)
            rec["regime_over_seed"] = round(float(rec["regime_range"] /
                                                  max(rec["seed_sd_inner"], 1e-9)), 3)
            actor_rows.append(rec)
            # standardised regime shift vs pooled seed noise
            pooled = np.nanmean(seed_sd[i])
            for p1, p2, tag in ((0, 1, "inner_retro"), (0, 2, "inner_covid"),
                                (1, 2, "retro_covid")):
                diff = point[i, p2] - point[i, p1]
                regime_rows.append({
                    "role": r, "aid": i, "shift": tag,
                    "delta": round(float(diff), 6),
                    "seed_sd_pooled": round(float(pooled), 6),
                    "z": round(float(diff / max(pooled, 1e-9)), 3),
                })
    pd.DataFrame(actor_rows).to_csv(
        os.path.join(C.TAB, "actor_seed_variance.csv"), index=False)
    pd.DataFrame(regime_rows).to_csv(
        os.path.join(C.TAB, "actor_regime_shift.csv"), index=False)

    # Feature-level seed stability
    feat_rows = []
    for p in periods:
        for r in ROLES:
            mats = []
            for s in range(C.N_SEEDS):
                z = caches[p][f"rep{s}"]
                n = z[f"feat_n_{r}"]
                mats.append(z[f"feat_sum_{r}"] / np.maximum(n, 1)[:, None])
            for i in range(n_actors):
                if not np.isfinite(mats[0][i]).any():
                    continue
                sp = [A.spearman(mats[a][i], mats[b][i])
                      for a, b in itertools.combinations(range(C.N_SEEDS), 2)]
                tops = [names[int(np.nanargmax(np.abs(mats[s][i])))]
                        for s in range(C.N_SEEDS)]
                top3 = [set(np.argsort(-np.abs(mats[s][i]))[:3].tolist())
                        for s in range(C.N_SEEDS)]
                j3 = [jaccard(top3[a], top3[b])
                      for a, b in itertools.combinations(range(C.N_SEEDS), 2)]
                vals, cnts = np.unique(tops, return_counts=True)
                feat_rows.append({
                    "period": p, "role": r, "aid": i,
                    "feature_spearman_median": round(float(np.nanmedian(sp)), 4),
                    "feature_spearman_min": round(float(np.nanmin(sp)), 4),
                    "top1_mode": vals[int(np.argmax(cnts))],
                    "top1_mode_share": round(float(cnts.max() / C.N_SEEDS), 2),
                    "top3_jaccard_median": round(float(np.nanmedian(j3)), 3),
                })
    pd.DataFrame(feat_rows).to_csv(
        os.path.join(C.TAB, "feature_stability.csv"), index=False)

    # ------------------------------------------------------------------
    # 6. Top-k churn: actor-month alert sets
    # ------------------------------------------------------------------
    churn_rows = []
    for p in periods:
        for r in ROLES:
            for score_name, kind in (("attribution", "attr"), ("risk", "risk")):
                scores = [actor_month_scores(caches[p][f"rep{s}"], r, kind)
                          for s in range(C.N_SEEDS)]
                point = actor_month_scores(caches[p]["point"], r, kind)
                M = point.shape[1]
                sd_cell = np.nanstd(np.stack(scores), axis=0, ddof=1)
                for k in (5, 10, 20):
                    pair_j, gaps = [], []
                    for m in range(M):
                        vals = point[np.isfinite(point[:, m]), m]
                        if len(vals) > k:
                            vals = np.sort(vals)[::-1]
                            gaps.append((vals[k - 1] - vals[k]) /
                                        max(float(np.nanmedian(sd_cell[:, m])), 1e-9))
                    for i, j in itertools.combinations(range(C.N_SEEDS), 2):
                        vals = []
                        for m in range(M):
                            a = topk_set(scores[i][:, m], k)
                            b = topk_set(scores[j][:, m], k)
                            if a and b:
                                vals.append(jaccard(a, b))
                        if vals:
                            pair_j.append(float(np.mean(vals)))
                    point_j = []
                    for s in range(C.N_SEEDS):
                        vals = []
                        for m in range(M):
                            a = topk_set(point[:, m], k)
                            b = topk_set(scores[s][:, m], k)
                            if a and b:
                                vals.append(jaccard(a, b))
                        if vals:
                            point_j.append(float(np.mean(vals)))
                    churn_rows.append({
                        "period": p, "role": r, "score": score_name, "k": k,
                        "seed_pair_mean_jaccard": round(float(np.mean(pair_j)), 4),
                        "seed_pair_min": round(float(np.min(pair_j)), 4),
                        "seed_pair_max": round(float(np.max(pair_j)), 4),
                        "point_vs_seed_mean_jaccard": round(float(np.mean(point_j)), 4),
                        "mean_churn": round(1 - float(np.mean(pair_j)), 4),
                        "median_boundary_gap_over_seed_sd": round(float(np.median(gaps)), 3)
                        if gaps else None,
                        "n_months": int(M),
                    })
    churn_df = pd.DataFrame(churn_rows)
    churn_df.to_csv(os.path.join(C.TAB, "topk_churn_actor_month.csv"), index=False)

    # Window-level top-k churn
    win_churn = []
    for p in periods:
        wnames = json.loads(str(caches[p]["point"]["meta"])).get("window_names", [])
        if not wnames:
            continue
        for r in ROLES:
            W = []
            for s in range(C.N_SEEDS):
                ww, _ = window_attribution(caches[p][f"rep{s}"], r, wnames)
                W.append(ww)
            for k in (5, 10, 20):
                for i, j in itertools.combinations(range(len(wnames)), 2):
                    vals = []
                    for s in range(C.N_SEEDS):
                        a = topk_set(W[s][i], k)
                        b = topk_set(W[s][j], k)
                        if a and b:
                            vals.append(jaccard(a, b))
                    win_churn.append({
                        "period": p, "role": r, "k": k,
                        "window_a": wnames[i], "window_b": wnames[j],
                        "seed_median_jaccard": round(float(np.median(vals)), 4),
                        "seed_min_jaccard": round(float(np.min(vals)), 4),
                        "n_seeds": len(vals),
                    })
    pd.DataFrame(win_churn).to_csv(
        os.path.join(C.TAB, "topk_churn_window.csv"), index=False)

    # ------------------------------------------------------------------
    # 7. Decision impact: attribution alert vs risk alert
    # ------------------------------------------------------------------
    decision_rows = []
    for p in periods:
        for r in ROLES:
            attr_scores = actor_month_scores(caches[p]["point"], r, "attr")
            risk_scores = actor_month_scores(caches[p]["point"], r, "risk")
            n = caches[p]["point"][f"count_{r}"]
            y = caches[p]["point"][f"y_sum_{r}"]
            cell_prec = np.where(n > 0, y / np.maximum(n, 1), np.nan)
            month_base = np.where(n.sum(axis=0) > 0,
                                  y.sum(axis=0) / np.maximum(n.sum(axis=0), 1), np.nan)
            M = n.shape[1]
            for k in (5, 10, 20):
                for tag, score in (("attribution", attr_scores),
                                   ("risk", risk_scores)):
                    per_month, lifts, overlaps = [], [], []
                    for m in range(M):
                        a = topk_set(score[:, m], k)
                        b = topk_set(risk_scores[:, m], k)
                        if not a:
                            continue
                        vals = [cell_prec[i, m] for i in a if np.isfinite(cell_prec[i, m])]
                        if not vals:
                            continue
                        prec = float(np.mean(vals))
                        base = float(month_base[m])
                        per_month.append(prec)
                        lifts.append(prec / base if base > 0 else np.nan)
                        overlaps.append(jaccard(a, b))
                    # month bootstrap CI for mean lift
                    rng = np.random.default_rng(BOOT_SEED + 401 + k +
                                                (0 if r == "source" else 1))
                    vals = np.asarray(lifts)
                    boot = []
                    for _ in range(N_BOOT):
                        take = rng.integers(0, len(vals), size=len(vals))
                        v = vals[take]
                        v = v[np.isfinite(v)]
                        if len(v):
                            boot.append(float(np.mean(v)))
                    seed_lifts = []
                    for s in range(C.N_SEEDS):
                        ss = actor_month_scores(caches[p][f"rep{s}"], r,
                                                "attr" if tag == "attribution" else "risk")
                        vv = []
                        for m in range(M):
                            a = topk_set(ss[:, m], k)
                            if not a:
                                continue
                            z = [cell_prec[i, m] for i in a
                                 if np.isfinite(cell_prec[i, m])]
                            if z and month_base[m] > 0:
                                vv.append(float(np.mean(z)) / float(month_base[m]))
                        if vv:
                            seed_lifts.append(float(np.mean(vv)))
                    decision_rows.append({
                        "period": p, "role": r, "k": k, "alert": tag,
                        "mean_cell_precision": round(float(np.nanmean(per_month)), 5),
                        "mean_lift_vs_month_base": round(float(np.nanmean(vals)), 4),
                        "median_lift": round(float(np.nanmedian(vals)), 4),
                        "lift_ci_lo": round(float(np.percentile(boot, 2.5)), 4)
                        if boot else None,
                        "lift_ci_hi": round(float(np.percentile(boot, 97.5)), 4)
                        if boot else None,
                        "lift_seed_median": round(float(np.median(seed_lifts)), 4)
                        if seed_lifts else None,
                        "lift_seed_min": round(float(np.min(seed_lifts)), 4)
                        if seed_lifts else None,
                        "lift_seed_max": round(float(np.max(seed_lifts)), 4)
                        if seed_lifts else None,
                        "mean_jaccard_with_risk_alert": round(float(np.nanmean(overlaps)), 4),
                        "n_months": int(len(per_month)),
                    })
    dec_df = pd.DataFrame(decision_rows)
    dec_df.to_csv(os.path.join(C.TAB, "decision_impact_actor_month.csv"), index=False)

    # ------------------------------------------------------------------
    # 7b. Conditional explanation stability inside the risk alert set
    # ------------------------------------------------------------------
    cond_rows = []
    for p in periods:
        for r in ROLES:
            risk = actor_month_scores(caches[p]["point"], r, "risk")
            attrs = [actor_month_scores(caches[p][f"rep{s}"], r, "attr")
                     for s in range(C.N_SEEDS)]
            point_attr = actor_month_scores(caches[p]["point"], r, "attr")
            for k in (5, 10, 20):
                cells = []
                for m in range(risk.shape[1]):
                    s = topk_set(risk[:, m], k)
                    cells.extend((i, m) for i in s)
                if not cells:
                    continue
                cells = np.array(cells, dtype=int)
                vals = [point_attr[cells[:, 0], cells[:, 1]]]
                vals += [a[cells[:, 0], cells[:, 1]] for a in attrs]
                pair_sp, pair_sg = [], []
                for i, j in itertools.combinations(range(C.N_SEEDS), 2):
                    pair_sp.append(A.spearman(vals[1 + i], vals[1 + j]))
                    pair_sg.append(A.sign_agreement(vals[1 + i], vals[1 + j]))
                cond_rows.append({
                    "period": p, "role": r, "k": k,
                    "n_cells": int(len(cells)),
                    "seed_pair_spearman_median": round(float(np.nanmedian(pair_sp)), 4),
                    "seed_pair_spearman_min": round(float(np.nanmin(pair_sp)), 4),
                    "seed_pair_sign_median": round(float(np.nanmedian(pair_sg)), 4),
                    "point_vs_seed_spearman_median": round(float(np.nanmedian(
                        [A.spearman(vals[0], vals[1 + s])
                         for s in range(C.N_SEEDS)])), 4),
                })
    cond_df = pd.DataFrame(cond_rows)
    cond_df.to_csv(os.path.join(C.TAB, "stability_conditional_on_risk.csv"),
                   index=False)

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    def _pick(df, **kw):
        m = np.ones(len(df), dtype=bool)
        for k, v in kw.items():
            m &= (df[k] == v).to_numpy()
        return df[m]

    summary = {
        "sprint": "B.2",
        "task": "stability / top-k churn / decision impact",
        "status": "provisional_holdout_not_integrated",
        "rubric": {"stable": STABLE, "conditional": CONDITIONAL,
                   "note": "fixed before inspecting stability numbers; reporting "
                           "convention only, not model selection"},
        "protocol": {
            "periods": C.PERIODS,
            "embargo": {"train": C.TRAIN_EMBARGO, "select": C.SELECT_EMBARGO,
                        "inner": 275, "retro": 335},
            "n_seed_replicates": C.N_SEEDS,
            "seed_semantics": "dyad-cluster bootstrap refits with fixed C",
            "n_actors": int(n_actors),
        },
        "seed_stability": {
            p: {r: {m: (_pick(seed_summary, period=p, role=r, metric=m,
                              comparison="seed_pair").iloc[0].to_dict()
                       if len(_pick(seed_summary, period=p, role=r, metric=m,
                                    comparison="seed_pair")) else None)
                   for m in ["spearman", "sign_agreement", "jaccard_pos@10",
                             "jaccard_pos@20"]}
                for r in ROLES} for p in periods},
        "sign_flip": pd.DataFrame(sign_rows).to_dict(orient="records"),
        "period_stability": period_df.to_dict(orient="records"),
        "window_stability": {
            p: window_pairs[window_pairs.period == p].groupby(
                ["role", "window_a", "window_b"]).agg(
                spearman_median=("spearman", "median"),
                jaccard10_median=("jaccard_pos@10", "median"),
                sign_median=("sign_agreement", "median")).reset_index().round(4)
            .to_dict(orient="records")
            for p in periods if p in set(window_pairs.period)},
        "variance_decomposition": var_df.to_dict(orient="records"),
        "topk_actor_month": churn_df.to_dict(orient="records"),
        "topk_window": pd.DataFrame(win_churn).to_dict(orient="records")
        if win_churn else [],
        "decision_impact_actor_month": dec_df.to_dict(orient="records"),
        "conditional_on_risk": cond_df.to_dict(orient="records"),
        "bootstrap": boot_df.to_dict(orient="records"),
        "runtime_seconds": elapsed(t_start),
    }
    C.atomic_json(os.path.join(C.OUT, "summary_stability.json"), summary)
    log(f"summary_stability.json written; total {elapsed(t_start)}s")
    rl.close()


if __name__ == "__main__":
    main()
