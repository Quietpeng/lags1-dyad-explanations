"""Recompute HGB occlusion agreement on a per-row (not summed) scale.

The streaming HGB occlusion cache stores the *sum* of ΔNLL over rows.  Actors
with more rows therefore look more important.  This post-process divides by the
actor/cell row count so the comparison with the logistic attribution (already a
per-row mean) is exposure-neutral, and overwrites the agreement tables.
"""

from __future__ import annotations

import itertools
import json
import os
import re

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import sprint_b_common as C
from framework import aggregation as A
from framework.config import load_config
from framework.utils import jdump, log

ROLES = ("source", "target")
KS = (5, 10, 20, 50)
N_BOOT = 1000
BOOT_SEED = C.SEED_BASE + 29


def _actor_agree(attr, logocc, hgbocc):
    out = {"n": int((np.isfinite(attr) & np.isfinite(hgbocc)).sum()),
           "spearman_attr_hgb": A.spearman(attr, hgbocc),
           "spearman_abs_attr_hgb": A.spearman(np.abs(attr), np.abs(hgbocc)),
           "sign_attr_hgb": A.sign_agreement(attr, hgbocc),
           "spearman_logocc_hgb": A.spearman(logocc, hgbocc),
           "spearman_abs_logocc_hgb": A.spearman(np.abs(logocc), np.abs(hgbocc)),
           "sign_logocc_hgb": A.sign_agreement(logocc, hgbocc),
           "spearman_attr_logocc": A.spearman(attr, logocc),
           "sign_attr_logocc": A.sign_agreement(attr, logocc)}
    for k in KS:
        out[f"jaccard_pos@{k}"] = A.topk_jaccard(attr, hgbocc, k, "pos")
        out[f"jaccard_abs@{k}"] = A.topk_jaccard(attr, hgbocc, k, "abs")
        out[f"jaccard_logocc_pos@{k}"] = A.topk_jaccard(logocc, hgbocc, k, "pos")
        out[f"jaccard_logocc_abs@{k}"] = A.topk_jaccard(logocc, hgbocc, k, "abs")
    return out


def _topk(scores, k):
    x = np.asarray(scores, dtype=np.float64)
    ok = np.isfinite(x)
    if ok.sum() < k:
        return set()
    return set(np.argsort(-np.where(ok, x, -np.inf), kind="stable")[:k].tolist())


def _jac(a, b):
    if not a or not b:
        return float("nan")
    return len(a & b) / len(a | b)


def _dyad_scores(Cmat, did, y, p, n_dyads, cols_s, cols_t):
    n = np.bincount(did, minlength=n_dyads).astype(np.float64)
    tot = np.bincount(did, weights=Cmat.sum(axis=1), minlength=n_dyads)
    src = np.bincount(did, weights=Cmat[:, cols_s].sum(axis=1), minlength=n_dyads)
    tgt = np.bincount(did, weights=Cmat[:, cols_t].sum(axis=1), minlength=n_dyads)
    ysum = np.bincount(did, weights=y, minlength=n_dyads)
    psum = np.bincount(did, weights=p, minlength=n_dyads)
    den = np.maximum(n, 1)
    return {"n": n, "y_sum": ysum,
            "risk": np.where(n > 0, psum / den, np.nan),
            "total": np.where(n > 0, tot / den, np.nan),
            "source": np.where(n > 0, src / den, np.nan),
            "target": np.where(n > 0, tgt / den, np.nan)}


def _boot_actor(a, b, reps, seed):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    idx = np.where(np.isfinite(a) & np.isfinite(b))[0]
    if len(idx) < 3:
        return None, None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        take = rng.choice(idx, size=len(idx), replace=True)
        v = A.spearman(a[take], b[take])
        if np.isfinite(v):
            vals.append(v)
    return A.ci(vals) if vals else (None, None)


def _boot_months(a, b, reps, seed):
    A_, M = a.shape
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        take = rng.integers(0, M, size=M)
        v = A.spearman(a[:, take].ravel(), b[:, take].ravel())
        if np.isfinite(v):
            vals.append(v)
    return A.ci(vals) if vals else (None, None)


def main() -> None:
    C.require_writable()
    path_h = os.path.join(C.OUT, "summary_hgb.json")
    H = json.load(open(path_h, encoding="utf-8")) if os.path.exists(path_h) else {}
    period_names = [p["name"] for p in C.PERIODS]
    actors = duckdb.connect().execute(
        "SELECT * FROM read_parquet('%s') ORDER BY aid" %
        os.path.join(load_config()["paths"]["cache_dir"], "actors.parquet")
        .replace("\\", "/")).df()
    agree_rows, actor_rows, fail_rows = [], [], []
    fail_like_rows = []
    for p in period_names:
        zp = np.load(os.path.join(C.AGG, f"agg_point_{p}.npz"), allow_pickle=False)
        zh = np.load(os.path.join(C.AGG, f"hgb_occ_{p}.npz"), allow_pickle=False)
        for role in ROLES:
            j = 0 if role == "source" else 1
            cnt = zp[f"count_{role}"]                        # (A, M)
            n_actor = cnt.sum(axis=1)
            attr = np.where(n_actor > 0,
                            zp[f"bucket_sum_{role}"][:, :, j].sum(axis=1) /
                            np.maximum(n_actor, 1), np.nan)
            log_sum = zp[f"occ_actor_total_{role}"]
            hgb_sum = zh[f"occ_total_{role}"]
            log_mean = np.where(n_actor > 0, log_sum / np.maximum(n_actor, 1), np.nan)
            hgb_mean = np.where(n_actor > 0, hgb_sum / np.maximum(n_actor, 1), np.nan)

            rec = {"period": p, "role": role, "level": "actor",
                   "n": int(np.isfinite(attr).sum()),
                   "normalization": "mean_delta_nll_per_row",
                   "spearman_attr_hgb_total": A.spearman(attr, hgb_sum),
                   "spearman_attr_hgb_mean": A.spearman(attr, hgb_mean),
                   "spearman_attr_hgb": A.spearman(attr, hgb_mean),
                   "spearman_abs_attr_hgb": A.spearman(np.abs(attr), np.abs(hgb_mean)),
                   "sign_attr_hgb": A.sign_agreement(attr, hgb_mean),
                   "spearman_logocc_hgb": A.spearman(log_mean, hgb_mean),
                   "spearman_abs_logocc_hgb": A.spearman(np.abs(log_mean), np.abs(hgb_mean)),
                   "sign_logocc_hgb": A.sign_agreement(log_mean, hgb_mean),
                   "spearman_attr_logocc": A.spearman(attr, log_mean),
                   "sign_attr_logocc": A.sign_agreement(attr, log_mean)}
            for k in KS:
                rec[f"jaccard_pos@{k}"] = A.topk_jaccard(attr, hgb_mean, k, "pos")
                rec[f"jaccard_abs@{k}"] = A.topk_jaccard(attr, hgb_mean, k, "abs")
                rec[f"jaccard_logocc_pos@{k}"] = A.topk_jaccard(log_mean, hgb_mean,
                                                                k, "pos")
                rec[f"jaccard_logocc_abs@{k}"] = A.topk_jaccard(log_mean, hgb_mean,
                                                                k, "abs")
            ci = _boot_actor(attr, hgb_mean, N_BOOT, BOOT_SEED + 7 * j + len(p))
            rec["spearman_ci_lo"], rec["spearman_ci_hi"] = ci
            ci2 = _boot_actor(log_mean, hgb_mean, N_BOOT, BOOT_SEED + 17 + 7 * j + len(p))
            rec["spearman_logocc_ci_lo"], rec["spearman_logocc_ci_hi"] = ci2
            agree_rows.append(rec)

            # actor-month, per-cell mean
            am_attr = np.where(cnt > 0, zp[f"bucket_sum_{role}"][:, :, j] /
                               np.maximum(cnt, 1), np.nan)
            am_log = np.where(cnt > 0, zp[f"occ_actor_month_{role}"] /
                              np.maximum(cnt, 1), np.nan)
            am_hgb = np.where(cnt > 0, zh[f"occ_month_{role}"] /
                              np.maximum(cnt, 1), np.nan)
            a, lo, hg = am_attr.ravel(), am_log.ravel(), am_hgb.ravel()
            recm = {"period": p, "role": role, "level": "actor_month",
                    "normalization": "mean_delta_nll_per_cell",
                    "n": int((np.isfinite(a) & np.isfinite(hg)).sum()),
                    "spearman_attr_hgb": A.spearman(a, hg),
                    "spearman_abs_attr_hgb": A.spearman(np.abs(a), np.abs(hg)),
                    "sign_attr_hgb": A.sign_agreement(a, hg),
                    "spearman_logocc_hgb": A.spearman(lo, hg),
                    "spearman_abs_logocc_hgb": A.spearman(np.abs(lo), np.abs(hg)),
                    "sign_logocc_hgb": A.sign_agreement(lo, hg),
                    "spearman_attr_logocc": A.spearman(a, lo)}
            for k in (10, 50, 100):
                recm[f"jaccard_pos@{k}"] = A.topk_jaccard(a, hg, k, "pos")
                recm[f"jaccard_abs@{k}"] = A.topk_jaccard(a, hg, k, "abs")
                recm[f"jaccard_logocc_abs@{k}"] = A.topk_jaccard(lo, hg, k, "abs")
            ci = _boot_months(am_attr, am_hgb, N_BOOT, BOOT_SEED + 101 + 7 * j + len(p))
            recm["spearman_ci_lo"], recm["spearman_ci_hi"] = ci
            ci2 = _boot_months(am_log, am_hgb, N_BOOT,
                               BOOT_SEED + 117 + 7 * j + len(p))
            recm["spearman_logocc_ci_lo"], recm["spearman_logocc_ci_hi"] = ci2
            agree_rows.append(recm)

            for i in range(len(actors)):
                if not np.isfinite(attr[i]):
                    continue
                actor_rows.append({
                    "period": p, "role": role, "aid": i,
                    "actor": actors["actor"].iloc[i],
                    "region": actors["region"].iloc[i],
                    "n_rows": int(n_actor[i]),
                    "attribution": round(float(attr[i]), 6),
                    "logistic_occlusion_mean_dnll": round(float(log_mean[i]), 6),
                    "hgb_occlusion_mean_dnll": round(float(hgb_mean[i]), 6),
                })

            ok = np.isfinite(attr) & np.isfinite(hgb_mean)
            ra = rankdata(np.abs(attr[ok]))
            rh = rankdata(np.abs(hgb_mean[ok]))
            diff = np.abs(ra - rh)
            idx = np.where(ok)[0]
            for o in np.argsort(-diff)[:10]:
                i = int(idx[o])
                fail_rows.append({
                    "period": p, "role": role, "aid": i,
                    "actor": actors["actor"].iloc[i],
                    "region": actors["region"].iloc[i],
                    "n_rows": int(n_actor[i]),
                    "attribution": round(float(attr[i]), 6),
                    "logistic_occlusion_mean_dnll": round(float(log_mean[i]), 6),
                    "hgb_occlusion_mean_dnll": round(float(hgb_mean[i]), 6),
                    "rank_abs_attr": int(ra[o]), "rank_abs_hgb": int(rh[o]),
                    "rank_gap": int(diff[o]),
                    "sign_disagreement": bool(attr[i] * hgb_mean[i] < 0),
                    "failure_type": "rank_gap",
                })

            # Like-for-like failures: logistic occlusion vs HGB occlusion.
            ok2 = np.isfinite(log_mean) & np.isfinite(hgb_mean)
            rl = rankdata(np.abs(log_mean[ok2]))
            rh2 = rankdata(np.abs(hgb_mean[ok2]))
            diff2 = np.abs(rl - rh2)
            idx2 = np.where(ok2)[0]
            for o in np.argsort(-diff2)[:10]:
                i = int(idx2[o])
                fail_like_rows.append({
                    "period": p, "role": role, "aid": i,
                    "actor": actors["actor"].iloc[i],
                    "region": actors["region"].iloc[i],
                    "n_rows": int(n_actor[i]),
                    "attribution": round(float(attr[i]), 6),
                    "logistic_occlusion_mean_dnll": round(float(log_mean[i]), 6),
                    "hgb_occlusion_mean_dnll": round(float(hgb_mean[i]), 6),
                    "rank_abs_logistic": int(rl[o]),
                    "rank_abs_hgb": int(rh2[o]),
                    "rank_gap": int(diff2[o]),
                    "sign_disagreement": bool(log_mean[i] * hgb_mean[i] < 0),
                    "failure_type": "rank_gap_like_for_like",
                })

    agree = pd.DataFrame(agree_rows)
    agree.to_csv(os.path.join(C.TAB, "hgb_occlusion_agreement.csv"), index=False)
    pd.DataFrame(actor_rows).to_csv(
        os.path.join(C.TAB, "hgb_occlusion_actor.csv"), index=False)
    pd.DataFrame(fail_rows).to_csv(
        os.path.join(C.TAB, "hgb_occlusion_failures.csv"), index=False)
    pd.DataFrame(fail_like_rows).to_csv(
        os.path.join(C.TAB, "hgb_occlusion_failures_like_for_like.csv"), index=False)
    H["agreement"] = {
        "normalization": "per-row mean delta-NLL (exposure neutral)",
        "actor": agree[agree.level == "actor"].to_dict(orient="records"),
        "actor_month": agree[agree.level == "actor_month"].to_dict(orient="records"),
        "failures_top": pd.DataFrame(fail_rows).head(40).to_dict(orient="records"),
        "failures_like_for_like_top": pd.DataFrame(fail_like_rows).head(40)
        .to_dict(orient="records"),
    }
    H["normalization_note"] = (
        "Totals and means are both reported in hgb_occlusion_agreement.csv; the "
        "headline uses per-row means so that actor exposure does not drive the "
        "rank correlation."
    )

    # ------------------------------------------------------------------
    # HGB metadata (the fit finished; only the dyad block crashed)
    # ------------------------------------------------------------------
    hm = pd.read_csv(os.path.join(C.TAB, "hgb_metrics.csv"))
    if len(hm):
        H["hgb"] = {
            "n_iter": int(hm["n_iter"].iloc[0]),
            "select_log_loss": float(hm["select_log_loss"].iloc[0]),
            "select_curve": [],
            "metrics": hm.to_dict(orient="records"),
            "occlusion": "actor-own feature block replaced by train mean",
        }

    # ------------------------------------------------------------------
    # Dyad-level stability / decision impact (no HGB refit)
    # ------------------------------------------------------------------
    cfg = load_config()
    b = C.load_splits(cfg)
    names = b["tr"]["feature_names"]
    n_dyads = b["n_dyads"]
    cols_s = [names.index(f) for f in A.SOURCE_OWN]
    cols_t = [names.index(f) for f in A.TARGET_OWN]
    inner = C.split_rows(b["va"], 252, 274)
    X_inner = inner["X"]
    did = inner["did"].astype(np.int64)
    y_inner = inner["y"].astype(np.float64)
    n_inner = len(y_inner)
    coefs = np.load(os.path.join(C.AGG, "coefs.npz"))
    cw, ci_ = coefs["coefs"], coefs["intercepts"]
    point_cache = np.load(os.path.join(C.CACHE, "contrib_inner_point.npz"),
                          allow_pickle=False)
    assert np.array_equal(point_cache["did"].astype(np.int64), did)
    assert np.array_equal(point_cache["y"].astype(np.int64),
                          y_inner.astype(np.int64))
    mu = b["tr"]["X"].mean(axis=0).astype(np.float32)
    sd = b["tr"]["X"].std(axis=0).astype(np.float32)
    sd = np.where(sd < 1e-8, 1.0, sd).astype(np.float32)
    fit_point = {"mu": mu, "sd": sd, "coef": cw[0], "intercept": float(ci_[0])}
    dyad = {}
    point_C = point_cache["contribution"].astype(np.float64)
    dyad["point"] = _dyad_scores(point_C, did, y_inner,
                                 point_cache["p"].astype(np.float64),
                                 n_dyads, cols_s, cols_t)
    C_rebuilt = A.standardized(fit_point, X_inner) * cw[0][None, :]
    log(f"dyad point rebuild max |dC|={np.abs(C_rebuilt - point_C).max():.2e}")
    del C_rebuilt, point_C
    for s in range(C.N_SEEDS):
        fit_s = {"mu": mu, "sd": sd, "coef": cw[1 + s],
                 "intercept": float(ci_[1 + s])}
        Cs = A.standardized(fit_s, X_inner) * cw[1 + s][None, :]
        ps = A.sigmoid(float(ci_[1 + s]) + Cs.sum(axis=1))
        dyad[f"rep{s}"] = _dyad_scores(Cs, did, y_inner, ps, n_dyads,
                                       cols_s, cols_t)
        del Cs
        log(f"dyad seed {s}: source mean="
            f"{np.nanmean(dyad[f'rep{s}']['source']):.4f}")

    dyad_stab, dyad_churn = [], []
    for key in ("total", "source", "target", "risk"):
        for i, j in itertools.combinations(range(C.N_SEEDS), 2):
            a = dyad[f"rep{i}"][key]
            bb = dyad[f"rep{j}"][key]
            rec = {"score": key, "seed_a": i, "seed_b": j,
                   "spearman": round(A.spearman(a, bb), 6),
                   "sign_agreement": round(A.sign_agreement(a, bb), 6)}
            for k in (10, 50, 100, 500):
                rec[f"jaccard_pos@{k}"] = round(A.topk_jaccard(a, bb, k, "pos"), 6)
            dyad_stab.append(rec)
        for k in (10, 50, 100, 500):
            vals = [A.topk_jaccard(dyad[f"rep{i}"][key], dyad[f"rep{j}"][key], k, "pos")
                    for i, j in itertools.combinations(range(C.N_SEEDS), 2)]
            pvals = [A.topk_jaccard(dyad["point"][key], dyad[f"rep{s}"][key], k, "pos")
                     for s in range(C.N_SEEDS)]
            stack = np.stack([dyad[f"rep{s}"][key] for s in range(C.N_SEEDS)])
            sd_dyad = np.nanstd(stack, axis=0, ddof=1)
            pv = dyad["point"][key]
            pv = np.sort(pv[np.isfinite(pv)])[::-1]
            gap = ((pv[k - 1] - pv[k]) / max(float(np.nanmedian(sd_dyad)), 1e-9)
                   if len(pv) > k else np.nan)
            dyad_churn.append({
                "score": key, "k": k,
                "seed_pair_mean_jaccard": round(float(np.mean(vals)), 4),
                "seed_pair_min": round(float(np.min(vals)), 4),
                "seed_pair_max": round(float(np.max(vals)), 4),
                "point_vs_seed_mean_jaccard": round(float(np.mean(pvals)), 4),
                "mean_churn": round(1 - float(np.mean(vals)), 4),
                "boundary_gap_over_seed_sd": round(float(gap), 3)
                if np.isfinite(gap) else None,
            })
    pd.DataFrame(dyad_stab).to_csv(
        os.path.join(C.TAB, "dyad_stability.csv"), index=False)
    pd.DataFrame(dyad_churn).to_csv(
        os.path.join(C.TAB, "topk_churn_dyad.csv"), index=False)

    dec_rows = []
    total_pos = float(y_inner.sum())
    base = total_pos / n_inner
    for score in ("risk", "total", "source", "target"):
        for k in (10, 50, 100, 500, 1000):
            sel = _topk(dyad["point"][score], k)
            if not sel:
                continue
            sel = np.array(sorted(sel), dtype=np.int64)
            rows = float(dyad["point"]["n"][sel].sum())
            pos = float(dyad["point"]["y_sum"][sel].sum())
            prec = pos / rows if rows else np.nan
            rec = {"alert": score, "k": k, "n_dyads": int(len(sel)),
                   "rows_reviewed": int(rows), "coverage": round(rows / n_inner, 4),
                   "positives": int(pos), "precision": round(prec, 5),
                   "recall": round(pos / total_pos, 5),
                   "lift_vs_base": round(prec / base, 4) if base else None,
                   "jaccard_with_risk": round(_jac(
                       set(sel.tolist()), _topk(dyad["point"]["risk"], k)), 4)}
            lifts = []
            for s in range(C.N_SEEDS):
                ss = _topk(dyad[f"rep{s}"][score], k)
                if not ss:
                    continue
                ss = np.array(sorted(ss), dtype=np.int64)
                rr = float(dyad[f"rep{s}"]["n"][ss].sum())
                pp = float(dyad[f"rep{s}"]["y_sum"][ss].sum())
                lifts.append((pp / rr) / base if rr else np.nan)
            rec["lift_seed_median"] = round(float(np.nanmedian(lifts)), 4)
            rec["lift_seed_min"] = round(float(np.nanmin(lifts)), 4)
            rec["lift_seed_max"] = round(float(np.nanmax(lifts)), 4)
            dec_rows.append(rec)
    dec = pd.DataFrame(dec_rows)
    dec.to_csv(os.path.join(C.TAB, "decision_impact_dyad.csv"), index=False)

    # Conditional explanation stability inside the dyad risk alert set.
    cond_rows = []
    for k in (10, 50, 100, 500):
        sel = _topk(dyad["point"]["risk"], k)
        if not sel:
            continue
        sel = np.array(sorted(sel), dtype=np.int64)
        for score in ("source", "target"):
            vals = [dyad["point"][score][sel]]
            vals += [dyad[f"rep{s}"][score][sel] for s in range(C.N_SEEDS)]
            sp = [A.spearman(vals[1 + i], vals[1 + j])
                  for i, j in itertools.combinations(range(C.N_SEEDS), 2)]
            sg = [A.sign_agreement(vals[1 + i], vals[1 + j])
                  for i, j in itertools.combinations(range(C.N_SEEDS), 2)]
            cond_rows.append({
                "k": k, "score": score, "n_dyads": int(len(sel)),
                "seed_pair_spearman_median": round(float(np.nanmedian(sp)), 4),
                "seed_pair_spearman_min": round(float(np.nanmin(sp)), 4),
                "seed_pair_sign_median": round(float(np.nanmedian(sg)), 4),
                "point_vs_seed_spearman_median": round(float(np.nanmedian(
                    [A.spearman(vals[0], vals[1 + s])
                     for s in range(C.N_SEEDS)])), 4),
            })
    cond_dyad = pd.DataFrame(cond_rows)
    cond_dyad.to_csv(os.path.join(C.TAB, "stability_conditional_on_risk_dyad.csv"),
                     index=False)

    pred_dir = os.path.join(C.OUT, "..", "holdout", "preds")
    files = sorted(f for f in os.listdir(pred_dir)
                   if f.endswith((".npz", ".csv", ".json"))) \
        if os.path.isdir(pred_dir) else []
    H["status"] = "provisional_holdout_not_integrated"
    H["holdout"] = {
        "available_at_audit": False,
        "completed_later": True,
        "used": False,
        "files": files,
        "note": ("The one-shot holdout completed after this audit was frozen; "
                 "its predictions were not used and the audit was not re-run. "
                 "All stability numbers remain provisional."),
    }
    H["dyad"] = {
        "n_rows": int(n_inner),
        "n_dyads": int(len(np.unique(did))),
        "stability": pd.DataFrame(dyad_stab).to_dict(orient="records"),
        "churn": pd.DataFrame(dyad_churn).to_dict(orient="records"),
        "decision_impact": dec.to_dict(orient="records"),
        "conditional_on_risk": cond_dyad.to_dict(orient="records"),
    }
    jdump(os.path.join(C.OUT, "summary_hgb.json"), H)
    log(f"HGB post-process: {len(agree)} agreement rows, "
        f"{len(actor_rows)} actor rows, {len(fail_rows)} failures, "
        f"{len(dec)} dyad decision rows")


if __name__ == "__main__":
    main()
