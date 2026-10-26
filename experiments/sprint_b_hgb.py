"""Sprint B.3 - HGB occlusion agreement and dyad-level top-k / decision impact.

The logistic model has an exact additive decomposition.  HGB does not.  This
script asks a narrower, falsifiable question: when the source-actor or
target-actor feature block is replaced by its training mean, does HGB's
out-of-sample delta-NLL rank actors the same way as the logistic attribution?

It also streams the five seed refits over the strict inner window to measure
dyad-level top-k churn and the precision/lift of explanation-based alerts.
CPU only; no test tuning; 2018-22 and COVID remain burned retrospective views.

Outputs:
  cache/hgb_occ_{period}.npz
  tables/hgb_metrics.csv
  tables/hgb_occlusion_agreement.csv
  tables/hgb_occlusion_actor.csv
  tables/hgb_occlusion_failures.csv
  tables/dyad_stability.csv
  tables/topk_churn_dyad.csv
  tables/decision_impact_dyad.csv
  summary_hgb.json
"""

from __future__ import annotations

import itertools
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import sprint_b_common as C
from framework import aggregation as A
from framework import models
from framework.config import load_config
from framework.utils import elapsed, jdump, log

ROLES = ("source", "target")
KS = (5, 10, 20, 50)
N_BOOT = 1000
BOOT_SEED = C.SEED_BASE + 23


def _period_rows(b: dict, p: dict) -> dict:
    src = b["va"] if p["lo"] < 276 else b["te"]
    return C.split_rows(src, p["lo"], p["hi"])


def _cols(names, feats):
    idx = {n: i for i, n in enumerate(names)}
    return [idx[f] for f in feats]


def _topk(scores, k, mode="pos"):
    x = np.asarray(scores, dtype=np.float64)
    ok = np.isfinite(x)
    if ok.sum() < k:
        return set()
    s = np.where(ok, x if mode == "pos" else np.abs(x), -np.inf)
    return set(np.argsort(-s, kind="stable")[:k].tolist())


def _jac(a, b):
    if not a or not b:
        return float("nan")
    return len(a & b) / len(a | b)


def _actor_occ_agreement(attr, logocc, hgbocc, k_list=KS):
    out = {}
    out["n"] = int((np.isfinite(attr) & np.isfinite(hgbocc)).sum())
    out["spearman_attr_hgb"] = A.spearman(attr, hgbocc)
    out["sign_attr_hgb"] = A.sign_agreement(attr, hgbocc)
    out["spearman_logocc_hgb"] = A.spearman(logocc, hgbocc)
    out["sign_logocc_hgb"] = A.sign_agreement(logocc, hgbocc)
    out["spearman_attr_logocc"] = A.spearman(attr, logocc)
    out["sign_attr_logocc"] = A.sign_agreement(attr, logocc)
    for k in k_list:
        out[f"jaccard_pos@{k}"] = A.topk_jaccard(attr, hgbocc, k, "pos")
        out[f"jaccard_abs@{k}"] = A.topk_jaccard(attr, hgbocc, k, "abs")
        out[f"jaccard_logocc_pos@{k}"] = A.topk_jaccard(logocc, hgbocc, k, "pos")
    return out


def _bootstrap_actor(a, b, reps, seed):
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
    if not vals:
        return None, None
    return A.ci(vals)


def _bootstrap_months_flat(a, b, reps, seed):
    """a,b are (actors, months); resample months and correlate flattened cells."""
    A_, M = a.shape
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        take = rng.integers(0, M, size=M)
        aa = a[:, take].ravel()
        bb = b[:, take].ravel()
        v = A.spearman(aa, bb)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return None, None
    return A.ci(vals)


def _dyad_scores(Cmat, did, y, p, n_dyads, cols_s, cols_t):
    n = np.bincount(did, minlength=n_dyads).astype(np.float64)
    tot = np.bincount(did, weights=Cmat.sum(axis=1), minlength=n_dyads)
    src = np.bincount(did, weights=Cmat[:, cols_s].sum(axis=1), minlength=n_dyads)
    tgt = np.bincount(did, weights=Cmat[:, cols_t].sum(axis=1), minlength=n_dyads)
    ysum = np.bincount(did, weights=y, minlength=n_dyads)
    psum = np.bincount(did, weights=p, minlength=n_dyads)
    den = np.maximum(n, 1)
    return {
        "n": n, "y_sum": ysum,
        "risk": np.where(n > 0, psum / den, np.nan),
        "total": np.where(n > 0, tot / den, np.nan),
        "source": np.where(n > 0, src / den, np.nan),
        "target": np.where(n > 0, tgt / den, np.nan),
    }


def main() -> None:
    rl = C.RunLog("sprint_b_hgb")
    t_start = time.time()
    cfg = load_config()
    C.ensure_dirs()
    b = C.load_splits(cfg)
    actors = b["actors"]
    names = b["tr"]["feature_names"]
    n_actors = len(actors)
    n_dyads = b["n_dyads"]
    cols_s = _cols(names, A.SOURCE_OWN)
    cols_t = _cols(names, A.TARGET_OWN)
    log(f"HGB: n_actors={n_actors} n_features={len(names)} "
        f"source_cols={len(cols_s)} target_cols={len(cols_t)}")

    tr, sel = b["tr"], C.select_rows(b["va"])
    w = C.base_weight(tr)
    t0 = time.time()
    hgb = models.fit_hgb(tr["X"], tr["y"], sel["X"], sel["y"], cfg,
                         sample_weight=w, iters=[50, 100, 150])
    log(f"HGB fit: iters={hgb['n_iter']} select_LL={hgb['select_log_loss']:.6f} "
        f"({elapsed(t0)}s) curve={hgb['select_curve']}")
    mu = tr["X"].mean(axis=0).astype(np.float32)
    sd = tr["X"].std(axis=0).astype(np.float32)
    sd = np.where(sd < 1e-8, 1.0, sd).astype(np.float32)
    mu_cols = {"source": mu[cols_s], "target": mu[cols_t]}

    # ------------------------------------------------------------------
    # HGB metrics + occlusion over the three periods
    # ------------------------------------------------------------------
    metrics_rows, occ = [], {}
    for p in C.PERIODS:
        d = _period_rows(b, p)
        n = len(d["y"])
        p_hgb = models.predict_hgb(hgb, d["X"])
        m = C.metrics_block(d["y"], p_hgb)
        metrics_rows.append({"model": "hgb_full", "period": p["name"],
                             "status": p["status"], "n": n,
                             "positives": int(d["y"].sum()),
                             "n_iter": hgb["n_iter"],
                             "select_log_loss": round(float(hgb["select_log_loss"]), 6),
                             **m})
        log(f"HGB metrics {p['name']}: LL={m['log_loss']:.5f} AUC={m['auc']:.4f} "
            f"ECE={m['ece']:.5f}")

        A_, M = n_actors, p["hi"] - p["lo"] + 1
        occ_tot = {r: np.zeros(A_) for r in ROLES}
        occ_mon = {r: np.zeros((A_, M)) for r in ROLES}
        for start in range(0, n, C.CHUNK):
            stop = min(start + C.CHUNK, n)
            sl = slice(start, stop)
            X = d["X"][sl]
            y = d["y"][sl]
            sid = d["sid"][sl].astype(np.int64)
            tid = d["tid"][sl].astype(np.int64)
            mth = d["t_idx"][sl].astype(np.int64) - p["lo"]
            pp = p_hgb[sl]
            base = A.nll(y, pp)
            for role, cols in (("source", cols_s), ("target", cols_t)):
                Xo = X.copy()
                Xo[:, cols] = mu_cols[role]
                po = models.predict_hgb(hgb, Xo)
                dn = A.nll(y, po) - base
                actor = sid if role == "source" else tid
                occ_tot[role] += np.bincount(actor, weights=dn, minlength=A_)
                flat = actor * M + mth
                occ_mon[role] += np.bincount(flat, weights=dn,
                                             minlength=A_ * M).reshape(A_, M)
            if start == 0:
                log(f"HGB occlusion {p['name']}: first chunk done")
        payload = {"meta": np.array(json.dumps({
            "period": p["name"], "lo": p["lo"], "hi": p["hi"],
            "model": "hgb_full", "n_iter": hgb["n_iter"],
            "select_log_loss": float(hgb["select_log_loss"]),
            "occlusion": "replace actor-own feature block with train mean",
        }))}
        for r in ROLES:
            payload[f"occ_total_{r}"] = occ_tot[r]
            payload[f"occ_month_{r}"] = occ_mon[r]
        np.savez_compressed(os.path.join(C.AGG, f"hgb_occ_{p['name']}.npz"), **payload)
        occ[p["name"]] = {"total": occ_tot, "month": occ_mon,
                          "p": p_hgb, "d": d}
        log(f"HGB occlusion {p['name']}: total source range "
            f"[{occ_tot['source'].min():.3f},{occ_tot['source'].max():.3f}]")

    pd.DataFrame(metrics_rows).to_csv(
        os.path.join(C.TAB, "hgb_metrics.csv"), index=False)

    # ------------------------------------------------------------------
    # Logistic vs HGB occlusion agreement
    # ------------------------------------------------------------------
    agree_rows, actor_rows, failure_rows = [], [], []
    for p in C.PERIODS:
        zpoint = np.load(os.path.join(C.AGG, f"agg_point_{p['name']}.npz"),
                         allow_pickle=False)
        for role in ROLES:
            j = 0 if role == "source" else 1
            cnt = zpoint[f"count_{role}"]
            attr = (zpoint[f"bucket_sum_{role}"][:, :, j].sum(axis=1) /
                    np.maximum(cnt.sum(axis=1), 1))
            attr = np.where(cnt.sum(axis=1) > 0, attr, np.nan)
            logocc = zpoint[f"occ_actor_total_{role}"]
            hgbocc = occ[p["name"]]["total"][role]
            rec = {"period": p["name"], "role": role, "level": "actor",
                   **_actor_occ_agreement(attr, logocc, hgbocc)}
            ci = _bootstrap_actor(attr, hgbocc, N_BOOT,
                                  BOOT_SEED + 11 * j + len(p["name"]))
            rec["spearman_ci_lo"], rec["spearman_ci_hi"] = ci
            agree_rows.append(rec)

            # actor-month level, common non-empty cells
            am_attr = np.where(cnt > 0, zpoint[f"bucket_sum_{role}"][:, :, j] /
                               np.maximum(cnt, 1), np.nan)
            am_log = np.where(cnt > 0, zpoint[f"occ_actor_month_{role}"], np.nan)
            am_hgb = np.where(cnt > 0, occ[p["name"]]["month"][role], np.nan)
            a = am_attr.ravel()
            lo = am_log.ravel()
            hg = am_hgb.ravel()
            recm = {"period": p["name"], "role": role, "level": "actor_month",
                    "n": int((np.isfinite(a) & np.isfinite(hg)).sum()),
                    "spearman_attr_hgb": A.spearman(a, hg),
                    "sign_attr_hgb": A.sign_agreement(a, hg),
                    "spearman_logocc_hgb": A.spearman(lo, hg),
                    "sign_logocc_hgb": A.sign_agreement(lo, hg),
                    "spearman_attr_logocc": A.spearman(a, lo)}
            for k in (10, 50, 100):
                recm[f"jaccard_pos@{k}"] = A.topk_jaccard(a, hg, k, "pos")
                recm[f"jaccard_abs@{k}"] = A.topk_jaccard(a, hg, k, "abs")
            ci = _bootstrap_months_flat(am_attr, am_hgb, N_BOOT,
                                        BOOT_SEED + 101 + 7 * j +
                                        len(p["name"]))
            recm["spearman_ci_lo"], recm["spearman_ci_hi"] = ci
            agree_rows.append(recm)

            # per-actor table for failure analysis
            for i in range(n_actors):
                if not np.isfinite(attr[i]):
                    continue
                actor_rows.append({
                    "period": p["name"], "role": role, "aid": i,
                    "actor": actors["actor"].iloc[i] if i < len(actors) else str(i),
                    "region": actors["region"].iloc[i] if i < len(actors) else "",
                    "n_rows": int(cnt.sum(axis=1)[i]),
                    "attribution": round(float(attr[i]), 6),
                    "logistic_occlusion_dnll": round(float(logocc[i]), 6),
                    "hgb_occlusion_dnll": round(float(hgbocc[i]), 6),
                    "hgb_over_logistic": round(float(hgbocc[i] /
                                                     max(abs(logocc[i]), 1e-9)), 4),
                })

            # rank divergence failure cases
            ok = np.isfinite(attr) & np.isfinite(hgbocc)
            ra = rankdata(np.abs(attr[ok]))
            rh = rankdata(np.abs(hgbocc[ok]))
            diff = np.abs(ra - rh)
            order = np.argsort(-diff)[:10]
            idx = np.where(ok)[0]
            for o in order:
                i = int(idx[o])
                failure_rows.append({
                    "period": p["name"], "role": role, "aid": i,
                    "actor": actors["actor"].iloc[i] if i < len(actors) else str(i),
                    "region": actors["region"].iloc[i] if i < len(actors) else "",
                    "n_rows": int(cnt.sum(axis=1)[i]),
                    "attribution": round(float(attr[i]), 6),
                    "logistic_occlusion_dnll": round(float(logocc[i]), 6),
                    "hgb_occlusion_dnll": round(float(hgbocc[i]), 6),
                    "rank_abs_attr": int(ra[o]), "rank_abs_hgb": int(rh[o]),
                    "rank_gap": int(diff[o]),
                    "sign_disagreement": bool(attr[i] * hgbocc[i] < 0),
                    "failure_type": "rank_gap",
                })
    agree = pd.DataFrame(agree_rows)
    agree.to_csv(os.path.join(C.TAB, "hgb_occlusion_agreement.csv"), index=False)
    pd.DataFrame(actor_rows).to_csv(
        os.path.join(C.TAB, "hgb_occlusion_actor.csv"), index=False)
    pd.DataFrame(failure_rows).to_csv(
        os.path.join(C.TAB, "hgb_occlusion_failures.csv"), index=False)

    # ------------------------------------------------------------------
    # Dyad-level top-k churn and decision impact (inner window only)
    # ------------------------------------------------------------------
    inner = C.split_rows(b["va"], 252, 274)
    X_inner = inner["X"]
    did = inner["did"].astype(np.int64)
    y_inner = inner["y"].astype(np.float64)
    n_inner = len(y_inner)
    coefs = np.load(os.path.join(C.AGG, "coefs.npz"))
    cw = coefs["coefs"]
    ci_ = coefs["intercepts"]
    log(f"dyad view: inner rows={n_inner:,} dyads={len(np.unique(did)):,} "
        f"n_dyads={n_dyads}")

    point_cache = np.load(os.path.join(C.CACHE, "contrib_inner_point.npz"),
                          allow_pickle=False)
    assert np.array_equal(point_cache["did"].astype(np.int64), did)
    assert np.array_equal(point_cache["y"].astype(np.int64),
                          y_inner.astype(np.int64))
    dyad = {}
    point_C = point_cache["contribution"].astype(np.float64)
    dyad["point"] = _dyad_scores(point_C, did, y_inner,
                                 point_cache["p"].astype(np.float64),
                                 n_dyads, cols_s, cols_t)
    # sanity: rebuild point C from stored scaler/coefficients
    fit_point = {"mu": mu, "sd": sd, "coef": cw[0], "intercept": float(ci_[0])}
    C_rebuilt = A.standardized(fit_point, X_inner) * cw[0][None, :]
    log(f"dyad point rebuild max |dC|={np.abs(C_rebuilt - point_C).max():.2e}")
    del C_rebuilt
    for s in range(C.N_SEEDS):
        fit_s = {"mu": mu, "sd": sd, "coef": cw[1 + s],
                 "intercept": float(ci_[1 + s])}
        Cs = A.standardized(fit_s, X_inner) * cw[1 + s][None, :]
        eta = float(ci_[1 + s]) + Cs.sum(axis=1)
        ps = A.sigmoid(eta)
        dyad[f"rep{s}"] = _dyad_scores(Cs, did, y_inner, ps, n_dyads,
                                       cols_s, cols_t)
        del Cs
        log(f"dyad seed {s}: source mean={np.nanmean(dyad[f'rep{s}']['source']):.4f}")

    # stability across seeds
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

    # decision impact over unique dyads
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
                   "positives": int(pos),
                   "precision": round(prec, 5),
                   "recall": round(pos / total_pos, 5),
                   "lift_vs_base": round(prec / base, 4) if base else None,
                   "jaccard_with_risk": round(_jac(
                       set(sel.tolist()), _topk(dyad["point"]["risk"], k)), 4)}
            # seed spread for the same alert rule
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

    # ------------------------------------------------------------------
    # Holdout availability (Sprint A integration hook)
    # ------------------------------------------------------------------
    holdout_dir = os.path.join(C.OUT, "..", "holdout")
    pred_dir = os.path.join(holdout_dir, "preds")
    holdout_files = []
    if os.path.isdir(pred_dir):
        holdout_files = sorted(f for f in os.listdir(pred_dir)
                               if f.endswith((".npz", ".csv", ".json")))
    holdout = {
        "available_at_audit": False,
        "completed_later": True,
        "used": False,
        "files": holdout_files,
        "note": ("The one-shot holdout completed after this audit was frozen; "
                 "its predictions were not used and the audit was not re-run. "
                 "All stability numbers remain provisional."),
    }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    hgb_actor = agree[agree.level == "actor"]
    hgb_am = agree[agree.level == "actor_month"]
    summary = {
        "sprint": "B.3",
        "task": "HGB occlusion agreement + dyad top-k / decision impact",
        "status": "provisional_holdout_not_integrated",
        "hgb": {
            "n_iter": int(hgb["n_iter"]),
            "select_log_loss": round(float(hgb["select_log_loss"]), 6),
            "select_curve": hgb["select_curve"],
            "metrics": pd.DataFrame(metrics_rows).to_dict(orient="records"),
            "occlusion": "actor-own feature block replaced by train mean",
        },
        "agreement": {
            "actor": hgb_actor.to_dict(orient="records"),
            "actor_month": hgb_am.to_dict(orient="records"),
            "failures_top": pd.DataFrame(failure_rows).head(40).to_dict(orient="records"),
        },
        "dyad": {
            "n_rows": int(n_inner),
            "n_dyads": int(len(np.unique(did))),
            "stability": pd.DataFrame(dyad_stab).to_dict(orient="records"),
            "churn": pd.DataFrame(dyad_churn).to_dict(orient="records"),
            "decision_impact": dec.to_dict(orient="records"),
        },
        "holdout": holdout,
        "runtime_seconds": elapsed(t_start),
    }
    C.atomic_json(os.path.join(C.OUT, "summary_hgb.json"), summary)
    log(f"summary_hgb.json written; total {elapsed(t_start)}s")
    rl.close()


if __name__ == "__main__":
    main()
