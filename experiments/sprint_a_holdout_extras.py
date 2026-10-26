#!/usr/bin/env python3
"""Post-run diagnostics for the one-shot Sprint-A unit-fresh holdout.

The confirmatory pass is ``experiments/sprint_a_run.py --part holdout``.
This script only *post-processes* the frozen outputs:

  parts/holdout_core.json
  preds/holdout_preds.npz

It re-exports the feature matrices in BLIND mode (test labels forced to 0)
so that no second 2018-22 label read occurs, then restores the holdout labels
from the saved prediction archive.  All extra models (dyad fixed effects,
leave-one-group-out) are fitted on train/select only; eval labels are used
for scoring only.

Outputs
-------
  report.md
  summary.json
  manifest.json
  parts/holdout_extras.json
  tables/*.csv
  preds/holdout_fe_scores.npz

The 2018-22 period is retrospective and unit-fresh, not time-fresh.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework import data, features, models, sprint_a
from framework import sprint_a_inference as inf
from framework.config import load_config
from framework.metrics import binary_metrics, within_entity_auc
from framework.utils import (Tee, host_info, jdump, jload, log, pkg_versions,
                             sha256_file)

BASE = os.path.join("results", "framework_pilot", "holdout")
PREREG = os.path.join(BASE, "preregistration.json")
ACCESS = os.path.join(BASE, "test_access_log.json")
LOCK = os.path.join(BASE, "run_once.json")
CORE = os.path.join(BASE, "parts", "holdout_core.json")
PREDS = os.path.join(BASE, "preds", "holdout_preds.npz")
EXTRAS = os.path.join(BASE, "parts", "holdout_extras.json")
FE_PREDS = os.path.join(BASE, "preds", "holdout_fe_scores.npz")
TABLES = os.path.join(BASE, "tables")
REPORT = os.path.join(BASE, "report.md")
SUMMARY = os.path.join(BASE, "summary.json")
MANIFEST = os.path.join(BASE, "manifest.json")

SEEDS = [20261023, 20261024, 20261025, 20261026, 20261027]
REPS = 1000
CONDLOGIT_CAP = 60          # max positives per training dyad for stable exact FE logit
CONDLOGIT_BOUND = 4.0       # beta box; the conditional likelihood is convex inside
ONESHOT_STATUS = "completed"

PRIMARY = "logistic49_S1_temperature"
PRIMARY_RAW = "logistic49_S1_none"
PRIMARY_ISO = "logistic49_S1_isotonic"
BASE_MARKOV = "base_per_dyad_markov"
BASE_BR = "base_dyad_base_rate"
BASE_RECENCY = "base_markov_recency"
BASE_GLOBAL = "base_global_transition"
BASE_WARM_MARKOV = "base_warm_markov"
BASE_WARM_BR = "base_warm_base_rate"
HGB = "hgb49_none"
LOGIT29 = "logistic29_S1_temperature"
LOGIT49_CLIP = "logistic49_S2_temperature"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _utc() -> str:
    return sprint_a.utc_now()


def _fmt(x, nd: int = 6) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float) and not math.isfinite(x):
        return "NA"
    return f"{float(x):.{nd}f}"


def _ci_str(ci, nd: int = 6) -> str:
    if not ci:
        return "NA"
    return f"[{_fmt(ci[0], nd)}, {_fmt(ci[1], nd)}]"


def _safe_auc(y, p) -> float | None:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return None


def _safe_weighted_auc(y, p, w) -> float | None:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, p, sample_weight=w))
    except ValueError:
        return None


def _norm_p(point: float, se: float) -> float:
    if se is None or not math.isfinite(se) or se <= 0:
        return 0.0 if point != 0 else 1.0
    return float(min(1.0, 2.0 * 0.5 * math.erfc(abs(point) / se / math.sqrt(2.0))))


def _boot_ci(draws, lo=2.5, hi=97.5):
    d = np.asarray(draws, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None, None
    return [float(np.percentile(d, lo)), float(np.percentile(d, hi))], float(np.std(d, ddof=1))


def _cluster_inverse(cid):
    _, inv = np.unique(np.asarray(cid), return_inverse=True)
    return inv.astype(np.int64), int(inv.max()) + 1


def _boot_indices(k: int, reps: int, rng):
    return rng.integers(0, k, size=(int(reps), int(k)))


def _weighted_auc_draws(y, score, cid, reps, seed, extra_weight=None):
    """Dyad-cluster bootstrap of a row-level AUC by row weights.

    Resampling dyads with replacement is equivalent to giving every row the
    multiplicity of its dyad.  ``extra_weight`` can carry an additional
    per-row weight (not used here).
    """
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    inv, k = _cluster_inverse(cid)
    rng = np.random.default_rng(int(seed))
    W = _boot_indices(k, reps, rng)
    # Multiplicity per row for each replicate would be (reps, n); too large.
    # Instead loop in blocks and build weights.
    out = np.empty(int(reps), dtype=np.float64)
    order = np.argsort(inv, kind="stable")
    inv_s = inv[order]
    y_s = y[order]
    s_s = score[order]
    sizes = np.bincount(inv_s, minlength=k)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    for r in range(int(reps)):
        mult = np.bincount(W[r], minlength=k).astype(np.float64)
        row_w = mult[inv_s]
        a = _safe_weighted_auc(y_s, s_s, row_w)
        out[r] = np.nan if a is None else a
    return out


# ---------------------------------------------------------------------------
# Within-dyad / between-dyad AUC
# ---------------------------------------------------------------------------
def per_dyad_auc(y, p, did):
    """Strict per-dyad AUC on dyads with both classes.

    Returns arrays for eligible dyads: did, auc, n_pos*n_neg pair weight,
    n_pos, n_neg, n.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    did = np.asarray(did, dtype=np.int64)
    order = np.argsort(did, kind="stable")
    ys, ps, ds = y[order], p[order], did[order]
    uniq, starts, counts = np.unique(ds, return_index=True, return_counts=True)
    rows = []
    for g, (u, s, n) in enumerate(zip(uniq, starts, counts)):
        yg = ys[s:s + n]
        k = int(yg.sum())
        if k <= 0 or k >= n:
            continue
        pg = ps[s:s + n]
        try:
            a = float(roc_auc_score(yg, pg))
        except ValueError:
            continue
        rows.append((int(u), a, int(k * (n - k)), k, int(n - k), int(n)))
    if not rows:
        return (np.array([], dtype=np.int64), np.array([]), np.array([]),
                np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                np.array([], dtype=np.int64))
    arr = np.asarray(rows, dtype=np.float64)
    return (arr[:, 0].astype(np.int64), arr[:, 1], arr[:, 2],
            arr[:, 3].astype(np.int64), arr[:, 4].astype(np.int64),
            arr[:, 5].astype(np.int64))


def within_summary(y, p, did, reps=REPS, seeds=SEEDS):
    """Strict within-dyad AUC mean and pair-weighted pooled AUC + dyad bootstrap."""
    a_did, auc, w, npos, nneg, n = per_dyad_auc(y, p, did)
    out = {
        "n_dyads_total": int(len(np.unique(did))),
        "n_dyads_eligible": int(len(auc)),
        "n_pairs": int(w.sum()),
        "mean": float(auc.mean()) if len(auc) else None,
        "pooled": float((auc * w).sum() / w.sum()) if len(auc) and w.sum() else None,
        "mean_ci": None, "pooled_ci": None,
        "seeds": [],
    }
    if len(auc) == 0:
        return out
    rng_master = np.random.default_rng(int(seeds[0]))
    for j, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        G = len(auc)
        idx = rng.integers(0, G, size=(int(reps), G))
        aa = auc[idx]
        ww = w[idx]
        m = aa.mean(axis=1)
        pooled = (aa * ww).sum(axis=1) / np.maximum(ww.sum(axis=1), 1e-12)
        mci, mse = _boot_ci(m)
        pci, pse = _boot_ci(pooled)
        out["seeds"].append({"seed": int(seed), "mean_ci": mci, "pooled_ci": pci,
                             "mean_se": mse, "pooled_se": pse})
        if j == 0:
            out["mean_ci"], out["pooled_ci"] = mci, pci
            out["mean_se"], out["pooled_se"] = mse, pse
    out["mean_ci_ll_spread"] = float(max(s["mean_ci"][0] for s in out["seeds"]) -
                                     min(s["mean_ci"][0] for s in out["seeds"]))
    out["mean_ci_ul_spread"] = float(max(s["mean_ci"][1] for s in out["seeds"]) -
                                     min(s["mean_ci"][1] for s in out["seeds"]))
    out["pooled_ci_ll_spread"] = float(max(s["pooled_ci"][0] for s in out["seeds"]) -
                                       min(s["pooled_ci"][0] for s in out["seeds"]))
    out["pooled_ci_ul_spread"] = float(max(s["pooled_ci"][1] for s in out["seeds"]) -
                                       min(s["pooled_ci"][1] for s in out["seeds"]))
    return out


def between_summary(y, p, did, reps=REPS, seeds=SEEDS):
    """Between-dyad AUC (P1-compatible binary, row-weighted, Spearman)."""
    from scipy.stats import spearmanr

    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    did = np.asarray(did, dtype=np.int64)
    _, inv = np.unique(did, return_inverse=True)
    k = int(inv.max()) + 1
    n = np.bincount(inv, minlength=k).astype(np.float64)
    ymean = np.bincount(inv, weights=y, minlength=k) / np.maximum(n, 1)
    pmean = np.bincount(inv, weights=p, minlength=k) / np.maximum(n, 1)
    lab = (ymean > 0.5).astype(np.int64)
    auc_bin = (_safe_auc(lab, pmean)
               if len(np.unique(lab)) > 1 else None)
    p_row = pmean[inv]
    auc_row = _safe_auc(y, p_row)
    rho = None
    try:
        rho = float(spearmanr(pmean, ymean).correlation)
    except Exception:
        pass
    out = {
        "n_dyads": int(k),
        "n_pos_dyads_binary": int(lab.sum()),
        "auc_binary": auc_bin,
        "auc_row_weighted": auc_row,
        "spearman": rho,
        "auc_binary_ci": None,
        "auc_row_weighted_ci": None,
        "spearman_ci": None,
        "seeds": [],
    }
    rng = np.random.default_rng(int(seeds[0]))
    draws_bin, draws_rho = [], []
    for _ in range(int(reps)):
        idx = rng.integers(0, k, size=k)
        lb = lab[idx]
        a = (_safe_auc(lb, pmean[idx]) if len(np.unique(lb)) > 1 else None)
        if a is not None:
            draws_bin.append(a)
        try:
            rr = spearmanr(pmean[idx], ymean[idx]).correlation
            if rr is not None and math.isfinite(float(rr)):
                draws_rho.append(float(rr))
        except Exception:
            pass
    out["auc_binary_ci"], _ = _boot_ci(draws_bin)
    out["spearman_ci"], _ = _boot_ci(draws_rho)
    # Row-weighted AUC is a sensitivity check; 200 dyad-bootstrap reps are
    # enough for its interval and keep the post-processing tractable.
    draws_row = []
    rng_row = np.random.default_rng(int(seeds[0]) + 991)
    for _ in range(int(min(reps, 200))):
        idx = rng_row.integers(0, k, size=k)
        mult = np.bincount(idx, minlength=k).astype(np.float64)
        arow = _safe_weighted_auc(y, p_row, mult[inv])
        if arow is not None:
            draws_row.append(arow)
    out["auc_row_weighted_ci"], _ = _boot_ci(draws_row)
    out["auc_row_weighted_reps"] = int(len(draws_row))
    # 5-seed endpoint spread (dyad-level binary AUC; row-weighted point only).
    for seed in seeds:
        rr = np.random.default_rng(int(seed))
        db = []
        for _ in range(int(min(reps, 300))):
            idx = rr.integers(0, k, size=k)
            lb = lab[idx]
            a = (_safe_auc(lb, pmean[idx]) if len(np.unique(lb)) > 1 else None)
            if a is not None:
                db.append(a)
        cb, _ = _boot_ci(db)
        out["seeds"].append({"seed": int(seed), "auc_binary_ci": cb})
    return out


def paired_within_delta(y, p_base, p_model, did, reps=REPS, seeds=SEEDS):
    """Paired strict within-dyad AUC delta (model - base) with dyad bootstrap."""
    a_did, auc_b, w_b, npos_b, nneg_b, n_b = per_dyad_auc(y, p_base, did)
    b_did, auc_m, w_m, npos_m, nneg_m, n_m = per_dyad_auc(y, p_model, did)
    common = np.intersect1d(a_did, b_did)
    if len(common) == 0:
        return {"n_common": 0}
    ib = {int(d): i for i, d in enumerate(a_did)}
    im = {int(d): i for i, d in enumerate(b_did)}
    ii = np.array([ib[int(d)] for d in common])
    jj = np.array([im[int(d)] for d in common])
    auc_b, auc_m = auc_b[ii], auc_m[jj]
    w = w_b[ii]
    d_mean = auc_m - auc_b
    d_pool = ((auc_m - auc_b) * w).sum() / w.sum()
    out = {"n_common": int(len(common)),
           "base_mean": float(auc_b.mean()), "model_mean": float(auc_m.mean()),
           "d_mean": float(d_mean.mean()),
           "base_pooled": float((auc_b * w).sum() / w.sum()),
           "model_pooled": float((auc_m * w).sum() / w.sum()),
           "d_pooled": float(d_pool),
           "d_mean_ci": None, "d_pooled_ci": None, "p_mean": None,
           "p_pooled": None, "seeds": []}
    rng = np.random.default_rng(int(seeds[0]))
    G = len(common)
    idx = rng.integers(0, G, size=(int(reps), G))
    dm = d_mean[idx].mean(axis=1)
    dp = ((d_mean[idx] * w[idx]).sum(axis=1) /
          np.maximum(w[idx].sum(axis=1), 1e-12))
    out["d_mean_ci"], out["d_mean_se"] = _boot_ci(dm)
    out["d_pooled_ci"], out["d_pooled_se"] = _boot_ci(dp)
    out["p_mean"] = float(min(1.0, 2.0 * min((dm <= 0).mean(), (dm >= 0).mean())))
    out["p_pooled"] = float(min(1.0, 2.0 * min((dp <= 0).mean(), (dp >= 0).mean())))
    for seed in seeds:
        rr = np.random.default_rng(int(seed))
        idx = rr.integers(0, G, size=(int(min(reps, 500)), G))
        dm = d_mean[idx].mean(axis=1)
        dp = ((d_mean[idx] * w[idx]).sum(axis=1) /
              np.maximum(w[idx].sum(axis=1), 1e-12))
        out["seeds"].append({"seed": int(seed),
                             "d_mean_ci": _boot_ci(dm)[0],
                             "d_pooled_ci": _boot_ci(dp)[0]})
    return out


def centered_within_auc(y, p, did):
    """P1-compatible within-entity AUC: AUC(y, p - dyad mean p)."""
    w = within_entity_auc(y, p, did)
    return w.get("within_auc"), w.get("between_auc")


def paired_centered_delta(y, p_base, p_model, did, reps=REPS, seed=SEEDS[0]):
    """Paired P1-compatible within-dyad AUC delta, dyad bootstrap by row weights."""
    y = np.asarray(y, dtype=np.int64)
    did = np.asarray(did, dtype=np.int64)
    _, inv = np.unique(did, return_inverse=True)
    k = int(inv.max()) + 1
    n = np.bincount(inv, minlength=k).astype(np.float64)
    base = np.asarray(p_base, dtype=np.float64)
    model = np.asarray(p_model, dtype=np.float64)
    mb = np.bincount(inv, weights=base, minlength=k) / np.maximum(n, 1)
    mm = np.bincount(inv, weights=model, minlength=k) / np.maximum(n, 1)
    pc_b = base - mb[inv]
    pc_m = model - mm[inv]
    a_b = _safe_auc(y, pc_b)
    a_m = _safe_auc(y, pc_m)
    out = {"base": a_b, "model": a_m, "d": None if a_b is None else a_m - a_b,
           "ci": None, "p": None}
    rng = np.random.default_rng(int(seed))
    draws = []
    for _ in range(int(reps)):
        idx = rng.integers(0, k, size=k)
        mult = np.bincount(idx, minlength=k).astype(np.float64)
        wrow = mult[inv]
        ab = _safe_weighted_auc(y, pc_b, wrow)
        am = _safe_weighted_auc(y, pc_m, wrow)
        if ab is not None and am is not None:
            draws.append(am - ab)
    out["ci"], out["se"] = _boot_ci(draws)
    if draws:
        d = np.asarray(draws)
        out["p"] = float(min(1.0, 2.0 * min((d <= 0).mean(), (d >= 0).mean())))
    return out


# ---------------------------------------------------------------------------
# Exact conditional logit (Chamberlain fixed-effects logit)
# ---------------------------------------------------------------------------
def _group_layout(did_sorted, y_sorted):
    uniq, starts, counts = np.unique(did_sorted, return_index=True,
                                     return_counts=True)
    k = np.add.reduceat(y_sorted, starts).astype(np.int64)
    return uniq, starts.astype(np.int64), counts.astype(np.int64), k


def _condlogit_prepare(X, y, did, cap=CONDLOGIT_CAP):
    order = np.argsort(did, kind="stable")
    Xs, ys, ds = X[order], y[order], did[order]
    uniq, starts, counts, k = _group_layout(ds, ys)
    keep = (k > 0) & (k < counts) & (k <= int(cap))
    row_keep = np.zeros(len(ys), dtype=bool)
    for g in np.where(keep)[0]:
        row_keep[starts[g]:starts[g] + counts[g]] = True
    Xk, yk, dk = Xs[row_keep], ys[row_keep], ds[row_keep]
    del Xs
    o = np.argsort(dk, kind="stable")
    Xk, yk, dk = Xk[o], yk[o], dk[o]
    uniq2, starts2, counts2, k2 = _group_layout(dk, yk)
    mu = Xk.mean(axis=0).astype(np.float64)
    sd = Xk.std(axis=0).astype(np.float64)
    sd = np.where(sd < 1e-10, 1.0, sd)
    Xz = ((Xk - mu) / sd).astype(np.float64)
    del Xk
    info = {
        "n_groups_total": int(len(uniq)),
        "n_groups_used": int(len(uniq2)),
        "n_rows_total": int(len(y)),
        "n_rows_used": int(len(yk)),
        "cap": int(cap),
        "k_min": int(k2.min()) if len(k2) else None,
        "k_max": int(k2.max()) if len(k2) else None,
        "k_median": float(np.median(k2)) if len(k2) else None,
    }
    return Xz, yk.astype(np.int64), starts2, counts2, k2, mu, sd, info


def _condlogit_nll_grad(beta, X, y, starts, counts, K):
    G = len(starts)
    logits = X @ beta
    gmax = np.maximum.reduceat(logits, starts)
    z = np.exp(logits - np.repeat(gmax, counts))
    max_n = int(counts.max())
    gid = np.repeat(np.arange(G), counts)
    pos = np.arange(len(y)) - np.repeat(starts, counts)
    zp = np.zeros((G, max_n), dtype=np.float64)
    zp[gid, pos] = z
    k = np.add.reduceat(y, starts).astype(np.int64)
    e = np.zeros((G, K + 1), dtype=np.float64)
    e[:, 0] = 1.0
    for t in range(max_n):
        zt = zp[:, t]
        for j in range(min(K, t + 1), 0, -1):
            e[:, j] += e[:, j - 1] * zt
    Z = e[np.arange(G), k]
    ar = np.arange(G)
    w = np.zeros(len(y), dtype=np.float64)
    for t in range(max_n):
        zt = zp[:, t]
        em = np.empty((G, K), dtype=np.float64)
        em[:, 0] = 1.0
        for j in range(1, K):
            em[:, j] = e[:, j] - zt * em[:, j - 1]
        idx = np.maximum(k - 1, 0)
        wt = zt * em[ar, idx] / np.maximum(Z, 1e-300)
        wt = np.clip(wt, 0.0, 1.0)
        mask = counts > t
        rows = starts[mask] + t
        w[rows] = wt[mask]
    nll = (-np.sum(y * logits) + np.sum(np.log(np.maximum(Z, 1e-300))) +
           np.sum(k * gmax))
    grad = X.T @ (w - y)
    return float(nll), grad


def fit_conditional_logit(X, y, did, cap=CONDLOGIT_CAP, bound=CONDLOGIT_BOUND,
                          maxiter=120):
    from scipy.optimize import minimize
    Xz, yk, starts, counts, k2, mu, sd, info = _condlogit_prepare(
        np.asarray(X), np.asarray(y, dtype=np.int64),
        np.asarray(did, dtype=np.int64), cap=cap)
    K = int(k2.max()) if len(k2) else 1
    b0 = np.zeros(Xz.shape[1], dtype=np.float64)
    res = minimize(lambda b: _condlogit_nll_grad(b, Xz, yk, starts, counts, K),
                   b0, jac=True, method="L-BFGS-B",
                   bounds=[(-float(bound), float(bound))] * Xz.shape[1],
                   options={"maxiter": int(maxiter), "ftol": 1e-11,
                            "gtol": 1e-6, "maxls": 50})
    info.update({"success": bool(res.success), "message": str(res.message),
                 "nit": int(res.nit), "nll": float(res.fun),
                 "grad_norm": float(np.linalg.norm(res.jac))
                 if getattr(res, "jac", None) is not None else None,
                 "beta": res.x.astype(np.float64), "mu": mu, "sd": sd})
    return info


def within_ols_fe(X, y, did, ridge=1e-6, chunk=500000):
    """Dyad fixed-effects linear probability model (within OLS)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    did = np.asarray(did, dtype=np.int64)
    _, inv = np.unique(did, return_inverse=True)
    k = int(inv.max()) + 1
    n = np.bincount(inv, minlength=k).astype(np.float64)
    ksum = np.bincount(inv, weights=y, minlength=k)
    ybar = ksum / np.maximum(n, 1)
    d = X.shape[1]
    XtX = np.zeros((d, d), dtype=np.float64)
    Xty = np.zeros(d, dtype=np.float64)
    for s in range(0, len(y), int(chunk)):
        e = min(len(y), s + int(chunk))
        Xi = X[s:e]
        yi = y[s:e]
        Xm = np.empty_like(Xi)
        for j in range(d):
            Xm[:, j] = np.bincount(inv[s:e], weights=Xi[:, j],
                                   minlength=k)[inv[s:e]] / np.maximum(n[inv[s:e]], 1)
        Xc = Xi - Xm
        yc = yi - ybar[inv[s:e]]
        XtX += Xc.T @ Xc
        Xty += Xc.T @ yc
    XtX.flat[::d + 1] += float(ridge)
    beta = np.linalg.solve(XtX, Xty)
    return beta


# ---------------------------------------------------------------------------
# Bootstrap for LOO per-row deltas
# ---------------------------------------------------------------------------
def boot_mean_delta(delta_rows, did, reps=REPS, seed=SEEDS[0]):
    delta_rows = np.asarray(delta_rows, dtype=np.float64)
    inv, k = _cluster_inverse(did)
    sums = np.bincount(inv, weights=delta_rows, minlength=k)
    ns = np.bincount(inv, minlength=k).astype(np.float64)
    rng = np.random.default_rng(int(seed))
    out = np.empty(int(reps), dtype=np.float64)
    for r in range(int(reps)):
        c = rng.integers(0, k, size=k)
        out[r] = sums[c].sum() / max(ns[c].sum(), 1.0)
    ci, se = _boot_ci(out)
    point = float(delta_rows.mean())
    p = float(min(1.0, 2.0 * min((out <= 0).mean(), (out >= 0).mean())))
    return {"delta": point, "ci": ci, "se": se, "p": p,
            "delta_normal_p": _norm_p(point, se), "n_clusters": int(k)}


# ---------------------------------------------------------------------------
# Load / verify
# ---------------------------------------------------------------------------
def load_inputs():
    for p in (PREREG, CORE, PREDS, LOCK):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    core = jload(CORE)
    prereg = jload(PREREG)
    run_once = jload(LOCK)
    if core["holdout"].get("blind") or core["holdout"].get("quick"):
        raise AssertionError("core is a blind/quick run, not the confirmatory pass")
    if run_once.get("status") != ONESHOT_STATUS:
        raise AssertionError(f"run_once status is {run_once.get('status')!r}")
    if run_once.get("blind") or run_once.get("quick"):
        raise AssertionError("run_once is blind/quick")
    file_hash = sha256_file(PREREG)
    frozen = open(PREREG + ".sha256", encoding="utf-8").read().split()[0]
    if file_hash != frozen:
        raise AssertionError("preregistration file hash changed")
    sprint_a.verify_preregistration(PREREG)
    if core["holdout"]["definition_sha256"] != prereg["hashes"]["definition_sha256"]:
        raise AssertionError("definition hash mismatch between core and prereg")
    if not core.get("assertions", {}).get("pass"):
        raise AssertionError(f"holdout assertions failed: {core.get('assertions')}")
    z = np.load(PREDS, allow_pickle=False)
    y = z["y"].astype(np.int64)
    did = z["did"].astype(np.int64)
    sid = z["sid"].astype(np.int64)
    tid = z["tid"].astype(np.int64)
    t_idx = z["t_idx"].astype(np.int64)
    preds = {k[2:]: z[k].astype(np.float64) for k in z.files if k.startswith("p_")}
    n = len(y)
    for k, v in preds.items():
        if len(v) != n:
            raise AssertionError(f"prediction length mismatch for {k}")
    if n != int(core["holdout"]["rows"]):
        raise AssertionError(f"rows {n} != core {core['holdout']['rows']}")
    if int(y.sum()) != int(core["holdout"]["positives"]):
        raise AssertionError("positive count mismatch")
    return core, prereg, run_once, y, did, sid, tid, t_idx, preds, file_hash


def build_blind_bundle(core, prereg):
    """Re-export features with test labels forced to 0; restore y from preds."""
    cfg = load_config()
    spec = {"dids": np.asarray(prereg["definition"]["dids"], dtype=np.int64),
            "counts": prereg["holdout"]["counts"],
            "definition": prereg["definition"]}
    b = sprint_a.build_holdout_bundle(cfg, spec, force_graph=False,
                                      label_mode="blind")
    return cfg, b


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
METRIC_KEYS = ("log_loss", "brier", "ece", "ece_equal_frequency", "auc", "ap",
               "accuracy", "macro_f1", "intercept", "slope")


def metric_rows(y, did, preds, keys, boot_keys=()):
    rows = []
    for key in keys:
        if key not in preds:
            continue
        p = preds[key]
        m = sprint_a.metrics_block(y, p)
        w = within_entity_auc(y, p, did)
        ws = within_summary(y, p, did, reps=200, seeds=SEEDS[:1]) if key in boot_keys else {}
        bs = between_summary(y, p, did, reps=200, seeds=SEEDS[:1]) if key in boot_keys else {}
        row = {"model": key, "n": m["n"], "positives": m["positives"],
               "pos_rate": m["pos_rate"], "log_loss": m["log_loss"],
               "brier": m["brier"], "ece": m["ece"],
               "ece_equal_frequency": m.get("ece_equal_frequency"),
               "auc": m["auc"], "ap": m["ap"],
               "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
               "calibration_intercept": m.get("intercept"),
               "calibration_slope": m.get("slope"),
               "within_auc_p1": w.get("within_auc"),
               "between_auc_p1": w.get("between_auc"),
               "within_auc_strict_mean": ws.get("mean"),
               "within_auc_strict_pooled": ws.get("pooled"),
               "within_eligible_dyads": ws.get("n_dyads_eligible"),
               "between_auc_row_weighted": bs.get("auc_row_weighted"),
               "between_spearman": bs.get("spearman")}
        rows.append(row)
    return rows


def write_csv(name, rows):
    os.makedirs(TABLES, exist_ok=True)
    df = pd.DataFrame(rows)
    path = os.path.join(TABLES, name)
    df.to_csv(path, index=False)
    return path


def primary_comparison_rows(core):
    rows = []
    for name, c in core["comparisons"].items():
        pt = c["point"]
        mw = c["multiway"]
        rows.append({
            "comparison": name,
            "base_log_loss": pt["base_log_loss"],
            "model_log_loss": pt["model_log_loss"],
            "d_log_loss": pt["d_log_loss"],
            "rel_gain": pt.get("rel_log_loss_gain"),
            "base_ece": pt["base_ece"], "model_ece": pt["model_ece"],
            "d_ece": pt["d_ece"],
            "base_brier": pt["base_brier"], "model_brier": pt["model_brier"],
            "d_brier": pt["d_brier"],
            "dyad_ci_ll": c["dims"]["dyad"]["ci_ll"],
            "source_ci_ll": c["dims"]["source"]["ci_ll"],
            "target_ci_ll": c["dims"]["target"]["ci_ll"],
            "multiway_ci_ll": mw["ci_ll"],
            "multiway_p_ll": mw.get("p_ll"),
            "multiway_ci_ece": mw.get("ci_ece"),
            "multiway_p_ece": mw.get("p_ece"),
            "multiway_ci_brier": mw.get("ci_brier"),
            "multiway_p_brier": mw.get("p_brier"),
        })
    return rows


def auc_level_rows(core):
    rows = []
    for key, a in core.get("auc_levels", {}).items():
        rows.append({"model": key, "auc": a["point"]["auc"], "ap": a["point"]["ap"],
                     "auc_dyad_ci": a["dims"]["dyad"]["ci_auc"],
                     "auc_source_ci": a["dims"]["source"]["ci_auc"],
                     "auc_target_ci": a["dims"]["target"]["ci_auc"],
                     "auc_multiway_ci": a.get("auc_multiway", {}).get("ci"),
                     "ap_dyad_ci": a["dims"]["dyad"]["ci_ap"],
                     "ap_source_ci": a["dims"]["source"]["ci_ap"],
                     "ap_target_ci": a["dims"]["target"]["ci_ap"],
                     "ap_multiway_ci": a.get("ap_multiway", {}).get("ci")})
    return rows


def auc_delta_rows(core):
    rows = []
    for name, a in core.get("auc_deltas", {}).items():
        rows.append({"comparison": name,
                     "d_auc": a["point"]["d_auc"],
                     "d_ap": a["point"]["d_ap"],
                     "d_auc_dyad_ci": a["dims"]["dyad"]["ci_auc"],
                     "d_auc_source_ci": a["dims"]["source"]["ci_auc"],
                     "d_auc_target_ci": a["dims"]["target"]["ci_auc"],
                     "d_auc_multiway_ci": a.get("multiway_auc", {}).get("ci"),
                     "d_auc_multiway_p": a.get("multiway_auc", {}).get("p"),
                     "d_ap_dyad_ci": a["dims"]["dyad"]["ci_ap"],
                     "d_ap_source_ci": a["dims"]["source"]["ci_ap"],
                     "d_ap_target_ci": a["dims"]["target"]["ci_ap"],
                     "d_ap_multiway_ci": a.get("multiway_ap", {}).get("ci"),
                     "d_ap_multiway_p": a.get("multiway_ap", {}).get("p")})
    return rows


def bh_rows(core, loo):
    fams = {
        "A_primary_ll": [
            ("logistic49_temp vs per_dyad_markov",
             core["comparisons"]["logistic49_temperature__vs__per_dyad_markov"]["multiway"].get("p_ll")),
            ("logistic49_temp vs dyad_base_rate",
             core["comparisons"]["logistic49_temperature__vs__dyad_base_rate"]["multiway"].get("p_ll")),
            ("logistic49_temp vs markov_recency",
             core["comparisons"]["logistic49_temperature__vs__markov_recency"]["multiway"].get("p_ll")),
            ("logistic49_temp vs global_transition",
             core["comparisons"]["logistic49_temperature__vs__global_transition"]["multiway"].get("p_ll")),
        ],
        "B_primary_other_metrics": [
            ("d_ece vs per_dyad_markov",
             core["comparisons"]["logistic49_temperature__vs__per_dyad_markov"]["multiway"].get("p_ece")),
            ("d_auc vs per_dyad_markov",
             core["auc_deltas"]["logistic49_temperature__vs__per_dyad_markov"].get("multiway_auc", {}).get("p")),
            ("d_ap vs per_dyad_markov",
             core["auc_deltas"]["logistic49_temperature__vs__per_dyad_markov"].get("multiway_ap", {}).get("p")),
        ],
        "C_capacity_feature": [
            ("hgb49 vs per_dyad_markov",
             core["comparisons"]["hgb49__vs__per_dyad_markov"]["multiway"].get("p_ll")),
            ("logistic29_temp vs per_dyad_markov",
             core["comparisons"]["logistic29_temperature__vs__per_dyad_markov"]["multiway"].get("p_ll")),
            ("hgb49 vs logistic49_temperature",
             core["comparisons"]["hgb49__vs__logistic49_temperature"]["multiway"].get("p_ll")),
        ],
        "D_group_attribution": [(r["group"], r["p"]) for r in loo],
    }
    rows = []
    for fam, items in fams.items():
        names = [x[0] for x in items]
        pvals = [float(x[1]) if x[1] is not None and math.isfinite(float(x[1])) else 1.0
                 for x in items]
        res = inf.bh_fdr(pvals, alpha=0.05)
        for name, p, q, rej in zip(names, res["p"], res["q"], res["reject"]):
            rows.append({"family": fam, "comparison": name, "p": p, "q": q,
                         "reject_0.05": bool(rej)})
    return rows


def anchor_rows(core):
    ab = core["feature_group_ablation"]
    rows = []
    for name, m in ab["anchors"].items():
        rows.append({"anchor": name, "log_loss": m["log_loss"], "brier": m["brier"],
                     "ece": m["ece"], "auc": m["auc"], "ap": m["ap"]})
    return rows


def permutation_rows(core):
    return [dict(r) for r in core["feature_group_ablation"]["grouped_permutation"]]


def calibration_rows(core):
    rows = []
    for spec, c in core.get("calibration", {}).items():
        rows.append({"spec": spec, "temperature_T": c.get("T"),
                     "select_ll_none": c.get("select_ll_none"),
                     "select_ll_temperature": c.get("select_ll_temperature"),
                     "select_ll_isotonic": c.get("select_ll_isotonic")})
    return rows


def logistic_rows(core):
    return [{"spec": k, "C": v, "select_log_loss": core["logistic_select_ll"].get(k)}
            for k, v in core.get("logistic_C", {}).items()]


def hgb_rows(core):
    rows = []
    for r in core.get("hgb_grid", []):
        rows.append(dict(r))
    if core.get("hgb_params"):
        rows.append({"selected": True, **core["hgb_params"]})
    return rows


def mde_rows(core):
    rows = []
    for k, v in core.get("mde", {}).items():
        rows.append({"contrast": k, **v})
    return rows


# ---------------------------------------------------------------------------
# FE check
# ---------------------------------------------------------------------------
def run_fe_check(cfg, b, y, did, preds):
    t_start = time.time()
    names = list(b["train"]["feature_names"])
    Xtr = b["train"]["X"]
    # Drop columns with no within-dyad variation on train (conditional
    # likelihood does not depend on them; this improves conditioning).
    did_tr = b["train"]["did"].astype(np.int64)
    _, inv = np.unique(did_tr, return_inverse=True)
    k = int(inv.max()) + 1
    n = np.bincount(inv, minlength=k).astype(np.float64)
    keep_cols = []
    for j in range(Xtr.shape[1]):
        # within-dyad variance via E[x^2]-E[x]^2
        m1 = np.bincount(inv, weights=Xtr[:, j].astype(np.float64),
                         minlength=k) / np.maximum(n, 1)
        m2 = np.bincount(inv, weights=Xtr[:, j].astype(np.float64) ** 2,
                         minlength=k) / np.maximum(n, 1)
        if np.nanmax(m2 - m1 ** 2) > 1e-12:
            keep_cols.append(j)
    keep_names = [names[j] for j in keep_cols]
    Xtr_k = Xtr[:, keep_cols]
    Xev_k = b["eval"]["X"][:, keep_cols]
    log(f"fe: kept {len(keep_cols)}/{len(names)} within-varying features")
    t0 = time.time()
    fe = fit_conditional_logit(Xtr_k, b["train"]["y"], did_tr)
    log(f"fe: conditional logit success={fe['success']} nit={fe['nit']} "
        f"n_groups={fe['n_groups_used']} ({time.time()-t0:.1f}s)")
    beta = fe["beta"]
    Xev_z = (np.asarray(Xev_k, dtype=np.float64) - fe["mu"]) / fe["sd"]
    score = Xev_z @ beta
    w_p1 = within_entity_auc(y, score, did)
    ws = within_summary(y, score, did, reps=REPS, seeds=SEEDS)
    # Within OLS sensitivity.
    t0 = time.time()
    beta_ols = within_ols_fe(np.asarray(Xtr_k, dtype=np.float64),
                             np.asarray(b["train"]["y"], dtype=np.float64),
                             did_tr)
    log(f"fe: within OLS done ({time.time()-t0:.1f}s)")
    score_ols = np.asarray(Xev_k, dtype=np.float64) @ beta_ols
    w_ols = within_entity_auc(y, score_ols, did)
    ws_ols = within_summary(y, score_ols, did, reps=REPS, seeds=SEEDS)
    # Primary / baseline rows for comparison.
    comp = {}
    for key in (PRIMARY, BASE_MARKOV, HGB):
        if key in preds:
            comp[key] = {"p1": within_entity_auc(y, preds[key], did),
                         "strict": within_summary(y, preds[key], did,
                                                  reps=REPS, seeds=SEEDS)}
    out = {"features_used": keep_names, "n_features_used": len(keep_names),
           "conditional_logit": {k: v for k, v in fe.items()
                                 if k not in ("beta", "mu", "sd")},
           "conditional_logit_within": {"p1": w_p1, "strict": ws},
           "within_ols": {"within": {"p1": w_ols, "strict": ws_ols}},
           "comparators": comp,
           "runtime_seconds": round(time.time() - t_start, 2)}
    return out, score, score_ols


# ---------------------------------------------------------------------------
# LOO
# ---------------------------------------------------------------------------
def run_loo(cfg, b, y, did, preds, core):
    names = list(b["train"]["feature_names"])
    idx = {n: i for i, n in enumerate(names)}
    groups = dict(features.DYAD_GROUPS)
    groups["graph"] = [n for n in features.GRAPH_FEATURES if n in idx]
    order = sprint_a.GROUP_ORDER
    full_C = float(core.get("logistic_C", {}).get("logistic49_S1", 1.0))
    p_full = preds[PRIMARY_RAW]
    full_ll = binary_metrics(y, p_full)["log_loss"]
    ytr = b["train"]["y"]
    ysel = b["select"]["y"]
    core_loo = {r["group"]: r for r in core["feature_group_ablation"]["loo"]}
    rows = []
    for gname in order:
        drop = set(groups.get(gname, []))
        keep = [f for f in names if f not in drop]
        ci = [idx[f] for f in keep]
        log(f"loo: fitting without {gname} ({len(keep)} features)")
        fit = models.fit_logistic(b["train"]["X"][:, ci], ytr,
                                  b["select"]["X"][:, ci], ysel, cfg,
                                  C_grid=[full_C], feature_names=keep)
        p = models.predict_logistic(fit, b["eval"]["X"][:, ci])
        ll = binary_metrics(y, p)["log_loss"]
        drow = (-(y * np.log(np.clip(p, 1e-12, 1)) +
                  (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))) -
                (-(y * np.log(np.clip(p_full, 1e-12, 1)) +
                   (1 - y) * np.log(np.clip(1 - p_full, 1e-12, 1)))))
        bt = boot_mean_delta(drow, did, reps=REPS, seed=SEEDS[0])
        rows.append({"group": gname, "n_features": len(keep),
                     "C": full_C, "log_loss_without": ll,
                     "delta_vs_full": ll - full_ll,
                     "delta_core": core_loo.get(gname, {}).get("delta_vs_full"),
                     "delta_boot": bt["delta"], "ci": bt["ci"],
                     "se": bt["se"], "p": bt["p"],
                     "p_normal": bt["delta_normal_p"],
                     "n_clusters": bt["n_clusters"]})
        log(f"loo: {gname} dLL={ll - full_ll:+.6f} ci={bt['ci']} p={bt['p']:.4f}")
    return rows, full_ll, full_C


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def build_summary(core, prereg, run_once, y, did, preds, within, between,
                  within_p1, within_deltas, fe, loo, bh, file_hash, t_idx):
    m = core["metrics"]
    comp = core["comparisons"]
    primary = comp["logistic49_temperature__vs__per_dyad_markov"]
    warm = comp.get("logistic49_temperature__vs__warm_markov")
    warm29 = comp.get("logistic29_temperature__vs__warm_markov")
    pt = primary["point"]
    mw = primary["multiway"]
    strict_primary = within[PRIMARY]
    p1_primary = within_p1[PRIMARY]
    between_primary = between[PRIMARY]
    strict_markov = within[BASE_MARKOV]
    p1_markov = within_p1[BASE_MARKOV]
    strict_hgb = within[HGB]
    p1_hgb = within_p1[HGB]
    fe_within = fe["conditional_logit_within"]
    sr_f7 = bool(mw["ci_ll"][0] <= 0.0 <= mw["ci_ll"][1])
    rel_gain = float(pt.get("rel_log_loss_gain", 0.0))
    gain_lt_2pct = bool(rel_gain < 0.02)
    hgb_ll = float(m[HGB]["log_loss"])
    logit_ll = float(m[PRIMARY]["log_loss"])
    hgb_rel = (logit_ll - hgb_ll) / max(logit_ll, 1e-12)
    hgb_beats_logistic_gt_0_5pct = bool(hgb_rel > 0.005)
    seed_ll = [s.get("multiway", {}).get("ci_ll") for s in core.get("seed_spread", [])]
    seed_ll = [x for x in seed_ll if x]
    seed_points = [s.get("point", {}).get("d_log_loss")
                   for s in core.get("seed_spread", [])]
    seed_points = [float(x) for x in seed_points
                   if x is not None and math.isfinite(float(x))]
    seed_point_spread = (max(seed_points) - min(seed_points)
                         if seed_points else None)
    seed_ll_spread = (max(x[0] for x in seed_ll) - min(x[0] for x in seed_ll)
                      if seed_ll else None)
    seed_ul_spread = (max(x[1] for x in seed_ll) - min(x[1] for x in seed_ll)
                      if seed_ll else None)
    seed_range = (max(x[1] for x in seed_ll) - min(x[0] for x in seed_ll)
                  if seed_ll else None)
    seed_spread_gt_0_5pct = bool(seed_point_spread is not None and
                                 seed_point_spread > 0.005)
    q_primary = next((r["q"] for r in bh
                      if r["family"] == "A_primary_ll" and
                      r["comparison"] == "logistic49_temp vs per_dyad_markov"),
                     None)
    q_logistic29 = next((r["q"] for r in bh
                         if r["family"] == "C_capacity_feature" and
                         r["comparison"] == "logistic29_temp vs per_dyad_markov"),
                        None)
    q_hgb_vs_logit = next((r["q"] for r in bh
                           if r["family"] == "C_capacity_feature" and
                           r["comparison"] == "hgb49 vs logistic49_temperature"),
                          None)
    auc_pd = core["auc_deltas"].get(
        "logistic49_temperature__vs__per_dyad_markov", {})
    auc_warm = core["auc_deltas"].get(
        "logistic49_temperature__vs__warm_markov", {})
    ece_pd_ci = mw.get("ci_ece")
    ece_pd_worse = bool(ece_pd_ci and ece_pd_ci[1] < 0)
    ll_pd_pos = bool(mw["ci_ll"][0] > 0)
    ll_pd_neg = bool(mw["ci_ll"][1] < 0)
    warm_ll = warm["point"]["d_log_loss"] if warm else None
    warm_ci = warm["multiway"]["ci_ll"] if warm else None
    warm_ece = warm["point"]["d_ece"] if warm else None
    warm_ece_ci = warm["multiway"]["ci_ece"] if warm else None
    warm_ll_worse = bool(warm_ci and warm_ci[1] < 0)
    warm_ll_better = bool(warm_ci and warm_ci[0] > 0)
    warm_ece_worse = bool(warm_ece_ci and warm_ece_ci[1] < 0)
    auc_pd_ci = auc_pd.get("multiway_auc", {}).get("ci")
    ap_pd_ci = auc_pd.get("multiway_ap", {}).get("ci")
    auc_warm_ci = auc_warm.get("multiway_auc", {}).get("ci")
    ap_warm_ci = auc_warm.get("multiway_ap", {}).get("ci")
    auc_ap_pd_better = bool(auc_pd_ci and ap_pd_ci and
                            auc_pd_ci[0] > 0 and ap_pd_ci[0] > 0)
    auc_ap_warm_better = bool(auc_warm_ci and ap_warm_ci and
                              auc_warm_ci[0] > 0 and ap_warm_ci[0] > 0)
    year_fe = core.get("feature_group_ablation", {}).get("anchors", {}).get(
        "calendar_year_fe_29", {})
    within_null = (
        (strict_primary.get("mean") is None or strict_primary["mean"] < 0.55) and
        (p1_primary.get("within_auc") is None or p1_primary["within_auc"] < 0.55) and
        (fe_within["strict"].get("mean") is None or fe_within["strict"]["mean"] < 0.55)
    )
    ece_pd_value = pt.get("d_ece")
    verdict = {
        "multiway_ci_includes_zero": sr_f7,
        "ci_excludes_zero": not sr_f7,
        "bh_q_primary": q_primary,
        "bh_q_logistic29": q_logistic29,
        "bh_q_hgb_vs_logistic": q_hgb_vs_logit,
        "bh_reject_primary": bool(q_primary is not None and q_primary < 0.05),
        "gain_rel": rel_gain,
        "gain_lt_2pct": gain_lt_2pct,
        "hgb_rel_improvement": hgb_rel,
        "hgb_beats_logistic_gt_0_5pct": hgb_beats_logistic_gt_0_5pct,
        "seed_ll_ci_ll_spread": seed_ll_spread,
        "seed_ll_ci_ul_spread": seed_ul_spread,
        "seed_ll_ci_range": seed_range,
        "seed_ll_ci_endpoint_spread": seed_range,
        "seed_point_spread": seed_point_spread,
        "seed_spread_gt_0_5pct": seed_spread_gt_0_5pct,
        "ll_beats_per_dyad_markov": bool(ll_pd_pos and q_primary is not None
                                          and q_primary < 0.05),
        "ll_worse_than_per_dyad_markov": ll_pd_neg,
        "ece_delta_vs_per_dyad_markov": ece_pd_value,
        "ece_worse_than_per_dyad_markov": ece_pd_worse,
        "auc_ap_beats_per_dyad_markov": auc_ap_pd_better,
        "beats_per_dyad_markov": bool((not sr_f7) and
                                      (q_primary is not None and q_primary < 0.05) and
                                      rel_gain > 0),
        "ll_delta_vs_warm_markov": warm_ll,
        "ll_ci_vs_warm_markov": warm_ci,
        "ll_worse_than_warm_markov": warm_ll_worse,
        "ll_better_than_warm_markov": warm_ll_better,
        "ece_delta_vs_warm_markov": warm_ece,
        "ece_worse_than_warm_markov": warm_ece_worse,
        "auc_ap_beats_warm_markov": auc_ap_warm_better,
        "logistic29_vs_warm_markov": warm29,
        "year_fe_anchor": year_fe,
        "within_dyad_timing": (
            "null" if within_null
            else "definition_dependent_no_robust_advantage"),
        "claim_level": (
            "dyad-level ranking/propensity, not month-level timing; "
            "calibration is not supported on 2018-22"
            if (within_null and ece_pd_worse)
            else ("dyad-level propensity/calibration, not month-level timing"
                  if within_null
                  else "mixed: ranking gain, no log-loss/calibration gain, "
                       "no robust within-dyad advantage")),
    }
    return {
        "schema": "q2-sprint-a-holdout-summary/1",
        "generated_utc": _utc(),
        "status": "confirmatory_full_holdout_completed",
        "holdout": {
            "rows": int(len(y)), "positives": int(y.sum()),
            "prevalence": float(y.mean()),
            "dyads": int(len(np.unique(did))),
            "period": "2018-01..2022-11",
            "unit_fresh": True, "time_fresh": False,
            "blind": False, "quick": False,
            "definition_sha256": prereg["hashes"]["definition_sha256"],
            "prereg_file_sha256": file_hash,
        },
        "primary": {
            "model": PRIMARY,
            "metrics": m[PRIMARY],
            "vs_per_dyad_markov": primary,
            "vs_warm_markov": comp.get("logistic49_temperature__vs__warm_markov"),
            "vs_warm_base_rate": comp.get("logistic49_temperature__vs__warm_base_rate"),
            "within_auc_p1": p1_primary.get("within_auc"),
            "within_auc_strict": strict_primary,
            "between_auc_p1": between_primary.get("auc_binary"),
            "between_auc_row_weighted": between_primary.get("auc_row_weighted"),
            "between_spearman": between_primary.get("spearman"),
        },
        "baselines": {k: m[k] for k in (BASE_MARKOV, BASE_BR, BASE_RECENCY,
                                        BASE_GLOBAL, BASE_WARM_MARKOV,
                                        BASE_WARM_BR) if k in m},
        "capacity_control": {k: m[k] for k in (HGB, "hgb49_temperature",
                                               "hgb49_isotonic") if k in m},
        "within_between": {
            "primary": strict_primary,
            "per_dyad_markov": strict_markov,
            "hgb49": strict_hgb,
            "p1": {"primary": p1_primary, "per_dyad_markov": p1_markov,
                   "hgb49": p1_hgb},
            "between_primary": between_primary,
            "between_per_dyad_markov": between[BASE_MARKOV],
            "between_hgb49": between[HGB],
            "paired": within_deltas,
        },
        "dyad_fe": {
            "conditional_logit": fe["conditional_logit"],
            "within": fe["conditional_logit_within"],
            "within_ols": fe["within_ols"],
            "comparators": fe["comparators"],
            "verdict": (
                "no within-dyad timing improvement" if within_null
                else "conditional_logit_did_not_converge; within-dyad LPM "
                     "descriptive only; no robust advantage"),
        },
        "feature_group_loo": loo,
        "bh": bh,
        "seed_spread": core.get("seed_spread"),
        "mde": core.get("mde"),
        "verdict": verdict,
        "paper_sentences": paper_sentences(core, p1_primary, strict_primary,
                                           p1_markov, fe_within, verdict,
                                           q_primary),
        "artifacts": {
            "core": CORE, "preds": PREDS, "extras": EXTRAS,
            "report": REPORT, "manifest": MANIFEST,
        },
    }


def _num(x, nd=4):
    return "NA" if x is None else f"{float(x):.{nd}f}"


def paper_sentences(core, p1_primary, strict_primary, p1_markov, fe_within,
                    verdict, q_primary):
    m = core["metrics"]
    c = core["comparisons"]["logistic49_temperature__vs__per_dyad_markov"]
    warm = core["comparisons"].get("logistic49_temperature__vs__warm_markov")
    warm29 = core["comparisons"].get("logistic29_temperature__vs__warm_markov")
    pt, mw = c["point"], c["multiway"]
    ll = m[PRIMARY]["log_loss"]
    brier = m[PRIMARY]["brier"]
    ece = m[PRIMARY]["ece"]
    auc = m[PRIMARY]["auc"]
    ap = m[PRIMARY]["ap"]
    dll = pt["d_log_loss"]
    rel = pt["rel_log_loss_gain"] * 100.0
    lo, hi = mw["ci_ll"]
    wp1 = p1_primary.get("within_auc")
    wp = strict_primary.get("mean")
    wpool = strict_primary.get("pooled")
    wm = p1_markov.get("within_auc")
    wf = fe_within["strict"].get("mean")
    wf_p1 = fe_within["p1"].get("within_auc")
    qtxt = "NA" if q_primary is None else f"{q_primary:.4g}"
    q29 = "NA" if verdict.get("bh_q_logistic29") is None else f"{verdict['bh_q_logistic29']:.4g}"
    s = []
    s.append(
        "In a pre-registered unit-fresh 20% unordered-dyad holdout "
        f"({core['holdout']['rows']:,} dyad-months; {core['holdout']['positives']:,} "
        f"positives; {core['holdout']['directed_dyads']:,} directed dyads; "
        "2018-01-–2022-11), the frozen temperature-calibrated 49-feature "
        "logistic model achieved "
        f"log-loss {ll:.5f}, Brier {brier:.5f}, ECE {ece:.5f}, AUC {auc:.4f}, "
        f"and AP {ap:.4f}.")
    s.append(
        f"Against the train-only per-dyad Markov baseline (log-loss "
        f"{pt['base_log_loss']:.5f}, which equals the global transition on "
        f"unit-fresh dyads), the primary model reduced log-loss by {dll:.5f} "
        f"({rel:.2f}% relative; 95% multiway CI [{lo:.5f}, {hi:.5f}]; "
        f"BH q={qtxt}), increased AUC by "
        f"{core['auc_deltas']['logistic49_temperature__vs__per_dyad_markov']['point']['d_auc']:.5f} "
        f"and AP by "
        f"{core['auc_deltas']['logistic49_temperature__vs__per_dyad_markov']['point']['d_ap']:.5f}, "
        f"but its ECE was worse by {abs(pt['d_ece']):.5f} "
        f"(95% multiway CI [{mw['ci_ece'][0]:.5f}, {mw['ci_ece'][1]:.5f}]).")
    if warm:
        wpt, wmw = warm["point"], warm["multiway"]
        wlo, whi = wmw["ci_ll"]
        wr = 100.0 * wpt["rel_log_loss_gain"]
        s.append(
            f"Against the stronger warm-history Markov baseline (log-loss "
            f"{wpt['base_log_loss']:.5f}, ECE {wpt['base_ece']:.5f}), the primary "
            f"model's log-loss was worse by {abs(wpt['d_log_loss']):.5f} "
            f"({abs(wr):.2f}% relative; 95% multiway CI [{wlo:.5f}, {whi:.5f}]) "
            f"and its ECE was worse by {abs(wpt['d_ece']):.5f} "
            f"(95% multiway CI [{wmw['ci_ece'][0]:.5f}, {wmw['ci_ece'][1]:.5f}]), "
            f"although its AUC was higher by "
            f"{core['auc_deltas']['logistic49_temperature__vs__warm_markov']['point']['d_auc']:.5f} "
            f"(95% CI "
            f"[{core['auc_deltas']['logistic49_temperature__vs__warm_markov']['multiway_auc']['ci'][0]:.5f}, "
            f"{core['auc_deltas']['logistic49_temperature__vs__warm_markov']['multiway_auc']['ci'][1]:.5f}]) "
            f"and AP by "
            f"{core['auc_deltas']['logistic49_temperature__vs__warm_markov']['point']['d_ap']:.5f}. "
            "The primary model therefore improves ranking but not log-loss or "
            "calibration over the best history baseline.")
    if warm29:
        p29 = m[LOGIT29]
        w29pt, w29mw = warm29["point"], warm29["multiway"]
        s.append(
            f"The pre-registered secondary 29-feature logistic model "
            f"(log-loss {p29['log_loss']:.5f}, ECE {p29['ece']:.5f}, "
            f"AUC {p29['auc']:.4f}, AP {p29['ap']:.4f}) beat the warm-history "
            f"Markov baseline on log-loss by {w29pt['d_log_loss']:.5f} "
            f"(95% multiway CI [{w29mw['ci_ll'][0]:.5f}, {w29mw['ci_ll'][1]:.5f}]; "
            f"BH q={q29}) but not on ECE "
            f"({w29pt['d_ece']:+.5f}; 95% CI "
            f"[{w29mw['ci_ece'][0]:.5f}, {w29mw['ci_ece'][1]:.5f}]); the 20 "
            "graph features in the primary do not help on this holdout.")
    yfe = verdict.get("year_fe_anchor") or {}
    if yfe:
        s.append(
            f"A 29-feature year-fixed-effects anchor achieved log-loss "
            f"{yfe['log_loss']:.5f}, ECE {yfe['ece']:.5f}, AUC {yfe['auc']:.4f}, "
            f"and AP {yfe['ap']:.4f}, better than the primary on log-loss; "
            "the calendar anchor is descriptive and not a prospective "
            "specification; we cannot cleanly separate calendar from graph "
            "effects.")
    within_txt = ("The model ranks dyads, not months within a dyad."
                  if verdict.get("within_dyad_timing") == "null"
                  else "The fixed-effects check is definition-dependent; there "
                       "is no robust within-dyad advantage, and any month-level "
                       "timing claim is exploratory.")
    s.append(
        f"Within-dyad AUC was {_num(wp1)} (P1-compatible) and {_num(wpool)} "
        f"(strict pair-weighted; mean {_num(wp)}) for the primary model, versus "
        f"{_num(wm)} (P1-compatible) for per-dyad Markov; the dyad "
        f"fixed-effects/conditional-logit check gave {_num(wf)} strict "
        f"(P1-compatible {_num(wf_p1)}). " + within_txt)
    if verdict.get("within_dyad_timing") == "null" and verdict.get("ece_worse_than_per_dyad_markov"):
        claim_txt = ("The confirmatory claim is therefore limited to dyad-level "
                     "ranking/propensity; we do not claim month-level timing, "
                     "calibrated risk on 2018-22, node-level prediction, or "
                     "causality.")
    elif verdict.get("within_dyad_timing") == "null":
        claim_txt = ("The confirmatory claim is therefore limited to dyad-level "
                     "event propensity and calibrated risk; we make no month-level "
                     "timing claim, no node-level prediction claim, and no "
                     "causal/lifecycle claim.")
    else:
        claim_txt = ("The confirmatory claim is limited to dyad-level event "
                     "propensity/ranking; the within-dyad result is "
                     "definition-dependent and exploratory, and we make no "
                     "month-level timing, node-level, causal or lifecycle "
                     "claim.")
    s.append(claim_txt)
    s.append(
        "Because the holdout is unit-fresh rather than time-fresh and the "
        "2018-22 period was retrospectively burned by earlier runs, these "
        "results are retrospective and should not be presented as a prospective "
        "forecast.")
    return s


def build_report(core, prereg, run_once, summary, rows, within_rows, between_rows,
                 within_delta_rows, fe, loo, bh, anchors, perms, calib, hgb,
                 logit, mde, file_hash, t_idx):
    m = core["metrics"]
    c = core["comparisons"]["logistic49_temperature__vs__per_dyad_markov"]
    pt, mw = c["point"], c["multiway"]
    v = summary["verdict"]
    lines = []
    w = lines.append
    w("# Sprint-A confirmatory unit-fresh dyad holdout")
    w("")
    w(f"Generated: {_utc()}  ")
    w("Status: **one-shot full holdout completed** (non-blind, non-quick).  ")
    w(f"Pre-registration file SHA-256: `{file_hash}`  ")
    w(f"Definition SHA-256: `{prereg['hashes']['definition_sha256']}`  ")
    w("Period: 2018-01–2022-11 (retrospective; unit-fresh, not time-fresh).")
    w("")
    w("## 1. Protocol and one-shot integrity")
    w("")
    w(f"- Holdout: {core['holdout']['rows']:,} dyad-months, "
      f"{core['holdout']['positives']:,} positives "
      f"({100*core['holdout']['positives']/core['holdout']['rows']:.2f}%), "
      f"{core['holdout']['directed_dyads']:,} directed dyads "
      f"({core['holdout']['unordered_pairs']:,} unordered pairs).")
    w("- Embargo rows dropped at t_idx = 239, 251, 275, 335; train < 239, "
      "select 240–250, eval 276–334.")
    w("- The holdout labels were read once, by `experiments/sprint_a_run.py "
      "--part holdout`; this report only post-processes the saved predictions.")
    w("- No 2018–22 labels were used for fitting, model selection, calibration, "
      "or hyper-parameter tuning.")
    w("- Fixed-effects and leave-one-group-out models below are fitted on "
      "train/select only; eval labels are used for scoring only.")
    w("")
    w("## 2. Headline metrics (full holdout)")
    w("")
    w("| model | LL | Brier | ECE | AUC | AP | within-dyad AUC (P1) | within-dyad AUC (strict) | between-dyad AUC (P1) |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["model"] not in (PRIMARY, PRIMARY_RAW, PRIMARY_ISO, LOGIT29,
                              LOGIT49_CLIP, HGB, BASE_MARKOV, BASE_BR,
                              BASE_RECENCY, BASE_GLOBAL, BASE_WARM_MARKOV,
                              BASE_WARM_BR):
            continue
        w(f"| `{r['model']}` | {_fmt(r['log_loss'])} | {_fmt(r['brier'])} | "
          f"{_fmt(r['ece'])} | {_fmt(r['auc'],4)} | {_fmt(r['ap'],4)} | "
          f"{_fmt(r['within_auc_p1'],4)} | {_fmt(r['within_auc_strict_pooled'],4)} | "
          f"{_fmt(r['between_auc_p1'],4)} |")
    w("")
    w(f"Primary vs per-dyad Markov: ΔLL = **{pt['d_log_loss']:+.6f}** "
      f"({100*pt['rel_log_loss_gain']:+.2f}% relative), 95% multiway CI "
      f"{_ci_str(mw['ci_ll'])}; ΔECE = **{pt['d_ece']:+.6f}** "
      f"(95% multiway CI {_ci_str(mw['ci_ece'])}). "
      f"Dyad-only CI {_ci_str(c['dims']['dyad']['ci_ll'])}; "
      f"source CI {_ci_str(c['dims']['source']['ci_ll'])}; "
      f"target CI {_ci_str(c['dims']['target']['ci_ll'])}.")
    w("")
    w("**Baseline caveat.** The train-only per-dyad Markov baseline collapses "
      "to the global transition for unit-fresh dyads (identical LL 0.265874); "
      "the warm-history Markov baseline (pre-2018 observed history only) is the "
      "fair strong comparator and is reported as a secondary row below. Against "
      "the warm-history baseline the primary model **loses on LL and ECE** but "
      "**wins on AUC/AP**; see Section 3 and the exact sentences in Section 9.")
    w("")
    w("## 3. Primary comparisons and multiplicity")
    w("")
    w("| comparison | ΔLL | rel. gain | multiway 95% CI | p (multiway) | BH q |")
    w("|---|---:|---:|---|---:|---:|")
    qmap = {(r["family"], r["comparison"]): r["q"] for r in bh}
    for name in ("logistic49_temperature__vs__per_dyad_markov",
                 "logistic49_temperature__vs__dyad_base_rate",
                 "logistic49_temperature__vs__markov_recency",
                 "logistic49_temperature__vs__global_transition",
                 "logistic49_temperature__vs__warm_markov",
                 "logistic49_temperature__vs__warm_base_rate",
                 "logistic29_temperature__vs__per_dyad_markov",
                 "logistic29_temperature__vs__warm_markov",
                 "hgb49__vs__per_dyad_markov",
                 "hgb49__vs__logistic49_temperature"):
        cc = core["comparisons"][name]
        pp = cc["point"]
        mm = cc["multiway"]
        fam = ("A_primary_ll" if "logistic49_temperature__vs__" in name and
               name.split("__vs__")[1] in ("per_dyad_markov", "dyad_base_rate",
                                            "markov_recency", "global_transition")
               else "C_capacity_feature")
        short = name.replace("logistic49_temperature__vs__", "logistic49_temp vs ")
        short = short.replace("logistic29_temperature__vs__", "logistic29_temp vs ")
        short = short.replace("hgb49__vs__", "hgb49 vs ")
        q = qmap.get((fam, short))
        qtxt = ("NA (secondary)" if (q is None and "warm_" in name)
                else ("NA" if q is None else f"{q:.4g}"))
        w(f"| `{name}` | {pp['d_log_loss']:.6f} | "
          f"{100*pp['rel_log_loss_gain']:.2f}% | {_ci_str(mm['ci_ll'])} | "
          f"{mm['p_ll']:.4g} | {qtxt} |")
    w("")
    w("BH families (pre-registered): A primary LL (4), B primary ECE/AUC/AP (3), "
      "C capacity/feature (3), D group attribution (9 LOO). See "
      "`tables/bh_families.csv`.")
    w("")
    w("Full dyad, source-actor, target-actor, and multiway intervals for ECE, "
      "Brier, AUC, and AP are in `tables/primary_comparisons.csv` and "
      "`tables/auc_ap_deltas.csv`.")
    w("")
    w("## 4. Within- vs between-dyad discrimination (key verdict)")
    w("")
    w(f"- Primary within-dyad AUC (P1-compatible centered score): "
      f"**{_fmt(summary['primary']['within_auc_p1'],4)}**.")
    w(f"- Primary strict per-dyad mean AUC: "
      f"**{_fmt(summary['within_between']['primary']['mean'],4)}**; "
      f"pair-weighted: **{_fmt(summary['within_between']['primary']['pooled'],4)}** "
      f"over {summary['within_between']['primary']['n_dyads_eligible']:,} dyads "
      f"with both classes.")
    w(f"- Per-dyad Markov: within-dyad AUC "
      f"{_fmt(summary['within_between']['per_dyad_markov']['mean'],4)} "
      f"(pooled {_fmt(summary['within_between']['per_dyad_markov']['pooled'],4)}; "
      f"P1-compatible {_fmt(summary['within_between']['p1']['per_dyad_markov'].get('within_auc'),4)}).")
    w(f"- HGB49: within-dyad AUC "
      f"{_fmt(summary['within_between']['hgb49']['mean'],4)} "
      f"(P1-compatible {_fmt(summary['within_between']['p1']['hgb49'].get('within_auc'),4)}).")
    pd = summary["within_between"]["paired"].get(f"{PRIMARY}__vs__{BASE_MARKOV}", {})
    w(f"- Paired strict within-dyad ΔAUC (primary − Markov): "
      f"{_fmt(pd.get('d_pooled'),4)} "
      f"(95% dyad bootstrap CI {_ci_str(pd.get('d_pooled_ci'))}).")
    w(f"- P1-compatible centered within-dyad ΔAUC: "
      f"{_fmt(pd.get('centered', {}).get('d'),4)} "
      f"(95% CI {_ci_str(pd.get('centered', {}).get('ci'))}).")
    w("")
    w("Within/between intervals use the dyad cluster bootstrap because the "
      "estimand is dyad-level. The primary LL/ECE/Brier/AUC/AP comparisons in "
      "Section 3 report dyad, source-actor, target-actor, and multiway intervals.")
    w("")
    if summary["dyad_fe"]["verdict"].startswith("no"):
        w("**Interpretation.** Both definitions of within-dyad AUC are near chance "
          "and the fixed-effects check does not raise them. The global AUC is "
          "therefore a between-dyad propensity signal, not month-level timing.")
    else:
        w("**Interpretation.** The fixed-effects check shows some within-dyad "
          "signal; the global AUC cannot be interpreted as pure between-dyad "
          "propensity, and any timing claim must be qualified as exploratory.")
    w("")
    w("## 5. Dyad fixed-effects / conditional logit")
    w("")
    fec = fe["conditional_logit"]
    w(f"Exact Chamberlain conditional logit on {fec['n_groups_used']:,} of "
      f"{fec['n_groups_total']:,} training dyads (dyads with 1–{fec['cap']} "
      f"positives; k median {fec['k_median']:.0f}); {fec['n_rows_used']:,} rows; "
      f"{fe['n_features_used']} within-varying features. "
      f"Optimizer success={fec['success']}, nit={fec['nit']}.")
    w("")
    w("| model | within-dyad AUC (P1) | strict mean | strict pooled |")
    w("|---|---:|---:|---:|")
    for label, d in (("primary logistic49", fe["comparators"][PRIMARY]),
                     ("per-dyad Markov", fe["comparators"][BASE_MARKOV]),
                     ("HGB49", fe["comparators"][HGB])):
        w(f"| `{label}` | {_fmt(d['p1'].get('within_auc'),4)} | "
          f"{_fmt(d['strict'].get('mean'),4)} | {_fmt(d['strict'].get('pooled'),4)} |")
    w(f"| dyad FE conditional logit | {_fmt(fe['conditional_logit_within']['p1'].get('within_auc'),4)} | "
      f"{_fmt(fe['conditional_logit_within']['strict'].get('mean'),4)} | "
      f"{_fmt(fe['conditional_logit_within']['strict'].get('pooled'),4)} |")
    w(f"| within-dyad OLS (LPM) | {_fmt(fe['within_ols']['within']['p1'].get('within_auc'),4)} | "
      f"{_fmt(fe['within_ols']['within']['strict'].get('mean'),4)} | "
      f"{_fmt(fe['within_ols']['within']['strict'].get('pooled'),4)} |")
    w("")
    w("**Verdict.** " + ("The conditional-logit check does not improve within-dyad "
      "discrimination; the claim is dyad-level propensity/calibration, not "
      "month-level timing." if summary["dyad_fe"]["verdict"].startswith("no")
      else "The conditional-logit check shows some within-dyad signal; the timing "
      "claim must be qualified and reported as exploratory."))
    w("")
    w("## 6. Feature-group anchors and attribution")
    w("")
    w("| anchor | LL | Brier | ECE | AUC | AP |")
    w("|---|---:|---:|---:|---:|---:|")
    for r in anchors:
        w(f"| `{r['anchor']}` | {_fmt(r['log_loss'])} | {_fmt(r['brier'])} | "
          f"{_fmt(r['ece'])} | {_fmt(r['auc'],4)} | {_fmt(r['ap'],4)} |")
    w("")
    w("| group | ΔLL without group | 95% dyad bootstrap CI | p |")
    w("|---|---:|---|---:|")
    for r in loo:
        w(f"| `{r['group']}` | {r['delta_vs_full']:+.6f} | {_ci_str(r['ci'])} | {r['p']:.4g} |")
    w("")
    w("**Grouped permutation** (all columns of a group permuted jointly; "
      "10 permutations; dyad bootstrap CI):")
    w("")
    w("| group | ΔLL permutation | 95% CI | SE |")
    w("|---|---:|---|---:|")
    for r in perms:
        w(f"| `{r['group']}` | {r['delta_log_loss']:+.6f} | "
          f"{_ci_str(r.get('delta_ci'))} | {_fmt(r.get('se'))} |")
    w("")
    w("Positive ΔLL means the group carries held-out signal. Full table: "
      "`tables/feature_group_permutation.csv`.")
    w("")
    w("## 7. Calibration, capacity, and sensitivity")
    w("")
    w("| sensitivity | LL | ECE | AUC | AP | note |")
    w("|---|---:|---:|---:|---:|---|")
    for key, note in ((PRIMARY_RAW, "raw log-odds"),
                      (PRIMARY, "temperature (primary)"),
                      (PRIMARY_ISO, "isotonic (sensitivity only)"),
                      (LOGIT29, "29-feature logistic, temperature"),
                      (LOGIT49_CLIP, "S2 clipped calendar"),
                      (HGB, "HGB49 capacity control"),
                      ("hgb49_temperature", "HGB49 temperature")):
        if key in m:
            w(f"| `{key}` | {_fmt(m[key]['log_loss'])} | {_fmt(m[key]['ece'])} | "
              f"{_fmt(m[key]['auc'],4)} | {_fmt(m[key]['ap'],4)} | {note} |")
    w("")
    w(f"HGB tuning parity: 49 features, 2×3 learning-rate × leaves grid plus "
      f"max-iter refinement; selected params `{core.get('hgb_params')}`. "
      f"Relative HGB vs logistic LL improvement = {100*v['hgb_rel_improvement']:.3f}%.")
    w("")
    w("## 8. MDE and stopping rules")
    w("")
    for k, mm in core.get("mde", {}).items():
        rel = mm.get("mde_paired_rel")
        w(f"- `{k}`: paired MDE {_fmt(mm.get('mde_paired'))} "
          f"({_num(100.0 * rel if rel is not None else None, 2)}% of base LL); "
          f"baseline-variance MDE {_fmt(mm.get('mde_baseline'))}.")
    w("")
    w(f"- **SR-F7** (multiway CI includes 0): "
      f"**{'YES — framework/audit claim only' if v['multiway_ci_includes_zero'] else 'NO'}**.")
    w(f"- **Gain < 2% relative**: **{'YES — exploratory' if v['gain_lt_2pct'] else 'NO'}** "
      f"({100*v['gain_rel']:.2f}%).")
    w(f"- **HGB beats logistic > 0.5%**: "
      f"**{'YES — drop no-capacity-headroom language' if v['hgb_beats_logistic_gt_0_5pct'] else 'NO'}** "
      f"({100*v['hgb_rel_improvement']:.3f}%).")
    w(f"- **Seed spread > 0.5%**: "
      f"**{'YES — drop seed-stability claim' if v['seed_spread_gt_0_5pct'] else 'NO'}** "
      f"(point-estimate spread {_fmt(v.get('seed_point_spread'))}; "
      f"CI endpoint range {_fmt(v['seed_ll_ci_endpoint_spread'])}).")
    w("The five seeds vary only the bootstrap weights; the fitted models are "
      "deterministic, so the point estimates are identical. The CI endpoint "
      "range is reported for completeness, but it is not a model-seed spread.")
    w("")
    w("## 9. Exact sentences the paper can use")
    w("")
    for s in summary["paper_sentences"]:
        w(f"> {s}")
        w("")
    w("## 10. Scope and limitations")
    w("")
    w("- Binary dyadic event prediction only; no node-level prediction.")
    w("- Unit-fresh 20% unordered-dyad holdout, not a time-fresh forecast; "
      "2018–22 was retrospectively burned by earlier G2/D2/B runs.")
    w("- The primary model is a transparent logistic model; HGB is a capacity "
      "control, not the interpretable model.")
    w("- The exact additive source/target actor explanations and stability "
      "audit are reported in the companion Sprint-B artifacts; this report "
      "covers prediction, calibration, and the within-dyad timing check.")
    w("")
    w("## 11. Artifacts")
    w("")
    w("- `parts/holdout_core.json` — one-shot core outputs.")
    w("- `preds/holdout_preds.npz` — row-level predictions and labels.")
    w("- `preds/holdout_fe_scores.npz` — dyad FE scores.")
    w("- `tables/*.csv` — metrics, comparisons, AUC/AP, BH, within/between, "
      "FE, LOO, anchors, permutation, calibration, HGB, seeds, MDE.")
    w("- `summary.json`, `manifest.json`, `run.log`, `test_access_log.json`.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    global REPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--skip-fe", action="store_true")
    ap.add_argument("--skip-loo", action="store_true")
    args = ap.parse_args()
    REPS = int(args.reps)
    os.makedirs(TABLES, exist_ok=True)
    log_path = os.path.join(BASE, "extras.log")
    tee = Tee(log_path)
    sys.stdout = tee
    sys.stderr = tee
    t0 = time.time()
    try:
        log(f"extras: host={host_info()}")
        log(f"extras: packages={pkg_versions()}")
        core, prereg, run_once, y, did, sid, tid, t_idx, preds, file_hash = load_inputs()
        log(f"extras: loaded rows={len(y):,} positives={int(y.sum()):,} "
            f"preds={len(preds)}")
        cfg, b = build_blind_bundle(core, prereg)
        # The blind re-export should have the same row order as the saved
        # predictions.  If it does not, align both by (t_idx, did) so that
        # labels and features stay row-matched.
        kb = np.stack([b["eval"]["t_idx"], b["eval"]["did"]], axis=1)
        kp = np.stack([t_idx, did], axis=1)
        if not np.array_equal(kb, kp):
            log("extras: eval order differs; aligning by (t_idx, did)")
            ob = np.lexsort((kb[:, 1], kb[:, 0]))
            op = np.lexsort((kp[:, 1], kp[:, 0]))
            b["eval"] = sprint_a.slice_dict(b["eval"], ob)
            y, did, sid, tid, t_idx = (y[op], did[op], sid[op],
                                       tid[op], t_idx[op])
            preds = {k: v[op] for k, v in preds.items()}
        b["eval"]["y"] = y.copy()
        log("extras: feature bundle re-exported in blind mode; labels restored "
            "from saved preds")

        # Primary metric rows.
        keys_all = list(preds)
        boot_keys = {PRIMARY, PRIMARY_RAW, PRIMARY_ISO, LOGIT29, LOGIT49_CLIP,
                     HGB, BASE_MARKOV, BASE_BR, BASE_RECENCY, BASE_GLOBAL,
                     BASE_WARM_MARKOV, BASE_WARM_BR}
        rows = metric_rows(y, did, preds, keys_all)
        rows_boot = metric_rows(y, did, preds,
                                [k for k in keys_all if k in boot_keys],
                                boot_keys=boot_keys)
        # Replace rows for boot keys with the richer versions.
        by_model = {r["model"]: r for r in rows}
        for r in rows_boot:
            by_model[r["model"]] = r
        rows = [by_model[k] for k in keys_all if k in by_model]
        write_csv("holdout_metrics.csv", rows)
        write_csv("holdout_metrics_main.csv",
                  [r for r in rows if r["model"] in boot_keys])
        log("extras: metrics table")

        # Core comparison tables.
        write_csv("primary_comparisons.csv", primary_comparison_rows(core))
        write_csv("auc_ap_levels.csv", auc_level_rows(core))
        write_csv("auc_ap_deltas.csv", auc_delta_rows(core))
        write_csv("calibration.csv", calibration_rows(core))
        write_csv("hgb_grid.csv", hgb_rows(core))
        write_csv("logistic_selection.csv", logistic_rows(core))
        write_csv("mde.csv", mde_rows(core))
        write_csv("seed_spread.csv", core.get("seed_spread", []))
        log("extras: core tables")

        # Within / between summary for key models.
        within = {}
        between = {}
        within_p1 = {}
        key_models = {PRIMARY, PRIMARY_RAW, PRIMARY_ISO, LOGIT29, HGB,
                      BASE_MARKOV}
        for key in [PRIMARY, PRIMARY_RAW, PRIMARY_ISO, LOGIT29, LOGIT49_CLIP,
                    HGB, BASE_MARKOV, BASE_BR, BASE_RECENCY, BASE_GLOBAL,
                    BASE_WARM_MARKOV, BASE_WARM_BR]:
            if key not in preds:
                continue
            within[key] = within_summary(y, preds[key], did)
            within_p1[key] = within_entity_auc(y, preds[key], did)
            if key in key_models:
                between[key] = between_summary(y, preds[key], did)
            else:
                between[key] = between_summary(y, preds[key], did,
                                               reps=min(REPS, 300),
                                               seeds=SEEDS[:1])
        within_rows = []
        for key in within:
            within_rows.append({"model": key, **within[key],
                                "p1_within_auc": within_p1[key].get("within_auc"),
                                "p1_between_auc": within_p1[key].get("between_auc"),
                                "between_auc_row_weighted": between[key].get("auc_row_weighted"),
                                "between_spearman": between[key].get("spearman")})
        write_csv("within_between_auc.csv", within_rows)

        # Paired within/between deltas.
        within_deltas = {}
        for base in (BASE_MARKOV, BASE_GLOBAL, BASE_BR, BASE_RECENCY, HGB):
            if base not in preds:
                continue
            d = paired_within_delta(y, preds[base], preds[PRIMARY], did)
            d["centered"] = paired_centered_delta(
                y, preds[base], preds[PRIMARY], did,
                reps=REPS if base == BASE_MARKOV else min(REPS, 500))
            within_deltas[f"{PRIMARY}__vs__{base}"] = d
        write_csv("within_between_deltas.csv",
                  [{"comparison": k, **v} for k, v in within_deltas.items()])
        log("extras: within/between AUC")

        # Dyad FE.
        if args.skip_fe:
            fe = {"conditional_logit": {},
                  "conditional_logit_within": {"p1": {}, "strict": {}},
                  "within_ols": {"within": {"p1": {}, "strict": {}}},
                  "comparators": {}}
            fe_score = fe_score_ols = None
        else:
            fe, fe_score, fe_score_ols = run_fe_check(cfg, b, y, did, preds)
            write_csv("dyad_fe.csv", [{
                "model": "conditional_logit",
                "p1_within_auc": fe["conditional_logit_within"]["p1"].get("within_auc"),
                "strict_mean": fe["conditional_logit_within"]["strict"].get("mean"),
                "strict_pooled": fe["conditional_logit_within"]["strict"].get("pooled"),
                "n_eligible": fe["conditional_logit_within"]["strict"].get("n_dyads_eligible"),
            }, {
                "model": "within_ols_lpm",
                "p1_within_auc": fe["within_ols"]["within"]["p1"].get("within_auc"),
                "strict_mean": fe["within_ols"]["within"]["strict"].get("mean"),
                "strict_pooled": fe["within_ols"]["within"]["strict"].get("pooled"),
                "n_eligible": fe["within_ols"]["within"]["strict"].get("n_dyads_eligible"),
            }])
            np.savez_compressed(FE_PREDS, y=y, did=did, t_idx=t_idx,
                                score_condlogit=fe_score.astype(np.float32),
                                score_within_ols=fe_score_ols.astype(np.float32))
            log("extras: dyad FE")

        # LOO.
        if args.skip_loo:
            loo, full_ll, full_C = [], None, None
        else:
            loo, full_ll, full_C = run_loo(cfg, b, y, did, preds, core)
            write_csv("feature_group_loo.csv", loo)
            write_csv("feature_group_permutation.csv",
                      permutation_rows(core))
            log("extras: LOO")
        write_csv("feature_group_anchors.csv", anchor_rows(core))
        write_csv("feature_group_cumulative.csv",
                  core["feature_group_ablation"].get("cumulative", []))

        bh = bh_rows(core, loo)
        write_csv("bh_families.csv", bh)

        summary = build_summary(core, prereg, run_once, y, did, preds,
                                within, between, within_p1, within_deltas,
                                fe, loo, bh, file_hash, t_idx)
        jdump(SUMMARY, summary)
        jdump(EXTRAS, {"summary": summary, "within": within, "between": between,
                       "within_deltas": within_deltas, "fe": fe, "loo": loo,
                       "bh": bh})
        report = build_report(core, prereg, run_once, summary, rows, within_rows,
                              between, within_deltas, fe, loo, bh, anchor_rows(core),
                              permutation_rows(core), calibration_rows(core),
                              hgb_rows(core), logistic_rows(core), mde_rows(core),
                              file_hash, t_idx)
        with open(REPORT, "w", encoding="utf-8") as fh:
            fh.write(report)
        log("extras: report/summary written")

        # Access log: one post-process entry, no new label read.
        doc = jload(ACCESS) if os.path.exists(ACCESS) else {"entries": []}
        entries = [e for e in doc.get("entries", [])
                   if e.get("kind") != "holdout_postprocess"]
        entries.append({
            "utc": _utc(),
            "kind": "holdout_postprocess",
            "labels_read": False,
            "what": ("post-hoc diagnostics (within/between AUC, dyad FE, LOO CIs) "
                     "from the saved one-shot predictions; test features "
                     "re-exported in blind mode (y=0) and labels restored from "
                     "preds/holdout_preds.npz; no new 2018-22 label read"),
            "inputs": [PREDS, CORE],
            "note": "post-processing only; the confirmatory read is the preceding holdout_eval entry",
        })
        doc["entries"] = entries
        jdump(ACCESS, doc)

        # Manifest.
        outputs = [REPORT, SUMMARY, EXTRAS, PREDS, LOCK, ACCESS,
                   os.path.join(BASE, "run.log"),
                   os.path.join(BASE, "run_blind_quick_aborted.log")]
        outputs += [FE_PREDS] if os.path.exists(FE_PREDS) else []
        outputs += sorted(os.path.join(TABLES, f) for f in os.listdir(TABLES)
                          if f.endswith(".csv"))
        outputs = [p for p in outputs if os.path.exists(p)]
        manifest = {
            "schema": "q2-sprint-a-holdout-manifest/1",
            "created_utc": _utc(),
            "status": "completed",
            "command": ("python3 experiments/sprint_a_run.py --part holdout; "
                        "python3 experiments/sprint_a_holdout_extras.py"),
            "host": host_info(),
            "packages": pkg_versions(),
            "prereg": {
                "path": PREREG,
                "file_sha256": file_hash,
                "definition_sha256": prereg["hashes"]["definition_sha256"],
                "pairs_sha256": prereg["hashes"]["pairs_sha256"],
                "dids_sha256": prereg["hashes"]["dids_sha256"],
                "unchanged": True,
            },
            "run_once": run_once,
            "holdout": core["holdout"],
            "counts": core.get("counts"),
            "assertions": core.get("assertions"),
            "seeds": SEEDS,
            "reps": REPS,
            "inputs": {p: sha256_file(p) for p in
                       (PREREG, CORE, PREDS, LOCK, ACCESS) if os.path.exists(p)},
            "code": {p: sha256_file(p) for p in
                     ("experiments/sprint_a_run.py",
                      "experiments/sprint_a_holdout_extras.py",
                      "framework/sprint_a.py",
                      "framework/sprint_a_inference.py",
                      "framework/models.py",
                      "framework/metrics.py") if os.path.exists(p)},
            "outputs": {p: {"sha256": sha256_file(p),
                            "bytes": os.path.getsize(p)} for p in outputs},
            "notes": [
                "one-shot confirmatory holdout; no second label read",
                "unit-fresh, not time-fresh; 2018-22 retrospectively burned",
                "blind/quick smoke log archived as run_blind_quick_aborted.log",
                "post-processing diagnostics use saved predictions; no test tuning",
            ],
            "runtime_seconds": round(time.time() - t0, 2),
        }
        jdump(MANIFEST, manifest)
        log(f"extras: manifest written; total {time.time()-t0:.1f}s")
        return 0
    finally:
        tee.close()
        sys.stdout = tee.stdout
        sys.stderr = tee.stderr


if __name__ == "__main__":
    raise SystemExit(main())
