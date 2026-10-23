"""Sprint-A inference: paired cluster bootstrap, multiway CIs, BH.

The P1 ``dual_cluster_gain`` drew separate bootstrap weights for the baseline
and the model, so its "paired" interval was not paired.  This module fixes that:
one weight vector per replicate is applied to both sides.

Clustering geometry
-------------------
Each directed dyad has a fixed source and target.  The source partition and the
target partition therefore intersect in the dyad partition.  The Cameron--
Gelbach--Miller two-way variance for source+target is

    Var(source + target) = Var(source) + Var(target) - Var(dyad)

and the three-way (dyad + source + target) variance reduces to the same
expression because dyad is nested in both actor partitions.  We floor the
combination at ``max(Var_s, Var_t, Var_d)`` and report it as ``multiway``.
The dyad-only and one-sided actor intervals are reported alongside it; no
claim is based on the dyad-only interval alone.
"""

from __future__ import annotations

import math

import numpy as np

from .bootstrap import aggregate, boot_weights, ci, cluster_sufficient
from .metrics import binary_metrics, icc_deff

Z_MDE = 2.801585  # z_{0.975} + z_{0.80}


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(float(z) / math.sqrt(2.0))


def _norm_p(point: float, se: float) -> float:
    if se <= 0:
        return 0.0 if point != 0 else 1.0
    return float(min(1.0, 2.0 * _norm_sf(abs(point) / se)))


def _var(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(np.var(x, ddof=1))


def paired_draws(y, p_base, p_model, cid, reps: int, seed: int) -> dict:
    """Paired bootstrap draws of base-minus-model metric differences."""
    rng = np.random.default_rng(int(seed))
    Sb = cluster_sufficient(y, p_base, cid)
    Sm = cluster_sufficient(y, p_model, cid)
    W = boot_weights(Sb["n_clusters"], int(reps), rng)
    B = aggregate(W, Sb)
    M = aggregate(W, Sm)
    return {
        "d_ll": B["log_loss"] - M["log_loss"],
        "d_ece": B["ece"] - M["ece"],
        "d_brier": B["brier"] - M["brier"],
        "n_clusters": int(Sb["n_clusters"]),
        "reps": int(reps),
    }


def paired_multiway(y, p_base, p_model, did, sid, tid,
                    reps: int = 1000, seed: int = 0) -> dict:
    """Paired multiway bootstrap for log-loss, ECE and Brier differences."""
    y = np.asarray(y, dtype=np.int64)
    p_base = np.asarray(p_base, dtype=np.float64)
    p_model = np.asarray(p_model, dtype=np.float64)
    base = binary_metrics(y, p_base)
    model = binary_metrics(y, p_model)
    point_ll = float(base["log_loss"] - model["log_loss"])
    point_ece = float(base["ece"] - model["ece"])
    point_brier = float(base["brier"] - model["brier"])
    point = {
        "base_log_loss": float(base["log_loss"]),
        "model_log_loss": float(model["log_loss"]),
        "d_log_loss": point_ll,
        "rel_log_loss_gain": point_ll / max(float(base["log_loss"]), 1e-12),
        "base_ece": float(base["ece"]),
        "model_ece": float(model["ece"]),
        "d_ece": point_ece,
        "base_brier": float(base["brier"]),
        "model_brier": float(model["brier"]),
        "d_brier": point_brier,
        "n": int(len(y)),
        "positives": int(y.sum()),
    }
    dims = {"dyad": np.asarray(did), "source": np.asarray(sid),
            "target": np.asarray(tid)}
    out = {"point": point, "dims": {}, "reps": int(reps)}
    for j, (name, cid) in enumerate(dims.items()):
        d = paired_draws(y, p_base, p_model, cid, reps, seed + 17 * j)
        out["dims"][name] = {
            "n_clusters": d["n_clusters"],
            "var_ll": _var(d["d_ll"]), "var_ece": _var(d["d_ece"]),
            "var_brier": _var(d["d_brier"]),
            "ci_ll": ci(d["d_ll"]), "ci_ece": ci(d["d_ece"]),
            "ci_brier": ci(d["d_brier"]),
            "se_ll": float(np.std(d["d_ll"], ddof=1)),
            "se_ece": float(np.std(d["d_ece"], ddof=1)),
            "se_brier": float(np.std(d["d_brier"], ddof=1)),
        }
    vs, vt, vd = (out["dims"]["source"], out["dims"]["target"],
                  out["dims"]["dyad"])
    v_s, v_t, v_d = vs["var_ll"], vt["var_ll"], vd["var_ll"]
    var_m = max(v_s + v_t - v_d, v_s, v_t, v_d)
    se_m = math.sqrt(max(var_m, 0.0))
    out["multiway"] = {
        "formula": "max(Var_source + Var_target - Var_dyad, "
                   "Var_source, Var_target, Var_dyad)",
        "var_ll": var_m, "se_ll": se_m,
        "ci_ll": [point_ll - 1.96 * se_m, point_ll + 1.96 * se_m],
        "p_ll": _norm_p(point_ll, se_m),
    }
    # ECE and Brier use the same variance combination.
    for metric in ("ece", "brier"):
        v_s, v_t, v_d = vs[f"var_{metric}"], vt[f"var_{metric}"], vd[f"var_{metric}"]
        var_m = max(v_s + v_t - v_d, v_s, v_t, v_d)
        se_m = math.sqrt(max(var_m, 0.0))
        pt = point[f"d_{metric}"]
        out["multiway"][f"var_{metric}"] = var_m
        out["multiway"][f"se_{metric}"] = se_m
        out["multiway"][f"ci_{metric}"] = [pt - 1.96 * se_m, pt + 1.96 * se_m]
        out["multiway"][f"p_{metric}"] = _norm_p(pt, se_m)
    out["multiway"]["n_clusters"] = {
        "dyad": vd["n_clusters"], "source": vs["n_clusters"],
        "target": vt["n_clusters"],
    }
    return out


# ---------------------------------------------------------------------------
# AUC / AP
# ---------------------------------------------------------------------------
def _auc_ap(y, score):
    from sklearn.metrics import average_precision_score, roc_auc_score

    try:
        return (float(roc_auc_score(y, score)),
                float(average_precision_score(y, score)))
    except ValueError:
        return None, None


def _resample_index(cid_inv, sizes, starts, rng, k):
    c = rng.integers(0, k, size=k)
    lengths = sizes[c]
    total = int(lengths.sum())
    if total == 0:
        return None
    rep_start = np.repeat(starts[c], lengths)
    offs = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    return rep_start + offs


def auc_ap_boot(y, score, cid, reps: int = 1000, seed: int = 0) -> dict:
    """Cluster bootstrap CI for AUC/AP on the full evaluation sample."""
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    _, inv = np.unique(np.asarray(cid), return_inverse=True)
    k = int(inv.max()) + 1
    sizes = np.bincount(inv).astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    order = np.argsort(inv, kind="stable")
    y_s, s_s = y[order], score[order]
    # Recompute starts on the sorted arrays.
    sizes = np.bincount(inv[order]).astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    rng = np.random.default_rng(int(seed))
    aucs, aps = [], []
    for _ in range(int(reps)):
        ridx = _resample_index(inv, sizes, starts, rng, k)
        if ridx is None:
            continue
        a, p = _auc_ap(y_s[ridx], s_s[ridx])
        if a is not None:
            aucs.append(a)
            aps.append(p)
    point_auc, point_ap = _auc_ap(y, score)
    return {"auc": point_auc, "ap": point_ap,
            "auc_ci": ci(aucs), "ap_ci": ci(aps),
            "auc_draws": np.asarray(aucs), "ap_draws": np.asarray(aps),
            "reps": int(len(aucs)), "n_clusters": int(k), "n": int(len(y))}


def paired_auc_ap_boot(y, score_base, score_model, cid,
                       reps: int = 1000, seed: int = 0) -> dict:
    """Paired cluster bootstrap for delta-AUC and delta-AP (model - base)."""
    y = np.asarray(y, dtype=np.int64)
    s0 = np.asarray(score_base, dtype=np.float64)
    s1 = np.asarray(score_model, dtype=np.float64)
    _, inv = np.unique(np.asarray(cid), return_inverse=True)
    k = int(inv.max()) + 1
    order = np.argsort(inv, kind="stable")
    y_s, s0_s, s1_s = y[order], s0[order], s1[order]
    sizes = np.bincount(inv[order]).astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    rng = np.random.default_rng(int(seed))
    da, dp = [], []
    for _ in range(int(reps)):
        ridx = _resample_index(inv, sizes, starts, rng, k)
        if ridx is None:
            continue
        a0, p0 = _auc_ap(y_s[ridx], s0_s[ridx])
        a1, p1 = _auc_ap(y_s[ridx], s1_s[ridx])
        if a0 is None or a1 is None:
            continue
        da.append(a1 - a0)
        dp.append(p1 - p0)
    a0, p0 = _auc_ap(y, s0)
    a1, p1 = _auc_ap(y, s1)
    return {
        "base_auc": a0, "model_auc": a1, "d_auc": None if a0 is None else a1 - a0,
        "base_ap": p0, "model_ap": p1, "d_ap": None if p0 is None else p1 - p0,
        "d_auc_ci": ci(da), "d_ap_ci": ci(dp),
        "d_auc_draws": np.asarray(da), "d_ap_draws": np.asarray(dp),
        "reps": int(len(da)), "n_clusters": int(k),
    }


def auc_ap_multiway(y, score, did, sid, tid, reps: int = 1000,
                    seed: int = 0) -> dict:
    """Multiway CIs for the *level* AUC/AP of one score."""
    dims = {"dyad": np.asarray(did), "source": np.asarray(sid),
            "target": np.asarray(tid)}
    out = {"dims": {}, "reps": int(reps)}
    for j, (name, cid) in enumerate(dims.items()):
        d = auc_ap_boot(y, score, cid, reps=reps, seed=seed + 19 * j)
        out["dims"][name] = {
            "n_clusters": d["n_clusters"],
            "var_auc": _var(d["auc_draws"]), "var_ap": _var(d["ap_draws"]),
            "ci_auc": d["auc_ci"], "ci_ap": d["ap_ci"],
        }
    out["point"] = {"auc": _auc_ap(np.asarray(y), np.asarray(score))[0],
                    "ap": _auc_ap(np.asarray(y), np.asarray(score))[1]}
    vs, vt, vd = (out["dims"]["source"], out["dims"]["target"],
                  out["dims"]["dyad"])
    for metric in ("auc", "ap"):
        v_s, v_t, v_d = vs[f"var_{metric}"], vt[f"var_{metric}"], vd[f"var_{metric}"]
        var_m = max(v_s + v_t - v_d, v_s, v_t, v_d)
        se_m = math.sqrt(max(var_m, 0.0))
        pt = out["point"][metric]
        out[f"{metric}_multiway"] = {
            "var": var_m, "se": se_m,
            "ci": [pt - 1.96 * se_m, pt + 1.96 * se_m],
        }
    return out


def paired_auc_ap_multiway(y, score_base, score_model, did, sid, tid,
                           reps: int = 1000, seed: int = 0) -> dict:
    """Multiway CI for the paired delta-AUC / delta-AP (model - base)."""
    dims = {"dyad": np.asarray(did), "source": np.asarray(sid),
            "target": np.asarray(tid)}
    out = {"dims": {}, "reps": int(reps)}
    for j, (name, cid) in enumerate(dims.items()):
        d = paired_auc_ap_boot(y, score_base, score_model, cid,
                               reps=reps, seed=seed + 23 * j)
        out["dims"][name] = {
            "n_clusters": d["n_clusters"],
            "var_auc": _var(d["d_auc_draws"]),
            "var_ap": _var(d["d_ap_draws"]),
            "ci_auc": d["d_auc_ci"], "ci_ap": d["d_ap_ci"],
        }
        out["point"] = {"d_auc": d["d_auc"], "d_ap": d["d_ap"],
                        "base_auc": d["base_auc"], "model_auc": d["model_auc"],
                        "base_ap": d["base_ap"], "model_ap": d["model_ap"]}
    vs, vt, vd = (out["dims"]["source"], out["dims"]["target"],
                  out["dims"]["dyad"])
    for metric in ("auc", "ap"):
        v_s, v_t, v_d = vs[f"var_{metric}"], vt[f"var_{metric}"], vd[f"var_{metric}"]
        var_m = max(v_s + v_t - v_d, v_s, v_t, v_d)
        se_m = math.sqrt(max(var_m, 0.0))
        pt = out["point"][f"d_{metric}"]
        out[f"multiway_{metric}"] = {
            "var": var_m, "se": se_m,
            "ci": [pt - 1.96 * se_m, pt + 1.96 * se_m],
            "p": _norm_p(pt, se_m),
        }
    out["formula"] = (
        "max(Var_source + Var_target - Var_dyad, Var_source, Var_target, Var_dyad)")
    return out


# ---------------------------------------------------------------------------
# Multiplicity and power
# ---------------------------------------------------------------------------
def bh_fdr(pvals, alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m, dtype=np.float64)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = int(order[rank])
        val = min(prev, float(p[i]) * m / (rank + 1))
        q[i] = val
        prev = val
    return {"p": p.tolist(), "q": q.tolist(),
            "reject": (q <= alpha).tolist(), "alpha": float(alpha),
            "m": int(m)}


def mde_pair(y, p_base, p_model, did, label: str = "") -> dict:
    """MDE for the paired contrast, plus a baseline-variance reference."""
    y = np.asarray(y, dtype=np.int64)
    p_base = np.clip(np.asarray(p_base, dtype=np.float64), 1e-12, 1 - 1e-12)
    p_model = np.clip(np.asarray(p_model, dtype=np.float64), 1e-12, 1 - 1e-12)
    loss_b = -(y * np.log(p_base) + (1 - y) * np.log(1 - p_base))
    loss_m = -(y * np.log(p_model) + (1 - y) * np.log(1 - p_model))
    diff = loss_b - loss_m
    sd = float(np.std(diff, ddof=1))
    deff = icc_deff(diff, did)
    mde = Z_MDE * sd / math.sqrt(deff["n_eff"]) if deff["n_eff"] > 0 else float("inf")
    sd_b = float(np.std(loss_b, ddof=1))
    deff_b = icc_deff(loss_b, did)
    mde_b = (Z_MDE * sd_b / math.sqrt(deff_b["n_eff"])
             if deff_b["n_eff"] > 0 else float("inf"))
    base_ll = float(np.mean(loss_b))
    return {
        "label": label,
        "base_log_loss": base_ll,
        "paired_sd": sd,
        "paired_deff": deff["deff"],
        "paired_n_eff": deff["n_eff"],
        "mde_paired": float(mde),
        "mde_paired_rel": float(mde / max(base_ll, 1e-12)),
        "baseline_sd": sd_b,
        "baseline_deff": deff_b["deff"],
        "baseline_n_eff": deff_b["n_eff"],
        "mde_baseline": float(mde_b),
        "mde_baseline_rel": float(mde_b / max(base_ll, 1e-12)),
    }
