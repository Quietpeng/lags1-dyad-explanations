"""Cluster bootstrap utilities.

Two clustering dimensions matter for this panel:

* **dyad** -- the standard G2 clustering; within-dyad rows are strongly
  dependent (persistence).
* **actor** -- actor-level heterogeneity is the node-level unit of the paper.

``dual_cluster_gain`` reports both one-way bootstraps and a multiway
(actor + dyad) interval using the inclusion-exclusion variance

    Var_multi = max(Var_actor + Var_dyad - Var_row, Var_actor, Var_dyad)

where ``Var_row`` is the iid row-level variance.  This is the standard
multiway cluster-robust combination; it is conservative when the two
partitions are nearly nested (they are not: actors cross dyads).
"""

from __future__ import annotations

import numpy as np

ECE_BINS = 10


def ci(arr, lo: float = 2.5, hi: float = 97.5) -> list | None:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return [round(float(np.percentile(arr, lo)), 6),
            round(float(np.percentile(arr, hi)), 6)]


def boot_weights(n_cl: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    W = np.empty((reps, n_cl), dtype=np.float64)
    for r in range(reps):
        W[r] = np.bincount(rng.integers(0, n_cl, size=n_cl), minlength=n_cl)
    return W


def cluster_sufficient(y, p, cid) -> dict:
    """Sufficient statistics per cluster for LL/Brier/ECE/accuracy/F1."""
    y = np.asarray(y, dtype=np.int64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    _, inv = np.unique(np.asarray(cid), return_inverse=True)
    k = int(inv.max()) + 1
    nll = -(y * np.log(p) + (1 - y) * np.log(1.0 - p))
    brier = (p - y) ** 2
    pred = (p >= 0.5).astype(np.int64)
    tp = ((pred == 1) & (y == 1)).astype(np.float64)
    fp = ((pred == 1) & (y == 0)).astype(np.float64)
    fn = ((pred == 0) & (y == 1)).astype(np.float64)
    tn = ((pred == 0) & (y == 0)).astype(np.float64)
    S = {
        "n": np.bincount(inv, minlength=k).astype(np.float64),
        "nll": np.bincount(inv, weights=nll, minlength=k),
        "brier": np.bincount(inv, weights=brier, minlength=k),
        "tp": np.bincount(inv, weights=tp, minlength=k),
        "fp": np.bincount(inv, weights=fp, minlength=k),
        "fn": np.bincount(inv, weights=fn, minlength=k),
        "tn": np.bincount(inv, weights=tn, minlength=k),
        "n_clusters": k,
    }
    bins = np.clip((p * ECE_BINS).astype(int), 0, ECE_BINS - 1)
    for b in range(ECE_BINS):
        m = bins == b
        S[f"bn{b}"] = np.bincount(inv[m], minlength=k).astype(np.float64)
        S[f"bp{b}"] = np.bincount(inv[m], weights=p[m], minlength=k)
        S[f"by{b}"] = np.bincount(inv[m], weights=y[m].astype(np.float64), minlength=k)
    return S


def aggregate(W: np.ndarray, S: dict) -> dict:
    N = np.maximum(W @ S["n"], 1.0)
    out = {
        "log_loss": (W @ S["nll"]) / N,
        "brier": (W @ S["brier"]) / N,
        "accuracy": (W @ (S["tp"] + S["tn"])) / N,
    }
    ece = np.zeros(W.shape[0])
    for b in range(ECE_BINS):
        bn = W @ S[f"bn{b}"]
        bp = W @ S[f"bp{b}"]
        by = W @ S[f"by{b}"]
        m = bn > 0
        contrib = np.zeros(W.shape[0])
        contrib[m] = (bn[m] / N[m]) * np.abs(by[m] / bn[m] - bp[m] / bn[m])
        ece += contrib
    out["ece"] = ece
    tp = W @ S["tp"]
    fp = W @ S["fp"]
    fn = W @ S["fn"]
    tn = W @ S["tn"]
    f1p = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)
    f1n = 2 * tn / np.maximum(2 * tn + fn + fp, 1e-12)
    out["f1_pos"] = f1p
    out["macro_f1"] = 0.5 * (f1p + f1n)
    return out


def _draws(y, p, cid, reps: int, rng) -> tuple[dict, np.ndarray]:
    S = cluster_sufficient(y, p, cid)
    W = boot_weights(S["n_clusters"], reps, rng)
    return aggregate(W, S), W


def bootstrap_metrics(y, p, cid, reps: int = 500, seed: int = 0) -> dict:
    """Point metrics plus bootstrap CIs for LL/Brier/ECE/accuracy/macro-F1."""
    rng = np.random.default_rng(seed)
    draws, _ = _draws(y, p, cid, reps, rng)
    point = {k: float(v[0]) if False else None for k, v in draws.items()}
    # Point values from the full sample, not from a bootstrap replicate.
    from .metrics import binary_metrics

    pm = binary_metrics(y, p)
    for k in ("log_loss", "brier", "ece", "accuracy", "macro_f1", "f1_pos"):
        point[k] = pm[k]
    out = {"point": {k: round(float(v), 6) for k, v in point.items()},
           "reps": int(reps), "n_clusters": int(np.unique(cid).size)}
    for k in ("log_loss", "brier", "ece", "accuracy", "macro_f1", "f1_pos"):
        out[k + "_ci"] = ci(draws[k])
    return out


def paired_gain(y, p_base, p_model, cid, reps: int = 500, seed: int = 0,
                base_ll: float | None = None) -> dict:
    """Paired cluster bootstrap for base - model gains."""
    rng = np.random.default_rng(seed)
    Sb = cluster_sufficient(y, p_base, cid)
    Sm = cluster_sufficient(y, p_model, cid)
    W = boot_weights(Sb["n_clusters"], reps, rng)
    B, M = aggregate(W, Sb), aggregate(W, Sm)
    from .metrics import binary_metrics

    bpm, mpm = binary_metrics(y, p_base), binary_metrics(y, p_model)
    d_ll = B["log_loss"] - M["log_loss"]
    rel = d_ll / np.maximum(B["log_loss"], 1e-12)
    d_brier = B["brier"] - M["brier"]
    d_ece = B["ece"] - M["ece"]
    d_f1 = M["macro_f1"] - B["macro_f1"]
    point = {
        "base_log_loss": round(float(bpm["log_loss"]), 6),
        "model_log_loss": round(float(mpm["log_loss"]), 6),
        "d_log_loss": round(float(bpm["log_loss"] - mpm["log_loss"]), 6),
        "rel_log_loss_gain": round(float(bpm["log_loss"] - mpm["log_loss"]) / max(bpm["log_loss"], 1e-12), 6),
        "d_brier": round(float(bpm["brier"] - mpm["brier"]), 6),
        "d_ece": round(float(bpm["ece"] - mpm["ece"]), 6),
        "rel_ece_gain": round(float(bpm["ece"] - mpm["ece"]) / max(bpm["ece"], 1e-12), 6),
        "d_macro_f1": round(float(mpm["macro_f1"] - bpm["macro_f1"]), 6),
    }
    ci_block = {
        "d_log_loss_ci": ci(d_ll),
        "rel_log_loss_gain_ci": ci(rel),
        "d_brier_ci": ci(d_brier),
        "d_ece_ci": ci(d_ece),
        "d_macro_f1_ci": ci(d_f1),
    }
    if base_ll is None:
        base_ll = bpm["log_loss"]
    return {"point": point, **ci_block,
            "mde": _mde_from_diff(
                -(y * np.log(np.clip(p_model, 1e-12, 1)) +
                  (1 - y) * np.log(np.clip(1 - p_model, 1e-12, 1))) +
                (y * np.log(np.clip(p_base, 1e-12, 1)) +
                 (1 - y) * np.log(np.clip(1 - p_base, 1e-12, 1))),
                cid, base_ll),
            "reps": int(reps)}


def _mde_from_diff(diff, cid, base_ll):
    from .metrics import mde_block

    return mde_block(diff, cid, base_ll)


def dual_cluster_gain(y, p_base, p_model, actor_cid, dyad_cid,
                      reps: int = 500, seed: int = 0) -> dict:
    """Actor-cluster, dyad-cluster and multiway (actor+dyad) intervals."""
    y = np.asarray(y, dtype=np.int64)
    p_base = np.asarray(p_base, dtype=np.float64)
    p_model = np.asarray(p_model, dtype=np.float64)
    n = len(y)
    row_ll = -(y * np.log(np.clip(p_model, 1e-12, 1)) +
               (1 - y) * np.log(np.clip(1 - p_model, 1e-12, 1)))
    base_ll = -(y * np.log(np.clip(p_base, 1e-12, 1)) +
                (1 - y) * np.log(np.clip(1 - p_base, 1e-12, 1)))
    diff = base_ll - row_ll
    point = float(diff.mean())
    se_row = float(diff.std(ddof=1) / np.sqrt(max(n, 1)))

    out = {"point_d_log_loss": round(point, 8),
           "se_row_iid": round(se_row, 8),
           "n": int(n)}
    se = {}
    for name, cid in (("actor", actor_cid), ("dyad", dyad_cid)):
        rng = np.random.default_rng(seed + (0 if name == "actor" else 1))
        draws, _ = _draws(y, p_model, cid, reps, rng)
        base_draws, _ = _draws(y, p_base, cid, reps, rng)
        d = base_draws["log_loss"] - draws["log_loss"]
        se[name] = float(np.std(d, ddof=1))
        out[f"{name}_ci"] = ci(d)
        out[f"{name}_se"] = round(se[name], 8)
    se_multi = float(np.sqrt(max(se["actor"] ** 2 + se["dyad"] ** 2 - se_row ** 2,
                                 se["actor"] ** 2, se["dyad"] ** 2)))
    out["multiway_se"] = round(se_multi, 8)
    out["multiway_ci"] = [round(point - 1.96 * se_multi, 8),
                          round(point + 1.96 * se_multi, 8)]
    out["multiway_note"] = ("Var = max(Var_actor + Var_dyad - Var_row, "
                            "Var_actor, Var_dyad)")
    return out


def auc_ap_ci(y, score, cid, reps: int = 200, seed: int = 0,
              max_rows: int = 300000) -> dict:
    """Cluster bootstrap CI for AUC/AP on a row subsample (cost control)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y)
    score = np.asarray(score)
    cid = np.asarray(cid)
    if len(y) > max_rows:
        rng0 = np.random.default_rng(seed)
        idx = np.sort(rng0.choice(len(y), size=max_rows, replace=False))
        y, score, cid = y[idx], score[idx], cid[idx]
    _, inv = np.unique(cid, return_inverse=True)
    k = int(inv.max()) + 1
    sizes = np.bincount(inv).astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    rng = np.random.default_rng(seed)
    aucs, aps = [], []
    for _ in range(reps):
        c = rng.integers(0, k, size=k)
        lengths = sizes[c]
        total = int(lengths.sum())
        if total == 0:
            continue
        rep_start = np.repeat(starts[c], lengths)
        offs = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        ridx = rep_start + offs
        try:
            aucs.append(roc_auc_score(y[ridx], score[ridx]))
            aps.append(average_precision_score(y[ridx], score[ridx]))
        except ValueError:
            continue
    try:
        point_auc = float(roc_auc_score(y, score))
        point_ap = float(average_precision_score(y, score))
    except ValueError:
        point_auc = point_ap = None
    return {
        "auc": round(point_auc, 6) if point_auc is not None else None,
        "ap": round(point_ap, 6) if point_ap is not None else None,
        "auc_ci": ci(aucs),
        "ap_ci": ci(aps),
        "reps": len(aucs),
        "subsample_rows": int(len(y)),
    }
