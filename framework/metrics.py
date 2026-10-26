"""Metrics for the framework.

Definitions follow the frozen G2 baseline where possible so that pilot numbers
are comparable with ``results/g2_cpu``:

* binary log-loss / Brier / ECE / accuracy / macro-F1 / AUC / AP
* ICC -> design effect -> n_eff (cluster = actor or dyad)
* minimum detectable effect (MDE) at 80% power, two-sided alpha = 0.05

The ECE here is a *binary* ECE with 10 equal-width bins on p(y=1).  The G2
``ece_top1`` is the multiclass analogue; the operator head uses that one.
"""

from __future__ import annotations

import math

import numpy as np

ECE_BINS = 10
Z_MDE = 2.801585  # z_{0.975} + z_{0.80}


def _safe_auc(y: np.ndarray, p: np.ndarray):
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        return float(roc_auc_score(y, p)), float(average_precision_score(y, p))
    except ValueError:
        return None, None


def binary_metrics(y, p, threshold: float = 0.5, ece_bins: int = ECE_BINS) -> dict:
    y = np.asarray(y, dtype=np.int64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    n = int(len(y))
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1.0 - p)))
    brier = float(np.mean((p - y) ** 2))

    bins = np.clip((p * ece_bins).astype(int), 0, ece_bins - 1)
    ece = 0.0
    for b in range(ece_bins):
        m = bins == b
        if m.any():
            ece += float(m.mean()) * abs(float(y[m].mean()) - float(p[m].mean()))

    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    f1_pos = 2.0 * tp / max(2.0 * tp + fp + fn, 1e-12)
    f1_neg = 2.0 * tn / max(2.0 * tn + fn + fp, 1e-12)
    auc, ap = _safe_auc(y, p) if len(np.unique(y)) > 1 else (None, None)
    return {
        "n": n,
        "positives": int(y.sum()),
        "pos_rate": round(float(y.mean()), 6),
        "log_loss": round(ll, 6),
        "brier": round(brier, 6),
        "ece": round(float(ece), 6),
        "accuracy": round(float((pred == y).mean()), 6),
        "f1_pos": round(float(f1_pos), 6),
        "macro_f1": round(float(0.5 * (f1_pos + f1_neg)), 6),
        "auc": None if auc is None else round(auc, 6),
        "ap": None if ap is None else round(ap, 6),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def top1_metrics(y, P, ece_bins: int = ECE_BINS) -> dict:
    """Multiclass metrics (same convention as g2 ``top1_metrics``)."""
    y = np.asarray(y, dtype=np.int64)
    P = np.asarray(P, dtype=np.float64)
    n, K = P.shape
    idx = np.arange(n)
    p_true = np.clip(P[idx, y], 1e-12, 1.0)
    ll = float(-np.mean(np.log(p_true)))
    Y = np.zeros_like(P)
    Y[idx, y] = 1.0
    brier = float(np.mean(np.sum((P - Y) ** 2, axis=1)))
    pred = P.argmax(axis=1)
    conf = P[idx, pred]
    correct = (pred == y).astype(np.float64)
    bin_i = np.clip(np.ceil(conf * ece_bins).astype(int) - 1, 0, ece_bins - 1)
    ece = 0.0
    for b in range(ece_bins):
        m = bin_i == b
        if m.any():
            ece += float(m.mean()) * abs(float(correct[m].mean()) - float(conf[m].mean()))
    cm = np.zeros((K, K), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    f1 = 2.0 * tp / np.maximum(2.0 * tp + fp + fn, 1e-12)
    out = {
        "n": int(n),
        "log_loss": round(ll, 6),
        "brier": round(brier, 6),
        "ece_top1": round(float(ece), 6),
        "accuracy": round(float(correct.mean()), 6),
        "macro_f1": round(float(f1.mean()), 6),
        "per_class_f1": [round(float(v), 6) for v in f1],
        "confusion": cm.tolist(),
    }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        out["auc_macro_ovr"] = round(float(roc_auc_score(
            y, P, average="macro", multi_class="ovr")), 6)
        out["ap_macro"] = round(float(average_precision_score(
            y, P, average="macro")), 6)
    except Exception:
        out["auc_macro_ovr"] = None
        out["ap_macro"] = None
    return out


def icc_deff(x, clusters) -> dict:
    """One-way ANOVA ICC and design effect (same estimator as g2_deff.py)."""
    x = np.asarray(x, dtype=np.float64)
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    k = int(inv.max()) + 1
    sizes = np.bincount(inv).astype(np.float64)
    sums = np.bincount(inv, weights=x)
    means = sums / np.maximum(sizes, 1.0)
    grand = float(x.mean())
    ssb = float(np.sum(sizes * (means - grand) ** 2))
    sst = float(np.sum((x - grand) ** 2))
    N = int(len(x))
    m0 = float(sizes.mean())
    msb = ssb / max(k - 1, 1)
    msw = (sst - ssb) / max(N - k, 1) if N > k else 0.0
    den = msb + (m0 - 1.0) * msw
    icc = max(0.0, (msb - msw) / den) if den > 0 else 0.0
    deff = 1.0 + (m0 - 1.0) * icc
    return {
        "icc": round(float(icc), 6),
        "deff": round(float(deff), 6),
        "n_eff": round(float(N / deff), 2) if deff > 0 else float("inf"),
        "clusters": k,
        "mean_size": round(m0, 3),
        "n": N,
    }


def mde_block(diff, clusters, base_ll: float | None = None) -> dict:
    """MDE for a paired per-row contrast, clustered."""
    diff = np.asarray(diff, dtype=np.float64)
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    st = icc_deff(diff, clusters)
    mde = Z_MDE * sd / math.sqrt(st["n_eff"]) if st["n_eff"] > 0 else float("inf")
    out = {"sd": round(sd, 6), **st, "mde": round(float(mde), 6)}
    if base_ll:
        out["mde_rel"] = round(float(mde / base_ll), 6)
    return out


def calibration_table(y, p, bins: int = 10) -> list[dict]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({
            "bin": b,
            "lo": round(float(edges[b]), 4),
            "hi": round(float(edges[b + 1]), 4),
            "n": int(m.sum()),
            "mean_p": round(float(p[m].mean()), 6) if m.any() else None,
            "mean_y": round(float(y[m].mean()), 6) if m.any() else None,
        })
    return rows


def reliability_slope(y, p) -> dict:
    """OLS calibration slope/intercept of y on logit(p)."""
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    z = np.log(p / (1 - p))
    A = np.column_stack([np.ones_like(z), z])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {"intercept": round(float(coef[0]), 6), "slope": round(float(coef[1]), 6)}


def within_entity_auc(y, p, cid) -> dict:
    """AUC after removing each entity's mean score.

    This separates *temporal* discrimination (does the model rank months
    within an actor/dyad?) from *between-entity* discrimination (does it know
    which entities are active-prone?).  A high global AUC with a ~0.5
    within-entity AUC is a propensity model, not a month-level forecast.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    cid = np.asarray(cid)
    _, inv = np.unique(cid, return_inverse=True)
    k = int(inv.max()) + 1
    means = np.bincount(inv, weights=p, minlength=k) / np.maximum(
        np.bincount(inv, minlength=k), 1)
    pc = p - means[inv]
    out = {"n": int(len(y)), "clusters": k,
           "within_auc": None, "between_auc": None}
    if len(np.unique(y)) > 1:
        try:
            out["within_auc"] = round(float(roc_auc_score(y, pc)), 6)
        except ValueError:
            pass
    # Between-entity: cluster mean outcome (binary if all clusters pure) vs
    # cluster mean score.
    ymean = np.bincount(inv, weights=y, minlength=k) / np.maximum(
        np.bincount(inv, minlength=k), 1)
    pmean = means
    lab = (ymean > 0.5).astype(np.int64)
    if len(np.unique(lab)) > 1:
        try:
            out["between_auc"] = round(float(roc_auc_score(lab, pmean)), 6)
        except ValueError:
            pass
    return out
