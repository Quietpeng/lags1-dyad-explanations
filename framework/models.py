"""Transparent per-row models, baselines, calibration and attribution helpers.

The primary model is a standardised L2 logistic regression.  It is intentionally
boring: coefficients are the explanation, and the node cards decompose each
forecast into ``coef_j * x_ij``.  HistGradientBoosting is a capacity control,
not the paper's interpretable model.
"""

from __future__ import annotations

import numpy as np

from .metrics import binary_metrics


def sigmoid(z):
    z = np.clip(np.asarray(z, dtype=np.float64), -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def scaler_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(X.mean(axis=0), dtype=np.float32)
    sd = np.asarray(X.std(axis=0), dtype=np.float32)
    sd = np.where(sd < 1e-8, 1.0, sd).astype(np.float32)
    return mu, sd


def scaler_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((X - mu) / sd).astype(np.float32)


def fit_logistic(Xtr, ytr, Xsel, ysel, cfg: dict, sample_weight=None,
                 C_grid=None, feature_names=None) -> dict:
    from sklearn.linear_model import LogisticRegression

    mcfg = cfg["models"]["logistic"]
    C_grid = list(C_grid if C_grid is not None else mcfg["C_grid"])
    mu, sd = scaler_fit(Xtr)
    Xtr_s = scaler_apply(Xtr, mu, sd)
    Xsel_s = scaler_apply(Xsel, mu, sd)
    best = None
    curve = []
    for C in C_grid:
        clf = LogisticRegression(
            C=float(C), max_iter=int(mcfg["max_iter"]), tol=float(mcfg["tol"]),
            solver=mcfg["solver"], random_state=int(cfg["seed"]),
        )
        clf.fit(Xtr_s, ytr, sample_weight=sample_weight)
        p = clf.predict_proba(Xsel_s)[:, 1]
        ll = binary_metrics(ysel, p)["log_loss"]
        curve.append({"C": float(C), "select_log_loss": ll})
        if best is None or ll < best[0]:
            best = (ll, float(C), clf)
    ll, C, clf = best
    return {
        "kind": "logistic",
        "model": clf,
        "C": C,
        "mu": mu,
        "sd": sd,
        "select_log_loss": ll,
        "select_curve": curve,
        "feature_names": list(feature_names) if feature_names else None,
        "coef": clf.coef_.ravel().astype(np.float64),
        "intercept": float(clf.intercept_[0]),
    }


def predict_logistic(fit: dict, X: np.ndarray) -> np.ndarray:
    Xs = scaler_apply(np.asarray(X), fit["mu"], fit["sd"])
    return fit["model"].predict_proba(Xs)[:, 1]


def fit_hgb(Xtr, ytr, Xsel, ysel, cfg: dict, sample_weight=None,
            iters=None) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier

    mcfg = cfg["models"]["hgb"]
    iters = list(iters if iters is not None else mcfg["max_iter_grid"])
    best = None
    curve = []
    for it in iters:
        clf = HistGradientBoostingClassifier(
            max_iter=int(it), early_stopping=False,
            learning_rate=float(mcfg["learning_rate"]),
            max_leaf_nodes=int(mcfg["max_leaf_nodes"]),
            max_bins=int(mcfg["max_bins"]),
            min_samples_leaf=int(mcfg["min_samples_leaf"]),
            random_state=int(cfg["seed"]),
        )
        clf.fit(Xtr, ytr, sample_weight=sample_weight)
        p = clf.predict_proba(Xsel)[:, 1]
        ll = binary_metrics(ysel, p)["log_loss"]
        curve.append({"iters": int(it), "select_log_loss": ll})
        if best is None or ll < best[0]:
            best = (ll, int(it), clf)
    ll, it, clf = best
    return {"kind": "hgb", "model": clf, "n_iter": it, "select_log_loss": ll,
            "select_curve": curve, "feature_names": None,
            "coef": None, "intercept": None}


def predict_hgb(fit: dict, X: np.ndarray) -> np.ndarray:
    return fit["model"].predict_proba(X)[:, 1]


def predict(fit: dict, X: np.ndarray) -> np.ndarray:
    return predict_logistic(fit, X) if fit["kind"] == "logistic" else predict_hgb(fit, X)


def apply_temperature(p, T: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    logits = np.log(p / (1.0 - p)) / float(T)
    return sigmoid(logits)


def fit_temperature(p_sel, y_sel, p_eval) -> dict:
    grid = np.concatenate([np.linspace(0.4, 3.0, 53), [1.0]])
    best = None
    for T in grid:
        ll = binary_metrics(y_sel, apply_temperature(p_sel, T))["log_loss"]
        if best is None or ll < best[0]:
            best = (ll, float(T))
    T = best[1]
    return {"T": T, "select_log_loss": best[0],
            "p_sel": apply_temperature(p_sel, T),
            "p_eval": apply_temperature(p_eval, T)}


def fit_isotonic(p_sel, y_sel, p_eval) -> dict:
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0 - 1e-6)
    iso.fit(p_sel, y_sel)
    return {"iso": iso, "p_sel": iso.predict(p_sel), "p_eval": iso.predict(p_eval)}


def calibrate_all(p_sel, y_sel, p_eval, cfg: dict) -> dict:
    out = {"none": {"p_sel": p_sel, "p_eval": p_eval, "select_log_loss":
                    binary_metrics(y_sel, p_sel)["log_loss"]}}
    for name in cfg["models"]["calibration"]:
        if name == "temperature":
            out["temperature"] = fit_temperature(p_sel, y_sel, p_eval)
        elif name == "isotonic":
            out["isotonic"] = fit_isotonic(p_sel, y_sel, p_eval)
    return out


# ---------------------------------------------------------------------------
# Baselines (same definitions as g2_cpu_baselines.icews_baselines_binary)
# ---------------------------------------------------------------------------
def baseline_persistence(train_d, eval_d) -> np.ndarray:
    a = train_d["active"].astype(np.int64)
    an = train_d["y"].astype(np.int64)
    pc = float((a == an).mean())
    a_eval = eval_d["active"].astype(np.int64)
    return np.where(a_eval == 1, pc, 1.0 - pc)


def baseline_global_transition(train_d, eval_d) -> np.ndarray:
    a = train_d["active"].astype(np.int64)
    an = train_d["y"].astype(np.int64)
    p_next = np.zeros((2, 2))
    for x in (0, 1):
        m = a == x
        p_next[x] = (np.bincount(an[m], minlength=2) + 1.0) / (m.sum() + 2.0)
    return p_next[eval_d["active"].astype(np.int64), 1]


def baseline_per_dyad_markov(train_d, eval_d, n_dyads: int) -> np.ndarray:
    a = train_d["active"].astype(np.int64)
    y = train_d["y"].astype(np.int64)
    did = train_d["did"]
    n1 = np.bincount(did[a == 1], minlength=n_dyads).astype(np.float64)
    n11 = np.bincount(did[(a == 1) & (y == 1)], minlength=n_dyads).astype(np.float64)
    n0 = np.bincount(did[a == 0], minlength=n_dyads).astype(np.float64)
    n01 = np.bincount(did[(a == 0) & (y == 1)], minlength=n_dyads).astype(np.float64)
    a_all = train_d["active"].astype(np.int64)
    an_all = train_d["y"].astype(np.int64)
    pg = np.zeros((2, 2))
    for x in (0, 1):
        m = a_all == x
        pg[x] = (np.bincount(an_all[m], minlength=2) + 1.0) / (m.sum() + 2.0)
    d_eval = eval_d["did"]
    x_eval = eval_d["active"].astype(np.int64)
    p1 = np.where(
        x_eval == 1,
        (n11[d_eval] + 2 * pg[1, 1]) / (n1[d_eval] + 2),
        (n01[d_eval] + 2 * pg[0, 1]) / (n0[d_eval] + 2),
    )
    return p1.astype(np.float64)


# ---------------------------------------------------------------------------
# Per-actor local models (node cards)
# ---------------------------------------------------------------------------
def fit_per_actor_logistic(d: dict, aid: int, cfg: dict) -> dict | None:
    from sklearn.linear_model import LogisticRegression

    m = d["aid"] == aid
    y = d["y"][m]
    n, pos = int(m.sum()), int(y.sum())
    min_pos = int(cfg["models"]["per_actor_local"]["min_pos"])
    if pos < min_pos or n - pos < min_pos:
        return None
    X = d["X"][m]
    mu, sd = scaler_fit(X)
    Xs = scaler_apply(X, mu, sd)
    clf = LogisticRegression(C=float(cfg["models"]["per_actor_local"]["C"]),
                             max_iter=300, solver="lbfgs",
                             random_state=int(cfg["seed"]))
    clf.fit(Xs, y)
    p = clf.predict_proba(Xs)[:, 1]
    return {
        "aid": int(aid), "n": n, "positives": pos,
        "coef": clf.coef_.ravel().astype(np.float64),
        "intercept": float(clf.intercept_[0]),
        "mu": mu, "sd": sd,
        "train_log_loss": binary_metrics(y, p)["log_loss"],
        "train_auc": binary_metrics(y, p)["auc"],
    }


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------
def linear_contributions(fit: dict, X: np.ndarray, names: list[str]) -> np.ndarray:
    """Per-row additive log-odds contributions, shape (n, p)."""
    Xs = scaler_apply(np.asarray(X), fit["mu"], fit["sd"])
    return Xs.astype(np.float64) * fit["coef"][None, :]


def linear_contribution_table(fit: dict, X: np.ndarray, names: list[str],
                              row_idx: int, top_k: int = 8) -> dict:
    C = linear_contributions(fit, X[row_idx:row_idx + 1], names)[0]
    order = np.argsort(-np.abs(C))[:top_k]
    return {
        "logit": round(float(fit["intercept"] + C.sum()), 6),
        "intercept": round(float(fit["intercept"]), 6),
        "terms": [
            {"feature": names[j], "value": round(float(X[row_idx, j]), 6),
             "contribution": round(float(C[j]), 6)}
            for j in order
        ],
    }


def group_contributions(fit: dict, X: np.ndarray, names: list[str],
                        groups: dict[str, list[str]]) -> np.ndarray:
    """Aggregate additive contributions by feature group."""
    C = linear_contributions(fit, X, names)
    idx = {n: i for i, n in enumerate(names)}
    cols = []
    for g, feats in groups.items():
        js = [idx[f] for f in feats if f in idx]
        cols.append(C[:, js].sum(axis=1) if js else np.zeros(len(X)))
    return np.column_stack(cols)


def permutation_importance(fit: dict, X, y, names, n_repeats: int = 3,
                           seed: int = 0, max_rows: int = 200000) -> list[dict]:
    rng = np.random.default_rng(seed)
    if len(y) > max_rows:
        idx = np.sort(rng.choice(len(y), size=max_rows, replace=False))
        X, y = X[idx], y[idx]
    base = binary_metrics(y, predict(fit, X))["log_loss"]
    rows = []
    for j, n in enumerate(names):
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            deltas.append(binary_metrics(y, predict(fit, Xp))["log_loss"] - base)
        rows.append({"feature": n, "delta_log_loss": round(float(np.mean(deltas)), 6),
                     "sd": round(float(np.std(deltas)), 6)})
    return sorted(rows, key=lambda r: -r["delta_log_loss"])


def occlusion_attribution(fit: dict, X, names, row_idx: int, baseline: np.ndarray,
                           top_k: int = 8) -> dict:
    """Model-agnostic per-row attribution: replace feature by train mean."""
    x = X[row_idx:row_idx + 1].copy()
    p0 = float(predict(fit, x)[0])
    rows = []
    for j, n in enumerate(names):
        xp = x.copy()
        xp[0, j] = baseline[j]
        p1 = float(predict(fit, xp)[0])
        rows.append({"feature": n, "value": round(float(x[0, j]), 6),
                     "delta_p": round(p1 - p0, 6)})
    rows = sorted(rows, key=lambda r: -abs(r["delta_p"]))[:top_k]
    return {"p": round(p0, 6), "terms": rows}


def nested_group_decomposition(Xtr, ytr, Xsel, ysel, Xeval, yeval, names,
                               group_order: list[str], groups: dict,
                               cfg: dict) -> list[dict]:
    """Fit cumulative feature-group logistic models and report eval LL."""
    idx = {n: i for i, n in enumerate(names)}
    active: list[str] = []
    rows = []
    for g in group_order:
        active = active + [f for f in groups.get(g, []) if f in idx]
        cols = [idx[f] for f in active]
        fit = fit_logistic(Xtr[:, cols], ytr, Xsel[:, cols], ysel,
                           cfg, feature_names=active)
        p = predict_logistic(fit, Xeval[:, cols])
        m = binary_metrics(yeval, p)
        rows.append({"groups": list(group_order[:group_order.index(g) + 1]),
                     "n_features": len(cols), "C": fit["C"],
                     **{k: m[k] for k in ("log_loss", "brier", "ece", "auc", "ap")}})
    return rows
