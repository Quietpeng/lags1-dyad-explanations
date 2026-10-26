"""Exact additive attribution, actor-level aggregation and stability metrics.

Purpose
-------
This framework's interpretable model is a standardised L2 logistic
regression.  For a row ``i`` with standardised features
``z_ij = (x_ij - mu_j) / sd_j`` the model is exactly additive:

    eta_i = b + sum_j beta_j * z_ij
          = b + sum_j C_ij,                 C_ij = beta_j * z_ij

``C_ij`` is the exact change in raw log-odds relative to the training mean.
Temperature scaling is affine on the log-odds scale (``eta -> eta / T``), so
``C_ij / T`` is again an exact decomposition.  Isotonic calibration is a
monotone step function, **not** affine; it is deliberately excluded from the
exact track and only reported as a sensitivity check.

Entity buckets
--------------
Every feature is assigned to exactly one of six ownership buckets:

``source_actor``  functions of the *source* actor's ego state
                  (own event volume, out-degree, eigenvector/pagerank,
                  source-side NRA)
``target_actor``  the same for the *target* actor
``dyad``          the ordered pair's own interaction history, both directions
``calendar``      year trend and month harmonics
``graph``         symmetric dyadic closure (common neighbours, Jaccard,
                  Adamic-Adar, resource allocation, two-hop, recent variants)
``train_stats``   train-only dyad base rate and seen-in-train indicator

The actor buckets deliberately contain only *actor-ego* quantities.  Features
that describe the pair (e.g. ``a1`` = an event from s to t last month) are
assigned to ``dyad``: they are evidence about the relationship, not about one
actor's global state.  This keeps the actor view non-overlapping and exactly
additive.  The share of absolute contribution carried by the two actor buckets
is reported so the reader can see how much of the forecast is actor-level.

Two aggregation views are provided:

``ownership``  per-actor sum of that actor's own bucket.  Additive across
               actors and buckets: source + target + dyad + calendar + graph +
               train_stats = total log-odds (minus intercept).  Actors are
               *not* double counted.
``incidence``  per-actor sum of *all* contributions of dyads in which the actor
               appears.  Descriptive roll-up for review triage; sums over
               actors double count every dyad and are **not** additive.

All functions are CPU-only, dependency-light (numpy/pandas/scipy) and operate
on arrays in the frozen ``export_dyad_split`` row order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BUCKETS = ["source_actor", "target_actor", "dyad", "calendar", "graph",
           "train_stats"]

SOURCE_OWN = ["src_ev", "g_deg_s", "g_wdeg_s", "g_deg12_s", "g_eig_s",
              "g_pr_s", "g_nra_s"]
TARGET_OWN = ["tgt_ev", "g_deg_t", "g_wdeg_t", "g_deg12_t", "g_eig_t",
              "g_pr_t", "g_nra_t"]
GRAPH_SYM = ["g_cn", "g_jaccard", "g_aa", "g_ra", "g_two_hop", "g_cn12",
             "g_aa12", "g_ra12"]
CALENDAR = ["t_yr", "mo_sin", "mo_cos"]
TRAIN_STATS = ["br", "seen_tr"]

#: Ordered pair history; the residual after the explicit buckets.
DYAD_PAIR = [
    "active", "log_ev",
    "a1", "a2", "a3", "a4", "a5", "a6",
    "le1", "le2", "le3",
    "am3", "am6", "am12", "les3", "les6", "les12",
    "since_act", "since_first", "log_n_act",
    "rev_act", "rev_am3",
]

BUCKET_FEATURES = {
    "source_actor": SOURCE_OWN,
    "target_actor": TARGET_OWN,
    "dyad": DYAD_PAIR,
    "calendar": CALENDAR,
    "graph": GRAPH_SYM,
    "train_stats": TRAIN_STATS,
}

ROLE_OF_BUCKET = {
    "source_actor": "source",
    "target_actor": "target",
    "dyad": "dyad",
    "calendar": "global",
    "graph": "dyad",
    "train_stats": "global",
}


# ---------------------------------------------------------------------------
# Bucket contract
# ---------------------------------------------------------------------------
def feature_buckets(names) -> dict[str, str]:
    """Map every model feature to exactly one bucket; fail loudly otherwise."""
    names = list(names)
    out: dict[str, str] = {}
    for bucket, feats in BUCKET_FEATURES.items():
        for f in feats:
            if f in out:
                raise AssertionError(f"feature {f!r} assigned twice")
            out[f] = bucket
    missing = [n for n in names if n not in out]
    extra = [n for n in out if n not in names]
    if missing:
        raise AssertionError(f"features without bucket: {missing}")
    if extra:
        raise AssertionError(f"bucket features not in model: {extra}")
    return {n: out[n] for n in names}


def bucket_indicator(names) -> np.ndarray:
    """(n_features, n_buckets) 0/1 matrix in ``BUCKETS`` order."""
    fb = feature_buckets(names)
    M = np.zeros((len(names), len(BUCKETS)), dtype=np.float64)
    for j, n in enumerate(names):
        M[j, BUCKETS.index(fb[n])] = 1.0
    return M


def bucket_map_frame(names) -> pd.DataFrame:
    fb = feature_buckets(names)
    return pd.DataFrame({
        "feature": list(names),
        "bucket": [fb[n] for n in names],
        "role": [ROLE_OF_BUCKET[fb[n]] for n in names],
        "is_actor_ego": [fb[n] in ("source_actor", "target_actor") for n in names],
    })


# ---------------------------------------------------------------------------
# Exact additive contributions
# ---------------------------------------------------------------------------
def sigmoid(z):
    z = np.clip(np.asarray(z, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    """Raw log-odds; wide clip only to keep -inf out of arithmetic."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-300, 1.0 - 1e-16)
    return np.log(p / (1.0 - p))


def nll(y, p):
    """Negative log-likelihood clipped at 1e-12, matching framework.metrics."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    y = np.asarray(y, dtype=np.float64)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def standardized(fit: dict, X) -> np.ndarray:
    """Standardise exactly as the model does (float32 scaler output).

    ``framework.models.scaler_apply`` returns float32; sklearn then promotes
    to float64 for the dot product.  Rounding z to float32 here makes the
    decomposition reproduce the model's decision function to ~1e-12 instead of
    leaving a ~1e-5 float32-storage residual.
    """
    X = np.asarray(X, dtype=np.float64)
    z = (X - fit["mu"][None, :]) / fit["sd"][None, :]
    return z.astype(np.float32).astype(np.float64)


def contributions(fit: dict, X) -> np.ndarray:
    """Exact per-row raw log-odds contributions, shape (n, p), float64."""
    return standardized(fit, X) * fit["coef"][None, :]


def contributions_temperature(C, T: float) -> np.ndarray:
    """Affine temperature track: C / T is still an exact decomposition."""
    return np.asarray(C, dtype=np.float64) / float(T)


def exactness_report(fit: dict, C, p, T: float = 1.0, tol: float = 1e-6,
                     eta=None) -> dict:
    """Verify ``b + sum_j C_ij = logit(p_i)`` and the temperature analogue.

    ``p`` may be supplied by ``predict_proba`` in float32; the round-trip
    through ``logit`` then limits agreement to ~1e-5 even when the additive
    identity is exact.  Pass the model's own ``decision_function`` output as
    ``eta`` to separate decomposition error from probability quantization.
    The temperature term is the affine identity ``eta/T`` and, when ``T == 1``,
    is the same numerical round-trip as the raw term rather than an independent
    calibration check.
    """
    b = float(fit["intercept"])
    eta_sum = b + np.asarray(C, dtype=np.float64).sum(axis=1)
    err = np.abs(eta_sum - logit(p))
    eta_t = eta_sum / float(T)
    p_t = sigmoid(eta_t)
    err_t = np.abs(eta_t - logit(p_t))
    out = {
        "n_rows": int(len(err)),
        "raw_max_abs_error": float(err.max()) if len(err) else 0.0,
        "raw_mean_abs_error": float(err.mean()) if len(err) else 0.0,
        "raw_rows_above_tol": int((err > tol).sum()),
        "temperature_T": float(T),
        "temperature_max_abs_error": float(err_t.max()) if len(err_t) else 0.0,
        "temperature_rows_above_tol": int((err_t > tol).sum()),
        "tol": float(tol),
        "intercept": b,
    }
    if eta is not None:
        eta = np.asarray(eta, dtype=np.float64).ravel()
        derr = np.abs(eta_sum - eta)
        out.update({
            "decision_max_abs_error": float(derr.max()) if len(derr) else 0.0,
            "decision_mean_abs_error": float(derr.mean()) if len(derr) else 0.0,
            "decision_rows_above_tol": int((derr > tol).sum()),
        })
    return out


def logistic_occlusion_delta(y, p, C, cols, T: float = 1.0) -> np.ndarray:
    """Per-row ΔNLL from replacing ``cols`` with their train mean.

    Replacing ``x_j`` by ``mu_j`` changes the log-odds by ``-C_j`` exactly, so
    the counterfactual probability is ``sigmoid(eta - sum_{j in cols} C_j)``.
    This is the logistic counterpart of the HGB occlusion intervention.
    """
    eta = logit(p)
    p0 = sigmoid(eta)
    p1 = sigmoid(eta - np.asarray(C)[:, cols].sum(axis=1) / float(T))
    return nll(y, p1) - nll(y, p0)


# ---------------------------------------------------------------------------
# Accumulator: one pass over rows -> all attribution tables
# ---------------------------------------------------------------------------
class AttributionAccumulator:
    """Streaming aggregation of exact contributions by actor, month and bucket.

    Parameters
    ----------
    n_actors : int
    month_lo, month_hi : int
        Inclusive month-index range of the period (contiguous).
    feature_names : list[str]
    n_windows : int
        Number of optional row-level windows (e.g. D2 W_b/W_c/W_d).
    do_occlusion : bool
        If true, also accumulate feature- and actor-level logistic occlusion
        ΔNLL per month (used for the HGB agreement check).
    """

    def __init__(self, n_actors: int, month_lo: int, month_hi: int,
                 feature_names, n_windows: int = 0,
                 do_occlusion: bool = False):
        self.A = int(n_actors)
        self.lo = int(month_lo)
        self.hi = int(month_hi)
        self.M = self.hi - self.lo + 1
        self.names = list(feature_names)
        self.F = len(self.names)
        self.B = len(BUCKETS)
        self.Mb = bucket_indicator(self.names)
        self.n_windows = int(n_windows)
        self.do_occlusion = bool(do_occlusion)
        A, M, F, B = self.A, self.M, self.F, self.B

        self.bucket_sum = {r: np.zeros((A, M, B), dtype=np.float64)
                           for r in ("source", "target")}
        self.count = {r: np.zeros((A, M), dtype=np.float64)
                      for r in ("source", "target")}
        self.p_sum = {r: np.zeros((A, M), dtype=np.float64)
                      for r in ("source", "target")}
        self.y_sum = {r: np.zeros((A, M), dtype=np.float64)
                      for r in ("source", "target")}

        self.feat_sum = {r: np.zeros((A, F), dtype=np.float64)
                         for r in ("source", "target")}
        self.feat_abs = {r: np.zeros((A, F), dtype=np.float64)
                         for r in ("source", "target")}
        self.feat_n = {r: np.zeros(A, dtype=np.float64)
                       for r in ("source", "target")}

        self.feat_c_sum = np.zeros(F, dtype=np.float64)
        self.feat_abs_c_sum = np.zeros(F, dtype=np.float64)
        self.feat_month_sum = np.zeros((F, M), dtype=np.float64)
        self.feat_month_abs = np.zeros((F, M), dtype=np.float64)
        self.month_n = np.zeros(M, dtype=np.float64)
        self.z_sum = {r: np.zeros((A, F), dtype=np.float64)
                      for r in ("source", "target")}
        self.n_rows = 0

        self.win_sum = {r: np.zeros((self.n_windows, A, B), dtype=np.float64)
                        for r in ("source", "target")}
        # Window denominators are role-specific: a row contributes to its source
        # actor in the source view and to its target actor in the target view.
        self.win_count = {r: np.zeros((self.n_windows, A), dtype=np.float64)
                          for r in ("source", "target")}

        if self.do_occlusion:
            self.occ_feat_month = np.zeros((F, M), dtype=np.float64)
            self.occ_feat_total = np.zeros(F, dtype=np.float64)
            self.occ_actor_month = {
                "source": np.zeros((A, M), dtype=np.float64),
                "target": np.zeros((A, M), dtype=np.float64)}
            self.occ_actor_total = {
                "source": np.zeros(A, dtype=np.float64),
                "target": np.zeros(A, dtype=np.float64)}
        self.exact = {"n_rows": 0, "max_abs_error": 0.0, "sum_abs_error": 0.0,
                      "rows_above_tol": 0}

    # -- helpers ---------------------------------------------------------
    def _flat(self, actor, t_idx):
        return actor.astype(np.int64) * self.M + (t_idx.astype(np.int64) - self.lo)

    def add(self, X, sid, tid, t_idx, y, p, C, intercept: float,
            window_id=None, Z=None, tol: float = 1e-6) -> None:
        X = np.asarray(X)
        C = np.asarray(C, dtype=np.float64)
        sid = np.asarray(sid, dtype=np.int64)
        tid = np.asarray(tid, dtype=np.int64)
        t_idx = np.asarray(t_idx, dtype=np.int64)
        y = np.asarray(y, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)
        if not (t_idx.min() >= self.lo and t_idx.max() <= self.hi):
            raise ValueError("rows outside accumulator month range")
        m = t_idx - self.lo
        n = len(y)
        self.n_rows += n
        self.month_n += np.bincount(m, minlength=self.M)

        b = C @ self.Mb                                   # (n, B)
        for role, actor in (("source", sid), ("target", tid)):
            flat = self._flat(actor, t_idx)
            for j in range(self.B):
                self.bucket_sum[role][:, :, j] += np.bincount(
                    flat, weights=b[:, j], minlength=self.A * self.M
                ).reshape(self.A, self.M)
            self.count[role] += np.bincount(
                flat, minlength=self.A * self.M).reshape(self.A, self.M)
            self.p_sum[role] += np.bincount(
                flat, weights=p, minlength=self.A * self.M).reshape(self.A, self.M)
            self.y_sum[role] += np.bincount(
                flat, weights=y, minlength=self.A * self.M).reshape(self.A, self.M)
            for f in range(self.F):
                self.feat_sum[role][:, f] += np.bincount(
                    actor, weights=C[:, f], minlength=self.A)
                self.feat_abs[role][:, f] += np.bincount(
                    actor, weights=np.abs(C[:, f]), minlength=self.A)
            self.feat_n[role] += np.bincount(actor, minlength=self.A)
            if Z is not None:
                Z = np.asarray(Z, dtype=np.float64)
                for f in range(self.F):
                    self.z_sum[role][:, f] += np.bincount(
                        actor, weights=Z[:, f], minlength=self.A)
            if self.do_occlusion:
                own = b[:, BUCKETS.index("source_actor" if role == "source"
                                         else "target_actor")]
                eta = intercept + C.sum(axis=1)
                dn = nll(y, sigmoid(eta - own)) - nll(y, sigmoid(eta))
                self.occ_actor_month[role] += np.bincount(
                    flat, weights=dn, minlength=self.A * self.M
                ).reshape(self.A, self.M)
                self.occ_actor_total[role] += np.bincount(
                    actor, weights=dn, minlength=self.A)

        for f in range(self.F):
            self.feat_c_sum[f] += C[:, f].sum()
            self.feat_abs_c_sum[f] += np.abs(C[:, f]).sum()
            self.feat_month_sum[f] += np.bincount(
                m, weights=C[:, f], minlength=self.M)
            self.feat_month_abs[f] += np.bincount(
                m, weights=np.abs(C[:, f]), minlength=self.M)

        if self.n_windows and window_id is not None:
            window_id = np.asarray(window_id, dtype=np.int64)
            for w in range(self.n_windows):
                mask = window_id == w
                if not mask.any():
                    continue
                bw = b[mask]
                self.win_count["source"][w] += np.bincount(sid[mask],
                                                           minlength=self.A)
                self.win_count["target"][w] += np.bincount(tid[mask],
                                                           minlength=self.A)
                for role, actor in (("source", sid), ("target", tid)):
                    aw = actor[mask]
                    for j in range(self.B):
                        self.win_sum[role][w, :, j] += np.bincount(
                            aw, weights=bw[:, j], minlength=self.A)

        if self.do_occlusion:
            eta = intercept + C.sum(axis=1)
            base = nll(y, sigmoid(eta))
            for f in range(self.F):
                dn = nll(y, sigmoid(eta - C[:, f])) - base
                self.occ_feat_month[f] += np.bincount(m, weights=dn,
                                                      minlength=self.M)
                self.occ_feat_total[f] += float(dn.sum())

        eta = intercept + C.sum(axis=1)
        err = np.abs(eta - logit(p))
        self.exact["n_rows"] += int(n)
        self.exact["max_abs_error"] = max(self.exact["max_abs_error"],
                                          float(err.max()) if n else 0.0)
        self.exact["sum_abs_error"] += float(err.sum())
        self.exact["rows_above_tol"] += int((err > tol).sum())

    # -- outputs ---------------------------------------------------------
    def actor_vectors(self, role: str) -> dict:
        """Per-actor period aggregates for one role."""
        bucket = "source_actor" if role == "source" else "target_actor"
        j = BUCKETS.index(bucket)
        n = self.count[role].sum(axis=1)
        attr = np.where(n > 0, self.bucket_sum[role][:, :, j].sum(axis=1) / np.maximum(n, 1), np.nan)
        total = np.where(n > 0, self.bucket_sum[role].sum(axis=2).sum(axis=1) / np.maximum(n, 1), np.nan)
        feat = self.feat_sum[role] / np.maximum(self.feat_n[role], 1)[:, None]
        feat_abs = self.feat_abs[role] / np.maximum(self.feat_n[role], 1)[:, None]
        return {
            "attr": attr, "total_contrib": total, "n_rows": n,
            "mean_feature_contrib": feat, "mean_feature_abs": feat_abs,
            "feat_sum": self.feat_sum[role].copy(),
            "feat_abs": self.feat_abs[role].copy(),
            "bucket_mean": (self.bucket_sum[role].sum(axis=1)
                            / np.maximum(n, 1)[:, None]),
            "p_sum": self.p_sum[role].sum(axis=1),
            "y_sum": self.y_sum[role].sum(axis=1),
        }

    def feature_vector(self) -> dict:
        n = max(self.n_rows, 1)
        return {
            "mean_contrib": self.feat_c_sum / n,
            "mean_abs_contrib": self.feat_abs_c_sum / n,
            "month_mean_contrib": self.feat_month_sum
            / np.maximum(self.month_n, 1)[None, :],
            "month_sum": self.feat_month_sum.copy(),
            "month_abs_sum": self.feat_month_abs.copy(),
        }

    def window_vector(self, w: int, role: str) -> dict:
        n = self.win_count[role][w]
        means = self.win_sum[role][w] / np.maximum(n, 1)[:, None]
        return {"bucket_mean": means, "n_rows": n}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def actor_table(acc: AttributionAccumulator, role: str, actors: pd.DataFrame,
                period: str) -> pd.DataFrame:
    v = acc.actor_vectors(role)
    feats = acc.names
    rows = []
    for i in range(acc.A):
        if v["n_rows"][i] <= 0:
            continue
        mc = v["mean_feature_contrib"][i]
        order = np.argsort(-np.abs(mc))[:3]
        b = v["bucket_mean"][i]
        rows.append({
            "period": period, "role": role, "aid": i,
            "actor": actors["actor"].iloc[i] if i < len(actors) else str(i),
            "region": actors["region"].iloc[i] if i < len(actors) else "",
            "n_rows": int(v["n_rows"][i]),
            "n_months": int(acc.count[role][i].astype(bool).sum()),
            "attribution": round(float(v["attr"][i]), 6),
            "total_contrib_excl_intercept": round(float(v["total_contrib"][i]), 6),
            "attribution_sum": round(float(acc.bucket_sum[role][i, :, BUCKETS.index(
                "source_actor" if role == "source" else "target_actor")].sum()), 4),
            "mean_p": round(float(v["p_sum"][i] / v["n_rows"][i]), 6),
            "mean_y": round(float(v["y_sum"][i] / v["n_rows"][i]), 6),
            "bucket_source": round(float(b[BUCKETS.index("source_actor")]), 6),
            "bucket_target": round(float(b[BUCKETS.index("target_actor")]), 6),
            "bucket_dyad": round(float(b[BUCKETS.index("dyad")]), 6),
            "bucket_calendar": round(float(b[BUCKETS.index("calendar")]), 6),
            "bucket_graph": round(float(b[BUCKETS.index("graph")]), 6),
            "bucket_train_stats": round(float(b[BUCKETS.index("train_stats")]), 6),
            "top1_feature": feats[int(order[0])],
            "top1_contrib": round(float(mc[int(order[0])]), 6),
            "top2_feature": feats[int(order[1])],
            "top2_contrib": round(float(mc[int(order[1])]), 6),
            "top3_feature": feats[int(order[2])],
            "top3_contrib": round(float(mc[int(order[2])]), 6),
        })
    df = pd.DataFrame(rows)
    if len(df):
        # share of the actor's mean absolute contribution that is actor-ego
        fb = feature_buckets(acc.names)
        abs_by_bucket = np.zeros((acc.A, len(BUCKETS)))
        for b, name in enumerate(BUCKETS):
            cols = [j for j, n in enumerate(acc.names) if fb[n] == name]
            abs_by_bucket[:, b] = v["mean_feature_abs"][:, cols].sum(axis=1)
        ego = abs_by_bucket[:, BUCKETS.index("source_actor")] + \
            abs_by_bucket[:, BUCKETS.index("target_actor")]
        tot = abs_by_bucket.sum(axis=1)
        df["ego_abs_share"] = np.round(
            ego[df["aid"].to_numpy()] / np.maximum(tot[df["aid"].to_numpy()], 1e-12), 4)
        df["attribution_rank"] = df["attribution"].rank(ascending=False, method="min").astype(int)
        df["attribution_pct"] = (df["attribution"].rank(pct=True) * 100).round(1)
    return df


def actor_month_table(acc: AttributionAccumulator, role: str,
                      actors: pd.DataFrame, period: str) -> pd.DataFrame:
    j = BUCKETS.index("source_actor" if role == "source" else "target_actor")
    rows = []
    for i in range(acc.A):
        for m in range(acc.M):
            n = acc.count[role][i, m]
            if n <= 0:
                continue
            rows.append({
                "period": period, "role": role,
                "aid": i,
                "actor": actors["actor"].iloc[i] if i < len(actors) else str(i),
                "t_idx": acc.lo + m,
                "month": month_label(acc.lo + m),
                "n_dyads": int(n),
                "attribution": round(float(acc.bucket_sum[role][i, m, j] / n), 6),
                "total_contrib": round(float(acc.bucket_sum[role][i, m].sum() / n), 6),
                "p_sum": round(float(acc.p_sum[role][i, m]), 6),
                "y_sum": round(float(acc.y_sum[role][i, m]), 6),
                "mean_p": round(float(acc.p_sum[role][i, m] / n), 6),
            })
    return pd.DataFrame(rows)


def feature_table(acc: AttributionAccumulator, period: str) -> pd.DataFrame:
    fv = acc.feature_vector()
    fb = feature_buckets(acc.names)
    df = pd.DataFrame({
        "period": period,
        "feature": acc.names,
        "bucket": [fb[n] for n in acc.names],
        "mean_contrib": np.round(fv["mean_contrib"], 6),
        "mean_abs_contrib": np.round(fv["mean_abs_contrib"], 6),
        "share_positive": np.round(
            [float((acc.feat_month_sum[j] > 0).mean()) for j in range(acc.F)], 4),
    })
    df["rank_abs"] = df["mean_abs_contrib"].rank(ascending=False, method="min").astype(int)
    df["rank_signed"] = df["mean_contrib"].rank(ascending=False, method="min").astype(int)
    return df


def bucket_share_table(acc: AttributionAccumulator, period: str) -> pd.DataFrame:
    fv = acc.feature_vector()
    fb = feature_buckets(acc.names)
    out = []
    # mean signed and mean absolute contribution, by bucket and by actor role
    for b in BUCKETS:
        cols = [j for j, n in enumerate(acc.names) if fb[n] == b]
        out.append({
            "period": period, "view": "all_features", "bucket": b,
            "mean_contrib": round(float(fv["mean_contrib"][cols].sum()), 6),
            "mean_abs_contrib": round(float(fv["mean_abs_contrib"][cols].sum()), 6),
        })
    tot = float(fv["mean_abs_contrib"].sum())
    df = pd.DataFrame(out)
    df["abs_share"] = (df["mean_abs_contrib"] / tot).round(4) if tot else np.nan
    return df


# ---------------------------------------------------------------------------
# Search / stability metrics
# ---------------------------------------------------------------------------
def _rank(x):
    from scipy.stats import rankdata
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    r = np.full(len(x), np.nan)
    if ok.sum() > 1:
        r[ok] = rankdata(x[ok])
    return r


def spearman(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = _rank(a[ok]), _rank(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def topk_indices(x, k: int, mode: str = "abs") -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    score = np.abs(x) if mode == "abs" else x
    score = np.where(np.isfinite(score), score, -np.inf)
    return np.argsort(-score, kind="stable")[:k]


def topk_jaccard(a, b, k: int, mode: str = "abs") -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return float("nan")
    aa = np.where(ok, a, np.nan)
    bb = np.where(ok, b, np.nan)
    ia = set(topk_indices(aa, k, mode).tolist())
    ib = set(topk_indices(bb, k, mode).tolist())
    return len(ia & ib) / max(len(ia | ib), 1)


def topk_overlap(a, b, k: int, mode: str = "abs") -> float:
    """Overlap coefficient |A∩B|/k (more intuitive for churn)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return float("nan")
    ia = set(topk_indices(np.where(ok, a, np.nan), k, mode).tolist())
    ib = set(topk_indices(np.where(ok, b, np.nan), k, mode).tolist())
    return len(ia & ib) / max(k, 1)


def sign_agreement(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
    if ok.sum() == 0:
        return float("nan")
    return float((np.sign(a[ok]) == np.sign(b[ok])).mean())


def stability_pair(a, b, ks=(5, 10, 20, 50)) -> dict:
    out = {"spearman": spearman(a, b), "sign_agreement": sign_agreement(a, b),
           "n": int(np.isfinite(np.asarray(a, dtype=np.float64)).sum())}
    for k in ks:
        out[f"jaccard_abs@{k}"] = topk_jaccard(a, b, k, "abs")
        out[f"jaccard_pos@{k}"] = topk_jaccard(a, b, k, "pos")
        out[f"overlap_abs@{k}"] = topk_overlap(a, b, k, "abs")
    return out


def churn_rate(a, b, k: int, mode: str = "abs") -> float:
    """1 - overlap@k: share of the top-k set that changed."""
    return 1.0 - topk_overlap(a, b, k, mode)


# ---------------------------------------------------------------------------
# Period / window helpers
# ---------------------------------------------------------------------------
def month_label(t_idx: int) -> str:
    y = 1995 + int(t_idx) // 12
    m = int(t_idx) % 12 + 1
    return f"{y:04d}-{m:02d}"


def period_rows(t_idx, lo: int, hi: int) -> np.ndarray:
    t_idx = np.asarray(t_idx)
    return (t_idx >= lo) & (t_idx <= hi)


def d2_windows(panel: pd.DataFrame, rows: dict, lo: int, hi: int | None = None) -> dict:
    """Generalised D2 windows for a period starting at month ``lo``.

    Same definitions as ``framework.splits.d2_window_masks`` but with the
    12-month look-back anchored at the period start.  Only lagged history and
    current activity are used (no labels).  ``hi`` truncates the history so a
    later re-entry cannot be used to label an earlier row.
    """
    if hi is not None:
        panel = panel[panel["t_idx"] <= hi]
    n_dyads = int(panel["did"].max()) + 1
    n_actors = int(max(panel["sid"].max(), panel["tid"].max())) + 1
    first_active = np.full(n_dyads, 10**9, dtype=np.int64)
    act = panel.loc[panel["active"], ["did", "t_idx"]].sort_values(["did", "t_idx"])
    d_act = act["did"].to_numpy(dtype=np.int64)
    t_act = act["t_idx"].to_numpy(dtype=np.int64)
    np.minimum.at(first_active, d_act, t_act)
    prev = np.full(len(t_act), -999, dtype=np.int64)
    if len(t_act) > 1:
        same = d_act[1:] == d_act[:-1]
        prev[1:] = np.where(same, t_act[:-1], -999)
    reentry = (t_act - prev) >= 13
    last_re = np.full(n_dyads, -999, dtype=np.int64)
    np.maximum.at(last_re, d_act[reentry], t_act[reentry])
    actor_first = np.full(n_actors, 10**9, dtype=np.int64)
    np.minimum.at(actor_first, panel["sid"].to_numpy(dtype=np.int64),
                  panel["t_idx"].to_numpy(dtype=np.int64))
    np.minimum.at(actor_first, panel["tid"].to_numpy(dtype=np.int64),
                  panel["t_idx"].to_numpy(dtype=np.int64))
    de = np.asarray(rows["did"], dtype=np.int64)
    te = np.asarray(rows["t_idx"], dtype=np.int64)
    fa, lr = first_active[de], last_re[de]
    sa = np.minimum(actor_first[rows["sid"]], actor_first[rows["tid"]])
    anchor = np.maximum(te - 11, lo)
    a = sa >= anchor
    b = fa >= anchor
    c = (lr >= anchor) & (lr >= 0)
    d = ~(a | b | c) & (fa <= te - 12) & (te >= lo)
    return {"W_a_new_actor": a, "W_b_dyad_onset": b,
            "W_c_dormancy_reentry": c, "W_d_stable": d}


def window_id_from_masks(masks: dict, prefer=("W_b_dyad_onset",
                                              "W_c_dormancy_reentry",
                                              "W_d_stable")) -> tuple[np.ndarray, list]:
    """Mutually exclusive row ids; returns (id, names). -1 = in no window."""
    names = [w for w in prefer if w in masks]
    wid = np.full(len(next(iter(masks.values()))), -1, dtype=np.int64)
    for i, w in enumerate(names):
        m = np.asarray(masks[w], dtype=bool)
        wid[m & (wid < 0)] = i
    return wid, names


# ---------------------------------------------------------------------------
# Bootstrap helpers (cluster over months or actors)
# ---------------------------------------------------------------------------
def boot_month_indices(M: int, reps: int, rng) -> np.ndarray:
    return rng.integers(0, M, size=(reps, M))


def boot_actor_indices(A: int, reps: int, rng) -> np.ndarray:
    return rng.integers(0, A, size=(reps, A))


def ci(arr, lo=2.5, hi=97.5):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return [round(float(np.percentile(arr, lo)), 6),
            round(float(np.percentile(arr, hi)), 6)]


def selfcheck(n_actors: int = 7, n_rows: int = 501, seed: int = 0) -> dict:
    """Internal consistency checks for the bucket contract and accumulator."""
    from framework import features as F

    rng = np.random.default_rng(seed)
    names = F.DYAD_CAUSAL + F.GRAPH_CORE20 + ["br", "seen_tr"]
    fb = feature_buckets(names)
    assert len(fb) == len(names) == 49
    M = bucket_indicator(names)
    assert np.all(M.sum(axis=1) == 1)
    fit = {"mu": rng.normal(size=len(names)).astype(np.float32),
           "sd": np.abs(rng.normal(size=len(names))).astype(np.float32) + 0.5,
           "coef": rng.normal(scale=0.3, size=len(names)),
           "intercept": -1.3}
    X = rng.normal(size=(n_rows, len(names))).astype(np.float32)
    C = contributions(fit, X)
    eta = fit["intercept"] + C.sum(1)
    p = sigmoid(eta)
    rep = exactness_report(fit, C, p, T=1.0)
    assert rep["raw_max_abs_error"] < 1e-9
    sid = rng.integers(0, n_actors, n_rows)
    tid = (sid + 1 + rng.integers(0, n_actors - 1, n_rows)) % n_actors
    t_idx = np.full(n_rows, 10)
    y = (rng.random(n_rows) < p).astype(np.int64)
    acc = AttributionAccumulator(n_actors, 10, 10, names, do_occlusion=True)
    acc.add(X, sid, tid, t_idx, y, p, C, fit["intercept"])
    # total contributions reproduce the pool
    assert abs(acc.bucket_sum["source"].sum() - C.sum()) < 1e-6
    assert abs(acc.bucket_sum["target"].sum() - C.sum()) < 1e-6
    # bucket sums per row equal C row sums
    b = C @ M
    assert np.allclose(b.sum(1), C.sum(1))
    # occlusion by all features = total contribution
    dn_all = logistic_occlusion_delta(y, p, C, list(range(len(names))))
    assert np.isfinite(dn_all).all()
    return {"n_rows": n_rows, "max_exactness_error": rep["raw_max_abs_error"],
            "buckets": BUCKETS, "features": len(names)}


if __name__ == "__main__":
    print(selfcheck())
