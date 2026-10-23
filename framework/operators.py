"""Five-class lifecycle operator head (synthetic-trained, diagnostic only).

The head is deliberately split into two transparent pieces:

1. ``P(event)`` -- a binary hazard on actor-month features.
2. ``P(type | event)`` -- a 5-class multinomial logistic over
   ``birth / death / merge / split / reentry``.

The joint ``P(type) = P(event) P(type | event)`` is reported for synthetic
worlds.  On real ICEWS the operator labels do not exist (0 clean
birth/death/merge/split events in the frozen test), so the head is **applied as
a diagnostic only**; it is not a causal or validated lifecycle claim.

The reference baselines are a majority-class look-up and a roster-composition
look-up (the SALS v2 triviality), so the head must beat both on synthetic test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import ACTOR_FEATURES
from .metrics import binary_metrics, top1_metrics
from .models import predict_logistic, scaler_apply, scaler_fit
from .synth import OPS

ROSTER_FEATURES = [
    "r_n_active", "r_fill", "r_mean_partners", "r_max_partners",
    "r_dormant_frac", "r_mean_events", "r_active_rate",
]
OPERATOR_FEATURES = ACTOR_FEATURES + ROSTER_FEATURES


def build_roster_context(con, cfg: dict) -> None:
    """Create ``roster_ctx`` keyed by month from ``actor_month``."""
    con.execute("""
      CREATE OR REPLACE TABLE roster_ctx AS
      SELECT m, t_idx,
        sum(out_active::INT)::DOUBLE AS r_n_active,
        avg(out_active::INT)::DOUBLE AS r_fill,
        avg(out_partners)::DOUBLE AS r_mean_partners,
        max(out_partners)::DOUBLE AS r_max_partners,
        avg((NOT out_active AND ever)::INT)::DOUBLE AS r_dormant_frac,
        avg(out_events)::DOUBLE AS r_mean_events,
        avg((out_events > 0)::INT)::DOUBLE AS r_active_rate
      FROM (
        SELECT *, max(out_active::INT) OVER (
          PARTITION BY actor ORDER BY t_idx
          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS ever
        FROM actor_month)
      GROUP BY m, t_idx
    """)


def export_operator_split(con, split: str, has_labels: bool) -> dict:
    """Export operator-head rows for one split.

    ``has_labels`` must be true only for synthetic panels; the function asserts
    that real data never silently receives a label.
    """
    cols = ", ".join(f"coalesce(a.{c}::DOUBLE, 0.0) AS {c}" for c in ACTOR_FEATURES)
    rcols = ", ".join(f"coalesce(r.{c}, 0.0) AS {c}" for c in ROSTER_FEATURES)
    op_expr = "coalesce(am.op_next, 0)::INT AS op_next" if has_labels else "0::INT AS op_next"
    src = "ab a JOIN actor_month am ON am.actor=a.actor AND am.t_idx=a.t_idx" if has_labels else "ab a"
    res = con.execute(f"""
      SELECT {cols}, {rcols}, {op_expr}, a.actor, a.aid, a.region, a.t_idx
      FROM {src}
      JOIN roster_ctx r ON r.t_idx = a.t_idx
      WHERE a.split='{split}' AND a.out_active_next IS NOT NULL
      ORDER BY a.aid, a.t_idx
    """).fetchnumpy()
    X = np.column_stack([res[c].astype(np.float32) for c in OPERATOR_FEATURES])
    return {
        "X": X, "feature_names": list(OPERATOR_FEATURES),
        "y_op": res["op_next"].astype(np.int64),
        "actor": res["actor"], "aid": res["aid"].astype(np.int16),
        "region": res["region"], "t_idx": res["t_idx"].astype(np.int16),
    }


class OperatorHead:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mu = None
        self.sd = None
        self.hazard = None
        self.type_model = None
        self.C_hazard = None
        self.C_type = None
        self.history = {}

    def fit(self, dtr: dict, dsel: dict) -> dict:
        from sklearn.linear_model import LogisticRegression

        Xtr, ytr = dtr["X"], dtr["y_op"]
        Xsel, ysel = dsel["X"], dsel["y_op"]
        self.mu, self.sd = scaler_fit(Xtr)
        Xtr_s = scaler_apply(Xtr, self.mu, self.sd)
        Xsel_s = scaler_apply(Xsel, self.mu, self.sd)
        ev_tr, ev_sel = ytr > 0, ysel > 0
        self.history["support"] = {
            "train_rows": int(len(ytr)), "train_events": int(ev_tr.sum()),
            "select_rows": int(len(ysel)), "select_events": int(ev_sel.sum()),
        }

        # Binary event hazard.
        best = None
        for C in self.cfg["operator_head"]["C_grid"]:
            clf = LogisticRegression(C=float(C), max_iter=600, solver="lbfgs",
                                     class_weight="balanced",
                                     random_state=int(self.cfg["seed"]))
            clf.fit(Xtr_s, ev_tr.astype(int))
            p = clf.predict_proba(Xsel_s)[:, 1]
            ll = binary_metrics(ev_sel.astype(int), p)["log_loss"]
            if best is None or ll < best[0]:
                best = (ll, float(C), clf)
        self.hazard_ll, self.C_hazard, self.hazard = best

        # 5-class conditional type.
        best = None
        for C in self.cfg["operator_head"]["C_grid"]:
            clf = LogisticRegression(C=float(C), max_iter=600, solver="lbfgs",
                                     class_weight="balanced",
                                     random_state=int(self.cfg["seed"]))
            clf.fit(Xtr_s[ev_tr], ytr[ev_tr] - 1)
            P = clf.predict_proba(Xsel_s[ev_sel])
            ll = top1_metrics(ysel[ev_sel] - 1, P)["log_loss"]
            if best is None or ll < best[0]:
                best = (ll, float(C), clf)
        self.type_ll, self.C_type, self.type_model = best
        return self.history

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Xs = scaler_apply(X, self.mu, self.sd)
        p_event = self.hazard.predict_proba(Xs)[:, 1]
        p_type = self.type_model.predict_proba(Xs)  # conditional on event
        joint = p_event[:, None] * p_type
        return p_event, p_type, joint

    def coefficients(self, top_k: int = 5) -> dict:
        out = {}
        coef = self.type_model.coef_  # [5, F]
        for k, name in enumerate(OPS):
            order = np.argsort(-np.abs(coef[k]))[:top_k]
            out[name] = [
                {"feature": OPERATOR_FEATURES[j], "coef": round(float(coef[k, j]), 5)}
                for j in order
            ]
        out["_hazard"] = [
            {"feature": OPERATOR_FEATURES[j], "coef": round(float(self.hazard.coef_[0, j]), 5)}
            for j in np.argsort(-np.abs(self.hazard.coef_[0]))[:top_k]
        ]
        return out


def lookup_type_baseline(dtr: dict, dsel: dict, bins: int = 4) -> dict:
    """Roster-composition look-up: bin by n_active x dormant_frac."""
    def key(d):
        a = pd.qcut(d["r_n_active"], bins, labels=False, duplicates="drop").astype(int)
        b = pd.qcut(d["r_dormant_frac"], bins, labels=False, duplicates="drop").astype(int)
        return a * bins + b

    ev = dtr["y_op"] > 0
    ktr = key(dtr)[ev]
    ytr = dtr["y_op"][ev]
    table = {}
    for k in np.unique(ktr):
        vals, counts = np.unique(ytr[ktr == k], return_counts=True)
        table[int(k)] = int(vals[np.argmax(counts)])
    majority = int(np.bincount(ytr).argmax())
    ksel = key(dsel)
    pred = np.array([table.get(int(k), majority) for k in ksel])
    ev_sel = dsel["y_op"] > 0
    P = np.zeros((len(pred), 5))
    P[np.arange(len(pred)), pred - 1] = 1.0
    return {"type_lookup": top1_metrics(dsel["y_op"][ev_sel] - 1, P[ev_sel]),
            "majority_class": majority}


def label_shuffle_control(dtr: dict, dsel: dict, cfg: dict, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d2 = dict(dtr)
    y = dtr["y_op"].copy()
    ev = np.where(y > 0)[0]
    y[ev] = rng.permutation(y[ev])
    d2["y_op"] = y
    head = OperatorHead(cfg)
    head.fit(d2, dsel)
    _, _, joint = head.predict(dsel["X"])
    ysel = dsel["y_op"]
    return {
        "joint_top1_accuracy": round(float((joint.argmax(1) + 1 == ysel).mean()), 6),
        "type_macro_f1": top1_metrics(ysel[ysel > 0] - 1,
                                      head.predict(dsel["X"])[1][ysel > 0])["macro_f1"],
        "note": "type labels shuffled across event rows; expect near-chance",
    }


def evaluate_head(head: OperatorHead, d: dict) -> dict:
    p_event, p_type, joint = head.predict(d["X"])
    y = d["y_op"]
    ev = y > 0
    out = {
        "hazard": binary_metrics(ev.astype(int), p_event),
        "type_conditional": top1_metrics(y[ev] - 1, p_type[ev]),
        "joint_top1_accuracy": round(float((joint.argmax(1) + 1 == y).mean()), 6),
        "event_support": int(ev.sum()),
        "class_support": {OPS[k]: int((y == k + 1).sum()) for k in range(5)},
    }
    return out


def apply_to_real(head: OperatorHead, d: dict) -> dict:
    """Diagnostic probabilities on real actor-months (no labels exist)."""
    p_event, p_type, joint = head.predict(d["X"])
    out = {
        "rows": int(len(y := d["y_op"])),
        "mean_p_event": round(float(p_event.mean()), 6),
        "mean_joint_by_type": {OPS[k]: round(float(joint[:, k].mean()), 6) for k in range(5)},
        "by_region": {},
        "note": ("synthetic-trained diagnostic; real ICEWS has no clean "
                 "birth/death/merge/split labels; not a causal claim"),
    }
    for region in sorted(set(d["region"])):
        m = d["region"] == region
        out["by_region"][region] = {
            "n": int(m.sum()),
            "mean_p_event": round(float(p_event[m].mean()), 6),
            "type_argmax_share": {
                OPS[k]: round(float((joint[m].argmax(1) == k).mean()), 6) for k in range(5)
            },
        }
    return out
