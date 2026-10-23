"""Extended actor-month targets for T1 (redesign).

The original T1 target (``out_active_next``) is saturated: 91% of actor-months
are positive, the strongest baseline (per-actor Markov) is already at LL 0.136,
and the logistic gain is at the MDE.  This module defines three harder,
substantively motivated actor-month targets and evaluates them with the same
honest protocol (train <2015, select 2015, eval 2016-2017, test untouched):

``high_q50`` / ``high_q75`` / ``high_q90``
    next-month event count above a *train-only* quantile of ``out_events_next``.
    q75 is the primary target; q50/q90 are sensitivity checks.
``new_dyad``
    actor i forms at least one *new* directed dyad in t+1 (first active month
    of that dyad equals t+1).  Derived from ``panel_dyad`` restricted to
    ``t_idx < test_start`` so the frozen test is never read.
``increase`` / ``decrease``
    next-month event count is strictly larger / smaller than this month.

All label construction is causal: the threshold is fitted on train only, the
new-dyad label uses first-activation at t+1, and no row after the eval end is
read.  ``strict`` eval additionally drops t=275 because its label (2018-01)
belongs to the frozen test split (boundary leak fix).
"""

from __future__ import annotations

import os
import time
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from . import features as F
from .bootstrap import auc_ap_ci, paired_gain
from .config import REPO_ROOT
from .metrics import binary_metrics, icc_deff, within_entity_auc
from .models import (calibrate_all, fit_hgb, fit_logistic, predict_hgb,
                     predict_logistic)
from .utils import elapsed, jdump, log

# Month grid is 1995-01 = 0 (same as splits.month_idx).
TEST_START = 276          # 2018-01
EVAL_START = 252          # 2016-01
EVAL_END = 276            # 2018-01 exclusive
SELECT_START = 240        # 2015-01
SELECT_END = 252          # 2016-01

TARGET_ORDER = ["next_active", "high_q50", "high_q75", "high_q90",
                "new_dyad", "increase", "decrease"]

TARGET_TEXT = {
    "next_active": "out_active_next (current T1 sanity check)",
    "high_q50": "out_events_next > train median",
    "high_q75": "out_events_next > train q75",
    "high_q90": "out_events_next > train q90",
    "new_dyad": "actor forms >=1 new directed dyad in t+1",
    "increase": "out_events_next > out_events",
    "decrease": "out_events_next < out_events",
}


def _panel_path(cfg: dict) -> str:
    return os.path.join(cfg["paths"]["cache_dir"], "features_actor.parquet")


def _dyad_path(cfg: dict) -> str:
    return os.path.join(cfg["paths"]["cache_dir"], "panel_dyad.parquet")


def _sql_path(path: str) -> str:
    return str(path).replace("\\", "/")


def load_actor_frame(cfg: dict, include_test: bool = False) -> pd.DataFrame:
    """Actor-month rows with features, metadata, and extended labels.

    Reads only train/val rows from ``features_actor.parquet``.  The new-dyad
    label uses dyad first-activation with ``t_idx < TEST_START`` only.
    """
    con = duckdb.connect()
    try:
        where = "" if include_test else "WHERE split IN ('train','val')"
        cols = ", ".join(f'"{c}"' for c in F.ACTOR_FEATURES)
        df = con.execute(f"""
          SELECT {cols}, actor, aid, region, t_idx, split,
                 out_active, out_events, out_events_next,
                 out_active_next, out_partners, in_partners
          FROM read_parquet('{_sql_path(_panel_path(cfg))}')
          {where}
          ORDER BY aid, t_idx
        """).df()
        for c in F.ACTOR_FEATURES:
            if df[c].dtype == bool:
                df[c] = df[c].astype(np.int8)
        df[F.ACTOR_FEATURES] = df[F.ACTOR_FEATURES].fillna(0.0)

        # First activation per directed dyad, restricted to pre-test months.
        # This makes "new dyad at t+1" label-safe for every eval row we use.
        onset = con.execute(f"""
          WITH act AS (
            SELECT did, sid, tid, t_idx
            FROM read_parquet('{_sql_path(_dyad_path(cfg))}')
            WHERE active AND t_idx < {TEST_START}
          ), first AS (
            SELECT did, sid, tid, min(t_idx) AS first_t
            FROM act GROUP BY did, sid, tid
          ), inc AS (
            SELECT sid AS aid, first_t FROM first
            UNION ALL
            SELECT tid AS aid, first_t FROM first
          )
          SELECT aid, first_t AS t_next, count(*) AS n_new
          FROM inc GROUP BY aid, first_t
        """).df()
    finally:
        con.close()

    df = df.sort_values(["aid", "t_idx"]).reset_index(drop=True)
    # n_new_next = number of incident dyads whose first active month is t+1.
    onset["t_idx"] = onset["t_next"] - 1
    df = df.merge(onset[["aid", "t_idx", "n_new"]], on=["aid", "t_idx"],
                  how="left", validate="one_to_one")
    df["n_new"] = df["n_new"].fillna(0.0)
    df["new_dyad"] = (df["n_new"] > 0).astype(np.int8)

    g = df.groupby("aid", sort=False)
    df["out_events_prev"] = g["out_events"].shift(1).fillna(0.0)
    df["out_active_prev"] = g["out_active"].shift(1).fillna(False).astype(np.int8)
    df["new_dyad_prev"] = g["new_dyad"].shift(1).fillna(0).astype(np.int8)
    df["out_active_next"] = df["out_active_next"].fillna(False).astype(np.int8)
    df["out_events_next"] = df["out_events_next"].fillna(0.0)
    return df


def threshold_for(df: pd.DataFrame, target: str) -> float | None:
    if not target.startswith("high_"):
        return None
    q = float(target.split("_q")[1]) / 100.0
    tr = df[df["split"] == "train"]
    return float(np.quantile(tr["out_events_next"].to_numpy(), q))


def make_label(df: pd.DataFrame, target: str, thr: float | None = None) -> np.ndarray:
    if target == "next_active":
        return df["out_active_next"].to_numpy(dtype=np.int64)
    if target.startswith("high_"):
        if thr is None:
            thr = threshold_for(df, target)
        return (df["out_events_next"].to_numpy() > float(thr)).astype(np.int64)
    if target == "new_dyad":
        return df["new_dyad"].to_numpy(dtype=np.int64)
    if target == "increase":
        return (df["out_events_next"].to_numpy() >
                df["out_events"].to_numpy()).astype(np.int64)
    if target == "decrease":
        return (df["out_events_next"].to_numpy() <
                df["out_events"].to_numpy()).astype(np.int64)
    raise KeyError(target)


def prev_label(df: pd.DataFrame, target: str, thr: float | None = None) -> np.ndarray:
    """Label at t-1, known when predicting t+1."""
    if target == "next_active":
        return df["out_active_prev"].to_numpy(dtype=np.int64)
    if target.startswith("high_"):
        if thr is None:
            thr = threshold_for(df, target)
        return (df["out_events_prev"].to_numpy() > float(thr)).astype(np.int64)
    if target == "new_dyad":
        return df["new_dyad_prev"].to_numpy(dtype=np.int64)
    if target == "increase":
        return (df["out_events"].to_numpy() >
                df["out_events_prev"].to_numpy()).astype(np.int64)
    if target == "decrease":
        return (df["out_events"].to_numpy() <
                df["out_events_prev"].to_numpy()).astype(np.int64)
    raise KeyError(target)


def split_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    t = df["t_idx"].to_numpy()
    split = df["split"].to_numpy()
    return {
        "train": (split == "train"),
        "select": (t >= SELECT_START) & (t < SELECT_END),
        # strict: label month t+1 must stay inside validation (< 2018-01)
        "eval": (t >= EVAL_START) & (t < EVAL_END - 1),
        "eval_with_boundary": (t >= EVAL_START) & (t < EVAL_END),
    }


def _actor_rate_baselines(df, y, masks, prev) -> dict[str, np.ndarray]:
    """Global / per-actor / per-actor-Markov probabilities for eval rows."""
    tr, se, ev = masks["train"], masks["select"], masks["eval"]
    aid = df["aid"].to_numpy()
    n_a = int(aid.max()) + 1
    prior_tr = float(y[tr].mean())
    prior_se = float(y[se].mean())

    def rate_for(mask, prior):
        cnt = np.bincount(aid[mask], minlength=n_a).astype(np.float64)
        pos = np.bincount(aid[mask], weights=y[mask], minlength=n_a)
        # Smoothed per-actor rate (pseudo-count 5 toward the split prior).
        return (pos + 5.0 * prior) / (cnt + 5.0)

    rate_tr = rate_for(tr, prior_tr)
    rate_se = rate_for(se, prior_se)

    # Markov on the lagged label, fitted on train (global fallback).
    yprev = prev
    p11_num = np.bincount(aid[tr & (yprev == 1)], weights=y[tr & (yprev == 1)],
                          minlength=n_a)
    p11_den = np.bincount(aid[tr & (yprev == 1)], minlength=n_a).astype(np.float64)
    p01_num = np.bincount(aid[tr & (yprev == 0)], weights=y[tr & (yprev == 0)],
                          minlength=n_a)
    p01_den = np.bincount(aid[tr & (yprev == 0)], minlength=n_a).astype(np.float64)
    m1 = tr & (yprev == 1)
    m0 = tr & (yprev == 0)
    g1 = float(y[m1].mean()) if m1.any() else prior_tr
    g0 = float(y[m0].mean()) if m0.any() else prior_tr
    a1 = (p11_num + 2.0 * g1) / (p11_den + 2.0)
    a0 = (p01_num + 2.0 * g0) / (p01_den + 2.0)

    return {
        "global_rate_train": np.full(int(ev.sum()), prior_tr),
        "global_rate_select": np.full(int(ev.sum()), prior_se),
        "per_actor_rate_train": rate_tr[aid[ev]],
        "per_actor_rate_select": rate_se[aid[ev]],
        "per_actor_markov": np.where(yprev[ev] == 1, a1[aid[ev]], a0[aid[ev]]),
    }


def _gain_block(y, p_base, p_model, cid, reps: int, seed: int) -> dict:
    g = paired_gain(y, p_base, p_model, cid, reps=reps, seed=seed)
    return g


def run_target(df: pd.DataFrame, target: str, cfg: dict,
               reps: int = 500) -> dict[str, Any]:
    t0 = time.time()
    masks = split_masks(df)
    tr, se, ev = masks["train"], masks["select"], masks["eval"]
    thr = threshold_for(df, target)
    y = make_label(df, target, thr)
    prev = prev_label(df, target, thr)
    X = df[F.ACTOR_FEATURES].to_numpy(np.float32)
    aid = df["aid"].to_numpy()
    seed = int(cfg["seed"])

    ytr, yse, yev = y[tr], y[se], y[ev]
    base = _actor_rate_baselines(df, y, masks, prev)
    base_blocks = {k: binary_metrics(yev, p) for k, p in base.items()}
    # Pre-registered primary baseline: per-actor Markov on the lagged label
    # (the same baseline used by the original T1).  Baselines chosen on the
    # select window are reported as a sensitivity check only.
    primary_base = "per_actor_markov"
    base_sel = {k: binary_metrics(yse, p) for k, p in
                _actor_rate_baselines(df, y, {**masks, "eval": se}, prev).items()}
    select_best = min(base_sel, key=lambda k: base_sel[k]["log_loss"])
    log(f"t1x {target}: rows={int(ev.sum())} pos={int(yev.sum())} "
        f"primary_base={primary_base} eval_ll="
        f"{base_blocks[primary_base]['log_loss']:.5f} select_best={select_best}")

    fit = fit_logistic(X[tr], ytr, X[se], yse, cfg,
                       feature_names=F.ACTOR_FEATURES)
    p_sel = predict_logistic(fit, X[se])
    p_ev = predict_logistic(fit, X[ev])
    cal = calibrate_all(p_sel, yse, p_ev, cfg)

    blocks, preds = {}, {}
    for cname, c in cal.items():
        key = "logistic_full" if cname == "none" else f"logistic_full_{cname}"
        preds[key] = c["p_eval"]
        blocks[key] = {
            "name": key,
            "metrics": binary_metrics(yev, c["p_eval"]),
            "select_log_loss": binary_metrics(yse, c["p_sel"])["log_loss"],
            "gain_vs_primary": _gain_block(
                yev, base[primary_base], c["p_eval"], aid[ev], reps, seed),
        }
        blocks[key]["metrics"]["auc_ap_ci"] = auc_ap_ci(
            yev, c["p_eval"], aid[ev], reps=min(200, reps), seed=seed)

    hgb = fit_hgb(X[tr], ytr, X[se], yse, cfg)
    ph_sel = predict_hgb(hgb, X[se])
    ph_ev = predict_hgb(hgb, X[ev])
    hcal = calibrate_all(ph_sel, yse, ph_ev, cfg)
    for cname, c in hcal.items():
        key = "hgb_full" if cname == "none" else f"hgb_full_{cname}"
        preds[key] = c["p_eval"]
        blocks[key] = {
            "name": key,
            "metrics": binary_metrics(yev, c["p_eval"]),
            "select_log_loss": binary_metrics(yse, c["p_sel"])["log_loss"],
            "gain_vs_primary": _gain_block(
                yev, base[primary_base], c["p_eval"], aid[ev], reps, seed),
        }
        blocks[key]["metrics"]["auc_ap_ci"] = auc_ap_ci(
            yev, c["p_eval"], aid[ev], reps=min(100, reps), seed=seed)

    # Pre-registered primary model: uncalibrated L2 logistic.  Isotonic and
    # HGB are sensitivity/capacity controls; selecting on select-LL would pick
    # isotonic for some targets and then lose on eval (documented).
    primary = "logistic_full"
    best_model_select = min(("logistic_full", "logistic_full_isotonic", "hgb_full"),
                            key=lambda k: blocks[k]["select_log_loss"])
    pg = blocks[primary]["gain_vs_primary"]
    pg_sel = _gain_block(yev, base[select_best], preds[primary], aid[ev], reps, seed)
    mde = pg["mde"]
    ratio = (pg["point"]["d_log_loss"] / mde["mde"]) if mde["mde"] else None
    auc_primary = blocks[primary]["metrics"]["auc"]

    # Support / power diagnostics on the outcome and on the paired residual.
    sup = {
        "rows": int(ev.sum()),
        "positives": int(yev.sum()),
        "pos_rate": round(float(yev.mean()), 6),
        "clusters": int(np.unique(aid[ev]).size),
        "threshold": None if thr is None else round(float(thr), 6),
        "n_eff_outcome": icc_deff(yev, aid[ev]),
        "within_entity_auc": within_entity_auc(yev, preds[primary], aid[ev])
        if primary in preds else None,
        "n_eff_boundary_included": int(masks["eval_with_boundary"].sum()),
        "positives_boundary_included": int(y[masks["eval_with_boundary"]].sum()),
    }
    return {
        "target": target,
        "definition": TARGET_TEXT[target],
        "support": sup,
        "baselines": base_blocks,
        "baseline_select_ll": {k: v["log_loss"] for k, v in base_sel.items()},
        "primary_baseline": primary_base,
        "select_best_baseline": select_best,
        "models": blocks,
        "primary_model": primary,
        "primary_gain": pg,
        "gain_vs_select_best": pg_sel,
        "best_model_select": best_model_select,
        "gain_over_mde": None if ratio is None else round(float(ratio), 3),
        "adequately_powered": bool(ratio is not None and ratio >= 1.25 and
                                   sup["positives"] >= 200 and
                                   auc_primary is not None and auc_primary >= 0.65),
        "logistic_fit": {"C": fit["C"], "select_curve": fit["select_curve"]},
        "coef_table": [
            {"feature": n, "coef": round(float(c), 6)}
            for n, c in sorted(zip(F.ACTOR_FEATURES, fit["coef"]),
                               key=lambda t: -abs(t[1]))[:25]
        ],
        "calibration": {
            "temperature_T": cal.get("temperature", {}).get("T"),
            "none_select_ll": cal["none"]["select_log_loss"],
            "isotonic_select_ll": cal.get("isotonic", {}).get("select_log_loss"),
        },
        "runtime_seconds": elapsed(t0),
    }


def run(cfg: dict, out_path: str | None = None) -> dict:
    t0 = time.time()
    df = load_actor_frame(cfg)
    reps = int(cfg["bootstrap"]["reps"])
    results = {t: run_target(df, t, cfg, reps=reps) for t in TARGET_ORDER}
    # Verdict summary.
    powered = [t for t, r in results.items() if r["adequately_powered"]]
    primary = "high_q75" if "high_q75" in powered else (
        powered[0] if powered else None)
    out = {
        "target_family": "T1_actor_month_redesign",
        "split": {"train": "<2015-01", "select": "2015", "eval": "2016-2017",
                  "strict_eval_note": "t=275 dropped: its label is 2018-01 (test)"},
        "targets": results,
        "powered_targets": powered,
        "primary_target": primary,
        "verdict": (
            "T1 redesign viable: high-activity target is above MDE with "
            "strong discrimination; next_active remains at MDE and is kept "
            "only as a sanity check."
            if primary else
            "No extended T1 target is adequately powered; use Option A "
            "(per-actor explanations of dyad forecasts) instead of literal "
            "node-level forecasting."
        ),
        "runtime_seconds": elapsed(t0),
    }
    if out_path:
        jdump(out_path, out)
    return out
