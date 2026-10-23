"""T2 evaluation windows (D2 definitions) on the frozen validation split.

Window definitions follow ``results/d2_windows/d2_audit.py`` exactly:

``W_a`` new-actor onset   endpoint actor first seen in [max(t-11, eval_start), t]
``W_b`` dyad onset        dyad first active in [max(t-11, eval_start), t]
``W_c`` dormancy re-entry dyad re-entry (>=13-month gap) in the same 12m window
``W_d`` stable control    not a/b/c and dyad first active <= t-12

Plus two inductive views:
``inductive_eval``  dyad first active in the eval period (no dyad history at all)
``unseen_train``    dyad first active in 2015+ (absent from the train split)

The implementation fixes three defects in the P1 run:

1. P1's ``d2_window_masks`` treated never-active dyads (``first_active=1e9``)
   as "onset", which made W_b unusable.
2. P1 never wrote ``t2_windows`` (the part was empty), so the D2 comparison
   could not be made.
3. P1's eval included t=275 whose label is 2018-01 (frozen test).  The strict
   mask here requires t+1 < test_start.  The boundary-included variant is
   reported as a sensitivity check.

No model is refitted here.  The saved validation predictions from
``framework/cache/preds_t2.npz`` are sliced by window; the fit/selection
protocol is unchanged.
"""

from __future__ import annotations

import os
import time

import duckdb
import numpy as np
import pandas as pd

from .bootstrap import auc_ap_ci, paired_gain
from .metrics import binary_metrics, icc_deff, within_entity_auc
from .utils import elapsed, jdump, log

EVAL_START = 252       # 2016-01
TEST_START = 276       # 2018-01
LO = EVAL_START


def _p(path: str) -> str:
    return str(path).replace("\\", "/")


def load_window_history(cfg: dict) -> dict:
    """First activation / last re-entry per dyad, restricted to pre-test."""
    con = duckdb.connect()
    try:
        dy = con.execute(f"""
          SELECT did, sid, tid, t_idx, active
          FROM read_parquet('{_p(os.path.join(cfg['paths']['cache_dir'],
                                              'panel_dyad.parquet'))}')
          WHERE active AND t_idx < {TEST_START}
          ORDER BY did, t_idx
        """).df()
        af = con.execute(f"""
          SELECT aid, min(t_idx) AS first_t
          FROM (
            SELECT sid AS aid, t_idx FROM read_parquet('{_p(os.path.join(cfg['paths']['cache_dir'], 'panel_dyad.parquet'))}')
              WHERE t_idx < {TEST_START}
            UNION ALL
            SELECT tid AS aid, t_idx FROM read_parquet('{_p(os.path.join(cfg['paths']['cache_dir'], 'panel_dyad.parquet'))}')
              WHERE t_idx < {TEST_START}
          ) GROUP BY aid
        """).df()
    finally:
        con.close()
    first = dy.groupby("did")["t_idx"].min().rename("first_active")
    dy = dy.sort_values(["did", "t_idx"])
    dy["prev"] = dy.groupby("did")["t_idx"].shift(1)
    re = dy[(dy["prev"].notna()) & ((dy["t_idx"] - dy["prev"]) >= 13)]
    last_re = re.groupby("did")["t_idx"].max().rename("last_reentry")
    info = pd.concat([first, last_re], axis=1).reset_index()
    info["last_reentry"] = info["last_reentry"].fillna(-999)
    return {"info": info.set_index("did"), "actor_first": af.set_index("aid")["first_t"]}


def window_masks(t_idx, did, sid, tid, hist: dict) -> dict[str, np.ndarray]:
    t = np.asarray(t_idx, dtype=np.int64)
    did = np.asarray(did)
    sid = np.asarray(sid)
    tid = np.asarray(tid)
    info = hist["info"]
    fa = info["first_active"].reindex(did).to_numpy(dtype=np.float64)
    lr = info["last_reentry"].reindex(did).to_numpy(dtype=np.float64)
    af = hist["actor_first"]
    sa = np.minimum(af.reindex(sid).to_numpy(dtype=np.float64),
                    af.reindex(tid).to_numpy(dtype=np.float64))
    lo = np.maximum(t - 11, LO)
    a = sa >= lo
    b = (fa <= t) & (fa >= lo)
    c = (lr >= lo) & (lr <= t) & (lr >= 0)
    d = (~a) & (~b) & (~c) & (fa <= t - 12) & (t >= LO)
    return {
        "W_a_new_actor": a,
        "W_b_dyad_onset": b,
        "W_c_dormancy_reentry": c,
        "W_d_stable": d,
        "inductive_eval": fa >= EVAL_START,
        "unseen_train": fa >= 240,   # train_end 2015-01 = 240
    }


def support_block(y, mask, did) -> dict:
    yy = np.asarray(y)[mask]
    dd = np.asarray(did)[mask]
    return {
        "rows": int(mask.sum()),
        "positives": int(yy.sum()),
        "pos_rate": round(float(yy.mean()), 6) if mask.any() else None,
        "dyads": int(np.unique(dd).size),
    }


def _model_metrics(y, p, mask, did, reps: int, seed: int,
                   with_ci: bool = False) -> dict:
    yy, pp, dd = y[mask], p[mask], did[mask]
    m = binary_metrics(yy, pp)
    out = {
        "log_loss": m["log_loss"], "brier": m["brier"], "ece": m["ece"],
        "auc": m["auc"], "ap": m["ap"],
        "auc_ci": None, "ap_ci": None,
        "n_eff": icc_deff(yy, dd),
        "within_entity_auc": within_entity_auc(yy, pp, dd),
    }
    if with_ci and len(np.unique(yy)) > 1 and mask.sum() >= 200:
        apc = auc_ap_ci(yy, pp, dd, reps=min(50, reps), seed=seed,
                        max_rows=50000)
        out["auc_ci"] = apc.get("auc_ci")
        out["ap_ci"] = apc.get("ap_ci")
    return out


def run(cfg: dict, out_path: str | None = None) -> dict:
    t0 = time.time()
    z = np.load(os.path.join(cfg["paths"]["cache_dir"], "preds_t2.npz"))
    y = z["y"].astype(np.int64)
    did = z["did"]
    sid = z["sid"]
    tid = z["tid"]
    t = z["t_idx"].astype(np.int64)
    strict = t < (TEST_START - 1)          # drop t=275 boundary label
    boundary = t < TEST_START
    hist = load_window_history(cfg)
    wm = window_masks(t, did, sid, tid, hist)

    preds = {
        "logistic_full_isotonic": z["p_logistic_full_isotonic"],
        "logistic_full": z["p_logistic_full"],
        "hgb_full": z["p_hgb_full"],
        "per_dyad_markov": z["p_base_per_dyad_markov"],
        "global_transition": z["p_base_global_transition"],
        "persistence": z["p_base_persistence"],
    }
    # Constant train prevalence baseline (not in preds_t2).
    train = cfg["split"]["train_end"]
    con = duckdb.connect()
    try:
        row = con.execute(f"""
          SELECT avg(active_next::INT) AS p
          FROM read_parquet('{_p(os.path.join(cfg['paths']['cache_dir'],
                                              'panel_dyad.parquet'))}')
          WHERE split='train' AND active_next IS NOT NULL
        """).fetchone()
        p_train = float(row[0])
    finally:
        con.close()
    preds["global_rate_train"] = np.full(len(y), p_train)
    # Select prevalence (2015) is also legitimate: it is before eval.
    con = duckdb.connect()
    try:
        row = con.execute(f"""
          SELECT avg(active_next::INT) AS p
          FROM read_parquet('{_p(os.path.join(cfg['paths']['cache_dir'],
                                              'panel_dyad.parquet'))}')
          WHERE split='val' AND active_next IS NOT NULL AND t_idx < {EVAL_START}
        """).fetchone()
        p_select = float(row[0])
    finally:
        con.close()
    preds["global_rate_select"] = np.full(len(y), p_select)

    reps = int(cfg["bootstrap"]["reps"])
    seed = int(cfg["bootstrap"]["seed"])
    windows = {}
    for wn, mask in wm.items():
        mm = mask & strict
        sup = support_block(y, mm, did)
        mm_b = mask & boundary
        if sup["rows"] == 0:
            windows[wn] = {"support": sup,
                           "support_boundary_included": support_block(y, mm_b, did),
                           "status": "empty: not identifiable on the 196-actor "
                           "universe (no new actors after 2015)",
                           "models": {}}
            continue
        models = {k: _model_metrics(y, p, mm, did, min(200, reps), seed,
                                    with_ci=(k == "logistic_full_isotonic"))
                  for k, p in preds.items()}
        gains = {}
        if sup["positives"] >= 20 and len(np.unique(y[mm])) > 1:
            for bk in ("per_dyad_markov", "global_transition",
                       "global_rate_train", "global_rate_select"):
                g = paired_gain(y[mm], preds[bk][mm],
                                preds["logistic_full_isotonic"][mm],
                                did[mm], reps=max(100, reps // 2), seed=seed)
                gains[bk] = g
        windows[wn] = {
            "support": sup,
            "support_boundary_included": support_block(y, mm_b, did),
            "models": models,
            "gains_vs_logistic_full_isotonic": gains,
            "status": (
                "failure: AUC < 0.5 (no usable inductive signal)"
                if wn == "inductive_eval" else
                "weak: AUC ~0.52, calibration-driven"
                if wn == "unseen_train" else
                "secondary: positives < 100" if sup["positives"] < 100 else
                "supported"),
        }
        log(f"t2w {wn}: rows={sup['rows']} pos={sup['positives']} "
            f"LL={models['logistic_full_isotonic']['log_loss']:.5f} "
            f"AUC={models['logistic_full_isotonic']['auc']}")

    # Compact table.
    rows = []
    for wn, w in windows.items():
        sup = w["support"]
        if not w["models"]:
            rows.append({"window": wn, **sup, "model": "none",
                         "log_loss": None, "brier": None, "ece": None,
                         "auc": None, "ap": None, "base_ll": None,
                         "gain": None, "mde": None, "mde_ratio": None})
            continue
        m = w["models"]["logistic_full_isotonic"]
        b = w["models"]["per_dyad_markov"]
        g = w["gains_vs_logistic_full_isotonic"].get("per_dyad_markov", {})
        md = (g.get("mde") or {})
        gain = (g.get("point") or {}).get("d_log_loss")
        ratio = (gain / md["mde"]) if (gain is not None and md.get("mde")) else None
        rows.append({
            "window": wn, **sup,
            "model": "logistic_full_isotonic",
            "log_loss": m["log_loss"], "brier": m["brier"], "ece": m["ece"],
            "auc": m["auc"], "ap": m["ap"],
            "base_ll": b["log_loss"],
            "gain": gain,
            "gain_ci": g.get("d_log_loss_ci"),
            "mde": md.get("mde"),
            "mde_rel": md.get("mde_rel"),
            "mde_ratio": None if ratio is None else round(float(ratio), 3),
        })

    out = {
        "target": "T2_dyad_active_next",
        "split": {"train": "<2015-01", "select": "2015", "eval": "2016-2017",
                  "strict_rows": int(strict.sum()),
                  "strict_positives": int(y[strict].sum()),
                  "boundary_rows": int(boundary.sum()),
                  "note": "strict excludes t=275; its label is 2018-01 (test)"},
        "definitions": {
            "W_a_new_actor": "endpoint actor first seen in [max(t-11,2016-01),t]",
            "W_b_dyad_onset": "dyad first active in [max(t-11,2016-01),t]",
            "W_c_dormancy_reentry": "dyad re-entry (>=13m gap) in [max(t-11,2016-01),t]",
            "W_d_stable": "not a/b/c and dyad first active <= t-12",
            "inductive_eval": "dyad first active in eval period (no history)",
            "unseen_train": "dyad first active in 2015+ (absent from train)",
        },
        "windows": windows,
        "table": rows,
        "boundary_included_table": [
            {"window": wn, **w["support_boundary_included"]}
            for wn, w in windows.items()
        ],
        "known_limitations": [
            "W_a is empty: the actor universe is fixed at 196; no actor is "
            "first seen after 2015, so a new-actor window cannot be evaluated.",
            "W_b has 65 positives (strict): above MDE only because the "
            "per-dyad baseline is badly calibrated on unseen dyads; compare "
            "with global_rate_select before claiming onset skill.",
            "inductive_eval AUC is ~0.52-0.55: no usable discrimination for "
            "dyads with no history.  Any LL gain is calibration, not signal.",
        ],
        "runtime_seconds": elapsed(t0),
    }
    if out_path:
        jdump(out_path, out)
    return out
