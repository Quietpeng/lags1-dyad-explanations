"""Sprint-A: pre-specified 20% unordered-dyad holdout with boundary embargo.

This module is deliberately separate from the frozen P1 artefacts.  It never
overwrites ``framework/cache/graph_*.npy``, ``results/framework_pilot/parts``
or any other frozen file.  All new caches use the ``_holdout`` tag and all
outputs go to ``results/framework_pilot/holdout``.

Design (frozen before any holdout label is read)
------------------------------------------------
Unit
    Unordered actor pair ``{min(sid,tid), max(sid,tid)}``.  A directed dyad
    ``did`` is the panel's row unit; holding out a pair removes *both*
    directions, so the reverse-dyad features cannot leak one direction into
    training while the other is evaluated.
Universe
    All unordered pairs with at least one row in the frozen 1995-01..2023-04
    panel (13,952 pairs; 26,434 directed dyads).
Strata
    ``ever_active`` = at least one month with ``active=1`` in the pre-embargo
    training period ``t_idx < 239`` (1995-01..2014-11).  Never-active pairs are
    the second stratum.  This is the memo's "stratified by ever-active".
Fraction / seed
    20% of each stratum, fixed seed 20261023.
Algorithm
    Sort pairs by ``(min(sid,tid), max(sid,tid))``.  Draw
    ``rng = np.random.default_rng(20261023)``; for stratum in ``(1, 0)``:
    ``idx = rng.permutation(n)``; ``k = max(1, floor(0.2*n + 0.5))``; take
    ``idx[:k]``.  The stratum order and the rounding rule are part of the
    internally hash-timestamped plan.
Evaluation
    The held-out pairs' 2018-01..2022-11 rows (``276 <= t_idx < 335``; 2022-12
    is embargoed because its label is 2023-01).  The period is retrospective
    (it was read by earlier G2 diagnostics), but the *units* are fresh: no
    held-out pair contributes a row to train, select, calibration or any
    train-only statistic.  This is a unit-fresh, not a time-fresh, holdout.

Embargo
-------
``t_idx in {239, 251, 275, 335}`` carry a label from the next period:
2014-12 -> 2015-01 (train -> select), 2015-12 -> 2016-01 (select -> eval),
2017-12 -> 2018-01 (inner eval -> holdout), 2022-12 -> 2023-01 (tail).
Those rows are dropped from fitting/selection/evaluation, and ``br`` is
recomputed from the remaining training rows.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import data, features, models
from .metrics import binary_metrics, reliability_slope
from .utils import jdump, jload, log

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------
HOLDOUT_FRAC = 0.20
HOLDOUT_SEED = 20261023
HOLDOUT_TAG = "_holdout"

TRAIN_END = 239          # 2015-01
SELECT_START = 240       # 2015-01
SELECT_END = 252         # 2016-01
INNER_EVAL_START = 252   # 2016-01
INNER_EVAL_END = 276     # 2018-01
HOLDOUT_EVAL_START = 276 # 2018-01
HOLDOUT_EVAL_END = 336   # 2023-01
TAIL_START = 336         # 2023-01

EMBARGO_T = (239, 251, 275, 335)
EMBARGO_DATES = {
    239: "2014-12", 251: "2015-12", 275: "2017-12", 335: "2022-12",
}

# Frozen P1 order: 27 causal + 20 graph + br + seen_tr.  The 29-feature set is
# the same 27 causal columns plus br/seen_tr (the order used when the graph
# group is absent).  The model matrices select by name, so the two orders are
# equivalent; the lists are kept exact for reporting.
FEATURES_29 = list(features.DYAD_CAUSAL) + ["br", "seen_tr"]
FEATURES_49 = list(features.DYAD_CAUSAL) + list(features.GRAPH_CORE20) + ["br", "seen_tr"]

GROUP_ORDER = [
    "calendar", "dyad_current", "dyad_lags", "dyad_rolling",
    "dyad_history", "reverse_dyad", "actor_ego", "train_stats", "graph",
]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def month_idx(date_str: str) -> int:
    y, m, _ = date_str.split("-")
    return (int(y) - 1995) * 12 + (int(m) - 1)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_json(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slice_dict(d: dict, mask: np.ndarray) -> dict:
    """Slice every row-aligned array; keep scalars/metadata untouched."""
    mask = np.asarray(mask, dtype=bool)
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(mask):
            out[k] = v[mask]
        else:
            out[k] = v
    return out


def _indices(names: list[str], wanted: list[str]) -> list[int]:
    idx = {n: i for i, n in enumerate(names)}
    missing = [w for w in wanted if w not in idx]
    if missing:
        raise KeyError(f"features missing: {missing}")
    return [idx[w] for w in wanted]


# ---------------------------------------------------------------------------
# Holdout definition
# ---------------------------------------------------------------------------
def define_holdout(con, frac: float = HOLDOUT_FRAC,
                   seed: int = HOLDOUT_SEED) -> dict:
    """Derive the unordered-pair holdout from the frozen panel.

    Only ``active`` (current-month activity) is touched for stratification; no
    ``active_next`` label from 2018-22 is read.  The function is deterministic
    and the returned ``definition_sha256`` is the plan hash.
    """
    rows = con.execute("""
        SELECT DISTINCT did::INTEGER AS did, sid::SMALLINT AS sid,
                        tid::SMALLINT AS tid
        FROM panel ORDER BY did
    """).fetchall()
    pair_of: dict[int, tuple[int, int]] = {}
    dids_of: dict[tuple[int, int], list[int]] = {}
    for did, sid, tid in rows:
        key = (int(min(sid, tid)), int(max(sid, tid)))
        pair_of[int(did)] = key
        dids_of.setdefault(key, []).append(int(did))

    act_rows = con.execute("""
        SELECT did::INTEGER AS did, max(active::INT) AS ever_active
        FROM panel WHERE t_idx < 239 GROUP BY did
    """).fetchall()
    ever: dict[tuple[int, int], int] = {}
    for did, a in act_rows:
        key = pair_of[int(did)]
        ever[key] = max(ever.get(key, 0), int(a))

    pairs = sorted(dids_of)
    pools = {
        1: [k for k in pairs if ever.get(k, 0) == 1],
        0: [k for k in pairs if ever.get(k, 0) == 0],
    }
    rng = np.random.default_rng(int(seed))
    chosen: list[tuple[int, int]] = []
    strata_counts = {}
    for stratum in (1, 0):
        pool = pools[stratum]
        n = len(pool)
        if n == 0:
            strata_counts[stratum] = {"n": 0, "k": 0}
            continue
        k = max(1, int(np.floor(float(frac) * n + 0.5)))
        idx = rng.permutation(n)[:k]
        chosen.extend([pool[int(i)] for i in idx])
        strata_counts[stratum] = {"n": int(n), "k": int(k)}
    chosen = sorted(chosen)
    dids = sorted({d for key in chosen for d in dids_of[key]})

    # Row counts per period for the held-out units (no labels).
    con.register("sprint_a_hd_tmp", pd.DataFrame({"did": np.asarray(dids, dtype=np.int64)}))
    counts = con.execute("""
        SELECT
          count(*) FILTER (WHERE t_idx < 239) AS train_rows,
          count(*) FILTER (WHERE t_idx >= 240 AND t_idx < 251) AS select_rows,
          count(*) FILTER (WHERE t_idx >= 252 AND t_idx < 275) AS inner_eval_rows,
          count(*) FILTER (WHERE t_idx >= 276 AND t_idx < 335) AS holdout_eval_rows,
          count(*) FILTER (WHERE t_idx >= 336) AS tail_rows,
          count(DISTINCT did) AS directed_dyads,
          count(DISTINCT (least(sid,tid)::VARCHAR || '|' || greatest(sid,tid)::VARCHAR)) AS unordered_pairs
        FROM panel p
        JOIN sprint_a_hd_tmp h USING (did)
    """).fetchone()
    con.unregister("sprint_a_hd_tmp")

    definition = {
        "unit": "unordered_actor_pair",
        "universe": ("all unordered actor pairs with >=1 row in the frozen "
                     "1995-01..2023-04 dyad-month panel"),
        "frac": float(frac),
        "seed": int(seed),
        "strata": ("ever_active_in_pre_embargo_train (max(active)=1 for any "
                   "month with t_idx<239); second stratum = never active"),
        "algorithm": ("pairs sorted by (min(sid,tid),max(sid,tid)); "
                      "rng=default_rng(seed); for stratum in (1,0): "
                      "idx=rng.permutation(n); "
                      "k=max(1,floor(frac*n+0.5)); take idx[:k]"),
        "eval_period": "2018-01..2022-11 (276<=t_idx<335)",
        "embargo_t_idx": list(EMBARGO_T),
        "pairs": [[int(a), int(b)] for a, b in chosen],
        "dids": [int(d) for d in dids],
    }
    return {
        "definition": definition,
        "definition_sha256": sha256_json(definition),
        "pairs": chosen,
        "dids": np.asarray(dids, dtype=np.int64),
        "strata_counts": strata_counts,
        "counts": {
            "unordered_pairs": int(len(definition["pairs"])),
            "directed_dyads": int(counts[5]),
            "train_rows": int(counts[0]),
            "select_rows": int(counts[1]),
            "inner_eval_rows": int(counts[2]),
            "holdout_eval_rows": int(counts[3]),
            "tail_rows": int(counts[4]),
        },
        "n_pairs_universe": int(len(pairs)),
        "n_dids_universe": int(len(rows)),
    }


def write_preregistration(path: str, payload: dict, force: bool = False) -> dict:
    """Write the frozen plan once; refuse silent overwrite."""
    if os.path.exists(path) and not force:
        raise FileExistsError(f"plan already frozen: {path}")
    payload = json.loads(json.dumps(payload))  # JSON-safe copy
    payload["hashes"] = payload.get("hashes", {})
    payload["hashes"]["definition_sha256"] = sha256_json(payload["definition"])
    payload["hashes"]["pairs_sha256"] = sha256_json(payload["definition"]["pairs"])
    payload["hashes"]["dids_sha256"] = sha256_json(payload["definition"]["dids"])
    payload.setdefault("created_utc", utc_now())
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True, default=str)
        fh.write("\n")
    file_hash = sha256_file(path)
    with open(path + ".sha256", "w", encoding="utf-8") as fh:
        fh.write(file_hash + "  " + os.path.basename(path) + "\n")
    return {"path": path, "file_sha256": file_hash,
            "definition_sha256": payload["hashes"]["definition_sha256"]}


def verify_preregistration(path: str) -> dict:
    payload = jload(path)
    recomputed = sha256_json(payload["definition"])
    frozen = payload.get("hashes", {}).get("definition_sha256")
    if frozen != recomputed:
        raise AssertionError(
            f"plan definition hash mismatch: {frozen} != {recomputed}")
    pairs_hash = sha256_json(payload["definition"]["pairs"])
    dids_hash = sha256_json(payload["definition"]["dids"])
    if payload["hashes"].get("pairs_sha256") != pairs_hash:
        raise AssertionError("plan pairs hash mismatch")
    if payload["hashes"].get("dids_sha256") != dids_hash:
        raise AssertionError("plan dids hash mismatch")
    return payload


# ---------------------------------------------------------------------------
# Holdout graph view and bundle
# ---------------------------------------------------------------------------
def make_panel_view(con, dids: np.ndarray) -> None:
    """Zero the held-out dyads' edges, keeping their rows for evaluation.

    ``active=0`` and ``n_events=0`` remove the held-out pairs from the
    undirected graph snapshots for *all* months.  The rows themselves remain so
    the held-out units can still be scored; their own lagged/current features
    come from the ordinary panel, not from this view.
    """
    con.register("sprint_a_drop_df", pd.DataFrame({"did": np.asarray(dids, dtype=np.int64)}))
    con.execute("DROP TABLE IF EXISTS sprint_a_drop")
    con.execute("CREATE TABLE sprint_a_drop AS SELECT DISTINCT did FROM sprint_a_drop_df")
    con.unregister("sprint_a_drop_df")
    con.execute("DROP VIEW IF EXISTS panel_sprint_a")
    con.execute("""
        CREATE VIEW panel_sprint_a AS
        SELECT p.s, p.t, p.m, p.t_idx, p.did, p.sid, p.tid,
               CASE WHEN d.did IS NULL THEN p.active ELSE FALSE END AS active,
               CASE WHEN d.did IS NULL THEN p.n_events ELSE 0 END AS n_events,
               p.active_next, p.root, p.root_next, p.split
        FROM panel p LEFT JOIN sprint_a_drop d USING (did)
    """)


def build_holdout_bundle(cfg: dict, spec: dict, force_graph: bool = False,
                         con=None, label_mode: str = "full") -> dict:
    """Load the frozen base features, rebuild graph/br without holdout, export.

    The train/select/eval dictionaries contain only rows allowed by the
    pre-specified split.  The full frozen caches are never modified.
    """
    own = con is None
    if own:
        con = data.connect(cfg)
        data.load_views(con, cfg)
    holdout_dids = np.asarray(spec["dids"], dtype=np.int64)
    hold_set = set(int(d) for d in holdout_dids)

    features.build_dyad_base(con, cfg, force=False)
    make_panel_view(con, holdout_dids)
    gm = features.build_graph_features(
        con, cfg, force=force_graph, tag=HOLDOUT_TAG, panel_view="panel_sprint_a")
    log(f"sprint_a: graph tag={HOLDOUT_TAG} rows_train={gm.get('rows_train')}")

    train_full = features.export_dyad_split(con, cfg, "train", gm,
                                            graph_tag=HOLDOUT_TAG)
    val_full = features.export_dyad_split(con, cfg, "val", gm,
                                          graph_tag=HOLDOUT_TAG)
    test_full = features.export_dyad_split(con, cfg, "test", gm,
                                           graph_tag=HOLDOUT_TAG,
                                           label_mode=label_mode)

    n_dyads = int(max(train_full["did"].max(), val_full["did"].max(),
                      test_full["did"].max())) + 1

    tr_mask = (train_full["t_idx"] < TRAIN_END) & ~np.isin(train_full["did"], holdout_dids)
    sel_mask = ((val_full["t_idx"] >= SELECT_START) &
                (val_full["t_idx"] < SELECT_END) &
                ~np.isin(val_full["did"], holdout_dids) &
                ~np.isin(val_full["t_idx"], EMBARGO_T))
    ev_mask = ((test_full["t_idx"] >= HOLDOUT_EVAL_START) &
               (test_full["t_idx"] < HOLDOUT_EVAL_END) &
               np.isin(test_full["did"], holdout_dids) &
               ~np.isin(test_full["t_idx"], EMBARGO_T))

    train_d = slice_dict(train_full, tr_mask)
    select_d = slice_dict(val_full, sel_mask)
    eval_d = slice_dict(test_full, ev_mask)
    del train_full, val_full, test_full
    gc.collect()

    stats = features.fit_dyad_stats(train_d, n_dyads)
    features.apply_dyad_stats(train_d, stats, True)
    features.apply_dyad_stats(select_d, stats, False)
    features.apply_dyad_stats(eval_d, stats, False)

    # Warm-history sufficient statistics for the *secondary* strong baselines.
    # Only pre-embargo history is used: t_idx < 275 (2017-11).  This is
    # observed history at prediction time, not holdout labels.
    con.register("sprint_a_hd_tmp", pd.DataFrame({"did": holdout_dids}))
    warm = con.execute("""
        SELECT did::INTEGER AS did, active::INT AS a, active_next::INT AS y
        FROM panel p JOIN sprint_a_hd_tmp h USING (did)
        WHERE t_idx < 275
    """).fetchall()
    con.unregister("sprint_a_hd_tmp")
    w_n = np.zeros(n_dyads); w_k = np.zeros(n_dyads)
    w_n1 = np.zeros(n_dyads); w_n11 = np.zeros(n_dyads)
    w_n0 = np.zeros(n_dyads); w_n01 = np.zeros(n_dyads)
    for did, a, y in warm:
        d = int(did); a = int(a); y = int(y)
        w_n[d] += 1; w_k[d] += y
        if a == 1:
            w_n1[d] += 1; w_n11[d] += y
        else:
            w_n0[d] += 1; w_n01[d] += y

    a = train_d["active"].astype(np.int64)
    y = train_d["y"].astype(np.int64)
    pg = np.zeros((2, 2))
    for x in (0, 1):
        m = a == x
        pg[x] = (np.bincount(y[m], minlength=2) + 1.0) / (m.sum() + 2.0)

    bundle = {
        "train": train_d, "select": select_d, "eval": eval_d,
        "stats": stats, "n_dyads": n_dyads, "graph_meta": gm,
        "holdout_dids": holdout_dids, "holdout_set": hold_set,
        "warm": {"n": w_n, "k": w_k, "n1": w_n1, "n11": w_n11,
                 "n0": w_n0, "n01": w_n01},
        "global_transition": pg,
        "global_rate": float(stats["global_rate"]),
        "counts": {
            "train": int(len(train_d["y"])), "train_pos": int(train_d["y"].sum()),
            "select": int(len(select_d["y"])), "select_pos": int(select_d["y"].sum()),
            "eval": int(len(eval_d["y"])), "eval_pos": int(eval_d["y"].sum()),
            "eval_dyads": int(len(np.unique(eval_d["did"]))),
            "eval_pairs": int(spec["counts"]["unordered_pairs"]),
            "holdout_dids": int(len(holdout_dids)),
        },
        "assertions": run_holdout_assertions(
            train_d, select_d, eval_d, holdout_dids, n_dyads,
            float(stats["global_rate"])),
    }
    if own:
        con.close()
    log("sprint_a: bundle counts " + json.dumps(bundle["counts"]))
    return bundle


def run_holdout_assertions(train_d, select_d, eval_d, holdout_dids,
                           n_dyads: int, global_rate: float | None = None) -> dict:
    hd = set(int(d) for d in holdout_dids)
    train_set = set(int(d) for d in np.unique(train_d["did"]))
    select_set = set(int(d) for d in np.unique(select_d["did"]))
    eval_set = set(int(d) for d in np.unique(eval_d["did"]))
    checks = {
        "train_holdout_overlap": len(train_set & hd),
        "select_holdout_overlap": len(select_set & hd),
        "eval_nonholdout": len(eval_set - hd),
        "train_rows": int(len(train_d["y"])),
        "select_rows": int(len(select_d["y"])),
        "eval_rows": int(len(eval_d["y"])),
        "train_embargo_rows": int(np.isin(train_d["t_idx"], EMBARGO_T).sum()),
        "select_embargo_rows": int(np.isin(select_d["t_idx"], EMBARGO_T).sum()),
        "eval_embargo_rows": int(np.isin(eval_d["t_idx"], EMBARGO_T).sum()),
        "eval_seen_tr_nonzero": int((eval_d["seen_tr"] > 0).sum()),
        "eval_br_global_rate_max_abs_diff": float(np.max(np.abs(
            eval_d["br"] - float(global_rate if global_rate is not None
                                 else train_d["br"].mean())))),
        "eval_dyads": int(len(eval_set)),
        "n_dyads": int(n_dyads),
    }
    checks["pass"] = bool(
        checks["train_holdout_overlap"] == 0
        and checks["select_holdout_overlap"] == 0
        and checks["eval_nonholdout"] == 0
        and checks["train_embargo_rows"] == 0
        and checks["select_embargo_rows"] == 0
        and checks["eval_embargo_rows"] == 0
        and checks["eval_seen_tr_nonzero"] == 0
    )
    return checks


# ---------------------------------------------------------------------------
# Calendar specifications
# ---------------------------------------------------------------------------
def set_clipped_calendar(X: np.ndarray, names: list[str], clip_value: float) -> np.ndarray:
    """Return a copy of ``X`` with ``t_yr`` clipped at the last train month."""
    X = np.array(X, copy=True)
    j = names.index("t_yr")
    X[:, j] = np.minimum(X[:, j], np.float32(clip_value))
    return X


def year_fe_matrix(X: np.ndarray, names: list[str], t_idx: np.ndarray):
    """Replace ``t_yr`` by year dummies with last-observed-year carry-forward.

    Levels 1996..2014 are estimated on train (1995 is the reference).  Any row
    after 2014 gets the 2014 dummy = 1, i.e. the last observed year effect is
    carried forward.  This is a *no-extrapolation* calendar anchor; it is not
    the frozen 29/49 primary specification.
    """
    j = names.index("t_yr")
    years = 1995 + (np.asarray(t_idx).astype(np.int64) // 12)
    year_levels = list(range(1996, 2015))
    dummies = np.zeros((len(years), len(year_levels)), dtype=np.float32)
    for col, yr in enumerate(year_levels):
        dummies[:, col] = (years == yr).astype(np.float32)
    dummies[years >= 2014, year_levels.index(2014)] = 1.0
    new_names = names[:j] + [f"yr_{y}" for y in year_levels] + names[j + 1:]
    X_new = np.column_stack([X[:, :j], dummies, X[:, j + 1:]])
    return X_new.astype(np.float32, copy=False), new_names


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def baseline_warm_base_rate(eval_d: dict, bundle: dict) -> np.ndarray:
    d = eval_d["did"].astype(np.int64)
    w = bundle["warm"]
    g = float(bundle["global_rate"])
    return (w["k"][d] + 2.0 * g) / (w["n"][d] + 2.0)


def baseline_warm_markov(eval_d: dict, bundle: dict) -> np.ndarray:
    d = eval_d["did"].astype(np.int64)
    a = eval_d["active"].astype(np.int64)
    w = bundle["warm"]
    pg = bundle["global_transition"]
    p1 = (w["n11"][d] + 2.0 * pg[1, 1]) / (w["n1"][d] + 2.0)
    p0 = (w["n01"][d] + 2.0 * pg[0, 1]) / (w["n0"][d] + 2.0)
    return np.where(a == 1, p1, p0).astype(np.float64)


def fit_markov_recency(cfg: dict, train_d: dict, select_d: dict,
                       eval_d: dict) -> dict:
    """Logistic on (active, since_act, log_n_act) -- the memo's baseline."""
    cols = ["active", "since_act", "log_n_act"]
    names = train_d["feature_names"]
    ci = _indices(names, cols)
    fit = models.fit_logistic(train_d["X"][:, ci], train_d["y"],
                              select_d["X"][:, ci], select_d["y"], cfg,
                              feature_names=cols)
    p_sel = models.predict_logistic(fit, select_d["X"][:, ci])
    p_eval = models.predict_logistic(fit, eval_d["X"][:, ci])
    return {"fit": fit, "p_sel": p_sel, "p_eval": p_eval, "cols": cols}


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------
def fit_logistic_spec(cfg: dict, train_d: dict, select_d: dict, eval_d: dict,
                      cols: list[str]) -> dict:
    names = train_d["feature_names"]
    ci = _indices(names, cols)
    if list(cols) == list(names):
        Xtr, Xsel, Xev = train_d["X"], select_d["X"], eval_d["X"]
    else:
        Xtr, Xsel, Xev = (train_d["X"][:, ci], select_d["X"][:, ci],
                          eval_d["X"][:, ci])
    fit = models.fit_logistic(Xtr, train_d["y"], Xsel, select_d["y"], cfg,
                              feature_names=cols)
    p_sel = models.predict_logistic(fit, Xsel)
    p_eval = models.predict_logistic(fit, Xev)
    cal = models.calibrate_all(p_sel, select_d["y"], p_eval, cfg)
    return {"fit": fit, "cols": cols, "ci": ci, "cal": cal,
            "p_sel_raw": p_sel, "p_eval_raw": p_eval}


def fit_hgb_grid(cfg: dict, train_d: dict, select_d: dict, eval_d: dict,
                 cols: list[str], grid: list[dict] | None = None,
                 refine: bool = True) -> dict:
    """HGB with a learning-rate x leaves grid and select-only tuning.

    Deterministic given the data and params (no row subsampling).  The grid is
    at least as wide as the logistic C grid, satisfying "tuning parity".
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    names = train_d["feature_names"]
    ci = _indices(names, cols)
    if list(cols) == list(names):
        Xtr, Xsel, Xev = train_d["X"], select_d["X"], eval_d["X"]
    else:
        Xtr, Xsel, Xev = (train_d["X"][:, ci], select_d["X"][:, ci],
                          eval_d["X"][:, ci])
    if grid is None:
        grid = [
            {"learning_rate": lr, "max_leaf_nodes": leaves, "max_iter": 120}
            for lr in (0.05, 0.10)
            for leaves in (15, 31, 63)
        ]
    rows, best = [], None
    for g in grid:
        clf = HistGradientBoostingClassifier(
            max_iter=int(g["max_iter"]), early_stopping=False,
            learning_rate=float(g["learning_rate"]),
            max_leaf_nodes=int(g["max_leaf_nodes"]),
            max_bins=int(cfg["models"]["hgb"]["max_bins"]),
            min_samples_leaf=int(cfg["models"]["hgb"]["min_samples_leaf"]),
            random_state=int(cfg["seed"]),
        )
        clf.fit(Xtr, train_d["y"])
        ll = binary_metrics(select_d["y"], clf.predict_proba(Xsel)[:, 1])["log_loss"]
        rows.append({**g, "select_log_loss": round(float(ll), 6)})
        if best is None or ll < best[0]:
            best = (ll, g, clf)
    # If the best is on an edge of the max_iter budget, extend it once.
    if refine:
        ll, g, _ = best
        edge = g["max_iter"] in (min(x["max_iter"] for x in grid),
                                 max(x["max_iter"] for x in grid))
        if edge:
            for it in (80, 160, 200):
                if it == g["max_iter"]:
                    continue
                clf = HistGradientBoostingClassifier(
                    max_iter=int(it), early_stopping=False,
                    learning_rate=float(g["learning_rate"]),
                    max_leaf_nodes=int(g["max_leaf_nodes"]),
                    max_bins=int(cfg["models"]["hgb"]["max_bins"]),
                    min_samples_leaf=int(cfg["models"]["hgb"]["min_samples_leaf"]),
                    random_state=int(cfg["seed"]),
                )
                clf.fit(Xtr, train_d["y"])
                v = binary_metrics(select_d["y"], clf.predict_proba(Xsel)[:, 1])["log_loss"]
                rows.append({**g, "max_iter": int(it),
                             "select_log_loss": round(float(v), 6)})
                if v < best[0]:
                    best = (v, {**g, "max_iter": int(it)}, clf)
    ll, g, clf = best
    p_sel = clf.predict_proba(Xsel)[:, 1]
    p_eval = clf.predict_proba(Xev)[:, 1]
    cal = models.calibrate_all(p_sel, select_d["y"], p_eval, cfg)
    return {"fit": {"kind": "hgb", "model": clf, "params": g,
                    "select_log_loss": float(ll), "coef": None, "intercept": None},
            "cols": cols, "ci": ci, "cal": cal, "grid": rows,
            "p_sel_raw": p_sel, "p_eval_raw": p_eval}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def ece_equal_frequency(y, p, bins: int = 10) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    edges = np.quantile(p, np.linspace(0.0, 1.0, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += float(m.mean()) * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(ece)


def metrics_block(y, p) -> dict:
    m = binary_metrics(y, p)
    m["ece_equal_frequency"] = round(ece_equal_frequency(y, p), 6)
    m.update(reliability_slope(y, p))
    return m


def block_metrics_for_preds(y, preds: dict) -> dict:
    return {k: metrics_block(y, v) for k, v in preds.items()}
