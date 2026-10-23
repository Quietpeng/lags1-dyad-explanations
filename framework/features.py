"""Feature construction for the two panels.

Two contracts are enforced everywhere:

1. **Causality**: every feature for a row at month ``t`` is a function of
   months ``<= t``; the label is month ``t+1``.  Lags/rolling/cumulative
   windows are backward-looking only.  Graph features use edges at ``<= t``.
2. **Train-only statistics**: the only learned per-entity statistics are
   ``br`` (dyad base rate) and ``seen_tr``.  They are computed from a caller
   supplied *statistics mask* (default: train only, leave-one-out inside the
   statistics rows), never from validation/test labels.

Feature sets
------------
``DYAD_FEATURES``     29 features, identical names to ``g2_cpu_baselines.BIN_FEATS``.
``GRAPH_FEATURES``    35 graph features, names/formulas copied from ``graph_probe.py``.
``GRAPH_CORE20``      20-feature subset used by the pilot (all 35 available).
``ACTOR_FEATURES``    actor-month T1 features (activity, degree, history, calendar,
                      region fixed effects).
"""

from __future__ import annotations

import gc
import os

import duckdb
import numpy as np

from .utils import jdump, log, sha256_file

# ---------------------------------------------------------------------------
# Frozen feature lists
# ---------------------------------------------------------------------------
DYAD_FEATURES = [
    "active", "log_ev",
    "a1", "a2", "a3", "a4", "a5", "a6",
    "le1", "le2", "le3",
    "am3", "am6", "am12", "les3", "les6", "les12",
    "since_act", "since_first", "log_n_act",
    "rev_act", "rev_am3", "src_ev", "tgt_ev",
    "br", "seen_tr", "t_yr", "mo_sin", "mo_cos",
]
DYAD_CAUSAL = [c for c in DYAD_FEATURES if c not in ("br", "seen_tr")]

DYAD_GROUPS = {
    "calendar": ["t_yr", "mo_sin", "mo_cos"],
    "dyad_current": ["active", "log_ev"],
    "dyad_lags": ["a1", "a2", "a3", "a4", "a5", "a6", "le1", "le2", "le3"],
    "dyad_rolling": ["am3", "am6", "am12", "les3", "les6", "les12"],
    "dyad_history": ["since_act", "since_first", "log_n_act"],
    "reverse_dyad": ["rev_act", "rev_am3"],
    "actor_ego": ["src_ev", "tgt_ev"],
    "train_stats": ["br", "seen_tr"],
}

GRAPH_FEATURES = [
    "g_deg_s", "g_deg_t", "g_wdeg_s", "g_wdeg_t",
    "g_deg12_s", "g_deg12_t", "g_wdeg12_s", "g_wdeg12_t",
    "g_eig_s", "g_eig_t", "g_pr_s", "g_pr_t",
    "g_clust_s", "g_clust_t", "g_nbdeg_s", "g_nbdeg_t",
    "g_newnb_s", "g_newnb_t",
    "g_cn", "g_jaccard", "g_aa", "g_ra", "g_tri_ratio", "g_pa",
    "g_two_hop",
    "g_cn12", "g_jaccard12", "g_aa12", "g_ra12",
    "g_nra_s", "g_nra_t", "g_nra12_s", "g_nra12_t",
    "g_nra_norm_s", "g_nra_norm_t",
]
GRAPH_LOG = {
    "g_deg_s", "g_deg_t", "g_wdeg_s", "g_wdeg_t",
    "g_deg12_s", "g_deg12_t", "g_wdeg12_s", "g_wdeg12_t",
    "g_nbdeg_s", "g_nbdeg_t", "g_newnb_s", "g_newnb_t",
    "g_cn", "g_aa", "g_ra", "g_pa", "g_cn12", "g_aa12", "g_ra12",
    "g_nra_s", "g_nra_t", "g_nra12_s", "g_nra12_t",
}
GRAPH_CORE20 = [
    "g_deg_s", "g_deg_t", "g_wdeg_s", "g_wdeg_t",
    "g_deg12_s", "g_deg12_t", "g_eig_s", "g_eig_t",
    "g_pr_s", "g_pr_t", "g_cn", "g_jaccard", "g_aa", "g_ra",
    "g_two_hop", "g_cn12", "g_aa12", "g_ra12", "g_nra_s", "g_nra_t",
]
GRAPH_GROUPS = {
    "graph_degree": ["g_deg_s", "g_deg_t", "g_wdeg_s", "g_wdeg_t",
                     "g_deg12_s", "g_deg12_t", "g_wdeg12_s", "g_wdeg12_t"],
    "graph_centrality": ["g_eig_s", "g_eig_t", "g_pr_s", "g_pr_t",
                         "g_clust_s", "g_clust_t"],
    "graph_neighborhood": ["g_nbdeg_s", "g_nbdeg_t", "g_newnb_s", "g_newnb_t",
                           "g_cn", "g_jaccard", "g_aa", "g_ra", "g_tri_ratio",
                           "g_pa", "g_two_hop"],
    "graph_recent": ["g_cn12", "g_jaccard12", "g_aa12", "g_ra12",
                     "g_nra_s", "g_nra_t", "g_nra12_s", "g_nra12_t",
                     "g_nra_norm_s", "g_nra_norm_t"],
}

ACTOR_FEATURES = [
    "out_active", "in_active", "active_any",
    "log_out_ev", "log_in_ev", "log_tot_ev",
    "out_partners", "in_partners", "out_partners_chg",
    "a1_out", "a2_out", "a3_out", "a4_out", "a5_out", "a6_out",
    "le1_out", "le2_out", "le3_out",
    "oam3", "oam6", "oam12", "oes3", "oes6", "oes12",
    "since_out", "since_first", "log_n_out_cum",
    "t_yr", "mo_sin", "mo_cos",
    "region_Asia", "region_Europe", "region_LatinAmericaCaribbean",
    "region_MiddleEast", "region_NorthAmerica", "region_Oceania",
]
ACTOR_GROUPS = {
    "actor_current": ["out_active", "in_active", "active_any",
                      "log_out_ev", "log_in_ev", "log_tot_ev"],
    "actor_degree": ["out_partners", "in_partners", "out_partners_chg"],
    "actor_lags": ["a1_out", "a2_out", "a3_out", "a4_out", "a5_out", "a6_out",
                   "le1_out", "le2_out", "le3_out"],
    "actor_rolling": ["oam3", "oam6", "oam12", "oes3", "oes6", "oes12"],
    "actor_history": ["since_out", "since_first", "log_n_out_cum"],
    "calendar": ["t_yr", "mo_sin", "mo_cos"],
    "region": [c for c in ACTOR_FEATURES if c.startswith("region_")],
}

LABELS = {"active_next", "root_next", "y", "out_active_next", "in_active_next",
          "out_events_next", "op_next"}


def assert_no_label(names) -> None:
    bad = sorted(set(names) & LABELS)
    if bad:
        raise AssertionError(f"label columns leaked into features: {bad}")


def feature_manifest() -> dict:
    assert_no_label(DYAD_FEATURES)
    assert_no_label(ACTOR_FEATURES)
    assert_no_label(GRAPH_FEATURES)
    return {
        "dyad": [{"name": n, "group": g, "available_at": "t"}
                 for g, cols in DYAD_GROUPS.items() for n in cols],
        "graph": [{"name": n, "group": next((g for g, c in GRAPH_GROUPS.items() if n in c),
                                            "graph_other"),
                   "available_at": "t"} for n in GRAPH_FEATURES],
        "actor": [{"name": n, "group": next((g for g, c in ACTOR_GROUPS.items() if n in c),
                                            "actor_other"),
                   "available_at": "t"} for n in ACTOR_FEATURES],
        "excluded_labels": sorted(LABELS),
        "train_only_statistics": ["br", "seen_tr"],
    }


# ---------------------------------------------------------------------------
# SQL builders (work for real and synthetic panels)
# ---------------------------------------------------------------------------
def _drop(con, name: str) -> None:
    """Drop a table or view regardless of which one currently exists."""
    for kind in ("TABLE", "VIEW"):
        try:
            con.execute(f"DROP {kind} IF EXISTS {name}")
        except Exception:
            pass


def build_dyad_base(con: duckdb.DuckDBPyConnection, cfg: dict,
                    force: bool = False, tag: str = "") -> str:
    """Create ``fb`` from a ``panel`` table/view (causal features only).

    The result is persisted to ``cache/features_dyad{tag}.parquet`` so later
    connections can reuse it without rebuilding the 81 s window-function pass.
    """
    cache = os.path.join(cfg["paths"]["cache_dir"], f"features_dyad{tag}.parquet")
    if not force and os.path.exists(cache):
        log("features: dyad base cache present")
        _drop(con, "fb")
        con.execute(f"CREATE OR REPLACE VIEW fb AS SELECT * FROM read_parquet('{cache}')")
        return cache
    _drop(con, "fb")
    log("features: dyad window functions")
    con.execute("""
      CREATE OR REPLACE TABLE f AS
      SELECT s, t, m, t_idx, n_events, active, active_next, root, root_next,
             split, did, sid, tid,
        coalesce(lag(active,1) OVER w, false) AS a1,
        coalesce(lag(active,2) OVER w, false) AS a2,
        coalesce(lag(active,3) OVER w, false) AS a3,
        coalesce(lag(active,4) OVER w, false) AS a4,
        coalesce(lag(active,5) OVER w, false) AS a5,
        coalesce(lag(active,6) OVER w, false) AS a6,
        coalesce(lag(n_events,1) OVER w, 0) AS e1,
        coalesce(lag(n_events,2) OVER w, 0) AS e2,
        coalesce(lag(n_events,3) OVER w, 0) AS e3,
        avg(active::INT) OVER (w ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS am3,
        avg(active::INT) OVER (w ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS am6,
        avg(active::INT) OVER (w ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS am12,
        sum(n_events) OVER (w ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS es3,
        sum(n_events) OVER (w ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS es6,
        sum(n_events) OVER (w ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS es12,
        max(CASE WHEN active THEN t_idx END) OVER (
            w ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS last_act,
        min(CASE WHEN active THEN t_idx END) OVER (
            w ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS first_act,
        sum(CASE WHEN active THEN 1 ELSE 0 END) OVER (
            w ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS n_act_cum
      FROM panel
      WINDOW w AS (PARTITION BY s, t ORDER BY t_idx)
    """)
    con.execute("""
      CREATE OR REPLACE TABLE rev AS
      SELECT s, t, m, active::INT AS r_active,
             avg(active::INT) OVER (PARTITION BY s, t ORDER BY m
                                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS r_am3
      FROM panel
    """)
    con.execute("""
      CREATE OR REPLACE TABLE act_m AS
      SELECT actor, m, sum(ev) AS ev FROM (
        SELECT s AS actor, m, sum(n_events) AS ev FROM panel GROUP BY 1,2
        UNION ALL
        SELECT t AS actor, m, sum(n_events) AS ev FROM panel GROUP BY 1,2)
      GROUP BY 1,2
    """)
    con.execute("""
      CREATE OR REPLACE TABLE fb AS
      SELECT f.*,
        ln(1+n_events)::DOUBLE AS log_ev,
        ln(1+e1)::DOUBLE AS le1, ln(1+e2)::DOUBLE AS le2, ln(1+e3)::DOUBLE AS le3,
        ln(1+es3)::DOUBLE AS les3, ln(1+es6)::DOUBLE AS les6,
        ln(1+es12)::DOUBLE AS les12,
        least(coalesce(t_idx-last_act, 999), 24)::DOUBLE AS since_act,
        least(coalesce(t_idx-first_act, 999), 120)::DOUBLE AS since_first,
        ln(1+n_act_cum)::DOUBLE AS log_n_act,
        coalesce(ra.r_active, 0)::DOUBLE AS rev_act,
        coalesce(ra.r_am3, 0)::DOUBLE AS rev_am3,
        ln(1+coalesce(sm.ev,0))::DOUBLE AS src_ev,
        ln(1+coalesce(tm.ev,0))::DOUBLE AS tgt_ev,
        (t_idx/12.0)::DOUBLE AS t_yr,
        sin(2*pi()*month(f.m)/12.0)::DOUBLE AS mo_sin,
        cos(2*pi()*month(f.m)/12.0)::DOUBLE AS mo_cos
      FROM f
      LEFT JOIN rev ra ON ra.s=f.t AND ra.t=f.s AND ra.m=f.m
      LEFT JOIN act_m sm ON sm.actor=f.s AND sm.m=f.m
      LEFT JOIN act_m tm ON tm.actor=f.t AND tm.m=f.m
    """)
    con.execute("DROP TABLE IF EXISTS f; DROP TABLE IF EXISTS rev; DROP TABLE IF EXISTS act_m;")
    con.execute(f"""
      COPY (SELECT * FROM fb ORDER BY t_idx, did)
      TO '{cache}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    log(f"features: dyad base written to {cache}")
    _drop(con, "fb")
    con.execute(f"CREATE OR REPLACE VIEW fb AS SELECT * FROM read_parquet('{cache}')")
    return cache


def build_actor_base(con: duckdb.DuckDBPyConnection, cfg: dict,
                     force: bool = False, tag: str = "") -> str:
    """Create ``ab`` from an ``actor_month`` table/view."""
    cache = os.path.join(cfg["paths"]["cache_dir"], f"features_actor{tag}.parquet")
    if not force and os.path.exists(cache):
        log("features: actor base cache present")
        _drop(con, "ab")
        con.execute(f"CREATE OR REPLACE VIEW ab AS SELECT * FROM read_parquet('{cache}')")
        return cache
    _drop(con, "ab")
    log("features: actor-month window functions")
    con.execute("""
      CREATE OR REPLACE TABLE af AS
      SELECT *,
        lag(out_active,1) OVER w AS a1_out,
        lag(out_active,2) OVER w AS a2_out,
        lag(out_active,3) OVER w AS a3_out,
        lag(out_active,4) OVER w AS a4_out,
        lag(out_active,5) OVER w AS a5_out,
        lag(out_active,6) OVER w AS a6_out,
        lag(out_events,1) OVER w AS e1_out,
        lag(out_events,2) OVER w AS e2_out,
        lag(out_events,3) OVER w AS e3_out,
        lag(out_partners,1) OVER w AS p1_out,
        avg(out_active::INT) OVER (w ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS oam3,
        avg(out_active::INT) OVER (w ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS oam6,
        avg(out_active::INT) OVER (w ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS oam12,
        sum(out_events) OVER (w ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS oes3_raw,
        sum(out_events) OVER (w ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS oes6_raw,
        sum(out_events) OVER (w ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS oes12_raw,
        max(CASE WHEN out_active THEN t_idx END) OVER (
            w ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS last_out,
        min(CASE WHEN out_active THEN t_idx END) OVER (
            w ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS first_out,
        sum(CASE WHEN out_active THEN 1 ELSE 0 END) OVER (
            w ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS n_out_cum
      FROM actor_month
      WINDOW w AS (PARTITION BY actor ORDER BY t_idx)
    """)
    con.execute("""
      CREATE OR REPLACE TABLE ab AS
      SELECT actor, aid, region, m, t_idx, split,
        out_active, in_active, (out_active OR in_active) AS active_any,
        out_events, in_events,
        ln(1+out_events)::DOUBLE AS log_out_ev,
        ln(1+in_events)::DOUBLE AS log_in_ev,
        ln(1+out_events+in_events)::DOUBLE AS log_tot_ev,
        out_partners::DOUBLE AS out_partners,
        in_partners::DOUBLE AS in_partners,
        coalesce(out_partners - p1_out, 0)::DOUBLE AS out_partners_chg,
        coalesce(a1_out, false) AS a1_out, coalesce(a2_out, false) AS a2_out,
        coalesce(a3_out, false) AS a3_out, coalesce(a4_out, false) AS a4_out,
        coalesce(a5_out, false) AS a5_out, coalesce(a6_out, false) AS a6_out,
        ln(1+coalesce(e1_out,0))::DOUBLE AS le1_out,
        ln(1+coalesce(e2_out,0))::DOUBLE AS le2_out,
        ln(1+coalesce(e3_out,0))::DOUBLE AS le3_out,
        oam3, oam6, oam12,
        ln(1+oes3_raw)::DOUBLE AS oes3,
        ln(1+oes6_raw)::DOUBLE AS oes6,
        ln(1+oes12_raw)::DOUBLE AS oes12,
        least(coalesce(t_idx-last_out, 999), 24)::DOUBLE AS since_out,
        least(coalesce(t_idx-first_out, 999), 120)::DOUBLE AS since_first,
        ln(1+n_out_cum)::DOUBLE AS log_n_out_cum,
        (t_idx/12.0)::DOUBLE AS t_yr,
        sin(2*pi()*month(m)/12.0)::DOUBLE AS mo_sin,
        cos(2*pi()*month(m)/12.0)::DOUBLE AS mo_cos,
        (region='Asia')::INT AS region_Asia,
        (region='Europe')::INT AS region_Europe,
        (region='LatinAmericaCaribbean')::INT AS region_LatinAmericaCaribbean,
        (region='MiddleEast')::INT AS region_MiddleEast,
        (region='NorthAmerica')::INT AS region_NorthAmerica,
        (region='Oceania')::INT AS region_Oceania,
        out_active_next, in_active_next, out_events_next
      FROM af
    """)
    con.execute("DROP TABLE IF EXISTS af;")
    con.execute(f"""
      COPY (SELECT * FROM ab ORDER BY aid, t_idx)
      TO '{cache}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    log(f"features: actor base written to {cache}")
    _drop(con, "ab")
    con.execute(f"CREATE OR REPLACE VIEW ab AS SELECT * FROM read_parquet('{cache}')")
    return cache


# ---------------------------------------------------------------------------
# Graph features (numpy; same formulas as graph_probe.py)
# ---------------------------------------------------------------------------
def _eigen_centrality(Bf: np.ndarray, eye: np.ndarray) -> np.ndarray:
    A = Bf + eye
    v = np.ones(A.shape[0], dtype=np.float32) / A.shape[0]
    for _ in range(40):
        v = A @ v
        n = float(np.linalg.norm(v))
        if n <= 0:
            break
        v /= n
    return (v / max(float(v.sum()), 1e-12)).astype(np.float32)


def _pagerank(Bf: np.ndarray, damping: float = 0.85, iters: int = 30) -> np.ndarray:
    n = Bf.shape[0]
    deg = Bf.sum(1)
    v = np.ones(n, dtype=np.float32) / n
    for _ in range(iters):
        v = (1.0 - damping) / n + damping * (Bf @ (v / np.maximum(deg, 1.0)))
        v /= max(float(v.sum()), 1e-12)
    return v.astype(np.float32)


def build_graph_features(con: duckdb.DuckDBPyConnection, cfg: dict,
                         force: bool = False, tag: str = "",
                         panel_view: str = "panel") -> dict:
    """Compute graph features for every dyad-month row and save split arrays.

    Reads ``panel`` (or ``panel_view``) ordered by ``(t_idx, did)``; the returned
    arrays are in that same order, and :func:`export_dyad_split` uses the same
    order.

    ``tag`` namespaces the cache files (``graph_<split><tag>.npy``,
    ``graph_meta<tag>.json``).  The default empty tag is the frozen P1 cache and
    must not be overwritten by a diagnostic run.  ``panel_view`` lets a caller
    supply a view with held-out dyad edges zeroed (sprint-A holdout); the
    default ``panel`` keeps the frozen behaviour byte-for-byte.
    """
    cache = cfg["paths"]["cache_dir"]
    meta_path = os.path.join(cache, f"graph_meta{tag}.json")
    names = (GRAPH_FEATURES if cfg["features"]["graph"] == "all35"
             else GRAPH_CORE20 if cfg["features"]["graph"] == "core20" else [])
    if not names:
        return {"features": [], "path": None}
    split_paths = {s: os.path.join(cache, f"graph_{s}{tag}.npy")
                   for s in ("train", "val", "test")}
    if not force and os.path.exists(meta_path) and all(os.path.exists(p) for p in split_paths.values()):
        from .utils import jload
        meta = jload(meta_path)
        if meta.get("features") == names and meta.get("panel_view", "panel") == panel_view:
            log("features: graph cache present")
            return meta

    log("features: loading panel arrays for graph snapshots")
    res = con.execute(f"""
      SELECT did::INTEGER AS did, sid::SMALLINT AS sid, tid::SMALLINT AS tid,
             t_idx::SMALLINT AS t_idx, n_events::FLOAT AS n_events,
             active::TINYINT AS act
      FROM {panel_view} ORDER BY t_idx, did
    """).fetchnumpy()
    did = res["did"].astype(np.int32)
    sid = res["sid"].astype(np.int16)
    tid = res["tid"].astype(np.int16)
    t_idx = res["t_idx"].astype(np.int16)
    n_events = res["n_events"].astype(np.float32)
    act = res["act"].astype(np.uint8)
    A = int(max(sid.max(), tid.max())) + 1
    M = int(t_idx.max()) + 1
    log(f"features: graph snapshots A={A} M={M} rows={len(did):,}")

    act_m = np.zeros((M, A, A), dtype=np.uint8)
    ev_m = np.zeros((M, A, A), dtype=np.float32)
    act_m[t_idx, sid, tid] = act
    ev_m[t_idx, sid, tid] = n_events
    B_all = act_m | act_m.transpose(0, 2, 1)
    W_all = ev_m + ev_m.transpose(0, 2, 1)
    del act_m, ev_m
    gc.collect()
    Cnt = np.cumsum(B_all, axis=0, dtype=np.uint16)
    WCum = np.cumsum(W_all, axis=0, dtype=np.float32)
    del B_all, W_all
    gc.collect()

    G = np.empty((len(t_idx), len(names)), dtype=np.float32)
    order = np.argsort(t_idx, kind="stable")
    t_sorted = t_idx[order]
    starts = np.searchsorted(t_sorted, np.arange(M + 1))
    eye = np.eye(A, dtype=np.float32)
    zero = np.zeros((A, A), dtype=np.float32)

    for m in range(M):
        lo, hi = int(starts[m]), int(starts[m + 1])
        if hi <= lo:
            continue
        rows = order[lo:hi]
        Cm = Cnt[m].astype(np.float32)
        Bc = Cm > 0
        Cp = Cnt[m - 12].astype(np.float32) if m >= 12 else zero
        Cr = Cm - Cp
        B12 = Cr > 0
        Cp2 = Cnt[m - 24].astype(np.float32) if m >= 24 else zero
        Bprev = (Cp - Cp2) > 0
        Wm = WCum[m]
        Wp = WCum[m - 12] if m >= 12 else 0.0
        W12 = Wm - Wp

        deg = Bc.sum(1).astype(np.float32)
        wdeg = Wm.sum(1)
        deg12 = B12.sum(1).astype(np.float32)
        wdeg12 = W12.sum(1)
        Bf = Bc.astype(np.float32)
        B12f = B12.astype(np.float32)
        BB = Bf @ Bf
        BB12 = B12f @ B12f
        invlog = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 2.0)), 0.0).astype(np.float32)
        invdeg = (1.0 / np.maximum(deg, 1.0)).astype(np.float32)
        invlog12 = np.where(deg12 > 1, 1.0 / np.log(np.maximum(deg12, 2.0)), 0.0).astype(np.float32)
        invdeg12 = (1.0 / np.maximum(deg12, 1.0)).astype(np.float32)
        AA = (Bf * invlog[None, :]) @ Bf
        RA = (Bf * invdeg[None, :]) @ Bf
        AA12 = (B12f * invlog12[None, :]) @ B12f
        RA12 = (B12f * invdeg12[None, :]) @ B12f
        NRA = Bf @ Cr
        NRA12 = B12f @ Cr
        Bdeg = Bf @ deg
        Bdeg12 = B12f @ deg12
        eig = _eigen_centrality(Bf, eye)
        pr = _pagerank(Bf)
        tri = (BB * Bf).sum(1)
        denom = deg * (deg - 1.0)
        clust = np.where(denom > 0, tri / np.maximum(denom, 1.0), 0.0).astype(np.float32)
        newnb = (B12 & ~Bprev).sum(1).astype(np.float32)

        s = sid[rows]
        t = tid[rows]
        ds, dt = deg[s], deg[t]
        ds12, dt12 = deg12[s], deg12[t]
        cn = BB[s, t]
        cn12 = BB12[s, t]
        edge = Bc[s, t]
        vals = {
            "g_deg_s": ds, "g_deg_t": dt,
            "g_wdeg_s": wdeg[s], "g_wdeg_t": wdeg[t],
            "g_deg12_s": ds12, "g_deg12_t": dt12,
            "g_wdeg12_s": wdeg12[s], "g_wdeg12_t": wdeg12[t],
            "g_eig_s": eig[s], "g_eig_t": eig[t],
            "g_pr_s": pr[s], "g_pr_t": pr[t],
            "g_clust_s": clust[s], "g_clust_t": clust[t],
            "g_nbdeg_s": Bdeg[s] / np.maximum(ds, 1.0),
            "g_nbdeg_t": Bdeg[t] / np.maximum(dt, 1.0),
            "g_newnb_s": newnb[s], "g_newnb_t": newnb[t],
            "g_cn": cn,
            "g_jaccard": cn / np.maximum(ds + dt - cn, 1.0),
            "g_aa": AA[s, t], "g_ra": RA[s, t],
            "g_tri_ratio": cn / np.maximum(np.minimum(ds, dt), 1.0),
            "g_pa": ds * dt,
            "g_two_hop": ((~edge) & (cn > 0)).astype(np.float32),
            "g_cn12": cn12,
            "g_jaccard12": cn12 / np.maximum(ds12 + dt12 - cn12, 1.0),
            "g_aa12": AA12[s, t], "g_ra12": RA12[s, t],
            "g_nra_s": NRA[s, t], "g_nra_t": NRA[t, s],
            "g_nra12_s": NRA12[s, t], "g_nra12_t": NRA12[t, s],
            "g_nra_norm_s": NRA[s, t] / np.maximum(ds, 1.0),
            "g_nra_norm_t": NRA[t, s] / np.maximum(dt, 1.0),
        }
        block = np.column_stack([vals[k].astype(np.float32) for k in names])
        for j, k in enumerate(names):
            if k in GRAPH_LOG:
                block[:, j] = np.log1p(np.maximum(block[:, j], 0.0))
        G[rows] = block
        if (m + 1) % 60 == 0 or m == M - 1:
            log(f"features: graph month {m + 1}/{M} rows={hi - lo}")

    del Cnt, WCum
    gc.collect()
    train = t_idx < _month_idx(cfg["split"]["train_end"])
    val = (t_idx >= _month_idx(cfg["split"]["train_end"])) & (
        t_idx < _month_idx(cfg["split"]["val_end"]))
    test = ~train & ~val
    for s, mask in (("train", train), ("val", val), ("test", test)):
        np.save(split_paths[s], G[mask])
    meta = {
        "features": names,
        "all_features": GRAPH_FEATURES,
        "tag": tag,
        "panel_view": panel_view,
        "rows_total": int(len(t_idx)),
        "rows_train": int(train.sum()),
        "rows_val": int(val.sum()),
        "rows_test": int(test.sum()),
        "actors": A, "months": M,
        "order": "panel ORDER BY t_idx, did",
        "cutoff": "edges at month <= t inclusive; label t+1",
        "graph": "undirected actor-actor; edge if either direction active; events summed",
        "log_features": sorted(GRAPH_LOG & set(names)),
        "train_mean": np.asarray(G[train].mean(0)).round(6).tolist(),
        "train_std": np.asarray(G[train].std(0)).round(6).tolist(),
        "files": {s: sha256_file(p) for s, p in split_paths.items()},
    }
    jdump(meta_path, meta)
    log(f"features: graph done; {len(names)} features")
    return meta


def _month_idx(date_str: str) -> int:
    y, m, _ = date_str.split("-")
    return (int(y) - 1995) * 12 + (int(m) - 1)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def load_graph_split(cfg: dict, split: str, names: list[str],
                     tag: str = "") -> np.ndarray:
    path = os.path.join(cfg["paths"]["cache_dir"], f"graph_{split}{tag}.npy")
    if not os.path.exists(path) or not names:
        return np.zeros((0, 0), dtype=np.float32)
    return np.load(path, mmap_mode="r")


def export_dyad_split(con: duckdb.DuckDBPyConnection, cfg: dict, split: str,
                      graph_meta: dict, graph_tag: str = "",
                      label_mode: str = "full") -> dict:
    """Export one split.  ``label_mode='blind'`` returns y=0 on the test split.

    Blind mode is used to validate the sprint-A holdout pipeline (graph rebuild,
    masks, br, assertions) before the one-shot label read.  It never selects
    ``active_next`` for the test split.
    """
    names = graph_meta.get("features", [])
    cols = ", ".join(f"coalesce({c}::DOUBLE, 0.0) AS {c}" for c in DYAD_CAUSAL)
    if split == "test" and label_mode == "blind":
        y_expr, root_expr = "0::INT AS y", "NULL AS root_next"
    else:
        y_expr, root_expr = "active_next::INT AS y", "root_next"
    res = con.execute(f"""
      SELECT {cols}, did, sid, tid, t_idx, active, {y_expr},
             {root_expr}, n_events
      FROM fb WHERE split='{split}' ORDER BY t_idx, did
    """).fetchnumpy()
    X = np.column_stack([res[c].astype(np.float32) for c in DYAD_CAUSAL])
    if names:
        G = load_graph_split(cfg, split, names, tag=graph_tag)
        if G.shape[0] != X.shape[0]:
            raise AssertionError(f"graph rows {G.shape[0]} != base rows {X.shape[0]} for {split}")
        X = np.column_stack([X, np.asarray(G, dtype=np.float32)])
    X = np.column_stack([X, np.zeros(X.shape[0], dtype=np.float32),
                         np.zeros(X.shape[0], dtype=np.float32)])
    out = {
        "X": X.astype(np.float32, copy=False),
        "feature_names": DYAD_CAUSAL + names + ["br", "seen_tr"],
        "y": res["y"].astype(np.int64),
        "did": res["did"].astype(np.int64),
        "sid": res["sid"].astype(np.int16),
        "tid": res["tid"].astype(np.int16),
        "t_idx": res["t_idx"].astype(np.int16),
        "active": res["active"].astype(np.int8),
        "root_next": res["root_next"],
        "n_events": res["n_events"].astype(np.float32),
    }
    return out


def fit_dyad_stats(train_d: dict, n_dyads: int,
                   exclude_mask: np.ndarray | None = None) -> dict:
    """Estimate per-dyad train counts from a *train* split dictionary.

    ``exclude_mask`` (over train rows) removes held-out actors/dyads from the
    statistics, so a cold-start regime cannot leak its own history into ``br``.
    """
    y = train_d["y"].astype(np.float64)
    did = train_d["did"]
    mask = np.ones(len(y), dtype=bool)
    if exclude_mask is not None:
        mask &= ~np.asarray(exclude_mask, dtype=bool)
    if not mask.any():
        raise AssertionError("empty dyad-statistics mask")
    n_tr = np.bincount(did[mask], minlength=n_dyads).astype(np.float64)
    k_tr = np.bincount(did[mask], weights=y[mask], minlength=n_dyads)
    return {
        "n_tr": n_tr,
        "k_tr": k_tr,
        "global_rate": float(y[mask].mean()),
        "n_stats_rows": int(mask.sum()),
        "n_stats_dyads": int((n_tr > 0).sum()),
    }


def apply_dyad_stats(d: dict, stats: dict, is_train: bool) -> dict:
    """Add ``br`` and ``seen_tr`` to a split using precomputed train counts."""
    y = d["y"].astype(np.float64)
    did = d["did"]
    n_tr, k_tr = stats["n_tr"], stats["k_tr"]
    global_rate = float(stats["global_rate"])
    n_row = n_tr[did]
    k_row = k_tr[did]
    br = np.full(len(y), global_rate, dtype=np.float64)
    if is_train:
        br = (k_row - y) / np.maximum(n_row - 1.0, 1.0)
    else:
        br = k_row / np.maximum(n_row, 1.0)
    br[n_row <= 0] = global_rate
    seen = (n_row > 0).astype(np.float64)
    names = list(d["feature_names"])
    if "br" in names and "seen_tr" in names:
        j_br, j_seen = names.index("br"), names.index("seen_tr")
        d["X"][:, j_br] = br.astype(np.float32)
        d["X"][:, j_seen] = seen.astype(np.float32)
    else:
        names = names + ["br", "seen_tr"]
        d["X"] = np.column_stack([d["X"], br.astype(np.float32),
                                  seen.astype(np.float32)]).astype(np.float32)
    d["feature_names"] = names
    d["br"] = br.astype(np.float32)
    d["seen_tr"] = seen.astype(np.float32)
    d["global_rate"] = global_rate
    return d


def export_actor_split(con: duckdb.DuckDBPyConnection, split: str) -> dict:
    cols = ", ".join(f"coalesce({c}::DOUBLE, 0.0) AS {c}" for c in ACTOR_FEATURES)
    res = con.execute(f"""
      SELECT {cols}, actor, aid, region, t_idx, out_active_next::INT AS y,
             out_events_next, out_events, out_active, out_partners, in_partners
      FROM ab WHERE split='{split}' AND out_active_next IS NOT NULL
      ORDER BY aid, t_idx
    """).fetchnumpy()
    X = np.column_stack([res[c].astype(np.float32) for c in ACTOR_FEATURES])
    return {
        "X": X,
        "y": res["y"].astype(np.int64),
        "actor": res["actor"],
        "aid": res["aid"].astype(np.int16),
        "region": res["region"],
        "t_idx": res["t_idx"].astype(np.int16),
        "out_events_next": res["out_events_next"].astype(np.float32),
        "out_events": res["out_events"].astype(np.float32),
        "out_active": res["out_active"].astype(np.int8),
        "out_partners": res["out_partners"].astype(np.float32),
        "in_partners": res["in_partners"].astype(np.float32),
        "feature_names": list(ACTOR_FEATURES),
    }


def export_actor_all(con: duckdb.DuckDBPyConnection) -> dict:
    """All actor-months (including the right-censored last month) for graph use."""
    cols = ", ".join(f"coalesce({c}::DOUBLE, 0.0) AS {c}" for c in ACTOR_FEATURES)
    res = con.execute(f"""
      SELECT {cols}, actor, aid, region, t_idx, out_active_next,
             out_events, out_active, out_partners
      FROM ab ORDER BY aid, t_idx
    """).fetchnumpy()
    X = np.column_stack([res[c].astype(np.float32) for c in ACTOR_FEATURES])
    return {
        "X": X, "actor": res["actor"], "aid": res["aid"].astype(np.int16),
        "region": res["region"], "t_idx": res["t_idx"].astype(np.int16),
        "out_active_next": res["out_active_next"],
        "out_events": res["out_events"].astype(np.float32),
        "out_active": res["out_active"].astype(np.int8),
        "out_partners": res["out_partners"].astype(np.float32),
        "feature_names": list(ACTOR_FEATURES),
    }


def export_panel_arrays(con: duckdb.DuckDBPyConnection) -> dict:
    res = con.execute("""
      SELECT did::INTEGER AS did, sid::SMALLINT AS sid, tid::SMALLINT AS tid,
             t_idx::SMALLINT AS t_idx, active::TINYINT AS active,
             n_events::FLOAT AS n_events, s AS s_iso, t AS t_iso
      FROM panel ORDER BY t_idx, did
    """).fetchnumpy()
    return {k: res[k] for k in res}
