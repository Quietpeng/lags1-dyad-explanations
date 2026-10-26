#!/usr/bin/env python3
"""Illustrative world-line preview for the dyadic forecast framework.

This is a **communication and debugging device**, not a validated multi-step
forecast.  It draws one sampled state path for a small, deliberately selected
set of dyads over two steps, using only the frozen inner-validation period
(2016-01..2017-11).  No test label is read.  No accuracy is computed.

Protocol
--------
anchor t
    Observed dyad features and the frozen one-step logistic model give p(t+1).
    One state is sampled for each dyad.
step t+1 -> t+2
    The dynamic state/lag/rolling features are updated from the sampled state;
    event magnitude is held at the anchor value (or the train mean positive
    event count when the anchor is zero).  Graph, reverse-dyad, actor-ego and
    train-only statistics are frozen at the anchor.  This is a **state-lag-only
    approximation**, not a generative feature model.

Outputs
-------
worldline_preview.png / .pdf
worldline_trace.json         metadata, per-anchor and aggregate transition tallies
run_manifest.json            command, versions, hashes, runtime, preview scope

The per-row exports (``worldline_trace.csv``, ``node_trace.csv``) are **not**
written by default: the public release carries a rendered illustration and
aggregate summaries only, with no row-level probabilities.  Pass
``--emit-row-traces`` to write them to a local debug directory; such files must
not be committed to the public tree.

The figure and every table carry the disclaimer
``ILLUSTRATIVE PREVIEW - NOT VALIDATED; NO MULTI-STEP ACCURACY CLAIM``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import duckdb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from framework import features as F  # noqa: E402
from framework.config import load_config  # noqa: E402
from framework.utils import jdump, sha256_file  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

DISCLAIMER = (
    "ILLUSTRATIVE PREVIEW - NOT VALIDATED; NO MULTI-STEP ACCURACY CLAIM. "
    "Step 1 uses the frozen one-step model. Step 2 uses a state-lag-only "
    "approximation with static structural features frozen at the anchor. "
    "Actor states are induced by the sampled dyad subset, not the full network. "
    "The sample is selected for visibility, not random. No test labels were read."
)

OUT_DEFAULT = os.path.join(REPO, "results", "worldline_preview")
COEF_PATH = os.path.join(REPO, "results", "framework_pilot", "stability",
                         "cache", "coefs.npz")
ANCHORS = [
    {"name": "2016-06", "t_idx": 257},
    {"name": "2016-12", "t_idx": 263},
    {"name": "2017-06", "t_idx": 269},
    {"name": "2017-09", "t_idx": 272},
]
INNER_EVAL_END = 274  # 2017-11; t=275 is embargoed (label crosses 2018-01)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -60, 60)))


def sha256_if_exists(path: str) -> str | None:
    return sha256_file(path) if os.path.exists(path) else None


def host_info() -> dict:
    import sklearn
    import matplotlib as mpl
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "duckdb": duckdb.__version__,
        "sklearn": sklearn.__version__,
        "matplotlib": mpl.__version__,
        "cpu_count": os.cpu_count(),
    }


# ---------------------------------------------------------------------------
# Data / model loading
# ---------------------------------------------------------------------------
def load_frozen_model() -> dict:
    z = np.load(COEF_PATH, allow_pickle=True)
    names = [str(x) for x in z["names"]]
    return {
        "names": names,
        "coef": np.asarray(z["coefs"][0], dtype=np.float64),
        "intercept": float(z["intercepts"][0]),
        "temperature": float(z["T"]),
        "path": COEF_PATH,
        "sha256": sha256_file(COEF_PATH),
    }


def train_scaler(con: duckdb.DuckDBPyConnection, names: list[str]) -> dict:
    """Reconstruct the train-only standardiser without loading all 6.3M rows.

    The base-feature moments come from DuckDB; graph moments come from the
    frozen ``graph_meta.json`` (the same float32 moments used when the graph
    array was written); ``br``/``seen_tr`` moments are computed from the train
    panel with the 2015-01 boundary row (t=239) excluded.
    """
    base_cols = F.DYAD_CAUSAL
    base_idx = {c: i for i, c in enumerate(base_cols)}
    sel = ", ".join(
        f"avg({c}::DOUBLE) AS mu_{i}, stddev_pop({c}::DOUBLE) AS sd_{i}"
        for i, c in enumerate(base_cols)
    )
    row = con.execute(
        "SELECT " + sel + " FROM read_parquet('framework/cache/features_dyad.parquet') "
        "WHERE split='train'"
    ).fetchone()
    mu_base = np.asarray(row[0::2], dtype=np.float32)
    sd_base = np.asarray(row[1::2], dtype=np.float32)

    gmeta = json.load(open(os.path.join(REPO, "framework", "cache",
                                        "graph_meta.json"), encoding="utf-8"))
    gmu = dict(zip(gmeta["features"], np.asarray(gmeta["train_mean"], dtype=np.float32)))
    gsd = dict(zip(gmeta["features"], np.asarray(gmeta["train_std"], dtype=np.float32)))

    q = """
    WITH st AS (
      SELECT did, count(*) n, sum(active_next::INT) k
      FROM read_parquet('framework/cache/panel_dyad.parquet')
      WHERE split='train' AND t_idx <> 239 GROUP BY did
    ), g AS (
      SELECT avg(active_next::DOUBLE) gr
      FROM read_parquet('framework/cache/panel_dyad.parquet')
      WHERE split='train' AND t_idx <> 239
    ), rows AS (
      SELECT p.did, p.active_next::DOUBLE y, st.n, st.k, g.gr
      FROM read_parquet('framework/cache/panel_dyad.parquet') p
      LEFT JOIN st USING (did) CROSS JOIN g
      WHERE p.split='train'
    )
    SELECT avg(br), stddev_pop(br) FROM (
      SELECT CASE WHEN n IS NULL OR n <= 0 THEN gr
                  ELSE (k - y) / greatest(n - 1, 1) END AS br
      FROM rows
    )
    """
    br_mu, br_sd = con.execute(q).fetchone()

    mu = np.empty(len(names), dtype=np.float32)
    sd = np.empty(len(names), dtype=np.float32)
    for j, name in enumerate(names):
        if name in base_idx:
            mu[j], sd[j] = mu_base[base_idx[name]], sd_base[base_idx[name]]
        elif name in gmu:
            mu[j], sd[j] = gmu[name], gsd[name]
        elif name == "br":
            mu[j], sd[j] = br_mu, br_sd
        elif name == "seen_tr":
            mu[j], sd[j] = 1.0, 1.0
        else:
            raise KeyError(f"no train moment for feature {name!r}")
    sd = np.where(sd < 1e-8, 1.0, sd).astype(np.float32)
    return {"mu": mu, "sd": sd,
            "provenance": {
                "base": "DuckDB avg/stddev_pop on features_dyad.parquet split=train",
                "graph": "graph_meta.json train_mean/train_std",
                "br": "train panel, t_idx<>239, train-row base-rate moments",
                "seen_tr": "constant 1 on train",
            }}


def load_val_and_stats(con: duckdb.DuckDBPyConnection, cfg: dict) -> tuple[dict, dict]:
    """Load validation rows only and attach train-only br/seen_tr."""
    con.execute("CREATE OR REPLACE VIEW fb AS SELECT * FROM read_parquet("
                "'framework/cache/features_dyad.parquet')")
    va = F.export_dyad_split(con, cfg, "val", {"features": F.GRAPH_CORE20})
    if int(va["t_idx"].max()) > 275:
        raise AssertionError("validation export unexpectedly includes later months")

    rows = con.execute(
        "SELECT did, count(*) n, sum(active_next::INT) k "
        "FROM read_parquet('framework/cache/panel_dyad.parquet') "
        "WHERE split='train' AND t_idx <> 239 GROUP BY did"
    ).fetchnumpy()
    n_dyads = int(max(rows["did"].max(), va["did"].max())) + 1
    n_tr = np.zeros(n_dyads, dtype=np.float64)
    k_tr = np.zeros(n_dyads, dtype=np.float64)
    n_tr[rows["did"]] = rows["n"]
    k_tr[rows["did"]] = rows["k"]
    global_rate = float(con.execute(
        "SELECT avg(active_next::DOUBLE) FROM read_parquet("
        "'framework/cache/panel_dyad.parquet') "
        "WHERE split='train' AND t_idx <> 239"
    ).fetchone()[0])
    stats = {"n_tr": n_tr, "k_tr": k_tr, "global_rate": global_rate,
             "n_stats_rows": int(n_tr.sum()),
             "n_stats_dyads": int((n_tr > 0).sum())}
    va = F.apply_dyad_stats(va, stats, False)
    return va, stats


def load_actors(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        "SELECT actor, aid, region FROM read_parquet("
        "'framework/cache/actors.parquet') ORDER BY aid"
    ).df()


def anchor_panel(con: duckdb.DuckDBPyConnection, t_idx: int) -> pd.DataFrame:
    return con.execute(
        "SELECT did, sid, tid, s, t, active, n_events "
        "FROM read_parquet('framework/cache/panel_dyad.parquet') "
        "WHERE split='val' AND t_idx = ? ORDER BY did", [int(t_idx)]
    ).df()


def load_history(con: duckdb.DuckDBPyConnection, dids: list[int], t_idx: int,
                 n_back: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Observed active/n_events for the last ``n_back`` months, ending at t_idx."""
    did_list = ",".join(str(int(d)) for d in sorted(set(dids)))
    df = con.execute(
        "SELECT did, t_idx, active, n_events "
        "FROM read_parquet('framework/cache/panel_dyad.parquet') "
        f"WHERE split='val' AND did IN ({did_list}) "
        f"AND t_idx BETWEEN {int(t_idx) - n_back + 1} AND {int(t_idx)} "
        "ORDER BY did, t_idx"
    ).df()
    row_of = {int(d): i for i, d in enumerate(dids)}
    hist_a = np.zeros((len(dids), n_back), dtype=np.float32)
    hist_e = np.zeros((len(dids), n_back), dtype=np.float32)
    for r in df.itertuples(index=False):
        i = row_of[int(r.did)]
        j = int(r.t_idx) - (int(t_idx) - n_back + 1)
        hist_a[i, j] = float(bool(r.active))
        hist_e[i, j] = float(r.n_events)
    return hist_a, hist_e


# ---------------------------------------------------------------------------
# Sample selection and rolling update
# ---------------------------------------------------------------------------
def select_sample(df_a: pd.DataFrame, n_sources: int, n_active: int,
                  n_inactive: int) -> pd.DataFrame:
    """Select a small, boundary-focused visibility sample.

    Source actors are sampled at evenly spaced ranks of the active-source
    distribution, so the scene includes both persistent and marginal sources.
    For each source, take up to ``n_active`` currently active dyads (mostly
    those closest to p=0.5, plus one persistent high-p dyad) and ``n_inactive``
    currently inactive dyads with the highest one-step p.  The goal is to make
    transitions and uncertainty visible.  This is not a random sample and must
    not be used for prevalence or accuracy statements.
    """
    counts = (df_a[df_a["active"]]
              .groupby(["sid", "s"]).size().reset_index(name="n_active")
              .sort_values(["n_active", "sid"], ascending=[False, True])
              .reset_index(drop=True))
    if n_sources >= len(counts):
        sources = counts["sid"].tolist()
    else:
        rank = np.linspace(0, len(counts) - 1, n_sources).round().astype(int)
        sources = counts.iloc[np.unique(rank)]["sid"].tolist()
    chosen = []
    for sid in sources:
        sub = df_a[df_a["sid"] == sid].copy()
        sub["uncertainty"] = np.abs(sub["p1"] - 0.5)
        boundary = (sub[sub["active"]]
                    .sort_values(["uncertainty", "p1"]).head(max(0, n_active - 1)))
        persistent = (sub[sub["active"]]
                      .sort_values("p1", ascending=False).head(1))
        act = (pd.concat([boundary, persistent], ignore_index=True)
               .drop_duplicates("did"))
        act["sample_role"] = np.where(
            act["did"].isin(boundary["did"]), "active_boundary", "active_persistent")
        inact = (sub[~sub["active"]]
                 .sort_values("p1", ascending=False).head(n_inactive)
                 .copy())
        inact["sample_role"] = "inactive_high_p1"
        take = pd.concat([act, inact], ignore_index=True)
        need = n_active + n_inactive - len(take)
        if need > 0:
            rest = sub[~sub["did"].isin(take["did"])].sort_values(
                "p1", ascending=False).head(need).copy()
            rest["sample_role"] = "fill_high_p1"
            take = pd.concat([take, rest], ignore_index=True)
        chosen.append(take)
    out = pd.concat(chosen, ignore_index=True)
    out = out.sort_values(["sid", "active", "p1"], ascending=[True, False, False])
    return out.reset_index(drop=True)


def next_features(x_prev: np.ndarray, hist_a: np.ndarray, hist_e: np.ndarray,
                  s_new: int, n_events_new: float, t_new: int,
                  index: dict[str, int]) -> np.ndarray:
    """One state-lag-only feature update.

    Exact for current active, six active lags, three event lags, three rolling
    activity/event windows, ``since_act``, ``since_first``, ``log_n_act`` and
    the calendar.  Static structural features (graph, reverse dyad, actor ego,
    br/seen_tr) stay at their anchor values by design.
    """
    x = x_prev.copy()
    old = {k: float(x_prev[index[k]]) for k in (
        "active", "a1", "a2", "a3", "a4", "a5", "log_ev", "le1", "le2",
        "since_act", "since_first", "log_n_act")}

    def put(name: str, value: float) -> None:
        x[index[name]] = value

    put("active", float(s_new))
    put("log_ev", float(np.log1p(n_events_new if s_new else 0.0)))
    put("a1", old["active"])
    put("a2", old["a1"])
    put("a3", old["a2"])
    put("a4", old["a3"])
    put("a5", old["a4"])
    put("a6", old["a5"])
    put("le1", old["log_ev"])
    put("le2", old["le1"])
    put("le3", old["le2"])

    put("am3", float(np.mean(np.concatenate([hist_a[-2:], [s_new]]))))
    put("am6", float(np.mean(np.concatenate([hist_a[-5:], [s_new]]))))
    put("am12", float(np.mean(np.concatenate([hist_a[-11:], [s_new]]))))
    es3 = float(hist_e[-2:].sum() + (n_events_new if s_new else 0.0))
    es6 = float(hist_e[-5:].sum() + (n_events_new if s_new else 0.0))
    es12 = float(hist_e[-11:].sum() + (n_events_new if s_new else 0.0))
    put("les3", float(np.log1p(es3)))
    put("les6", float(np.log1p(es6)))
    put("les12", float(np.log1p(es12)))

    since_act = 0.0 if s_new else min(old["since_act"] + 1.0, 24.0)
    put("since_act", since_act)
    put("since_first", min(old["since_first"] + 1.0, 120.0))
    put("log_n_act", float(np.log1p(np.expm1(old["log_n_act"]) + s_new)))

    month = int(t_new) % 12  # t_idx=0 is 1995-01
    put("t_yr", float(t_new) / 12.0)
    put("mo_sin", float(np.sin(2.0 * np.pi * month / 12.0)))
    put("mo_cos", float(np.cos(2.0 * np.pi * month / 12.0)))
    return x


def predict_logit(x: np.ndarray, model: dict, scaler: dict) -> float:
    z = (x - scaler["mu"]) / scaler["sd"]
    logit = float(z @ model["coef"] + model["intercept"])
    return logit / model["temperature"]


# ---------------------------------------------------------------------------
# One anchor
# ---------------------------------------------------------------------------
def run_anchor(con: duckdb.DuckDBPyConnection, va: dict,
               model: dict, scaler: dict, anchor: dict, args: argparse.Namespace,
               global_event_mean: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    t0 = int(anchor["t_idx"])
    if t0 + 2 > INNER_EVAL_END:
        raise AssertionError(f"anchor {anchor['name']} would enter the embargo row")

    names = va["feature_names"]
    if names != model["names"]:
        raise AssertionError("feature order mismatch between val export and coefficients")
    index = {n: i for i, n in enumerate(names)}

    df_a = anchor_panel(con, t0)
    # Model p1 for every dyad at the anchor (computed from frozen coefficients).
    meta = pd.DataFrame({"did": va["did"], "t_idx": va["t_idx"],
                         "row": np.arange(len(va["did"]))})
    meta = meta[meta["t_idx"] == t0].set_index("did")
    if len(meta) != len(df_a):
        raise AssertionError(f"anchor row count mismatch at t={t0}: "
                             f"{len(meta)} vs {len(df_a)}")
    X_all = va["X"]
    p_by_did = np.full(int(df_a["did"].max()) + 1, np.nan, dtype=np.float64)
    for did, row in meta["row"].items():
        x = X_all[int(row)]
        p_by_did[int(did)] = sigmoid(predict_logit(x, model, scaler))
    df_a = df_a.copy()
    df_a["p1"] = p_by_did[df_a["did"].values]
    if not np.isfinite(df_a["p1"]).all():
        raise AssertionError("non-finite p1 at anchor")

    sample = select_sample(df_a, args.n_sources, args.n_active, args.n_inactive)
    dids = sample["did"].astype(int).tolist()
    hist_a, hist_e = load_history(con, dids, t0, n_back=12)

    # Anchor features and observed state.
    X0 = np.vstack([X_all[int(meta.loc[d, "row"])] for d in dids])
    observed = np.asarray(sample["active"].astype(int), dtype=np.int8)
    if not np.array_equal(observed, X0[:, index["active"]].astype(np.int8)):
        raise AssertionError("panel active state and feature active disagree")

    rng = np.random.default_rng(int(args.seed) + 1009 * ANCHORS.index(anchor))
    p1 = np.array([sigmoid(predict_logit(X0[i], model, scaler))
                   for i in range(len(dids))])
    s1 = (rng.random(len(dids)) < p1).astype(np.int8)

    n_sim = np.asarray(sample["n_events"].astype(float), dtype=np.float64)
    n_sim = np.where(n_sim > 0, n_sim, global_event_mean)
    p2 = np.zeros(len(dids), dtype=np.float64)
    s2 = np.zeros(len(dids), dtype=np.int8)
    hist_a1 = np.zeros_like(hist_a)
    hist_e1 = np.zeros_like(hist_e)
    for i in range(len(dids)):
        ne = float(n_sim[i]) if s1[i] else 0.0
        X1 = next_features(X0[i], hist_a[i], hist_e[i], int(s1[i]), ne,
                           t0 + 1, index)
        p2[i] = sigmoid(predict_logit(X1, model, scaler))
        s2[i] = int(rng.random() < p2[i])
        hist_a1[i] = np.concatenate([hist_a[i, 1:], [s1[i]]])
        hist_e1[i] = np.concatenate([hist_e[i, 1:], [ne]])

    states = np.column_stack([observed, s1, s2]).astype(np.int8)
    probs = np.column_stack([np.full(len(dids), np.nan), p1, p2])

    # Dyad trace.
    rows = []
    for i, r in sample.reset_index(drop=True).iterrows():
        for step, tt in enumerate((t0, t0 + 1, t0 + 2)):
            rows.append({
                "anchor": anchor["name"],
                "anchor_t_idx": t0,
                "step": step,
                "t_idx": tt,
                "date": month_label(tt),
                "did": int(r["did"]),
                "source_actor": r["s"],
                "target_actor": r["t"],
                "source_aid": int(r["sid"]),
                "target_aid": int(r["tid"]),
                "sample_role": r["sample_role"],
                "state": int(states[i, step]),
                "state_origin": "observed" if step == 0 else "sampled",
                "feature_mode": ("observed" if step == 0 else
                                 ("one_step_model" if step == 1 else
                                  "state_lag_only_approx")),
                "p_model": (None if step == 0 else round(float(probs[i, step]), 6)),
                "anchor_n_events": round(float(r["n_events"]), 4),
                "sim_n_events": (None if step == 0 else
                                 round(float(n_sim[i] if states[i, step] else 0.0), 4)),
            })
    trace = pd.DataFrame(rows)

    # Node trace (role-specific; induced by the sampled dyad subset).
    node_rows = []
    roles = (("source", "sid", "s"), ("target", "tid", "t"))
    for role, id_col, name_col in roles:
        actors = (sample[[id_col, name_col]].drop_duplicates()
                  .sort_values(id_col).reset_index(drop=True))
        for aid, aname in actors.itertuples(index=False):
            mask = sample[id_col].values == aid
            vals = [int(states[mask, step].sum() > 0) for step in range(3)]
            counts = [int(states[mask, step].sum()) for step in range(3)]
            for step, tt in enumerate((t0, t0 + 1, t0 + 2)):
                node_rows.append({
                    "anchor": anchor["name"],
                    "anchor_t_idx": t0,
                    "step": step,
                    "t_idx": tt,
                    "date": month_label(tt),
                    "role": role,
                    "actor_aid": int(aid),
                    "actor": aname,
                    "active": vals[step],
                    "n_active_sampled_dyads": counts[step],
                    "state_prev": (None if step == 0 else vals[step - 1]),
                    "transition": (None if step == 0 else
                                   f"{vals[step - 1]}->{vals[step]}"),
                })
    node_trace = pd.DataFrame(node_rows)
    if "state_prev" in node_trace.columns:
        node_trace["state_prev"] = node_trace["state_prev"].astype("Int64")

    # Transition counts per window.
    def counts(prev, curr):
        return {f"{a}{b}": int(np.sum((prev == a) & (curr == b)))
                for a in (0, 1) for b in (0, 1)}

    dyad_trans = {"step1": counts(states[:, 0], states[:, 1]),
                  "step2": counts(states[:, 1], states[:, 2])}
    node_trans = {}
    for role, _, _ in roles:
        sub = node_trace[node_trace["role"] == role]
        by_step = {}
        for step in (1, 2):
            prev = sub[sub["step"] == step - 1].set_index("actor_aid")["active"]
            curr = sub[sub["step"] == step].set_index("actor_aid")["active"]
            common = prev.index.intersection(curr.index)
            by_step[f"step{step}"] = counts(prev.loc[common].values,
                                            curr.loc[common].values)
        node_trans[role] = by_step

    info = {
        "anchor": anchor["name"],
        "t_idx": t0,
        "dates": [month_label(t0), month_label(t0 + 1), month_label(t0 + 2)],
        "n_dyads": int(len(sample)),
        "n_sources": int(sample["sid"].nunique()),
        "n_targets": int(sample["tid"].nunique()),
        "sample_roles": sample["sample_role"].value_counts().to_dict(),
        "observed_active": int(observed.sum()),
        "sampled_active_step1": int(s1.sum()),
        "sampled_active_step2": int(s2.sum()),
        "p1_summary": summary(p1),
        "p2_summary": summary(p2),
        "dyad_transitions": dyad_trans,
        "node_transitions": node_trans,
        "state_paths": {
            "".join(str(int(v)) for v in states[i]): int(sum(
                "".join(str(int(v)) for v in states[j]) ==
                "".join(str(int(v)) for v in states[i]) for j in range(len(dids))))
            for i in range(len(dids))
        },
    }
    return trace, node_trace, info


def summary(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    return {"mean": round(float(x.mean()), 6),
            "median": round(float(np.median(x)), 6),
            "min": round(float(x.min()), 6),
            "max": round(float(x.max()), 6)}


def month_label(t_idx: int) -> str:
    year = 1995 + int(t_idx) // 12
    month = int(t_idx) % 12 + 1
    return f"{year:04d}-{month:02d}"


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
STATE_CMAP = ListedColormap(["#ece7df", "#1f6f78"])
TRANS_COLORS = {"00": "#d9d4cc", "01": "#2a9d8f",
                "10": "#e76f51", "11": "#264653"}


def _ribbon(ax, states: np.ndarray, probs: np.ndarray, labels: list[str],
            title: str, roles: list[str] | None = None,
            counts: np.ndarray | None = None) -> None:
    ax.imshow(states, aspect="auto", cmap=STATE_CMAP, vmin=0, vmax=1,
              interpolation="nearest")
    for i in range(states.shape[0]):
        for j in range(states.shape[1]):
            txt = None
            if counts is not None and np.isfinite(counts[i, j]):
                txt = f"{int(counts[i, j])}"
            elif j > 0 and np.isfinite(probs[i, j]):
                txt = f"{probs[i, j]:.2f}"
            if txt is not None:
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=5.8,
                        color="white" if states[i, j] else "#3a3a3a")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["t\n(observed)", "t+1\n(sampled)", "t+2\n(sampled)"],
                       fontsize=7.5)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=6.2)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title, fontsize=9, pad=6)
    if roles is not None:
        # A light separator between source and target rows.
        n_source = sum(1 for r in roles if r == "source")
        if 0 < n_source < len(roles):
            ax.axhline(n_source - 0.5, color="#222222", lw=0.8, ls="--")
    ax.legend(handles=[Patch(facecolor=STATE_CMAP(1), label="active"),
                       Patch(facecolor=STATE_CMAP(0), label="inactive")],
              loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=2,
              fontsize=6, frameon=False)


def anonymize_export(trace: pd.DataFrame,
                     node_trace: pd.DataFrame) -> tuple[pd.DataFrame,
                                                        pd.DataFrame]:
    """Return CSV-export copies with preview-local dyad/actor labels.

    The figure keeps ISO3 labels so the communication artifact stays readable.
    The machine-readable traces must not carry a dyad id, actor id or ISO3 name
    that can be joined to the licensed ICEWS panel, so the export replaces them
    with ``dyad_XX`` / ``actor_XX`` labels that are local to this preview.
    """
    actor_ids = {
        int(v)
        for v in pd.concat(
            [trace["source_aid"], trace["target_aid"], node_trace["actor_aid"]]
        ).dropna()
    }
    dyad_ids = sorted(int(v) for v in trace["did"].dropna())
    amap = {aid: f"actor_{k:02d}" for k, aid in enumerate(sorted(actor_ids), 1)}
    dmap = {did: f"dyad_{k:02d}" for k, did in enumerate(dyad_ids, 1)}

    trace_export = pd.DataFrame({
        "anchor": trace["anchor"],
        "anchor_t_idx": trace["anchor_t_idx"],
        "step": trace["step"],
        "t_idx": trace["t_idx"],
        "date": trace["date"],
        "dyad_label": trace["did"].map(lambda v: dmap[int(v)]),
        "source_label": trace["source_aid"].map(lambda v: amap[int(v)]),
        "target_label": trace["target_aid"].map(lambda v: amap[int(v)]),
        "sample_role": trace["sample_role"],
        "state": trace["state"],
        "state_origin": trace["state_origin"],
        "feature_mode": trace["feature_mode"],
        "p_model": trace["p_model"],
        "anchor_n_events": trace["anchor_n_events"],
        "sim_n_events": trace["sim_n_events"],
    })
    node_export = pd.DataFrame({
        "anchor": node_trace["anchor"],
        "anchor_t_idx": node_trace["anchor_t_idx"],
        "step": node_trace["step"],
        "t_idx": node_trace["t_idx"],
        "date": node_trace["date"],
        "role": node_trace["role"],
        "actor_label": node_trace["actor_aid"].map(lambda v: amap[int(v)]),
        "active": node_trace["active"],
        "n_active_sampled_dyads": node_trace["n_active_sampled_dyads"],
        "state_prev": node_trace["state_prev"],
        "transition": node_trace["transition"],
    })
    return trace_export, node_export


def plot_preview(infos: list[dict], traces: dict[str, pd.DataFrame],
                 node_traces: dict[str, pd.DataFrame], agg: dict,
                 out_png: str, out_pdf: str) -> None:
    showcase = [infos[0]["anchor"], infos[2]["anchor"]] if len(infos) >= 3 else [i["anchor"] for i in infos[:2]]
    fig = plt.figure(figsize=(15.5, 11.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 0.95, 0.75],
                          hspace=0.50, wspace=0.22)
    for col, aname in enumerate(showcase[:2]):
        tr = traces[aname]
        nt = node_traces[aname]
        info = next(i for i in infos if i["anchor"] == aname)

        # Dyad ribbon.
        tr2 = tr.copy()
        tr2["p_model"] = pd.to_numeric(tr2["p_model"], errors="coerce")
        piv = tr2.pivot_table(index=["source_actor", "target_actor", "did"],
                              columns="step", values="state")
        pp = tr2.pivot_table(index=["source_actor", "target_actor", "did"],
                             columns="step", values="p_model").reindex(
                                 columns=[0, 1, 2])
        piv = piv.reindex(columns=[0, 1, 2])
        order = np.lexsort((piv[0].values, -piv[1].values, -piv[[0, 1, 2]].sum(1).values))
        st = piv.iloc[order].values.astype(np.int8)
        pr = pp.iloc[order].values.astype(float)
        labels = []
        for key, row in zip(piv.iloc[order].index, st):
            labels.append(f"{key[0]}->{key[1]}  " + "".join(str(int(v)) for v in row))
        ax = fig.add_subplot(gs[0, col])
        _ribbon(ax, st, pr, labels,
                f"{aname}: dyad states (n={info['n_dyads']}; "
                f"active t={info['observed_active']})")
        ax.set_ylabel("source -> target  (state path)", fontsize=7)

        # Node ribbon (source block then target block).  Binary state is the
        # colour; the number in each cell is the count of active sampled dyads.
        nrows = []
        for role in ("source", "target"):
            sub = nt[nt["role"] == role]
            piv_n = sub.pivot_table(index=["actor_aid", "actor"], columns="step",
                                    values="active").reindex(columns=[0, 1, 2])
            cnt_n = sub.pivot_table(index=["actor_aid", "actor"], columns="step",
                                    values="n_active_sampled_dyads").reindex(
                                        columns=[0, 1, 2])
            order_n = np.lexsort((piv_n[0].values,
                                  -piv_n[[0, 1, 2]].sum(1).values))
            for pos, ((aid, aname2), row) in enumerate(zip(
                    piv_n.iloc[order_n].index, piv_n.iloc[order_n].values)):
                nrows.append((role, f"{role[0].upper()}:{aname2}",
                              row.astype(np.int8),
                              cnt_n.iloc[order_n].values[pos].astype(float)))
        st_n = np.vstack([r[2] for r in nrows])
        pr_n = np.full_like(st_n, np.nan, dtype=float)
        ct_n = np.vstack([r[3] for r in nrows])
        ax2 = fig.add_subplot(gs[1, col])
        _ribbon(ax2, st_n, pr_n, [r[1] for r in nrows],
                f"{aname}: actor states induced by the sampled dyads "
                f"(cell = active dyad count)",
                roles=[r[0] for r in nrows], counts=ct_n)
        ax2.set_ylabel("actor (S=source, T=target)", fontsize=7)

    # Per-window transition bars, aggregated across anchors.
    ax3 = fig.add_subplot(gs[2, :])
    groups = [
        ("dyads\nt->t+1", agg["dyad"]["step1"]),
        ("dyads\nt+1->t+2", agg["dyad"]["step2"]),
        ("source nodes\nt->t+1", agg["source"]["step1"]),
        ("source nodes\nt+1->t+2", agg["source"]["step2"]),
        ("target nodes\nt->t+1", agg["target"]["step1"]),
        ("target nodes\nt+1->t+2", agg["target"]["step2"]),
    ]
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    for key in ("00", "01", "10", "11"):
        vals = np.array([g[1][key] for g in groups], dtype=float)
        ax3.bar(x, vals, bottom=bottom, color=TRANS_COLORS[key],
                edgecolor="white", linewidth=0.7, label=key)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 3:
                ax3.text(xi, b + v / 2, str(int(v)), ha="center", va="center",
                         fontsize=7, color="white" if key in ("01", "11") else "#222")
        bottom += vals
    for xi, g in enumerate(groups):
        ax3.text(xi, bottom[xi] + 0.6, f"n={int(bottom[xi])}", ha="center",
                 va="bottom", fontsize=7)
    ax3.set_xticks(x)
    ax3.set_xticklabels([g[0] for g in groups], fontsize=7.5)
    ax3.set_ylabel("count (summed over anchors)", fontsize=8)
    ax3.set_title("Per-window transitions: dyads and induced actor states",
                  fontsize=9, pad=8)
    ax3.legend(title="state transition", ncol=4, fontsize=7,
               title_fontsize=7, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, -0.24))
    ax3.spines[["top", "right"]].set_visible(False)

    fig.suptitle("World-line preview (inner validation 2016-17): "
                 "two representative anchors",
                 fontsize=13, y=0.985)
    fig.text(0.5, 0.038,
             "Figure shows 2016-06 and 2017-06; the trace covers all four "
             "anchors (2016-06, 2016-12, 2017-06, 2017-09).",
             ha="center", va="bottom", fontsize=6.8, color="#444444")
    fig.text(0.5, 0.012, DISCLAIMER, ha="center", va="bottom", fontsize=7.2,
             wrap=True,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#fdecea",
                       edgecolor="#b03a2e", linewidth=0.8))
    fig.subplots_adjust(left=0.16, right=0.985, top=0.94, bottom=0.15)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=OUT_DEFAULT)
    p.add_argument("--seed", type=int, default=20261023)
    p.add_argument("--n-sources", type=int, default=5)
    p.add_argument("--n-active", type=int, default=3)
    p.add_argument("--n-inactive", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--emit-row-traces", action="store_true",
                   help=("also write the per-row worldline_trace.csv / "
                         "node_trace.csv exports (local debugging only; not "
                         "for the public tree)"))
    p.add_argument("--only-anchor", default=None,
                   help="optional anchor label, e.g. 2016-06 (quick test)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = load_config()
    if cfg["test_policy"]["allow_test"]:
        raise RuntimeError("this preview is validation-only; allow_test must be false")

    anchors = ANCHORS
    if args.only_anchor:
        anchors = [a for a in ANCHORS if a["name"] == args.only_anchor]
        if not anchors:
            raise ValueError(f"unknown anchor {args.only_anchor!r}")

    con = duckdb.connect()
    con.execute(f"SET threads={max(1, int(args.threads))}")
    va, stats = load_val_and_stats(con, cfg)
    actors = load_actors(con)
    model = load_frozen_model()
    scaler = train_scaler(con, va["feature_names"])

    global_event_mean = float(con.execute(
        "SELECT avg(n_events) FROM read_parquet("
        "'framework/cache/panel_dyad.parquet') "
        "WHERE split='train' AND active"
    ).fetchone()[0])
    print(f"data: val rows={len(va['y']):,}; train stats rows={stats['n_stats_rows']:,}; "
          f"global positive event mean={global_event_mean:.3f}")

    all_traces, all_nodes, infos = [], [], []
    for anchor in anchors:
        tr, nt, info = run_anchor(con, va, model, scaler, anchor, args,
                                  global_event_mean)
        all_traces.append(tr)
        all_nodes.append(nt)
        infos.append(info)
        print(f"anchor {anchor['name']}: dyads={info['n_dyads']} "
              f"active(t)={info['observed_active']} "
              f"active(t+1)={info['sampled_active_step1']} "
              f"active(t+2)={info['sampled_active_step2']}")
    con.close()

    trace = pd.concat(all_traces, ignore_index=True)
    node_trace = pd.concat(all_nodes, ignore_index=True)

    # Aggregate transition counts across anchors.
    def zero():
        return {k: 0 for k in ("00", "01", "10", "11")}
    agg = {"dyad": {"step1": zero(), "step2": zero()},
           "source": {"step1": zero(), "step2": zero()},
           "target": {"step1": zero(), "step2": zero()}}
    for info in infos:
        for key in ("step1", "step2"):
            for k, v in info["dyad_transitions"][key].items():
                agg["dyad"][key][k] += v
            for role in ("source", "target"):
                for k, v in info["node_transitions"][role][key].items():
                    agg[role][key][k] += v

    traces = {i["anchor"]: tr for i, tr in zip(infos, all_traces)}
    nodes = {i["anchor"]: nt for i, nt in zip(infos, all_nodes)}

    csv_trace = os.path.join(args.out_dir, "worldline_trace.csv")
    csv_nodes = os.path.join(args.out_dir, "node_trace.csv")
    png = os.path.join(args.out_dir, "worldline_preview.png")
    pdf = os.path.join(args.out_dir, "worldline_preview.pdf")
    json_path = os.path.join(args.out_dir, "worldline_trace.json")
    manifest_path = os.path.join(args.out_dir, "run_manifest.json")

    trace_export, node_export = anonymize_export(trace, node_trace)
    if args.emit_row_traces:
        trace_export.to_csv(csv_trace, index=False, float_format="%.6f")
        node_export.to_csv(csv_nodes, index=False)
    plot_preview(infos, traces, nodes, agg, png, pdf)

    payload = {
        "schema": "q2-worldline-preview/1",
        "status": "illustrative_preview_not_validated",
        "disclaimer": DISCLAIMER,
        "no_multi_step_accuracy_claim": True,
        "no_node_level_forecast_claim": True,
        "no_test_labels_read": True,
        "no_test_tuning": True,
        "no_synthetic_or_t1": True,
        "labels_used_for_scoring": False,
        "accuracy_metrics": None,
        "figure_note": ("figure shows 2016-06 and 2017-06; the trace covers "
                        "all four anchors"),
        "period": {
            "name": "inner_validation_2016_17",
            "anchors": [i["anchor"] for i in infos],
            "t_idx": [i["t_idx"] for i in infos],
            "note": "2016-01..2017-11 only; t=275 (2017-12) is embargoed",
        },
        "model": {
            "name": "frozen Sprint B logistic49",
            "source": os.path.relpath(COEF_PATH, REPO),
            "sha256": model["sha256"],
            "n_features": len(model["names"]),
            "temperature": model["temperature"],
            "feature_order": model["names"],
            "scaler": scaler["provenance"],
        },
        "sample_selection": {
            "rule": ("n_sources source actors at evenly spaced ranks of the "
                     "active-source distribution; per source, up to n_active "
                     "currently active dyads (mostly closest to p=0.5, plus one "
                     "persistent high-p dyad) and n_inactive currently inactive "
                     "high-p dyads"),
            "n_sources": int(args.n_sources),
            "n_active": int(args.n_active),
            "n_inactive": int(args.n_inactive),
            "random": False,
            "use": "visibility only; no prevalence or accuracy inference",
        },
        "step_semantics": {
            "step0": "observed anchor state",
            "step1": "sampled from the frozen one-step model p(t+1)",
            "step2": ("sampled from a state-lag-only approximation p(t+2); "
                      "active/lag/rolling/history/calendar updated from the "
                      "sampled state; graph, reverse-dyad, actor-ego, br/seen_tr "
                      "frozen at anchor; event magnitude fixed at anchor value"),
        },
        "anchors": infos,
        "aggregate_transitions": agg,
        "files": {
            "figure_png": os.path.basename(png),
            "figure_pdf": os.path.basename(pdf),
            "aggregate_json": os.path.basename(json_path),
            **({"trace_csv": os.path.basename(csv_trace),
                "node_trace_csv": os.path.basename(csv_nodes)}
               if args.emit_row_traces else {}),
        },
        "row_counts": {
            "dyad_step_rows": int(len(trace)),
            "node_rows": int(len(node_trace)),
            "shipped_as_files": bool(args.emit_row_traces),
            "note": ("per-row traces are withheld unless --emit-row-traces is "
                     "passed; the public tree ships only the rendered figure "
                     "and the aggregate summaries in this JSON"),
        },
        "no_row_level_probabilities": not args.emit_row_traces,
        "machine_readable_traces": {
            "anonymized": True,
            "shipped": bool(args.emit_row_traces),
            "labels": (
                "when generated locally, preview-local dyad_XX / actor_XX labels "
                "replace ICEWS dyad ids, actor ids and ISO3 names; no such "
                "export is distributed"
            ),
            "scope": (
                "240 dyad-step rows over four inner-validation anchors; "
                "20 sampled dyads per anchor (71 distinct across anchors); "
                "2016-01..2017-11 only; no 2018-22 holdout row"
            ),
            "figure_note": (
                "the print figure labels the 20 country dyads for readability; "
                "those labels are not join keys and no probability is exported"
            ),
            "release_rule": (
                "no row-level predictions or per-row probabilities are shipped; "
                "only the rendered PNG/PDF illustration and the per-anchor/"
                "aggregate summary blocks in this JSON are distributed"
            ),
        },
        "runtime_seconds": round(time.time() - t_start, 2),
        "created_utc": utc_now(),
    }
    jdump(json_path, payload)

    manifest = {
        "script": os.path.relpath(os.path.abspath(__file__), REPO),
        "argv": sys.argv,
        "created_utc": utc_now(),
        "host": host_info(),
        "inputs": {
            "features_dyad.parquet": sha256_if_exists(
                os.path.join(REPO, "framework", "cache", "features_dyad.parquet")),
            "panel_dyad.parquet": sha256_if_exists(
                os.path.join(REPO, "framework", "cache", "panel_dyad.parquet")),
            "graph_val.npy": sha256_if_exists(
                os.path.join(REPO, "framework", "cache", "graph_val.npy")),
            "graph_meta.json": sha256_if_exists(
                os.path.join(REPO, "framework", "cache", "graph_meta.json")),
            "actors.parquet": sha256_if_exists(
                os.path.join(REPO, "framework", "cache", "actors.parquet")),
            "coefs.npz": sha256_if_exists(COEF_PATH),
        },
        "outputs": {
            "worldline_preview.png": sha256_if_exists(png),
            "worldline_preview.pdf": sha256_if_exists(pdf),
            **({"worldline_trace.csv": sha256_if_exists(csv_trace),
                "node_trace.csv": sha256_if_exists(csv_nodes)}
               if args.emit_row_traces else {}),
            "worldline_trace.json": sha256_if_exists(json_path),
        },
        "runtime_seconds": round(time.time() - t_start, 2),
        "preview_scope": {
            "period": "inner_validation_2016_17 (2016-01..2017-11; max t_idx 274)",
            "sampled_dyads_per_anchor": int(args.n_sources) * (
                int(args.n_active) + int(args.n_inactive)),
            "distinct_dyads_across_anchors": int(trace["did"].nunique()),
            "holdout_rows": 0,
            "row_level_files_shipped": 2 if args.emit_row_traces else 0,
            "machine_readable_traces_anonymized": True,
            "release_rule": (
                "no row-level predictions: the per-row CSVs are withheld "
                "unless --emit-row-traces is passed; only the rendered "
                "illustration and the aggregate JSON summaries ship; no "
                "headline number depends on this directory"
            ),
        },
        "withheld_exports": {
            "worldline_trace.csv": {
                "rows": int(len(trace)), "reason": "row-level state/p_model"},
            "node_trace.csv": {
                "rows": int(len(node_trace)),
                "reason": "row-level induced actor state"},
        },
    }
    jdump(manifest_path, manifest)

    print(f"wrote {png}")
    print(f"wrote {pdf}")
    if args.emit_row_traces:
        print(f"wrote {csv_trace} ({len(trace)} rows)")
        print(f"wrote {csv_nodes} ({len(node_trace)} rows)")
    else:
        print("row-level CSVs withheld (pass --emit-row-traces for a local debug copy)")
    print(f"wrote {json_path}")
    print(f"wrote {manifest_path}")
    print(f"runtime {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
