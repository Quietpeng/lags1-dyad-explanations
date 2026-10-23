"""Interpretability artefacts: node cards, attribution, deletion/insertion,
counterfactuals, edge probes, failure cases and human-audit materials.

Primary T1 model: L2 logistic on ``ACTOR_FEATURES`` for ``high_q75`` (the
redesigned target).  Primary T2 model: L2 logistic on the base-29 + core-20
graph features (same feature set as the P1 headline), refitted here so that
per-row additive contributions can be computed exactly.

All artefacts are additive log-odds for the logistic model:

    logit_i = intercept + sum_j coef_j * (x_ij - mu_j) / sd_j

so a card term ``coef_j * z_ij`` is the exact change in log-odds relative to the
training mean, not a SHAP approximation.  HGB is used only as a capacity
control in the main results; it is not interpreted.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd

from . import features as F
from . import models, targets
from .config import REPO_ROOT
from .metrics import binary_metrics, reliability_slope
from .utils import elapsed, jdump, log

TEST_START = 276
EVAL_START, EVAL_END = 252, TEST_START


def _p(path: str) -> str:
    return str(path).replace("\\", "/")


def _out(cfg: dict, *parts: str) -> str:
    p = os.path.join(cfg["paths"]["out_dir"], *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _slice(d: dict, mask: np.ndarray) -> dict:
    return {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# T1 high_q75
# ---------------------------------------------------------------------------
def fit_t1(cfg: dict) -> dict:
    df = targets.load_actor_frame(cfg)
    masks = targets.split_masks(df)
    thr = targets.threshold_for(df, "high_q75")
    y = targets.make_label(df, "high_q75", thr)
    X = df[F.ACTOR_FEATURES].to_numpy(np.float32)
    fit = models.fit_logistic(X[masks["train"]], y[masks["train"]],
                              X[masks["select"]], y[masks["select"]], cfg,
                              feature_names=F.ACTOR_FEATURES)
    ev = masks["eval"]
    p = models.predict_logistic(fit, X[ev])
    C = models.linear_contributions(fit, X[ev], F.ACTOR_FEATURES)
    return {"df": df, "masks": masks, "y": y, "X": X, "fit": fit,
            "p": p, "C": C, "ev": ev, "thr": thr}


def t1_node_cards(t1: dict, cfg: dict, top_n: int = 40) -> tuple[pd.DataFrame, list]:
    df, ev, y, p, C = t1["df"], t1["ev"], t1["y"], t1["p"], t1["C"]
    sub = df[ev].reset_index(drop=True)
    names = F.ACTOR_FEATURES
    rows, detailed = [], []
    for aid, g in sub.groupby("aid", sort=False):
        idx = g.index.to_numpy()
        yy, pp, cc = y[ev][idx], p[idx], C[idx]
        m = binary_metrics(yy, pp)
        slope = reliability_slope(yy, pp) if yy.min() != yy.max() else {"slope": None}
        mean_c = cc.mean(axis=0)
        order = np.argsort(mean_c)
        groups = {
            gname: float(mean_c[[names.index(f) for f in feats if f in names]].sum())
            for gname, feats in F.ACTOR_GROUPS.items()
        }
        # Per-actor local model (only if enough positives/negatives).
        d = {"aid": df["aid"].to_numpy(), "X": t1["X"], "y": t1["y"]}
        try:
            lm = models.fit_per_actor_logistic(d, int(aid), cfg)
        except Exception:
            lm = None
        top_pos = names[int(order[-1])]
        top_neg = names[int(order[0])]
        card = (f"{g['actor'].iloc[0]} ({g['region'].iloc[0]}): eval n={len(g)}, "
                f"pos={int(yy.sum())}, mean p={pp.mean():.3f}, LL={m['log_loss']:.3f}. "
                f"Top +: {top_pos} ({mean_c[int(order[-1])]:+.2f}); "
                f"Top -: {top_neg} ({mean_c[int(order[0])]:+.2f}).")
        rows.append({
            "aid": int(aid), "actor": g["actor"].iloc[0], "region": g["region"].iloc[0],
            "n_eval": int(len(g)), "positives": int(yy.sum()),
            "pos_rate": round(float(yy.mean()), 4),
            "mean_p": round(float(pp.mean()), 4), "mean_y": round(float(yy.mean()), 4),
            "log_loss": round(float(m["log_loss"]), 5), "auc": m["auc"],
            "cal_slope": slope["slope"],
            "top_pos_feature": top_pos, "top_pos_contrib": round(float(mean_c[int(order[-1])]), 4),
            "top_neg_feature": top_neg, "top_neg_contrib": round(float(mean_c[int(order[0])]), 4),
            "local_n": None if lm is None else lm["n"],
            "local_pos": None if lm is None else lm["positives"],
            "local_train_ll": None if lm is None else lm["train_log_loss"],
            "local_train_auc": None if lm is None else lm["train_auc"],
            "local_top_feature": None if lm is None else names[int(np.argmax(np.abs(lm["coef"])))],
            "local_top_coef": None if lm is None else round(float(np.max(np.abs(lm["coef"]))), 4),
            "card": card,
        })
        if len(detailed) < top_n:
            month_cards = []
            for j in np.argsort(-pp)[:3]:
                terms = [{"feature": names[k], "value": round(float(t1["X"][ev][idx[j], k]), 4),
                          "contribution": round(float(cc[j, k]), 4)}
                         for k in np.argsort(-np.abs(cc[j]))[:6]]
                month_cards.append({
                    "t_idx": int(g["t_idx"].iloc[j]), "y": int(yy[j]),
                    "p": round(float(pp[j]), 4),
                    "terms": terms,
                })
            detailed.append({
                "aid": int(aid), "actor": g["actor"].iloc[0],
                "region": g["region"].iloc[0], "card": card,
                "mean_contributions": [
                    {"feature": names[k], "mean_contribution": round(float(mean_c[k]), 4)}
                    for k in np.argsort(-np.abs(mean_c))[:10]
                ],
                "group_contributions": {k: round(v, 4) for k, v in groups.items()},
                "top_months": month_cards,
                "local_model": None if lm is None else {
                    "n": lm["n"], "positives": lm["positives"],
                    "train_log_loss": lm["train_log_loss"],
                    "train_auc": lm["train_auc"],
                    "top_coef": [{"feature": names[k], "coef": round(float(lm["coef"][k]), 4)}
                                 for k in np.argsort(-np.abs(lm["coef"]))[:5]],
                },
            })
    return pd.DataFrame(rows).sort_values("log_loss", ascending=False), detailed


def t1_attribution(t1: dict, cfg: dict) -> tuple[pd.DataFrame, dict]:
    df, ev, y, p, C = t1["df"], t1["ev"], t1["y"], t1["p"], t1["C"]
    names = F.ACTOR_FEATURES
    coef = t1["fit"]["coef"]
    mean_abs = np.abs(C).mean(axis=0)
    perm = models.permutation_importance(t1["fit"], t1["X"][ev], y[ev], names,
                                         n_repeats=3, seed=int(cfg["seed"]),
                                         max_rows=200000)
    perm_map = {r["feature"]: r["delta_log_loss"] for r in perm}
    gmap = {}
    for gname, feats in F.ACTOR_GROUPS.items():
        for f in feats:
            gmap[f] = gname
    tab = pd.DataFrame([
        {"feature": n, "coef": round(float(c), 5),
         "mean_abs_contrib": round(float(mean_abs[j]), 5),
         "perm_delta_log_loss": perm_map.get(n),
         "group": gmap.get(n, "other")}
        for j, (n, c) in enumerate(zip(names, coef))
    ]).sort_values("mean_abs_contrib", ascending=False)
    group = (tab.groupby("group")["mean_abs_contrib"].sum()
             .sort_values(ascending=False).round(4).to_dict())
    return tab, {"group_mean_abs_contrib": group,
                 "top_perm": perm[:15]}


def t1_deletion_insertion(t1: dict, cfg: dict, n_sample: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(int(cfg["seed"]))
    ev = t1["ev"]
    X, y, p = t1["X"][ev], t1["y"][ev], t1["p"]
    idx = np.sort(rng.choice(len(y), size=min(n_sample, len(y)), replace=False))
    X, y = X[idx], y[idx]
    C = models.linear_contributions(t1["fit"], X, F.ACTOR_FEATURES)
    mu = t1["fit"]["mu"]
    base_ll = binary_metrics(y, p[idx])["log_loss"]
    p_mean = models.predict_logistic(t1["fit"], np.tile(mu, (len(X), 1)).astype(np.float32))
    mean_ll = binary_metrics(y, p_mean)["log_loss"]
    rows = []
    for k in (1, 3, 5, 10):
        top = np.argsort(-np.abs(C), axis=1)[:, :k]
        Xd = X.copy()
        for i in range(len(X)):
            Xd[i, top[i]] = mu[top[i]]
        pd_ = models.predict_logistic(t1["fit"], Xd)
        # insertion: start at train mean, insert the top-k actual values
        X0 = np.tile(mu, (len(X), 1)).astype(np.float32)
        p0 = models.predict_logistic(t1["fit"], X0)
        Xi = X0.copy()
        for i in range(len(X)):
            Xi[i, top[i]] = X[i, top[i]]
        pi = models.predict_logistic(t1["fit"], Xi)
        rows.append({
            "target": "T1_high_q75", "k": k,
            "deletion_mean_dp": round(float(np.mean(pd_ - p[idx])), 5),
            "deletion_ll": binary_metrics(y, pd_)["log_loss"],
            "deletion_dll": round(binary_metrics(y, pd_)["log_loss"] - base_ll, 5),
            "insertion_mean_dp": round(float(np.mean(pi - p0)), 5),
            "insertion_ll": binary_metrics(y, pi)["log_loss"],
            "mean_baseline_ll": mean_ll,
            "insertion_gain_vs_mean": round(mean_ll - binary_metrics(y, pi)["log_loss"], 5),
            "insertion_dll_vs_actual": round(binary_metrics(y, pi)["log_loss"] - base_ll, 5),
        })
    return pd.DataFrame(rows)


def t1_counterfactual(t1: dict, cfg: dict) -> pd.DataFrame:
    ev = t1["ev"]
    X, p = t1["X"][ev], t1["p"]
    names = F.ACTOR_FEATURES
    idx = {n: i for i, n in enumerate(names)}
    df = t1["df"][ev].reset_index(drop=True)
    mu = t1["fit"]["mu"]
    rows = []
    active = df["out_active"].to_numpy() == 1
    for scenario, edits in {
        "no_events_last_month": [("a1_out", 0.0), ("le1_out", 0.0),
                                 ("oam3", float(mu[idx["oam3"]])),
                                 ("oes3", float(mu[idx["oes3"]]))],
        "degree_collapse": [("out_partners", float(mu[idx["out_partners"]])),
                            ("out_partners_chg", 0.0)],
        "cold_start_all_history": [(n, float(mu[idx[n]])) for n in
                                   ("a1_out", "a2_out", "a3_out", "a4_out",
                                    "a5_out", "a6_out", "oam3", "oam6", "oam12",
                                    "oes3", "oes6", "oes12", "since_out")],
    }.items():
        Xc = X.copy()
        for n, v in edits:
            Xc[:, idx[n]] = v
        pc = models.predict_logistic(t1["fit"], Xc)
        d = pc - p
        rows.append({
            "target": "T1_high_q75", "scenario": scenario,
            "n": int(len(X)),
            "mean_dp": round(float(d.mean()), 5),
            "median_dp": round(float(np.median(d)), 5),
            "share_down": round(float((d < 0).mean()), 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# T2 full model
# ---------------------------------------------------------------------------
def fit_t2(cfg: dict) -> dict:
    con = duckdb.connect()
    con.execute(f"CREATE OR REPLACE VIEW fb AS SELECT * FROM read_parquet('"
                f"{_p(os.path.join(cfg['paths']['cache_dir'], 'features_dyad.parquet'))}')")
    gm = {"features": F.GRAPH_CORE20}
    tr = F.export_dyad_split(con, cfg, "train", gm)
    va = F.export_dyad_split(con, cfg, "val", gm)
    con.close()
    stats = F.fit_dyad_stats(tr, int(max(tr["did"].max(), va["did"].max())) + 1)
    tr = F.apply_dyad_stats(tr, stats, True)
    va = F.apply_dyad_stats(va, stats, False)
    t = va["t_idx"]
    sel = _slice(va, (t >= 240) & (t < 252))
    ev = _slice(va, (t >= EVAL_START) & (t < EVAL_END - 1))
    fit = models.fit_logistic(tr["X"], tr["y"], sel["X"], sel["y"], cfg,
                              C_grid=[0.3, 1.0], feature_names=tr["feature_names"])
    p = models.predict_logistic(fit, ev["X"])
    C = models.linear_contributions(fit, ev["X"], tr["feature_names"])
    return {"train": tr, "select": sel, "eval": ev, "fit": fit, "p": p, "C": C,
            "names": tr["feature_names"]}


def t2_attribution(t2: dict, cfg: dict) -> tuple[pd.DataFrame, dict]:
    names, coef, C = t2["names"], t2["fit"]["coef"], t2["C"]
    mean_abs = np.abs(C).mean(axis=0)
    perm = models.permutation_importance(t2["fit"], t2["eval"]["X"],
                                         t2["eval"]["y"], names, n_repeats=2,
                                         seed=int(cfg["seed"]), max_rows=120000)
    pm = {r["feature"]: r["delta_log_loss"] for r in perm}
    groups = {}
    for gname, feats in F.DYAD_GROUPS.items():
        for f in feats:
            groups[f] = gname
    for f in F.GRAPH_CORE20:
        groups[f] = "graph"
    tab = pd.DataFrame([
        {"feature": n, "coef": round(float(c), 5),
         "mean_abs_contrib": round(float(mean_abs[j]), 5),
         "perm_delta_log_loss": pm.get(n), "group": groups.get(n, "other")}
        for j, (n, c) in enumerate(zip(names, coef))
    ]).sort_values("mean_abs_contrib", ascending=False)
    return tab, {"group_mean_abs_contrib": tab.groupby("group")["mean_abs_contrib"]
                 .sum().sort_values(ascending=False).round(4).to_dict(),
                 "top_perm": perm[:15]}


def _adj_snapshots(cfg: dict, months: set[int]):
    """Cumulative and 12-month weighted adjacency snapshots at eval months."""
    con = duckdb.connect()
    try:
        e = con.execute(f"""
          SELECT sid, tid, t_idx, n_events FROM read_parquet('{_p(os.path.join(cfg['paths']['cache_dir'], 'panel_dyad.parquet'))}')
          WHERE active AND t_idx < {TEST_START} ORDER BY t_idx
        """).df()
    finally:
        con.close()
    n_a = int(max(e["sid"].max(), e["tid"].max())) + 1
    adj = [set() for _ in range(n_a)]
    wadj = [dict() for _ in range(n_a)]
    adj12 = [set() for _ in range(n_a)]
    wadj12 = [dict() for _ in range(n_a)]
    # deque of (t, s, d, w) to expire the 12-month window
    recent = []
    by_t = {int(k): g for k, g in e.groupby("t_idx")}
    out = {}
    for t in range(0, max(months) + 1):
        g = by_t.get(t)
        if g is not None:
            for s, d, w in zip(g["sid"].to_numpy(), g["tid"].to_numpy(),
                               g["n_events"].to_numpy()):
                s, d, w = int(s), int(d), float(w)
                adj[s].add(d); adj[d].add(s)
                wadj[s][d] = wadj[s].get(d, 0.0) + w
                wadj[d][s] = wadj[d].get(s, 0.0) + w
                adj12[s].add(d); adj12[d].add(s)
                wadj12[s][d] = wadj12[s].get(d, 0.0) + w
                wadj12[d][s] = wadj12[d].get(s, 0.0) + w
                recent.append((t, s, d, w))
        # expire months <= t-12 (keep t-11..t inclusive)
        while recent and recent[0][0] <= t - 12:
            _, s, d, w = recent.pop(0)
            for a, b in ((s, d), (d, s)):
                wadj12[a][b] -= w
                if wadj12[a][b] <= 1e-9:
                    wadj12[a].pop(b, None)
                    adj12[a].discard(b)
        if t in months:
            out[t] = {
                "adj": [set(a) for a in adj],
                "wadj": [dict(a) for a in wadj],
                "adj12": [set(a) for a in adj12],
                "wadj12": [dict(a) for a in wadj12],
            }
    return out


def _gfeat(snap, s, t, drop=None):
    Ns, Nt = set(snap["adj"][s]), set(snap["adj"][t])
    Ns12, Nt12 = set(snap["adj12"][s]), set(snap["adj12"][t])
    Ws, Wt = dict(snap["wadj"][s]), dict(snap["wadj"][t])
    Ws12, Wt12 = dict(snap["wadj12"][s]), dict(snap["wadj12"][t])
    if drop is not None:
        for N in (Ns, Nt, Ns12, Nt12):
            N.discard(drop)
        for W in (Ws, Wt, Ws12, Wt12):
            W.pop(drop, None)
    inter, inter12 = Ns & Nt, Ns12 & Nt12
    ds, dt, cn = len(Ns), len(Nt), len(inter)
    ds12, dt12, cn12 = len(Ns12), len(Nt12), len(inter12)
    deg_all = [len(x) for x in snap["adj"]]
    deg12_all = [len(x) for x in snap["adj12"]]
    aa = sum(1.0 / np.log(max(deg_all[j], 2)) for j in inter) if inter else 0.0
    ra = sum(1.0 / max(deg_all[j], 1) for j in inter) if inter else 0.0
    aa12 = sum(1.0 / np.log(max(deg12_all[j], 2)) for j in inter12) if inter12 else 0.0
    ra12 = sum(1.0 / max(deg12_all[j], 1) for j in inter12) if inter12 else 0.0
    # NRA[s,t] = sum_j B[s,j] * Cr[j,t] (edges j-t in the last 12 months)
    nra_s = sum(snap["wadj12"][j].get(t, 0.0) > 0 for j in Ns)
    nra_t = sum(snap["wadj12"][j].get(s, 0.0) > 0 for j in Nt)
    return {
        "g_deg_s": ds, "g_deg_t": dt,
        "g_wdeg_s": sum(Ws.values()), "g_wdeg_t": sum(Wt.values()),
        "g_deg12_s": ds12, "g_deg12_t": dt12,
        "g_wdeg12_s": sum(Ws12.values()), "g_wdeg12_t": sum(Wt12.values()),
        "g_cn": cn, "g_jaccard": cn / max(ds + dt - cn, 1),
        "g_aa": aa, "g_ra": ra,
        "g_tri_ratio": cn / max(min(ds, dt), 1),
        "g_pa": ds * dt,
        "g_two_hop": 1.0 if (t not in Ns and cn > 0) else 0.0,
        "g_cn12": cn12, "g_jaccard12": cn12 / max(ds12 + dt12 - cn12, 1),
        "g_aa12": aa12, "g_ra12": ra12,
        "g_nra_s": nra_s, "g_nra_t": nra_t,
        "g_nra_norm_s": nra_s / max(ds, 1), "g_nra_norm_t": nra_t / max(dt, 1),
    }


def t2_edge_probe(t2: dict, cfg: dict, n_rows: int = 60) -> pd.DataFrame:
    ev = t2["eval"]
    y, p, did, tid, sid, t_idx = (ev["y"], t2["p"], ev["did"], ev["tid"],
                                  ev["sid"], ev["t_idx"])
    months = set(int(x) for x in np.unique(t_idx))
    snaps = _adj_snapshots(cfg, months)
    names = t2["names"]
    idx = {n: i for i, n in enumerate(names)}
    log_names = set(F.GRAPH_LOG)
    rng = np.random.default_rng(int(cfg["seed"]))
    # One row per dyad, sampled across the predicted-probability range where
    # graph structure can still move the forecast (not the saturated tail).
    cand = []
    for d_ in np.unique(did):
        ii = np.where(did == d_)[0]
        ii = ii[(p[ii] >= 0.15) & (p[ii] <= 0.97)]
        if len(ii):
            cand.append(int(ii[np.argmax(p[ii])]))
    rng.shuffle(cand)
    rows = []
    for i in cand:
        snap = snaps.get(int(t_idx[i]))
        if snap is None:
            continue
        s, t = int(sid[i]), int(tid[i])
        inter = set(snap["adj"][s]) & set(snap["adj"][t])
        if not inter:
            continue
        j = max(inter, key=lambda x: len(snap["adj"][x]))
        cf = _gfeat(snap, s, t, drop=j)
        x = ev["X"][i].copy()
        for k, v in cf.items():
            if k not in idx:
                continue
            x[idx[k]] = np.log1p(max(v, 0.0)) if k in log_names else v
        p_cf = float(models.predict_logistic(t2["fit"], x[None, :])[0])
        noncommon = [a for a in snap["adj"][s] if a not in inter and a != t]
        jw = float(snap["wadj"][s].get(j, 0.0))
        if noncommon:
            jr = min(noncommon, key=lambda a: (
                abs(len(snap["adj"][a]) - len(snap["adj"][j])),
                abs(float(snap["wadj"][s].get(a, 0.0)) - jw)))
        else:
            jr = None
        if jr is not None:
            cf2 = _gfeat(snap, s, t, drop=jr)
            x2 = ev["X"][i].copy()
            for k, v in cf2.items():
                if k not in idx:
                    continue
                x2[idx[k]] = np.log1p(max(v, 0.0)) if k in log_names else v
            p_cf2 = float(models.predict_logistic(t2["fit"], x2[None, :])[0])
        else:
            p_cf2 = np.nan
        rows.append({
            "did": int(did[i]), "sid": s, "tid": t, "t_idx": int(t_idx[i]),
            "y": int(y[i]), "p": round(float(p[i]), 5),
            "neighbor": j, "neighbor_degree": len(snap["adj"][j]),
            "edge_weight": round(jw, 1),
            "control_degree": None if jr is None else len(snap["adj"][jr]),
            "control_weight": None if jr is None else round(float(snap["wadj"][s].get(jr, 0.0)), 1),
            "cn": len(inter),
            "dp_remove_common": round(p_cf - float(p[i]), 5),
            "dp_remove_noncommon": round(p_cf2 - float(p[i]), 5) if jr is not None else None,
            "p_cf_common": round(p_cf, 5),
        })
        if len(rows) >= n_rows:
            break
    return pd.DataFrame(rows)


def t2_counterfactual(t2: dict, cfg: dict) -> pd.DataFrame:
    ev = t2["eval"]
    X, p = ev["X"], t2["p"]
    names = t2["names"]
    idx = {n: i for i, n in enumerate(names)}
    mu = t2["fit"]["mu"]
    rows = []
    scenarios = {
        "no_recent_edges": ["a1", "a2", "a3"],
        "no_history": ["a1", "a2", "a3", "a4", "a5", "a6", "le1", "le2", "le3",
                       "am3", "am6", "am12", "since_act", "log_n_act"],
        "low_graph": ["g_cn", "g_aa", "g_ra", "g_jaccard", "g_two_hop"],
    }
    for name, cols in scenarios.items():
        Xc = X.copy()
        for c in cols:
            if c in idx:
                Xc[:, idx[c]] = mu[idx[c]]
        pc = models.predict_logistic(t2["fit"], Xc)
        d = pc - p
        rows.append({
            "target": "T2_dyad_active_next", "scenario": name,
            "n": int(len(X)), "mean_dp": round(float(d.mean()), 5),
            "median_dp": round(float(np.median(d)), 5),
            "share_down": round(float((d < 0).mean()), 4),
        })
    return pd.DataFrame(rows)


def t2_deletion_insertion(t2: dict, cfg: dict, n_sample: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(int(cfg["seed"]))
    ev = t2["eval"]
    idx = np.sort(rng.choice(len(ev["y"]), size=min(n_sample, len(ev["y"])), replace=False))
    X, y, p = ev["X"][idx], ev["y"][idx], t2["p"][idx]
    names = t2["names"]
    C = models.linear_contributions(t2["fit"], X, names)
    mu = t2["fit"]["mu"]
    base_ll = binary_metrics(y, p)["log_loss"]
    p_mean = models.predict_logistic(t2["fit"], np.tile(mu, (len(X), 1)).astype(np.float32))
    mean_ll = binary_metrics(y, p_mean)["log_loss"]
    rows = []
    for k in (1, 3, 5, 10):
        top = np.argsort(-np.abs(C), axis=1)[:, :k]
        Xd = X.copy()
        for i in range(len(X)):
            Xd[i, top[i]] = mu[top[i]]
        pd_ = models.predict_logistic(t2["fit"], Xd)
        X0 = np.tile(mu, (len(X), 1)).astype(np.float32)
        p0 = models.predict_logistic(t2["fit"], X0)
        Xi = X0.copy()
        for i in range(len(X)):
            Xi[i, top[i]] = X[i, top[i]]
        pi = models.predict_logistic(t2["fit"], Xi)
        rows.append({
            "target": "T2", "k": k,
            "deletion_mean_dp": round(float(np.mean(pd_ - p)), 5),
            "deletion_dll": round(binary_metrics(y, pd_)["log_loss"] - base_ll, 5),
            "insertion_mean_dp": round(float(np.mean(pi - p0)), 5),
            "insertion_ll": binary_metrics(y, pi)["log_loss"],
            "mean_baseline_ll": mean_ll,
            "insertion_gain_vs_mean": round(mean_ll - binary_metrics(y, pi)["log_loss"], 5),
            "insertion_dll_vs_actual": round(binary_metrics(y, pi)["log_loss"] - base_ll, 5),
        })
    return pd.DataFrame(rows)


def failure_cases_t1(t1: dict, k: int = 25) -> pd.DataFrame:
    ev = t1["ev"]
    df = t1["df"][ev].reset_index(drop=True)
    y, p = t1["y"][ev], t1["p"]
    nll = -(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))
    idx = np.argsort(-nll)[:k]
    out = df.iloc[idx][["actor", "region", "t_idx"]].copy()
    out["y"] = y[idx]
    out["p"] = np.round(p[idx], 4)
    out["nll"] = np.round(nll[idx], 4)
    out["out_events"] = np.round(df["out_events"].to_numpy()[idx], 1)
    out["out_events_next"] = np.round(df["out_events_next"].to_numpy()[idx], 1)
    return out


def failure_cases_t2(t2: dict, k: int = 25) -> pd.DataFrame:
    ev = t2["eval"]
    y, p = ev["y"], t2["p"]
    nll = -(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))
    idx = np.argsort(-nll)[:k]
    out = pd.DataFrame({
        "did": ev["did"][idx], "sid": ev["sid"][idx], "tid": ev["tid"][idx],
        "t_idx": ev["t_idx"][idx], "y": y[idx], "p": np.round(p[idx], 4),
        "nll": np.round(nll[idx], 4),
        "active": ev["active"][idx],
        "log_ev": np.round(ev["X"][idx, t2["names"].index("log_ev")], 3),
    })
    return out


# ---------------------------------------------------------------------------
# Human audit
# ---------------------------------------------------------------------------
def audit_sample_t1(t1: dict, k: int = 120) -> pd.DataFrame:
    ev = t1["ev"]
    df = t1["df"][ev].reset_index(drop=True)
    y, p = t1["y"][ev], t1["p"]
    C = t1["C"]
    names = F.ACTOR_FEATURES
    rng = np.random.default_rng(int(t1["fit"]["seed"]) if "seed" in t1["fit"] else 0)
    bins = np.digitize(p, [0.1, 0.3, 0.6, 0.9])
    rows = []
    for yv in (0, 1):
        for b in range(5):
            m = np.where((y == yv) & (bins == b))[0]
            if len(m) == 0:
                continue
            take = rng.choice(m, size=min(max(k // 10, 1), len(m)), replace=False)
            for i in take:
                top = np.argsort(-np.abs(C[i]))[:5]
                rows.append({
                    "item_id": len(rows) + 1,
                    "actor": df["actor"].iloc[i], "region": df["region"].iloc[i],
                    "t_idx": int(df["t_idx"].iloc[i]),
                    "y_high_q75": int(y[i]), "p": round(float(p[i]), 4),
                    "out_events": round(float(df["out_events"].iloc[i]), 1),
                    "out_events_next": round(float(df["out_events_next"].iloc[i]), 1),
                    "top_features": "; ".join(
                        f"{names[j]}={df.iloc[i][names[j]]:.3g} ({C[i, j]:+.2f})"
                        for j in top),
                })
    return pd.DataFrame(rows[:k])


def audit_sample_t2(t2: dict, k: int = 120) -> pd.DataFrame:
    ev = t2["eval"]
    y, p = ev["y"], t2["p"]
    C = t2["C"]
    names = t2["names"]
    rng = np.random.default_rng(11)
    bins = np.digitize(p, [0.05, 0.15, 0.4, 0.7])
    rows = []
    for yv in (0, 1):
        for b in range(5):
            m = np.where((y == yv) & (bins == b))[0]
            if len(m) == 0:
                continue
            take = rng.choice(m, size=min(max(k // 10, 1), len(m)), replace=False)
            for i in take:
                top = np.argsort(-np.abs(C[i]))[:5]
                rows.append({
                    "item_id": len(rows) + 1,
                    "did": int(ev["did"][i]), "sid": int(ev["sid"][i]),
                    "tid": int(ev["tid"][i]), "t_idx": int(ev["t_idx"][i]),
                    "y": int(y[i]), "p": round(float(p[i]), 4),
                    "top_features": "; ".join(
                        f"{names[j]}={ev['X'][i, j]:.3g} ({C[i, j]:+.2f})"
                        for j in top),
                })
    return pd.DataFrame(rows[:k])


CHECKLIST = """# Human audit checklist (framework pilot)

Two independent raters score each item blind to the outcome.  The question is
**not** "was the prediction right?" but "is the stated explanation a plausible
reading of the available evidence at t?".

For each item, score 1-5:

1. **Availability** — every feature cited is observable at month t (no label,
   no t+1 information).  (1 = clear violation, 5 = all available)
2. **Direction** — the sign of the cited contribution matches domain
   intuition (e.g. recent activity raises high-activity risk).  (1 = opposite,
   5 = clearly consistent)
3. **Magnitude** — the cited contribution is large enough to matter for this
   forecast, not a rounding artefact.  (1 = negligible, 5 = dominant)
4. **Completeness** — the card names the main driver; no obvious omitted
   driver that would reverse the story.  (1 = misleading, 5 = complete)
5. **Actionability** — a policy analyst could state what would change the
   forecast.  (1 = no, 5 = yes, with a concrete counterfactual)

Stop rules: any item with Availability <= 2 is escalated to the PI; if more
than 5% of items violate availability, the card template is rejected.

Kappa: weighted Cohen's kappa on the 5-point scores, target >= 0.60 per
dimension.  Disagreements are adjudicated by a third rater; the adjudicated
set is reported separately.
"""

KAPPA = {
    "design": "2 raters x 120 items, 20 duplicated items for intra-rater check",
    "blinding": "outcome hidden; model p shown only after scoring",
    "scale": "1-5 per dimension (availability, direction, magnitude, completeness, actionability)",
    "metric": "quadratic-weighted Cohen's kappa per dimension",
    "target_kappa": 0.60,
    "stop_rule": "availability <= 2 on any item -> PI escalation; >5% availability violations -> reject template",
    "adjudication": "third rater resolves disagreements; report raw and adjudicated kappa",
    "sample": "stratified by outcome x predicted-probability quintile",
}


def run(cfg: dict, out_path: str | None = None) -> dict:
    t0 = time.time()
    t1 = fit_t1(cfg)
    t2 = fit_t2(cfg)
    log(f"interpret: T1 LL={binary_metrics(t1['y'][t1['ev']], t1['p'])['log_loss']:.5f} "
        f"T2 LL={binary_metrics(t2['eval']['y'], t2['p'])['log_loss']:.5f}")

    cards, detailed = t1_node_cards(t1, cfg)
    attr1, attr1_meta = t1_attribution(t1, cfg)
    di1 = t1_deletion_insertion(t1, cfg)
    cf1 = t1_counterfactual(t1, cfg)
    attr2, attr2_meta = t2_attribution(t2, cfg)
    edge = t2_edge_probe(t2, cfg)
    cf2 = t2_counterfactual(t2, cfg)
    di2 = t2_deletion_insertion(t2, cfg)
    fail1 = failure_cases_t1(t1)
    fail2 = failure_cases_t2(t2)
    sample1 = audit_sample_t1(t1)
    sample2 = audit_sample_t2(t2)

    tabdir = os.path.join(cfg["paths"]["out_dir"], "tables")
    auddir = os.path.join(cfg["paths"]["out_dir"], "audit")
    os.makedirs(tabdir, exist_ok=True)
    os.makedirs(auddir, exist_ok=True)
    cards.to_csv(os.path.join(tabdir, "node_cards_actor.csv"), index=False)
    attr1.to_csv(os.path.join(tabdir, "feature_attribution_t1.csv"), index=False)
    attr2.to_csv(os.path.join(tabdir, "feature_attribution_t2.csv"), index=False)
    di1.to_csv(os.path.join(tabdir, "deletion_insertion_t1.csv"), index=False)
    di2.to_csv(os.path.join(tabdir, "deletion_insertion_t2.csv"), index=False)
    cf1.to_csv(os.path.join(tabdir, "counterfactual_t1.csv"), index=False)
    cf2.to_csv(os.path.join(tabdir, "counterfactual_t2.csv"), index=False)
    edge.to_csv(os.path.join(tabdir, "edge_attribution.csv"), index=False)
    fail1.to_csv(os.path.join(tabdir, "failure_cases_t1.csv"), index=False)
    fail2.to_csv(os.path.join(tabdir, "failure_cases_t2.csv"), index=False)
    sample1.to_csv(os.path.join(auddir, "sample_t1.csv"), index=False)
    sample2.to_csv(os.path.join(auddir, "sample_t2.csv"), index=False)
    with open(os.path.join(auddir, "checklist.md"), "w", encoding="utf-8") as f:
        f.write(CHECKLIST)
    jdump(os.path.join(auddir, "kappa_plan.json"), KAPPA)
    jdump(os.path.join(cfg["paths"]["out_dir"], "parts", "node_cards.json"),
          {"top_actors": detailed, "n_actors": int(len(cards))})

    out = {
        "t1_primary": "high_q75",
        "t1_metrics": binary_metrics(t1["y"][t1["ev"]], t1["p"]),
        "t1_threshold": t1["thr"],
        "t2_metrics": binary_metrics(t2["eval"]["y"], t2["p"]),
        "t1_feature_attribution": attr1_meta,
        "t2_feature_attribution": attr2_meta,
        "deletion_insertion_t1": di1.to_dict(orient="records"),
        "deletion_insertion_t2": di2.to_dict(orient="records"),
        "counterfactual_t1": cf1.to_dict(orient="records"),
        "counterfactual_t2": cf2.to_dict(orient="records"),
        "edge_probe": {
            "n": int(len(edge)),
            "mean_dp_remove_common": round(float(edge["dp_remove_common"].mean()), 5),
            "median_dp_remove_common": round(float(edge["dp_remove_common"].median()), 5),
            "mean_dp_remove_noncommon": round(float(edge["dp_remove_noncommon"].dropna().mean()), 5),
            "share_abs_gt_0.01": round(float((edge["dp_remove_common"].abs() > 0.01).mean()), 4),
        },
        "node_cards": {
            "n_actors": int(len(cards)),
            "mean_card_ll": round(float(cards["log_loss"].mean()), 5),
            "worst_actors": cards.dropna(subset=["auc"]).head(5)[
                ["actor", "log_loss", "auc"]].to_dict(orient="records"),
            "best_actors": cards.dropna(subset=["auc"]).tail(5)[
                ["actor", "log_loss", "auc"]].to_dict(orient="records"),
        },
        "failure_cases": {
            "t1_n": int(len(fail1)), "t2_n": int(len(fail2)),
            "t1_top": fail1.head(5).to_dict(orient="records"),
            "t2_top": fail2.head(5).to_dict(orient="records"),
        },
        "audit_materials": {
            "checklist": os.path.join(auddir, "checklist.md"),
            "sample_t1": os.path.join(auddir, "sample_t1.csv"),
            "sample_t2": os.path.join(auddir, "sample_t2.csv"),
            "kappa_plan": os.path.join(auddir, "kappa_plan.json"),
            "sample_t1_n": int(len(sample1)), "sample_t2_n": int(len(sample2)),
        },
        "cache": {
            "preds_t1_high_q75": os.path.join(cfg["paths"]["cache_dir"],
                                              "preds_t1_highq75.npz"),
        },
        "runtime_seconds": elapsed(t0),
    }
    np.savez_compressed(
        os.path.join(cfg["paths"]["cache_dir"], "preds_t1_highq75.npz"),
        y=t1["y"][t1["ev"]], aid=t1["df"]["aid"].to_numpy()[t1["ev"]],
        t_idx=t1["df"]["t_idx"].to_numpy()[t1["ev"]], p=t1["p"],
        contribution=t1["C"].astype(np.float32),
        feature_names=np.array(F.ACTOR_FEATURES),
    )
    if out_path:
        jdump(out_path, out)
    return out
