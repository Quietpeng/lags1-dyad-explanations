"""Validation regimes and evaluation windows.

Regimes
-------
``temporal``       train <2015, select 2015, eval 2016-2017 (test never touched)
``actor_holdout``  20% of actors removed from train/select; evaluated on their
                   eval-window dyads as a cold-start stress test
``dyad_holdout``   20% of dyads removed from train/select; evaluated on their
                   eval-window rows with pre-eval history zeroed
``region_holdout`` one macro-region removed from train/select; evaluated on its
                   eval-window dyads

The held-out entity is removed from *both* train and the selection window.  The
evaluation rows are never used for fitting or hyper-parameter selection.

D2 windows (optional, for comparability with ``results/d2_windows``):
``W_a`` new-actor onset, ``W_b`` dyad onset (12m), ``W_c`` dormancy re-entry
(12m), ``W_d`` stable control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import tail_mask


def month_idx(date_str: str) -> int:
    y, m, _ = date_str.split("-")
    return (int(y) - 1995) * 12 + (int(m) - 1)


def temporal_masks(t_idx: np.ndarray, cfg: dict) -> dict:
    t_idx = np.asarray(t_idx)
    s = cfg["split"]
    tr_end = month_idx(s["train_end"])
    sel_start, sel_end = month_idx(s["select_start"]), month_idx(s["select_end"])
    ev_start, ev_end = month_idx(s["eval_start"]), month_idx(s["eval_end"])
    val_end = month_idx(s["val_end"])
    return {
        "train": t_idx < tr_end,
        "select": (t_idx >= sel_start) & (t_idx < sel_end),
        "eval": (t_idx >= ev_start) & (t_idx < ev_end),
        "val": (t_idx >= tr_end) & (t_idx < val_end),
        "test": t_idx >= val_end,
        "tail": tail_mask(t_idx, cfg),
    }


def holdout_actors(actors: pd.DataFrame, train_activity: dict[str, int],
                   frac: float, seed: int, region: str | None = None) -> set[str]:
    """Stratified actor holdout (region x activity tercile)."""
    rng = np.random.default_rng(seed)
    df = actors.copy()
    df["activity"] = df["actor"].map(train_activity).fillna(0).astype(float)
    df["stratum"] = df["region"].astype(str) + "|" + pd.qcut(
        df["activity"].rank(method="first"), 3, labels=False).astype(str)
    held: set[str] = set()
    if region is not None:
        held = set(df.loc[df["region"] == region, "actor"])
    else:
        for _, g in df.groupby("stratum"):
            k = max(1, int(round(frac * len(g))))
            held.update(rng.choice(g["actor"].to_numpy(), size=min(k, len(g)),
                                   replace=False).tolist())
    return held


def holdout_dyads(train_d: dict, frac: float, seed: int) -> set[int]:
    """Stratified dyad holdout: ever-active vs never-active in train."""
    rng = np.random.default_rng(seed)
    did = train_d["did"]
    y = train_d["y"]
    ever = np.zeros(int(did.max()) + 1, dtype=bool)
    ever[did] = True
    active = np.zeros(int(did.max()) + 1, dtype=bool)
    active[did[y == 1]] = True
    held: set[int] = set()
    for pool in (np.where(active)[0], np.where(ever & ~active)[0]):
        k = max(1, int(round(frac * len(pool))))
        if len(pool):
            held.update(rng.choice(pool, size=min(k, len(pool)),
                                   replace=False).tolist())
    return held


def actor_rows(d: dict, actors: set) -> np.ndarray:
    """Rows whose source or target is in the held-out set (aids or ISO3)."""
    if not actors:
        return np.zeros(len(d["y"]), dtype=bool)
    vals = np.array(sorted(actors))
    if vals.dtype.kind in "iu":
        return np.isin(d["sid"], vals) | np.isin(d["tid"], vals)
    s_iso = d.get("sid_iso")
    t_iso = d.get("tid_iso")
    if s_iso is None or t_iso is None:
        raise AssertionError("actor_rows needs aid ints or sid_iso/tid_iso")
    return np.isin(s_iso, vals) | np.isin(t_iso, vals)


def regime_masks(cfg: dict, train_d: dict, select_d: dict, eval_d: dict,
                 actors: pd.DataFrame, actor_train_activity: dict) -> dict:
    """Return per-regime masks over the supplied split dictionaries."""
    regimes = cfg["validation"]["regimes"]
    frac = float(cfg["validation"]["holdout_frac"])
    seed = int(cfg["validation"]["holdout_seed"])
    iso_to_aid = dict(zip(actors["actor"], actors["aid"].astype(int)))
    out: dict[str, dict] = {}

    tmask = temporal_masks(eval_d["t_idx"], cfg)
    out["temporal"] = {
        "train": np.ones(len(train_d["y"]), dtype=bool),
        "select": np.ones(len(select_d["y"]), dtype=bool),
        "eval": tmask["eval"],
        "heldout_actors": set(),
        "heldout_actor_iso": set(),
        "heldout_dyads": set(),
        "kind": "temporal",
        "note": "standard temporal split; no entity holdout",
    }

    if "actor_holdout" in regimes:
        held_iso = holdout_actors(actors, actor_train_activity, frac, seed)
        held = {iso_to_aid[a] for a in held_iso if a in iso_to_aid}
        out["actor_holdout"] = {
            "train": ~actor_rows(train_d, held),
            "select": ~actor_rows(select_d, held),
            "eval": actor_rows(eval_d, held),
            "heldout_actors": held,
            "heldout_actor_iso": held_iso,
            "heldout_dyads": set(),
            "kind": "actor",
            "note": f"{len(held)} actors held out; ego features imputed",
        }

    if "dyad_holdout" in regimes:
        held_d = holdout_dyads(train_d, frac, seed + 1)
        out["dyad_holdout"] = {
            "train": ~np.isin(train_d["did"], list(held_d)),
            "select": ~np.isin(select_d["did"], list(held_d)),
            "eval": np.isin(eval_d["did"], list(held_d)),
            "heldout_actors": set(),
            "heldout_actor_iso": set(),
            "heldout_dyads": held_d,
            "kind": "dyad",
            "note": f"{len(held_d)} dyads held out; pre-eval history zeroed",
        }

    if "region_holdout" in regimes:
        region = cfg["validation"]["region"]
        held_iso = set(actors.loc[actors["region"] == region, "actor"])
        held = {iso_to_aid[a] for a in held_iso if a in iso_to_aid}
        out["region_holdout"] = {
            "train": ~actor_rows(train_d, held),
            "select": ~actor_rows(select_d, held),
            "eval": actor_rows(eval_d, held),
            "heldout_actors": held,
            "heldout_actor_iso": held_iso,
            "heldout_dyads": set(),
            "kind": "region",
            "region": region,
            "note": f"region {region}: {len(held)} actors held out",
        }
    return out


def cold_start_impute(X: np.ndarray, names: list[str], d: dict, regime: dict,
                      mu: np.ndarray) -> np.ndarray:
    """Impute entity-specific features to the training mean (standardised 0).

    For held-out actors we blank ego/graph features of that endpoint and the
    dyad-history block.  For held-out dyads we blank the dyad-history block
    only.  Current-month ``active`` / ``log_ev`` stay observed because they are
    available at prediction time.
    """
    X = X.copy()
    idx = {n: i for i, n in enumerate(names)}
    held_a = regime.get("heldout_actors") or set()
    held_d = regime.get("heldout_dyads") or set()
    if not held_a and not held_d:
        return X

    history = [
        "a1", "a2", "a3", "a4", "a5", "a6", "le1", "le2", "le3",
        "am3", "am6", "am12", "les3", "les6", "les12",
        "since_act", "since_first", "log_n_act", "rev_act", "rev_am3",
        "br", "seen_tr",
    ]
    ego_s = ["src_ev", "g_src_ev"] + [n for n in names if n.endswith("_s")]
    ego_t = ["tgt_ev", "g_tgt_ev"] + [n for n in names if n.endswith("_t")]

    rows_dyad = np.isin(d["did"], list(held_d)) if held_d else np.zeros(len(X), dtype=bool)
    rows_actor = actor_rows(d, held_a) if held_a else np.zeros(len(X), dtype=bool)
    if rows_dyad.any():
        cols = [idx[n] for n in history if n in idx]
        X[np.ix_(rows_dyad, cols)] = mu[cols]
    if rows_actor.any():
        cols = [idx[n] for n in history if n in idx]
        X[np.ix_(rows_actor, cols)] = mu[cols]
        s_held = np.isin(d["sid"], list(held_a))
        t_held = np.isin(d["tid"], list(held_a))
        cs = [idx[n] for n in ego_s if n in idx]
        ct = [idx[n] for n in ego_t if n in idx]
        if s_held.any() and cs:
            X[np.ix_(s_held, cs)] = mu[cs]
        if t_held.any() and ct:
            X[np.ix_(t_held, ct)] = mu[ct]
    return X


def actor_cold_start_impute(X: np.ndarray, names: list[str], d: dict,
                            regime: dict, mu: np.ndarray) -> np.ndarray:
    X = X.copy()
    held = regime.get("heldout_actors") or set()
    if not held:
        return X
    idx = {n: i for i, n in enumerate(names)}
    rows = np.isin(d["aid"], list(held))
    # Everything except region dummies is entity history; region is observable.
    cols = [idx[n] for n in names if not n.startswith("region_") and n in idx]
    if rows.any() and cols:
        X[np.ix_(rows, cols)] = mu[cols]
    return X


def d2_window_masks(panel: pd.DataFrame, eval_d: dict, cfg: dict) -> dict:
    """Compute W_a..W_d masks for *eval rows* using pre-test panel history.

    Corrected 2026-10-23:
    * never-active dyads (no first activation) are not treated as onsets;
    * a re-entry must be <= t (a future re-entry is not in the trailing window);
    * only months < test_start are read, so the frozen test is not touched.
    """
    lo = month_idx(cfg["split"]["eval_start"])
    test_start = month_idx(cfg["split"]["val_end"])
    p = panel.loc[panel["t_idx"] < test_start]
    n_dyads = int(p["did"].max()) + 1
    n_actors = int(max(p["sid"].max(), p["tid"].max())) + 1

    first_active = np.full(n_dyads, 10**9, dtype=np.int64)
    act = p.loc[p["active"], ["did", "t_idx"]].sort_values(["did", "t_idx"])
    d_act = act["did"].to_numpy(dtype=np.int64)
    t_act = act["t_idx"].to_numpy(dtype=np.int64)
    if len(d_act):
        np.minimum.at(first_active, d_act, t_act)
    prev = np.full(len(t_act), -999, dtype=np.int64)
    if len(t_act) > 1:
        same = d_act[1:] == d_act[:-1]
        prev[1:] = np.where(same, t_act[:-1], -999)
    reentry = (t_act - prev) >= 13
    last_re = np.full(n_dyads, -999, dtype=np.int64)
    if reentry.any():
        np.maximum.at(last_re, d_act[reentry], t_act[reentry])

    actor_first = np.full(n_actors, 10**9, dtype=np.int64)
    np.minimum.at(actor_first, p["sid"].to_numpy(dtype=np.int64),
                  p["t_idx"].to_numpy(dtype=np.int64))
    np.minimum.at(actor_first, p["tid"].to_numpy(dtype=np.int64),
                  p["t_idx"].to_numpy(dtype=np.int64))

    de = np.asarray(eval_d["did"], dtype=np.int64)
    te = np.asarray(eval_d["t_idx"], dtype=np.int64)
    fa, lr = first_active[de], last_re[de]
    sa = np.minimum(actor_first[eval_d["sid"]], actor_first[eval_d["tid"]])
    a = sa >= np.maximum(te - 11, lo)
    b = (fa <= te) & (fa >= np.maximum(te - 11, lo))
    c = (lr >= np.maximum(te - 11, lo)) & (lr <= te) & (lr >= 0)
    d = ~(a | b | c) & (fa <= te - 12) & (te >= lo)
    return {"W_a_new_actor": a, "W_b_dyad_onset": b,
            "W_c_dormancy_reentry": c, "W_d_stable": d}
