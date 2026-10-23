"""Synthetic diversity generator (train-only) with K schemes and controls.

This is **not** SALS v2 and not a calibrated simulator.  It is a lightweight
panel-level generator whose *nuisance* moments (activity, persistence, degree,
event counts) are estimated from the real training split only; the *operator*
block is a pre-registered prior varied across K schemes, as required by the
synthetic protocol.  The generator writes latent operator labels for the
5-class operator head; those labels are never available on real data.

Schemes
-------
``base``, ``birth_heavy``, ``merge_split_heavy``, ``high_noise``,
``low_coupling``, ``high_coupling``, ``random_events``, ``shuffled_kernels``.

Controls
--------
* ``random_events``: same event rate, random actor/type, no roster coupling.
* ``shuffled_kernels``: event timing/actors preserved, operator labels permuted
  relative to the applied kernel (breaks label -> outcome semantics).
* ``no_events`` (added on demand): nuisance-only parametric bootstrap.

Everything is deterministic given ``(scheme, world_seed, moments)``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from .regions import REGION_NAMES
from .utils import jdump, log, sha256_file

OPS = ["birth", "death", "merge", "split", "reentry"]
OP_CODE = {name: i + 1 for i, name in enumerate(OPS)}
CODE_OP = {0: "none", **{v: k for k, v in OP_CODE.items()}}


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def estimate_moments(train_dyad: dict, train_actor: dict) -> dict:
    """Estimate nuisance moments from real TRAIN only (no val/test reads)."""
    y = train_dyad["y"].astype(np.float64)
    a = train_dyad["active"].astype(np.float64)
    ev = train_dyad["n_events"].astype(np.float64)
    p11 = float(y[a == 1].mean()) if (a == 1).any() else 0.7
    p01 = float(y[a == 0].mean()) if (a == 0).any() else 0.05
    deg = train_actor["out_partners"].astype(np.float64)
    mu_a = float(np.log1p(train_actor["out_events"]).mean())
    sd_a = float(np.log1p(train_actor["out_events"]).std()) + 1e-6
    return {
        "p_active": float(y.mean()),
        "p11": p11,
        "p01": p01,
        "mean_events_active": float(ev[ev > 0].mean()) if (ev > 0).any() else 2.0,
        "degree_q": np.quantile(deg, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist(),
        "actor_log_events_mean": mu_a,
        "actor_log_events_sd": sd_a,
        "n_train_rows": int(len(y)),
        "source": "real train split only",
    }


@dataclass
class World:
    dyad: pd.DataFrame
    actor: pd.DataFrame
    events: pd.DataFrame
    meta: dict


def generate_world(cfg: dict, scheme_name: str, world_seed: int, moments: dict,
                   world_id: int = 0) -> World:
    scfg = cfg["synth"]
    schemes = scfg["schemes"]
    if scheme_name not in schemes:
        raise KeyError(scheme_name)
    scheme = schemes[scheme_name]
    n_actors = int(scfg["n_actors"])
    n_months = int(scfg["n_months"])
    max_actors = int(scfg["max_actors"])
    rng = np.random.default_rng(world_seed)
    A = max(max_actors, n_actors + 8)
    T = n_months

    activity = rng.normal(moments["actor_log_events_mean"],
                          max(0.15, moments["actor_log_events_sd"]), A)
    bloc = rng.integers(0, len(REGION_NAMES), A)
    alive = np.zeros(A, dtype=bool)
    alive[:n_actors] = True
    ever = np.zeros(A, dtype=bool)
    S = np.zeros((A, A), dtype=np.float32)
    S_last = np.zeros((A, A), dtype=np.float32)
    E = np.zeros((T, A, A), dtype=np.int32)
    op_current = np.zeros((T, A), dtype=np.int8)
    op_next = np.zeros((T, A), dtype=np.int8)
    event_rows: list[dict] = []

    # Initial sparse graph, degree drawn from the real train degree quantiles.
    q = np.asarray(moments["degree_q"], dtype=float)
    pdeg = np.array([0.15, 0.2, 0.3, 0.2, 0.15])
    for i in range(n_actors):
        d = int(np.clip(rng.choice(q, p=pdeg), 1, n_actors - 1))
        cand = [j for j in range(n_actors) if j != i]
        # Prefer same bloc.
        w = np.array([3.0 if bloc[j] == bloc[i] else 1.0 for j in cand], dtype=float)
        w /= w.sum()
        picks = rng.choice(cand, size=min(d, len(cand)), replace=False, p=w)
        S[i, picks] = 1.0
        S[picks, i] = 1.0

    eta = 0.0
    phi, sigma = 0.9, 0.1
    base_hazard = float(np.clip(scfg["event_rate"] * float(scheme.get("hazard", 1.0)),
                                1e-4, 0.5))
    coupling = float(scheme.get("coupling", 0.5))
    noise = float(scheme.get("noise", 0.35))
    mix = np.asarray(scheme["mix"], dtype=np.float64)
    mix = mix / mix.sum()
    shuffle_kernels = bool(scheme.get("shuffle_kernels", False))
    random_events = bool(scheme.get("random", False))
    no_events = bool(scheme.get("no_events", False))

    for t in range(T):
        eta = phi * eta + sigma * rng.normal()
        idx = np.where(alive)[0]
        n_alive = len(idx)
        fill = min(n_alive / max(n_actors, 1), 1.5)
        deg = (S > 0).sum(1)
        mean_deg = float(deg[idx].mean() / max(n_alive, 1)) if n_alive else 0.0
        dormant = float((~alive & ever).sum()) / max(n_actors, 1)
        recent = float(E[t - 1].sum()) / max(n_alive, 1) if t > 0 else 0.0

        # ---- one latent operator event per month (diversity, not calibration)
        if not no_events and n_alive > 1:
            if random_events:
                hazard = base_hazard
            else:
                hazard = _sigmoid(
                    np.log(base_hazard / (1 - base_hazard))
                    + coupling * (fill - 0.7)
                    + 0.5 * mean_deg
                    + 0.3 * np.log1p(recent)
                    + 0.6 * eta
                    + noise * rng.normal())
            if rng.random() < min(max(hazard, 0.0), 0.9):
                if random_events:
                    p = np.ones(5) / 5.0
                else:
                    base = np.log(mix + 1e-9)
                    feats = np.array([
                        2.0 * (1.0 - fill) + 0.5 * dormant,
                        2.0 * fill + 0.5 * (1.0 - mean_deg),
                        1.5 * fill + 1.0 * mean_deg,
                        1.5 * (1.0 - fill) + 0.5 * (1.0 - dormant),
                        2.0 * dormant,
                    ])
                    logits = base + coupling * feats + noise * rng.gumbel(size=5)
                    p = _softmax(logits)
                op_i = int(rng.choice(5, p=p))
                op_name = OPS[op_i]
                kernel_i = int(rng.integers(0, 5)) if shuffle_kernels else op_i
                actor = None
                other = None
                if kernel_i == 0:  # birth
                    dead = np.where(~alive & (np.arange(A) >= n_actors))[0]
                    if len(dead):
                        actor = int(rng.choice(dead))
                        alive[actor] = True
                        activity[actor] = rng.normal(moments["actor_log_events_mean"],
                                                     moments["actor_log_events_sd"])
                        bloc[actor] = int(rng.integers(0, len(REGION_NAMES)))
                        cand = idx
                        if len(cand):
                            k = int(rng.integers(1, 4))
                            picks = rng.choice(cand, size=min(k, len(cand)), replace=False)
                            S[actor, picks] = 1.0
                            S[picks, actor] = 1.0
                elif kernel_i == 1:  # death
                    cand = idx
                    if len(cand) > max(2, n_actors // 4):
                        actor = int(rng.choice(cand))
                        S_last[actor] = S[actor]
                        alive[actor] = False
                        S[actor, :] = 0
                        S[:, actor] = 0
                elif kernel_i == 2:  # merge
                    if len(idx) >= 2:
                        i, j = rng.choice(idx, size=2, replace=False)
                        actor, other = int(i), int(j)
                        activity[actor] = 0.5 * (activity[actor] + activity[other]) + 0.2
                        S[actor, :] = np.maximum(S[actor, :], S[other, :])
                        S[:, actor] = np.maximum(S[:, actor], S[:, other])
                        S[actor, actor] = 0.0
                        alive[other] = False
                        S[other, :] = 0
                        S[:, other] = 0
                elif kernel_i == 3:  # split
                    dead = np.where(~alive & (np.arange(A) >= n_actors))[0]
                    if len(dead) and len(idx):
                        actor = int(rng.choice(idx))
                        child = int(rng.choice(dead))
                        alive[child] = True
                        activity[child] = 0.8 * activity[actor]
                        bloc[child] = bloc[actor]
                        mask = (S[actor] > 0) & (rng.random(A) < 0.6)
                        S[child, mask] = 1.0
                        S[mask, child] = 1.0
                        S[actor, :] *= 0.6
                        other = child
                else:  # reentry
                    dead = np.where(~alive & ever)[0]
                    if len(dead):
                        actor = int(rng.choice(dead))
                        alive[actor] = True
                        activity[actor] = 0.8 * activity[actor]
                        mask = (S_last[actor] > 0) & (rng.random(A) < 0.5)
                        S[actor, mask] = 1.0
                        S[mask, actor] = 1.0
                if actor is not None:
                    op_current[t, actor] = OP_CODE[op_name]
                    if t > 0:
                        op_next[t - 1, actor] = OP_CODE[op_name]
                    event_rows.append({
                        "world": world_id, "scheme": scheme_name, "month": t,
                        "t_idx": t, "op": op_name, "op_code": OP_CODE[op_name],
                        "actor": actor, "other": -1 if other is None else other,
                        "kernel": OPS[kernel_i],
                    })

        # ---- dyad process on the surviving roster
        idx = np.where(alive)[0]
        if len(idx) < 2:
            continue
        sub = S[np.ix_(idx, idx)]
        a = activity[idx]
        b = bloc[idx]
        B = (sub > 0).astype(np.float32)
        deg_i = B.sum(1)
        cn = (B @ B) / np.maximum(deg_i[:, None], 1.0)
        same = (b[:, None] == b[None, :]).astype(np.float32)
        p_on = _sigmoid(-2.4 + 0.7 * (a[:, None] + a[None, :])
                        + 0.5 * same + 0.4 * cn + 0.3 * eta)
        p_off = _sigmoid(-1.1 - 1.1 * sub + 0.2 * eta)
        lat = sub > 0
        u = rng.random(sub.shape)
        new_lat = np.where(lat, u < (1.0 - p_off), u < p_on)
        mu = max(0.6, moments["mean_events_active"] * 0.8)
        lam = mu * np.exp(0.4 * (a[:, None] + a[None, :]) + 0.3 * eta) * (0.4 + sub)
        n_ev = rng.poisson(np.clip(lam, 0, 50)).astype(np.int32)
        n_ev[~new_lat] = 0
        n_ev[np.arange(len(idx)), np.arange(len(idx))] = 0
        # Direct each unordered pair randomly; keep counts integer.
        iu, ju = np.triu_indices(len(idx), k=1)
        swap = rng.random(len(iu)) < 0.5
        ii, jj = idx[iu], idx[ju]
        E[t, ii[~swap], jj[~swap]] = n_ev[iu[~swap], ju[~swap]]
        E[t, jj[swap], ii[swap]] = n_ev[iu[swap], ju[swap]]
        S[np.ix_(idx, idx)] = new_lat.astype(np.float32)
        ever[idx] |= n_ev.sum(1) > 0

    # ---- assemble actor-month panel (all A actors x T months)
    out_events = E.sum(axis=2)          # [T,A] outgoing
    in_events = E.sum(axis=1)           # [T,A] incoming
    out_partners = (E > 0).sum(axis=2)
    in_partners = (E > 0).sum(axis=1)
    out_active = out_events > 0
    in_active = in_events > 0
    actor_rows = []
    for t in range(T):
        for i in range(A):
            actor_rows.append((
                int(i), t, int(out_events[t, i]), int(in_events[t, i]),
                int(out_partners[t, i]), int(in_partners[t, i]),
                bool(out_active[t, i]), bool(in_active[t, i]),
                int(op_current[t, i]), int(op_next[t, i]) if t < T - 1 else 0,
                int(alive[i]),
            ))
    actor = pd.DataFrame(actor_rows, columns=[
        "actor_idx", "t_idx", "out_events", "in_events", "out_partners",
        "in_partners", "out_active", "in_active", "op", "op_next", "alive"])
    actor["actor"] = "S" + actor["actor_idx"].astype(str).str.zfill(3)
    actor["aid"] = actor["actor_idx"].astype(np.int16)
    actor["region"] = [REGION_NAMES[int(bloc[i])] for i in actor["actor_idx"]]
    yy = 1995 + actor["t_idx"].to_numpy() // 12
    mm = actor["t_idx"].to_numpy() % 12 + 1
    actor["m"] = pd.to_datetime({"year": yy, "month": mm, "day": 1})
    # Synthetic split by month fraction.
    tr = int(round(scfg["train_frac"] * T))
    va = int(round((scfg["train_frac"] + scfg["val_frac"]) * T))
    actor["split"] = np.where(actor["t_idx"] < tr, "train",
                              np.where(actor["t_idx"] < va, "val", "test"))
    actor["out_active_next"] = actor.groupby("actor")["out_active"].shift(-1)
    actor["in_active_next"] = actor.groupby("actor")["in_active"].shift(-1)
    actor["out_events_next"] = actor.groupby("actor")["out_events"].shift(-1)
    actor["out_active_next"] = actor["out_active_next"].fillna(False).astype(bool)
    actor["in_active_next"] = actor["in_active_next"].fillna(False).astype(bool)
    actor["out_events_next"] = actor["out_events_next"].fillna(0).astype(np.int64)

    # ---- assemble directed dyad-month panel for ever-eventful directed pairs
    any_dir = (E > 0).any(axis=0)
    pairs = np.argwhere(any_dir)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    rows = []
    united = {}
    for s, tgt in pairs:
        key = (int(s), int(tgt))
        united[key] = len(rows)
        for t in range(T):
            n = int(E[t, s, tgt])
            rows.append((key[0], key[1], t, n))
    dyad = pd.DataFrame(rows, columns=["sid", "tid", "t_idx", "n_events"])
    dyad["active"] = dyad["n_events"] > 0
    dyad = dyad.sort_values(["sid", "tid", "t_idx"]).reset_index(drop=True)
    dyad["active_next"] = dyad.groupby(["sid", "tid"])["active"].shift(-1)
    dyad["active_next"] = dyad["active_next"].astype("boolean").fillna(False).astype(bool)
    dyad["did"] = dyad.groupby(["sid", "tid"], sort=False).ngroup().astype(np.int32)
    dyad["actor_s"] = "S" + dyad["sid"].astype(str).str.zfill(3)
    dyad["actor_t"] = "S" + dyad["tid"].astype(str).str.zfill(3)
    dyad["region_s"] = [REGION_NAMES[int(bloc[i])] for i in dyad["sid"]]
    dyad["region_t"] = [REGION_NAMES[int(bloc[i])] for i in dyad["tid"]]
    dyad["same_region"] = dyad["region_s"] == dyad["region_t"]
    dyad["split"] = np.where(dyad["t_idx"] < tr, "train",
                             np.where(dyad["t_idx"] < va, "val", "test"))
    yy = 1995 + dyad["t_idx"].to_numpy() // 12
    mm = dyad["t_idx"].to_numpy() % 12 + 1
    dyad["m"] = pd.to_datetime({"year": yy, "month": mm, "day": 1})
    dyad["root"] = "00"
    dyad["root_next"] = "00"
    # Bounded size: if a world explodes, subsample dyads (deterministically).
    max_rows = int(scfg.get("max_rows_per_world", 400000))
    if len(dyad) > max_rows:
        keep = dyad["did"].drop_duplicates().sample(
            frac=max_rows / len(dyad), random_state=world_seed).to_numpy()
        dyad = dyad[dyad["did"].isin(keep)].reset_index(drop=True)
        dyad["did"] = dyad.groupby(["sid", "tid"], sort=False).ngroup().astype(np.int32)

    events = pd.DataFrame(event_rows)
    meta = {
        "scheme": scheme_name, "world": int(world_id), "seed": int(world_seed),
        "n_actors": A, "n_months": T,
        "dyad_rows": int(len(dyad)), "actor_rows": int(len(actor)),
        "events": int(len(events)),
        "event_by_op": events["op"].value_counts().to_dict() if len(events) else {},
        "stipulated": {k: v for k, v in scheme.items()},
        "nuisance_moments": {k: v for k, v in moments.items() if k != "degree_q"},
        "note": "synthetic diversity prior; operator parameters stipulated, not calibrated",
    }
    return World(dyad=dyad, actor=actor, events=events, meta=meta)


def _register_and_build(con: duckdb.DuckDBPyConnection, world: World,
                        cfg: dict, world_tag: str) -> None:
    """Register a world and build the same feature tables as the real panel."""
    from . import features as F

    d = world.dyad.rename(columns={"actor_s": "s", "actor_t": "t"}).copy()
    d["m"] = d["m"].dt.date
    a = world.actor.rename(columns={"actor_idx": "aid"}).copy()
    actors = a[["actor", "aid", "region"]].drop_duplicates("aid")
    con.register("syn_dyad_df", d)
    con.register("syn_actor_df", a)
    con.register("syn_actors_df", actors)
    con.execute("CREATE OR REPLACE TABLE panel AS SELECT * FROM syn_dyad_df")
    con.execute("CREATE OR REPLACE TABLE actor_month AS SELECT * FROM syn_actor_df")
    con.execute("CREATE OR REPLACE TABLE actors AS SELECT * FROM syn_actors_df")
    F.build_dyad_base(con, cfg, tag=world_tag)
    F.build_actor_base(con, cfg, tag=world_tag)
    con.execute("DROP TABLE IF EXISTS panel; DROP TABLE IF EXISTS actor_month; DROP TABLE IF EXISTS actors;")


def _write_parquet(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, path: str) -> None:
    con.register("_tmp_write", df)
    con.execute(f"COPY (SELECT * FROM _tmp_write) TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.unregister("_tmp_write")


def _read_parquet(con: duckdb.DuckDBPyConnection, path: str) -> pd.DataFrame:
    return con.execute(f"SELECT * FROM read_parquet('{path}')").df()


def generate_corpus(cfg: dict, train_dyad: dict, train_actor: dict,
                    force: bool = False, cache_subdir: str = "synth",
                    tag_prefix: str = "_synth") -> dict:
    """Generate K schemes x worlds, cache parquet + feature tables.

    ``cache_subdir`` / ``tag_prefix`` let a full-scale corpus live next to the
    original smoke-scale cache without overwriting it.
    """
    cache = os.path.join(cfg["paths"]["cache_dir"], cache_subdir)
    os.makedirs(cache, exist_ok=True)
    moments = estimate_moments(train_dyad, train_actor)
    jdump(os.path.join(cache, "moments.json"), moments)
    schemes = list(cfg["synth"]["schemes"].keys())
    k = int(cfg["synth"]["K_schemes"])
    # Keep the requested K plus the two mandatory controls.
    mandatory = [s for s in ("random_events", "shuffled_kernels") if s in schemes]
    chosen = schemes[:max(k, 1)]
    for s in mandatory:
        if s not in chosen:
            chosen.append(s)
    worlds = int(cfg["synth"]["worlds_per_scheme"])
    manifest = {"moments": moments, "schemes": chosen, "worlds_per_scheme": worlds,
                "files": {}, "worlds": []}
    con = duckdb.connect()
    con.execute("SET threads=4;")
    for si, scheme in enumerate(chosen):
        for w in range(worlds):
            world_id = si * worlds + w
            seed = int(hashlib.sha256(f"{scheme}|{w}|{cfg['seed']}".encode()).hexdigest()[:8], 16)
            tag = f"{tag_prefix}_{scheme}_{w}"
            dyad_path = os.path.join(cache, f"dyad_{scheme}_{w}.parquet")
            actor_path = os.path.join(cache, f"actor_{scheme}_{w}.parquet")
            ev_path = os.path.join(cache, f"events_{scheme}_{w}.parquet")
            meta_path = os.path.join(cache, f"meta_{scheme}_{w}.json")
            if not force and all(os.path.exists(p) for p in
                                 (dyad_path, actor_path, ev_path, meta_path)):
                log(f"synth: cache {scheme} w{w}")
            else:
                log(f"synth: generating {scheme} w{w} (seed {seed})")
                world = generate_world(cfg, scheme, seed, moments, world_id)
                _write_parquet(con, world.dyad, dyad_path)
                _write_parquet(con, world.actor, actor_path)
                _write_parquet(con, world.events, ev_path)
                jdump(meta_path, world.meta)
                log(f"synth: {scheme} w{w} dyad={len(world.dyad):,} "
                    f"events={len(world.events):,}")
            _register_and_build(con, _load_world(con, dyad_path, actor_path, ev_path),
                                cfg, tag)
            manifest["files"][f"{scheme}_{w}"] = {
                "dyad": sha256_file(dyad_path), "actor": sha256_file(actor_path),
                "events": sha256_file(ev_path),
            }
            manifest["worlds"].append({"scheme": scheme, "world": w,
                                       "seed": seed, "tag": tag})
    con.close()
    manifest["corpus_sha256"] = hashlib.sha256(
        json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()
    jdump(os.path.join(cache, "corpus_manifest.json"), manifest)
    return manifest


def _load_world(con: duckdb.DuckDBPyConnection, dyad_path: str, actor_path: str,
                ev_path: str) -> World:
    return World(_read_parquet(con, dyad_path), _read_parquet(con, actor_path),
                 _read_parquet(con, ev_path), {})


def load_world_features(cfg: dict, scheme: str, w: int, split: str,
                        tag_prefix: str = "_synth") -> dict:
    """Load synthetic feature matrices for one world/split (base 29 only)."""
    from . import features as F

    cache = os.path.join(cfg["paths"]["cache_dir"], "synth")
    con = duckdb.connect()
    con.execute("SET threads=4;")
    tag = f"{tag_prefix}_{scheme}_{w}"
    con.execute(f"CREATE OR REPLACE VIEW fb AS SELECT * FROM read_parquet('"
                f"{os.path.join(cfg['paths']['cache_dir'], 'features_dyad' + tag + '.parquet')}')")
    con.execute(f"CREATE OR REPLACE VIEW ab AS SELECT * FROM read_parquet('"
                f"{os.path.join(cfg['paths']['cache_dir'], 'features_actor' + tag + '.parquet')}')")
    d = F.export_dyad_split(con, cfg, split, {"features": []})
    a = F.export_actor_split(con, split)
    con.close()
    return {"dyad": d, "actor": a}
