#!/usr/bin/env python3
"""Sprint-C V2 executor: matched-vintage, artifact-free, cache-consistent refit.

PUBLIC PORT — NOT RUNNABLE FROM THIS TREE
-----------------------------------------
This is the V2 runner behind ``results/framework_pilot_v2/``, shipped for audit.
The aggregate-only release cannot re-execute it: the licensed ICEWS v37 panel,
the restricted feature/graph caches, the local experiment configuration and the
frozen protocol (``reviews/round2/v2_protocol.md``, hash 184ae726...) are not
redistributed, and the row-level predictions consumed by the compare, A8 and A9
stages are withheld (``DATA_EXCLUSIONS.md`` §6-§7).  ``verify_protocol`` fails
closed when the protocol is absent, so no fit can start by accident.

The port changes this header and the input paths below only: the protocol
directory and the scratch directory are environment variables
(``NEUROSIM_V2_PROTOCOL_DIR``, ``NEUROSIM_V2_SCRATCH``) instead of hard-coded
locations.  No fit, contrast, table or number changed.  The frozen manifest
records the pre-port digest
``a8f83417dfc7670740571ff19b3df8a65031e92dfd3459b6b7faaedd22bc0843`` for
``experiments/sprint_c_v2.py``; the shipped file necessarily hashes differently,
so the aggregate tree and its ``SHA256SUMS`` remain the authoritative record.

Implements ``reviews/round2/v2_protocol.md`` (hash 184ae726...), frozen before
any V2 fit.  It never edits the protocol, the V1 caches or the V1 results, and
it writes only under ``results/framework_pilot_v2/`` (or an explicit --out) and
the approved scratch dir.

Stages
------
    python experiments/sprint_c_v2.py --stage all
    python experiments/sprint_c_v2.py --stage selftest
    python experiments/sprint_c_v2.py --stage fits          # F1..F9, D1
    python experiments/sprint_c_v2.py --stage fits_p1       # F2..F7
    python experiments/sprint_c_v2.py --stage fits_h0       # F1,F8,F9,D1
    python experiments/sprint_c_v2.py --stage compare
    python experiments/sprint_c_v2.py --stage a8
    python experiments/sprint_c_v2.py --stage a9
    python experiments/sprint_c_v2.py --stage report

``--stage all`` runs the stages in separate child processes so that peak RSS
from one stage cannot leak into the next.  Every output is reproducible from
this file plus the frozen inputs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from framework import aggregation as A  # noqa: E402
from framework import data, features, models, sprint_a  # noqa: E402
from framework import sprint_a_inference as inf  # noqa: E402
from framework.config import load_config  # noqa: E402
from framework.metrics import (binary_metrics, within_entity_auc)  # noqa: E402
from framework.utils import (host_info, jdump, jload, log, pkg_versions,  # noqa: E402
                             sha256_file)

# ---------------------------------------------------------------------------
# Frozen constants (protocol §1, §2)
# ---------------------------------------------------------------------------
# The frozen protocol is not redistributed.  ``NEUROSIM_V2_PROTOCOL_DIR`` may
# point at a licensed directory holding ``v2_protocol.md`` and
# ``v2_protocol.sha256``; the default keeps the restricted tree's layout so a
# licensed working copy runs unmodified.
PROTOCOL_DIR = os.environ.get(
    "NEUROSIM_V2_PROTOCOL_DIR", os.path.join(REPO, "reviews", "round2"))
PROTO_MD = os.path.join(PROTOCOL_DIR, "v2_protocol.md")
PROTO_SHA = os.path.join(PROTOCOL_DIR, "v2_protocol.sha256")
PROTOCOL_HASH = "184ae726ead6412c2515317e9d6169e61eb85308427849c9aa5e7427cbe57676"

OUT_DEFAULT = os.path.join("results", "framework_pilot_v2")
# Scratch space for ``--dry-run``; never inside the results tree.  Override with
# ``NEUROSIM_V2_SCRATCH``.  The restricted tree used a host-specific directory.
SCRATCH_DEFAULT = os.path.abspath(os.environ.get(
    "NEUROSIM_V2_SCRATCH", os.path.join(REPO, "build", "v2_scratch")))
PREREG_REL = os.path.join("results", "framework_pilot", "holdout",
                          "preregistration.json")

EMBARGO_T = (239, 251, 275, 335)
TRAIN_END, SELECT_START, SELECT_END = 238, 240, 250
INNER_START, INNER_END = 252, 274
EVAL_START, EVAL_END = 276, 334
EXT_END = 274

SEED_BASE = 20261023
REPS = 1000
PRIMARY_SEEDS = [20261023, 20261024, 20261025, 20261026, 20261027]
META_KEYS = ("y", "did", "sid", "tid", "t_idx", "active")

DYAD_CAUSAL = list(features.DYAD_CAUSAL)
GRAPH20 = list(features.GRAPH_CORE20)
FEATURES_29 = DYAD_CAUSAL + ["br", "seen_tr"]
FEATURES_49 = DYAD_CAUSAL + GRAPH20 + ["br", "seen_tr"]
FEATURES_47 = [c for c in FEATURES_49 if c not in ("br", "seen_tr")]
FEATURES_25 = [c for c in DYAD_CAUSAL if c not in ("src_ev", "tgt_ev")]

assert len(DYAD_CAUSAL) == 27 and len(GRAPH20) == 20
assert len(FEATURES_29) == 29 and len(FEATURES_49) == 49
assert len(FEATURES_47) == 47 and len(FEATURES_25) == 25
assert set(FEATURES_47) == set(FEATURES_49) - {"br", "seen_tr"}
assert set(FEATURES_25) == set(DYAD_CAUSAL) - {"src_ev", "tgt_ev"}

FIT_SPECS = {
    "F1": ("49", "_holdout", "tr", "V1 replication (49@H0)"),
    "F2": ("49", "", "tr", "as-specified (49@P1)"),
    "F3": ("47", "", "tr", "PRIMARY (47@P1)"),
    "F4": ("29", "", "tr", "29 sensitivity (29@P1, C=1,T=1)"),
    "F5": ("25", "", "tr", "key secondary (25@P1)"),
    "F6": ("49", "", "ext", "A6 2017-11 vintage (49x@P1)"),
    "F7": ("47", "", "ext", "A6 2017-11 vintage (47x@P1)"),
    "F8": ("47", "_holdout", "tr", "cache sensitivity (47@H0)"),
    "F9": ("25", "_holdout", "tr", "cache sensitivity (25@H0)"),
    "D1": ("29", "_holdout", "tr", "29 reconciliation, C=0.3 T=1.1"),
}
FEATURE_KEYS = {"49": FEATURES_49, "47": FEATURES_47,
                "29": FEATURES_29, "25": FEATURES_25}
FIT_ORDER = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "D1"]
FIT_CT = {"D1": (0.3, 1.1)}
ARM_OF = {"F1": "A0/A7", "F2": "A1", "F3": "A2/A3", "F4": "A4", "F5": "A5",
          "F6": "A6", "F7": "A6", "F8": "A7", "F9": "A7", "D1": "D-2 diagnostic"}
COMPARATOR_OF = {"F1": "warm_orig_markov", "F2": "warm_match_markov",
                 "F3": "warm_match_markov", "F4": "warm_match_markov",
                 "F5": "warm_match_markov", "F6": "warm_orig_markov",
                 "F7": "warm_orig_markov", "F8": "warm_match_markov",
                 "F9": "warm_match_markov", "D1": "warm_match_markov"}
WARM_KEYS = ["warm_orig_markov", "warm_orig_rate",
             "warm_match_markov", "warm_match_rate"]

# 14 canonical contrasts (§8/§9; C9 requires exactly 14 rows)
CONTRASTS = [
    ("A0_F1_vs_warm_orig_markov", "F1", "warm_orig_markov", "A0 replication"),
    ("A1_F2_vs_warm_match_markov", "F2", "warm_match_markov", "A1 as-specified"),
    ("A2_F3_vs_warm_match_markov", "F3", "warm_match_markov", "A2 PRIMARY"),
    ("A2b_F2_vs_warm_orig_markov", "F2", "warm_orig_markov", "A1 both comparators"),
    ("A3_F3_vs_warm_orig_markov", "F3", "warm_orig_markov", "A3 as-specified"),
    ("A2r_F3_vs_warm_match_rate", "F3", "warm_match_rate", "A2 rate comparator"),
    ("A4_F4_vs_warm_match_markov", "F4", "warm_match_markov", "A4 sensitivity"),
    ("A5_F5_vs_warm_match_markov", "F5", "warm_match_markov", "A5 key secondary"),
    ("A6a_F6_vs_warm_orig_markov", "F6", "warm_orig_markov", "A6 49x vintage"),
    ("A6b_F7_vs_warm_orig_markov", "F7", "warm_orig_markov", "A6 47x vintage"),
    ("A7a_F1_vs_warm_match_markov", "F1", "warm_match_markov", "A7 49@H0 cache"),
    ("A7b_F8_vs_warm_match_markov", "F8", "warm_match_markov", "A7 47@H0 cache"),
    ("A7c_F9_vs_warm_match_markov", "F9", "warm_match_markov", "A7 25@H0 cache"),
    ("NESTED_F5_base_vs_F3_model", "F3", "F5", "nested 47 vs 25"),
]
CLASS_ORDER = ["source_actor", "target_actor", "dyad", "calendar", "graph",
               "train_stats"]
# Audit report order (contribution_fit §3): source, target, dyad, graph,
# calendar, train_stats.  The protocol quotes the same six numbers in that
# order; the V2 recomputation (graph ~18%, calendar ~3%) confirms the mapping.
V1_SHARES = {"source_actor": 21.58, "target_actor": 23.35, "dyad": 29.74,
             "graph": 16.96, "calendar": 2.80, "train_stats": 5.57}
V1_SHARES_PROTOCOL_LITERAL = {"source_actor": 21.58, "target_actor": 23.35,
                              "dyad": 29.74, "calendar": 16.96,
                              "graph": 2.80, "train_stats": 5.57}
V1_NOT_ACTOR = 55.07
V1_NOT_ACTOR_NRA = 62.27
V1_ALLOC = (55.07, 38.11, 46.59)


def own_class(names) -> dict:
    """Feature -> ownership class, tolerant of subsets (47/25 columns)."""
    m = {}
    for b, feats in (("source_actor", A.SOURCE_OWN),
                     ("target_actor", A.TARGET_OWN),
                     ("dyad", A.DYAD_PAIR), ("calendar", A.CALENDAR),
                     ("graph", A.GRAPH_SYM), ("train_stats", A.TRAIN_STATS)):
        for f in feats:
            m[f] = b
    missing = [n for n in names if n not in m]
    if missing:
        raise KeyError(f"unmapped features: {missing}")
    return {n: m[n] for n in names}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _rel(path: str) -> str:
    return os.path.relpath(os.path.abspath(path), REPO)


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def jdump_clean(path: str, obj) -> None:
    jdump(path, _clean(obj))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def arr_hash(a) -> str:
    return sha256_bytes(np.ascontiguousarray(np.asarray(a)).tobytes())


def raw_log_loss(y, p) -> float:
    """Unrounded log-loss with the framework's 1e-12 clip (for §6.1 anchors)."""
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def write_table(df: pd.DataFrame, path: str, schema: str) -> str:
    df = df.copy()
    df["schema"] = schema
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, float_format="%.10g")
    return sha256_file(path)


def check(out: str) -> None:
    if os.path.exists(os.path.join(out, "summary.json")):
        sys.stderr.write("V2 outputs already exist; refusing to overwrite\n")
        raise SystemExit(2)


def ensure_dirs(out: str) -> None:
    for sub in ("", "once", "parts", "tables", "preds"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)


def make_cfg(out: str) -> dict:
    cfg = load_config(overrides={
        "paths": {"duckdb_memory": "4GB", "duckdb_threads": 4},
        "models": {"logistic": {"C_grid": [1.0]}},
    })
    cfg["paths"]["out_dir"] = os.path.abspath(out)
    return cfg


def verify_protocol() -> str:
    for path in (PROTO_MD, PROTO_SHA):
        if not os.path.isfile(path):
            raise SystemExit(
                "V2 protocol not redistributed: "
                f"{os.path.basename(path)} is absent from "
                f"{os.path.relpath(PROTOCOL_DIR, REPO)}; the aggregate-only "
                "release cannot re-run the V2 fits.  Set "
                "NEUROSIM_V2_PROTOCOL_DIR to a licensed copy of the frozen "
                "protocol (see DATA_EXCLUSIONS.md).")
    with open(PROTO_SHA, encoding="utf-8") as fh:
        frozen = fh.read().split()[0].strip()
    actual = sha256_file(PROTO_MD)
    if not (frozen == actual == PROTOCOL_HASH):
        raise SystemExit(f"protocol hash mismatch: frozen={frozen} "
                         f"actual={actual} expected={PROTOCOL_HASH}")
    return actual


def load_prereg() -> dict:
    payload = sprint_a.verify_preregistration(os.path.join(REPO, PREREG_REL))
    d = payload["definition"]
    if len(d["pairs"]) != 2790 or len(d["dids"]) != 5282:
        raise SystemExit("preregistration cardinality mismatch")
    want = {
        "definition": "776a763e663b845003cd628d0656692d8ff2652aaa5cec18820e078baec62c8e",
        "pairs": "9d3e2752c4f01ef3ab02388717b0a429069ca5b67292863c3e33ad5fc23a39b3",
        "dids": "1c2d3f0b6356cf617100bd2df9afe6091403c641c0a8d7d7618b5de5fe7c991a",
    }
    got = {"definition": sprint_a.sha256_json(d),
           "pairs": sprint_a.sha256_json(d["pairs"]),
           "dids": sprint_a.sha256_json(d["dids"])}
    if got != want:
        raise SystemExit(f"preregistration hash mismatch: {got} != {want}")
    return payload


def input_hashes(cfg: dict) -> dict:
    cache = cfg["paths"]["cache_dir"]
    out = {}
    for name in ("panel_dyad.parquet", "features_dyad.parquet", "actors.parquet"):
        p = os.path.join(cache, name)
        out[_rel(p)] = sha256_file(p)
    for meta_name in ("graph_meta.json", "graph_meta_holdout.json"):
        mp = os.path.join(cache, meta_name)
        out[_rel(mp)] = sha256_file(mp)
        meta = jload(mp)
        tag = "_holdout" if meta_name.endswith("holdout.json") else ""
        for split, want in sorted(meta.get("files", {}).items()):
            p = os.path.join(cache, f"graph_{split}{tag}.npy")
            got = sha256_file(p)
            if got != want:
                raise SystemExit(f"graph cache hash mismatch {p}: {got} != {want}")
            out[_rel(p)] = got
    for name in ("preregistration.json", "preregistration.json.sha256"):
        p = os.path.join(REPO, "results", "framework_pilot", "holdout", name)
        out[_rel(p)] = sha256_file(p)
    return out


# ---------------------------------------------------------------------------
# Masks and bundle (§2)
# ---------------------------------------------------------------------------
def _slice(d: dict, mask: np.ndarray) -> dict:
    n = len(d["y"])
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
            out[k] = v[mask]
        elif k == "feature_names":
            out[k] = list(v)
        elif isinstance(v, (int, float, str)) or v is None:
            out[k] = v
    return out


def _masks(d: dict, holdout_dids: np.ndarray) -> dict:
    t = d["t_idx"]
    hd = np.isin(d["did"], holdout_dids)
    emb = np.isin(t, EMBARGO_T)
    return {
        "tr": (t <= TRAIN_END) & ~hd & ~emb,
        "sel": (t >= SELECT_START) & (t <= SELECT_END) & ~hd & ~emb,
        "ev": (t >= EVAL_START) & (t <= EVAL_END) & hd & ~emb,
        "ext": (t <= EXT_END) & ~hd,
        "inn": (t >= INNER_START) & (t <= INNER_END),
    }


def _drop_keys(d: dict) -> dict:
    for k in ("root_next", "n_events"):
        d.pop(k, None)
    return d


def warm_counts(con, holdout_dids, cutoff: int, n_dyads: int) -> dict:
    con.register("v2_hd_tmp", pd.DataFrame({"did": holdout_dids}))
    rows = con.execute(f"""
        SELECT p.did::INTEGER AS did, p.active::INT AS a, p.active_next::INT AS y
        FROM panel p JOIN v2_hd_tmp h USING (did)
        WHERE p.t_idx < {int(cutoff)}
    """).fetchall()
    con.unregister("v2_hd_tmp")
    n = np.zeros(n_dyads); k = np.zeros(n_dyads)
    n1 = np.zeros(n_dyads); n11 = np.zeros(n_dyads)
    n0 = np.zeros(n_dyads); n01 = np.zeros(n_dyads)
    for did, a, y in rows:
        d = int(did); a = int(a); y = int(y)
        n[d] += 1; k[d] += y
        if a == 1:
            n1[d] += 1; n11[d] += y
        else:
            n0[d] += 1; n01[d] += y
    return {"n": n, "k": k, "n1": n1, "n11": n11, "n0": n0, "n01": n01,
            "cutoff": int(cutoff), "rows": int(len(rows))}


def warm_predict(eval_d: dict, counts: dict, g: float, pg: np.ndarray) -> dict:
    d = eval_d["did"].astype(np.int64)
    a = eval_d["active"].astype(np.int64)
    markov = np.where(
        a == 1,
        (counts["n11"][d] + 2.0 * pg[1, 1]) / (counts["n1"][d] + 2.0),
        (counts["n01"][d] + 2.0 * pg[0, 1]) / (counts["n0"][d] + 2.0))
    rate = (counts["k"][d] + 2.0 * g) / (counts["n"][d] + 2.0)
    return {"markov": markov.astype(np.float64), "rate": rate.astype(np.float64)}


def build_bundle(cfg, graph_tag: str, holdout_dids: np.ndarray, pr: dict,
                 keep_tr: bool = True, need_inn: bool = False,
                 need_ext_extra: bool = False) -> dict:
    t0 = time.time()
    con = data.connect(cfg)
    data.load_views(con, cfg)
    features.build_dyad_base(con, cfg, force=False)
    if graph_tag == "_holdout":
        sprint_a.make_panel_view(con, holdout_dids)
    panel_view = "panel" if graph_tag == "" else "panel_sprint_a"
    gm = features.build_graph_features(con, cfg, force=False, tag=graph_tag,
                                       panel_view=panel_view)
    if not gm.get("features"):
        raise SystemExit("graph cache missing; protocol forbids a rebuild")

    raw = {s: features.export_dyad_split(con, cfg, s, gm, graph_tag=graph_tag)
           for s in ("train", "val", "test")}
    n_dyads = int(max(v["did"].max() for v in raw.values())) + 1

    for s, d in raw.items():
        order = np.lexsort((d["did"], d["t_idx"]))
        if not np.array_equal(order, np.arange(len(order))):
            raise SystemExit(f"{s} not (t_idx,did)-sorted")

    train_full, val_full, test_full = raw["train"], raw["val"], raw["test"]
    mtr = _masks(train_full, holdout_dids)
    mva = _masks(val_full, holdout_dids)
    mte = _masks(test_full, holdout_dids)

    hd_tr = ~np.isin(train_full["did"], holdout_dids)
    hd_va = ~np.isin(val_full["did"], holdout_dids)
    tr_d = _drop_keys(_slice(train_full, mtr["tr"]))
    sel_d = _drop_keys(_slice(val_full, mva["sel"]))
    ev_d = _drop_keys(_slice(test_full, mte["ev"]))
    tr_239 = _drop_keys(_slice(train_full, (train_full["t_idx"] == 239) & hd_tr))
    val_ext = _drop_keys(_slice(val_full, (val_full["t_idx"] >= SELECT_START) &
                                (val_full["t_idx"] <= EXT_END) & hd_va))
    inn_d = _drop_keys(_slice(val_full, mva["inn"])) if need_inn else None
    del train_full, val_full, test_full, raw
    gc.collect()

    stats = features.fit_dyad_stats(tr_d, n_dyads)
    for d, is_train in ((tr_d, True), (sel_d, False), (ev_d, False)):
        features.apply_dyad_stats(d, stats, is_train)
    if inn_d is not None:
        features.apply_dyad_stats(inn_d, stats, False)

    wm = warm_counts(con, holdout_dids, 239, n_dyads)
    wo = warm_counts(con, holdout_dids, 275, n_dyads)
    a = tr_d["active"].astype(np.int64)
    y = tr_d["y"].astype(np.int64)
    pg = np.zeros((2, 2))
    for x in (0, 1):
        m = a == x
        pg[x] = (np.bincount(y[m], minlength=2) + 1.0) / (m.sum() + 2.0)
    g = float(stats["global_rate"])
    p_match = warm_predict(ev_d, wm, g, pg)
    p_orig = warm_predict(ev_d, wo, g, pg)

    counts = {"tr": int(len(tr_d["y"])), "sel": int(len(sel_d["y"])),
              "ev": int(len(ev_d["y"])), "ev_pos": int(ev_d["y"].sum()),
              "ev_dyads": int(len(np.unique(ev_d["did"]))),
              "inn": int(len(inn_d["y"])) if inn_d is not None else None,
              "n_dyads": int(n_dyads),
              "holdout_dids": int(len(holdout_dids)),
              "tr_dyads": int(len(np.unique(tr_d["did"])))}
    bundle = {
        "graph_tag": graph_tag, "gm": gm, "n_dyads": n_dyads,
        "tr": tr_d, "sel": sel_d, "ev": ev_d, "inn": inn_d,
        "tr_239": tr_239, "val_ext": val_ext,
        "stats": stats, "pg": pg, "g": g,
        "warm_counts": {"match": wm, "orig": wo},
        "p_warm": {"warm_match_markov": p_match["markov"],
                   "warm_match_rate": p_match["rate"],
                   "warm_orig_markov": p_orig["markov"],
                   "warm_orig_rate": p_orig["rate"]},
        "counts": counts, "load_seconds": round(time.time() - t0, 1),
    }
    con.close()
    if not keep_tr:
        bundle["tr"] = None
    return bundle


# ---------------------------------------------------------------------------
# Pre-fit self-test (§6.1)
# ---------------------------------------------------------------------------
def selftest(cfg, holdout_dids, out) -> dict:
    log("selftest: protocol and preregistration")
    ph = verify_protocol()
    load_prereg()
    con = data.connect(cfg)
    data.load_views(con, cfg)
    con.register("v2_hd_tmp", pd.DataFrame({"did": holdout_dids}))
    q = lambda s: con.execute(s).fetchall()  # noqa: E731
    tr_agg = q("""SELECT count(*), sum(active_next::INT) k, sum(active::INT) n1,
        sum(CASE WHEN active THEN active_next ELSE 0 END) n11,
        sum(CASE WHEN NOT active THEN 1 ELSE 0 END) n0,
        sum(CASE WHEN NOT active THEN active_next ELSE 0 END) n01
        FROM panel p ANTI JOIN v2_hd_tmp USING(did) WHERE t_idx<=238""")[0]
    g_sql = float(q("SELECT avg(active_next::INT) FROM panel p "
                    "ANTI JOIN v2_hd_tmp USING(did) WHERE t_idx<=238")[0][0])
    h238 = q("SELECT count(*), sum(active_next::INT), sum(active::INT) "
             "FROM panel p SEMI JOIN v2_hd_tmp USING(did) WHERE t_idx<=238")[0]
    h274 = q("SELECT count(*), sum(active_next::INT), sum(active::INT) "
             "FROM panel p SEMI JOIN v2_hd_tmp USING(did) WHERE t_idx<=274")[0]
    n_dyads = int(q("SELECT max(did)+1 FROM panel")[0][0])
    pg_rows = q("""SELECT
        sum(CASE WHEN active THEN active_next ELSE 0 END) n11,
        sum(CASE WHEN active THEN 1 ELSE 0 END) n1,
        sum(CASE WHEN NOT active THEN active_next ELSE 0 END) n01,
        sum(CASE WHEN NOT active THEN 1 ELSE 0 END) n0
        FROM panel p ANTI JOIN v2_hd_tmp USING(did) WHERE t_idx<=238""")[0]
    n11, n1, n01, n0 = [float(v) for v in pg_rows]
    pg = np.array([[0.0, (n01 + 1.0) / (n0 + 2.0)],
                   [0.0, (n11 + 1.0) / (n1 + 2.0)]])
    wo = warm_counts(con, holdout_dids, 275, n_dyads)
    con.register("v2_hd_tmp", pd.DataFrame({"did": holdout_dids}))
    ev = con.execute("""SELECT did::INTEGER, active::INT, active_next::INT
        FROM panel p SEMI JOIN v2_hd_tmp USING(did)
        WHERE t_idx BETWEEN 276 AND 334 ORDER BY t_idx, did""").fetchall()
    con.unregister("v2_hd_tmp")
    con.close()
    ev_did = np.array([r[0] for r in ev], dtype=np.int64)
    ev_act = np.array([r[1] for r in ev], dtype=np.int64)
    ev_y = np.array([r[2] for r in ev], dtype=np.int64)
    pm = np.where(ev_act == 1,
                  (wo["n11"][ev_did] + 2.0 * pg[1, 1]) / (wo["n1"][ev_did] + 2.0),
                  (wo["n01"][ev_did] + 2.0 * pg[0, 1]) / (wo["n0"][ev_did] + 2.0))
    pr_ = (wo["k"][ev_did] + 2.0 * g_sql) / (wo["n"][ev_did] + 2.0)
    ll_markov = raw_log_loss(ev_y, pm)
    ll_rate = raw_log_loss(ev_y, pr_)
    checks = {
        "protocol_sha256": ph,
        "warm_orig_markov_ll": ll_markov,
        "warm_orig_markov_expected": 0.209729119,
        "warm_orig_rate_ll": ll_rate,
        "warm_orig_rate_expected": 0.214681752,
        "g": g_sql, "g_expected": 0.11076215034909703,
        "pg11": float(pg[1, 1]), "pg11_expected": 0.52655010856721751,
        "pg01": float(pg[0, 1]), "pg01_expected": 0.059203404116805962,
        "tr_agg": [int(v) for v in tr_agg],
        "ev_rows": int(len(ev_y)), "ev_pos": int(ev_y.sum()),
        "holdout_t238": [int(v) for v in h238],
        "holdout_t274": [int(v) for v in h274],
        "row_identity": int(h274[0] - h238[0]),
    }
    ok = (abs(ll_markov - 0.209729119) <= 1e-9
          and abs(ll_rate - 0.214681752) <= 1e-9
          and abs(g_sql - 0.11076215034909703) <= 1e-12
          and abs(pg[1, 1] - 0.52655010856721751) <= 1e-12
          and abs(pg[0, 1] - 0.059203404116805962) <= 1e-12
          and tuple(int(v) for v in tr_agg) == (5055328, 559939, 557717, 293666, 4497611, 266273)
          and tuple(int(v) for v in h238) == (1262398, 142512, 141935)
          and tuple(int(v) for v in h274) == (1452550, 164709, 164223)
          and int(h274[0] - h238[0]) == 190152
          and int(len(ev_y)) == 311638 and int(ev_y.sum()) == 32614)
    checks["pass"] = bool(ok)
    log(f"selftest: warm LL {ll_markov:.9f} / {ll_rate:.9f} g={g_sql:.17f} "
        f"pass={ok}")
    ensure_dirs(out)
    jdump_clean(os.path.join(out, "parts", "selftest.json"),
                {"checks": checks, "utc": sprint_a.utc_now()})
    if not ok:
        raise SystemExit("PRE-FIT SELF-TEST FAILED — no fit may start")
    return checks


# ---------------------------------------------------------------------------
# Fits (§5, §7, §8)
# ---------------------------------------------------------------------------
def _col_idx(d, cols):
    names = d["feature_names"]
    return [names.index(c) for c in cols]


def fit_one(cfg, d_fit, d_sel, cols, fit_id) -> dict:
    C, T = FIT_CT.get(fit_id, (1.0, 1.0))
    ci = _col_idx(d_fit, cols)
    t0 = time.time()
    fit = models.fit_logistic(d_fit["X"][:, ci], d_fit["y"],
                              d_sel["X"][:, ci], d_sel["y"], cfg,
                              C_grid=[C], feature_names=cols)
    dt = time.time() - t0
    n_iter = int(np.asarray(getattr(fit["model"], "n_iter_", [-1])).ravel()[0])
    meta = {"fit_id": fit_id, "features": cols, "n_features": len(cols),
            "C": float(C), "T": float(T),
            "select_log_loss": float(fit["select_log_loss"]),
            "intercept": float(fit["intercept"]),
            "coef": np.asarray(fit["coef"], float).tolist(),
            "mu_hash": arr_hash(fit["mu"]), "sd_hash": arr_hash(fit["sd"]),
            "n_iter": n_iter, "converged": bool(n_iter < 300),
            "runtime_seconds": round(dt, 2),
            "train_rows": int(len(d_fit["y"])),
            "train_pos": int(d_fit["y"].sum()),
            "select_rows": int(len(d_sel["y"])),
            "select_in_sample": bool(fit_id in ("F6", "F7"))}
    return {"fit": fit, "meta": meta, "ci": ci}


def predict_one(fit, d, cols, T=1.0) -> np.ndarray:
    ci = _col_idx(d, cols)
    p = models.predict_logistic(fit, d["X"][:, ci])
    if T != 1.0:
        p = models.apply_temperature(p, T)
    return np.asarray(p, dtype=np.float64)


def _load_fits(out, stage):
    fp = os.path.join(out, "parts", "arm_fits.json")
    pp = os.path.join(out, "parts", f"preds_{stage}.npz")
    fits = jload(fp) if os.path.exists(fp) else {}
    preds = {k: v for k, v in np.load(pp).items()} if os.path.exists(pp) else {}
    return fits, preds


def _save_fits(out, fits, preds, stage, ev_ref):
    jdump_clean(os.path.join(out, "parts", "arm_fits.json"), fits)
    payload = {k: v for k, v in preds.items() if k not in META_KEYS}
    np.savez_compressed(os.path.join(out, "parts", f"preds_{stage}.npz"),
                        y=ev_ref["y"], did=ev_ref["did"], sid=ev_ref["sid"],
                        tid=ev_ref["tid"], t_idx=ev_ref["t_idx"],
                        active=ev_ref["active"], **payload)


def _save_mu_sd(out, store: dict) -> None:
    path = os.path.join(out, "parts", "fit_mu_sd.npz")
    cur = dict(np.load(path)) if os.path.exists(path) else {}
    cur.update(store)
    np.savez_compressed(path, **cur)


def _save_warm_counts(out, b) -> None:
    path = os.path.join(out, "parts", "warm_counts.npz")
    cur = dict(np.load(path)) if os.path.exists(path) else {}
    for label in ("match", "orig"):
        c = b["warm_counts"][label]
        for k in ("n", "k", "n1", "n11", "n0", "n01"):
            cur[f"{label}_{k}"] = c[k]
        cur[f"{label}_cutoff"] = np.asarray(c["cutoff"])
        cur[f"{label}_rows"] = np.asarray(c["rows"])
    np.savez_compressed(path, **cur)


def _assert_eval(ev, b):
    if len(ev["y"]) != 311638 or int(ev["y"].sum()) != 32614:
        raise SystemExit(f"C3: eval rows {len(ev['y'])} / pos {int(ev['y'].sum())}")
    if int((ev["seen_tr"] > 0).sum()) != 0:
        raise SystemExit("C5: eval seen_tr nonzero")
    if abs(float(np.max(np.abs(ev["br"] - b["g"])))) >= 1e-12:
        raise SystemExit("C5: eval br != global rate")
    if len(np.unique(ev["did"])) != 5282:
        raise SystemExit("C3: eval dyads != 5282")
    actors = np.union1d(ev["sid"], ev["tid"])
    if len(actors) != 196 or actors.min() != 0 or actors.max() != 195:
        raise SystemExit("C4: actor coverage != {0..195}")
    if len(np.unique(ev["t_idx"])) != 59:
        raise SystemExit("C3: eval months != 59")
    if b["counts"]["tr"] != 5055328 or b["counts"]["n_dyads"] != 26434:
        raise SystemExit(f"C3: tr/n_dyads {b['counts']}")


def stage_fits_p1(cfg, out, pr, holdout_dids) -> None:
    log("fits_p1: building P1 bundle")
    b = build_bundle(cfg, "", holdout_dids, pr, keep_tr=True,
                     need_inn=False, need_ext_extra=True)
    log(f"fits_p1: bundle {b['counts']} in {b['load_seconds']}s")
    ev = b["ev"]
    _assert_eval(ev, b)
    fits, preds = _load_fits(out, "p1")
    if "did" in preds:
        if not np.array_equal(preds["did"], ev["did"]):
            raise SystemExit("fits_p1: eval row order changed on resume")
    for k in WARM_KEYS:
        p = np.asarray(b["p_warm"][k], dtype=np.float32)
        if k in preds:
            if arr_hash(preds[k]) != arr_hash(p):
                raise SystemExit(f"C6: warm vector moved: {k}")
        preds[k] = p
    _save_warm_counts(out, b)
    jdump_clean(os.path.join(out, "parts", "warm_counts_hash.json"), {
        "p_vectors": {k: arr_hash(preds[k]) for k in WARM_KEYS},
        "match": {k: arr_hash(v) for k, v in b["warm_counts"]["match"].items()
                  if isinstance(v, np.ndarray)},
        "orig": {k: arr_hash(v) for k, v in b["warm_counts"]["orig"].items()
                 if isinstance(v, np.ndarray)},
        "g": b["g"], "pg": b["pg"].tolist()})

    mu_sd = {}
    for fid in ("F2", "F3", "F4", "F5"):
        cols = FEATURE_KEYS[FIT_SPECS[fid][0]]
        if fid in fits and fid in preds:
            log(f"fits_p1: {fid} present, skip")
            continue
        r = fit_one(cfg, b["tr"], b["sel"], cols, fid)
        preds[fid] = np.asarray(predict_one(r["fit"], ev, cols,
                                            FIT_CT.get(fid, (1, 1))[1]),
                                dtype=np.float32)
        fits[fid] = r["meta"]
        mu_sd[f"{fid}_mu"] = r["fit"]["mu"]
        mu_sd[f"{fid}_sd"] = r["fit"]["sd"]
        log(f"fits_p1: {fid} LL={binary_metrics(ev['y'], preds[fid])['log_loss']:.6f} "
            f"({r['meta']['runtime_seconds']}s, iter={r['meta']['n_iter']})")
        del r
        gc.collect()
    if mu_sd:
        _save_mu_sd(out, mu_sd)
    _save_fits(out, fits, preds, "p1", ev)

    # A8 structural checks while the training rows are still resident
    sp = os.path.join(out, "parts", "a8_structural.json")
    if not os.path.exists(sp):
        log("fits_p1: A8 structural checks")
        jdump_clean(sp, a8_structural(b))

    # A6 extended window: tr (t<=238) + t=239 + val 240..274  (5,816,800 rows)
    if not all(f in fits and f in preds for f in ("F6", "F7")):
        log("fits_p1: building A6 extended window")
        parts = [b["tr"], b["tr_239"], b["val_ext"]]
        ext = {k: np.concatenate([p[k] for p in parts], axis=0)
               for k in ("X", "y", "did", "sid", "tid", "t_idx", "active")}
        ext["feature_names"] = list(b["tr"]["feature_names"])
        del parts
        b["tr"] = None
        b["tr_239"] = None
        b["val_ext"] = None
        gc.collect()
        if len(ext["y"]) != 5816800:
            raise SystemExit(f"A6 window rows {len(ext['y'])} != 5816800")
        if len(np.unique(ext["did"])) != 21152:
            raise SystemExit("A6 window dyads != 21152")
        stats_ext = features.fit_dyad_stats(ext, b["n_dyads"])
        features.apply_dyad_stats(ext, stats_ext, True)
        jdump_clean(os.path.join(out, "parts", "a6_window.json"), {
            "rows": int(len(ext["y"])), "pos": int(ext["y"].sum()),
            "dyads": int(len(np.unique(ext["did"]))),
            "stats_rows": int(stats_ext["n_stats_rows"]),
            "months": int(len(np.unique(ext["t_idx"]))),
            "assertion": "t<=274, holdout excluded; t=239 and t=251 kept, "
                         "t=275 dropped"})
        mu_sd = {}
        for fid in ("F6", "F7"):
            if fid in fits and fid in preds:
                continue
            cols = FEATURE_KEYS[FIT_SPECS[fid][0]]
            r = fit_one(cfg, ext, b["sel"], cols, fid)
            preds[fid] = np.asarray(predict_one(r["fit"], ev, cols,
                                                FIT_CT.get(fid, (1, 1))[1]),
                                    dtype=np.float32)
            fits[fid] = r["meta"]
            mu_sd[f"{fid}_mu"] = r["fit"]["mu"]
            mu_sd[f"{fid}_sd"] = r["fit"]["sd"]
            log(f"fits_p1: {fid} LL={binary_metrics(ev['y'], preds[fid])['log_loss']:.6f} "
                f"({r['meta']['runtime_seconds']}s, iter={r['meta']['n_iter']})")
            del r
            gc.collect()
        _save_mu_sd(out, mu_sd)
        del ext
        gc.collect()
        _save_fits(out, fits, preds, "p1", ev)
    del b
    gc.collect()


def stage_fits_h0(cfg, out, pr, holdout_dids) -> None:
    log("fits_h0: building H0 bundle")
    b = build_bundle(cfg, "_holdout", holdout_dids, pr, keep_tr=True)
    log(f"fits_h0: bundle {b['counts']} in {b['load_seconds']}s")
    ev = b["ev"]
    _assert_eval(ev, b)
    fits, preds = _load_fits(out, "h0")
    for k in WARM_KEYS:
        p = np.asarray(b["p_warm"][k], dtype=np.float32)
        if k in preds and arr_hash(preds[k]) != arr_hash(p):
            raise SystemExit(f"C6: warm vector moved across caches: {k}")
        preds[k] = p
    mu_sd = {}
    for fid in ("F1", "F8", "F9", "D1"):
        cols = FEATURE_KEYS[FIT_SPECS[fid][0]]
        if fid in fits and fid in preds:
            log(f"fits_h0: {fid} present, skip")
            continue
        r = fit_one(cfg, b["tr"], b["sel"], cols, fid)
        preds[fid] = np.asarray(predict_one(r["fit"], ev, cols,
                                            FIT_CT.get(fid, (1, 1))[1]),
                                dtype=np.float32)
        fits[fid] = r["meta"]
        mu_sd[f"{fid}_mu"] = r["fit"]["mu"]
        mu_sd[f"{fid}_sd"] = r["fit"]["sd"]
        log(f"fits_h0: {fid} LL={binary_metrics(ev['y'], preds[fid])['log_loss']:.6f} "
            f"({r['meta']['runtime_seconds']}s, iter={r['meta']['n_iter']})")
        del r
        gc.collect()
    if mu_sd:
        _save_mu_sd(out, mu_sd)
    _save_fits(out, fits, preds, "h0", ev)
    del b
    gc.collect()


def stage_fits(cfg, out, pr, holdout_dids) -> None:
    stage_fits_p1(cfg, out, pr, holdout_dids)
    stage_fits_h0(cfg, out, pr, holdout_dids)


# ---------------------------------------------------------------------------
# A8 structural checks (no fit) — §10.5
# ---------------------------------------------------------------------------
def a8_structural(b) -> dict:
    tr = b["tr"]
    names = tr["feature_names"]
    t200 = tr["t_idx"] == 200
    s = tr["sid"][t200].astype(np.int64)
    t = tr["tid"][t200].astype(np.int64)
    X200 = tr["X"][t200]
    Xfull = tr["X"]
    feats = ["src_ev", "g_deg_s", "g_wdeg_s", "g_deg12_s", "g_eig_s",
             "g_pr_s", "g_nra_s"]
    cells = {}
    for actor in np.unique(s):
        m = s == actor
        if len(np.unique(t[m])) >= 20:
            cells[int(actor)] = {f: X200[m, names.index(f)] for f in feats}
    pb = {}
    for f in feats:
        nd = [len(np.unique(v[f])) for v in cells.values()]
        pb[f] = {"cells": int(len(nd)),
                 "cells_gt1": int(sum(1 for x in nd if x > 1)),
                 "max_distinct": int(max(nd)) if nd else 0,
                 "median_distinct": float(np.median(nd)) if nd else 0.0,
                 "total_distinct_values": int(len(np.unique(X200[:, names.index(f)])))}
    rng = np.random.default_rng(0)
    n = len(tr["y"])
    sample = np.sort(rng.choice(n, size=min(4000, n), replace=False))
    A_ = 196
    key = (tr["sid"].astype(np.int64) * A_ + tr["tid"].astype(np.int64)) * 340 \
        + tr["t_idx"].astype(np.int64)
    order = np.argsort(key)
    ks = key[order]
    pairs = [("g_deg_s", "g_deg_t"), ("g_wdeg_s", "g_wdeg_t"),
             ("g_deg12_s", "g_deg12_t"), ("g_eig_s", "g_eig_t"),
             ("g_pr_s", "g_pr_t"), ("g_nra_s", "g_nra_t"),
             ("src_ev", "tgt_ev")]
    kr = (tr["tid"][sample].astype(np.int64) * A_
          + tr["sid"][sample].astype(np.int64)) * 340 + tr["t_idx"][sample]
    pos = np.clip(np.searchsorted(ks, kr), 0, len(ks) - 1)
    valid = ks[pos] == kr
    rev = {}
    for fs, ft in pairs:
        a_ = Xfull[sample, names.index(fs)].astype(np.float64)
        b_ = Xfull[order[pos], names.index(ft)].astype(np.float64)
        tol = 1e-5 * np.maximum(1.0, np.abs(a_))
        ok = (np.abs(a_ - b_) <= tol) & valid
        rev[f"{fs}={ft}"] = {
            "matched": int(valid.sum()), "pass": int(ok.sum()),
            "rate": round(float(ok.sum() / max(int(valid.sum()), 1)), 6)}
    nra = np.abs(X200[:, names.index("g_nra_s")].astype(np.float64)
                 - X200[:, names.index("g_nra_t")].astype(np.float64))
    return {
        "partner_blindness_t200": {
            "rows": int(t200.sum()), "actors_ge20_partners": int(len(cells)),
            "features": pb,
            "statement": "6 of 7 source-side features are actor-level; "
                         "g_nra_s is dyad-level"},
        "reversal_invariance": {
            "sample_rows": int(len(sample)), "features": rev,
            "statement": "7 of 7 pairs role-blind at the input level"},
        "g_nra_symmetry_t200": {
            "mean_abs_g_nra_s_minus_t": float(nra.mean()),
            "max_abs": float(nra.max()), "n_rows": int(len(nra))},
    }


# ---------------------------------------------------------------------------
# Metrics (§9.1)
# ---------------------------------------------------------------------------
def within_variants(y, p, did) -> dict:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    _, inv = np.unique(np.asarray(did), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    inv = inv[order]
    ys, ps = y[order], p[order]
    bounds = np.searchsorted(inv, np.arange(inv.max() + 2))
    aucs, weights = [], []
    for g in range(inv.max() + 1):
        lo, hi = bounds[g], bounds[g + 1]
        yy, pp = ys[lo:hi], ps[lo:hi]
        npos = int(yy.sum())
        nneg = len(yy) - npos
        if npos and nneg:
            aucs.append(roc_auc_score(yy, pp))
            weights.append(npos * nneg)
    aucs = np.asarray(aucs, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    pooled = float((aucs * weights).sum() / weights.sum()) if len(aucs) else None
    return {"within_auc_strict_mean": round(float(aucs.mean()), 6) if len(aucs) else None,
            "within_auc_strict_pooled": round(pooled, 6) if pooled is not None else None,
            "within_eligible_dyads": int(len(aucs)),
            "n_pairs": int(weights.sum())}


def point_metrics(y, p, did) -> dict:
    m = sprint_a.metrics_block(y, p)
    w = within_entity_auc(y, p, did)
    out = {"n": m["n"], "positives": m["positives"], "pos_rate": m["pos_rate"],
           "log_loss": m["log_loss"], "brier": m["brier"], "ece": m["ece"],
           "ece_equal_frequency": m["ece_equal_frequency"], "auc": m["auc"],
           "ap": m["ap"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
           "calibration_intercept": m["intercept"],
           "calibration_slope": m["slope"], "within_auc_p1": w["within_auc"],
           "between_auc_p1": w["between_auc"]}
    out.update(within_variants(y, p, did))
    return out


# ---------------------------------------------------------------------------
# Contrasts / BH (§9.2–9.4)
# ---------------------------------------------------------------------------
def contrast(y, p_base, p_model, did, sid, tid, reps=REPS, seed=SEED_BASE):
    r = inf.paired_multiway(y, p_base, p_model, did, sid, tid, reps=reps,
                            seed=seed)
    bm = binary_metrics(y, p_base)
    mm = binary_metrics(y, p_model)
    pt = r["point"]
    pt["base_auc"], pt["model_auc"] = bm["auc"], mm["auc"]
    pt["d_auc"] = None if bm["auc"] is None else mm["auc"] - bm["auc"]
    pt["base_ap"], pt["model_ap"] = bm["ap"], mm["ap"]
    pt["d_ap"] = None if bm["ap"] is None else mm["ap"] - bm["ap"]
    return r


def bh_families(p: dict) -> dict:
    fams = {
        "F1_primary": ["A2_dll", "A2_dece", "A2_dbrier"],
        "F2_key_secondary": ["A5_dll", "A5_dece", "A5_dbrier", "NESTED_dll"],
        "F3_sensitivity": ["A1_dll", "A3_dll", "A4_dll", "A6a_dll", "A6b_dll",
                           "A7a_dll", "A7b_dll", "A7c_dll", "A2rate_dll"],
        "F4_cache_effect": ["CACHE49_dll", "CACHE47_dll", "CACHE25_dll"],
    }
    out = {"families": {}, "pvals": dict(p)}
    for name, keys in fams.items():
        vals = [float(p[k]) for k in keys]
        r = inf.bh_fdr(vals, alpha=0.05)
        out["families"][name] = {
            "m": len(keys),
            "members": [{"test": k, "p": float(pp), "q": float(q),
                         "reject": bool(rj)}
                        for k, pp, q, rj in zip(keys, r["p"], r["q"], r["reject"])]}
    allr = inf.bh_fdr([float(p[k]) for k in sorted(p)], alpha=0.05)
    out["global_sensitivity"] = {
        "m": len(p), "alpha": 0.05,
        "members": [{"test": k, "p": float(pp), "q": float(q),
                     "reject": bool(rj)}
                    for k, pp, q, rj in zip(sorted(p), allr["p"], allr["q"],
                                            allr["reject"])]}
    return out


# ---------------------------------------------------------------------------
# Compare stage
# ---------------------------------------------------------------------------
def merge_preds(out: str) -> dict:
    parts = [np.load(os.path.join(out, "parts", f"preds_{s}.npz"))
             for s in ("p1", "h0")
             if os.path.exists(os.path.join(out, "parts", f"preds_{s}.npz"))]
    if len(parts) != 2:
        raise SystemExit("compare: both fits_p1 and fits_h0 predictions required")
    base = parts[0]
    out_d = {k: base[k] for k in META_KEYS}
    for p in parts:
        for k in META_KEYS:
            if not np.array_equal(base[k], p[k]):
                raise SystemExit(f"compare: eval rows differ across caches ({k})")
    for p in parts:
        for k in p.files:
            if k not in out_d:
                out_d[k] = np.asarray(p[k], dtype=np.float32)
    return out_d


def stage_compare(cfg, out) -> None:
    P = merge_preds(out)
    y = P["y"].astype(np.int64)
    did, sid, tid = P["did"], P["sid"], P["tid"]
    if len(y) != 311638 or int(y.sum()) != 32614:
        raise SystemExit("compare: eval rows changed")
    fits = jload(os.path.join(out, "parts", "arm_fits.json"))
    preds = {k: P[k].astype(np.float64) for k in P if k not in META_KEYS}
    for fid in FIT_ORDER:
        if fid not in preds:
            raise SystemExit(f"compare: missing predictions for {fid}")

    rows = []
    for key in FIT_ORDER + WARM_KEYS:
        m = point_metrics(y, preds[key], did)
        row = {"predictor": key,
               "kind": "fit" if key in FIT_ORDER else "warm",
               "features": fits[key]["n_features"] if key in fits else None,
               "cache": (("H0" if FIT_SPECS[key][1] == "_holdout" else "P1")
                         if key in FIT_SPECS else None),
               "C": fits[key]["C"] if key in fits else None,
               "T": fits[key]["T"] if key in fits else None,
               "arm": ARM_OF.get(key), "comparator": COMPARATOR_OF.get(key)}
        row.update(m)
        rows.append(row)
    arms_main = pd.DataFrame(rows)

    comps, crows = {}, []
    for name, model_key, base_key, note in CONTRASTS:
        r = contrast(y, preds[base_key], preds[model_key], did, sid, tid)
        comps[name] = r
        crows.append({
            "contrast": name, "model": model_key, "baseline": base_key,
            "note": note, "d_log_loss": r["point"]["d_log_loss"],
            "d_ece": r["point"]["d_ece"], "d_brier": r["point"]["d_brier"],
            "rel_log_loss_gain": r["point"]["rel_log_loss_gain"],
            "d_auc": r["point"]["d_auc"], "d_ap": r["point"]["d_ap"],
            "ll_ci_lo": r["multiway"]["ci_ll"][0],
            "ll_ci_hi": r["multiway"]["ci_ll"][1],
            "ll_se_multiway": r["multiway"]["se_ll"],
            "p_ll": r["multiway"]["p_ll"], "p_ece": r["multiway"]["p_ece"],
            "p_brier": r["multiway"]["p_brier"],
            "ci_ece": str(r["multiway"]["ci_ece"]),
            "ci_brier": str(r["multiway"]["ci_brier"]),
            "var_dyad": r["dims"]["dyad"]["var_ll"],
            "var_source": r["dims"]["source"]["var_ll"],
            "var_target": r["dims"]["target"]["var_ll"],
            "dyad_ci_lo": r["dims"]["dyad"]["ci_ll"][0],
            "dyad_ci_hi": r["dims"]["dyad"]["ci_ll"][1],
            "source_ci_lo": r["dims"]["source"]["ci_ll"][0],
            "source_ci_hi": r["dims"]["source"]["ci_ll"][1],
            "target_ci_lo": r["dims"]["target"]["ci_ll"][0],
            "target_ci_hi": r["dims"]["target"]["ci_ll"][1]})
    arms_contrasts = pd.DataFrame(crows)
    if len(arms_contrasts) != 14:
        raise SystemExit("C9: arms_contrasts must have 14 rows")

    mrows = []
    for fid in FIT_ORDER:
        for wk in WARM_KEYS:
            r = contrast(y, preds[wk], preds[fid], did, sid, tid, reps=200)
            mrows.append({"fit": fid, "comparator": wk,
                          "d_log_loss": r["point"]["d_log_loss"],
                          "rel_log_loss_gain": r["point"]["rel_log_loss_gain"],
                          "ci_lo": r["multiway"]["ci_ll"][0],
                          "ci_hi": r["multiway"]["ci_ll"][1],
                          "p_ll": r["multiway"]["p_ll"],
                          "d_ece": r["point"]["d_ece"],
                          "d_brier": r["point"]["d_brier"]})
    comparator_matrix = pd.DataFrame(mrows)

    seed_spread = []
    for s in PRIMARY_SEEDS:
        r = contrast(y, preds["warm_match_markov"], preds["F3"], did, sid,
                     tid, seed=s)
        seed_spread.append({"seed": s, "point": r["point"]["d_log_loss"],
                            "ci_ll": r["multiway"]["ci_ll"],
                            "p_ll": r["multiway"]["p_ll"]})

    auc = {}
    cp = os.path.join(out, "parts", "comparisons.json")
    prev = jload(cp) if os.path.exists(cp) else {}
    auc_key = sha256_bytes(("|".join([
        arr_hash(preds["F3"]), arr_hash(preds["F5"]),
        arr_hash(preds["warm_match_markov"]), str(REPS), str(SEED_BASE),
        str(len(y))])).encode())
    if (prev.get("auc_ap_meta", {}).get("key") == auc_key
            and "auc_ap" in prev):
        auc = prev["auc_ap"]
        log("compare: reusing stored AUC/AP multiway (same preds/seed/reps)")
    else:
        for name, base_key, model_key in (
                ("A2_auc_ap", "warm_match_markov", "F3"),
                ("A5_auc_ap", "warm_match_markov", "F5")):
            log(f"compare: paired AUC/AP multiway {name}")
            auc[name] = inf.paired_auc_ap_multiway(
                y, preds[base_key], preds[model_key], did, sid, tid,
                reps=REPS, seed=SEED_BASE)
            log(f"compare: {name} dAUC={auc[name]['point']['d_auc']:.6f} "
                f"dAP={auc[name]['point']['d_ap']:.6f}")

    mde = {}
    for name, key in (("A2", "A2_F3_vs_warm_match_markov"),
                      ("A5", "A5_F5_vs_warm_match_markov")):
        base_key = "warm_match_markov"
        model_key = "F3" if name == "A2" else "F5"
        m = inf.mde_pair(y, preds[base_key], preds[model_key], did, label=name)
        se = comps[key]["multiway"]["se_ll"]
        m["mde_multiway"] = 2.801585 * se
        m["mde_multiway_rel"] = m["mde_multiway"] / max(m["base_log_loss"], 1e-12)
        mde[name] = m

    pv = {
        "A2_dll": comps["A2_F3_vs_warm_match_markov"]["multiway"]["p_ll"],
        "A2_dece": comps["A2_F3_vs_warm_match_markov"]["multiway"]["p_ece"],
        "A2_dbrier": comps["A2_F3_vs_warm_match_markov"]["multiway"]["p_brier"],
        "A5_dll": comps["A5_F5_vs_warm_match_markov"]["multiway"]["p_ll"],
        "A5_dece": comps["A5_F5_vs_warm_match_markov"]["multiway"]["p_ece"],
        "A5_dbrier": comps["A5_F5_vs_warm_match_markov"]["multiway"]["p_brier"],
        "NESTED_dll": comps["NESTED_F5_base_vs_F3_model"]["multiway"]["p_ll"],
        "A1_dll": comps["A1_F2_vs_warm_match_markov"]["multiway"]["p_ll"],
        "A3_dll": comps["A3_F3_vs_warm_orig_markov"]["multiway"]["p_ll"],
        "A4_dll": comps["A4_F4_vs_warm_match_markov"]["multiway"]["p_ll"],
        "A6a_dll": comps["A6a_F6_vs_warm_orig_markov"]["multiway"]["p_ll"],
        "A6b_dll": comps["A6b_F7_vs_warm_orig_markov"]["multiway"]["p_ll"],
        "A7a_dll": comps["A7a_F1_vs_warm_match_markov"]["multiway"]["p_ll"],
        "A7b_dll": comps["A7b_F8_vs_warm_match_markov"]["multiway"]["p_ll"],
        "A7c_dll": comps["A7c_F9_vs_warm_match_markov"]["multiway"]["p_ll"],
        "A2rate_dll": comps["A2r_F3_vs_warm_match_rate"]["multiway"]["p_ll"],
    }
    cache = {}
    for feats, h0, p1 in (("49", "F1", "F2"), ("47", "F8", "F3"),
                          ("25", "F9", "F5")):
        r = contrast(y, preds[h0], preds[p1], did, sid, tid)
        cache[feats] = {"p1": p1, "h0": h0,
                        "d_ll_h0_minus_p1": r["point"]["d_log_loss"],
                        "ci_ll": r["multiway"]["ci_ll"],
                        "p_ll": r["multiway"]["p_ll"],
                        "ll_p1": r["point"]["model_log_loss"],
                        "ll_h0": r["point"]["base_log_loss"]}
        pv[f"CACHE{feats}_dll"] = r["multiway"]["p_ll"]
    bh = bh_families(pv)

    W = np.load(os.path.join(out, "parts", "warm_counts.npz"))
    ws = []
    act = P["active"].astype(np.int64)
    d = did.astype(np.int64)
    for label in ("match", "orig"):
        n = W[f"{label}_n"]; n1 = W[f"{label}_n1"]; n0 = W[f"{label}_n0"]
        ws.append({"cutoff": int(W[f"{label}_cutoff"]),
                   "eval_rows": int(len(d)),
                   "rows_n0": int((n[d] == 0).sum()),
                   "rows_active_n1_0": int(((act == 1) & (n1[d] == 0)).sum()),
                   "rows_inactive_n0_0": int(((act == 0) & (n0[d] == 0)).sum()),
                   "markov_at_global": int((((act == 1) & (n1[d] == 0)) |
                                            ((act == 0) & (n0[d] == 0))).sum())})
    warm_support = pd.DataFrame(ws)

    within = pd.DataFrame([{"predictor": k, **within_variants(y, preds[k], did)}
                           for k in FIT_ORDER + WARM_KEYS])

    a7rows = []
    for feats in ("49", "47", "25"):
        for cache_name, idx in (("P1", 0), ("H0", 1)):
            fid = {"49": ("F2", "F1"), "47": ("F3", "F8"),
                   "25": ("F5", "F9")}[feats][idx]
            mo = contrast(y, preds["warm_orig_markov"], preds[fid], did, sid,
                          tid, reps=200)
            mm = contrast(y, preds["warm_match_markov"], preds[fid], did, sid,
                          tid, reps=200)
            a7rows.append({"features": feats, "cache": cache_name, "fit": fid,
                           "ll_model": mo["point"]["model_log_loss"],
                           "ll_warm_orig_markov": mo["point"]["base_log_loss"],
                           "dll_vs_orig": mo["point"]["d_log_loss"],
                           "ll_warm_match_markov": mm["point"]["base_log_loss"],
                           "dll_vs_match": mm["point"]["d_log_loss"]})
    a7 = pd.DataFrame(a7rows)

    rec = []

    def add_rec(arm, quantity, observed, expected, tol=1e-6):
        delta = None if expected is None else float(observed) - float(expected)
        if expected is None:
            status = "INFO"
        elif abs(delta) <= tol:
            status = "PASS"
        elif abs(delta) <= 1e-4:
            status = "PASS_WITH_NOTE"
        else:
            status = "STOP"
        rec.append({"arm": arm, "quantity": quantity, "observed": observed,
                    "expected": expected, "delta": delta, "status": status})

    ll = {r.predictor: r.log_loss for r in arms_main.itertuples()}
    add_rec("A0", "F1 log_loss", ll["F1"], 0.217875)
    add_rec("A0", "warm_orig_markov log_loss", ll["warm_orig_markov"], 0.209729)
    add_rec("A7", "F8 log_loss", ll["F8"], 0.193850)
    add_rec("A7", "F9 log_loss", ll["F9"], 0.197023)
    add_rec("A0", "warm_orig_rate log_loss", ll["warm_orig_rate"], 0.214682)
    add_rec("D1", "D1 log_loss (C=0.3,T=1.1)", ll["D1"], 0.205219)
    # Frozen anchor: warm_orig_markov - F8 = +0.015879 (model gain).  The
    # protocol writes "F8 - 0.209729 = +0.015879"; the frozen number is the
    # gain 0.209729 - 0.193850, so the operands are ordered accordingly.
    add_rec("A7", "warm_orig_markov - F8 (gain)",
            ll["warm_orig_markov"] - ll["F8"], 0.015879)
    reconciliation = pd.DataFrame(rec)
    hard_stop = reconciliation["status"].eq("STOP").any()

    np.savez_compressed(os.path.join(out, "preds", "v2_preds.npz"), **P)
    jdump_clean(os.path.join(out, "parts", "comparisons.json"),
                {"contrasts": comps, "seed_spread": seed_spread,
                 "auc_ap": auc, "auc_ap_meta": {"key": auc_key,
                                                "reps": REPS,
                                                "seed": SEED_BASE},
                 "mde": mde, "bh": bh, "cache": cache})
    write_table(arms_main, os.path.join(out, "tables", "arms_main.csv"),
                "v2/arms_main/1")
    write_table(arms_contrasts, os.path.join(out, "tables", "arms_contrasts.csv"),
                "v2/arms_contrasts/1")
    write_table(comparator_matrix,
                os.path.join(out, "tables", "comparator_matrix.csv"),
                "v2/comparator_matrix/1")
    write_table(warm_support, os.path.join(out, "tables", "warm_support.csv"),
                "v2/warm_support/1")
    write_table(within, os.path.join(out, "tables", "within_dyad.csv"),
                "v2/within_dyad/1")
    write_table(a7, os.path.join(out, "tables", "a7_cache_matrix.csv"),
                "v2/a7_cache_matrix/1")
    write_table(reconciliation, os.path.join(out, "tables", "reconciliation.csv"),
                "v2/reconciliation/1")
    bhrows = []
    for fam, block in bh["families"].items():
        for m in block["members"]:
            bhrows.append({"family": fam, "test": m["test"], "p": m["p"],
                           "q": m["q"], "reject": m["reject"], "m": block["m"]})
    write_table(pd.DataFrame(bhrows),
                os.path.join(out, "tables", "bh_families.csv"),
                "v2/bh_families/1")
    mderows = [{"arm": k, **{kk: vv for kk, vv in v.items()
                             if not isinstance(vv, (list, dict))}}
               for k, v in mde.items()]
    write_table(pd.DataFrame(mderows), os.path.join(out, "tables", "mde.csv"),
                "v2/mde/1")
    arows = []
    for name, r in auc.items():
        arows.append({"contrast": name,
                      "base_auc": r["point"]["base_auc"],
                      "model_auc": r["point"]["model_auc"],
                      "d_auc": r["point"]["d_auc"],
                      "d_auc_ci_lo": r["multiway_auc"]["ci"][0],
                      "d_auc_ci_hi": r["multiway_auc"]["ci"][1],
                      "d_auc_p": r["multiway_auc"]["p"],
                      "base_ap": r["point"]["base_ap"],
                      "model_ap": r["point"]["model_ap"],
                      "d_ap": r["point"]["d_ap"],
                      "d_ap_ci_lo": r["multiway_ap"]["ci"][0],
                      "d_ap_ci_hi": r["multiway_ap"]["ci"][1],
                      "d_ap_p": r["multiway_ap"]["p"]})
    write_table(pd.DataFrame(arows), os.path.join(out, "tables", "auc_ap.csv"),
                "v2/auc_ap/1")
    if hard_stop:
        bad = reconciliation[reconciliation.status == "STOP"]
        raise SystemExit("C1 STOP: anchor mismatch\n" + bad.to_string())
    log("compare: tables written")


# ---------------------------------------------------------------------------
# A8 ownership (§10)
# ---------------------------------------------------------------------------
def _meta_fit(meta, mu, sd):
    return {"mu": np.asarray(mu, dtype=np.float64),
            "sd": np.asarray(sd, dtype=np.float64),
            "coef": np.asarray(meta["coef"], dtype=np.float64),
            "intercept": float(meta["intercept"])}


def contributions_for(out, fid, X, cols):
    fits = jload(os.path.join(out, "parts", "arm_fits.json"))
    MS = np.load(os.path.join(out, "parts", "fit_mu_sd.npz"))
    meta = fits[fid]
    ci = [meta["features"].index(c) for c in cols]
    fit = _meta_fit(meta, MS[f"{fid}_mu"], MS[f"{fid}_sd"])
    return A.contributions(fit, X[:, ci]), fit, ci


def pred_for(out, fid, X, cols):
    fits = jload(os.path.join(out, "parts", "arm_fits.json"))
    MS = np.load(os.path.join(out, "parts", "fit_mu_sd.npz"))
    meta = fits[fid]
    ci = [meta["features"].index(c) for c in cols]
    fit = _meta_fit(meta, MS[f"{fid}_mu"], MS[f"{fid}_sd"])
    z = A.standardized(fit, X[:, ci])
    return A.sigmoid(z @ fit["coef"] + fit["intercept"])


def class_shares(C, names):
    fb = own_class(names)
    mean_abs = np.abs(C).mean(axis=0)
    total = float(mean_abs.sum())
    sums = {b: 0.0 for b in CLASS_ORDER}
    for j, n in enumerate(names):
        sums[fb[n]] += float(mean_abs[j])
    shares = {b: (100.0 * sums[b] / total if total else float("nan"))
              for b in CLASS_ORDER}
    return {"class_sum": sums, "shares_pct": shares, "total": total,
            "n_rows": int(C.shape[0]), "n_features": len(names)}


def stage_a8(cfg, out, pr, holdout_dids) -> None:
    fits = jload(os.path.join(out, "parts", "arm_fits.json"))
    for fid in ("F2", "F3"):
        if fid not in fits:
            raise SystemExit(f"a8: fit {fid} missing; run fits first")
    b = build_bundle(cfg, "", holdout_dids, pr, keep_tr=False, need_inn=True)
    ev, inn = b["ev"], b["inn"]

    outj = {}
    for fid, period, d in (("F2", "ev", ev), ("F2", "inn", inn),
                           ("F3", "ev", ev)):
        cols = fits[fid]["features"]
        C, _, _ = contributions_for(out, fid, d["X"], cols)
        outj[f"{fid}@{period}"] = class_shares(C, cols)
        del C
        gc.collect()

    names2 = list(fits["F2"]["features"])
    C, fit2, _ = contributions_for(out, "F2", inn["X"], names2)
    fb = own_class(names2)
    mean_abs = np.abs(C).mean(axis=0)
    total = float(mean_abs.sum())
    nra = {"g_nra_s", "g_nra_t"}
    summ = {b_: 0.0 for b_ in CLASS_ORDER}
    summ["actor_dyad_interaction"] = 0.0
    for j, n in enumerate(names2):
        if n in nra:
            summ["actor_dyad_interaction"] += float(mean_abs[j])
        else:
            summ[fb[n]] += float(mean_abs[j])
    shares = {k: 100.0 * v / total for k, v in summ.items()}
    nra_block = {"shares_pct": shares,
                 "actor_owned_pct": shares["source_actor"] + shares["target_actor"],
                 "not_actor_pct": 100.0 - shares["source_actor"] - shares["target_actor"],
                 "n_actor_owned_features": 12,
                 "v1_not_actor_target": V1_NOT_ACTOR_NRA,
                 "delta_vs_v1": (100.0 - shares["source_actor"]
                                 - shares["target_actor"] - V1_NOT_ACTOR_NRA)}
    s = class_shares(C, names2)["shares_pct"]
    base = s["dyad"] + s["calendar"] + s["train_stats"]
    alloc = {"a_graph_unowned_not_actor": 100.0 - s["source_actor"] - s["target_actor"],
             "b_graph_all_actor_not_actor": base,
             "c_graph_half_not_actor": base + 0.5 * s["graph"],
             "v1_triple": list(V1_ALLOC),
             "span_pct": (100.0 - s["source_actor"] - s["target_actor"]) - base}
    del C
    gc.collect()

    # exactness identity on 50,000 sampled eval rows (§10)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(ev["y"]), size=50000, replace=False))
    cols2 = fits["F2"]["features"]
    C50, fit50, _ = contributions_for(out, "F2", ev["X"][idx], cols2)
    p50 = pred_for(out, "F2", ev["X"][idx], cols2)
    ex = A.exactness_report(fit50, C50, p50)
    exactness = {"n_rows": int(ex["n_rows"]),
                 "raw_max_abs_error": ex["raw_max_abs_error"],
                 "raw_mean_abs_error": ex["raw_mean_abs_error"],
                 "rows_above_tol": ex["raw_rows_above_tol"],
                 "pass": bool(ex["raw_max_abs_error"] < 1e-5)}
    del C50
    gc.collect()

    structural = jload(os.path.join(out, "parts", "a8_structural.json"))
    dup_pairs = 5
    n_actor_owned = len(A.SOURCE_OWN) + len(A.TARGET_OWN)
    role_unique = sorted(set(A.SOURCE_OWN + A.TARGET_OWN)
                         - {"g_deg_s", "g_wdeg_s", "g_deg12_s", "g_eig_s",
                            "g_pr_s", "g_deg_t", "g_wdeg_t", "g_deg12_t",
                            "g_eig_t", "g_pr_t"})
    roles = {
        "n_actor_owned": {"value": n_actor_owned, "of": 49,
                          "definition": "SOURCE_OWN + TARGET_OWN columns"},
        "n_duplicate_graph_pairs": {
            "value": dup_pairs,
            "definition": "deg, wdeg, deg12, eig, pr: g_X_s(i,j,t)=g_X_t(j,i,t)"},
        "n_role_unique": {"value": len(role_unique), "of": 49,
                          "features": role_unique,
                          "definition": "actor-owned columns not in one of the "
                                        "five duplicate pairs"},
        "n_partner_blind": {
            "value": n_actor_owned - 2, "of": n_actor_owned,
            "definition": "actor-owned columns constant across partners at "
                          "fixed (actor, month); only g_nra_* varies"},
    }
    res = {"periods": outj,
           "v1_targets": {"shares_pct": V1_SHARES,
                          "shares_pct_protocol_literal": V1_SHARES_PROTOCOL_LITERAL,
                          "not_actor_pct": V1_NOT_ACTOR,
                          "order_note": "V1 16.96 is the graph class and 2.80 "
                                        "the calendar class; the protocol lists "
                                        "the six numbers without labelling that "
                                        "pair.  The V2 recomputation (graph "
                                        "18.17, calendar 3.23) identifies it."},
           "delta_vs_v1_F2inn": {b_: outj["F2@inn"]["shares_pct"][b_] - V1_SHARES[b_]
                                 for b_ in CLASS_ORDER},
           "train_rows_v1": 6344160, "train_rows_v2": 5055328,
           "train_row_delta": 5055328 - 6344160,
           "g_nra_reclassification_F2inn": nra_block,
           "allocation_range_F2inn": alloc,
           "role_counts": roles,
           "structural": structural,
           "exactness": exactness,
           "note": "V1 shares came from a 6,344,160-row Sprint-B fit that "
                   "included held-out rows; V2 does not reproduce them and "
                   "does not re-fit to do so (D-5)."}
    jdump_clean(os.path.join(out, "parts", "a8_ownership.json"), res)
    rows = []
    for per, cs in outj.items():
        for b_ in CLASS_ORDER:
            rows.append({"period": per, "class": b_,
                         "class_sum": cs["class_sum"][b_],
                         "share_pct": cs["shares_pct"][b_],
                         "n_rows": cs["n_rows"], "n_features": cs["n_features"]})
        rows.append({"period": per, "class": "TOTAL", "class_sum": cs["total"],
                     "share_pct": 100.0, "n_rows": cs["n_rows"],
                     "n_features": cs["n_features"]})
    write_table(pd.DataFrame(rows), os.path.join(out, "tables", "a8_ownership.csv"),
                "v2/a8_ownership/1")
    arows = [{"convention": k, "not_actor_pct": v}
             for k, v in (("a_graph_unowned", alloc["a_graph_unowned_not_actor"]),
                          ("b_graph_all_actor", alloc["b_graph_all_actor_not_actor"]),
                          ("c_graph_half", alloc["c_graph_half_not_actor"]))]
    arows += [{"convention": f"v1_{i}", "not_actor_pct": v}
              for i, v in enumerate(V1_ALLOC)]
    write_table(pd.DataFrame(arows),
                os.path.join(out, "tables", "a8_allocations.csv"),
                "v2/a8_allocations/1")
    srows = []
    for f, v in structural["partner_blindness_t200"]["features"].items():
        srows.append({"check": "partner_blindness", "feature": f,
                      **{k: v[k] for k in v}})
    for f, v in structural["reversal_invariance"]["features"].items():
        srows.append({"check": "reversal_invariance", "feature": f,
                      **{k: v[k] for k in v}})
    srows.append({"check": "g_nra_symmetry", "feature": "g_nra_s - g_nra_t",
                  "mean_abs": structural["g_nra_symmetry_t200"]
                  ["mean_abs_g_nra_s_minus_t"]})
    write_table(pd.DataFrame(srows),
                os.path.join(out, "tables", "a8_structural.csv"),
                "v2/a8_structural/1")
    del b
    gc.collect()
    log("a8: ownership written")


# ---------------------------------------------------------------------------
# A9 stability (§11)
# ---------------------------------------------------------------------------
def _pair_stats(stack: np.ndarray, ks=(10, 50)) -> dict:
    n = stack.shape[0]
    sp = []
    jac = {k: [] for k in ks}
    for i in range(n):
        for j in range(i + 1, n):
            sp.append(A.spearman(stack[i], stack[j]))
            for k in ks:
                jac[k].append(A.topk_jaccard(stack[i], stack[j], k, "abs"))
    sp = np.asarray([v for v in sp if np.isfinite(v)], dtype=np.float64)
    out = {"pairs": int(len(sp)),
           "spearman_median": float(np.median(sp)) if len(sp) else None,
           "spearman_min": float(np.min(sp)) if len(sp) else None,
           "spearman_max": float(np.max(sp)) if len(sp) else None}
    for k in ks:
        v = np.asarray([x for x in jac[k] if np.isfinite(x)], dtype=np.float64)
        out[f"jaccard@{k}_median"] = float(np.median(v)) if len(v) else None
        out[f"jaccard@{k}_min"] = float(np.min(v)) if len(v) else None
        out[f"jaccard@{k}_max"] = float(np.max(v)) if len(v) else None
    return out


def _bootstrap_weights(did, rep):
    uniq, inv = np.unique(did, return_inverse=True)
    if len(uniq) != 21152:
        raise SystemExit(f"A9: n_clusters {len(uniq)} != 21152")
    rng = np.random.default_rng(SEED_BASE + 1009 * rep)
    cnt = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)))
    if int(cnt.sum()) != 21152:
        raise SystemExit("A9: multinomial mass mismatch")
    return cnt[inv].astype(np.float64)


def _padded_acc(cols, C, X, ev, p, intercept):
    """Accumulator with br/seen_tr zero-padded so feature_buckets sees 49 cols."""
    names = list(cols)
    pad = [c for c in ("br", "seen_tr") if c not in names]
    if pad:
        names = names + pad
        Cpad = np.column_stack([C, np.zeros((C.shape[0], len(pad)))])
    else:
        Cpad = C
    acc = A.AttributionAccumulator(196, EVAL_START, EVAL_END, names)
    acc.add(X, ev["sid"], ev["tid"], ev["t_idx"], ev["y"], p, Cpad, intercept)
    return acc, Cpad


def _one_refit(cfg, cols, Xtr, ytr, Xsel, ysel, Xev, ev, inv, rep, intercept_pad):
    w = _bootstrap_weights_from_inv(inv, rep)
    t0 = time.time()
    fit = models.fit_logistic(Xtr, ytr, Xsel, ysel, cfg, sample_weight=w,
                              C_grid=[1.0], feature_names=cols)
    p = models.predict_logistic(fit, Xev)
    C = A.contributions(fit, Xev)
    acc, Cpad = _padded_acc(cols, C, Xev, ev, p, float(fit["intercept"]))
    out = {"seconds": time.time() - t0,
           "coef": np.asarray(fit["coef"], dtype=np.float64),
           "intercept": float(fit["intercept"])}
    for role in ("source", "target"):
        v = acc.actor_vectors(role)
        out[f"attr_{role}"] = v["attr"]
        out[f"feat_{role}"] = v["mean_feature_contrib"].astype(np.float32)
    fv = acc.feature_vector()
    fb = own_class(cols)
    total = float(fv["mean_abs_contrib"].sum())
    sh = {b_: 0.0 for b_ in A.BUCKETS}
    for j, n in enumerate(cols):
        sh[fb[n]] += float(fv["mean_abs_contrib"][j])
    out["not_actor_share"] = 100.0 * (1.0 - (sh["source_actor"]
                                             + sh["target_actor"]) / total)
    return out


def _bootstrap_weights_from_inv(inv, rep):
    k = int(inv.max()) + 1
    if k != 21152:
        raise SystemExit(f"A9: n_clusters {k} != 21152")
    rng = np.random.default_rng(SEED_BASE + 1009 * rep)
    cnt = rng.multinomial(k, np.full(k, 1.0 / k))
    if int(cnt.sum()) != 21152:
        raise SystemExit("A9: multinomial mass mismatch")
    return cnt[inv].astype(np.float64)


def stage_a9(cfg, out, pr, holdout_dids, n_refits_cli: int) -> None:
    fits = jload(os.path.join(out, "parts", "arm_fits.json"))
    MS = np.load(os.path.join(out, "parts", "fit_mu_sd.npz"))
    cache_res, big = {}, {}
    t_rep = None
    n_refits = int(n_refits_cli)
    for tag, label in (("", "P1"), ("_holdout", "H0")):
        fid = "F3" if label == "P1" else "F8"
        cols = fits[fid]["features"]
        log(f"a9: building {label} bundle")
        b = build_bundle(cfg, tag, holdout_dids, pr, keep_tr=True)
        ci_tr = _col_idx(b["tr"], cols)
        Xtr = np.ascontiguousarray(b["tr"]["X"][:, ci_tr])
        ytr = b["tr"]["y"]
        ci_sel = _col_idx(b["sel"], cols)
        Xsel = np.ascontiguousarray(b["sel"]["X"][:, ci_sel])
        ysel = b["sel"]["y"]
        ci_ev = _col_idx(b["ev"], cols)
        Xev = np.ascontiguousarray(b["ev"]["X"][:, ci_ev])
        ev = b["ev"]
        uniq, inv = np.unique(b["tr"]["did"], return_inverse=True)
        del b
        gc.collect()
        if not len(uniq) == 21152:
            raise SystemExit("A9: training dyads != 21152")

        reps = {}
        if t_rep is not None:
            n_refits = 20 if 100.0 * t_rep > 6 * 3600 else int(n_refits_cli)
        for rep in range(n_refits):
            r = _one_refit(cfg, cols, Xtr, ytr, Xsel, ysel, Xev, ev, inv, rep,
                           None)
            if t_rep is None:
                t_rep = float(r["seconds"])
                n_refits = 20 if 100.0 * t_rep > 6 * 3600 else int(n_refits_cli)
                log(f"a9: t_rep={t_rep:.1f}s -> n_refits={n_refits} "
                    f"(rule: N=20 iff 100*t_rep > 6h)")
            reps[rep] = r
            if rep == 0 or (rep + 1) % 5 == 0:
                log(f"a9: {label} rep {rep + 1}/{n_refits} ({r['seconds']:.1f}s)")
        del Xtr, ytr, Xsel, ysel
        gc.collect()

        # point attributions from the frozen (non-bootstrap) fit
        meta = fits[fid]
        mu, sd = np.asarray(MS[f"{fid}_mu"], float), np.asarray(MS[f"{fid}_sd"], float)
        fitp = _meta_fit(meta, mu, sd)
        p = A.sigmoid(A.standardized(fitp, Xev) @ fitp["coef"] + fitp["intercept"])
        C = A.contributions(fitp, Xev)
        acc, Cpad = _padded_acc(cols, C, Xev, ev, p,
                                float(meta["intercept"]))
        point = {role: acc.actor_vectors(role)["attr"]
                 for role in ("source", "target")}
        fv = acc.feature_vector()
        fb = own_class(cols)
        total = float(fv["mean_abs_contrib"].sum())
        sh = {b_: 0.0 for b_ in A.BUCKETS}
        for j, n in enumerate(cols):
            sh[fb[n]] += float(fv["mean_abs_contrib"][j])
        point["not_actor_share"] = 100.0 * (1.0 - (sh["source_actor"]
                                                   + sh["target_actor"]) / total)
        point["coef"] = np.asarray(meta["coef"], dtype=np.float64)
        del C, p
        gc.collect()

        block = {"label": label, "fit": fid, "n_refits": n_refits,
                 "point": point, "t_rep_seconds": t_rep}
        for role in ("source", "target"):
            stack = np.vstack([reps[r][f"attr_{role}"] for r in range(n_refits)])
            block[f"stability_{role}"] = _pair_stats(stack)
            block[f"replicate_sd_{role}_median"] = float(
                np.nanmedian(np.nanstd(stack, axis=0, ddof=1)))
        coefs = np.vstack([reps[r]["coef"] for r in range(n_refits)])
        flips = (np.sign(coefs) != np.sign(point["coef"])[None, :]).mean(axis=0)
        block["coef_sign_flip_rate_median"] = float(np.median(flips))
        block["coef_sign_flip_rate_max"] = float(np.max(flips))
        block["coef_sign_flip"] = {c: float(f) for c, f in zip(cols, flips)}
        shares = np.array([reps[r]["not_actor_share"] for r in range(n_refits)])
        block["not_actor_share"] = {"min": float(shares.min()),
                                    "median": float(np.median(shares)),
                                    "max": float(shares.max()),
                                    "point": float(point["not_actor_share"])}
        block["per_replicate_not_actor_share"] = shares.tolist()
        cache_res[label] = block
        for rep, r in reps.items():
            for role in ("source", "target"):
                big[f"{label}_attr_{role}_{rep}"] = np.asarray(r[f"attr_{role}"])
                big[f"{label}_feat_{role}_{rep}"] = np.asarray(r[f"feat_{role}"],
                                                               dtype=np.float32)
            big[f"{label}_coef_{rep}"] = np.asarray(r["coef"], dtype=np.float64)
        del reps, Xev, ev
        gc.collect()

    p1, h0 = cache_res["P1"]["point"], cache_res["H0"]["point"]
    cross = {"spearman_source": A.spearman(p1["source"], h0["source"]),
             "spearman_target": A.spearman(p1["target"], h0["target"]),
             "jaccard10_source": A.topk_jaccard(p1["source"], h0["source"], 10),
             "jaccard10_target": A.topk_jaccard(p1["target"], h0["target"], 10),
             "coef_sign_flips": int((np.sign(p1["coef"]) != np.sign(h0["coef"])).sum()),
             "n_coef": int(len(p1["coef"])),
             "delta_not_actor_share": float(p1["not_actor_share"]
                                            - h0["not_actor_share"])}
    res = {"n_refits": n_refits, "t_rep_seconds": t_rep,
           "contingency_rule": "N=20 iff 100*t_rep > 6h (symmetric, D-6)",
           "caches": cache_res, "cross_cache": cross,
           "status": "provisional"}
    jdump_clean(os.path.join(out, "parts", "a9_stability.json"), res)
    np.savez_compressed(os.path.join(out, "parts", "a9_attributions.npz"), **big)

    rows = []
    for label, block in cache_res.items():
        for role in ("source", "target"):
            for k, v in block[f"stability_{role}"].items():
                rows.append({"cache": label, "role": role, "metric": k, "value": v})
            rows.append({"cache": label, "role": role,
                         "metric": "replicate_sd_median",
                         "value": block[f"replicate_sd_{role}_median"]})
        rows.append({"cache": label, "role": "-",
                     "metric": "coef_sign_flip_rate_median",
                     "value": block["coef_sign_flip_rate_median"]})
        for k, v in block["not_actor_share"].items():
            rows.append({"cache": label, "role": "-",
                         "metric": f"not_actor_share_{k}", "value": v})
    write_table(pd.DataFrame(rows), os.path.join(out, "tables", "a9_stability.csv"),
                "v2/a9_stability/1")
    write_table(pd.DataFrame([{"metric": k, "value": v}
                              for k, v in cross.items()]),
                os.path.join(out, "tables", "a9_crosscache.csv"),
                "v2/a9_crosscache/1")
    fliprows = [{"cache": label, "feature": f, "sign_flip_rate": v}
                for label, block in cache_res.items()
                for f, v in block["coef_sign_flip"].items()]
    write_table(pd.DataFrame(fliprows),
                os.path.join(out, "tables", "a9_coef_flip.csv"),
                "v2/a9_coef_flip/1")
    log(f"a9: done n_refits={n_refits} t_rep={t_rep:.1f}s")


# ---------------------------------------------------------------------------
# Report (§13)
# ---------------------------------------------------------------------------
def stage_report(cfg, out) -> None:
    parts = os.path.join(out, "parts")
    fits = jload(os.path.join(parts, "arm_fits.json"))
    comps = jload(os.path.join(parts, "comparisons.json"))
    selft = jload(os.path.join(parts, "selftest.json"))
    recon = pd.read_csv(os.path.join(out, "tables", "reconciliation.csv"))
    arms_main = pd.read_csv(os.path.join(out, "tables", "arms_main.csv"))
    contrasts = pd.read_csv(os.path.join(out, "tables", "arms_contrasts.csv"))
    a8 = jload(os.path.join(parts, "a8_ownership.json")) \
        if os.path.exists(os.path.join(parts, "a8_ownership.json")) else None
    a9 = jload(os.path.join(parts, "a9_stability.json")) \
        if os.path.exists(os.path.join(parts, "a9_stability.json")) else None
    a6 = jload(os.path.join(parts, "a6_window.json")) \
        if os.path.exists(os.path.join(parts, "a6_window.json")) else None
    ll = {r.predictor: r.log_loss for r in arms_main.itertuples()}
    # §12 claim rule, applied mechanically to the frozen thresholds
    a2c = comps["contrasts"]["A2_F3_vs_warm_match_markov"]
    nst = comps["contrasts"]["NESTED_F5_base_vs_F3_model"]
    q_a2 = [m["q"] for m in comps["bh"]["families"]["F1_primary"]["members"]
            if m["test"] == "A2_dll"][0]
    d_a2 = float(a2c["point"]["d_log_loss"])
    ci_a2 = [float(v) for v in a2c["multiway"]["ci_ll"]]
    ci_n = [float(v) for v in nst["multiway"]["ci_ll"]]
    cond = {"d_gt_0": bool(d_a2 > 0),
            "ci_excludes_0": bool(ci_a2[0] > 0 or ci_a2[1] < 0),
            "q_F1_le_0.05": bool(q_a2 <= 0.05),
            "nested_ci_excludes_0": bool(ci_n[0] > 0 or ci_n[1] < 0)}
    if all(cond.values()):
        verdict = ("support: the 47-feature contract beats matched persistence "
                   "and the nested 47-vs-25 contrast excludes 0; the supported "
                   "claim is bounded by the 22 added columns (20 graph + 2 ego)")
    elif cond["d_gt_0"] and cond["ci_excludes_0"] and cond["q_F1_le_0.05"] \
            and not cond["nested_ci_excludes_0"]:
        verdict = ("refutation of the actor-role reading: the 22 columns the "
                   "47-feature model adds over 25 add nothing identifiable")
    elif d_a2 < 0 and cond["ci_excludes_0"]:
        verdict = "refutation: artifact-free model loses to matched persistence"
    elif not cond["ci_excludes_0"]:
        verdict = "null: no identified difference vs matched-vintage persistence"
    else:
        verdict = "inconclusive"
    claim_rule = {"primary_contrast": "A2_F3_vs_warm_match_markov",
                  "d_log_loss": d_a2, "ci_ll": ci_a2,
                  "p_multiway": float(a2c["multiway"]["p_ll"]), "q_F1": q_a2,
                  "nested_d_log_loss": float(nst["point"]["d_log_loss"]),
                  "nested_ci_ll": ci_n,
                  "nested_p": float(nst["multiway"]["p_ll"]),
                  "conditions": cond, "verdict": verdict,
                  "A5_beats_A2": bool(nst["point"]["d_log_loss"] < 0
                                      and ci_n[1] < 0),
                  "note": "post hoc secondary on a burned holdout; the V1 "
                          "49/H0/original-warm result remains the "
                          "pre-registered primary"}
    ih = input_hashes(cfg)
    gm1, gm2 = jload(os.path.join(cfg["paths"]["cache_dir"], "graph_meta.json")), \
        jload(os.path.join(cfg["paths"]["cache_dir"], "graph_meta_holdout.json"))
    c2_ok = bool(gm1.get("rows_total") == gm2.get("rows_total") == 8987560
                 and gm1.get("actors") == gm2.get("actors") == 196
                 and gm1.get("months") == gm2.get("months") == 340
                 and gm1.get("order") == gm2.get("order")
                 and set(gm1.get("files", {})) == set(gm2.get("files", {})))
    # C10: rebuild the published point metrics from preds/v2_preds.npz
    Pv = merge_preds(out)
    pm3 = point_metrics(Pv["y"].astype(np.int64), Pv["F3"].astype(np.float64),
                        Pv["did"])
    row3 = arms_main[arms_main.predictor == "F3"].iloc[0]
    c10_ok = bool(abs(pm3["log_loss"] - float(row3["log_loss"])) < 1e-9
                  and abs(pm3["auc"] - float(row3["auc"])) < 1e-9
                  and abs(pm3["ece"] - float(row3["ece"])) < 1e-9)
    checks = {
        "C1_anchors": {"pass": bool(not recon["status"].eq("STOP").any()),
                       "detail": recon[["arm", "quantity", "observed",
                                        "expected", "delta", "status"]]
                       .to_dict("records")},
        "C2_cache_alignment": {
            "pass": c2_ok,
            "detail": {"rows_total": gm1.get("rows_total"),
                       "actors": gm1.get("actors"),
                       "months": gm1.get("months"),
                       "order_equal": gm1.get("order") == gm2.get("order"),
                       "note": "export asserts (t_idx,did) sort; graph file "
                               "hashes verified against each graph_meta"}},
        "C3_split_arithmetic": {
            "pass": bool(int(arms_main["n"].iloc[0]) == 311638
                         and int(arms_main["positives"].iloc[0]) == 32614),
            "detail": {"expected": [5055328, 232672, 311638, 32614, 5282, 5816800],
                       "tr": 5055328, "sel": 232672,
                       "ev": int(arms_main["n"].iloc[0]),
                       "ev_pos": int(arms_main["positives"].iloc[0]),
                       "ev_dyads": 5282, "ext": 5816800}},
        "C4_actor_coverage": {"pass": True,
                              "detail": "unique(sid|tid) over ev = {0..195}"},
        "C5_no_leakage": {"pass": bool(selft["checks"]["pass"]),
                          "detail": "self-test + eval seen_tr==0 + br==global"},
        "C6_invariance": {"pass": True,
                          "detail": "warm vectors hashed once; asserted equal "
                                    "in fits_h0"},
        "C7_estimator": {
            "pass": all(f["C"] == (0.3 if k == "D1" else 1.0)
                        and f["T"] == (1.1 if k == "D1" else 1.0)
                        and f["converged"] for k, f in fits.items()),
            "detail": {k: {"C": f["C"], "T": f["T"], "n_iter": f["n_iter"],
                           "converged": f["converged"]} for k, f in fits.items()}},
        "C8_exactness": {"pass": bool(a8 and a8["exactness"]["pass"]),
                         "detail": (a8 or {}).get("exactness")},
        "C9_completeness": {
            "pass": bool(len(arms_main) == 14 and len(contrasts) == 14
                         and a8 is not None and a9 is not None),
            "detail": {"arms_main_rows": int(len(arms_main)),
                       "contrast_rows": int(len(contrasts)),
                       "suppressed": [], "bh_tests": 19}},
        "C10_reproducibility": {
            "pass": c10_ok,
            "detail": {"rebuilt_from": "preds/v2_preds.npz",
                       "fit_checked": "F3",
                       "log_loss": pm3["log_loss"], "auc": pm3["auc"],
                       "ece": pm3["ece"],
                       "table_values": {"log_loss": float(row3["log_loss"]),
                                        "auc": float(row3["auc"]),
                                        "ece": float(row3["ece"])}}},
        "C11_no_overwrite": {"pass": True,
                             "detail": "summary.json blocks a rerun (exit 2)"},
    }
    total_bytes = sum(os.path.getsize(os.path.join(dp, f))
                      for dp, _, fs in os.walk(out) for f in fs)
    checks["C12_disk"] = {"pass": bool(total_bytes < 500 * 1024 * 1024),
                          "detail": {"bytes": int(total_bytes)}}

    runtime = jload(os.path.join(parts, "runtimes.json")) \
        if os.path.exists(os.path.join(parts, "runtimes.json")) else {}
    summary = {
        "schema": "q2-v2/1", "status": "completed",
        "protocol_sha256": PROTOCOL_HASH,
        "protocol_status": "FROZEN_BEFORE_V2_FIT",
        "code_sha256": sha256_file(os.path.abspath(__file__)),
        "post_hoc": True,
        "test_access": {"post_hoc": True,
                        "n_rows": int(arms_main["n"].iloc[0]),
                        "period": "2018-01..2022-11 (burned, unit-fresh)"},
        "checks": checks,
        "selftest": selft["checks"],
        "row_counts": {"ev": int(arms_main["n"].iloc[0]), "tr": 5055328,
                       "sel": 232672, "ext": 5816800, "inn": 607982},
        "fits": {k: {kk: vv for kk, vv in v.items() if kk != "coef"}
                 for k, v in fits.items()},
        "arm_results": {"log_loss": ll,
                        "contrasts": {r["contrast"]: r for r in
                                      contrasts.to_dict("records")},
                        "seed_spread": comps.get("seed_spread"),
                        "auc_ap": comps.get("auc_ap"),
                        "mde": comps.get("mde"), "bh": comps.get("bh"),
                        "cache_effect": comps.get("cache")},
        "claim_rule": claim_rule,
        "a6_window": a6, "a8": a8, "a9": a9,
        "reconciliation": recon.to_dict("records"),
        "suppressed": [],
        "unresolved": [
            "V2 is post hoc on a burned holdout; it cannot be confirmatory.",
            "A8 shares differ from V1 by construction (5.06M vs 6.34M train rows).",
            "A9 is provisional and cannot upgrade or refute the primary."],
        "runtime": runtime}
    manifest = {
        "schema": "q2-v2-manifest/1",
        "protocol_sha256": PROTOCOL_HASH,
        "protocol_status": "FROZEN_BEFORE_V2_FIT",
        "code_path": _rel(os.path.abspath(__file__)),
        "code_sha256": sha256_file(os.path.abspath(__file__)),
        "host": host_info(), "packages": pkg_versions(),
        "inputs": ih,
        "row_counts": summary["row_counts"],
        "n_fits": len(fits), "n_contrasts": len(contrasts),
        "arms": {k: {"log_loss": ll[k]} for k in ll},
        "arm_results": {"contrasts": {r["contrast"]: {
            "d_log_loss": r["d_log_loss"], "ll_ci": [r["ll_ci_lo"], r["ll_ci_hi"]],
            "p_ll": r["p_ll"], "d_ece": r["d_ece"], "d_brier": r["d_brier"]}
            for r in contrasts.to_dict("records")},
            "bh_families": comps.get("bh", {}).get("families"),
            "seed_spread": comps.get("seed_spread"),
            "a9_n_refits": (a9 or {}).get("n_refits"),
            "a9_t_rep_seconds": (a9 or {}).get("t_rep_seconds")},
        "checks": {k: v["pass"] for k, v in checks.items()},
        "runtime": runtime, "unresolved": summary["unresolved"],
        "DRY_RUN": bool(os.path.commonpath([out, SCRATCH_DEFAULT]) == SCRATCH_DEFAULT)}
    jdump_clean(os.path.join(out, "summary.json"), summary)
    jdump_clean(os.path.join(out, "manifest.json"), manifest)

    once = os.path.join(out, "once", "run_once_v2.json")
    doc = jload(once) if os.path.exists(once) else {"attempts": 1}
    doc.update({"status": "completed", "completed_utc": sprint_a.utc_now()})
    jdump_clean(once, doc)
    log("report: summary, manifest and SUMS written")


# ---------------------------------------------------------------------------
# Finalisation: SUMS after the log is closed (protocol §13.1)
# ---------------------------------------------------------------------------
def finalize(out: str) -> None:
    rt = jload(os.path.join(out, "parts", "runtimes.json")) \
        if os.path.exists(os.path.join(out, "parts", "runtimes.json")) else {}
    mp = os.path.join(out, "manifest.json")
    if os.path.exists(mp):
        man = jload(mp)
        man["runtime"] = rt
        man["total_seconds"] = round(float(sum(rt.values())), 1)
        jdump_clean(mp, man)
    sums = {}
    for dp, _, fs in os.walk(out):
        for f in sorted(fs):
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, out)
            if rel == "SUMS.sha256":
                continue
            sums[rel] = sha256_file(p)
    with open(os.path.join(out, "SUMS.sha256"), "w", encoding="utf-8") as fh:
        for rel in sorted(sums):
            fh.write(f"{sums[rel]}  {rel}\n")
    print(f"[finalize] SUMS.sha256 over {len(sums)} files", flush=True)


# ---------------------------------------------------------------------------
# Stage driver
# ---------------------------------------------------------------------------
STAGES = {"selftest", "fits", "fits_p1", "fits_h0", "compare", "a8", "a9",
          "report", "all"}
STAGE_ORDER = ["selftest", "fits_p1", "fits_h0", "compare", "a8", "a9", "report"]


def _record_runtime(out, stage, seconds):
    p = os.path.join(out, "parts", "runtimes.json")
    doc = jload(p) if os.path.exists(p) else {}
    doc[stage] = round(float(seconds), 1)
    jdump_clean(p, doc)


def run_stage(stage, out, n_refits):
    t0 = time.time()
    ensure_dirs(out)
    verify_protocol()
    pr = load_prereg()
    holdout_dids = np.asarray(pr["definition"]["dids"], dtype=np.int64)
    cfg = make_cfg(out)
    if stage == "selftest":
        selftest(cfg, holdout_dids, out)
    elif stage == "fits_p1":
        stage_fits_p1(cfg, out, pr, holdout_dids)
    elif stage == "fits_h0":
        stage_fits_h0(cfg, out, pr, holdout_dids)
    elif stage == "fits":
        stage_fits(cfg, out, pr, holdout_dids)
    elif stage == "compare":
        stage_compare(cfg, out)
    elif stage == "a8":
        stage_a8(cfg, out, pr, holdout_dids)
    elif stage == "a9":
        stage_a9(cfg, out, pr, holdout_dids, n_refits)
    elif stage == "report":
        stage_report(cfg, out)
    else:
        raise SystemExit(f"unknown stage {stage}")
    _record_runtime(out, stage, time.time() - t0)
    log(f"stage {stage}: {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=sorted(STAGES))
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-refits", type=int, default=50)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    if args.dry_run and os.path.commonpath([out, REPO]) == REPO:
        out = os.path.join(SCRATCH_DEFAULT, "dry_run")
        sys.stderr.write(f"dry run forced to {out}\n")
    check(out)
    ensure_dirs(out)
    ph = verify_protocol()

    if args.stage == "all":
        runlog = os.path.join(out, "run.log")
        with open(runlog, "w", encoding="utf-8") as fh:
            fh.write(ph + "\n")
        once = os.path.join(out, "once", "run_once_v2.json")
        doc = jload(once) if os.path.exists(once) else {"attempts": 0}
        doc["attempts"] = int(doc.get("attempts", 0)) + 1
        doc["status"] = "started"
        doc["started_utc"] = sprint_a.utc_now()
        jdump_clean(once, doc)
        for s in STAGE_ORDER:
            with open(runlog, "a", encoding="utf-8") as fh:
                fh.write(f"\n=== stage {s} ===\n")
            log(f"=== stage {s} ===")
            cmd = [sys.executable, os.path.abspath(__file__), "--stage", s,
                   "--out", out, "--n-refits", str(args.n_refits)]
            subprocess.run(cmd, check=True, cwd=REPO)
        return

    from framework.utils import Tee
    runlog = os.path.join(out, "run.log")
    if not os.path.exists(runlog):
        with open(runlog, "w", encoding="utf-8") as fh:
            fh.write(ph + "\n")
    tee = Tee(runlog)
    sys.stdout = tee
    sys.stderr = tee
    try:
        run_stage(args.stage, out, args.n_refits)
    finally:
        sys.stdout = tee.stdout
        sys.stderr = tee.stderr
        tee.close()
        if args.stage == "report":
            finalize(out)


if __name__ == "__main__":
    main()
