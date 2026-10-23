"""Shared plumbing for Sprint B (exact actor explanations + stability audit).

CPU-only.  No GPU, no PyG/DGL, no new data, no synthetic/T1 reruns.  The
functions here only load the frozen framework caches, fit the single
interpretable logistic model (plus dyad-cluster resample replicates) and stream
exact contributions through :class:`framework.aggregation.AttributionAccumulator`.

Frozen protocol used by every Sprint B script
---------------------------------------------
train     t < 240   (1995-01..2014-11), t=239 dropped (label crosses 2015-01)
select    t 240..250 (2015-01..2015-11), t=251 dropped (label crosses 2016-01)
inner     t 252..274 (2016-01..2017-11), t=275 dropped (label is 2018-01)
retro     t 276..334 (2018-01..2022-11), t=335 dropped (label is 2023-01)
covid     t 300..323 (2020-01..2021-12)

The embargo follows docs/15_FRAMEWORK_PILOT.md section 3.2 (239/251/275/335).
The P1 parts are *not* regenerated; this audit is therefore not row-for-row
comparable with the P1 tables (607,982 vs 634,416 inner rows).
"""

from __future__ import annotations

import json
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from framework import aggregation as A            # noqa: E402
from framework import features as F               # noqa: E402
from framework import models                      # noqa: E402
from framework.config import load_config           # noqa: E402
from framework.metrics import binary_metrics       # noqa: E402
from framework.utils import (Tee, elapsed, host_info, jdump, jload,  # noqa: E402
                             log, pkg_versions, sha256_file)

SPRINT = "sprint_b"
OUT = os.path.join(REPO, "results", "framework_pilot", "stability")
TAB = os.path.join(OUT, "tables")
AGG = os.path.join(OUT, "cache")
CACHE = os.path.join(REPO, "framework", "cache", SPRINT)

TRAIN_EMBARGO = 239
SELECT_EMBARGO = 251
PERIODS = [
    {"name": "inner_2016_17", "lo": 252, "hi": 274,
     "label": "2016-01..2017-11 (inner validation, embargoed)",
     "status": "inner_validation"},
    {"name": "retro_2018_22", "lo": 276, "hi": 334,
     "label": "2018-01..2022-11 (burned retrospective, secondary)",
     "status": "burned_retrospective"},
    {"name": "covid_2020_21", "lo": 300, "hi": 323,
     "label": "2020-01..2021-12 (COVID sub-window, secondary)",
     "status": "burned_retrospective"},
]
PERIOD_BY_NAME = {p["name"]: p for p in PERIODS}
N_SEEDS = 5
SEED_BASE = 20261023
CHUNK = 250_000


def ensure_dirs() -> None:
    for d in (OUT, TAB, AGG, CACHE):
        os.makedirs(d, exist_ok=True)


class RunLog:
    def __init__(self, name: str):
        ensure_dirs()
        self.path = os.path.join(OUT, "run.log")
        self.tee = Tee(self.path)
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self.tee
        sys.stderr = self.tee
        self.name = name
        log(f"=== {name} start ===")

    def close(self, status: str = "ok"):
        log(f"=== {self.name} {status} ===")
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self.tee.close()


def load_splits(cfg: dict) -> dict:
    t0 = time.time()
    con = duckdb.connect()
    con.execute(
        "CREATE OR REPLACE VIEW fb AS SELECT * FROM read_parquet('%s')" % os.path.join(
            cfg["paths"]["cache_dir"], "features_dyad.parquet").replace("\\", "/"))
    gm = {"features": F.GRAPH_CORE20}
    tr = F.export_dyad_split(con, cfg, "train", gm)
    va = F.export_dyad_split(con, cfg, "val", gm)
    te = F.export_dyad_split(con, cfg, "test", gm)
    panel = con.execute(
        "SELECT did, sid, tid, t_idx, active FROM read_parquet('%s') ORDER BY did, t_idx"
        % os.path.join(cfg["paths"]["cache_dir"], "panel_dyad.parquet").replace("\\", "/")
    ).df()
    actors = con.execute("SELECT * FROM read_parquet('%s') ORDER BY aid"
                         % os.path.join(cfg["paths"]["cache_dir"], "actors.parquet")
                         .replace("\\", "/")).df()
    con.close()
    n_dyads = int(max(tr["did"].max(), va["did"].max(), te["did"].max())) + 1
    exclude = tr["t_idx"] == TRAIN_EMBARGO
    stats = F.fit_dyad_stats(tr, n_dyads, exclude_mask=exclude)
    tr = F.apply_dyad_stats(tr, stats, True)
    va = F.apply_dyad_stats(va, stats, False)
    te = F.apply_dyad_stats(te, stats, False)
    assert len(actors) == int(max(te["sid"].max(), te["tid"].max())) + 1
    log(f"load: train={tr['X'].shape} val={va['X'].shape} test={te['X'].shape} "
        f"panel={panel.shape} actors={len(actors)} ({elapsed(t0)}s)")
    return {"tr": tr, "va": va, "te": te, "stats": stats, "actors": actors,
            "panel": panel, "n_dyads": n_dyads,
            "exclude_train": exclude}


def split_rows(d: dict, lo: int, hi: int) -> dict:
    m = (d["t_idx"] >= lo) & (d["t_idx"] <= hi)
    return {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def base_weight(tr: dict) -> np.ndarray:
    return (tr["t_idx"] != TRAIN_EMBARGO).astype(np.float64)


def select_rows(va: dict) -> dict:
    return split_rows(va, 240, 250)


def fit_point(cfg: dict, b: dict) -> dict:
    tr, sel = b["tr"], select_rows(b["va"])
    w = base_weight(tr)
    t0 = time.time()
    fit = models.fit_logistic(tr["X"], tr["y"], sel["X"], sel["y"], cfg,
                              sample_weight=w, feature_names=tr["feature_names"])
    p_sel = models.predict_logistic(fit, sel["X"])
    temp = models.fit_temperature(p_sel, sel["y"], p_sel)
    iso = models.fit_isotonic(p_sel, sel["y"], p_sel)
    out = {"fit": fit, "C": fit["C"], "T": float(temp["T"]),
           "iso": iso["iso"], "p_sel": p_sel,
           "select_log_loss": float(fit["select_log_loss"]),
           "temp_select_log_loss": float(temp["select_log_loss"]),
           "calibration_curve": {
               "C_grid": cfg["models"]["logistic"]["C_grid"],
               "select_curve": fit["select_curve"]},
           "runtime": elapsed(t0)}
    log(f"fit point: C={fit['C']} select_LL={fit['select_log_loss']:.6f} "
        f"T={out['T']:.3f} ({out['runtime']}s)")
    return out


def seed_weights(tr: dict, n_dyads: int, rep: int) -> np.ndarray:
    """Dyad-cluster bootstrap replicate weights (integer counts per dyad)."""
    rng = np.random.default_rng(SEED_BASE + 1009 * rep)
    cnt = rng.multinomial(n_dyads, np.full(n_dyads, 1.0 / n_dyads))
    w = cnt[tr["did"]].astype(np.float64)
    w[tr["t_idx"] == TRAIN_EMBARGO] = 0.0
    return w


def fit_seed(cfg: dict, b: dict, point: dict, rep: int) -> dict:
    tr, sel = b["tr"], select_rows(b["va"])
    t0 = time.time()
    fit = models.fit_logistic(tr["X"], tr["y"], sel["X"], sel["y"], cfg,
                              sample_weight=seed_weights(tr, b["n_dyads"], rep),
                              C_grid=[float(point["fit"]["C"])],
                              feature_names=tr["feature_names"])
    log(f"fit seed {rep}: C={fit['C']} select_LL={fit['select_log_loss']:.6f} "
        f"({elapsed(t0)}s)")
    return fit


def run_period(acc: A.AttributionAccumulator, d: dict, fit: dict, T: float,
               use_z: bool = False, store_C: np.ndarray | None = None,
               window_id: np.ndarray | None = None) -> None:
    """Stream one period through the accumulator, chunk by chunk."""
    n = len(d["y"])
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        sl = slice(start, stop)
        X = d["X"][sl]
        C = A.contributions(fit, X)
        if T != 1.0:
            C = A.contributions_temperature(C, T)
        p = models.predict_logistic(fit, X)
        if T != 1.0:
            p = models.apply_temperature(p, T)
        Z = A.standardized(fit, X) if use_z else None
        wid = None if window_id is None else window_id[sl]
        acc.add(X, d["sid"][sl], d["tid"][sl], d["t_idx"][sl], d["y"][sl],
                p, C, float(fit["intercept"]) / float(T),
                window_id=wid, Z=Z)
        if store_C is not None:
            store_C[sl] = C.astype(np.float32)


def save_agg(acc: A.AttributionAccumulator, path: str, meta: dict) -> None:
    payload = {"meta": np.array(json.dumps(meta))}
    for r in ("source", "target"):
        payload[f"bucket_sum_{r}"] = acc.bucket_sum[r].astype(np.float32)
        payload[f"count_{r}"] = acc.count[r].astype(np.float32)
        payload[f"p_sum_{r}"] = acc.p_sum[r].astype(np.float32)
        payload[f"y_sum_{r}"] = acc.y_sum[r].astype(np.float32)
        payload[f"feat_sum_{r}"] = acc.feat_sum[r].astype(np.float32)
        payload[f"feat_abs_{r}"] = acc.feat_abs[r].astype(np.float32)
        payload[f"feat_n_{r}"] = acc.feat_n[r].astype(np.float32)
        payload[f"z_sum_{r}"] = acc.z_sum[r].astype(np.float32)
    payload["feat_c_sum"] = acc.feat_c_sum.astype(np.float32)
    payload["feat_abs_c_sum"] = acc.feat_abs_c_sum.astype(np.float32)
    payload["feat_month_sum"] = acc.feat_month_sum.astype(np.float32)
    payload["feat_month_abs"] = acc.feat_month_abs.astype(np.float32)
    payload["month_n"] = acc.month_n.astype(np.float32)
    payload["win_sum_source"] = acc.win_sum["source"].astype(np.float32)
    payload["win_sum_target"] = acc.win_sum["target"].astype(np.float32)
    payload["win_count_source"] = acc.win_count["source"].astype(np.float32)
    payload["win_count_target"] = acc.win_count["target"].astype(np.float32)
    # Backward-compatible alias; old caches only had the source denominator.
    payload["win_count"] = acc.win_count["source"].astype(np.float32)
    if acc.do_occlusion:
        payload["occ_feat_month"] = acc.occ_feat_month.astype(np.float32)
        payload["occ_feat_total"] = acc.occ_feat_total.astype(np.float32)
        payload["occ_actor_month_source"] = acc.occ_actor_month["source"].astype(np.float32)
        payload["occ_actor_month_target"] = acc.occ_actor_month["target"].astype(np.float32)
        payload["occ_actor_total_source"] = acc.occ_actor_total["source"].astype(np.float32)
        payload["occ_actor_total_target"] = acc.occ_actor_total["target"].astype(np.float32)
    np.savez_compressed(path, **payload)


def load_agg(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    out["meta"] = json.loads(str(out["meta"]))
    return out


def atomic_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    jdump(tmp, obj)
    os.replace(tmp, path)


def input_hashes(cfg: dict) -> dict:
    cache = cfg["paths"]["cache_dir"]
    files = ["features_dyad.parquet", "panel_dyad.parquet", "actors.parquet",
             "graph_meta.json", "features_actor.parquet"]
    out = {}
    for f in files:
        p = os.path.join(cache, f)
        if os.path.exists(p):
            out[f] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    return out


def metrics_block(y, p) -> dict:
    m = binary_metrics(y, p)
    return {k: (round(float(v), 6) if isinstance(v, (int, float, np.floating))
                else v) for k, v in m.items()}


def period_frame(d: dict) -> pd.DataFrame:
    return pd.DataFrame({"did": d["did"], "sid": d["sid"], "tid": d["tid"],
                         "t_idx": d["t_idx"], "y": d["y"]})


def ensure_summary(name: str) -> dict:
    path = os.path.join(OUT, f"summary_{name}.json")
    return jload(path) if os.path.exists(path) else {}
