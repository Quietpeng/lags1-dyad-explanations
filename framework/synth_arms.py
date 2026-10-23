"""Synthetic-data arms for T1/T2.

The synthetic corpus is generated only from real **train** moments (see
``synth.estimate_moments``) and stored under ``framework/cache/synth_full`` with
feature tags ``_synthfull_*`` so the original smoke cache is untouched.

Arms
----
``A0_real``                 real train only (reference)
``A2_synth``                synthetic train only (transfer control)
``A3_real_synth``           real + base synthetic, synthetic total weight 0.3x real
``A5_random_events``        real + random-event control, same weight
``A6_shuffled_kernels``     real + shuffled-kernel control, same weight
``A7_real_volume_matched``  real only, subsampled to the synthetic row count
``A8_real_synth_equal``     real + base synthetic, synthetic total weight 1.0x real

``synth_weight`` is a **total-weight** ratio, not a per-row weight: synthetic
rows receive ``w * n_real / n_synth`` so that a value of 0.3 means synthetic
data contribute 23% of the effective training weight.  This avoids the common
mistake of reporting "weight 0.3" when synthetic rows are only 4% of the pool.

Hyper-parameter C is selected once on the real select window (2015) using a
1M-row subsample of real train, then held fixed for every arm.  Calibration is
fit on the real select window for every arm.  Evaluation is the strict 2016-17
window (t<275); the frozen test is never read.
"""

from __future__ import annotations

import os
import time

import duckdb
import numpy as np

from . import features as F
from . import models, synth
from .bootstrap import paired_gain
from .metrics import binary_metrics
from .utils import elapsed, jdump, log

SELECT_START, SELECT_END = 240, 252
EVAL_START, EVAL_END = 252, 275     # strict: t+1 < 276
SYNTH_SCHEMES = ["base", "random_events", "shuffled_kernels"]
TAG_PREFIX = "_synthfull"


def _p(path: str) -> str:
    return str(path).replace("\\", "/")


def _dyad_dict(df, mask, names) -> dict:
    return {
        "X": np.column_stack([df[c].astype(np.float32) for c in names])[mask],
        "y": df["active_next"].astype(np.int64)[mask],
        "did": df["did"].astype(np.int64)[mask],
        "t_idx": df["t_idx"].astype(np.int64)[mask],
        "feature_names": list(names),
    }


def load_real_dyad(cfg: dict) -> dict:
    """Real train/select/eval base-29 dyad matrices (train stats only)."""
    con = duckdb.connect()
    try:
        cols = ", ".join(F.DYAD_CAUSAL)
        df = con.execute(f"""
          SELECT {cols}, did, t_idx, split, active_next
          FROM read_parquet('{_p(os.path.join(cfg['paths']['cache_dir'], 'features_dyad.parquet'))}')
          WHERE split IN ('train','val')
          ORDER BY t_idx, did
        """).df()
    finally:
        con.close()
    names = list(F.DYAD_CAUSAL)
    tr_m = (df["split"] == "train").to_numpy()
    se_m = ((df["t_idx"] >= SELECT_START) & (df["t_idx"] < SELECT_END)).to_numpy()
    ev_m = ((df["t_idx"] >= EVAL_START) & (df["t_idx"] < EVAL_END)).to_numpy()
    n_dyads = int(df["did"].max()) + 1
    tr = _dyad_dict(df, tr_m, names)
    se = _dyad_dict(df, se_m, names)
    ev = _dyad_dict(df, ev_m, names)
    stats = F.fit_dyad_stats(tr, n_dyads)
    tr = F.apply_dyad_stats(tr, stats, True)
    se = F.apply_dyad_stats(se, stats, False)
    ev = F.apply_dyad_stats(ev, stats, False)
    return {"train": tr, "select": se, "eval": ev, "n_dyads": n_dyads,
            "stats": stats}


def load_real_actor(cfg: dict) -> dict:
    con = duckdb.connect()
    try:
        cols = ", ".join(F.ACTOR_FEATURES)
        df = con.execute(f"""
          SELECT {cols}, aid, t_idx, split, out_active_next
          FROM read_parquet('{_p(os.path.join(cfg['paths']['cache_dir'], 'features_actor.parquet'))}')
          WHERE split IN ('train','val') ORDER BY aid, t_idx
        """).df()
    finally:
        con.close()
    for c in F.ACTOR_FEATURES:
        if df[c].dtype == bool:
            df[c] = df[c].astype(np.int8)
    df[F.ACTOR_FEATURES] = df[F.ACTOR_FEATURES].fillna(0.0)
    names = list(F.ACTOR_FEATURES)
    tr_m = (df["split"] == "train").to_numpy()
    se_m = ((df["t_idx"] >= SELECT_START) & (df["t_idx"] < SELECT_END)).to_numpy()
    ev_m = ((df["t_idx"] >= EVAL_START) & (df["t_idx"] < EVAL_END)).to_numpy()

    def block(m):
        return {"X": np.column_stack([df[c].astype(np.float32) for c in names])[m],
                "y": df["out_active_next"].astype(np.int64)[m].to_numpy(),
                "aid": df["aid"].astype(np.int64)[m].to_numpy(),
                "t_idx": df["t_idx"].astype(np.int64)[m].to_numpy(),
                "feature_names": names}
    return {"train": block(tr_m), "select": block(se_m), "eval": block(ev_m)}


def load_synth_dyad(cfg: dict, scheme: str) -> dict:
    w = synth.load_world_features(cfg, scheme, 0, "train", tag_prefix=TAG_PREFIX)
    d = w["dyad"]
    n_dyads = int(d["did"].max()) + 1
    stats = F.fit_dyad_stats(d, n_dyads)
    d = F.apply_dyad_stats(d, stats, True)
    return d


def load_synth_actor(cfg: dict, scheme: str) -> dict:
    w = synth.load_world_features(cfg, scheme, 0, "train", tag_prefix=TAG_PREFIX)
    return w["actor"]


def _fit_arm(Xtr, ytr, wtr, tr_names, real, C, cfg):
    fit = models.fit_logistic(Xtr, ytr, real["select"]["X"], real["select"]["y"],
                              cfg, sample_weight=wtr, C_grid=[C],
                              feature_names=tr_names)
    p_sel = models.predict_logistic(fit, real["select"]["X"])
    p_ev = models.predict_logistic(fit, real["eval"]["X"])
    cal = models.calibrate_all(p_sel, real["select"]["y"], p_ev, cfg)
    return fit, cal


def _arm_block(name, cal, real, base_p, cid, reps, seed, extra) -> dict:
    from .metrics import icc_deff  # local to keep import surface small

    y = real["eval"]["y"]
    out = {"arm": name, **extra}
    for cname in ("none", "isotonic"):
        p = cal[cname]["p_eval"]
        m = binary_metrics(y, p)
        g = paired_gain(y, base_p, p, cid, reps=reps, seed=seed)
        out[cname] = {
            "metrics": m,
            "select_log_loss": binary_metrics(real["select"]["y"],
                                              cal[cname]["p_sel"])["log_loss"],
            "vs_A0": g,
        }
        out[cname]["n_eff_diff"] = icc_deff(
            -(y * np.log(np.clip(p, 1e-12, 1)) +
              (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))) +
            (y * np.log(np.clip(base_p, 1e-12, 1)) +
             (1 - y) * np.log(np.clip(1 - base_p, 1e-12, 1))), cid)
    return out


def _select_c(cfg, real, seed: int = 7) -> float:
    """C selected on real select with a 1M-row real-train subsample."""
    X, y = real["train"]["X"], real["train"]["y"]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(y), size=min(1_000_000, len(y)), replace=False))
    best = None
    for C in (0.3, 1.0):
        fit = models.fit_logistic(X[idx], y[idx], real["select"]["X"],
                                  real["select"]["y"], cfg, C_grid=[C],
                                  feature_names=real["train"]["feature_names"])
        ll = fit["select_log_loss"]
        log(f"syntharms: C={C} select_ll={ll:.5f}")
        if best is None or ll < best[0]:
            best = (ll, C)
    return float(best[1])


def run(cfg: dict, out_path: str | None = None) -> dict:
    t0 = time.time()
    reps = int(cfg["bootstrap"]["reps"]) // 2
    seed = int(cfg["bootstrap"]["seed"])

    out: dict = {"protocol": {
        "synthetic_train_only": True,
        "schemes": SYNTH_SCHEMES,
        "tag_prefix": TAG_PREFIX,
        "weight_semantics": "synth total weight = w * real total weight",
        "selection": "C on real select (1M real-train subsample), fixed for all arms",
        "calibration": "isotonic on real select for every arm",
        "eval": "strict 2016-17 (t<275); test never read",
    }}

    # ---------------- T2 ----------------
    real = load_real_dyad(cfg)
    C2 = _select_c(cfg, real)
    log(f"syntharms T2: n_real={len(real['train']['y'])} C*={C2}")
    base_fit, base_cal = _fit_arm(real["train"]["X"], real["train"]["y"],
                                  None, real["train"]["feature_names"], real, C2, cfg)
    base_p = base_cal["isotonic"]["p_eval"]
    cid2 = real["eval"]["did"]
    n_real = len(real["train"]["y"])
    t2 = {
        "n_real_train": n_real, "C_star": C2,
        "n_synth_train": {}, "arms": {}, "dose_response": [],
    }
    t2["arms"]["A0_real"] = _arm_block(
        "A0_real", base_cal, real, base_p, cid2, reps, seed,
        {"n_train": n_real, "synth_share": 0.0})

    synth_data = {s: load_synth_dyad(cfg, s) for s in SYNTH_SCHEMES}
    for s, d in synth_data.items():
        t2["n_synth_train"][s] = int(len(d["y"]))

    def run_aug(name, scheme, w_total, extra_note=""):
        s = synth_data[scheme]
        n_s = len(s["y"])
        sw = w_total * n_real / max(n_s, 1)
        X = np.vstack([real["train"]["X"], s["X"]])
        y = np.concatenate([real["train"]["y"], s["y"]])
        w = np.concatenate([np.ones(n_real, dtype=np.float64),
                            np.full(n_s, sw, dtype=np.float64)])
        fit, cal = _fit_arm(X, y, w, real["train"]["feature_names"], real, C2, cfg)
        share = (sw * n_s) / (n_real + sw * n_s)
        block = _arm_block(name, cal, real, base_p, cid2, reps, seed,
                           {"n_train": int(len(y)), "n_synth": n_s,
                            "synth_weight_total": w_total,
                            "synth_share": round(float(share), 4),
                            "scheme": scheme, "note": extra_note})
        t2["arms"][name] = block
        log(f"syntharms T2 {name}: LL_iso={block['isotonic']['metrics']['log_loss']:.5f} "
            f"Δ={block['isotonic']['vs_A0']['point']['d_log_loss']:.5f}")
        return block

    # synthetic only
    s = synth_data["base"]
    fit, cal = _fit_arm(s["X"], s["y"], None, s["feature_names"], real, C2, cfg)
    t2["arms"]["A2_synth"] = _arm_block(
        "A2_synth", cal, real, base_p, cid2, reps, seed,
        {"n_train": int(len(s["y"])), "n_synth": int(len(s["y"])),
         "synth_share": 1.0, "scheme": "base"})

    run_aug("A3_real_synth", "base", 0.3)
    run_aug("A5_random_events", "random_events", 0.3)
    run_aug("A6_shuffled_kernels", "shuffled_kernels", 0.3)
    run_aug("A8_real_synth_equal", "base", 1.0)

    # real-only volume control
    rng = np.random.default_rng(seed)
    n_syn = len(synth_data["base"]["y"])
    idx = np.sort(rng.choice(n_real, size=min(n_syn, n_real), replace=False))
    fit, cal = _fit_arm(real["train"]["X"][idx], real["train"]["y"][idx],
                        None, real["train"]["feature_names"], real, C2, cfg)
    t2["arms"]["A7_real_volume_matched"] = _arm_block(
        "A7_real_volume_matched", cal, real, base_p, cid2, reps, seed,
        {"n_train": int(len(idx)), "synth_share": 0.0,
         "note": "real-only subsample with the same row count as A2"})

    # dose-response on the base scheme
    for w_total in (0.1, 0.3, 0.5, 1.0):
        b = run_aug(f"dose_w{w_total}", "base", w_total)
        g = b["isotonic"]["vs_A0"]
        t2["dose_response"].append({
            "w_total": w_total, "synth_share": b["synth_share"],
            "d_log_loss": g["point"]["d_log_loss"],
            "d_log_loss_ci": g["d_log_loss_ci"],
            "d_ece": g["point"]["d_ece"], "d_ece_ci": g["d_ece_ci"],
            "log_loss": b["isotonic"]["metrics"]["log_loss"],
            "auc": b["isotonic"]["metrics"]["auc"],
        })
    out["t2"] = t2

    # ---------------- T1 (next_active, comparable across worlds) ----------------
    ra = load_real_actor(cfg)
    Ca = _select_c(cfg, ra)
    base_fit, base_cal = _fit_arm(ra["train"]["X"], ra["train"]["y"],
                                  None, ra["train"]["feature_names"], ra, Ca, cfg)
    base_pa = base_cal["isotonic"]["p_eval"]
    cida = ra["eval"]["aid"]
    n_ra = len(ra["train"]["y"])
    t1 = {"n_real_train": n_ra, "C_star": Ca, "arms": {}, "n_synth_train": {}}
    t1["arms"]["A0_real"] = _arm_block(
        "A0_real", base_cal, ra, base_pa, cida, reps, seed,
        {"n_train": n_ra, "synth_share": 0.0})
    sa = {s: load_synth_actor(cfg, s) for s in SYNTH_SCHEMES}
    for s, d in sa.items():
        t1["n_synth_train"][s] = int(len(d["y"]))

    def run_aug_actor(name, scheme, w_total):
        d = sa[scheme]
        n_s = len(d["y"])
        sw = w_total * n_ra / max(n_s, 1)
        X = np.vstack([ra["train"]["X"], d["X"]])
        y = np.concatenate([ra["train"]["y"], d["y"]])
        w = np.concatenate([np.ones(n_ra), np.full(n_s, sw)])
        fit, cal = _fit_arm(X, y, w, ra["train"]["feature_names"], ra, Ca, cfg)
        share = (sw * n_s) / (n_ra + sw * n_s)
        b = _arm_block(name, cal, ra, base_pa, cida, reps, seed,
                       {"n_train": int(len(y)), "n_synth": n_s,
                        "synth_weight_total": w_total,
                        "synth_share": round(float(share), 4),
                        "scheme": scheme})
        t1["arms"][name] = b
        log(f"syntharms T1 {name}: LL_iso={b['isotonic']['metrics']['log_loss']:.5f} "
            f"Δ={b['isotonic']['vs_A0']['point']['d_log_loss']:.5f}")

    d = sa["base"]
    fit, cal = _fit_arm(d["X"], d["y"], None, d["feature_names"], ra, Ca, cfg)
    t1["arms"]["A2_synth"] = _arm_block(
        "A2_synth", cal, ra, base_pa, cida, reps, seed,
        {"n_train": int(len(d["y"])), "synth_share": 1.0, "scheme": "base"})
    run_aug_actor("A3_real_synth", "base", 0.3)
    run_aug_actor("A5_random_events", "random_events", 0.3)
    run_aug_actor("A6_shuffled_kernels", "shuffled_kernels", 0.3)
    run_aug_actor("A8_real_synth_equal", "base", 1.0)
    out["t1"] = t1
    out["runtime_seconds"] = elapsed(t0)
    if out_path:
        jdump(out_path, out)
    return out
