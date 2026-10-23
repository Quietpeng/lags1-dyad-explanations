"""Pilot pipeline: panel -> features -> T2/T1 -> graph -> synthetic -> regimes -> cards.

Every stage writes a JSON part under ``results/framework_pilot/parts`` and, where
useful, prediction arrays under ``framework/cache/preds``.  The frozen test
split is never exported unless ``test_policy.allow_test`` is true; the pilot
uses validation only (select 2015, eval 2016-2017).
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from . import SCHEMA, data, features, graph, models, operators, synth
from .bootstrap import auc_ap_ci, dual_cluster_gain, paired_gain
from .config import ensure_dirs
from .metrics import binary_metrics, calibration_table, icc_deff, mde_block
from .splits import (actor_cold_start_impute, actor_rows, cold_start_impute,
                     d2_window_masks, regime_masks, temporal_masks)
from .utils import elapsed, jdump, jload, log, sha256_file

PARTS = "parts"


def _part_path(cfg: dict, name: str) -> str:
    return os.path.join(cfg["paths"]["out_dir"], PARTS, f"{name}.json")


def save_part(cfg: dict, name: str, obj) -> None:
    jdump(_part_path(cfg, name), obj)


def _slice(d: dict, mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v[mask]
        else:
            out[k] = v
    return out


def _cols(names: list[str], wanted: list[str]) -> list[int]:
    idx = {n: i for i, n in enumerate(names)}
    missing = [w for w in wanted if w not in idx]
    if missing:
        raise KeyError(f"features missing: {missing}")
    return [idx[w] for w in wanted]


def prepare(cfg: dict, force: bool = False, allow_test: bool = False) -> dict:
    t0 = time.time()
    meta = data.build_panels(cfg, force=force)
    ensure_dirs(cfg)
    con = data.connect(cfg)
    data.load_views(con, cfg)
    features.build_dyad_base(con, cfg, force=force)
    features.build_actor_base(con, cfg, force=force)
    gm = features.build_graph_features(con, cfg, force=force)
    tr = features.export_dyad_split(con, cfg, "train", gm)
    va = features.export_dyad_split(con, cfg, "val", gm)
    n_dyads = int(max(tr["did"].max(), va["did"].max())) + 1
    stats = features.fit_dyad_stats(tr, n_dyads)
    tr = features.apply_dyad_stats(tr, stats, True)
    va = features.apply_dyad_stats(va, stats, False)
    actor_tr = features.export_actor_split(con, "train")
    actor_va = features.export_actor_split(con, "val")
    actors_df = con.execute("SELECT * FROM actors ORDER BY aid").df()
    panel = features.export_panel_arrays(con)
    test = None
    if allow_test:
        test = features.export_dyad_split(con, cfg, "test", gm)
        test = features.apply_dyad_stats(test, stats, False)
    log(f"prepare: train={tr['X'].shape} val={va['X'].shape} "
        f"actor_train={actor_tr['X'].shape}")
    return {
        "cfg": cfg, "con": con, "panel_meta": meta, "graph_meta": gm,
        "train": tr, "val": va, "actor_train": actor_tr, "actor_val": actor_va,
        "actors": actors_df, "panel": panel, "n_dyads": n_dyads,
        "stats": stats, "test": test,
        "prepare_seconds": elapsed(t0),
    }


def _eval_block(name, y, p, did, sid, base_p=None, base_name=None,
                reps=500, seed=0, auc=True) -> dict:
    m = binary_metrics(y, p)
    cluster = icc_deff(-(y * np.log(np.clip(p, 1e-12, 1)) +
                        (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))), did)
    out = {"name": name, "metrics": m, "cluster_dyad": cluster,
           "deff": cluster, "mde": mde_block(
               -(y * np.log(np.clip(p, 1e-12, 1)) +
                 (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))), did, m["log_loss"])}
    if auc:
        out["auc_ap"] = auc_ap_ci(y, p, did, reps=min(200, reps), seed=seed)
    if base_p is not None:
        out["vs_" + (base_name or "base")] = dual_cluster_gain(
            y, base_p, p, sid, did, reps=reps, seed=seed)
    return out


# ---------------------------------------------------------------------------
# T2: dyad-month active next month
# ---------------------------------------------------------------------------
def run_t2(cfg: dict, b: dict) -> dict:
    t0 = time.time()
    tr, va = b["train"], b["val"]
    masks = temporal_masks(va["t_idx"], cfg)
    sel = _slice(va, masks["select"])
    ev = _slice(va, masks["eval"])
    names = tr["feature_names"]
    n_dyads = b["n_dyads"]
    reps = int(cfg["bootstrap"]["reps"])
    log(f"t2: train={tr['X'].shape} select={sel['X'].shape} eval={ev['X'].shape}")

    base = {
        "persistence": models.baseline_persistence(tr, ev),
        "global_transition": models.baseline_global_transition(tr, ev),
        "per_dyad_markov": models.baseline_per_dyad_markov(tr, ev, n_dyads),
    }
    base_blocks = {k: _eval_block(k, ev["y"], p, ev["did"], ev["sid"],
                                  reps=min(100, reps)) for k, p in base.items()}
    strongest = min(base_blocks, key=lambda k: base_blocks[k]["metrics"]["log_loss"])
    log(f"t2: strongest baseline {strongest} "
        f"LL={base_blocks[strongest]['metrics']['log_loss']:.5f}")

    tiers = {
        "logistic_t1": ["active", "log_ev"],
        "logistic_t2": ["active", "log_ev", "br", "seen_tr"],
        "logistic_full": names,
    }
    fits, model_blocks, preds = {}, {}, {}
    for name, cols in tiers.items():
        ci = _cols(names, cols)
        fit = models.fit_logistic(tr["X"][:, ci], tr["y"], sel["X"][:, ci],
                                  sel["y"], cfg, feature_names=cols)
        p_sel = models.predict_logistic(fit, sel["X"][:, ci])
        p_ev = models.predict_logistic(fit, ev["X"][:, ci])
        cal = models.calibrate_all(p_sel, sel["y"], p_ev, cfg)
        fits[name] = {"fit": fit, "cols": cols, "ci": ci, "cal": cal}
        for cname, c in cal.items():
            key = name if cname == "none" else f"{name}_{cname}"
            preds[key] = {"p_sel": c["p_sel"], "p_eval": c["p_eval"]}
            model_blocks[key] = _eval_block(
                key, ev["y"], c["p_eval"], ev["did"], ev["sid"],
                base_p=base[strongest], base_name=strongest,
                reps=reps if key.endswith(("none", "isotonic", "temperature")) else 50,
                seed=int(cfg["bootstrap"]["seed"]))
            log(f"t2: {key} LL={model_blocks[key]['metrics']['log_loss']:.5f}")
        log(f"t2: {name} fit C={fit['C']} select_ll={fit['select_log_loss']:.5f}")

    # HGB capacity control
    hgb = models.fit_hgb(tr["X"], tr["y"], sel["X"], sel["y"], cfg)
    p_sel = models.predict_hgb(hgb, sel["X"])
    p_ev = models.predict_hgb(hgb, ev["X"])
    cal = models.calibrate_all(p_sel, sel["y"], p_ev, cfg)
    fits["hgb_full"] = {"fit": hgb, "cols": names, "cal": cal}
    for cname, c in cal.items():
        key = "hgb_full" if cname == "none" else f"hgb_full_{cname}"
        preds[key] = {"p_sel": c["p_sel"], "p_eval": c["p_eval"]}
        model_blocks[key] = _eval_block(
            key, ev["y"], c["p_eval"], ev["did"], ev["sid"],
            base_p=base[strongest], base_name=strongest,
            reps=min(100, reps), seed=int(cfg["bootstrap"]["seed"]))
    log(f"t2: hgb iters={hgb['n_iter']} LL={model_blocks['hgb_full']['metrics']['log_loss']:.5f}")

    # Feature-group decomposition (single C for cost).
    cfg1 = json.loads(json.dumps(cfg))
    cfg1["models"]["logistic"]["C_grid"] = [1.0]
    groups = dict(features.DYAD_GROUPS)
    groups["graph"] = [n for n in features.GRAPH_FEATURES if n in names]
    order = ["calendar", "dyad_current", "dyad_lags", "dyad_rolling",
             "dyad_history", "reverse_dyad", "actor_ego", "train_stats", "graph"]
    decomp = models.nested_group_decomposition(
        tr["X"], tr["y"], sel["X"], sel["y"], ev["X"], ev["y"], names,
        order, groups, cfg1)

    # Global permutation importance on a subsample.
    imp = models.permutation_importance(
        models.fit_logistic(tr["X"], tr["y"], sel["X"], sel["y"], cfg,
                            C_grid=[fits["logistic_full"]["fit"]["C"]],
                            feature_names=names),
        ev["X"], ev["y"], names, n_repeats=2,
        seed=int(cfg["seed"]), max_rows=100000)
    imp_hgb = models.permutation_importance(
        hgb, ev["X"], ev["y"], names, n_repeats=2,
        seed=int(cfg["seed"]), max_rows=100000)

    # D2 windows on eval.
    windows = {}
    if cfg["validation"]["windows"]:
        wm = d2_window_masks(pd.DataFrame({
            "did": b["panel"]["did"], "t_idx": b["panel"]["t_idx"],
            "active": b["panel"]["active"], "sid": b["panel"]["sid"],
            "tid": b["panel"]["tid"]}), ev, cfg)
        for wn, m in wm.items():
            if int(m.sum()) < 100:
                continue
            windows[wn] = {"rows": int(m.sum())}
            for key in ("logistic_full_isotonic", "logistic_full", "hgb_full",
                        "per_dyad_markov"):
                if key in model_blocks:
                    windows[wn][key] = binary_metrics(ev["y"][m], preds[key]["p_eval"][m])
                elif key == "per_dyad_markov":
                    windows[wn][key] = binary_metrics(ev["y"][m], base["per_dyad_markov"][m])
        save_part(cfg, "t2_windows", windows)

    out = {
        "target": "T2_dyad_active_next",
        "split": {"train": len(tr["y"]), "select": len(sel["y"]), "eval": len(ev["y"]),
                  "eval_positives": int(ev["y"].sum())},
        "baselines": base_blocks,
        "strongest_baseline": strongest,
        "models": model_blocks,
        "fits": {k: {kk: vv for kk, vv in v.items() if kk in
                     ("C", "n_iter", "select_log_loss", "select_curve", "cols")}
                 for k, v in fits.items()},
        "calibration": {
            "logistic_full": {
                "temperature_T": fits["logistic_full"]["cal"].get("temperature", {}).get("T"),
                "isotonic_select_log_loss": fits["logistic_full"]["cal"].get("isotonic", {}).get("select_log_loss"),
                "none_select_log_loss": fits["logistic_full"]["cal"]["none"]["select_log_loss"],
            }
        },
        "feature_group_decomposition": decomp,
        "permutation_importance_logistic": imp[:20],
        "permutation_importance_hgb": imp_hgb[:20],
        "windows": windows,
        "coef_table": [
            {"feature": n, "coef": round(float(c), 6)}
            for n, c in sorted(zip(names, fits["logistic_full"]["fit"]["coef"]),
                               key=lambda t: -abs(t[1]))[:30]
        ],
        "runtime_seconds": elapsed(t0),
    }
    # Save predictions for report/cards.
    np.savez_compressed(
        os.path.join(cfg["paths"]["cache_dir"], "preds_t2.npz"),
        y=ev["y"], did=ev["did"], sid=ev["sid"], tid=ev["tid"], t_idx=ev["t_idx"],
        **{f"p_{k}": v["p_eval"] for k, v in preds.items()},
        **{f"p_base_{k}": v for k, v in base.items()},
    )
    save_part(cfg, "t2", out)
    return out


# ---------------------------------------------------------------------------
# T1: actor-month activity next month
# ---------------------------------------------------------------------------
def run_t1(cfg: dict, b: dict) -> dict:
    t0 = time.time()
    tr, va = b["actor_train"], b["actor_val"]
    masks = temporal_masks(va["t_idx"], cfg)
    sel = _slice(va, masks["select"])
    ev = _slice(va, masks["eval"])
    names = tr["feature_names"]
    reps = int(cfg["bootstrap"]["reps"])
    log(f"t1: train={tr['X'].shape} select={sel['X'].shape} eval={ev['X'].shape}")

    # Baselines
    base = {
        "global_rate": np.full(len(ev["y"]), float(tr["y"].mean())),
        "persistence": ev["out_active"].astype(np.float64),
    }
    # per-actor Markov
    n_a = int(max(tr["aid"].max(), ev["aid"].max())) + 1
    a = tr["out_active"].astype(np.int64)
    y = tr["y"].astype(np.int64)
    n1 = np.bincount(tr["aid"][a == 1], minlength=n_a).astype(float)
    n11 = np.bincount(tr["aid"][(a == 1) & (y == 1)], minlength=n_a).astype(float)
    n0 = np.bincount(tr["aid"][a == 0], minlength=n_a).astype(float)
    n01 = np.bincount(tr["aid"][(a == 0) & (y == 1)], minlength=n_a).astype(float)
    base["per_actor_markov"] = np.where(
        ev["out_active"] == 1,
        (n11[ev["aid"]] + 2 * y.mean()) / (n1[ev["aid"]] + 2),
        (n01[ev["aid"]] + 2 * y.mean()) / (n0[ev["aid"]] + 2))
    base_blocks = {k: _eval_block(k, ev["y"], p, ev["aid"], ev["aid"],
                                  reps=min(100, reps)) for k, p in base.items()}
    strongest = min(base_blocks, key=lambda k: base_blocks[k]["metrics"]["log_loss"])

    fit = models.fit_logistic(tr["X"], tr["y"], sel["X"], sel["y"], cfg,
                              feature_names=names)
    p_sel = models.predict_logistic(fit, sel["X"])
    p_ev = models.predict_logistic(fit, ev["X"])
    cal = models.calibrate_all(p_sel, sel["y"], p_ev, cfg)
    blocks, preds = {}, {}
    for cname, c in cal.items():
        key = "logistic_full" if cname == "none" else f"logistic_full_{cname}"
        preds[key] = c["p_eval"]
        blocks[key] = _eval_block(key, ev["y"], c["p_eval"], ev["aid"], ev["aid"],
                                  base_p=base[strongest], base_name=strongest,
                                  reps=reps, seed=int(cfg["bootstrap"]["seed"]))
        log(f"t1: {key} LL={blocks[key]['metrics']['log_loss']:.5f}")

    hgb = models.fit_hgb(tr["X"], tr["y"], sel["X"], sel["y"], cfg)
    ph_sel = models.predict_hgb(hgb, sel["X"])
    ph_ev = models.predict_hgb(hgb, ev["X"])
    hcal = models.calibrate_all(ph_sel, sel["y"], ph_ev, cfg)
    for cname, c in hcal.items():
        key = "hgb_full" if cname == "none" else f"hgb_full_{cname}"
        preds[key] = c["p_eval"]
        blocks[key] = _eval_block(key, ev["y"], c["p_eval"], ev["aid"], ev["aid"],
                                  base_p=base[strongest], base_name=strongest,
                                  reps=min(100, reps), seed=int(cfg["bootstrap"]["seed"]))

    # Per-actor local models.
    local = {}
    for aid in sorted(set(tr["aid"].tolist())):
        lm = models.fit_per_actor_logistic(tr, aid, cfg)
        if lm is not None:
            local[int(aid)] = lm
    local_rows = np.where(np.isin(ev["aid"], list(local)))[0]
    if len(local_rows):
        p_local = np.zeros(len(local_rows))
        for k, aid in enumerate(ev["aid"][local_rows]):
            lm = local[int(aid)]
            p_local[k] = models.sigmoid(
                float(lm["intercept"]) + float(
                    (models.scaler_apply(ev["X"][local_rows[k:k + 1]], lm["mu"], lm["sd"])
                     @ lm["coef"])[0]))
        local_block = binary_metrics(ev["y"][local_rows], p_local)
        local_block["n_actors"] = len(local)
        local_block["rows"] = int(len(local_rows))
    else:
        local_block = {"n_actors": len(local), "rows": 0}

    out = {
        "target": "T1_actor_active_next",
        "split": {"train": len(tr["y"]), "select": len(sel["y"]), "eval": len(ev["y"]),
                  "eval_positives": int(ev["y"].sum())},
        "baselines": base_blocks,
        "strongest_baseline": strongest,
        "models": blocks,
        "local_per_actor": local_block,
        "local_models": {str(k): {"n": v["n"], "positives": v["positives"],
                                  "train_log_loss": v["train_log_loss"],
                                  "train_auc": v["train_auc"]}
                         for k, v in local.items()},
        "coef_table": [
            {"feature": n, "coef": round(float(c), 6)}
            for n, c in sorted(zip(names, fit["coef"]), key=lambda t: -abs(t[1]))[:30]
        ],
        "calibration": {"T": cal["temperature"]["T"] if "temperature" in cal else None},
        "runtime_seconds": elapsed(t0),
    }
    np.savez_compressed(
        os.path.join(cfg["paths"]["cache_dir"], "preds_t1.npz"),
        y=ev["y"], aid=ev["aid"], t_idx=ev["t_idx"], region=ev["region"],
        **{f"p_{k}": v for k, v in preds.items()},
        **{f"p_base_{k}": v for k, v in base.items()},
        logistic_coef=fit["coef"], logistic_mu=fit["mu"], logistic_sd=fit["sd"],
        logistic_intercept=fit["intercept"],
        **{f"local_coef_{k}": v["coef"] for k, v in local.items()},
    )
    save_part(cfg, "t1", out)
    return out
