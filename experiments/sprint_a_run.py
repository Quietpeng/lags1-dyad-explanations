#!/usr/bin/env python3
"""Sprint-A runner: reproduction -> embargo diagnostic -> one-shot holdout.

    python3 experiments/sprint_a_run.py --part repro
    python3 experiments/sprint_a_run.py --part inner
    python3 experiments/sprint_a_run.py --part holdout --blind --quick   # smoke
    python3 experiments/sprint_a_run.py --part holdout                   # one shot
    python3 experiments/sprint_a_run.py --part report

Only ``--part holdout`` (without ``--blind``) reads 2018-22 holdout labels,
exactly once.  The frozen P1 caches and result JSONs are read-only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework import data, features, models, sprint_a
from framework import sprint_a_inference as inf
from framework.config import load_config
from framework.metrics import binary_metrics
from framework.splits import temporal_masks
from framework.utils import (Tee, host_info, jdump, jload, log, pkg_versions,
                             sha256_file)

BASE = os.path.join("results", "framework_pilot", "holdout")
PREREG = os.path.join(BASE, "preregistration.json")
ACCESS = os.path.join(BASE, "test_access_log.json")
LOCK = os.path.join(BASE, "run_once.json")
PARTS = os.path.join(BASE, "parts")
TABLES = os.path.join(BASE, "tables")
PREDS = os.path.join(BASE, "preds")

SEEDS = [20261023, 20261024, 20261025, 20261026, 20261027]
REPS = 1000


# ---------------------------------------------------------------------------
# Access / lock helpers
# ---------------------------------------------------------------------------
def append_access(entry: dict) -> None:
    os.makedirs(os.path.dirname(ACCESS), exist_ok=True)
    if os.path.exists(ACCESS):
        doc = jload(ACCESS)
    else:
        doc = {"policy": ("one-shot holdout; every post-freeze 2018-22 read for "
                          "the holdout is logged; three earlier exploratory "
                          "passes (G2, D2, B) predate this log, are outside its "
                          "scope and are disclosed in the manuscript"),
               "entries": []}
    doc.setdefault("entries", []).append(entry)
    jdump(ACCESS, doc)


def lock_start(blind: bool) -> dict:
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    if blind:
        return {"status": "blind_smoke", "attempts": 0}
    if os.path.exists(LOCK):
        doc = jload(LOCK)
        if doc.get("status") == "completed":
            raise RuntimeError(
                "holdout already completed; refusing a second confirmatory run. "
                "See results/framework_pilot/holdout/run_once.json")
        doc["attempts"] = int(doc.get("attempts", 0)) + 1
        doc["status"] = "started"
        doc["last_started_utc"] = sprint_a.utc_now()
        jdump(LOCK, doc)
        return doc
    doc = {"status": "started", "attempts": 1,
           "first_started_utc": sprint_a.utc_now()}
    jdump(LOCK, doc)
    return doc


def lock_finish(info: dict) -> None:
    if info.get("status") == "blind_smoke":
        return
    doc = jload(LOCK) if os.path.exists(LOCK) else {}
    doc.update(info)
    doc["status"] = "completed"
    doc["completed_utc"] = sprint_a.utc_now()
    jdump(LOCK, doc)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def frozen_p1(cfg):
    con = data.connect(cfg)
    data.load_views(con, cfg)
    features.build_dyad_base(con, cfg, force=False)
    gm = features.build_graph_features(con, cfg, force=False)
    tr = features.export_dyad_split(con, cfg, "train", gm)
    va = features.export_dyad_split(con, cfg, "val", gm)
    n_dyads = int(max(tr["did"].max(), va["did"].max())) + 1
    return con, gm, tr, va, n_dyads


def _masks_embargo(t_idx, start, end):
    return ((np.asarray(t_idx) >= start) & (np.asarray(t_idx) < end) &
            ~np.isin(np.asarray(t_idx), sprint_a.EMBARGO_T))


def _eval_preds(y, preds, selected=None):
    keys = selected if selected is not None else list(preds)
    return {k: sprint_a.metrics_block(y, preds[k]) for k in keys}


def _save_preds(path, y, did, sid, tid, t_idx, preds):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path, y=np.asarray(y, dtype=np.int8), did=np.asarray(did, dtype=np.int32),
        sid=np.asarray(sid, dtype=np.int16), tid=np.asarray(tid, dtype=np.int16),
        t_idx=np.asarray(t_idx, dtype=np.int16),
        **{f"p_{k}": np.asarray(v, dtype=np.float32) for k, v in preds.items()})


# ---------------------------------------------------------------------------
# Part: P1 reproduction (inner validation, no holdout)
# ---------------------------------------------------------------------------
def part_repro(cfg) -> dict:
    t0 = time.time()
    con, gm, tr, va, n_dyads = frozen_p1(cfg)
    stats = features.fit_dyad_stats(tr, n_dyads)
    features.apply_dyad_stats(tr, stats, True)
    features.apply_dyad_stats(va, stats, False)
    masks = temporal_masks(va["t_idx"], cfg)
    sel = sprint_a.slice_dict(va, masks["select"])
    ev = sprint_a.slice_dict(va, masks["eval"])
    log(f"repro: train={tr['X'].shape} select={sel['X'].shape} eval={ev['X'].shape}")

    base = {
        "global_transition": models.baseline_global_transition(tr, ev),
        "per_dyad_markov": models.baseline_per_dyad_markov(tr, ev, n_dyads),
    }
    out = {"split": {"train": int(len(tr["y"])), "select": int(len(sel["y"])),
                     "eval": int(len(ev["y"])), "eval_pos": int(ev["y"].sum())},
           "baselines": {k: binary_metrics(ev["y"], p) for k, p in base.items()}}
    for name, cols in (("logistic29", sprint_a.FEATURES_29),
                       ("logistic49", sprint_a.FEATURES_49)):
        r = sprint_a.fit_logistic_spec(cfg, tr, sel, ev, cols)
        for cal, p in (("none", r["cal"]["none"]["p_eval"]),
                       ("temperature", r["cal"]["temperature"]["p_eval"]),
                       ("isotonic", r["cal"]["isotonic"]["p_eval"])):
            out[f"{name}_{cal}"] = binary_metrics(ev["y"], p)
    ref = jload("results/framework_pilot/parts/t2.json")
    ref_ll = ref["models"]["logistic_full"]["metrics"]["log_loss"]
    ref_markov = ref["baselines"]["per_dyad_markov"]["metrics"]["log_loss"]
    got_ll = out["logistic49_none"]["log_loss"]
    got_markov = out["baselines"]["per_dyad_markov"]["log_loss"]
    out["reference"] = {"logistic_full": ref_ll, "per_dyad_markov": ref_markov}
    out["delta_vs_reference"] = {"logistic_full": got_ll - ref_ll,
                                 "per_dyad_markov": got_markov - ref_markov}
    out["pass"] = bool(abs(got_ll - ref_ll) < 1e-4 and
                       abs(got_markov - ref_markov) < 1e-4)
    out["runtime_seconds"] = round(time.time() - t0, 2)
    con.close()
    jdump(os.path.join(PARTS, "repro.json"), out)
    log(f"repro: logistic49 LL={got_ll:.6f} ref={ref_ll:.6f} "
        f"markov={got_markov:.6f} ref={ref_markov:.6f} pass={out['pass']}")
    return out


# ---------------------------------------------------------------------------
# Part: boundary-embargo diagnostic on 2016-17 (inner validation, no holdout)
# ---------------------------------------------------------------------------
def part_inner(cfg) -> dict:
    t0 = time.time()
    con, gm, tr, va, n_dyads = frozen_p1(cfg)

    tr_mask = (tr["t_idx"] < sprint_a.TRAIN_END) & ~np.isin(tr["t_idx"], sprint_a.EMBARGO_T)
    sel_mask = _masks_embargo(va["t_idx"], sprint_a.SELECT_START, sprint_a.SELECT_END)
    ev_mask = _masks_embargo(va["t_idx"], sprint_a.INNER_EVAL_START, sprint_a.INNER_EVAL_END)
    train_d = sprint_a.slice_dict(tr, tr_mask)
    select_d = sprint_a.slice_dict(va, sel_mask)
    eval_d = sprint_a.slice_dict(va, ev_mask)

    stats_emb = features.fit_dyad_stats(train_d, n_dyads)
    features.apply_dyad_stats(train_d, stats_emb, True)
    features.apply_dyad_stats(select_d, stats_emb, False)
    features.apply_dyad_stats(eval_d, stats_emb, False)

    # br without the embargo, for the delta table.  ``tr`` is the full frozen
    # train split and includes 2014-12 (t_idx=239); ``train_d`` excludes it.
    tr_no = tr
    stats_no = features.fit_dyad_stats(tr_no, n_dyads)
    n_no = stats_no["n_tr"][eval_d["did"]]
    br_no = np.where(n_no > 0,
                     stats_no["k_tr"][eval_d["did"]] / np.maximum(n_no, 1.0),
                     float(stats_no["global_rate"]))
    d_br = eval_d["br"] - br_no
    br_delta = {
        "eval_rows_changed": int((np.abs(d_br) > 1e-12).sum()),
        "max_abs": float(np.max(np.abs(d_br))),
        "mean_abs": float(np.mean(np.abs(d_br))),
        "train_rows_dropped": int(len(tr) - len(train_d)),
        "train_rows_after": int(len(train_d)),
    }

    base = {
        "global_transition": models.baseline_global_transition(train_d, eval_d),
        "per_dyad_markov": models.baseline_per_dyad_markov(train_d, eval_d, n_dyads),
        "dyad_base_rate": eval_d["br"].astype(np.float64),
    }
    out = {"split": {"train": int(len(train_d["y"])),
                     "select": int(len(select_d["y"])),
                     "eval": int(len(eval_d["y"])),
                     "eval_pos": int(eval_d["y"].sum())},
           "embargo": {"t_idx": list(sprint_a.EMBARGO_T),
                       "train_dropped": int(len(tr) - len(train_d)),
                       "select_dropped": int((~sel_mask &
                                              (va["t_idx"] >= 240) &
                                              (va["t_idx"] < 252)).sum()),
                       "eval_dropped": int((~ev_mask &
                                            (va["t_idx"] >= 252) &
                                            (va["t_idx"] < 276)).sum()),
                       "br_delta": br_delta},
           "baselines": {k: binary_metrics(eval_d["y"], p) for k, p in base.items()}}
    preds = dict(base)
    for name, cols in (("logistic29", sprint_a.FEATURES_29),
                       ("logistic49", sprint_a.FEATURES_49)):
        r = sprint_a.fit_logistic_spec(cfg, train_d, select_d, eval_d, cols)
        for cal in ("none", "temperature", "isotonic"):
            preds[f"{name}_{cal}"] = r["cal"][cal]["p_eval"]
            out[f"{name}_{cal}"] = binary_metrics(eval_d["y"], r["cal"][cal]["p_eval"])
    ref = jload("results/framework_pilot/parts/t2.json")
    ref_ll = ref["models"]["logistic_full"]["metrics"]["log_loss"]
    ref_markov = ref["baselines"]["per_dyad_markov"]["metrics"]["log_loss"]
    out["reference"] = {"logistic_full": ref_ll, "per_dyad_markov": ref_markov}
    out["delta_vs_reference"] = {
        "logistic49_none": out["logistic49_none"]["log_loss"] - ref_ll,
        "per_dyad_markov": out["baselines"]["per_dyad_markov"]["log_loss"] - ref_markov,
    }
    out["runtime_seconds"] = round(time.time() - t0, 2)
    con.close()
    jdump(os.path.join(PARTS, "embargo_inner.json"), out)
    _save_preds(os.path.join(PREDS, "inner_embargo_preds.npz"),
                eval_d["y"], eval_d["did"], eval_d["sid"], eval_d["tid"],
                eval_d["t_idx"], preds)
    log(f"inner: logistic49 LL={out['logistic49_none']['log_loss']:.6f} "
        f"markov={out['baselines']['per_dyad_markov']['log_loss']:.6f} "
        f"br_delta_max={br_delta['max_abs']:.6g}")
    return out


# ---------------------------------------------------------------------------
# Part: one-shot holdout
# ---------------------------------------------------------------------------
def part_holdout(cfg, blind: bool = False, quick: bool = False) -> dict:
    t0 = time.time()
    payload = sprint_a.verify_preregistration(PREREG)
    spec = {"dids": np.asarray(payload["definition"]["dids"], dtype=np.int64),
            "counts": payload["holdout"]["counts"],
            "definition": payload["definition"]}
    lock = lock_start(blind)
    if not blind:
        append_access({"utc": sprint_a.utc_now(), "kind": "holdout_eval",
                       "labels_read": True, "one_shot": True,
                       "n_rows_expected": int(payload["holdout"]["counts"]["holdout_eval_rows"]),
                       "note": "one-shot confirmatory read; no re-run permitted"})
    log(f"holdout: prereg={payload['hashes']['definition_sha256']} blind={blind}")

    b = sprint_a.build_holdout_bundle(
        cfg, spec, force_graph=False, label_mode="blind" if blind else "full")
    if blind:
        rng = np.random.default_rng(7)
        b["eval"]["y"] = (rng.random(len(b["eval"]["y"])) < 0.10).astype(np.int64)
        log("holdout: BLIND smoke labels injected; metrics are not results")
    if quick:
        rng = np.random.default_rng(11)
        for key in ("train", "select", "eval"):
            n = len(b[key]["y"])
            n_keep = min(n, {"train": 200000, "select": 50000, "eval": 50000}[key])
            idx = np.sort(rng.choice(n, n_keep, replace=False))
            b[key] = sprint_a.slice_dict(b[key], np.isin(np.arange(n), idx))
        log("holdout: QUICK subsample; results are not final")
    y = b["eval"]["y"]
    did, sid, tid = b["eval"]["did"], b["eval"]["sid"], b["eval"]["tid"]
    log(f"holdout: eval rows={len(y):,} positives={int(y.sum()):,} "
        f"dyads={len(np.unique(did))}")

    # Baselines ------------------------------------------------------------
    preds = {
        "base_global_transition": models.baseline_global_transition(b["train"], b["eval"]),
        "base_per_dyad_markov": models.baseline_per_dyad_markov(
            b["train"], b["eval"], b["n_dyads"]),
        "base_dyad_base_rate": b["eval"]["br"].astype(np.float64),
        "base_warm_base_rate": sprint_a.baseline_warm_base_rate(b["eval"], b),
        "base_warm_markov": sprint_a.baseline_warm_markov(b["eval"], b),
    }
    mr = sprint_a.fit_markov_recency(cfg, b["train"], b["select"], b["eval"])
    preds["base_markov_recency"] = mr["p_eval"]

    # Logistic specifications --------------------------------------------
    specs = {}
    for name, cols in (("logistic49", sprint_a.FEATURES_49),
                       ("logistic29", sprint_a.FEATURES_29)):
        log(f"holdout: fitting {name} S1")
        r = sprint_a.fit_logistic_spec(cfg, b["train"], b["select"], b["eval"], cols)
        specs[f"{name}_S1"] = r
        for cal in ("none", "temperature", "isotonic"):
            preds[f"{name}_S1_{cal}"] = r["cal"][cal]["p_eval"]

    # S2: clip t_yr at the last train month, in place, then restore.
    j = b["train"]["feature_names"].index("t_yr")
    clip = float((sprint_a.TRAIN_END - 1) / 12.0)
    saved = {k: b[k]["X"][:, j].copy() for k in ("train", "select", "eval")}
    for k in ("train", "select", "eval"):
        b[k]["X"][:, j] = np.minimum(b[k]["X"][:, j], np.float32(clip))
    for name, cols in (("logistic49", sprint_a.FEATURES_49),
                       ("logistic29", sprint_a.FEATURES_29)):
        log(f"holdout: fitting {name} S2 (clipped calendar)")
        r = sprint_a.fit_logistic_spec(cfg, b["train"], b["select"], b["eval"], cols)
        specs[f"{name}_S2"] = r
        for cal in ("none", "temperature", "isotonic"):
            preds[f"{name}_S2_{cal}"] = r["cal"][cal]["p_eval"]
    for k in ("train", "select", "eval"):
        b[k]["X"][:, j] = saved[k]
    del saved

    # HGB capacity control ------------------------------------------------
    log("holdout: fitting HGB49 grid")
    hgb = sprint_a.fit_hgb_grid(cfg, b["train"], b["select"], b["eval"],
                                sprint_a.FEATURES_49)
    for cal in ("none", "temperature", "isotonic"):
        preds[f"hgb49_{cal}"] = hgb["cal"][cal]["p_eval"]
    specs["hgb49"] = hgb

    metrics = {k: sprint_a.metrics_block(y, v) for k, v in preds.items()}
    log("holdout: point metrics done")

    # Primary comparisons -------------------------------------------------
    comparisons = {}
    comp_specs = [
        ("logistic49_temperature__vs__per_dyad_markov",
         "base_per_dyad_markov", "logistic49_S1_temperature"),
        ("logistic49_temperature__vs__dyad_base_rate",
         "base_dyad_base_rate", "logistic49_S1_temperature"),
        ("logistic49_temperature__vs__markov_recency",
         "base_markov_recency", "logistic49_S1_temperature"),
        ("logistic49_temperature__vs__global_transition",
         "base_global_transition", "logistic49_S1_temperature"),
        ("logistic49_temperature__vs__warm_markov",
         "base_warm_markov", "logistic49_S1_temperature"),
        ("logistic49_temperature__vs__warm_base_rate",
         "base_warm_base_rate", "logistic49_S1_temperature"),
        ("logistic29_temperature__vs__per_dyad_markov",
         "base_per_dyad_markov", "logistic29_S1_temperature"),
        ("logistic29_temperature__vs__warm_markov",
         "base_warm_markov", "logistic29_S1_temperature"),
        ("logistic49_clip_temperature__vs__per_dyad_markov",
         "base_per_dyad_markov", "logistic49_S2_temperature"),
        ("logistic49_clip_temperature__vs__warm_markov",
         "base_warm_markov", "logistic49_S2_temperature"),
        ("hgb49__vs__per_dyad_markov", "base_per_dyad_markov", "hgb49_none"),
        ("hgb49__vs__logistic49_temperature", "logistic49_S1_temperature", "hgb49_none"),
    ]
    for name, bkey, mkey in comp_specs:
        log(f"holdout: paired multiway {name}")
        comparisons[name] = inf.paired_multiway(
            y, preds[bkey], preds[mkey], did, sid, tid, reps=REPS, seed=SEEDS[0])

    # Five-seed spread for the primary contrast.
    seed_spread = []
    for s in SEEDS:
        r = inf.paired_multiway(y, preds["base_per_dyad_markov"],
                                preds["logistic49_S1_temperature"], did, sid, tid,
                                reps=REPS, seed=s)
        seed_spread.append({"seed": s, "point": r["point"],
                            "multiway": r["multiway"],
                            "dyad_ci": r["dims"]["dyad"]["ci_ll"]})
        log(f"holdout: seed {s} dLL={r['point']['d_log_loss']:.6f} "
            f"multiway_ci={r['multiway']['ci_ll']}")

    # AUC/AP level and paired delta for the primary contrast.
    log("holdout: AUC/AP multiway (primary)")
    auc_levels = {}
    for key in ("logistic49_S1_temperature", "logistic29_S1_temperature",
                "hgb49_none", "base_per_dyad_markov", "base_warm_markov"):
        auc_levels[key] = inf.auc_ap_multiway(
            y, preds[key], did, sid, tid, reps=REPS, seed=SEEDS[0])
        log(f"holdout: auc_ap {key} done")
    auc_deltas = {}
    for name, bkey, mkey in (
            ("logistic49_temperature__vs__per_dyad_markov",
             "base_per_dyad_markov", "logistic49_S1_temperature"),
            ("logistic49_temperature__vs__warm_markov",
             "base_warm_markov", "logistic49_S1_temperature"),
            ("logistic29_temperature__vs__per_dyad_markov",
             "base_per_dyad_markov", "logistic29_S1_temperature"),
            ("hgb49__vs__logistic49_temperature",
             "logistic49_S1_temperature", "hgb49_none")):
        auc_deltas[name] = inf.paired_auc_ap_multiway(
            y, preds[bkey], preds[mkey], did, sid, tid, reps=REPS, seed=SEEDS[0])
        log(f"holdout: paired auc_ap {name} done")

    mde = {
        "logistic49_temperature__vs__per_dyad_markov":
            inf.mde_pair(y, preds["base_per_dyad_markov"],
                         preds["logistic49_S1_temperature"], did),
        "logistic49_temperature__vs__warm_markov":
            inf.mde_pair(y, preds["base_warm_markov"],
                         preds["logistic49_S1_temperature"], did),
    }

    # Feature-group ablation ---------------------------------------------
    log("holdout: feature-group ablation")
    groups = dict(features.DYAD_GROUPS)
    groups["graph"] = [n for n in features.GRAPH_FEATURES
                       if n in b["train"]["feature_names"]]
    ablation = feature_group_ablation(cfg, b, groups, preds, y, did, quick,
                                      specs=specs)

    out = {
        "target": "T2_dyad_active_next",
        "holdout": {
            "definition_sha256": payload["hashes"]["definition_sha256"],
            "unordered_pairs": int(payload["holdout"]["counts"]["unordered_pairs"]),
            "directed_dyads": int(payload["holdout"]["counts"]["directed_dyads"]),
            "period": "2018-01..2022-11 (retrospective; unit-fresh)",
            "rows": int(len(y)), "positives": int(y.sum()),
            "prevalence": float(y.mean()),
            "blind": bool(blind), "quick": bool(quick),
        },
        "counts": b["counts"],
        "assertions": b["assertions"],
        "metrics": metrics,
        "comparisons": comparisons,
        "seed_spread": seed_spread,
        "auc_levels": auc_levels,
        "auc_deltas": auc_deltas,
        "mde": mde,
        "feature_group_ablation": ablation,
        "hgb_grid": hgb["grid"],
        "hgb_params": hgb["fit"]["params"],
        "logistic_C": {k: v["fit"]["C"] for k, v in specs.items()
                       if v["fit"].get("kind") == "logistic"},
        "logistic_select_ll": {k: v["fit"]["select_log_loss"] for k, v in specs.items()
                               if v["fit"].get("kind") == "logistic"},
        "calibration": {
            k: {"T": v["cal"]["temperature"]["T"],
                "select_ll_none": v["cal"]["none"]["select_log_loss"],
                "select_ll_temperature": v["cal"]["temperature"]["select_log_loss"],
                "select_ll_isotonic": v["cal"]["isotonic"].get("select_log_loss")}
            for k, v in specs.items() if "cal" in v
        },
        "feature_sets": {"n29": len(sprint_a.FEATURES_29),
                         "n49": len(sprint_a.FEATURES_49),
                         "features_29": sprint_a.FEATURES_29,
                         "features_49": sprint_a.FEATURES_49},
        "runtime_seconds": round(time.time() - t0, 2),
    }
    jdump(os.path.join(PARTS, "holdout_core.json"), out)
    _save_preds(os.path.join(PREDS, "holdout_preds.npz"),
                y, did, sid, tid, b["eval"]["t_idx"], preds)
    lock_finish({"rows": int(len(y)), "positives": int(y.sum()),
                 "definition_sha256": payload["hashes"]["definition_sha256"],
                 "blind": bool(blind), "quick": bool(quick),
                 "runtime_seconds": out["runtime_seconds"]})
    log(f"holdout: done in {out['runtime_seconds']}s")
    return out


# ---------------------------------------------------------------------------
# Feature-group ablation
# ---------------------------------------------------------------------------
def feature_group_ablation(cfg, b, groups, preds, y, did, quick=False,
                           specs=None) -> dict:
    from framework import models as M

    names = b["train"]["feature_names"]
    order = sprint_a.GROUP_ORDER
    idx = {n: i for i, n in enumerate(names)}
    C_grid = [0.1, 1.0, 3.0] if not quick else [1.0]
    n_perm = 2 if quick else 10

    # Anchor rows.
    anchors = {}
    g = b["train"]
    anchors["intercept_only"] = np.full(len(y), float(g["y"].mean()),
                                        dtype=np.float64)
    anchors["global_transition"] = models.baseline_global_transition(
        g, b["eval"])
    # calendar-only (linear) and clipped-calendar.
    for name, clip in (("calendar_linear", None), ("calendar_clipped", True)):
        cols = ["t_yr", "mo_sin", "mo_cos"]
        ci = [idx[c] for c in cols]
        Xtr, Xsel, Xev = g["X"][:, ci], b["select"]["X"][:, ci], b["eval"]["X"][:, ci]
        if clip:
            Xtr = Xtr.copy(); Xsel = Xsel.copy(); Xev = Xev.copy()
            Xtr[:, 0] = np.minimum(Xtr[:, 0], np.float32((sprint_a.TRAIN_END - 1) / 12.0))
            Xsel[:, 0] = np.minimum(Xsel[:, 0], np.float32((sprint_a.TRAIN_END - 1) / 12.0))
            Xev[:, 0] = np.minimum(Xev[:, 0], np.float32((sprint_a.TRAIN_END - 1) / 12.0))
        fit = M.fit_logistic(Xtr, g["y"], Xsel, b["select"]["y"], cfg,
                             C_grid=[1.0], feature_names=cols)
        anchors[name] = M.predict_logistic(fit, Xev)
    # year FE anchor on the 29-feature base.
    cols29 = sprint_a.FEATURES_29
    Xtr29 = g["X"][:, [idx[c] for c in cols29]]
    Xsel29 = b["select"]["X"][:, [idx[c] for c in cols29]]
    Xev29 = b["eval"]["X"][:, [idx[c] for c in cols29]]
    Xtr_fe, names_fe = sprint_a.year_fe_matrix(Xtr29, cols29, g["t_idx"])
    Xsel_fe, _ = sprint_a.year_fe_matrix(Xsel29, cols29, b["select"]["t_idx"])
    Xev_fe, _ = sprint_a.year_fe_matrix(Xev29, cols29, b["eval"]["t_idx"])
    fit_fe = M.fit_logistic(Xtr_fe, g["y"], Xsel_fe, b["select"]["y"], cfg,
                            C_grid=[1.0], feature_names=names_fe)
    anchors["calendar_year_fe_29"] = M.predict_logistic(fit_fe, Xev_fe)
    anchor_metrics = {k: sprint_a.metrics_block(y, v) for k, v in anchors.items()}

    # Cumulative path.
    cumulative, active = [], []
    for gname in order:
        active = active + [f for f in groups.get(gname, []) if f in idx]
        ci = [idx[f] for f in active]
        fit = M.fit_logistic(g["X"][:, ci], g["y"], b["select"]["X"][:, ci],
                             b["select"]["y"], cfg, C_grid=C_grid,
                             feature_names=active)
        p = M.predict_logistic(fit, b["eval"]["X"][:, ci])
        m = sprint_a.metrics_block(y, p)
        cumulative.append({"groups": list(order[:order.index(gname) + 1]),
                           "n_features": len(ci), "C": fit["C"],
                           **{k: m[k] for k in ("log_loss", "ece", "auc", "ap")}})
        log(f"ablation: cumulative {gname} n={len(ci)} LL={m['log_loss']:.6f}")
    full_ll = cumulative[-1]["log_loss"]

    # LOO at the primary full model's selected C and raw log-loss.
    full_C = float(specs["logistic49_S1"]["fit"]["C"])
    full_ll = sprint_a.metrics_block(
        y, preds["logistic49_S1_none"])["log_loss"]
    loo = []
    for gname in order:
        keep = [f for f in names if f not in set(groups.get(gname, []))]
        ci = [idx[f] for f in keep]
        fit = M.fit_logistic(g["X"][:, ci], g["y"], b["select"]["X"][:, ci],
                             b["select"]["y"], cfg, C_grid=[full_C],
                             feature_names=keep)
        p = M.predict_logistic(fit, b["eval"]["X"][:, ci])
        ll = binary_metrics(y, p)["log_loss"]
        loo.append({"group": gname, "n_features": len(ci),
                    "log_loss_without": ll, "delta_vs_full": ll - full_ll})
        log(f"ablation: LOO {gname} LL={ll:.6f} delta={ll - full_ll:+.6f}")

    # Grouped permutation on the raw log-odds model.  All columns of a group
    # are permuted with the same row permutation, preserving within-group
    # correlation.  The reported delta is the mean held-out LL increase; its CI
    # is a dyad-cluster bootstrap of the per-row mean delta.
    full_fit = specs["logistic49_S1"]["fit"]
    full_p = preds["logistic49_S1_none"]
    full_loss = -(y * np.log(np.clip(full_p, 1e-12, 1)) +
                  (1 - y) * np.log(np.clip(1 - full_p, 1e-12, 1)))
    perm = []
    rng = np.random.default_rng(20261023)
    for gname in order:
        cols = [idx[f] for f in groups.get(gname, []) if f in idx]
        if not cols:
            continue
        deltas = np.zeros(len(y), dtype=np.float64)
        for _ in range(n_perm):
            Xp = b["eval"]["X"].copy()
            perm_idx = rng.permutation(len(y))
            Xp[:, cols] = Xp[perm_idx][:, cols]
            pp = models.predict_logistic(full_fit, Xp)
            deltas += (-(y * np.log(np.clip(pp, 1e-12, 1)) +
                         (1 - y) * np.log(np.clip(1 - pp, 1e-12, 1))) - full_loss)
        deltas /= n_perm
        boot = _boot_mean_delta(deltas, did, reps=1000 if not quick else 100,
                                seed=20261023)
        perm.append({"group": gname, "n_features": len(cols),
                     "delta_log_loss": float(deltas.mean()),
                     "delta_ci": boot["ci"], "se": boot["se"]})
        log(f"ablation: perm {gname} delta={deltas.mean():+.6f} ci={boot['ci']}")

    return {"anchors": anchor_metrics, "cumulative": cumulative,
            "loo": loo, "grouped_permutation": perm,
            "full_log_loss": full_ll, "full_C": full_C}


def _boot_mean_delta(deltas, cid, reps=1000, seed=0):
    """Cluster bootstrap of the mean per-row delta."""
    _, inv = np.unique(np.asarray(cid), return_inverse=True)
    k = int(inv.max()) + 1
    sums = np.bincount(inv, weights=np.asarray(deltas, dtype=np.float64),
                       minlength=k)
    ns = np.bincount(inv, minlength=k).astype(np.float64)
    rng = np.random.default_rng(seed)
    out = np.empty(reps, dtype=np.float64)
    for r in range(reps):
        c = rng.integers(0, k, size=k)
        out[r] = sums[c].sum() / max(ns[c].sum(), 1.0)
    return {"ci": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))],
            "se": float(np.std(out, ddof=1))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["repro", "inner", "holdout", "report", "all"])
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(PARTS, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(PREDS, exist_ok=True)
    log_path = os.path.join(BASE, "run.log")
    tee = Tee(log_path)
    sys.stdout = tee
    sys.stderr = tee
    cfg = load_config()
    log(f"sprint_a: host={host_info()}")
    log(f"sprint_a: packages={pkg_versions()}")
    try:
        if args.part in ("repro", "all"):
            part_repro(cfg)
        if args.part in ("inner", "all"):
            part_inner(cfg)
        if args.part in ("holdout", "all"):
            part_holdout(cfg, blind=args.blind, quick=args.quick)
        if args.part in ("report", "all"):
            from sprint_a_report import build_report
            build_report(BASE)
    finally:
        tee.close()
        sys.stdout = tee.stdout
        sys.stderr = tee.stderr
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
