"""Configuration loading and validation.

JSON only (no PyYAML dependency).  A config file may override any default via a
recursive merge.  All paths are resolved relative to the repository root, which
is inferred from this file (``<repo>/framework/config.py``).
"""

from __future__ import annotations

import copy
import os
from typing import Any

from .utils import jload

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS: dict[str, Any] = {
    "schema": "q2-framework/1",
    "seed": 20261023,
    "paths": {
        "icews_glob": "data/icews/icews_events/*/*.parquet",
        "cache_dir": "framework/cache",
        "out_dir": "results/framework_pilot",
    },
    "split": {
        # Frozen three-way split confirmed by the PI.  The test split is never
        # read for fitting/selection; its dates are recorded for the manifest.
        "train_end": "2015-01-01",
        "val_end": "2018-01-01",
        "tail_start": "2023-01-01",
        # Honest inner protocol inside validation:
        #   select window = first year of val (hyper-parameters)
        #   eval window   = remaining val years (reported)
        "select_start": "2015-01-01",
        "select_end": "2016-01-01",
        "eval_start": "2016-01-01",
        "eval_end": "2018-01-01",
    },
    "features": {
        "dyad": "g2_29",
        "graph": "core20",  # core20 | all35 | none
        "actor": "actor_v1",
        "future_graph_control": False,
    },
    "models": {
        "logistic": {
            "C_grid": [0.1, 0.3, 1.0, 3.0],
            "max_iter": 300,
            "tol": 0.0001,
            "solver": "lbfgs",
        },
        "hgb": {
            "max_iter_grid": [50, 100, 150],
            "learning_rate": 0.1,
            "max_leaf_nodes": 31,
            "max_bins": 128,
            "min_samples_leaf": 100,
        },
        "calibration": ["none", "temperature", "isotonic"],
        "per_actor_local": {"enabled": True, "min_pos": 30, "C": 1.0},
    },
    "graph_model": {
        "enabled": True,
        "dim": 8,
        "epochs": 10,
        "lr": 0.002,
        "l2": 1e-3,
        "max_rows_per_month": 4000,
        "seed": 0,
        "val_every": 1,
        "grad_clip": 5.0,
    },
    "operator_head": {
        "enabled": True,
        "classes": ["birth", "death", "merge", "split", "reentry"],
        "C_grid": [0.1, 1.0],
        "max_iter": 300,
        "train_worlds": [0, 1, 2, 3],
        "test_worlds": [4],
    },
    "synth": {
        "enabled": True,
        "K_schemes": 5,
        "worlds_per_scheme": 2,
        "n_actors": 80,
        "n_months": 120,
        "train_frac": 0.6,
        "val_frac": 0.2,
        "event_rate": 0.28,
        "max_actors": 110,
        "max_rows_per_world": 400000,
        "schemes": {
            "base": {"hazard": 1.0, "coupling": 0.5, "noise": 0.35,
                     "mix": [0.25, 0.20, 0.15, 0.15, 0.25]},
            "birth_heavy": {"hazard": 1.1, "coupling": 0.6, "noise": 0.35,
                            "mix": [0.55, 0.10, 0.05, 0.10, 0.20]},
            "merge_split_heavy": {"hazard": 1.1, "coupling": 0.7, "noise": 0.35,
                                  "mix": [0.10, 0.10, 0.40, 0.30, 0.10]},
            "high_noise": {"hazard": 1.2, "coupling": 0.4, "noise": 0.90,
                           "mix": [0.25, 0.20, 0.15, 0.15, 0.25]},
            "low_coupling": {"hazard": 1.0, "coupling": 0.1, "noise": 0.35,
                             "mix": [0.25, 0.20, 0.15, 0.15, 0.25]},
            "high_coupling": {"hazard": 1.2, "coupling": 0.95, "noise": 0.35,
                              "mix": [0.25, 0.20, 0.15, 0.15, 0.25]},
            "random_events": {"hazard": 1.0, "coupling": 0.0, "noise": 0.0,
                              "mix": [0.20, 0.20, 0.20, 0.20, 0.20],
                              "random": True},
            "shuffled_kernels": {"hazard": 1.0, "coupling": 0.5, "noise": 0.35,
                                 "mix": [0.25, 0.20, 0.15, 0.15, 0.25],
                                 "shuffle_kernels": True},
        },
        # Arms use schemes by name; absent schemes fall back to "base".
        "arms": {
            "A0_real": {"sources": ["real"]},
            "A2_synth": {"sources": ["synth"]},
            "A3_real_synth": {"sources": ["real", "synth"], "synth_weight": 0.3},
            "A5_random_events": {"sources": ["real", "synth:random_events"],
                                 "synth_weight": 0.3},
            "A6_param_bootstrap": {"sources": ["real", "synth:shuffled_kernels"],
                                   "synth_weight": 0.3},
        },
    },
    "validation": {
        "regimes": ["temporal", "actor_holdout", "dyad_holdout", "region_holdout"],
        "holdout_frac": 0.2,
        "holdout_seed": 20261023,
        "region": "Africa",
        "min_eval_rows": 500,
        "min_eval_clusters": 30,
        "windows": True,
    },
    "bootstrap": {
        "reps": 500,
        "reps_auc": 200,
        "seed": 20261024,
        "auc_max_rows": 300000,
    },
    "report": {
        "node_cards": True,
        "node_cards_top_n": 196,
        "max_top_features": 8,
        "smoke": False,
    },
    "test_policy": {
        "allow_test": False,
        "note": "Frozen 2018-2022+ tail. P1 pilot must not read test labels.",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve_paths(cfg: dict) -> dict:
    p = cfg["paths"]
    for key in ("cache_dir", "out_dir"):
        if not os.path.isabs(p[key]):
            p[key] = os.path.normpath(os.path.join(REPO_ROOT, p[key]))
    if not os.path.isabs(p["icews_glob"]):
        p["icews_glob"] = os.path.normpath(os.path.join(REPO_ROOT, p["icews_glob"]))
    return cfg


def load_config(path: str | None = None, overrides: dict | None = None) -> dict:
    """Load a config, optionally extending another JSON file.

    A config may contain ``"extends": "base.json"`` (relative to its own
    directory).  The base is loaded first, then the child keys are merged on
    top.  This keeps the stage configs small and auditable.
    """
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        raw = jload(path)
        base = raw.pop("extends", None)
        if base:
            base_path = os.path.join(os.path.dirname(os.path.abspath(path)), base)
            cfg = _deep_merge(cfg, load_config(base_path))
        cfg = _deep_merge(cfg, raw)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    cfg = _resolve_paths(cfg)

    # Basic invariants.
    split = cfg["split"]
    assert split["train_end"] <= split["select_start"] <= split["select_end"]
    assert split["eval_start"] >= split["select_end"]
    assert split["eval_end"] <= split["val_end"]
    assert cfg["validation"]["holdout_frac"] < 0.5
    assert cfg["bootstrap"]["reps"] > 0
    return cfg


def ensure_dirs(cfg: dict) -> None:
    for key in ("cache_dir", "out_dir"):
        os.makedirs(cfg["paths"][key], exist_ok=True)
    os.makedirs(os.path.join(cfg["paths"]["out_dir"], "parts"), exist_ok=True)
    os.makedirs(os.path.join(cfg["paths"]["out_dir"], "tables"), exist_ok=True)
