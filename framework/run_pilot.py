"""Command-line entry point for the framework pilot.

Examples
--------
    python -m framework.run_pilot --stage corpus
    python -m framework.run_pilot --stage t1
    python -m framework.run_pilot --stage all
    python -m framework.run_pilot --config configs/smoke.json --stage all

Stages are independent except that ``synth`` requires ``corpus``.  ``report``
reads the four parts and the leakage audit; it never refits a model.
"""

from __future__ import annotations

import argparse
import os
import time

import duckdb
import numpy as np

from . import audit, interpret, reporting, synth, synth_arms, targets, windows
from .config import ensure_dirs, load_config
from .utils import elapsed, log


def build_corpus(cfg: dict, force: bool = False) -> dict:
    """Generate the synthetic corpus from real-train moments only."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from framework import features as F

    con = duckdb.connect()
    p = lambda n: str(os.path.join(cfg["paths"]["cache_dir"], n)).replace("\\", "/")
    try:
        fb = con.execute(f"""
          SELECT active, active_next, n_events
          FROM read_parquet('{p("features_dyad.parquet")}')
          WHERE split='train'
        """).fetchnumpy()
        ab = con.execute(f"""
          SELECT out_events, out_partners
          FROM read_parquet('{p("features_actor.parquet")}')
          WHERE split='train'
        """).fetchnumpy()
    finally:
        con.close()
    train_d = {"y": fb["active_next"].astype(np.int64),
               "active": fb["active"].astype(np.int64),
               "n_events": fb["n_events"].astype(float)}
    train_a = {"out_events": ab["out_events"].astype(float),
               "out_partners": ab["out_partners"].astype(float)}
    return synth.generate_corpus(
        cfg, train_d, train_a, force=force,
        cache_subdir=cfg["synth"].get("corpus_subdir", "synth_full"),
        tag_prefix=cfg["synth"].get("tag_prefix", "_synthfull"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Q2 framework pilot runner")
    ap.add_argument("--config", default="configs/framework_pilot.json")
    ap.add_argument("--stage", default="all",
                    choices=["corpus", "t1", "t2", "synth", "interp", "audit",
                             "report", "all"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    t0 = time.time()
    log(f"run_pilot: config={args.config} stage={args.stage}")
    if args.stage in ("corpus", "all"):
        man = build_corpus(cfg, force=args.force)
        log(f"corpus: {man['corpus_sha256'][:16]} worlds={len(man['worlds'])}")
    if args.stage in ("t1", "all"):
        targets.run(cfg, os.path.join(cfg["paths"]["out_dir"], "parts",
                                      "t1_redesign.json"))
    if args.stage in ("t2", "all"):
        windows.run(cfg, os.path.join(cfg["paths"]["out_dir"], "parts",
                                      "t2_windows.json"))
    if args.stage in ("synth", "all"):
        synth_arms.run(cfg, os.path.join(cfg["paths"]["out_dir"], "parts",
                                         "synth_arms.json"))
    if args.stage in ("interp", "all"):
        interpret.run(cfg, os.path.join(cfg["paths"]["out_dir"], "parts",
                                          "interpretability.json"))
    if args.stage in ("audit", "all"):
        a = audit.run(cfg, os.path.join(cfg["paths"]["out_dir"], "audit",
                                        "framework_leakage_audit.json"))
        audit.write_markdown(a, os.path.join(cfg["paths"]["out_dir"], "audit",
                                             "framework_leakage_audit.md"))
    if args.stage in ("report", "all"):
        reporting.run(cfg)
    log(f"run_pilot: done in {elapsed(t0):.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
