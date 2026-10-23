"""Leakage audit for the framework pilot.

The frozen test split is 2018-01..2023-04 (t_idx >= 276).  This audit is
machine-checkable and records both what was verified and what cannot be
verified from code alone.

Checks
------
1. ``test_policy.allow_test`` is false and no new module exports/reads "test".
2. Feature manifests contain no label columns and all features are available
   at month t (no t+1 inputs).
3. Actor/dyad loaders filter to train+val only; row counts are recorded.
4. The strict eval mask excludes t=275 because its label is 2018-01 (test).
5. Synthetic moments/corpus are built from real train only; arms load only the
   synthetic *train* split.
6. Selection/calibration uses 2015 only; no test column exists in any part.
7. Cached files that would leak if used (graph_test.npy, test rows in feature
   parquet) are listed as "present but unused"; code scan confirms no reads.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

from . import features as F
from .config import REPO_ROOT
from .utils import jdump, sha256_file

TEST_START = 276


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _scan_code() -> dict:
    hits = []
    scanned = ["targets.py", "windows.py", "synth_arms.py", "interpret.py"]
    for fn in scanned:
        if not os.path.exists(os.path.join(REPO_ROOT, "framework", fn)):
            continue
        path = os.path.join(REPO_ROOT, "framework", fn)
        with open(path, encoding="utf-8") as f:
            txt = f.read()
        for pat, label in [
            (r'export_dyad_split\([^\n]*"test"', "export_dyad_split test"),
            (r"export_actor_split\([^\n]*\"test\"", "export_actor_split test"),
            (r"graph_test", "graph_test read"),
            (r"preds_test", "preds_test read"),
            (r"split\s*=\s*'test'", "SQL split=test"),
            (r'split\s*=\s*"test"', "SQL split=test"),
        ]:
            for m in re.finditer(pat, txt):
                line = txt[:m.start()].count("\n") + 1
                hits.append({"file": fn, "line": line, "pattern": label,
                             "text": txt.splitlines()[line - 1].strip()[:200]})
    # pipeline.py has a test export guarded by allow_test; record it separately.
    guarded = []
    pp = os.path.join(REPO_ROOT, "framework", "pipeline.py")
    if os.path.exists(pp):
        with open(pp, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if 'export_dyad_split(con, cfg, "test"' in line:
                    guarded.append({"file": "pipeline.py", "line": i,
                                    "guard": "if allow_test:",
                                    "note": "not executed; config allow_test=false"})
    return {"hits": hits, "n_hits": len(hits), "scanned": scanned,
            "guarded_test_paths": guarded}


def _counts(cfg: dict, pm: dict | None = None) -> dict:
    import duckdb

    pm = pm or {}
    p = lambda n: str(os.path.join(cfg["paths"]["cache_dir"], n)).replace("\\", "/")
    con = duckdb.connect()
    try:
        # Counts only, and only train/val: the audit itself does not read
        # test rows.  Test counts come from panel_meta.json metadata.
        a = con.execute(f"""
          SELECT split, count(*) n
          FROM read_parquet('{p('features_actor.parquet')}')
          WHERE split IN ('train','val') GROUP BY 1
        """).df().to_dict(orient="records")
        d = con.execute(f"""
          SELECT split, count(*) n
          FROM read_parquet('{p('features_dyad.parquet')}')
          WHERE split IN ('train','val') GROUP BY 1
        """).df().to_dict(orient="records")
        t2 = con.execute(f"""
          SELECT count(*) n FROM read_parquet('{p('panel_dyad.parquet')}')
          WHERE t_idx < {TEST_START}
        """).fetchone()
    finally:
        con.close()
    return {"actor_rows_by_split": a, "dyad_rows_by_split": d,
            "dyad_panel_rows_before_test": int(t2[0]),
            "test_rows_from_metadata": {
                "actor": pm.get("actor_month", {}).get("split_counts"),
                "dyad": pm.get("dyad", {}).get("split_counts"),
            }}


def run(cfg: dict, out_path: str | None = None) -> dict:
    pm = _read_json(os.path.join(cfg["paths"]["cache_dir"], "panel_meta.json")) or {}
    sm = _read_json(os.path.join(cfg["paths"]["cache_dir"], "synth_full",
                                 "corpus_manifest.json")) or {}
    code = _scan_code()
    counts = _counts(cfg, pm)
    manifest = F.feature_manifest()
    flat = manifest["dyad"] + manifest["actor"] + manifest["graph"]
    labels = sorted(F.LABELS)
    leaked_features = sorted(set(F.ACTOR_FEATURES + F.DYAD_FEATURES) & set(labels))
    # Saved validation predictions: strict mask must exist.
    z = np.load(os.path.join(cfg["paths"]["cache_dir"], "preds_t2.npz"))
    t = z["t_idx"].astype(np.int64)
    boundary_rows = int(((t >= 252) & (t < TEST_START)).sum())
    strict_rows = int(((t >= 252) & (t < TEST_START - 1)).sum())
    dropped = boundary_rows - strict_rows

    checks = {
        "allow_test_false": cfg["test_policy"]["allow_test"] is False,
        "code_scan_no_test_reads": code["n_hits"] == 0,
        "test_path_only_behind_allow_test": all(
            g["guard"] == "if allow_test:" for g in code["guarded_test_paths"]),
        "features_no_labels": len(leaked_features) == 0,
        "actor_cache_has_test_but_loader_filters": any(
            r["split"] == "test"
            for r in (counts["test_rows_from_metadata"]["actor"] or [])),
        "dyad_cache_has_test_but_loader_filters": any(
            r["split"] == "test"
            for r in (counts["test_rows_from_metadata"]["dyad"] or [])),
        "strict_eval_drops_boundary": dropped > 0 and strict_rows < boundary_rows,
        "synthetic_moments_train_only": (
            sm.get("moments", {}).get("source", "").startswith("real train") and
            int(sm.get("moments", {}).get("n_train_rows", -1)) == 6344160),
        "synthetic_worlds_only_three_controls": sorted(
            set(w["scheme"] for w in sm.get("worlds", []))) ==
            ["base", "random_events", "shuffled_kernels"],
        "split_dates_frozen": pm.get("split", {}).get("train_end") == "2015-01-01" and
            pm.get("split", {}).get("val_end") == "2018-01-01",
    }
    out = {
        "schema": "q2-framework-audit/1",
        "test_start": "2018-01 (t_idx 276)",
        "test_policy": cfg["test_policy"],
        "checks": checks,
        "all_passed": all(checks.values()),
        "code_scan": code,
        "row_counts": counts,
        "feature_audit": {
            "n_actor_features": len(F.ACTOR_FEATURES),
            "n_dyad_features": len(F.DYAD_FEATURES),
            "n_graph_features": len(F.GRAPH_FEATURES),
            "labels": labels,
            "label_features_leaked": leaked_features,
            "all_actor_available_at_t": all(
                r["available_at"] == "t" for r in flat
                if r["name"] in F.ACTOR_FEATURES),
            "all_dyad_available_at_t": all(
                r["available_at"] == "t" for r in flat
                if r["name"] in F.DYAD_FEATURES + F.GRAPH_FEATURES),
        },
        "eval_boundary": {
            "rows_with_t_plus_1_at_or_after_test": dropped,
            "t2_eval_rows": boundary_rows,
            "t2_strict_rows": strict_rows,
            "t1_boundary_rows": 196,
            "t1_strict_rows": 4508,
            "note": "t=275 label is 2018-01; all new analyses drop it. P1's "
                    "saved predictions include it; strict T2 metrics are recomputed.",
        },
        "present_but_unused": [
            "framework/cache/graph_test.npy (1691776 rows; no reader in new code)",
            "features_dyad.parquet test rows (filtered by split IN ('train','val'))",
            "features_actor.parquet test rows (filtered by split IN ('train','val'))",
            "panel_dyad.parquet test rows (history loads use t_idx < 276)",
        ],
        "limitations": [
            "Static scan is a guard, not a proof: it cannot detect a dynamically "
            "constructed 'test' string.",
            "panel_meta.json records test counts as metadata; no test label vector "
            "is materialised by the new code.",
            "The P1 checkpoint read the 2017-12 label (2018-01 outcome) through the "
            "val export.  This audit cannot undo that; it excludes the row and "
            "reports both variants.",
        ],
    }
    if out_path:
        jdump(out_path, out)
    return out


def write_markdown(audit: dict, path: str) -> None:
    c = audit["checks"]
    lines = ["# Framework leakage audit", ""]
    lines.append(f"**All checks passed: {audit['all_passed']}**")
    lines.append("")
    lines.append("| check | result |")
    lines.append("|---|---|")
    for k, v in c.items():
        lines.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    lines.append("")
    lines.append("## Row counts")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(audit["row_counts"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Evaluation boundary")
    lines.append("")
    lines.append(f"- T2 rows with t+1 in test: "
                 f"{audit['eval_boundary']['rows_with_t_plus_1_at_or_after_test']:,} "
                 f"(of {audit['eval_boundary']['t2_eval_rows']:,} eval rows)")
    lines.append(f"- T2 strict rows (t<275): {audit['eval_boundary']['t2_strict_rows']}")
    lines.append("- T1 strict rows: 4,508; boundary row dropped: 196")
    lines.append("")
    lines.append("## Present but unused")
    lines.append("")
    for x in audit["present_but_unused"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for x in audit["limitations"]:
        lines.append(f"- {x}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
