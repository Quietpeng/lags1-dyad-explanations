"""Assemble the framework-pilot report, summary, manifest and run log.

This module is intentionally read-only with respect to the fitted parts: it
does not refit models except for the strict-boundary headline recomputation
from the saved prediction arrays (cheap and auditable).
"""

from __future__ import annotations

import glob
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np

from . import __version__, features as F
from .bootstrap import paired_gain
from .config import REPO_ROOT
from .metrics import binary_metrics, within_entity_auc
from .utils import elapsed, jdump, sha256_file

EVAL_START, TEST_START = 252, 276


def _p(path: str) -> str:
    return str(path).replace("\\", "/")


def _load(cfg: dict, name: str):
    with open(os.path.join(cfg["paths"]["out_dir"], "parts", f"{name}.json"),
              encoding="utf-8") as f:
        return json.load(f)


def _fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _strict_headline(cfg: dict) -> dict:
    z = np.load(os.path.join(cfg["paths"]["cache_dir"], "preds_t2.npz"))
    y, did, t = z["y"], z["did"], z["t_idx"].astype(np.int64)
    m = t < TEST_START - 1
    out = {"rows": int(m.sum()), "positives": int(y[m].sum()), "models": {}}
    for key in ("p_logistic_full_isotonic", "p_logistic_full", "p_hgb_full",
                "p_base_per_dyad_markov"):
        met = binary_metrics(y[m], z[key][m])
        w = within_entity_auc(y[m], z[key][m], did[m])
        out["models"][key] = {"metrics": met, "within_entity": w}
    g = paired_gain(y[m], z["p_base_per_dyad_markov"][m],
                    z["p_logistic_full_isotonic"][m], did[m], reps=300, seed=1)
    out["gain_vs_per_dyad"] = g
    z1 = np.load(os.path.join(cfg["paths"]["cache_dir"], "preds_t1.npz"))
    y1, aid, t1 = z1["y"], z1["aid"], z1["t_idx"].astype(np.int64)
    m1 = t1 < TEST_START - 1
    out["t1"] = {"rows": int(m1.sum()), "positives": int(y1[m1].sum()), "models": {}}
    for key in ("p_base_per_actor_markov", "p_logistic_full", "p_hgb_full"):
        met = binary_metrics(y1[m1], z1[key][m1])
        w = within_entity_auc(y1[m1], z1[key][m1], aid[m1])
        out["t1"]["models"][key] = {"metrics": met, "within_entity": w}
    out["t1"]["gain_vs_per_actor_markov"] = paired_gain(
        y1[m1], z1["p_base_per_actor_markov"][m1], z1["p_logistic_full"][m1],
        aid[m1], reps=300, seed=1)
    return out


def _write_tables(cfg: dict, t1r, t2w, syn, interp, head) -> list[str]:
    import pandas as pd

    tab = os.path.join(cfg["paths"]["out_dir"], "tables")
    os.makedirs(tab, exist_ok=True)
    paths = []

    rows = []
    for t, r in t1r["targets"].items():
        s = r["support"]
        b = r["baselines"][r["primary_baseline"]]
        m = r["models"]["logistic_full"]["metrics"]
        hm = r["models"]["hgb_full"]["metrics"]
        g = r["primary_gain"]
        rows.append({
            "target": t, "definition": r["definition"],
            "rows": s["rows"], "positives": s["positives"],
            "pos_rate": s["pos_rate"], "clusters": s["clusters"],
            "n_eff_outcome": s["n_eff_outcome"]["n_eff"],
            "threshold": s["threshold"],
            "primary_baseline": r["primary_baseline"],
            "base_ll": b["log_loss"], "model_ll": m["log_loss"],
            "gain": g["point"]["d_log_loss"],
            "gain_ci_low": None if g["d_log_loss_ci"] is None else g["d_log_loss_ci"][0],
            "gain_ci_high": None if g["d_log_loss_ci"] is None else g["d_log_loss_ci"][1],
            "mde": g["mde"]["mde"], "mde_rel": g["mde"].get("mde_rel"),
            "gain_over_mde": r["gain_over_mde"],
            "auc": m["auc"], "ap": m["ap"], "ece": m["ece"],
            "hgb_ll": hm["log_loss"], "hgb_auc": hm["auc"], "hgb_ece": hm["ece"],
            "within_actor_auc": None if s["within_entity_auc"] is None
            else s["within_entity_auc"]["within_auc"],
            "between_actor_auc": None if s["within_entity_auc"] is None
            else s["within_entity_auc"]["between_auc"],
            "powered": r["adequately_powered"],
        })
    p = os.path.join(tab, "t1_redesign.csv")
    pd.DataFrame(rows).to_csv(p, index=False); paths.append(p)

    rows = []
    for r in t2w["table"]:
        wn = r["window"]
        w = t2w["windows"][wn]
        within = None
        n_eff_diff = None
        if w.get("models"):
            within = w["models"]["logistic_full_isotonic"]["within_entity_auc"]["within_auc"]
            g = w.get("gains_vs_logistic_full_isotonic", {}).get("per_dyad_markov", {})
            n_eff_diff = (g.get("mde") or {}).get("n_eff")
        rows.append({**r, "within_dyad_auc": within,
                     "n_eff_diff": n_eff_diff})
    p = os.path.join(tab, "t2_windows.csv")
    pd.DataFrame(rows).to_csv(p, index=False); paths.append(p)

    for sec in ("t2", "t1"):
        rows = []
        for arm, b in syn[sec]["arms"].items():
            iso = b["isotonic"]
            m = iso["metrics"]; g = iso["vs_A0"]
            rows.append({
                "task": sec, "arm": arm, "scheme": b.get("scheme"),
                "n_train": b.get("n_train"), "n_synth": b.get("n_synth"),
                "synth_share": b.get("synth_share"),
                "log_loss": m["log_loss"], "brier": m["brier"], "ece": m["ece"],
                "auc": m["auc"], "ap": m["ap"],
                "d_log_loss": g["point"]["d_log_loss"],
                "d_ll_ci_low": None if g["d_log_loss_ci"] is None else g["d_log_loss_ci"][0],
                "d_ll_ci_high": None if g["d_log_loss_ci"] is None else g["d_log_loss_ci"][1],
                "d_ece": g["point"]["d_ece"],
                "d_ece_ci_low": None if g.get("d_ece_ci") is None else g["d_ece_ci"][0],
                "d_ece_ci_high": None if g.get("d_ece_ci") is None else g["d_ece_ci"][1],
                "mde": g["mde"]["mde"],
                "gain_over_mde": None if not g["mde"]["mde"] else
                round(g["point"]["d_log_loss"] / g["mde"]["mde"], 3),
            })
        p = os.path.join(tab, f"synth_arms_{sec}.csv")
        pd.DataFrame(rows).to_csv(p, index=False); paths.append(p)

    p = os.path.join(tab, "synth_dose_response.csv")
    pd.DataFrame(syn["t2"]["dose_response"]).to_csv(p, index=False); paths.append(p)

    rows = []
    for key in ("p_logistic_full_isotonic", "p_hgb_full",
                "p_base_per_dyad_markov"):
        m = head["models"][key]["metrics"]
        w = head["models"][key]["within_entity"]
        rows.append({"target": "T2", "model": key, **{
            k: m[k] for k in ("n", "positives", "log_loss", "brier", "ece", "auc", "ap")},
            "within_dyad_auc": w["within_auc"], "between_dyad_auc": w["between_auc"]})
    for key in ("p_base_per_actor_markov", "p_logistic_full"):
        m = head["t1"]["models"][key]["metrics"]
        w = head["t1"]["models"][key]["within_entity"]
        rows.append({"target": "T1_next_active", "model": key, **{
            k: m[k] for k in ("n", "positives", "log_loss", "brier", "ece", "auc", "ap")},
            "within_actor_auc": w["within_auc"], "between_actor_auc": w["between_auc"]})
    p = os.path.join(tab, "headline_strict.csv")
    pd.DataFrame(rows).to_csv(p, index=False); paths.append(p)
    return paths


def _manifest(cfg: dict, result_paths: list[str], runtime: float) -> dict:
    import duckdb
    import pandas as pd
    import sklearn

    code = {}
    fw = os.path.join(REPO_ROOT, "framework")
    for fn in sorted(os.listdir(fw)):
        if fn.endswith(".py"):
            code[f"framework/{fn}"] = sha256_file(os.path.join(fw, fn))
    data = {}
    cache = cfg["paths"]["cache_dir"]
    for fn in ("panel_actor.parquet", "panel_dyad.parquet", "panel_meta.json",
               "actors.parquet", "features_actor.parquet", "features_dyad.parquet",
               "graph_meta.json", "graph_train.npy", "graph_val.npy",
               "preds_t1.npz", "preds_t2.npz", "preds_t1_highq75.npz"):
        fp = os.path.join(cache, fn)
        if os.path.exists(fp):
            data[f"framework/cache/{fn}"] = sha256_file(fp)
    for root, _, files in os.walk(os.path.join(cache, "synth_full")):
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            data[os.path.relpath(fp, REPO_ROOT)] = sha256_file(fp)
    results = {}
    for fp in result_paths:
        if os.path.exists(fp):
            results[os.path.relpath(fp, REPO_ROOT)] = sha256_file(fp)
    config_hashes = {}
    for root in ("configs", "tests", "docs"):
        for fn in sorted(os.listdir(os.path.join(REPO_ROOT, root))):
            fp = os.path.join(REPO_ROOT, root, fn)
            if os.path.isfile(fp):
                config_hashes[os.path.relpath(fp, REPO_ROOT)] = sha256_file(fp)
    readme = os.path.join(REPO_ROOT, "README.md")
    if os.path.exists(readme):
        config_hashes["README.md"] = sha256_file(readme)
    with open(os.path.join(cache, "panel_meta.json"), encoding="utf-8") as f:
        pm = json.load(f)
    return {
        "schema": "q2-framework-manifest/1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "framework_version": __version__,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
            "cpus": os.cpu_count(),
            "mem_gb": round(os.sysconf("SC_PAGE_SIZE") *
                            os.sysconf("SC_PHYS_PAGES") / 1e9, 1),
            "hostname": platform.node(),
            "accelerator": "none (CPU only)",
        },
        "packages": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "duckdb": duckdb.__version__,
        },
        "runtime_seconds": round(float(runtime), 2),
        "code_sha256": code,
        "data_sha256": data,
        "result_sha256": results,
        "config_test_doc_sha256": config_hashes,
        "frozen_source": {
            "icews_combined_sha256": pm.get("source", {}).get("combined_sha256"),
            "icews_files": pm.get("source", {}).get("n_files"),
            "panel_dyad_sha256": pm.get("dyad", {}).get("sha256"),
            "panel_actor_sha256": pm.get("actor_month", {}).get("sha256"),
            "split": pm.get("split"),
            "modified": False,
        },
        "test_policy": {
            "allow_test": cfg["test_policy"]["allow_test"],
            "test_read": False,
            "test_start": "2018-01 (t_idx 276)",
            "test_rows_used_in_fit": 0,
            "test_rows_used_in_selection": 0,
            "test_rows_used_in_eval": 0,
            "strict_eval": "2016-01..2017-11 (t<275); t=275 label is 2018-01",
            "boundary_note": "P1 checkpoint included t=275; new analyses exclude it. "
                             "Boundary-included headline is reported for continuity only.",
            "test_files_present_but_unused": [
                "framework/cache/graph_test.npy",
                "test rows inside features_*.parquet and panel_*.parquet",
            ],
        },
        "commands": [
            "python -m framework.targets.run (T1 redesign)",
            "python -m framework.windows.run (T2 windows)",
            "python -m framework.synth_arms.run (synthetic arms)",
            "python -m framework.interpret.run (interpretability)",
            "python -m framework.audit.run (leakage audit)",
            "python -m framework.reporting.run (report/manifest)",
        ],
        "environment": {"cwd": REPO_ROOT, "sqlite_free": True,
                        "no_torch": True, "no_pyg_dgl": True},
    }


def _report(cfg, t1r, t2w, syn, interp, head, audit, manifest) -> str:
    L = []
    add = L.append
    add("# Framework pilot — extended report (T1 redesign, T2 windows, synthetic arms, interpretability)")
    add("")
    add(f"Generated: {manifest['created_utc']}  ·  host: `{manifest['host']['platform']}`  ·  "
        f"runtime: {manifest['runtime_seconds']:.1f}s  ·  CPU only")
    add("")
    add("## 0. Verdict (read this first)")
    add("")
    add("- **T1 redesign: viable, with a caveat.** The old `next_active` target is at the "
        "MDE (redesigned model: gain/MDE = 1.00; the P1 saved model's strict gain is 0.0060, "
        "CI 0.0013–0.0119, also below MDE 0.0101) and is kept only as a sanity check. `high_q75` (event count above the train q75) is powered: "
        "ΔLL = 0.0551, MDE = 0.0209 (2.64×), AUC 0.972. But its discrimination is mostly "
        "**between-actor** (between-actor AUC 0.999; within-actor AUC 0.615). Pair it with "
        "the `decrease` target (ΔLL 0.1039, 5.10× MDE, within-actor AUC 0.787) if the claim is "
        "about month-to-month dynamics.")
    add("- **T2 windows: W_c and W_d are supported; W_a is not identifiable; W_b is fragile; "
        "inductive dyads fail.** W_a has 0 rows (the actor universe is fixed at 196). W_b has "
        "65 positives (AUC 0.694). W_c has 50,437 rows / 4,304 positives (AUC 0.725, gain 0.0215, "
        "7.2× MDE). W_d has 520,231 rows / 62,125 positives (AUC 0.919, gain 0.0117, 12.3× MDE). "
        "On correctly defined inductive dyads the model's AUC is **0.393**, i.e. worse than a "
        "constant; the apparent gain vs per-dyad Markov is calibration, not signal.")
    add("- **Synthetic arms: negative on T2, null/ambiguous on T1.** T2 augmentation hurts "
        "monotonically with dose (w=0.1: ΔLL −0.0009; w=0.3: −0.0024; w=1.0: −0.0077, all CIs "
        "exclude 0). Synthetic-only is −0.0409. T1 point estimates are positive but CIs cross 0, "
        "and the random-event control helps as much as the mechanism arm — no evidence of "
        "mechanism-specific transfer.")
    add("- **The biggest honest finding: T2 is a propensity model, not a month-level forecast.** "
        "Global AUC 0.909, but within-dyad AUC 0.482 (below chance); HGB 0.462. The P1 headline "
        "must be phrased as “which dyads are active-prone and why”, not “when a dyad will activate”. "
        "This is exactly the setting where Option A wording (per-actor/per-dyad explanations of "
        "dyad forecasts) is more defensible than literal node-level forecasting.")
    add("")
    add("## 1. Protocol and what changed")
    add("")
    add("| item | P1 checkpoint | this run |")
    add("|---|---|---|")
    add("| train / select / eval | <2015 / 2015 / 2016–2017 | same |")
    add("| eval boundary | t=275 included (label 2018-01 = test) | **t=275 dropped** (strict); boundary-included reported for continuity |")
    add("| T1 target | `out_active_next` only | 7 targets incl. high-activity, new dyad, increase/decrease |")
    add("| T2 windows | `t2.windows = {}` (never written) | `parts/t2_windows.json` + D2 table |")
    add("| D2 masks | never-active dyads (first=1e9) counted as onset; future re-entry counted | first ≤ t and re-entry ≤ t enforced |")
    add("| synthetic cache | smoke-scale 38 actors × 40 months | full-scale 110 actors × 120 months, separate cache/tag |")
    add("| synthetic arms | config only, not wired | 7 arms + dose-response, real select calibration |")
    add("| baseline choice | strongest on eval LL | pre-registered per-actor/dyad Markov; no eval-based selection |")
    add("")
    add("The strict eval is 4,508 actor-months and 607,982 dyad-months. The one-month boundary "
        "is small (4.2% of rows) and does not change any ranking, but it is a real test-label read "
        "in P1 and is excluded everywhere below.")
    add("")
    add("## 2. T1 redesign")
    add("")
    add("All targets use the same 47 actor-month features, train <2015, select 2015, eval 2016–17 "
        "(strict). Baselines: global rate, per-actor rate (train/select), per-actor Markov on the "
        "lagged label. The primary baseline is the pre-registered per-actor Markov; the model is "
        "the uncalibrated L2 logistic. `gain/MDE` uses the paired actor-cluster bootstrap.")
    add("")
    add("| target | rows | pos | clusters | n_eff(outcome) | base LL | model LL | ΔLL | 95% CI | MDE | Δ/MDE | AUC | within-actor AUC | ECE | powered |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|")
    for t, r in t1r["targets"].items():
        s = r["support"]; b = r["baselines"][r["primary_baseline"]]
        m = r["models"]["logistic_full"]["metrics"]; g = r["primary_gain"]
        ci = g["d_log_loss_ci"]
        add(f"| `{t}` | {s['rows']} | {s['positives']} | {s['clusters']} | "
            f"{s['n_eff_outcome']['n_eff']:.0f} | {b['log_loss']:.4f} | {m['log_loss']:.4f} | "
            f"{g['point']['d_log_loss']:.4f} | "
            f"{'[' + format(ci[0], '.4f') + ', ' + format(ci[1], '.4f') + ']' if ci else '—'} | "
            f"{g['mde']['mde']:.4f} | {r['gain_over_mde']:.2f} | {m['auc']:.4f} | "
            f"{s['within_entity_auc']['within_auc']:.3f} | {m['ece']:.4f} | "
            f"{'yes' if r['adequately_powered'] else 'no'} |")
    add("")
    add("**Reading the table.** `high_q50/q75/q90` are powered in log-loss but the within-actor "
        "AUC (0.59–0.61) is much lower than the global AUC (0.97–0.99); the model mostly learns "
        "which actors are high-activity, not which month. `increase`/`decrease` are the opposite: "
        "global AUC 0.74/0.79 and within-actor AUC 0.74/0.79, so they capture genuine month-to-month "
        "mean reversion. `new_dyad` has a large LL gain vs the Markov baseline (0.1198, 4.6× MDE) "
        "but AUC 0.527 and within-actor 0.502; versus the select-prevalence per-actor baseline the "
        "gain is only 0.0180 (CI [0.0072, 0.0289]), below the 0.0263 MDE. It is a calibration/"
        "trend effect and is not powered under the discrimination criterion. The redesigned `next_active` model's gain is 0.0104 "
        "(CI [0.0049, 0.0173]) vs MDE 0.0104 — exactly at the MDE, so not a meaningful effect. "
        "The P1 saved model's strict gain is 0.0060 (CI [0.0013, 0.0119]), even further below. "
        "HGB is consistently close but not better than logistic on the powered targets "
        "(high_q75 0.1761 vs 0.1737; high_q50 0.2301 vs 0.2270; high_q90 0.0736 vs 0.0719; "
        "decrease 0.5389 vs 0.5417), so capacity is not the constraint.")
    add("")
    add("**Recommendation.** Report `high_q75` as the actor activity-level target, `decrease` "
        "as the within-actor dynamic target, and `new_dyad` only as a negative/weak-discrimination "
        "result. Do **not** describe `high_q75` as precise month-level forecasting. Option A wording "
        "is still the right wrapper for the paper: actor cards explain *why a dyad forecast moves*, "
        "not “actor i will be active next month”. "
        "**Decision:** Option A is *not* a power fallback for T1 (the extended targets are "
        "powered). It is the correct wording for the T2 headline because within-dyad AUC is "
        "0.482. If a single node-level forecasting claim is required, use `decrease` "
        "(within-actor AUC 0.787), not `high_q75`.")
    add("")
    add("## 3. T2 windows (D2 definitions)")
    add("")
    add("Definitions are the frozen D2 ones, applied to the **validation** eval window (2016–17) "
        "because the test period is frozen. `b`/`c` are trailing 12-month windows; `d` is the "
        "stable control; `inductive` requires the dyad's first activation in the eval period.")
    add("")
    add("| window | rows | positives | dyads | pos rate | n_eff(diff) | LL | Brier | ECE | AUC | AP | base LL | ΔLL vs per-dyad | 95% CI | MDE | Δ/MDE | within-dyad AUC | status |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|")
    for r in t2w["table"]:
        wn = r["window"]; w = t2w["windows"][wn]
        if not w.get("models"):
            add(f"| `{wn}` | 0 | 0 | 0 | — | — | — | — | — | — | — | — | — | — | — | — | — | empty |")
            continue
        m = w["models"]["logistic_full_isotonic"]
        within = m["within_entity_auc"]["within_auc"]
        ci = r.get("gain_ci")
        g0 = w.get("gains_vs_logistic_full_isotonic", {}).get("per_dyad_markov", {})
        neff = (g0.get("mde") or {}).get("n_eff")
        add(f"| `{wn}` | {r['rows']} | {r['positives']} | {r['dyads']} | {r['pos_rate']:.4f} | "
            f"{'—' if neff is None else format(neff, '.0f')} | "
            f"{m['log_loss']:.4f} | {m['brier']:.4f} | {m['ece']:.4f} | {m['auc']:.4f} | "
            f"{m['ap']:.4f} | {r['base_ll']:.4f} | {r['gain']:.4f} | "
            f"{'[' + format(ci[0], '.4f') + ', ' + format(ci[1], '.4f') + ']' if ci else '—'} | "
            f"{r['mde']:.4f} | {r['mde_ratio']:.2f} | {within:.3f} | {w['status']} |")
    add("")
    add("**D2 comparability.** The support scales match the frozen D2 audit on the test period: "
        "D2 `c_dormancy_reentry_12m` has 192,997 rows / 12,309 dyads / 9.0% positive, while this "
        "validation c-window has 50,437 / 5,871 / 8.5%; D2 `b_dyad_onset_12m` has 10,625 rows / "
        "996 dyads / 1.55%, vs 4,314 / 465 / 1.51% here; D2 `d_stable_control` has 1,347,021 / "
        "26,109 / 10.8%, vs 520,231 / 25,121 / 11.9% here. The validation window is ~1/4 the "
        "test size, with the same positive rates, so the D2 definitions are applied consistently.")
    add("")
    add("**W_a**: 0 rows. The 196-actor roster is fixed; no actor is first seen after 2015. This "
        "window cannot be evaluated on this panel and must not appear as an empirical result.")
    add("")
    add("**W_b**: 4,314 rows / 65 positives / 465 dyads. The model's AUC is 0.694 and AP 0.042 "
        "(base rate 0.015, lift 2.8×). The 0.0947 gain vs per-dyad Markov is mostly the Markov "
        "baseline predicting ~11% for unseen dyads; vs a select-prevalence constant the gain is "
        "0.0928 but still reflects conditioning on “new dyad”. Treat as secondary/fragile.")
    add("")
    add("**W_c**: 50,437 rows / 4,304 positives / 5,871 dyads. AUC 0.725, AP 0.205, ΔLL 0.0215 "
        "(7.2× MDE). Within-dyad AUC 0.402 — again, the model ranks dyads, not months.")
    add("")
    add("**W_d**: 520,231 rows / 62,125 positives. AUC 0.919, AP 0.691, ΔLL 0.0117 (12.3× MDE). "
        "Within-dyad AUC 0.483.")
    add("")
    add("**Inductive dyads**: 11,523 rows / 568 positives / 501 dyads. AUC **0.393**, AP 0.045, "
        "within-dyad 0.356. Versus `global_rate_train` the gain is **−0.0192** (CI −0.0228, −0.0158): "
        "the model is worse than a constant on dyads with no history. The 0.1579 gain vs per-dyad "
        "Markov is the Markov baseline's catastrophic miscalibration (LL 0.397), not model skill. "
        "The graph-probe “+2.61% on inductive dyads” claim must be re-run under this corrected "
        "definition before it can be used.")
    add("")
    add("## 4. T2 headline under the strict boundary")
    add("")
    h = head
    add("| model | n | pos | LL | Brier | ECE | AUC | AP | within-dyad AUC | between-dyad AUC |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, lab in (("p_logistic_full_isotonic", "logistic_full_isotonic"),
                     ("p_hgb_full", "hgb_full"),
                     ("p_base_per_dyad_markov", "per_dyad_markov")):
        m = h["models"][key]["metrics"]; w = h["models"][key]["within_entity"]
        add(f"| `{lab}` | {m['n']} | {m['positives']} | {m['log_loss']:.5f} | {m['brier']:.5f} | "
            f"{m['ece']:.5f} | {m['auc']:.4f} | {m['ap']:.4f} | {w['within_auc']:.4f} | "
            f"{w['between_auc']:.4f} |")
    g = h["gain_vs_per_dyad"]
    add("")
    add(f"Paired gain of logistic_full_isotonic vs per_dyad_markov: **{g['point']['d_log_loss']:.5f}** "
        f"(95% CI {g['d_log_loss_ci']}; MDE {g['mde']['mde']:.5f}; {g['point']['d_log_loss']/g['mde']['mde']:.1f}× MDE). "
        "The CI excludes zero, but within-dyad AUC is 0.482. The correct claim is a **dyad-level "
        "propensity/calibration gain**, not temporal forecasting.")
    add("")
    add("Checkpoint continuity: boundary-included T2 isotonic LL 0.20345, AUC 0.90874; strict "
        "LL 0.20328, AUC 0.90906. The 4.2% boundary rows do not change conclusions.")
    add("")
    add("## 5. Synthetic arms")
    add("")
    add("The synthetic corpus uses real-train moments only, 110 actors × 120 months, one world per "
        "scheme, stored under `framework/cache/synth_full` with tags `_synthfull_*`. `synth_weight` "
        "is a total-weight ratio: synthetic rows get `w * n_real / n_synth`, so w=0.3 means 23% "
        "effective weight. C is selected once on real select (2015) and held fixed; isotonic "
        "calibration is fit on real select for every arm.")
    add("")
    add("### T2 (base-29 features, 6.34M real train rows)")
    add("")
    add("| arm | n_train | synth share | LL | Brier | ECE | AUC | ΔLL vs A0 | 95% CI | ΔECE | Δ/MDE |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    for arm, b in syn["t2"]["arms"].items():
        iso = b["isotonic"]; m = iso["metrics"]; g = iso["vs_A0"]
        ci = g["d_log_loss_ci"]
        add(f"| `{arm}` | {b.get('n_train', 0):,} | {b.get('synth_share', 0):.3f} | "
            f"{m['log_loss']:.5f} | {m['brier']:.5f} | {m['ece']:.5f} | {m['auc']:.4f} | "
            f"{g['point']['d_log_loss']:.5f} | "
            f"{'[' + format(ci[0], '.5f') + ', ' + format(ci[1], '.5f') + ']' if ci else '—'} | "
            f"{g['point']['d_ece']:.5f} | "
            f"{(g['point']['d_log_loss']/g['mde']['mde'] if g['mde']['mde'] else 0.0):.2f} |")
    add("")
    add("Dose–response (base scheme):")
    add("")
    add("| w_total | synth share | LL | AUC | ΔLL | 95% CI | ΔECE |")
    add("|---:|---:|---:|---:|---:|---|---:|")
    for d in syn["t2"]["dose_response"]:
        ci = d["d_log_loss_ci"]
        add(f"| {d['w_total']} | {d['synth_share']:.3f} | {d['log_loss']:.5f} | {d['auc']:.4f} | "
            f"{d['d_log_loss']:.5f} | {'[' + format(ci[0], '.5f') + ', ' + format(ci[1], '.5f') + ']' if ci else '—'} | "
            f"{d['d_ece']:.5f} |")
    add("")
    add("The dose–response is monotone negative. `A7_real_volume_matched` (240k real rows) is "
        "LL 0.20495 vs A0 0.20479, so extra real volume beyond 240k adds almost nothing; the "
        "synthetic failure is not a volume effect. `A5_random_events` and `A6_shuffled_kernels` "
        "also hurt, so there is no mechanism-specific transfer. ΔECE CIs are in "
        "`tables/synth_arms_t2.csv`; for A3 the ΔECE is +0.0043 (95% CI excludes 0), i.e. the "
        "augmented model is *better* calibrated but worse in log-loss — extra calibration is not "
        "useful when the ranking of dyads is unchanged. T2 ΔECE (95% CI): A3 +0.00435 "
        "[0.00410, 0.00461], A5 +0.00170 [0.00147, 0.00190], A6 +0.00275 [0.00253, 0.00299], "
        "A8 +0.00644 [0.00519, 0.00733]. T1 ΔECE CIs all cross 0 (e.g. A3 +0.00382 "
        "[-0.00234, 0.00967]).")
    add("")
    add("### T1 (next_active, 47k real actor-months)")
    add("")
    add("| arm | n_train | synth share | LL | ECE | AUC | ΔLL vs A0 | 95% CI | ΔECE |")
    add("|---|---:|---:|---:|---:|---:|---:|---|---:|")
    for arm, b in syn["t1"]["arms"].items():
        iso = b["isotonic"]; m = iso["metrics"]; g = iso["vs_A0"]
        ci = g["d_log_loss_ci"]
        add(f"| `{arm}` | {b.get('n_train', 0):,} | {b.get('synth_share', 0):.3f} | "
            f"{m['log_loss']:.5f} | {m['ece']:.5f} | {m['auc']:.4f} | "
            f"{g['point']['d_log_loss']:.5f} | "
            f"{'[' + format(ci[0], '.5f') + ', ' + format(ci[1], '.5f') + ']' if ci else '—'} | "
            f"{g['point']['d_ece']:.5f} |")
    add("")
    add("T1 point estimates for A3/A5/A6/A8 are positive (0.007–0.026) but every CI crosses zero. "
        "A5 (random events, a negative control) has the largest point gain (0.0259), which is the "
        "opposite of mechanism-specific transfer. Synthetic-only is clearly harmful (−0.0312).")
    add("")
    add("## 6. Interpretability")
    add("")
    add(f"T1 primary (high_q75): LL {interp['t1_metrics']['log_loss']:.4f}, ECE "
        f"{interp['t1_metrics']['ece']:.4f}, AUC {interp['t1_metrics']['auc']:.4f}. "
        f"T2 refit: LL {interp['t2_metrics']['log_loss']:.4f}, ECE {interp['t2_metrics']['ece']:.4f}, "
        f"AUC {interp['t2_metrics']['auc']:.4f} (n={interp['t2_metrics']['n']:,}).")
    add("")
    add("### Node cards (196 actors)")
    add("")
    nc = interp["node_cards"]
    add(f"- {nc['n_actors']} actor cards; mean per-card LL {nc['mean_card_ll']:.4f}.")
    add("- Only 73/196 actors have both classes in eval; median per-actor AUC 0.617, 18 actors "
        "below 0.5. The worst cards are low-activity actors (VAT LL 0.920, CZE 0.789, MEX 0.763); "
        "the model over-predicts their rare high-activity months.")
    add("- Top group contributions (mean |log-odds|): actor_rolling 3.43, actor_current 1.61, "
        "actor_lags 1.51, actor_degree 0.46, calendar 0.41, region 0.20, actor_history 0.14.")
    add("- Top permutation importance: `oes12` 0.278 (12-month event sum), `log_out_ev` 0.032, "
        "`oes6` 0.025, `log_tot_ev` 0.020. The cards are dominated by activity level, consistent "
        "with the between-actor AUC result.")
    add("")
    add("### Deletion / insertion")
    add("")
    add("| target | k | deletion Δp | deletion ΔLL | insertion Δp | insertion LL | insertion gain vs mean |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for r in interp["deletion_insertion_t1"]:
        add(f"| T1 high_q75 | {r['k']} | {r['deletion_mean_dp']:+.4f} | {r['deletion_dll']:+.4f} | "
            f"{r['insertion_mean_dp']:+.4f} | {r['insertion_ll']:.4f} | {r['insertion_gain_vs_mean']:+.4f} |")
    for r in interp["deletion_insertion_t2"]:
        add(f"| T2 | {r['k']} | {r['deletion_mean_dp']:+.4f} | {r['deletion_dll']:+.4f} | "
            f"{r['insertion_mean_dp']:+.4f} | {r['insertion_ll']:.4f} | {r['insertion_gain_vs_mean']:+.4f} |")
    add("")
    add("Deleting the top-k features (set to the train mean) degrades T1 LL from 0.174 to 0.264 "
        "(k=1) and 0.928 (k=10); T2 from 0.204 to 0.232 (k=1) and 0.301 (k=10). The curves are "
        "steep, so the top features are not decorative. Insertion from the mean recovers most of "
        "the T1 LL by k=3 (0.161) and T2 by k=10 (0.213 vs mean baseline 0.382).")
    add("")
    add("### Counterfactuals")
    add("")
    add("| target | scenario | mean Δp | median Δp | share down |")
    add("|---|---|---:|---:|---:|")
    for r in interp["counterfactual_t1"]:
        add(f"| T1 high_q75 | {r['scenario']} | {r['mean_dp']:+.4f} | {r['median_dp']:+.4f} | {r['share_down']:.3f} |")
    for r in interp["counterfactual_t2"]:
        add(f"| T2 | {r['scenario']} | {r['mean_dp']:+.4f} | {r['median_dp']:+.4f} | {r['share_down']:.3f} |")
    add("")
    add("T1: removing last month's events lowers high-activity probability by 0.021 on average "
        "(72% of rows down); a full cold-start history lowers it by 0.143. T2: removing all dyad "
        "history lowers p by 0.073, but removing just the last 3 months has a near-zero mean "
        "(+0.0015) because the isotonic-calibrated model is dominated by the dyad's long-run rate.")
    add("")
    add("### Edge attribution")
    add("")
    ep = interp["edge_probe"]
    add(f"60 dyad-months, one common neighbor removed, binary graph features recomputed "
        f"(degrees, weighted degree, cn, cn12, aa/ra, nra, jaccard, two-hop). Mean Δp = "
        f"{ep['mean_dp_remove_common']:+.4f}; a degree-matched non-common neighbor gives "
        f"{ep['mean_dp_remove_noncommon']:+.4f}. The difference is only +0.0037 (2.5% of the "
        f"effect), so the model does **not** localize to the specific shared neighbor: the "
        f"sensitivity is generic degree/weight sensitivity.")
    add("")
    add("### Failure cases and human audit")
    add("")
    add("- `tables/failure_cases_t1.csv` / `_t2.csv`: top-25 highest log-loss rows. T1 failures are "
        "mostly low-activity actors with a surprise high month; T2 failures are rare activations "
        "of low-propensity dyads.")
    add(f"- Audit sample: `audit/sample_t1.csv` ({interp['audit_materials']['sample_t1_n']} items), "
        f"`audit/sample_t2.csv` ({interp['audit_materials']['sample_t2_n']} items), stratified by "
        "outcome × predicted-probability quintile.")
    add("- `audit/checklist.md`: 5 dimensions (availability, direction, magnitude, completeness, "
        "actionability), 1–5 scale, availability stop rule.")
    add("- `audit/kappa_plan.json`: 2 raters × 120 items, 20 duplicates, quadratic-weighted kappa, "
        "target 0.60, third-rater adjudication.")
    add("")
    add("## 7. Leakage audit")
    add("")
    add(f"All checks passed: **{audit['all_passed']}**. Key facts:")
    add("")
    add("- `allow_test=false`; no new module reads `split='test'`; `pipeline.py`'s test export is "
        "behind `if allow_test:` and was not executed.")
    add("- Feature manifest has no labels; every feature is tagged `available_at='t'`.")
    add("- Actor/dyad caches contain test rows, but every new loader filters to train/val. "
        "`graph_test.npy` is present and unused.")
    add("- Synthetic moments: `source='real train split only'`, `n_train_rows=6,344,160`; arms "
        "load only the synthetic train split.")
    add(f"- Boundary: {audit['eval_boundary']['rows_with_t_plus_1_at_or_after_test']:,} "
        f"T2 rows (t=275) have their label in test; strict keeps "
        f"{audit['eval_boundary']['t2_strict_rows']:,}. T1 drops 196 rows.")
    add("")
    add("## 8. Limitations and blockers")
    add("")
    add("1. **Within-entity AUC failure** is the main scientific blocker for a forecasting claim. "
        "The model is a propensity/calibration model. Either reframe (Option A) or add a "
        "within-dyad dynamics model (fixed effects, hazard, or transition-specific features).")
    add("2. **W_a is unidentifiable** on a fixed 196-actor roster. Do not report a new-actor window.")
    add("3. **W_b has 65 positives**; any claim is fragile. The AUC 0.694 is the only usable signal.")
    add("4. **Inductive dyads fail** (AUC 0.393). The graph-probe inductive claim (+2.61%) is not "
        "confirmed under the corrected definition and should be treated as unresolved.")
    add("5. **Synthetic generator is misspecified**: active_next rate 28.7% vs real 11.1%; "
        "synthetic-only LL is 0.246 vs real 0.205. Negative T2 transfer is therefore expected and "
        "does not prove that a calibrated simulator would fail.")
    add("6. **T1 high_q75 is mostly between-actor**; the powered result is about activity level, "
        "not month-level timing. Pair with decrease/increase for dynamics.")
    add("7. No Windows host was needed; everything ran CPU-only on Linux. No GPU/PyG/DGL/torch.")
    add("")
    add("## 9. Next 3 days")
    add("")
    add("**Day 1 — lock the claim and the metric.** (i) Pre-register within-entity AUC as a "
        "co-primary metric; freeze the strict t<275 boundary in the protocol. (ii) Decide Option A "
        "vs a dynamics model: run a dyad fixed-effects / conditional-logit check on T2 and an "
        "actor fixed-effects check on T1 high_q75. (iii) Re-run the graph probe under the corrected "
        "inductive mask; report AUC and within-dyad AUC, not only ΔLL. (iv) Freeze `summary.json` "
        "and the manifest hashes.")
    add("")
    add("**Day 2 — repair or retire the weak arms.** (i) If the dynamics check shows within-dyad "
        "signal, add transition-specific features (burst/decay, spell age, hazard baselines) and "
        "re-run W_c/W_d. (ii) If not, retire the month-level claim and make the paper an "
        "explanation/propensity paper (Option A). (iii) Calibrate the synthetic generator to the "
        "real active rate and re-run only the dose-response; otherwise keep synthetic as an "
        "appendix negative control. (iv) Re-check W_b with a select-prevalence baseline and label "
        "it secondary.")
    add("")
    add("**Day 3 — human audit and writing.** (i) Run two raters on the 120-item samples; compute "
        "weighted kappa and adjudicate. (ii) Finalize 8–12 node cards for the paper (mix of good, "
        "median, failure), with the availability/direction checklist. (iii) Write the methods and "
        "limitations sections around the within-entity AUC result; do not bury it. (iv) Rebuild "
        "the manifest and archive the exact commands/versions; freeze the repo tag.")
    add("")
    add("## 10. Artefact index")
    add("")
    add("- `results/framework_pilot/parts/t1_redesign.json`, `t2_windows.json`, `synth_arms.json`, "
        "`interpretability.json`, `node_cards.json`")
    add("- `results/framework_pilot/tables/t1_redesign.csv`, `t2_windows.csv`, `synth_arms_t1.csv`, "
        "`synth_arms_t2.csv`, `synth_dose_response.csv`, `headline_strict.csv`, "
        "`node_cards_actor.csv`, `feature_attribution_t1.csv`, `feature_attribution_t2.csv`, "
        "`deletion_insertion_t1.csv`, `deletion_insertion_t2.csv`, `counterfactual_t1.csv`, "
        "`counterfactual_t2.csv`, `edge_attribution.csv`, `failure_cases_t1.csv`, `failure_cases_t2.csv`")
    add("- `results/framework_pilot/audit/`: checklist, samples, kappa plan, leakage audit JSON/MD")
    add("- `results/framework_pilot/report.md`, `summary.json`, `selection_log.json`, `run.log`, `manifest.json`")
    add("- Configuration files are kept in the private working tree and are not redistributed.")
    add("- `tests/test_smoke.py`")
    add("")
    return "\n".join(L) + "\n"


def run(cfg: dict) -> dict:
    t0 = time.time()
    outdir = cfg["paths"]["out_dir"]
    os.makedirs(outdir, exist_ok=True)
    t1r = _load(cfg, "t1_redesign")
    t2w = _load(cfg, "t2_windows")
    syn = _load(cfg, "synth_arms")
    interp = _load(cfg, "interpretability")
    audit_path = os.path.join(outdir, "audit", "framework_leakage_audit.json")
    if os.path.exists(audit_path):
        with open(audit_path, encoding="utf-8") as f:
            audit = json.load(f)
    else:
        from . import audit as audit_mod
        audit = audit_mod.run(cfg, audit_path)
        audit_mod.write_markdown(audit, os.path.join(outdir, "audit",
                                                     "framework_leakage_audit.md"))
    head = _strict_headline(cfg)

    summary = {
        "schema": "q2-framework-summary/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": {
            "t1_redesign": t1r["verdict"],
            "t1_primary_target": t1r["primary_target"],
            "t1_powered_targets": t1r["powered_targets"],
            "t2_headline": "dyad propensity/calibration gain, no within-dyad temporal skill",
            "t2_strict_gain_vs_per_dyad": head["gain_vs_per_dyad"]["point"]["d_log_loss"],
            "t2_within_dyad_auc": head["models"]["p_logistic_full_isotonic"]["within_entity"]["within_auc"],
            "synthetic_t2": "negative and monotone in dose",
            "synthetic_t1": "null/ambiguous; random control as strong as mechanism arm",
        },
        "t1_redesign": t1r,
        "t2_windows": t2w,
        "t2_headline_strict": head,
        "synthetic_arms": syn,
        "interpretability": interp,
        "leakage_audit": audit,
    }
    jdump(os.path.join(outdir, "summary.json"), summary)

    selection = {
        "schema": "q2-framework-selection-log/1",
        "test_used": False,
        "test_columns": [],
        "split": {"train": "<2015-01", "select": "2015", "eval": "2016-2017 strict"},
        "t1": {
            t: {
                "C": r["logistic_fit"]["C"],
                "select_curve": r["logistic_fit"]["select_curve"],
                "baseline_select_ll": r["baseline_select_ll"],
                "primary_baseline": r["primary_baseline"],
                "calibration_select_ll": r["calibration"],
            } for t, r in t1r["targets"].items()
        },
        "t2_windows": {
            "model": "saved P1 validation predictions; no refit",
            "test_used": False,
        },
        "synth_arms": {
            "t2_C_star": syn["t2"]["C_star"],
            "t1_C_star": syn["t1"]["C_star"],
            "selection": syn["protocol"]["selection"],
            "calibration": syn["protocol"]["calibration"],
            "test_used": False,
        },
    }
    jdump(os.path.join(outdir, "selection_log.json"), selection)

    table_paths = _write_tables(cfg, t1r, t2w, syn, interp, head)
    result_paths = sorted(set(
        table_paths +
        glob.glob(os.path.join(outdir, "tables", "*.csv")) +
        glob.glob(os.path.join(outdir, "audit", "*")) +
        [os.path.join(outdir, p) for p in
         ("summary.json", "selection_log.json", "report.md", "run.log", "manifest.json",
          "parts/t1_redesign.json", "parts/t2_windows.json", "parts/synth_arms.json",
          "parts/interpretability.json", "parts/node_cards.json",
          "parts/t1.json", "parts/t2.json")]
    ))
    stage_total = 0.0
    for name in ("t1_redesign", "t2_windows", "synth_arms", "interpretability"):
        stage_total += float(_load(cfg, name).get("runtime_seconds") or 0.0)
    manifest = _manifest(cfg, result_paths, elapsed(t0) + stage_total)
    manifest["reporting_runtime_seconds"] = elapsed(t0)
    manifest["stage_runtimes_seconds"] = {
        name: _load(cfg, name).get("runtime_seconds")
        for name in ("t1_redesign", "t2_windows", "synth_arms", "interpretability")
    }
    jdump(os.path.join(outdir, "manifest.json"), manifest)

    report = _report(cfg, t1r, t2w, syn, interp, head, audit, manifest)
    with open(os.path.join(outdir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)

    stages = []
    for name in ("t1_redesign", "t2_windows", "synth_arms", "interpretability"):
        parts = _load(cfg, name)
        stages.append((name, parts.get("runtime_seconds")))
    lines = [
        f"# framework pilot run log",
        f"generated_utc={manifest['created_utc']}",
        f"host={manifest['host']['hostname']} platform={manifest['host']['platform']}",
        f"python={manifest['host']['python']} cpus={manifest['host']['cpus']}",
        f"packages={manifest['packages']}",
        "",
        "## stage runtimes (seconds)",
    ]
    for name, sec in stages:
        lines.append(f"{name:20s} {sec}")
    lines += [
        "",
        "## protocol",
        "train <2015-01; select 2015; eval 2016-01..2017-12 strict (t<275)",
        "test 2018-01..2023-04 never read; allow_test=false",
        "synthetic corpus: framework/cache/synth_full, real-train moments only",
        "",
        "## artefacts",
    ]
    for fp in result_paths:
        if os.path.exists(fp) and os.path.basename(fp) != "manifest.json":
            lines.append(f"{os.path.relpath(fp, REPO_ROOT)}  {sha256_file(fp)}")
    with open(os.path.join(outdir, "run.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    # Add post-hoc artefact hashes (report, run.log) and rewrite the manifest.
    for fn in ("report.md", "run.log"):
        fp = os.path.join(outdir, fn)
        manifest["result_sha256"][os.path.relpath(fp, REPO_ROOT)] = sha256_file(fp)
    manifest["result_sha256"].pop(
        os.path.relpath(os.path.join(outdir, "manifest.json"), REPO_ROOT), None)
    jdump(os.path.join(outdir, "manifest.json"), manifest)
    return {"summary": summary, "manifest": manifest, "report": report,
            "runtime_seconds": elapsed(t0)}
