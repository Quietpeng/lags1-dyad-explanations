"""Sprint B.1 - exact additive decomposition and actor-level explanation tables.

Outputs (results/framework_pilot/stability):
  tables/actor_{source,target}_{period}.csv         per-actor attribution
  tables/actor_month_{source,target}_{period}.csv   per-actor-month attribution
  tables/feature_attribution_periods.csv            exact feature contributions
  tables/bucket_shares.csv                          7-bucket contract shares
  tables/exactness_check.csv                        sum == raw log-odds
  tables/calibration_sensitivity.csv                raw/temperature/isotonic
  tables/collinearity.csv                           VIF / max |r| / seed coef SD
  cache/agg_{point,rep0..4}_{period}.npz            streaming aggregates
  cache/zbar_{period}.npz                           actor mean standardised features
  cache/coefs.npz                                   point + 5 replicate coefficients
  summary_decompose.json

Row-level dyad samples, labels and prediction columns are **not** written: the
public release excludes row-level data (see `DATA_EXCLUSIONS.md`). The
contribution cache is restricted and is not redistributed.

The exact track is raw log-odds or temperature scaling (affine).  Isotonic is
reported only as a sensitivity check and is never decomposed.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

import sprint_b_common as C
from framework import aggregation as A
from framework import models
from framework.config import load_config
from framework.metrics import binary_metrics
from framework.utils import elapsed, jdump, jload, log


def _period_source(b: dict, p: dict) -> dict:
    src = b["va"] if p["lo"] < 276 else b["te"]
    return C.split_rows(src, p["lo"], p["hi"])


def _windows(b: dict, rows: dict, p: dict) -> tuple[np.ndarray | None, list]:
    if p["name"] == "covid_2020_21":
        return None, []
    masks = A.d2_windows(b["panel"], rows, p["lo"], p["hi"])
    wid, names = A.window_id_from_masks(masks)
    return wid, names


def main() -> None:
    rl = C.RunLog("sprint_b_decompose")
    t_start = time.time()
    cfg = load_config()
    C.ensure_dirs()
    b = C.load_splits(cfg)
    actors = b["actors"]
    n_actors = len(actors)
    names = b["tr"]["feature_names"]
    A.feature_buckets(names)               # fail loudly if the contract drifts
    log("selfcheck: " + json.dumps(A.selfcheck()))

    point = C.fit_point(cfg, b)
    fit0 = point["fit"]
    T = float(point["T"])

    # ---- period rows and D2 windows -------------------------------------
    rows, wids, wnames = {}, {}, {}
    for p in C.PERIODS:
        rows[p["name"]] = _period_source(b, p)
        wids[p["name"]], wnames[p["name"]] = _windows(b, rows[p["name"]], p)
        d = rows[p["name"]]
        log(f"period {p['name']}: rows={len(d['y']):,} pos={int(d['y'].sum()):,} "
            f"months={p['lo']}..{p['hi']} windows={wnames[p['name']]}")
        if wids[p["name"]] is not None:
            cnt = np.bincount(wids[p["name"]][wids[p["name"]] >= 0],
                              minlength=len(wnames[p["name"]]))
            log(f"  window rows: {dict(zip(wnames[p['name']], cnt.tolist()))}")

    # ---- point-model pass ------------------------------------------------
    coefs = np.zeros((1 + C.N_SEEDS, len(names)))
    intercepts = np.zeros(1 + C.N_SEEDS)
    coefs[0] = fit0["coef"]
    intercepts[0] = fit0["intercept"]
    accs, metrics = {}, {}
    inner_C = None
    for p in C.PERIODS:
        d = rows[p["name"]]
        nw = len(wnames[p["name"]])
        acc = A.AttributionAccumulator(n_actors, p["lo"], p["hi"], names,
                                       n_windows=nw, do_occlusion=True)
        store = None
        if p["name"] == "inner_2016_17":
            store = np.empty((len(d["y"]), len(names)), dtype=np.float32)
        t0 = time.time()
        C.run_period(acc, d, fit0, T, use_z=True, store_C=store,
                     window_id=wids[p["name"]])
        log(f"pass point {p['name']}: {elapsed(t0)}s exact_max="
            f"{acc.exact['max_abs_error']:.2e} bad={acc.exact['rows_above_tol']}")
        accs[p["name"]] = acc
        if store is not None:
            inner_C = store

        # metrics on the same rows
        p_raw = models.predict_logistic(fit0, d["X"])
        p_T = models.apply_temperature(p_raw, T) if T != 1.0 else p_raw
        p_iso = point["iso"].predict(p_raw)
        m_raw = C.metrics_block(d["y"], p_raw)
        m_T = C.metrics_block(d["y"], p_T)
        m_iso = C.metrics_block(d["y"], p_iso)
        dist = float(np.mean(np.abs(A.logit(p_iso) - A.logit(p_T))))
        metrics[p["name"]] = {
            "n": int(len(d["y"])), "positives": int(d["y"].sum()),
            "pos_rate": round(float(d["y"].mean()), 6),
            "log_loss_raw": m_raw["log_loss"], "log_loss_temperature": m_T["log_loss"],
            "log_loss_isotonic": m_iso["log_loss"],
            "ece_raw": m_raw["ece"], "ece_temperature": m_T["ece"],
            "ece_isotonic": m_iso["ece"],
            "auc": m_raw["auc"], "ap": m_raw["ap"],
            "mean_abs_isotonic_logit_distortion": round(dist, 6),
            "temperature_T": T,
        }
        log(f"metrics {p['name']}: LL raw={m_raw['log_loss']:.5f} "
            f"T={m_T['log_loss']:.5f} iso={m_iso['log_loss']:.5f} "
            f"ECE raw={m_raw['ece']:.5f}")

        meta = {"model": "point", "period": p["name"], "lo": p["lo"], "hi": p["hi"],
                "T": T, "window_names": wnames[p["name"]],
                "exact": acc.exact, "n_rows": acc.n_rows,
                "source": "inner_validation" if p["name"] == "inner_2016_17"
                else "burned_retrospective"}
        C.save_agg(acc, os.path.join(C.AGG, f"agg_point_{p['name']}.npz"), meta)

    # exactness on a random sample against the model's own probability
    exact_rows = []
    rng = np.random.default_rng(C.SEED_BASE)
    for p in C.PERIODS:
        d = rows[p["name"]]
        idx = np.sort(rng.choice(len(d["y"]), size=min(50000, len(d["y"])),
                                 replace=False))
        Csam = A.contributions(fit0, d["X"][idx])
        p_raw = models.predict_logistic(fit0, d["X"][idx])
        p_T = models.apply_temperature(p_raw, T) if T != 1.0 else p_raw
        eta_sk = fit0["model"].decision_function(
            models.scaler_apply(d["X"][idx], fit0["mu"], fit0["sd"])).astype(np.float64) / float(T)
        fitT = {**fit0, "intercept": float(fit0["intercept"]) / T}
        rep = A.exactness_report(fitT, A.contributions_temperature(Csam, T), p_T,
                                 T=1.0, tol=1e-4, eta=eta_sk)
        rep.update({"period": p["name"], "sample_rows": len(idx),
                    "accumulator_max_abs_error": accs[p["name"]].exact["max_abs_error"],
                    "accumulator_rows_above_tol": accs[p["name"]].exact["rows_above_tol"],
                    "accumulator_tol": 1e-6})
        exact_rows.append(rep)
        log(f"exactness {p['name']}: sample max={rep['raw_max_abs_error']:.2e} "
            f"decision max={rep.get('decision_max_abs_error', float('nan')):.2e} "
            f"full max={rep['accumulator_max_abs_error']:.2e}")
    exact_df = pd.DataFrame(exact_rows)
    exact_df.to_csv(os.path.join(C.TAB, "exactness_check.csv"), index=False)

    # ---- seed replicates -------------------------------------------------
    for rep in range(C.N_SEEDS):
        fit = C.fit_seed(cfg, b, point, rep)
        coefs[1 + rep] = fit["coef"]
        intercepts[1 + rep] = fit["intercept"]
        for p in C.PERIODS:
            d = rows[p["name"]]
            acc = A.AttributionAccumulator(n_actors, p["lo"], p["hi"], names,
                                           n_windows=len(wnames[p["name"]]),
                                           do_occlusion=False)
            t0 = time.time()
            C.run_period(acc, d, fit, T, use_z=False,
                         window_id=wids[p["name"]])
            log(f"pass rep{rep} {p['name']}: {elapsed(t0)}s exact_max="
                f"{acc.exact['max_abs_error']:.2e}")
            meta = {"model": f"rep{rep}", "period": p["name"], "lo": p["lo"],
                    "hi": p["hi"], "T": T, "window_names": wnames[p["name"]],
                    "exact": acc.exact, "n_rows": acc.n_rows,
                    "source": "inner_validation" if p["name"] == "inner_2016_17"
                    else "burned_retrospective"}
            C.save_agg(acc, os.path.join(C.AGG, f"agg_rep{rep}_{p['name']}.npz"), meta)

    np.savez_compressed(os.path.join(C.AGG, "coefs.npz"),
                        coefs=coefs, intercepts=intercepts, T=T,
                        C=float(fit0["C"]), names=np.array(names),
                        rep_seeds=np.array([C.SEED_BASE + 1009 * i
                                            for i in range(C.N_SEEDS)]))

    # ---- tables (point model) -------------------------------------------
    # Public export: aggregate attribution only. Drop predicted-probability and
    # observed-rate columns so a preview run cannot recreate the excluded schema.
    restricted = ("mean_p", "mean_y", "p_sum", "y_sum")

    def public_export(frame):
        return frame.drop(columns=[c for c in restricted if c in frame.columns])

    actor_frames = []
    for p in C.PERIODS:
        acc = accs[p["name"]]
        for role in ("source", "target"):
            df = A.actor_table(acc, role, actors, p["name"])
            actor_frames.append(df)
            public_export(df).to_csv(
                os.path.join(C.TAB, f"actor_{role}_{p['name']}.csv"), index=False)
            am = A.actor_month_table(acc, role, actors, p["name"])
            public_export(am).to_csv(
                os.path.join(C.TAB, f"actor_month_{role}_{p['name']}.csv"),
                index=False)
            log(f"actor table {role} {p['name']}: {len(df)} actors, "
                f"mean ego share={df['ego_abs_share'].mean():.3f}")
    public_export(pd.concat(actor_frames, ignore_index=True)).to_csv(
        os.path.join(C.TAB, "actor_attribution_all_periods.csv"), index=False)

    feat_frames, share_frames = [], []
    for p in C.PERIODS:
        acc = accs[p["name"]]
        ft = A.feature_table(acc, p["name"])
        feat_frames.append(ft)
        bs = A.bucket_share_table(acc, p["name"])
        share_frames.append(bs)
        log(f"bucket abs shares {p['name']}: " +
            ", ".join(f"{r.bucket}={r.abs_share:.3f}" for r in bs.itertuples()))
    pd.concat(feat_frames, ignore_index=True).to_csv(
        os.path.join(C.TAB, "feature_attribution_periods.csv"), index=False)
    pd.concat(share_frames, ignore_index=True).to_csv(
        os.path.join(C.TAB, "bucket_shares.csv"), index=False)
    A.bucket_map_frame(names).to_csv(os.path.join(C.TAB, "feature_bucket_map.csv"),
                                     index=False)

    # calibration sensitivity
    cal_rows = []
    for p in C.PERIODS:
        m = metrics[p["name"]]
        cal_rows.append({
            "model": "logistic_full", "period": p["name"], "status": p["status"],
            "n": m["n"], "positives": m["positives"], "T": m["temperature_T"],
            "log_loss_raw": m["log_loss_raw"],
            "log_loss_temperature": m["log_loss_temperature"],
            "log_loss_isotonic": m["log_loss_isotonic"],
            "ece_raw": m["ece_raw"], "ece_temperature": m["ece_temperature"],
            "ece_isotonic": m["ece_isotonic"],
            "mean_abs_isotonic_logit_distortion": m["mean_abs_isotonic_logit_distortion"],
            "exact_decomposition_applies_to": "raw log-odds and temperature only",
            "isotonic_decomposed": False,
        })
    pd.DataFrame(cal_rows).to_csv(
        os.path.join(C.TAB, "calibration_sensitivity.csv"), index=False)

    # zbar (mean standardised features per actor/role/period) for the exact
    # seed-variance and regime-shift decompositions
    for p in C.PERIODS:
        acc = accs[p["name"]]
        payload = {"meta": np.array(json.dumps({"period": p["name"], "lo": p["lo"],
                                                "hi": p["hi"], "T": T}))}
        for role in ("source", "target"):
            n = acc.feat_n[role]
            payload[f"z_{role}"] = (acc.z_sum[role]
                                    / np.maximum(n, 1)[:, None]).astype(np.float32)
            payload[f"n_{role}"] = n.astype(np.float32)
        np.savez_compressed(os.path.join(C.AGG, f"zbar_{p['name']}.npz"), **payload)

    # full inner contribution cache (restricted; never redistributed)
    d = rows["inner_2016_17"]
    p_raw = models.predict_logistic(fit0, d["X"])
    p_T = models.apply_temperature(p_raw, T) if T != 1.0 else p_raw
    rng = np.random.default_rng(C.SEED_BASE + 7)
    np.savez_compressed(
        os.path.join(C.CACHE, "contrib_inner_point.npz"),
        contribution=inner_C.astype(np.float32), y=d["y"].astype(np.int64),
        did=d["did"].astype(np.int64), sid=d["sid"].astype(np.int16),
        tid=d["tid"].astype(np.int16), t_idx=d["t_idx"].astype(np.int16),
        p=p_T.astype(np.float32), feature_names=np.array(names), T=T)
    log(f"full contribution cache {inner_C.nbytes / 1e6:.0f} MB; "
        "no row-level sample CSV is written")

    # ---- collinearity / coefficient stability ---------------------------
    valid = np.where(b["tr"]["t_idx"] != C.TRAIN_EMBARGO)[0]
    idx = rng.choice(valid, size=min(300000, len(valid)), replace=False)
    Z = A.standardized(fit0, b["tr"]["X"][idx])
    sd_z = Z.std(axis=0)
    const = sd_z < 1e-12
    R = np.eye(len(names))
    if (~const).sum() > 1:
        with np.errstate(invalid="ignore", divide="ignore"):
            Rc = np.corrcoef(Z[:, ~const], rowvar=False)
        Rc = np.nan_to_num(Rc, nan=0.0)
        np.fill_diagonal(Rc, 1.0)
        R[np.ix_(~const, ~const)] = Rc
    try:
        vif = np.diag(np.linalg.inv(R))
    except np.linalg.LinAlgError:
        vif = np.diag(np.linalg.pinv(R))
    vif = np.where(const, 1.0, vif)
    cond = float(np.linalg.cond(R))
    max_r, partner = np.zeros(len(names)), np.full(len(names), "", dtype=object)
    for j in range(len(names)):
        others = np.abs(R[j]).copy()
        others[j] = -1
        k = int(np.argmax(others))
        max_r[j] = float(R[j, k])
        partner[j] = names[k]
    coef_sd = coefs[1:].std(axis=0, ddof=1)
    flip = (np.sign(coefs[1:]) != np.sign(coefs[0])[None, :]).mean(axis=0)
    fb = A.feature_buckets(names)
    pd.DataFrame({
        "feature": names, "bucket": [fb[n] for n in names],
        "coef_point": np.round(coefs[0], 6),
        "coef_sd_reps": np.round(coef_sd, 6),
        "coef_sign_flip_rate": np.round(flip, 3),
        "vif": np.round(vif, 3), "max_abs_corr": np.round(np.abs(max_r), 4),
        "corr_partner": partner, "corr_signed": np.round(max_r, 4),
    }).sort_values("vif", ascending=False).to_csv(
        os.path.join(C.TAB, "collinearity.csv"), index=False)
    log(f"collinearity: condition number={cond:.1f} max VIF={vif.max():.1f}")

    # ---- P1 reference comparison ---------------------------------------
    p1_note = {"available": False}
    p1_path = os.path.join(cfg["paths"]["cache_dir"], "preds_t2.npz")
    if os.path.exists(p1_path):
        z = np.load(p1_path)
        m = (z["t_idx"] >= 252) & (z["t_idx"] <= 274)
        p1 = z["p_logistic_full"][m]
        if len(p1) == len(d["y"]):
            diff = p_T - p1
            p1_note = {
                "available": True, "n": int(len(p1)),
                "max_abs_diff": round(float(np.abs(diff).max()), 6),
                "mean_abs_diff": round(float(np.abs(diff).mean()), 6),
                "corr": round(float(np.corrcoef(p1, p_T)[0, 1]), 6),
                "log_loss_p1_preds": round(float(binary_metrics(d["y"], p1)["log_loss"]), 6),
                "log_loss_refit": round(float(binary_metrics(d["y"], p_T)["log_loss"]), 6),
                "note": ("P1 preds were produced before the current graph cache; "
                         "the refit is the consistent source for Sprint B"),
            }
    log(f"P1 reference: {p1_note}")

    # ---- summary ---------------------------------------------------------
    summary = {
        "sprint": "B",
        "task": "exact additive actor explanations + stability audit",
        "status": "provisional_holdout_not_integrated",
        "protocol": {
            "embargo": {"train": C.TRAIN_EMBARGO, "select": C.SELECT_EMBARGO,
                        "inner": 275, "retro": 335},
            "periods": C.PERIODS,
            "train_rows": int(len(b["tr"]["y"])),
            "train_rows_used": int((b["tr"]["t_idx"] != C.TRAIN_EMBARGO).sum()),
            "select_rows": int(len(C.select_rows(b["va"])["y"])),
            "n_actors": int(n_actors), "n_features": int(len(names)),
            "n_seed_replicates": int(C.N_SEEDS),
            "seed_semantics": ("dyad-cluster bootstrap refits with fixed C; the "
                               "lbfgs point fit is deterministic, so optimizer "
                               "seed noise is exactly zero"),
        },
        "model": {
            "name": "logistic_full", "C": float(fit0["C"]),
            "temperature_T": T,
            "select_log_loss": float(fit0["select_log_loss"]),
            "features": names,
            "coefs_top_abs": [
                {"feature": n, "coef": round(float(c), 5)}
                for n, c in sorted(zip(names, coefs[0]), key=lambda t: -abs(t[1]))[:20]],
        },
        "metrics": metrics,
        "exactness": exact_df.to_dict(orient="records"),
        "bucket_contract": {b: feats for b, feats in A.BUCKET_FEATURES.items()},
        "bucket_shares": pd.concat(share_frames, ignore_index=True).to_dict(orient="records"),
        "actor_headline": {
            p["name"]: {
                role: {
                    "top_positive": A.actor_table(accs[p["name"]], role, actors,
                                                  p["name"]).nlargest(
                        5, "attribution")[["actor", "attribution", "ego_abs_share"]]
                    .to_dict(orient="records"),
                    "top_negative": A.actor_table(accs[p["name"]], role, actors,
                                                  p["name"]).nsmallest(
                        5, "attribution")[["actor", "attribution", "ego_abs_share"]]
                    .to_dict(orient="records"),
                } for role in ("source", "target")
            } for p in C.PERIODS
        },
        "collinearity": {"condition_number": round(cond, 2),
                         "max_vif": round(float(vif.max()), 2),
                         "n_high_vif_gt10": int((vif > 10).sum())},
        "p1_reference": p1_note,
        "test_access_log": [{
            "split": "test", "months": "276..334 (2018-01..2022-11)",
            "rows": int(len(rows["retro_2018_22"]["y"])),
            "purpose": "burned retrospective secondary view (explicitly requested)",
            "confirmatory": False, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                        time.gmtime()),
        }],
        "provisional": {
            "fresh_dyad_holdout": "not available in this sprint (no Sprint A artifact)",
            "confirmatory_claim_allowed": False,
            "note": ("All inner-window numbers are inner validation; 2018-22 and "
                     "COVID are burned retrospective secondary. No test tuning "
                     "was performed."),
        },
        "input_hashes": C.input_hashes(cfg),
        "runtime_seconds": elapsed(t_start),
    }
    from framework.utils import host_info, pkg_versions
    summary["host"] = {**host_info(), **pkg_versions()}
    C.atomic_json(os.path.join(C.OUT, "summary_decompose.json"), summary)
    log(f"summary_decompose.json written; total {elapsed(t_start)}s")
    rl.close()


if __name__ == "__main__":
    main()
