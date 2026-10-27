#!/usr/bin/env python3
"""Freeze the Sprint-A 20% unordered-dyad holdout before any label is read.

Run once, before ``sprint_a_run.py --part holdout``.  The script reads only
``active`` (current-month activity) for the train-period stratification and row
counts; it never reads ``active_next`` on 2018-22.

    python3 experiments/sprint_a_prereg.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework import data, features, sprint_a
from framework.config import load_config
from framework.utils import jdump, jload

OUT = "results/framework_pilot/holdout/preregistration.json"
ACCESS = "results/framework_pilot/holdout/test_access_log.json"


def main() -> int:
    cfg = load_config()
    con = data.connect(cfg)
    data.load_views(con, cfg)
    spec = sprint_a.define_holdout(con, sprint_a.HOLDOUT_FRAC,
                                   sprint_a.HOLDOUT_SEED)
    con.close()

    definition = spec["definition"]
    payload = {
        "schema": "q2-sprint-a-prereg/1",
        "definition": definition,
        "created_utc": sprint_a.utc_now(),
        "status": "FROZEN_BEFORE_HOLDOUT_LABEL_READ",
        "one_shot": True,
        "paper_scope": {
            "claim": ("binary dyadic event prediction + source/target actor "
                      "exact additive explanations + stability audit; no "
                      "node-level prediction, no lifecycle causal claim"),
            "period": ("2018-01..2022-11 retrospective; unit-fresh 20% "
                       "unordered-dyad holdout; not a time-fresh test"),
            "forbidden": ["synthetic/T1", "new data", "GPU", "test tuning",
                          "2018-22 confirmatory headline without the holdout"],
        },
        "holdout": {
            "definition": definition,
            "counts": spec["counts"],
            "strata_counts": spec["strata_counts"],
            "universe": {
                "unordered_pairs": spec["n_pairs_universe"],
                "directed_dyads": spec["n_dids_universe"],
            },
        },
        "embargo": {
            "t_idx": list(sprint_a.EMBARGO_T),
            "dates": {str(k): v for k, v in sprint_a.EMBARGO_DATES.items()},
            "rationale": ("the label at month t is activity at t+1; the last "
                          "month of a period labels the next period. Dropping "
                          "the boundary row removes the one-month label leak."),
            "handling": ("drop from train/select/eval; recompute br from the "
                         "remaining train rows; graph features are not rebuilt "
                         "for the embargo because they use edges at <=t, not "
                         "the row's label."),
            "expected_drops": {
                "239": "2014-12 train row (label 2015-01 select)",
                "251": "2015-12 select row (label 2016-01 eval)",
                "275": "2017-12 eval row (label 2018-01 holdout)",
                "335": "2022-12 holdout row (label 2023-01 tail)",
            },
        },
        "splits": {
            "train": "t_idx < 239, holdout dyads removed, t_idx not in embargo",
            "select": ("240 <= t_idx < 251 (2015-01..2015-11), holdout dyads "
                       "removed, used for C / HGB / calibration"),
            "eval": ("276 <= t_idx < 335 (2018-01..2022-11), holdout dyads "
                     "only, embargo row 335 removed"),
            "calibration": "select only; no eval labels",
            "tail": "t_idx >= 336 not used",
        },
        "feature_sets": {
            "n29": len(sprint_a.FEATURES_29),
            "n49": len(sprint_a.FEATURES_49),
            "features_29": sprint_a.FEATURES_29,
            "features_49": sprint_a.FEATURES_49,
            "graph_core20": list(features.GRAPH_CORE20),
            "note": ("exact frozen P1 contract: 27 causal + br + seen_tr = 29; "
                     "+20 graph core features = 49. No feature is added or "
                     "removed after seeing the holdout."),
        },
        "calendar_specs": {
            "S1_primary": ("frozen t_yr = t_idx/12, linear extrapolation as in "
                           "P1; this is the plan-declared primary because it "
                           "is the exact frozen feature contract."),
            "S2_sensitivity": ("t_yr clipped at 238/12 (2014-11, last train "
                               "month), a no-extrapolation calendar encoding "
                               "motivated by the 2026-10-23 methodology review; "
                               "added on the day of the run, per the plan's "
                               "self-reported timestamp, before the "
                               "confirmatory read."),
            "year_fe_anchor": ("year dummies 1996..2014 with 2014 carried "
                               "forward for 2015+; used as a feature-group "
                               "anchor, not as the primary model."),
        },
        "models": {
            "primary": "logistic49 + temperature (S1)",
            "primary_secondary": "logistic29 + temperature (S1)",
            "capacity_control": "HGB49 with 2x3 lr x leaves grid (max_iter 120, refined 80/160/200 if at edge)",
            "calibration": ["none (raw log-odds)", "temperature (primary)",
                            "isotonic (sensitivity only)"],
            "logistic_C_grid": [0.1, 0.3, 1.0, 3.0],
            "hgb_grid": {"learning_rate": [0.05, 0.10],
                         "max_leaf_nodes": [15, 31, 63],
                         "max_iter": 120,
                         "refine_max_iter": [80, 160, 200],
                         "max_bins": 128, "min_samples_leaf": 100},
            "selection": "select 2015 only; no 2018-22 tuning",
        },
        "baselines": {
            "global_transition": "P(y=1|active) from train, +1/+2 smoothing",
            "per_dyad_markov": "train transition counts per directed dyad, +2 prior; collapses to global transition for unseen dyads",
            "dyad_base_rate": "br from train counts; collapses to global rate for unseen dyads",
            "markov_recency": "logistic on active, since_act, log_n_act; C tuned on select",
            "secondary_warm_history": {
                "warm_dyad_base_rate": "pre-2018 (t_idx<275) dyad active_next rate, +2 prior",
                "warm_dyad_markov": "pre-2018 (t_idx<275) dyad transition counts, +2 prior",
                "why": ("the train-only per-dyad baselines are degenerate on a "
                        "unit holdout; warm-history versions use only observed "
                        "history available at prediction time, not holdout labels"),
            },
        },
        "inference": {
            "paired": "one bootstrap weight vector applied to base and model",
            "clusters": ["dyad", "source actor", "target actor"],
            "multiway": ("max(Var_source + Var_target - Var_dyad, Var_source, "
                         "Var_target, Var_dyad); dyad is the source-target "
                         "intersection, so this is also the three-way "
                         "dyad+source+target interval"),
            "reps": 1000,
            "seeds": [20261023, 20261024, 20261025, 20261026, 20261027],
            "auc_ap": ("full holdout, no subsample; 1,000 cluster-bootstrap "
                       "reps for the primary comparison; 1 seed for secondary "
                       "comparisons, 5 seeds for LL/ECE"),
            "bh_families": {
                "A_primary_ll": ["logistic49_temp vs per_dyad_markov",
                                 "logistic49_temp vs dyad_base_rate",
                                 "logistic49_temp vs markov_recency",
                                 "logistic49_temp vs global_transition"],
                "B_primary_other_metrics": ["d_ece vs per_dyad_markov",
                                             "d_auc vs per_dyad_markov",
                                             "d_ap vs per_dyad_markov"],
                "C_capacity_feature": ["hgb49 vs per_dyad_markov",
                                       "logistic29_temp vs per_dyad_markov",
                                       "hgb49 vs logistic49_temp"],
                "D_group_attribution": ["leave-one-group-out delta LL for 9 groups"],
            },
            "claim_rule": ("report dyad/source/target/actor-two-way/multiway CIs "
                           "together; if the multiway CI includes 0, do not claim "
                           "an accuracy improvement (SR-F7); if the gain is <2% "
                           "relative, label it exploratory."),
        },
        "endpoints": {
            "primary": ["log_loss", "ece", "auc", "ap"],
            "secondary": ["brier", "calibration_slope", "calibration_intercept",
                          "ece_equal_frequency"],
            "mde": ("paired MDE at 80% power, alpha=0.05, dyad-clustered; also "
                    "baseline-variance MDE as a conservative reference"),
            "no_selective_ci": ("all four clustering views are reported for the "
                                "primary contrast; no dyad-only claim"),
        },
        "test_policy": {
            "allow_test": False,
            "holdout_label_read": "exactly once, in sprint_a_run.py --part holdout",
            "access_log": "results/framework_pilot/holdout/test_access_log.json",
            "burn": ("2018-22 is retrospectively burned by earlier G2/D2/B "
                     "runs; the holdout is unit-fresh, not time-fresh."),
        },
        "stopping_rules": {
            "SR-F1": "no fresh holdout -> no confirmatory abstract (this run satisfies it by unit freshness)",
            "SR-F7": "multiway CI includes 0 -> framework/audit claim only",
            "gain_lt_2pct": "report as exploratory, not a headline accuracy claim",
            "hgb_beats_logistic_gt_0.5pct": "drop 'no capacity headroom' language",
            "seed_spread_gt_0.5pct": "drop the seed-stability claim",
        },
        "hashes": {},
    }

    result = sprint_a.write_preregistration(OUT, payload)

    # The one-shot ledger starts before the holdout run.  The pre-prereg
    # support count below was an aggregate over *all* dyads, used only to size
    # the holdout; it did not reveal holdout-specific labels and no model was
    # fitted on 2018-22.
    if os.path.exists(ACCESS):
        log = jload(ACCESS)
    else:
        log = {
            "policy": ("one-shot unit-fresh holdout; every post-freeze 2018-22 "
                       "read for the holdout is logged; three earlier "
                       "exploratory passes (G2, D2, B) predate this log, are "
                       "outside its scope and are disclosed in the manuscript; "
                       "the confirmatory run is allowed once"),
            "out_of_scope": {
                "scope": ("reads that predate the 2026-10-23 preregistration "
                          "freeze, and the three earlier exploratory passes "
                          "disclosed in the manuscript (G2, D2, B)"),
                "related_logs": {
                    "results/framework_pilot/stability/summary_decompose.json":
                        ("the post-freeze stability audit logs its own "
                         "burned-retrospective 2018-22 read there "
                         "(2026-10-23T14:57:50Z; confirmatory=false); it is a "
                         "separate ledger, not a holdout read"),
                },
                "prior_passes": [
                    {"label": label,
                     "status": ("pre-freeze exploratory pass on 2018-22; "
                                "predates this log")}
                    for label in ("G2", "D2", "B")
                ],
                "disclosure": ("The manuscript discloses these passes as "
                               "burned exploration. Only post-freeze reads plus "
                               "the single pre-freeze aggregate support count "
                               "that sized the holdout are recorded here."),
                "no_invented_entries": ("No dated entry is added for the prior "
                                        "passes: no timestamp, count or access "
                                        "record was kept for them, and "
                                        "inventing one would misstate the "
                                        "ledger."),
            },
            "entries": [],
        }
    log.setdefault("entries", [])
    log["entries"].extend([
        {
            "utc": None,
            "kind": "pre_prereg_support_count",
            "labels_read": True,
            "scope": "aggregate over all 26,434 dyads, 2018-01..2022-12",
            "what": ("rows=1,586,040; positives=164,272; used only to size "
                     "the holdout (the protocol cites ~317k rows / ~5,287 "
                     "dyads). No holdout-specific labels and no model fit."),
            "note": "wall time not captured; query ran before the prereg freeze",
        },
        {
            "utc": sprint_a.utc_now(),
            "kind": "preregistration",
            "labels_read": False,
            "what": ("holdout definition frozen; definition_sha256="
                     + result["definition_sha256"]),
        },
    ])
    jdump(ACCESS, log)

    print(json.dumps(result, indent=2))
    print("counts:", json.dumps(spec["counts"], indent=2))
    print("strata:", json.dumps(spec["strata_counts"], indent=2))
    print("universe:", spec["n_pairs_universe"], "pairs",
          spec["n_dids_universe"], "directed dyads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
