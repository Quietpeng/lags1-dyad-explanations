"""Focused unit tests for the inference helpers (no data, no network, no I/O).

R2C5 m04 noted that ``framework/bootstrap.py``, ``framework/sprint_a.py``,
``framework/sprint_a_inference.py`` and ``framework/regions.py`` were imported
by no test.  These tests exercise the pure helpers on synthetic arrays; they do
not read ICEWS data, configuration, caches or frozen predictions and they do not
recompute a headline number.
"""

from __future__ import annotations

import math
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework import bootstrap, regions, sprint_a, sprint_a_inference  # noqa: E402
from framework.metrics import binary_metrics  # noqa: E402


class TestBootstrap(unittest.TestCase):
    def setUp(self):
        self.y = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        self.p = np.array([0.9, 0.1, 0.8, 0.6, 0.2, 0.3, 0.7, 0.25])
        self.cid = np.array([1, 1, 2, 2, 3, 3, 4, 4])

    def test_cluster_sufficient_counts(self):
        S = bootstrap.cluster_sufficient(self.y, self.p, self.cid)
        self.assertEqual(S["n_clusters"], 4)
        np.testing.assert_allclose(S["n"], [2, 2, 2, 2])
        self.assertAlmostEqual(float(S["n"].sum()), 8.0)
        self.assertEqual(float(np.sum(S["nll"])),
                         float(np.sum(-(self.y * np.log(
                             np.clip(self.p, 1e-12, 1 - 1e-12)) +
                             (1 - self.y) * np.log(
                                 1 - np.clip(self.p, 1e-12, 1 - 1e-12))))))

    def test_boot_weights_are_seeded_and_complete(self):
        first = bootstrap.boot_weights(5, 20, np.random.default_rng(7))
        second = bootstrap.boot_weights(5, 20, np.random.default_rng(7))
        self.assertEqual(first.shape, (20, 5))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first.sum(axis=1), np.full(20, 5))
        self.assertTrue((first >= 0).all())

    def test_full_weight_aggregate_matches_point_metrics(self):
        S = bootstrap.cluster_sufficient(self.y, self.p, self.cid)
        out = bootstrap.aggregate(np.ones((1, S["n_clusters"])), S)
        point = binary_metrics(self.y, self.p)
        self.assertAlmostEqual(float(out["log_loss"][0]), point["log_loss"], places=5)
        self.assertAlmostEqual(float(out["brier"][0]), point["brier"], places=5)
        self.assertAlmostEqual(float(out["accuracy"][0]), point["accuracy"], places=5)

    def test_ci_rounds_and_handles_empty(self):
        self.assertEqual(bootstrap.ci([0.1, 0.2, 0.3, 0.4]),
                         [0.1075, 0.3925])
        self.assertEqual(bootstrap.ci([np.nan, np.inf]), None)

    def test_paired_gain_of_identical_predictions_is_zero(self):
        out = bootstrap.paired_gain(
            self.y, self.p, self.p, self.cid, reps=40, seed=3)
        self.assertEqual(out["point"]["d_log_loss"], 0.0)
        self.assertEqual(out["point"]["d_brier"], 0.0)
        self.assertEqual(out["d_log_loss_ci"], [0.0, 0.0])
        self.assertEqual(out["reps"], 40)

    def test_dual_cluster_multiway_floor(self):
        out = bootstrap.dual_cluster_gain(
            self.y, self.p * 0.5 + 0.05, self.p, self.cid,
            np.array([1, 1, 2, 2, 1, 1, 2, 2]), reps=30, seed=2)
        floor = max(out["actor_se"], out["dyad_se"])
        self.assertGreaterEqual(out["multiway_se"] + 1e-12, floor)
        self.assertEqual(out["n"], 8)


class TestSprintAHelpers(unittest.TestCase):
    def test_month_idx_is_1995_based(self):
        self.assertEqual(sprint_a.month_idx("1995-01-01"), 0)
        self.assertEqual(sprint_a.month_idx("2015-02-01"), 241)

    def test_sha256_json_is_canonical(self):
        self.assertEqual(sprint_a.sha256_json({"a": 1, "b": [2]}),
                         sprint_a.sha256_json({"b": [2], "a": 1}))
        self.assertNotEqual(sprint_a.sha256_json({"a": 1}),
                            sprint_a.sha256_json({"a": 2}))

    def test_slice_dict_slices_rows_and_keeps_scalars(self):
        d = {"did": np.arange(4), "meta": "x", "count": 4}
        out = sprint_a.slice_dict(d, np.array([True, False, True, False]))
        np.testing.assert_array_equal(out["did"], [0, 2])
        self.assertEqual(out["meta"], "x")
        self.assertEqual(out["count"], 4)

    def test_clipped_calendar_does_not_mutate_input(self):
        X = np.array([[1.0, 2020.0], [2.0, 1999.0]], dtype=np.float32)
        out = sprint_a.set_clipped_calendar(X, ["f", "t_yr"], 2014.0)
        np.testing.assert_array_equal(out[:, 1], [2014.0, 1999.0])
        np.testing.assert_array_equal(X[:, 1], [2020.0, 1999.0])

    def test_year_fe_matrix_carries_2014_forward(self):
        X = np.array([[1.0, 2020.0], [2.0, 1997.0], [3.0, 1995.0]],
                     dtype=np.float32)
        out, names = sprint_a.year_fe_matrix(
            X, ["f", "t_yr"], np.array([300, 24, 0]))
        self.assertEqual(names[1:20], [f"yr_{y}" for y in range(1996, 2015)])
        self.assertEqual(out.shape, (3, 20))
        self.assertEqual(float(out[0, names.index("yr_2014")]), 1.0)
        self.assertEqual(float(out[1, names.index("yr_1997")]), 1.0)
        np.testing.assert_array_equal(out[:, 0], [1.0, 2.0, 3.0])
        # 1995 is the reference level: no dummy fires.
        self.assertEqual(float(out[2, 1:].sum()), 0.0)

    def test_warm_baselines_use_global_prior(self):
        eval_d = {"did": np.array([0, 1]), "active": np.array([1, 0])}
        bundle = {
            "warm": {"k": np.array([2.0, 0.0]), "n": np.array([4.0, 0.0]),
                     "n11": np.array([1.0, 0.0]), "n1": np.array([2.0, 0.0]),
                     "n01": np.array([1.0, 0.0]), "n0": np.array([2.0, 0.0])},
            "global_rate": 0.25,
            "global_transition": np.array([[0.9, 0.1], [0.2, 0.8]]),
        }
        base = sprint_a.baseline_warm_base_rate(eval_d, bundle)
        np.testing.assert_allclose(base, [(2 + 0.5) / 6, 0.25])
        markov = sprint_a.baseline_warm_markov(eval_d, bundle)
        np.testing.assert_allclose(markov, [(1 + 1.6) / 4, 0.1])

    def test_ece_is_zero_for_calibrated_half_probabilities(self):
        y = np.array([0, 1, 0, 1])
        p = np.full(4, 0.5)
        self.assertAlmostEqual(sprint_a.ece_equal_frequency(y, p), 0.0)
        block = sprint_a.metrics_block(y, p)
        for key in ("log_loss", "brier", "ece", "ece_equal_frequency"):
            self.assertIn(key, block)

    def test_holdout_assertions_pass_and_fail_closed(self):
        train = {"did": np.array([0, 1]), "y": np.array([0, 1]),
                 "t_idx": np.array([0, 1]), "br": np.array([0.5, 0.5])}
        select = {"did": np.array([0, 1]), "y": np.array([0, 1]),
                  "t_idx": np.array([10, 11])}
        evald = {"did": np.array([2, 3]), "y": np.array([0, 1]),
                 "t_idx": np.array([300, 301]), "seen_tr": np.array([0, 0]),
                 "br": np.array([0.5, 0.5])}
        out = sprint_a.run_holdout_assertions(
            train, select, evald, np.array([2, 3]), n_dyads=4,
            global_rate=0.5)
        self.assertTrue(out["pass"])
        self.assertEqual(out["eval_dyads"], 2)
        leaked = dict(evald, did=np.array([0, 3]))
        failed = sprint_a.run_holdout_assertions(
            train, select, leaked, np.array([2, 3]), n_dyads=4,
            global_rate=0.5)
        self.assertFalse(failed["pass"])
        self.assertEqual(failed["eval_nonholdout"], 1)

    def test_preregistration_roundtrip_and_overwrite_guard(self):
        payload = {"definition": {"pairs": [[1, 2]], "dids": [3]},
                   "created_utc": "2026-01-01T00:00:00+00:00"}
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "preregistration.json")
            written = sprint_a.write_preregistration(path, payload)
            self.assertEqual(written["definition_sha256"],
                             sprint_a.sha256_json(payload["definition"]))
            loaded = sprint_a.verify_preregistration(path)
            self.assertEqual(loaded["definition"], payload["definition"])
            with self.assertRaises(FileExistsError):
                sprint_a.write_preregistration(path, payload)
            tampered = dict(loaded)
            tampered["definition"] = {"pairs": [[9, 9]], "dids": [3]}
            Path(path).write_text(
                json.dumps(tampered), encoding="utf-8")
            with self.assertRaises(AssertionError):
                sprint_a.verify_preregistration(path)


class TestSprintAInference(unittest.TestCase):
    def test_p_display_renders_underflow_as_a_bound(self):
        self.assertEqual(sprint_a_inference.p_display(0.0), "<1e-300")
        self.assertEqual(sprint_a_inference.p_display(0.02), 0.02)
        self.assertEqual(sprint_a_inference.p_display(None), "")
        self.assertEqual(sprint_a_inference.boot_p_display(0.0, 1000), "<0.001")
        self.assertEqual(sprint_a_inference.boot_p_display(0.2, 1000), 0.2)

    def test_norm_p_is_two_sided_normal(self):
        self.assertEqual(sprint_a_inference._norm_p(0.0, 0.0), 1.0)
        self.assertEqual(sprint_a_inference._norm_p(1.0, 0.0), 0.0)
        self.assertAlmostEqual(
            sprint_a_inference._norm_p(1.959963984540054, 1.0), 0.05, places=8)
        self.assertAlmostEqual(sprint_a_inference._norm_p(0.0, 1.0), 1.0)

    def test_bh_fdr_is_monotone_and_rejects_small_p(self):
        out = sprint_a_inference.bh_fdr([0.001, 0.02, 0.5])
        self.assertEqual(out["m"], 3)
        self.assertEqual(out["reject"], [True, True, False])
        self.assertTrue(all(q <= 1.0 for q in out["q"]))
        ordered = [q for _, q in sorted(zip(out["p"], out["q"]))]
        self.assertEqual(ordered, sorted(ordered))

    def test_paired_multiway_reports_the_variance_floor(self):
        rng = np.random.default_rng(11)
        y = (rng.random(60) < 0.3).astype(int)
        p_base = np.clip(rng.random(60), 0.01, 0.99)
        p_model = np.clip(p_base + rng.normal(0, 0.02, 60), 0.01, 0.99)
        did = np.arange(60) // 2
        sid = np.arange(60) % 6
        tid = (np.arange(60) + 3) % 7
        out = sprint_a_inference.paired_multiway(
            y, p_base, p_model, did, sid, tid, reps=25, seed=5)
        self.assertEqual(out["point"]["n"], 60)
        self.assertEqual(out["multiway"]["n_clusters"]["dyad"],
                         out["dims"]["dyad"]["n_clusters"])
        floor = max(out["dims"]["source"]["var_ll"],
                    out["dims"]["target"]["var_ll"],
                    out["dims"]["dyad"]["var_ll"])
        self.assertGreaterEqual(out["multiway"]["var_ll"] + 1e-12, floor)
        self.assertTrue(math.isfinite(out["multiway"]["p_ll"]))

    def test_mde_pair_is_finite_for_a_paired_contrast(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        base = np.full(8, 0.4)
        model = np.full(8, 0.45)
        out = sprint_a_inference.mde_pair(
            y, base, model, np.arange(8) // 2, label="synthetic")
        self.assertEqual(out["label"], "synthetic")
        self.assertTrue(math.isfinite(out["mde_paired"]))
        self.assertTrue(math.isfinite(out["mde_baseline"]))


class TestRegions(unittest.TestCase):
    def test_known_and_unknown_iso3(self):
        self.assertEqual(regions.region_of("USA"), "NorthAmerica")
        self.assertEqual(regions.region_of("DEU"), "Europe")
        self.assertEqual(regions.region_of("ZZZ"), "Other")

    def test_region_map_has_no_duplicate_actors(self):
        flat = [iso for isos in regions.REGIONS.values() for iso in isos]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(set(flat), set(regions.ISO_TO_REGION))
        self.assertEqual(len(regions.REGION_NAMES), len(regions.REGIONS))

    def test_assert_coverage_closed_over_the_map(self):
        sample = ["USA", "DEU", "AFG"]
        self.assertIsNone(regions.assert_coverage(sample))
        with self.assertRaises(AssertionError):
            regions.assert_coverage(sample + ["ZZZ"])


if __name__ == "__main__":
    unittest.main()
