"""Smoke tests for the framework pilot (CPU only, no data files).

Run with::

    python \
        -m unittest discover -s tests -v

The tests do not read the frozen test split and do not touch ``data/``.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from framework import features as F  # noqa: E402
from framework import synth, targets, windows  # noqa: E402
from framework.config import load_config  # noqa: E402
from framework.metrics import binary_metrics, mde_block, within_entity_auc  # noqa: E402


class TestConfig(unittest.TestCase):
    def test_config_invariants_without_repo_configs(self):
        cfg = load_config(overrides={"synth": {"n_months": 36}})
        self.assertFalse(cfg["test_policy"]["allow_test"])
        self.assertLessEqual(cfg["split"]["train_end"], cfg["split"]["select_start"])
        self.assertLessEqual(cfg["split"]["select_end"], cfg["split"]["eval_start"])
        self.assertEqual(cfg["synth"]["n_months"], 36)

    def test_config_override_merge(self):
        cfg = load_config(overrides={"extensions": {"t1_redesign": {"primary_target": "high_q75"}}})
        self.assertEqual(cfg["extensions"]["t1_redesign"]["primary_target"], "high_q75")
        self.assertFalse(cfg["test_policy"]["allow_test"])


class TestFeatures(unittest.TestCase):
    def test_manifest_has_no_labels(self):
        man = F.feature_manifest()
        flat = man["dyad"] + man["actor"] + man["graph"]
        names = {r["name"] for r in flat}
        self.assertFalse(names & set(F.LABELS))
        self.assertTrue(all(r["available_at"] == "t" for r in flat))

    def test_dyad_groups_cover_base_features(self):
        covered = {n for cols in F.DYAD_GROUPS.values() for n in cols}
        self.assertTrue(set(F.DYAD_CAUSAL).issubset(covered | {"active", "log_ev"}))


class TestMetrics(unittest.TestCase):
    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.01, 0.02, 0.98, 0.99])
        m = binary_metrics(y, p)
        self.assertLess(m["log_loss"], 0.05)
        self.assertEqual(m["auc"], 1.0)
        self.assertEqual(m["ap"], 1.0)

    def test_within_entity_auc_separates_levels(self):
        rng = np.random.default_rng(0)
        cid = np.repeat(np.arange(40), 25)
        level = np.repeat(rng.normal(0, 1, 40), 25)
        p = 1 / (1 + np.exp(-(level * 3 + rng.normal(0, 0.2, len(cid)))))
        y = (rng.random(len(cid)) < p).astype(int)
        w = within_entity_auc(y, p, cid)
        self.assertIsNotNone(w["between_auc"])
        self.assertGreater(w["between_auc"], 0.8)
        self.assertLess(abs(w["within_auc"] - 0.5), 0.12)

    def test_mde_zero_for_constant_diff(self):
        d = np.ones(100)
        m = mde_block(d, np.repeat(np.arange(10), 10), 0.5)
        self.assertEqual(m["mde"], 0.0)


class TestSplitsAndLabels(unittest.TestCase):
    def _df(self):
        n = 20
        return pd.DataFrame({
            "aid": np.arange(n),
            "split": np.array(["train"] * 12 + ["val"] * 8),
            "t_idx": np.array(list(range(12)) + [240, 241, 251, 252, 273, 274, 275, 276]),
            "out_events": np.arange(n, dtype=float),
            "out_events_next": np.arange(n, dtype=float) + 1,
            "out_active": np.ones(n, dtype=int),
            "out_active_next": np.ones(n, dtype=int),
            "new_dyad": np.zeros(n, dtype=int),
            "out_events_prev": np.zeros(n, dtype=float),
            "out_active_prev": np.ones(n, dtype=int),
            "new_dyad_prev": np.zeros(n, dtype=int),
        })

    def test_threshold_train_only(self):
        df = self._df()
        thr = targets.threshold_for(df, "high_q75")
        self.assertAlmostEqual(thr, float(np.quantile(np.arange(12) + 1, 0.75)))

    def test_strict_eval_drops_boundary(self):
        df = self._df()
        m = targets.split_masks(df)
        self.assertTrue(m["eval"][15])         # t=252
        self.assertTrue(m["eval"][17])         # t=274
        self.assertFalse(m["eval"][18])        # t=275 boundary
        self.assertFalse(m["eval"][19])        # t=276 test


class TestWindows(unittest.TestCase):
    def test_window_masks(self):
        info = pd.DataFrame({
            "first_active": [0, 252, 0, 0],
            "last_reentry": [-999, -999, 252, -999],
        }, index=pd.Index([0, 1, 2, 3], name="did"))
        hist = {"info": info,
                "actor_first": pd.Series([252, 252, 0, 0], index=[10, 11, 12, 13])}
        out = windows.window_masks(
            np.array([252, 252, 252, 252]),
            np.array([0, 1, 2, 3]),
            np.array([10, 11, 12, 13]),
            np.array([11, 10, 10, 12]),
            hist)
        self.assertTrue(out["W_a_new_actor"][0])
        self.assertTrue(out["W_b_dyad_onset"][1])
        self.assertFalse(out["W_b_dyad_onset"][0])
        self.assertTrue(out["W_c_dormancy_reentry"][2])
        self.assertTrue(out["W_d_stable"][3])
        self.assertFalse(out["W_d_stable"][0])


class TestSynth(unittest.TestCase):
    def test_small_world_shapes(self):
        cfg = load_config()
        cfg["synth"]["n_actors"] = 12
        cfg["synth"]["n_months"] = 12
        cfg["synth"]["max_actors"] = 16
        moments = {
            "p_active": 0.11, "p11": 0.5, "p01": 0.06,
            "mean_events_active": 5.0,
            "degree_q": [1.0, 3.0, 8.0, 15.0, 25.0],
            "actor_log_events_mean": 3.0, "actor_log_events_sd": 1.5,
        }
        w = synth.generate_world(cfg, "base", 123, moments, 0)
        self.assertEqual(len(w.actor), 20 * 12)
        self.assertGreater(len(w.dyad), 0)
        self.assertTrue(set(w.dyad["split"]).issubset({"train", "val", "test"}))


if __name__ == "__main__":
    unittest.main()
