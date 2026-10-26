"""Unit checks for the Sprint B attribution plumbing (CPU only, no data)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from framework import aggregation as A  # noqa: E402
from framework import features as F  # noqa: E402


class TestExactness(unittest.TestCase):
    def test_decision_function_track_is_exact(self):
        rng = np.random.default_rng(7)
        names = F.DYAD_CAUSAL + F.GRAPH_CORE20 + ["br", "seen_tr"]
        fit = {
            "mu": rng.normal(size=len(names)).astype(np.float32),
            "sd": (np.abs(rng.normal(size=len(names))) + 0.5).astype(np.float32),
            "coef": rng.normal(scale=0.2, size=len(names)),
            "intercept": -2.0,
        }
        X = rng.normal(size=(200, len(names))).astype(np.float32)
        C = A.contributions(fit, X)
        eta = fit["intercept"] + C.sum(axis=1)
        rep = A.exactness_report(fit, C, A.sigmoid(eta), tol=1e-9, eta=eta)
        self.assertLess(rep["decision_max_abs_error"], 1e-9)
        self.assertEqual(rep["decision_rows_above_tol"], 0)

    def test_temperature_is_affine(self):
        rng = np.random.default_rng(8)
        names = F.DYAD_CAUSAL + F.GRAPH_CORE20 + ["br", "seen_tr"]
        fit = {
            "mu": np.zeros(len(names), dtype=np.float32),
            "sd": np.ones(len(names), dtype=np.float32),
            "coef": rng.normal(size=len(names)),
            "intercept": -1.0,
        }
        X = rng.normal(size=(50, len(names))).astype(np.float32)
        C = A.contributions(fit, X)
        eta = (fit["intercept"] + C.sum(axis=1)) / 2.0
        rep = A.exactness_report({**fit, "intercept": fit["intercept"] / 2.0},
                                 C / 2.0, A.sigmoid(eta), tol=1e-12, eta=eta)
        self.assertLess(rep["temperature_max_abs_error"], 1e-12)
        self.assertLess(rep["decision_max_abs_error"], 1e-12)


class TestWindowDenominators(unittest.TestCase):
    def test_source_and_target_window_counts_differ(self):
        rng = np.random.default_rng(9)
        names = F.DYAD_CAUSAL + F.GRAPH_CORE20 + ["br", "seen_tr"]
        n_actors = 6
        A_ = 5
        X = rng.normal(size=(40, len(names))).astype(np.float32)
        fit = {
            "mu": np.zeros(len(names), dtype=np.float32),
            "sd": np.ones(len(names), dtype=np.float32),
            "coef": rng.normal(scale=0.1, size=len(names)),
            "intercept": -1.0,
        }
        C = A.contributions(fit, X)
        sid = rng.integers(0, n_actors, size=40)
        tid = (sid + 1 + rng.integers(0, n_actors - 1, size=40)) % n_actors
        t_idx = np.full(40, 3)
        y = rng.integers(0, 2, size=40)
        p = A.sigmoid(fit["intercept"] + C.sum(axis=1))
        wid = rng.integers(0, 2, size=40)
        acc = A.AttributionAccumulator(n_actors, 3, 3, names, n_windows=2,
                                       do_occlusion=False)
        acc.add(X, sid, tid, t_idx, y, p, C, fit["intercept"], window_id=wid)
        expected_s = np.stack([np.bincount(sid[wid == w], minlength=n_actors)
                               for w in range(2)])
        expected_t = np.stack([np.bincount(tid[wid == w], minlength=n_actors)
                               for w in range(2)])
        np.testing.assert_array_equal(acc.win_count["source"], expected_s)
        np.testing.assert_array_equal(acc.win_count["target"], expected_t)
        self.assertFalse(np.array_equal(acc.win_count["source"],
                                        acc.win_count["target"]))


if __name__ == "__main__":
    unittest.main()
