"""Model, metric and calibration contract tests (CPU only, no data)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.config import load_config  # noqa: E402
from framework.metrics import binary_metrics, within_entity_auc  # noqa: E402
from framework.models import (  # noqa: E402
    apply_temperature,
    baseline_global_transition,
    baseline_per_dyad_markov,
    baseline_persistence,
    fit_logistic,
    fit_temperature,
    linear_contributions,
    predict_logistic,
    scaler_apply,
    scaler_fit,
    sigmoid,
)


class TestScalingAndSigmoid(unittest.TestCase):
    def test_sigmoid_is_bounded_and_centred(self):
        self.assertAlmostEqual(float(sigmoid(0.0)), 0.5)
        values = sigmoid(np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0]))
        self.assertTrue(np.all(values >= 0.0))
        self.assertTrue(np.all(values <= 1.0))

    def test_constant_column_gets_unit_scale(self):
        X = np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]], dtype=np.float32)
        mu, sd = scaler_fit(X)
        np.testing.assert_allclose(mu, [1.0, 5.0])
        np.testing.assert_allclose(sd, [1.0, 1.0])
        np.testing.assert_allclose(scaler_apply(X, mu, sd), np.zeros_like(X))


class TestLogisticContract(unittest.TestCase):
    def _fit(self):
        rng = np.random.default_rng(11)
        X = rng.normal(size=(240, 3)).astype(np.float32)
        logits = 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.3
        y = (rng.random(len(X)) < sigmoid(logits)).astype(int)
        cfg = load_config()
        fit = fit_logistic(
            X[:160], y[:160], X[160:], y[160:], cfg,
            C_grid=[1.0], feature_names=["a", "b", "c"],
        )
        return fit, X

    def test_fit_selection_and_shapes(self):
        fit, X = self._fit()
        self.assertEqual(fit["kind"], "logistic")
        self.assertEqual(fit["C"], 1.0)
        self.assertEqual(fit["coef"].shape, (3,))
        self.assertEqual(len(fit["select_curve"]), 1)
        p = predict_logistic(fit, X)
        self.assertEqual(p.shape, (len(X),))
        self.assertTrue(np.all((p > 0) & (p < 1)))

    def test_additive_contributions_reproduce_the_decision_function(self):
        fit, X = self._fit()
        C = linear_contributions(fit, X, ["a", "b", "c"])
        eta = fit["intercept"] + C.sum(axis=1)
        np.testing.assert_allclose(sigmoid(eta), predict_logistic(fit, X), atol=1e-7)

    def test_temperature_is_affine_and_monotone(self):
        p = np.array([0.05, 0.2, 0.5, 0.8, 0.95])
        np.testing.assert_allclose(apply_temperature(p, 1.0), p, atol=1e-12)
        warmer = apply_temperature(p, 2.0)
        self.assertTrue(np.all(np.diff(warmer) > 0))
        self.assertLess(np.max(np.abs(warmer - 0.5)), np.max(np.abs(p - 0.5)))

    def test_temperature_fit_reports_a_grid_value(self):
        rng = np.random.default_rng(12)
        p_sel = rng.uniform(0.01, 0.99, 200)
        y_sel = (rng.random(200) < p_sel).astype(int)
        p_eval = rng.uniform(0.01, 0.99, 50)
        result = fit_temperature(p_sel, y_sel, p_eval)
        self.assertIn("T", result)
        self.assertGreaterEqual(result["T"], 0.4)
        self.assertLessEqual(result["T"], 3.0)
        self.assertEqual(result["p_eval"].shape, p_eval.shape)


class TestBaselines(unittest.TestCase):
    def _data(self):
        train = {
            "active": np.array([0, 0, 1, 1, 0, 1]),
            "y": np.array([0, 0, 1, 0, 1, 1]),
            "did": np.array([0, 1, 0, 1, 0, 1]),
        }
        evaluate = {
            "active": np.array([0, 1, 0, 1]),
            "did": np.array([0, 1, 9, 9]),
        }
        return train, evaluate

    def test_global_transition_matches_manual_counts(self):
        train, evaluate = self._data()
        p = baseline_global_transition(train, evaluate)
        self.assertEqual(p.shape, (4,))
        expected_active = (2 + 1.0) / (3 + 2.0)
        expected_inactive = (1 + 1.0) / (3 + 2.0)
        np.testing.assert_allclose(
            p, [expected_inactive, expected_active, expected_inactive, expected_active]
        )

    def test_per_dyad_markov_falls_back_to_global_for_unseen_dyads(self):
        train, evaluate = self._data()
        p = baseline_per_dyad_markov(train, evaluate, n_dyads=10)
        global_p = baseline_global_transition(train, evaluate)
        np.testing.assert_allclose(p[2:], global_p[2:])

    def test_persistence_returns_two_probabilities(self):
        train, evaluate = self._data()
        p = baseline_persistence(train, evaluate)
        pc = float((train["active"] == train["y"]).mean())
        self.assertTrue(set(np.round(p, 8)).issubset({round(pc, 8), round(1 - pc, 8)}))


class TestMetricsContract(unittest.TestCase):
    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.01, 0.02, 0.98, 0.99])
        metrics = binary_metrics(y, p)
        self.assertLess(metrics["log_loss"], 0.05)
        self.assertEqual(metrics["auc"], 1.0)
        self.assertEqual(metrics["ap"], 1.0)

    def test_within_entity_auc_separates_levels(self):
        rng = np.random.default_rng(13)
        cid = np.repeat(np.arange(40), 25)
        level = np.repeat(rng.normal(0, 1, 40), 25)
        p = 1 / (1 + np.exp(-(level * 3 + rng.normal(0, 0.2, len(cid)))))
        y = (rng.random(len(cid)) < p).astype(int)
        result = within_entity_auc(y, p, cid)
        self.assertIsNotNone(result["between_auc"])
        self.assertGreater(result["between_auc"], 0.8)
        self.assertLess(abs(result["within_auc"] - 0.5), 0.12)


if __name__ == "__main__":
    unittest.main()
