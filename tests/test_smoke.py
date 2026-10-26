"""Smoke tests for the public paper path (CPU only, no data files).

Run with::

    python3 -m unittest discover -s tests -v

The tests do not read ICEWS data, experiment configuration, caches, the frozen
test split or row-level predictions.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.config import load_config  # noqa: E402
from framework.utils import json_default, sha256_text  # noqa: E402


class TestConfig(unittest.TestCase):
    def test_defaults_are_safe_without_configs(self):
        cfg = load_config()
        self.assertFalse(cfg["test_policy"]["allow_test"])
        self.assertLessEqual(cfg["split"]["train_end"], cfg["split"]["select_start"])
        self.assertLessEqual(cfg["split"]["select_end"], cfg["split"]["eval_start"])
        self.assertGreater(cfg["bootstrap"]["reps"], 0)

    def test_override_merge_does_not_enable_test(self):
        cfg = load_config(overrides={"split": {"train_end": "2014-01-01"}})
        self.assertEqual(cfg["split"]["train_end"], "2014-01-01")
        self.assertFalse(cfg["test_policy"]["allow_test"])

    def test_resolved_paths_stay_inside_release(self):
        cfg = load_config()
        for key in ("cache_dir", "out_dir", "icews_glob"):
            path = Path(cfg["paths"][key]).resolve()
            self.assertTrue(path.is_relative_to(ROOT), (key, path))


class TestUtils(unittest.TestCase):
    def test_sha256_text_is_stable(self):
        self.assertEqual(sha256_text("abc"), sha256_text("abc"))
        self.assertNotEqual(sha256_text("abc"), sha256_text("abd"))

    def test_json_default_rejects_unknown_objects(self):
        self.assertEqual(json_default(np.int64(3)), 3)
        self.assertEqual(json_default(np.float32(0.5)), 0.5)
        with self.assertRaises(TypeError):
            json_default(object())


class TestLegacyRemoval(unittest.TestCase):
    def test_legacy_modules_are_not_shipped(self):
        names = (
            "audit", "graph", "interpret", "operators", "pipeline", "reporting",
            "run_pilot", "synth", "synth_arms", "targets", "windows",
        )
        for name in names:
            self.assertIsNone(
                importlib.util.find_spec(f"framework.{name}"),
                f"legacy module framework.{name} is still importable",
            )

    def test_paper_modules_import(self):
        names = (
            "aggregation", "bootstrap", "config", "data", "features", "metrics",
            "models", "regions", "splits", "sprint_a", "sprint_a_inference",
            "utils",
        )
        for name in names:
            self.assertIsNotNone(
                importlib.util.find_spec(f"framework.{name}"),
                f"paper module framework.{name} is missing",
            )


if __name__ == "__main__":
    unittest.main()
