"""Smoke tests for the public paper path (CPU only, no data files).

Run with::

    python3 -m unittest discover -s tests -v

The tests do not read ICEWS data, experiment configuration, caches, the frozen
test split or row-level predictions.
"""

from __future__ import annotations

import ast
import importlib.util
import os
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




ABS_TOKENS = ("/" + "home/", "/" + "tmp/", "/" + "Users/", "C:" + "\\Users")
V2_RUNNER = ROOT / "experiments" / "sprint_c_v2.py"
V2_ARCHIVE = ROOT / "experiments" / "sprint_c_v2_fit_repro.py"


def _load_v2(name: str = "sprint_c_v2_public"):
    spec = importlib.util.spec_from_file_location(name, V2_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _body_after_docstring(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    first = tree.body[0]
    if not (isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)):
        raise AssertionError(f"{path.name} has no module docstring")
    return source.splitlines()[first.end_lineno:]


class TestV2CodePort(unittest.TestCase):
    """The V2 runner ships for audit and is honest about its inputs."""

    def test_both_v2_scripts_are_shipped_and_compile(self):
        for path in (V2_RUNNER, V2_ARCHIVE):
            self.assertTrue(path.is_file(), path.name)
            compile(path.read_text(encoding="utf-8"), str(path.name), "exec")

    def test_v2_runner_imports_relative_to_the_public_tree(self):
        module = _load_v2()
        self.assertEqual(len(module.DYAD_CAUSAL), 27)
        self.assertEqual(len(module.GRAPH20), 20)
        for name, expected in (("FEATURES_49", 49), ("FEATURES_47", 47),
                               ("FEATURES_29", 29), ("FEATURES_25", 25)):
            self.assertEqual(len(getattr(module, name)), expected, name)
        self.assertEqual(len(module.CONTRASTS), 14)
        self.assertEqual(len(module.FIT_ORDER), 10)
        self.assertEqual(module.REPO, str(ROOT))
        self.assertTrue(
            os.path.isdir(os.path.join(module.REPO, "results",
                                       "framework_pilot", "holdout")))
        self.assertEqual(module.PROTOCOL_HASH,
                         "184ae726ead6412c2515317e9d6169e61eb85308427849c9aa5e7427cbe57676")

    def test_v2_runner_fails_closed_without_the_protocol(self):
        module = _load_v2("sprint_c_v2_fail_closed")
        if os.environ.get("NEUROSIM_V2_PROTOCOL_DIR") or os.path.isfile(
                module.PROTO_MD):
            self.skipTest("a licensed protocol directory is configured")
        with self.assertRaises(SystemExit) as caught:
            module.verify_protocol()
        message = str(caught.exception)
        self.assertIn("not redistributed", message)
        self.assertIn("NEUROSIM_V2_PROTOCOL_DIR", message)

    def test_v2_scripts_carry_no_host_paths(self):
        for path in (V2_RUNNER, V2_ARCHIVE):
            text = path.read_text(encoding="utf-8")
            for token in ABS_TOKENS:
                self.assertNotIn(token, text, f"{path.name}: {token}")

    def test_v2_runner_declares_the_aggregate_only_limit(self):
        doc = ast.get_docstring(ast.parse(
            V2_RUNNER.read_text(encoding="utf-8"))) or ""
        flat = " ".join(doc.split()).lower()
        self.assertIn("not runnable from this tree", flat)
        self.assertIn("licensed icews v37", flat)
        self.assertIn("a8f83417dfc7670740571ff19b3df8a65031e92dfd3459b6b7faaedd22bc0843",
                      doc)
        archive = ast.get_docstring(ast.parse(
            V2_ARCHIVE.read_text(encoding="utf-8"))) or ""
        self.assertIn("archival duplicate", archive.lower())

    def test_archival_copy_matches_the_runner_body(self):
        self.assertEqual(_body_after_docstring(V2_RUNNER),
                         _body_after_docstring(V2_ARCHIVE))


if __name__ == "__main__":
    unittest.main()
