"""Notebook-contract tests: executed, aggregate-only, path-clean outputs."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
ABS_TOKENS = ("/" + "home/", "/" + "tmp/", "/" + "Users/", "C:" + "\\Users")
PALETTE = ("#35525F", "#A2664F", "#84957A", "#C2A05F")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sources(nb: dict) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])


class TestNotebookSuite(unittest.TestCase):
    def test_five_notebooks_are_present_and_strict_json(self):
        paths = sorted(NOTEBOOKS.glob("*.ipynb"))
        self.assertEqual([p.name for p in paths], [
            "01_confirmatory_holdout.ipynb",
            "02_stability_audit.ipynb",
            "03_worldline_preview.ipynb",
            "04_exact_attribution.ipynb",
            "05_training_pipeline.ipynb",
        ])
        for path in paths:
            nb = load(path)
            self.assertEqual(nb["nbformat"], 4)
            self.assertEqual(nb["metadata"]["kernelspec"]["name"], "python3")

    def test_every_code_cell_is_executed_without_error(self):
        for path in sorted(NOTEBOOKS.glob("*.ipynb")):
            nb = load(path)
            for index, cell in enumerate(nb["cells"]):
                if cell.get("cell_type") != "code":
                    continue
                self.assertIsInstance(cell.get("execution_count"), int,
                                      f"{path.name} cell {index}")
                outputs = cell.get("outputs", [])
                self.assertTrue(outputs, f"{path.name} cell {index} has no output")
                self.assertFalse(
                    any(out.get("output_type") == "error" for out in outputs),
                    f"{path.name} cell {index} has an error output",
                )

    def test_outputs_contain_no_absolute_paths_or_tracebacks(self):
        for path in sorted(NOTEBOOKS.glob("*.ipynb")):
            text = path.read_text(encoding="utf-8")
            for token in ABS_TOKENS:
                self.assertNotIn(token, text, f"{path.name}: {token}")
            self.assertNotIn("Traceback (most recent call last)", text)

        # MN29: text/plain and text/html outputs are scanned, not just errors.
        import importlib.util
        import shutil
        import tempfile

        spec = importlib.util.spec_from_file_location(
            "check_notebooks_mn29", ROOT / "scripts" / "verify_release.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "notebooks"
            shutil.copytree(NOTEBOOKS, target)
            path = target / "03_worldline_preview.ipynb"
            nb = load(path)
            code = next(cell for cell in nb["cells"]
                        if cell.get("cell_type") == "code")
            code.setdefault("outputs", []).append({
                "output_type": "display_data",
                "data": {"text/plain": ["mean_p  0.123\n"]},
                "metadata": {},
            })
            path.write_text(json.dumps(nb), encoding="utf-8")
            errors = module.check_notebooks(Path(tmp))
            self.assertTrue(any("mean_p" in error for error in errors), errors)

    def test_notebooks_do_not_read_restricted_inputs(self):
        forbidden = ("read_parquet", "duckdb", "data/icews", "preds/", ".npz")
        for path in sorted(NOTEBOOKS.glob("*.ipynb")):
            source = sources(load(path))
            for token in forbidden:
                self.assertNotIn(token, source, f"{path.name}: {token}")

    def test_notebook_01_guards_empty_tables_and_uses_the_palette(self):
        source = " ".join(sources(load(NOTEBOOKS / "01_confirmatory_holdout.ipynb")).split())
        self.assertIn("comparisons.empty", source)
        self.assertIn('"comparison" not in comparisons.columns', source)
        self.assertIn("negative = better", source)
        for colour in PALETTE:
            self.assertIn(colour, source)
        self.assertNotIn("#1f4e79", source)
        self.assertNotIn('print("repo root:", ROOT)', source)

    def test_notebook_01_explains_the_secondary_model(self):
        source = sources(load(NOTEBOOKS / "01_confirmatory_holdout.ipynb")).lower()
        self.assertIn("plan-declared secondary", source)
        self.assertIn("post-hoc", source)
        self.assertIn("not a new confirmatory test", source)

    def test_notebook_02_reports_the_post_holdout_status(self):
        text = (NOTEBOOKS / "02_stability_audit.ipynb").read_text(encoding="utf-8")
        for phrase in ("provisional_holdout_not_integrated", "available_at_audit",
                       "completed_later", "used", "not integrated"):
            self.assertIn(phrase, text)
        self.assertNotIn("provisional_no_fresh_holdout", text)

    def test_notebook_03_is_labelled_illustrative(self):
        source = sources(load(NOTEBOOKS / "03_worldline_preview.ipynb"))
        lowered = " ".join(source.lower().split())
        self.assertIn("ILLUSTRATIVE PREVIEW - NOT VALIDATED", source)
        self.assertIn("no multi-step accuracy claim", lowered)
        # MJ02: same 4 x 20 non-random inner-period scope as §6.4.
        self.assertIn("four anchors", lowered)
        self.assertIn("20 sampled", lowered)
        self.assertIn("non-random", lowered)
        self.assertIn("inner-validation", lowered)
        self.assertIn("analyses nothing here", lowered)
        self.assertIn("not a paper result", lowered)

    def test_notebook_04_states_the_ownership_convention(self):
        source = " ".join(
            sources(load(NOTEBOOKS / "04_exact_attribution.ipynb")).split())
        lowered = source.lower()
        for phrase in ("six ownership classes", "seven-class",
                       "actor-dyad interaction", "55.1 / 62.3",
                       "post-hoc secondary", "near-collinearity"):
            self.assertIn(phrase, lowered, phrase)
        # the six-class table and the 62.3 seven-class view are both present
        self.assertIn("a8_ownership.csv", source)
        self.assertIn("a8_allocations.csv", source)
        self.assertIn("decision_impact_dyad.csv", source)
        # the exact additive identity is stated, with its tolerance
        self.assertIn("logit(p_i)", source)
        self.assertIn("1e-5", source)

    def test_notebook_05_states_the_pipeline_and_the_restriction(self):
        source = " ".join(
            sources(load(NOTEBOOKS / "05_training_pipeline.ipynb")).split())
        lowered = source.lower()
        for phrase in ("27 + 20 + 2", "49", "47", "29", "25",
                       "matched-vintage", "holdout-zeroed",
                       "c = 1.0", "logistic", "calibration",
                       "a0", "a7", "synthetic"):
            self.assertIn(phrase.lower(), lowered, phrase)
        self.assertIn("icews v37", lowered)
        self.assertIn("not redistributed", lowered)
        # the demo is synthetic and says so, next to the fit it runs
        self.assertIn("SYNTHETIC", source)
        self.assertIn("framework.models.fit_logistic", source)

    def test_figures_are_embedded_and_metadata_is_current(self):
        expected_images = {
            "01_confirmatory_holdout.ipynb",
            "03_worldline_preview.ipynb",
            "04_exact_attribution.ipynb",
            "05_training_pipeline.ipynb",
        }
        for path in sorted(NOTEBOOKS.glob("*.ipynb")):
            nb = load(path)
            images = [
                output for cell in nb["cells"] if cell.get("cell_type") == "code"
                for output in cell.get("outputs", [])
                if "image/png" in output.get("data", {})
            ]
            if path.name in expected_images:
                self.assertTrue(images, f"{path.name}: no embedded PNG")
            version = nb["metadata"]["language_info"]["version"]
            major_minor = tuple(int(part) for part in version.split(".")[:2])
            self.assertGreaterEqual(
                major_minor, (3, 12),
                f"{path.name}: {version} is below the declared 3.12 floor")


if __name__ == "__main__":
    unittest.main()
