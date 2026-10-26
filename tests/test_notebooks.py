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
    def test_three_notebooks_are_present_and_strict_json(self):
        paths = sorted(NOTEBOOKS.glob("*.ipynb"))
        self.assertEqual([p.name for p in paths], [
            "01_confirmatory_holdout.ipynb",
            "02_stability_audit.ipynb",
            "03_worldline_preview.ipynb",
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
        self.assertIn("pre-registered secondary", source)
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
        self.assertIn("ILLUSTRATIVE PREVIEW - NOT VALIDATED", source)
        self.assertIn("no multi-step accuracy claim", source.lower())

    def test_figures_are_embedded_and_metadata_is_current(self):
        expected_images = {"01_confirmatory_holdout.ipynb", "03_worldline_preview.ipynb"}
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
            self.assertTrue(version.startswith("3.13"), f"{path.name}: {version}")


if __name__ == "__main__":
    unittest.main()
