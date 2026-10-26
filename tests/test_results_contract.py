"""Result-contract tests: frozen aggregates, provenance and claim boundaries.

These tests read only files shipped in the public tree. They do not read ICEWS
data, experiment configuration, caches or row-level predictions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
HOLDOUT = RESULTS / "framework_pilot" / "holdout"
STABILITY = RESULTS / "framework_pilot" / "stability"
sys.path.insert(0, str(ROOT))


def strict_load(path: Path) -> dict:
    def reject(token: str):
        raise AssertionError(f"non-finite JSON constant {token!r} in {path}")

    obj = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    json.dumps(obj, allow_nan=False)
    return obj


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparison_rows() -> dict[str, dict]:
    with (HOLDOUT / "tables" / "primary_comparisons.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        return {row["comparison"]: row for row in csv.DictReader(handle)}


class TestFrozenHeadlines(unittest.TestCase):
    def test_holdout_status_and_design(self):
        summary = strict_load(HOLDOUT / "summary.json")
        self.assertEqual(summary["status"], "confirmatory_full_holdout_completed")
        holdout = summary["holdout"]
        self.assertEqual(holdout["rows"], 311638)
        self.assertEqual(holdout["positives"], 32614)
        self.assertEqual(holdout["dyads"], 5282)
        self.assertTrue(holdout["unit_fresh"])
        self.assertFalse(holdout["time_fresh"])

    def test_primary_metrics_are_frozen(self):
        metrics = strict_load(HOLDOUT / "summary.json")["primary"]["metrics"]
        self.assertEqual(metrics["log_loss"], 0.217875)
        self.assertEqual(metrics["brier"], 0.06401)
        self.assertEqual(metrics["ece"], 0.042112)
        self.assertEqual(metrics["auc"], 0.910152)
        self.assertEqual(metrics["ap"], 0.671863)

    def test_primary_comparisons_are_frozen(self):
        rows = comparison_rows()
        primary = rows["logistic49_temperature__vs__per_dyad_markov"]
        self.assertAlmostEqual(float(primary["d_log_loss"]), 0.047999, places=6)
        self.assertAlmostEqual(float(primary["d_ece"]), -0.038642, places=6)
        warm = rows["logistic49_temperature__vs__warm_markov"]
        self.assertAlmostEqual(float(warm["d_log_loss"]), -0.008146, places=6)
        secondary = rows["logistic29_temperature__vs__warm_markov"]
        self.assertAlmostEqual(float(secondary["d_log_loss"]), 0.004510, places=6)

    def test_readme_quotes_the_frozen_numbers(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for value in ("0.217875", "0.910152", "0.671863", "0.047999",
                      "0.008146", "0.004510"):
            self.assertIn(value, readme)

    def test_readme_uses_one_sign_convention(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("Δ = model − comparator", readme)
        self.assertIn("negative = better", readme)

    def test_readme_keeps_the_secondary_boundary(self):
        readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").lower().split())
        self.assertIn("pre-registered 29-feature secondary", readme)
        self.assertIn("post-hoc", readme)
        self.assertIn("not a new confirmatory test", readme)

    def test_readme_keeps_the_claim_boundaries(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
        for phrase in ("provisional", "no node-level", "not time-fresh"):
            self.assertIn(phrase, readme)


class TestProvenance(unittest.TestCase):
    def test_sha256sums_cover_every_distributed_holdout_file(self):
        entries = {}
        for line in (HOLDOUT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, rel = line.split("  ", 1)
            entries[rel] = digest
        self.assertNotIn("SHA256SUMS", " ".join(entries))
        actual = {
            path.relative_to(ROOT).as_posix()
            for path in HOLDOUT.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        }
        self.assertEqual(actual, set(entries))
        for rel, digest in entries.items():
            self.assertEqual(sha256(ROOT / rel), digest, rel)

    def test_preregistration_checksum_is_repo_relative(self):
        line = (HOLDOUT / "preregistration.json.sha256").read_text(
            encoding="utf-8"
        ).strip()
        digest, rel = line.split()
        self.assertEqual(rel, "results/framework_pilot/holdout/preregistration.json")
        self.assertEqual(sha256(ROOT / rel), digest)
        summary = strict_load(HOLDOUT / "summary.json")
        self.assertEqual(summary["holdout"]["prereg_file_sha256"], digest)

    def test_manifest_marks_the_four_non_redistributed_outputs(self):
        manifest = strict_load(HOLDOUT / "manifest.json")
        excluded = {
            rel for rel, meta in manifest["outputs"].items()
            if meta.get("redistributed") is False
        }
        self.assertEqual(len(excluded), 4)
        for rel in excluded:
            self.assertFalse((ROOT / rel).exists(), rel)
            self.assertTrue(manifest["outputs"][rel].get("exclusion_reason"))
        self.assertFalse(
            manifest["release_annotations"]["numbers_changed"]
        )

    def test_summary_annotates_the_prediction_exclusion(self):
        summary = strict_load(HOLDOUT / "summary.json")
        status = summary["release_annotations"]["artifact_redistribution"]
        preds = status["results/framework_pilot/holdout/preds/holdout_preds.npz"]
        self.assertFalse(preds["redistributed"])
        self.assertTrue(preds["reason"])

    def test_all_results_json_is_strict(self):
        for path in sorted(RESULTS.rglob("*.json")):
            strict_load(path)

    def test_no_restricted_directories_are_present(self):
        for name in ("paper", "docs", "configs", "data"):
            self.assertFalse((ROOT / name).exists(), name)
        self.assertFalse(list(RESULTS.rglob("preds")))
        self.assertFalse(list(RESULTS.rglob("*.npz")))


class TestStabilityStatus(unittest.TestCase):
    def test_summary_states_the_holdout_timing(self):
        summary = strict_load(STABILITY / "summary.json")
        self.assertEqual(summary["status"], "provisional_holdout_not_integrated")
        holdout = summary["holdout"]
        self.assertFalse(holdout["available_at_audit"])
        self.assertTrue(holdout["completed_later"])
        self.assertFalse(holdout["used"])
        self.assertEqual(holdout["completed_utc"], "2026-10-23T07:45:19Z")

    def test_report_carries_the_release_banner(self):
        report = (STABILITY / "report.md").read_text(encoding="utf-8").lower()
        for phrase in ("frozen at", "holdout completed after", "not integrated",
                       "available_at_audit", "provisional"):
            self.assertIn(phrase, report)

    def test_summaries_use_the_same_status(self):
        for name in ("summary_decompose.json", "summary_hgb.json",
                     "summary_stability.json"):
            summary = strict_load(STABILITY / name)
            self.assertEqual(summary["status"], "provisional_holdout_not_integrated")


class TestReleaseMetadata(unittest.TestCase):
    def test_results_readme_documents_terms_and_exclusions(self):
        text = " ".join((RESULTS / "README.md").read_text(encoding="utf-8").lower().split())
        for phrase in ("derived-data terms", "not redistributed", "sign convention",
                       "row-level", "icews"):
            self.assertIn(phrase, text)

    def test_package_docstring_has_no_node_level_claim(self):
        text = (ROOT / "framework" / "__init__.py").read_text(encoding="utf-8")
        self.assertNotIn("node-level framework", text)
        self.assertNotIn("GraphSAGE-style encoder", text)
        self.assertIn("no node-level prediction", text)


class TestDataExclusions(unittest.TestCase):
    RATE_KEYS = {"mean_p", "p_sum", "y_sum", "mean_y", "p_model"}
    ACTOR_TABLES = [
        *(f"actor_month_{role}_{period}.csv"
          for role in ("source", "target")
          for period in ("inner_2016_17", "retro_2018_22", "covid_2020_21")),
        *(f"actor_{role}_{period}.csv"
          for role in ("source", "target")
          for period in ("inner_2016_17", "retro_2018_22", "covid_2020_21")),
        "actor_attribution_all_periods.csv",
    ]

    def test_dyad_contribution_sample_is_absent(self):
        hits = [p.as_posix() for p in RESULTS.rglob("*") if "dyad_contributions" in p.name]
        self.assertEqual(hits, [])

    def test_actor_tables_have_no_probability_or_outcome_columns(self):
        for name in self.ACTOR_TABLES:
            path = STABILITY / "tables" / name
            self.assertTrue(path.is_file(), name)
            with path.open(encoding="utf-8", newline="") as handle:
                header = set(next(csv.reader(handle)))
            self.assertFalse(header & self.RATE_KEYS, (name, header & self.RATE_KEYS))

    def test_no_distributed_csv_is_row_level(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "verify_release", ROOT / "scripts" / "verify_release.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.check_row_level_data(ROOT), [])

    def test_verifier_flags_a_synthetic_row_level_file(self):
        import importlib.util
        import tempfile
        spec = importlib.util.spec_from_file_location(
            "verify_release_negative", ROOT / "scripts" / "verify_release.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results" / "x"
            results.mkdir(parents=True)
            (results / "leak.csv").write_text(
                "did,month,y,p,c_a1\n1,2017-01,1,0.91,0.2\n", encoding="utf-8")
            errors = module.check_row_level_data(Path(tmp))
            self.assertTrue(any("row-level label+probability" in e for e in errors), errors)
            (results / "leak.csv").unlink()
            (results / "actor.csv").write_text(
                "actor,mean_p\nAFG,0.14\n", encoding="utf-8")
            errors = module.check_row_level_data(Path(tmp))
            self.assertTrue(any("prediction/label" in e or "rate" in e for e in errors), errors)
            (results / "actor.csv").unlink()
            wl = Path(tmp) / "results" / "worldline_preview"
            wl.mkdir(parents=True)
            (wl / "worldline_trace.csv").write_text(
                "dyad_label,p_model,state\ndyad_1,0.5,1\n", encoding="utf-8")
            errors = module.check_row_level_data(Path(tmp))
            self.assertTrue(any("world-line export is present" in e for e in errors), errors)

    def test_stability_json_has_no_rate_keys(self):
        def keys(obj):
            found = set()
            if isinstance(obj, dict):
                for key, value in obj.items():
                    found.add(str(key))
                    found |= keys(value)
            elif isinstance(obj, list):
                for value in obj:
                    found |= keys(value)
            return found

        for name in ("summary.json", "summary_decompose.json", "summary_hgb.json",
                     "summary_stability.json"):
            payload = strict_load(STABILITY / name)
            self.assertFalse(keys(payload) & self.RATE_KEYS, name)

    def test_worldline_row_level_exports_are_absent(self):
        base = STABILITY.parent.parent / "worldline_preview"
        for name in ("worldline_trace.csv", "node_trace.csv"):
            self.assertFalse((base / name).exists(), name)
        self.assertEqual(sorted(p.name for p in base.glob("*.csv")), [])

    def test_worldline_aggregate_is_bounded_and_probability_free(self):
        base = STABILITY.parent.parent / "worldline_preview"
        meta = strict_load(base / "worldline_trace.json")
        self.assertTrue(meta["no_row_level_probabilities"])
        self.assertFalse(meta["machine_readable_traces"]["shipped"])
        self.assertEqual(len(meta["anchors"]), 4)
        self.assertLessEqual(max(int(t) for t in meta["period"]["t_idx"]), 274)
        for anchor in meta["anchors"]:
            for forbidden in ("did", "aid", "dyad_label", "actor_label",
                              "p_model", "state"):
                self.assertNotIn(forbidden, anchor)
        manifest = strict_load(base / "run_manifest.json")
        self.assertEqual(manifest["preview_scope"]["row_level_files_shipped"], 0)
        self.assertFalse([rel for rel in manifest["outputs"] if rel.endswith(".csv")])
        for rel, digest in manifest["outputs"].items():
            self.assertEqual(sha256(base / rel), digest, rel)

    def test_data_exclusions_document_lists_the_removal(self):
        text = (ROOT / "DATA_EXCLUSIONS.md").read_text(encoding="utf-8")
        self.assertIn("dyad_contributions_inner_sample.csv", text)
        self.assertIn("mean_p", text)
        self.assertIn("worldline_trace.csv", text)

    def test_stability_sums_cover_every_distributed_file(self):
        entries = {}
        for line in (STABILITY / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, rel = line.split("  ", 1)
            entries[rel] = digest
        self.assertEqual(len(entries), 47)
        actual = {
            path.relative_to(ROOT).as_posix()
            for path in STABILITY.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        }
        self.assertEqual(actual, set(entries))
        for rel, digest in entries.items():
            self.assertEqual(sha256(ROOT / rel), digest, rel)


if __name__ == "__main__":
    unittest.main()
