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

    # -- R2C2 M01: the audit status literal must state the true timing --------
    def test_audit_status_literal_is_truthful(self):
        """The grid was frozen mid-run, so the label may not claim "pre-holdout"."""
        literal = "frozen_during_confirmatory_run_holdout_rows_excluded"
        summary = strict_load(STABILITY / "summary.json")
        self.assertEqual(summary["scope"]["audit_design"], literal)
        self.assertEqual(summary["holdout"]["audit_design"], literal)
        hgb = strict_load(STABILITY / "summary_hgb.json")
        self.assertEqual(hgb["holdout"]["audit_design"], literal)
        holdout = summary["holdout"]
        self.assertEqual(holdout["confirmatory_run_start_utc"],
                         "2026-10-23T05:26:55.031577Z")
        self.assertEqual(holdout["design_frozen_utc"],
                         "2026-10-23T06:31:40Z")
        self.assertEqual(holdout["completed_utc"], "2026-10-23T07:45:19Z")
        # Freeze strictly after the read started and before it completed.
        self.assertLess(holdout["confirmatory_run_start_utc"],
                        holdout["design_frozen_utc"])
        self.assertLess(holdout["design_frozen_utc"], holdout["completed_utc"])

    def test_no_shipped_stability_file_uses_the_legacy_label(self):
        legacy = "pre_holdout_holdout_excluded"
        for path in sorted(STABILITY.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".md", ".csv"}:
                self.assertNotIn(legacy, path.read_text(encoding="utf-8"), path)

    def test_report_says_the_freeze_was_mid_run(self):
        report = " ".join(
            (STABILITY / "report.md").read_text(encoding="utf-8").lower().split())
        self.assertIn("after the confirmatory read began", report)
        self.assertIn("frozen_during_confirmatory_run_holdout_rows_excluded", report)
        self.assertNotIn("is a pre-holdout design", report)

    def test_verifier_rejects_the_legacy_label(self):
        import importlib.util
        import tempfile
        spec = importlib.util.spec_from_file_location(
            "verify_release_legacy", ROOT / "scripts" / "verify_release.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "results" / "framework_pilot" / "stability"
            base.mkdir(parents=True)
            (base / "summary.json").write_text(
                '{"scope": {"audit_design": "pre_holdout_holdout_excluded"}}',
                encoding="utf-8")
            errors = module.check_legacy_stability_label(Path(tmp))
            self.assertTrue(
                any("legacy audit label" in e for e in errors), errors)
            (base / "summary.json").write_text(
                '{"scope": {"audit_design": '
                '"frozen_during_confirmatory_run_holdout_rows_excluded"}}',
                encoding="utf-8")
            self.assertEqual(module.check_legacy_stability_label(Path(tmp)), [])


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

    def test_manifest_code_annotations_match_shipped_bytes(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "verify_release_manifest_positive", ROOT / "scripts" / "verify_release.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.check_manifest_code(ROOT), [])
        manifest = strict_load(HOLDOUT / "manifest.json")
        annotations = manifest["release_annotations"]["code_annotations"]
        self.assertEqual(set(annotations["current_sha256"]), set(manifest["code"]))
        for rel, digest in annotations["current_sha256"].items():
            self.assertEqual(sha256(ROOT / rel), digest, rel)

    def test_verifier_rejects_a_tampered_code_hash(self):
        import importlib.util
        import tempfile
        spec = importlib.util.spec_from_file_location(
            "verify_release_manifest_negative", ROOT / "scripts" / "verify_release.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "results" / "framework_pilot" / "holdout"
            base.mkdir(parents=True)
            code = Path(tmp) / "experiments" / "synthetic.py"
            code.parent.mkdir(parents=True)
            code.write_text("print('synthetic')\n", encoding="utf-8")
            shipped = sha256(code)
            manifest = {
                "code": {"experiments/synthetic.py": shipped},
                "release_annotations": {"code_annotations": {
                    "current_sha256": {"experiments/synthetic.py": shipped},
                    "changed_since_freeze": {},
                    "p_display": {"files": ["experiments/synthetic.py"],
                                  "symbols": []},
                }},
            }
            target = base / "manifest.json"
            target.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(module.check_manifest_code(Path(tmp)), [])
            # A tampered current hash is caught.
            manifest["release_annotations"]["code_annotations"][
                "current_sha256"]["experiments/synthetic.py"] = "0" * 64
            target.write_text(json.dumps(manifest), encoding="utf-8")
            errors = module.check_manifest_code(Path(tmp))
            self.assertTrue(any("current code hash mismatch" in e for e in errors), errors)
            # A post-freeze change without an annotation is caught.
            manifest["release_annotations"]["code_annotations"][
                "current_sha256"]["experiments/synthetic.py"] = shipped
            manifest["code"]["experiments/synthetic.py"] = "1" * 64
            target.write_text(json.dumps(manifest), encoding="utf-8")
            errors = module.check_manifest_code(Path(tmp))
            self.assertTrue(any("without annotation" in e for e in errors), errors)
            # The shipped tree itself is clean.
            self.assertEqual(module.check_manifest_code(ROOT), [])

    def test_verifier_rejects_a_tampered_row_level_column(self):
        import importlib.util
        import tempfile
        spec = importlib.util.spec_from_file_location(
            "verify_release_row_tamper", ROOT / "scripts" / "verify_release.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = (STABILITY / "tables" / "actor_month_source_retro_2018_22.csv")
        with source.open(encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
        self.assertIn("aid", header)
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results" / "x"
            results.mkdir(parents=True)
            (results / "tampered.csv").write_text(
                ",".join(header + ["p"]) + "\n" + ",".join(["0"] * len(header) + ["0.5"]) + "\n",
                encoding="utf-8")
            errors = module.check_row_level_data(Path(tmp))
            self.assertTrue(
                any("with unit identifiers" in e for e in errors), errors)

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


class TestSetOverlapRange(unittest.TestCase):
    """R2C2 M02: absolute attribution is 0.00-0.33, log-occlusion 0.05-0.43."""

    def rows(self):
        with (STABILITY / "tables" / "hgb_occlusion_agreement.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            return list(csv.DictReader(handle))

    def values(self, level: str, column: str) -> list[float]:
        out = []
        for row in self.rows():
            if row["level"] != level or not row.get(column):
                continue
            out.append(float(row[column]))
        return out

    def test_actor_level_ranges_are_distinct(self):
        abs_j = self.values("actor", "jaccard_abs@10")
        logocc = self.values("actor", "jaccard_logocc_pos@10")
        self.assertEqual(len(abs_j), 6)      # 3 periods x 2 roles
        self.assertEqual(len(logocc), 6)
        self.assertAlmostEqual(min(abs_j), 0.0, places=12)
        self.assertAlmostEqual(max(abs_j), 0.3333, places=4)
        self.assertAlmostEqual(min(logocc), 0.0526, places=4)
        self.assertAlmostEqual(max(logocc), 0.4286, places=4)

    def test_actor_month_rows_are_absolute_zero_and_logocc_blank(self):
        abs_j = self.values("actor_month", "jaccard_abs@10")
        self.assertEqual(len(abs_j), 6)
        self.assertTrue(all(value == 0.0 for value in abs_j))
        self.assertEqual(self.values("actor_month", "jaccard_logocc_pos@10"), [])

    def test_signed_overlap_is_zero(self):
        signed = self.values("actor", "jaccard_pos@10")
        self.assertTrue(all(value == 0.0 for value in signed))

    def test_report_generator_does_not_conflate_the_ranges(self):
        text = (ROOT / "experiments" / "sprint_b_report.py").read_text(
            encoding="utf-8")
        flat = " ".join(text.split())
        self.assertIn("0.00-0.33", flat)
        # The conflation the paper used to print must be gone.
        self.assertNotIn("absolute-importance overlap is only 0.05-0.43", flat)
        self.assertNotIn("top-10 absolute overlap 0.05-0.43", flat)
        # Every quoted 0.05-0.43 must sit next to a log-occlusion scope.
        start = 0
        while True:
            pos = flat.find("0.05-0.43", start)
            if pos < 0:
                break
            window = flat[max(0, pos - 160):pos + 160]
            self.assertIn("log-occlusion", window, window)
            start = pos + 1


class TestFamilyBFloors(unittest.TestCase):
    """R2C2 CL-4: p-values below numerical resolution are bounds, not 0.0."""

    def test_bh_family_b_uses_a_bound_and_a_censoring_field(self):
        with (HOLDOUT / "tables" / "bh_families.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        row = next(r for r in rows
                   if r["family"] == "B_primary_other_metrics"
                   and r["comparison"] == "d_ap vs per_dyad_markov")
        self.assertEqual(row["p"], "<1e-300")
        self.assertEqual(row["q"], "<1e-300")
        self.assertEqual(row["censoring"], "p_normal_tail_below_float64")
        self.assertEqual(row["reject_0.05"], "True")
        for other in rows:
            self.assertNotEqual(other["p"], "0.0", other["comparison"])

    def test_auc_ap_delta_p_is_a_bound(self):
        with (HOLDOUT / "tables" / "auc_ap_deltas.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        bounded = [r for r in rows if r["d_ap_multiway_p"] == "<1e-300"]
        self.assertEqual(len(bounded), 2)
        for row in bounded:
            self.assertEqual(row["censoring"],
                             "d_ap_multiway_p_normal_tail_below_float64")
        self.assertFalse([r for r in rows if r["d_ap_multiway_p"] == "0.0"])

    def test_within_between_bootstrap_floor_is_a_bound(self):
        with (HOLDOUT / "tables" / "within_between_deltas.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertFalse([r for r in rows if "0.0" in (r["p_mean"], r["p_pooled"])])
        bounded = [r for r in rows
                   if r["censoring"] ==
                   "two_sided_paired_bootstrap_tail_zero_of_1000"]
        self.assertEqual(len(bounded), 4)
        for row in bounded:
            self.assertEqual(row["p_mean"] if row["p_mean"] == "<0.001"
                             else row["p_pooled"], "<0.001")
        # No delta, CI, SE or ordering changed.
        primary = next(r for r in rows if r["comparison"].endswith(
            "__vs__base_per_dyad_markov"))
        self.assertEqual(primary["d_pooled"], "0.024024662485383222")
        self.assertEqual(primary["d_mean"], "-0.004688250097412408")

    def test_summary_mirrors_the_bounds(self):
        summary = strict_load(HOLDOUT / "summary.json")
        row = next(r for r in summary["bh"]
                   if r["comparison"] == "d_ap vs per_dyad_markov")
        self.assertEqual(row["p"], "<1e-300")
        self.assertEqual(row["q"], "<1e-300")
        self.assertTrue(row["reject_0.05"])
        paired = summary["within_between"]["paired"][
            "logistic49_S1_temperature__vs__base_dyad_base_rate"]
        self.assertEqual(paired["p_mean"], "<0.001")
        self.assertEqual(paired["p_pooled"], "<0.001")


class TestWorldlinePreview(unittest.TestCase):
    """R6 X01: ship the paper's A1 render, not the superseded 164 mm preview."""

    def base(self) -> Path:
        return STABILITY.parent.parent / "worldline_preview"

    def test_pdf_is_the_paper_a1_render(self):
        pdf = (self.base() / "worldline_preview.pdf").read_bytes()
        self.assertIn(b"/MediaBox [ 0 0 425.1968503937 330 ]", pdf)
        self.assertNotIn(b"465 380", pdf)
        self.assertNotIn(b"DejaVu", pdf)
        self.assertNotIn(b"/Type3", pdf)
        self.assertIn(b"/FontFile2", pdf)
        self.assertEqual(sha256(self.base() / "worldline_preview.pdf"),
                         "77081960c20db85ff3748cc06edd3d38ceb9cf69fb93687ff612f5deec8bb3b1")
        self.assertEqual(sha256(self.base() / "worldline_preview.png"),
                         "3b6e0fb29706a5c8af1bf219120d4c125467bc37ecc9ad48f25bd3df046e02d4")

    def test_manifest_records_the_provenance(self):
        manifest = strict_load(self.base() / "run_manifest.json")
        provenance = manifest["figure_provenance"]
        self.assertEqual(provenance["width_mm"], 150.0)
        self.assertFalse(provenance["baked_key"])
        self.assertFalse(provenance["regenerable_from_this_tree"])
        self.assertEqual(provenance["pdf_sha256"],
                         "77081960c20db85ff3748cc06edd3d38ceb9cf69fb93687ff612f5deec8bb3b1")
        self.assertNotIn("465x380", manifest["figure_revision"])

    def test_docs_do_not_advertise_the_old_geometry(self):
        # m01 (R2C3): CODE_MAP.md and notebooks/README.md carried the stale
        # 164 mm / 465x380 pt sentence and were outside the original guard.
        for rel in ("results/README.md", "results/worldline_preview/README.md",
                    "notebooks/03_worldline_preview.ipynb", "CODE_MAP.md",
                    "notebooks/README.md"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            self.assertNotIn("164 mm", text, rel)
            self.assertNotIn("465 × 380", text, rel)
            self.assertNotIn("465x380", text, rel)
            self.assertIn("150", text, rel)

    def test_aggregate_probability_summaries_are_documented(self):
        # m02 (R2C3): the retained per-anchor p1/p2 distributions must be named
        # in the inventory, while no row-level value may ship.
        trace = strict_load(self.base() / "worldline_trace.json")
        for anchor in trace["anchors"]:
            for key in ("p1_summary", "p2_summary"):
                block = anchor[key]
                self.assertEqual(sorted(block), ["max", "mean", "median", "min"])
                for value in block.values():
                    self.assertIsInstance(value, float)
        for rel in ("results/README.md", "DATA_EXCLUSIONS.md",
                    "results/worldline_preview/README.md"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            for token in ("p1_summary", "p2_summary"):
                self.assertIn(token, text, rel)
            self.assertIn("no per-dyad", text.lower(), rel)
        scope = strict_load(self.base() / "run_manifest.json")["preview_scope"]
        self.assertEqual(scope["row_level_files_shipped"], 0)
        self.assertIn("anchors[*].p1_summary",
                      scope["aggregate_probability_summaries_shipped"])

    def test_verifier_rejects_the_superseded_preview(self):
        import importlib.util
        import tempfile
        spec = importlib.util.spec_from_file_location(
            "check_aggregates_old_preview", ROOT / "scripts" / "check_aggregates.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "results" / "worldline_preview"
            target.mkdir(parents=True)
            for name in ("README.md", "run_manifest.json", "worldline_trace.json",
                         "worldline_preview.png"):
                (target / name).write_bytes((self.base() / name).read_bytes())
            # The superseded 465x380 pt DejaVu print version.
            (target / "worldline_preview.pdf").write_bytes(
                b"%PDF-1.4\n/MediaBox [ 0 0 465 380 ]\n/FontName /DejaVuSans\n")
            errors = module.check_worldline(Path(tmp))
            joined = " | ".join(errors)
            self.assertIn("465 x 380", joined)
            self.assertIn("425.197 x 330", joined)
            self.assertIn("byte-identical", joined)


class TestR2C7EvidenceFlags(unittest.TestCase):
    """P2.2/P2.9/year-FE status: evidence flags and comparator wording."""

    def fe_rows(self) -> dict[str, dict]:
        with (HOLDOUT / "tables" / "dyad_fe.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            return {row["model"]: row for row in csv.DictReader(handle)}

    def test_non_converged_conditional_logit_is_not_evidence(self):
        rows = self.fe_rows()
        self.assertEqual(rows["conditional_logit"]["converged"], "false")
        self.assertEqual(rows["conditional_logit"]["used_as_evidence"], "false")
        extras = strict_load(HOLDOUT / "parts" / "holdout_extras.json")
        self.assertFalse(extras["fe"]["conditional_logit"]["success"])

    def test_dyad_fe_table_values_match_the_frozen_json(self):
        rows = self.fe_rows()
        extras = strict_load(HOLDOUT / "parts" / "holdout_extras.json")["fe"]
        pairs = {
            "conditional_logit": extras["conditional_logit_within"],
            "within_ols_lpm": extras["within_ols"]["within"],
        }
        for name, block in pairs.items():
            self.assertEqual(rows[name]["p1_within_auc"],
                             f"{block['p1']['within_auc']:.6f}")
            self.assertEqual(float(rows[name]["strict_mean"]), block["strict"]["mean"])
            self.assertEqual(float(rows[name]["strict_pooled"]),
                             block["strict"]["pooled"])
        self.assertEqual(rows["within_ols_lpm"]["converged"], "true")

    def test_comparator_wording_is_two_directional(self):
        report = (HOLDOUT / "report.md").read_text(encoding="utf-8").lower()
        self.assertIn("unequal in two directions", report)
        for stale in ("information-" + "advantaged", "information-" + "matched"):
            self.assertNotIn(stale, report)
        summary = strict_load(HOLDOUT / "summary.json")
        joined = " ".join(summary["paper_sentences"]).lower()
        self.assertNotIn("information-" + "matched", joined)
        self.assertIn("unequal in two directions", joined)

    def test_year_fe_status_is_plan_declared_and_post_hoc(self):
        summary = strict_load(HOLDOUT / "summary.json")
        sentence = next(s for s in summary["paper_sentences"]
                        if "year-fixed-effects anchor" in s)
        self.assertIn("plan-declared", sentence)
        self.assertIn("post-hoc, C-confounded and uncorrected", sentence)
        self.assertIn("not a prospective model", sentence)
        self.assertNotIn("pre-registered " + "descriptive", sentence)
        report = (HOLDOUT / "report.md").read_text(encoding="utf-8")
        self.assertIn(sentence, report)


if __name__ == "__main__":
    unittest.main()
