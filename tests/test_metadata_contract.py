"""Metadata and release-hygiene contracts (R2C4).

Closes the documentation/metadata defects from the R2C3 repository review:

* M01 — the declared Python floor must match the pinned wheels
  (``numpy==2.5.4`` and ``scipy==1.18.1`` publish no cp311 wheels, so the
  floor is 3.12 and the CI matrix must not run a 3.11 leg).
* m03 — ``make dist`` must archive the committed tree only: an untracked file
  never enters the archive and a modified tracked file aborts the build.

The figure-geometry (m01) and world-line inventory (m02) contracts live next
to the frozen results in ``test_results_contract.py``.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "dyad-explanations-0.1.1"


def pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


class TestPythonSupportFloor(unittest.TestCase):
    """M01: one declared floor across packaging, tooling, docs and CI."""

    def test_requires_python_is_312(self):
        project = pyproject()["project"]
        version = tuple(int(part) for part in
                        re.findall(r"\d+", project["requires-python"])[:2])
        self.assertGreaterEqual(version, (3, 12),
                                "pinned numpy/scipy have no cp311 wheels")

    def test_classifiers_do_not_claim_311(self):
        classifiers = pyproject()["project"]["classifiers"]
        self.assertNotIn("Programming Language :: Python :: 3.11", classifiers)
        self.assertIn("Programming Language :: Python :: 3.12", classifiers)
        self.assertIn("Programming Language :: Python :: 3.13", classifiers)

    def test_doctor_uses_the_same_floor(self):
        text = (ROOT / "scripts" / "doctor.py").read_text(encoding="utf-8")
        match = re.search(r"^MIN_PYTHON = \((\d+), (\d+)\)$", text, re.MULTILINE)
        self.assertIsNotNone(match, "MIN_PYTHON not found in scripts/doctor.py")
        self.assertEqual((int(match.group(1)), int(match.group(2))), (3, 12))

    def test_docs_and_ci_state_the_312_floor(self):
        for rel in ("README.md", "REPRODUCING.md", "notebooks/README.md"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            self.assertNotIn("3.11+", text, rel)
            self.assertNotIn(">= 3.11", text, rel)
            self.assertIn("3.12", text, rel)
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertNotIn("3.11", ci)
        self.assertIn('"3.12"', ci)

    def test_pyproject_and_requirements_pins_agree(self):
        deps = set(pyproject()["project"]["dependencies"])
        pins = {line.strip() for line in
                (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
                if line.strip()}
        self.assertEqual(pins, deps)
        self.assertTrue(any(pin.startswith("numpy==") for pin in pins))
        self.assertTrue(any(pin.startswith("scipy==") for pin in pins))


def git(cwd: Path, *args: str) -> str:
    """Run git in ``cwd`` and return stdout (raises on a non-zero exit)."""
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout


class TestDistArchivesHead(unittest.TestCase):
    """m03: ``make dist`` builds from HEAD, never from the working tree."""

    def setUp(self):
        if shutil.which("git") is None or shutil.which("make") is None:
            self.skipTest("git and make are required for the dist contract")
        if not (ROOT / ".git").exists():
            self.skipTest("not a git checkout (extracted release archive)")
        self.tmp = Path(tempfile.mkdtemp(prefix="dist-contract-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        # Fixture = this checkout's tracked files, committed as-is. Untracked
        # scratch files (and build/, .venv/) are deliberately not copied.
        listed = git(ROOT, "ls-files", "-z").split("\0")
        for rel in filter(None, listed):
            source, target = ROOT / rel, self.repo / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "dist-contract@example.invalid")
        self.git("config", "user.name", "dist contract")
        self.git("add", "-A")
        self.git("-c", "commit.gpgsign=false", "commit", "-q", "-m", "fixture")

    def git(self, *args: str) -> str:
        return git(self.repo, *args)

    def build(self) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["make", "dist", f"PY={sys.executable}"], cwd=self.repo,
            capture_output=True, text=True,
        )

    def archived_files(self) -> set[str]:
        archive = self.repo / "build" / "dist" / f"{PREFIX}.tar.gz"
        self.assertTrue(archive.is_file(), "make dist produced no archive")
        names = set()
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                if member.isdir() or not member.name.startswith(PREFIX + "/"):
                    continue
                names.add(member.name[len(PREFIX) + 1:])
        return names

    def test_untracked_file_is_excluded_and_archive_still_verifies(self):
        (self.repo / "scratch_note.md").write_text("local scratch\n",
                                                   encoding="utf-8")
        proc = self.build()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        archived = self.archived_files()
        self.assertNotIn("scratch_note.md", archived)
        tracked = set(self.git("ls-tree", "-r", "--name-only", "HEAD").split())
        self.assertEqual(archived, tracked,
                         "archive file list must equal the HEAD tree")
        self.assertIn("verify: OK", proc.stdout)

    def test_modified_tracked_file_aborts_the_build(self):
        (self.repo / "README.md").write_text("edited in the working tree\n",
                                            encoding="utf-8")
        proc = self.build()
        self.assertNotEqual(proc.returncode, 0,
                            "make dist must refuse a dirty tracked tree")
        self.assertIn("tracked files are modified", proc.stdout)
        self.assertFalse(
            (self.repo / "build" / "dist" / f"{PREFIX}.tar.gz").exists(),
            "no archive may be produced from a dirty tracked tree")

    def test_makefile_no_longer_stages_the_working_tree(self):
        recipe = (ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertNotIn("git add -A", recipe)
        self.assertNotIn("read-tree HEAD", recipe)
        self.assertIn("git archive", recipe)
        self.assertIn("HEAD", recipe)
        self.assertIn("verify_release.py --root", recipe)


def _verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_release_metadata", ROOT / "scripts" / "verify_release.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFIER = _verifier()
CFF = VERIFIER.parse_cff(ROOT / "CITATION.cff")
ZENODO = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
README = (ROOT / "README.md").read_text(encoding="utf-8")
README_FLAT = " ".join(README.lower().split())


class TestCitationLicenceContract(unittest.TestCase):
    """P2.4/R2C9: split code/deposit licences, pending authors, title and floor."""

    def test_cff_code_licence_is_valid_spdx_mit(self):
        self.assertEqual(CFF["license"], "MIT")
        self.assertNotEqual(CFF["license"], "other-open")

    def test_zenodo_deposit_licence_stays_other_open(self):
        self.assertEqual(ZENODO["license"], "other-open")

    def test_metadata_state_the_code_only_carve_out(self):
        abstract = " ".join((CFF.get("abstract") or "").split()).lower()
        message = " ".join((CFF.get("message") or "").split()).lower()
        notes = " ".join((ZENODO.get("notes") or "").split()).lower()
        self.assertIn("not mit", abstract)
        self.assertIn("code only", message)
        self.assertIn("not mit", notes)
        self.assertIn("other-open", notes)

    def test_cff_validates_against_the_schema_when_available(self):
        if shutil.which("cffconvert") is None:
            self.skipTest("cffconvert is not installed")
        proc = subprocess.run(
            ["cffconvert", "--validate", "-i", "CITATION.cff"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_pyproject_code_licence_is_mit(self):
        self.assertEqual(pyproject()["project"]["license"]["text"], "MIT")

    def test_license_file_grants_mit_for_code_only(self):
        text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("MIT License", text)
        self.assertIn("covers the source code only", text.lower())
        self.assertIn("not MIT-licensed", text)

    def test_notice_carves_out_the_aggregate_results(self):
        text = (ROOT / "NOTICE").read_text(encoding="utf-8")
        self.assertIn("not MIT-licensed", text)
        self.assertIn("doi:10.7910/DVN/28075", text)

    def test_titles_match_the_manuscript_title(self):
        self.assertEqual(CFF["title"], VERIFIER.RELEASE_TITLE)
        self.assertEqual(ZENODO["title"], VERIFIER.RELEASE_TITLE)
        self.assertEqual(pyproject()["project"]["description"].rstrip("."),
                         VERIFIER.RELEASE_TITLE)
        self.assertIn("one actor signal, near-collinearity, and a within-panel "
                      "stability audit", README_FLAT)

    def test_no_superseded_title_framing(self):
        old_framings = (
            "exact attribution, " + "near-collinearity",
            "portability " + "audit",
        )
        for name in ("README.md", "CITATION.cff", ".zenodo.json", "pyproject.toml",
                     "notebooks/README.md"):
            text = (ROOT / name).read_text(encoding="utf-8").lower()
            for old_framing in old_framings:
                self.assertNotIn(old_framing, text, name)

    def test_no_unqualified_registration_status_survives(self):
        banned = (
            "pre-" + "registered primary",
            "pre-" + "registered secondary",
            "pre-" + "registered confirmatory",
            "pre-" + "registered one-shot",
            "pre-" + "registered holdout",
        )
        surfaces = (
            "README.md", "CITATION.cff", ".zenodo.json", "REPRODUCING.md",
            "results/README.md", "results/framework_pilot/holdout/report.md",
            "results/framework_pilot/holdout/summary.json",
            "results/framework_pilot/holdout/parts/holdout_extras.json",
            "experiments/STABILITY_VARIANT.md",
            "notebooks/01_confirmatory_holdout.ipynb",
            "notebooks/02_stability_audit.ipynb",
        )
        offenders = []
        for rel in surfaces:
            text = (ROOT / rel).read_text(encoding="utf-8").lower()
            offenders += [f"{rel}: {phrase}" for phrase in banned if phrase in text]
        self.assertEqual(offenders, [])

    def test_claim_boundary_carries_the_qualifier(self):
        readme = " ".join(
            (ROOT / "results" / "README.md").read_text(encoding="utf-8").lower().split()
        )
        self.assertIn("internally hash-timestamped", readme)
        self.assertIn("only the split has an independently written pre-run timestamp",
                      readme)
        self.assertIn("plan-declared primary", readme)

    def test_author_metadata_is_pending_and_not_invented(self):
        names = [row["name"] for row in CFF["authors"]]
        zen_names = [row["name"] for row in ZENODO["creators"]]
        self.assertEqual(names, zen_names)
        self.assertIn("pending", CFF["abstract"].lower())
        self.assertIn("pending", ZENODO["notes"].lower())
        payload = json.dumps([CFF, ZENODO]) + README
        self.assertIsNone(VERIFIER.ORCID_RE.search(payload))
        self.assertIsNone(VERIFIER.DOI_RE.search(json.dumps([CFF, ZENODO])))

    def test_required_test_modules_and_floor(self):
        modules = sorted(path.name for path in (ROOT / "tests").glob("test_*.py"))
        self.assertEqual(modules, sorted([
            "test_attribution.py", "test_features_contract.py",
            "test_inference_units.py", "test_metadata_contract.py",
            "test_models.py", "test_notebooks.py", "test_results_contract.py",
            "test_smoke.py", "test_splits_contract.py",
        ]))
        self.assertGreaterEqual(VERIFIER._count_tests(ROOT),
                                VERIFIER.MIN_TEST_COUNT)

    def test_metadata_contract_is_clean(self):
        self.assertEqual(VERIFIER.check_metadata_contract(ROOT), [])

    def test_metadata_contract_flags_a_licence_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            for name in ("CITATION.cff", ".zenodo.json", "pyproject.toml",
                         "LICENSE", "NOTICE", "README.md"):
                shutil.copy2(ROOT / name, target / name)
            (target / "notebooks").mkdir()
            shutil.copy2(ROOT / "notebooks" / "README.md",
                         target / "notebooks" / "README.md")
            shutil.copytree(ROOT / "tests", target / "tests",
                            ignore=shutil.ignore_patterns("__pycache__"))
            cff = (target / "CITATION.cff").read_text(encoding="utf-8")
            (target / "CITATION.cff").write_text(
                cff.replace("license: MIT", "license: other-open"), encoding="utf-8")
            errors = VERIFIER.check_metadata_contract(target)
            self.assertTrue(any("license" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
