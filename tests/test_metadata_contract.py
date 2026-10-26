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


if __name__ == "__main__":
    unittest.main()
