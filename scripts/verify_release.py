#!/usr/bin/env python3
"""Verify the distributable public release.

The verifier is intentionally restricted to files that are actually shipped.
It never reads ICEWS data, experiment configuration, caches or row-level
predictions, and it fails if any of those appear in the tree.

Checks
------
* required public files exist, with matching CFF/Zenodo/pyproject licence and
  title metadata, pending author metadata, and the required test modules/floor;
* no ``paper/``, ``docs/``, ``configs/``, ``data/``, ``preds/``, caches or logs;
* every ``results/**/*.json`` is strict JSON (no ``NaN``/``Infinity``);
* ``results/framework_pilot/holdout/SHA256SUMS`` and
  ``results/framework_pilot/stability/SHA256SUMS`` cover every distributed
  file in their directories and every digest matches;
* the stability scope is the frozen 50-refit, C=1.0, holdout-excluded design;
* ``experiments/STABILITY_VARIANT.md`` exists and no public document claims a
  five-seed audit or the legacy ``provisional_no_fresh_holdout`` status;
* the pre-registration checksum uses a repo-relative path and matches;
* manifest entries for non-redistributed predictions/logs are marked
  ``redistributed: false`` and the files are absent;
* every ``manifest.code`` entry has a ``code_annotations.current_sha256``
  digest equal to the shipped bytes, and any post-freeze change is annotated;
* all three notebooks contain executed, error-free, aggregate-only outputs;
* no absolute home/temporary path or credential-like string appears in text;
* no forbidden path is tracked by git (when the verifier runs in a git tree).

Usage::

    python3 scripts/verify_release.py
    python3 scripts/verify_release.py --root build/dist/verify
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HOLDOUT_REL = Path("results/framework_pilot/holdout")
STABILITY_REL = Path("results/framework_pilot/stability")
STABILITY_VARIANT_REL = Path("experiments/STABILITY_VARIANT.md")
STABILITY_DOCS = (
    "README.md",
    "REPRODUCING.md",
    "CODE_MAP.md",
    "RELEASE_NOTES.md",
    "results/README.md",
    "notebooks/README.md",
    "CITATION.cff",
    ".zenodo.json",
)
FORBIDDEN_DIRS = {
    "paper",
    "docs",
    "configs",
    "data",
    "preds",
    "__pycache__",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".git",
}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".npz", ".npy", ".parquet", ".log"}
TEXT_SUFFIXES = {
    ".md", ".py", ".json", ".csv", ".cff", ".yml", ".yaml", ".toml",
    ".txt", ".cfg", ".ini", ".ipynb", ".tex", ".in",
}
TEXT_NAMES = {
    "Makefile", "NOTICE", "LICENSE", "CITATION.cff", ".gitignore",
    ".gitattributes", "requirements.txt", "requirements.lock",
}
SKIP_DIRS = {".git", ".venv", "venv", "build", "dist", "node_modules",
             ".tox", ".nox", "site-packages"}
REQUIRED = [
    "README.md",
    "REPRODUCING.md",
    "CODE_MAP.md",
    "DATA_EXCLUSIONS.md",
    "NOTICE",
    "LICENSE",
    "CITATION.cff",
    ".zenodo.json",
    ".gitignore",
    ".gitattributes",
    "Makefile",
    "pyproject.toml",
    "requirements.txt",
    "requirements.lock",
    "scripts/verify_release.py",
    "scripts/execute_notebooks.py",
    "scripts/check_aggregates.py",
    "scripts/doctor.py",
    "framework/__init__.py",
    "framework/aggregation.py",
    "framework/bootstrap.py",
    "framework/config.py",
    "framework/data.py",
    "framework/features.py",
    "framework/metrics.py",
    "framework/models.py",
    "framework/regions.py",
    "framework/splits.py",
    "framework/sprint_a.py",
    "framework/sprint_a_inference.py",
    "framework/utils.py",
    "tests/test_attribution.py",
    "tests/test_smoke.py",
    "tests/test_features_contract.py",
    "tests/test_splits_contract.py",
    "tests/test_models.py",
    "tests/test_results_contract.py",
    "tests/test_notebooks.py",
    "tests/test_inference_units.py",
    "tests/test_metadata_contract.py",
    "notebooks/README.md",
    "notebooks/01_confirmatory_holdout.ipynb",
    "notebooks/02_stability_audit.ipynb",
    "notebooks/03_worldline_preview.ipynb",
    "results/README.md",
    str(HOLDOUT_REL / "SHA256SUMS"),
    str(HOLDOUT_REL / "manifest.json"),
    str(HOLDOUT_REL / "report.md"),
    str(HOLDOUT_REL / "summary.json"),
    str(HOLDOUT_REL / "preregistration.json"),
    str(HOLDOUT_REL / "preregistration.json.sha256"),
    str(HOLDOUT_REL / "run_once.json"),
    str(HOLDOUT_REL / "test_access_log.json"),
    str(HOLDOUT_REL / "CORRECTIONS.md"),
    "experiments/STABILITY_VARIANT.md",
    "results/framework_pilot/stability/SHA256SUMS",
    "results/framework_pilot/stability/report.md",
    "results/framework_pilot/stability/summary.json",
    "results/framework_pilot/stability/summary_decompose.json",
    "results/framework_pilot/stability/summary_hgb.json",
    "results/framework_pilot/stability/summary_stability.json",
    "results/worldline_preview/README.md",
    "results/worldline_preview/run_manifest.json",
    "results/worldline_preview/worldline_trace.json",
    "results/worldline_preview/worldline_preview.pdf",
    "results/worldline_preview/worldline_preview.png",
]

ABS_PATHS = ("/" + "home/", "/" + "tmp/", "/" + "Users/", "C:" + "\\Users")
SECRET_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(
        r"(?i)(api[_-]?key|secret|password|passwd|access[_-]?token|bearer)"
        r"\s*[:=]\s*[\"'][^\"']{8,}"
    ),
)
SUM_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")

# The public world-line preview is a standalone, repo-only illustration: its
# digests and geometry are pinned so a replaced or superseded render fails, but
# the check makes no claim that the preview is a manuscript exhibit (S01).
# 150 mm = 425.197 pt wide, Arimo, no baked key.
WG_PDF_MEDIABOX = b"/MediaBox [ 0 0 425.1968503937 330 ]"
WG_PDF_SHA256 = "77081960c20db85ff3748cc06edd3d38ceb9cf69fb93687ff612f5deec8bb3b1"
WG_PNG_SHA256 = "3b6e0fb29706a5c8af1bf219120d4c125467bc37ecc9ad48f25bd3df046e02d4"
# S01: the preview is repo-only. These phrases would re-assert that it is a
# manuscript exhibit, which the paper does not contain.
PAPER_EXHIBIT_ASSERTIONS = (
    "paper's Figure A1", "paper Figure A1",
    "paper's Appendix A1", "paper Appendix A1",
    "figA1_worldline", "A1 print version", "A1 render",
)

# Row-level / named-rate boundary (see DATA_EXCLUSIONS.md). The world-line
# preview ships a rendered illustration and aggregate JSON only; these per-row
# exports were removed and must not reappear.
WORLDLINE_WITHHELD = {
    "results/worldline_preview/worldline_trace.csv",
    "results/worldline_preview/node_trace.csv",
}
PREDICTION_COLUMNS = {
    "y", "p", "mean_p", "p_sum", "y_sum", "mean_y", "p_model",
}
IDENTIFIER_COLUMNS = {
    "did", "sid", "tid", "aid", "actor", "actor_label", "source", "target",
    "source_actor", "target_actor", "source_aid", "target_aid", "dyad",
    "dyad_label", "month",
}
RATE_JSON_KEYS = {"mean_p", "p_sum", "y_sum", "mean_y", "p_model"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_load(path: Path) -> dict:
    def reject(token: str):
        raise ValueError(f"non-finite JSON constant {token!r}")

    obj = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    json.dumps(obj, allow_nan=False)
    return obj


def check_required(root: Path) -> list[str]:
    missing = [rel for rel in REQUIRED if not (root / rel).is_file()]
    return [f"required file missing: {rel}" for rel in missing]


# ---------------------------------------------------------------------------
# Citation / licence / test-suite contract (no PyYAML dependency)
# ---------------------------------------------------------------------------
RELEASE_TITLE = (
    "Actor-role components of dyadic event forecasts: exact attribution, "
    "near-collinearity, and a within-panel stability audit"
)
MIN_TEST_COUNT = 123
ORCID_RE = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b")
DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"']+")
CFF_LINE = re.compile(r"^([A-Za-z0-9_.-]+):\s*(.*)$")


def _cff_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _cff_sequence(block: list[str]) -> list:
    items: list = []
    current: dict | None = None
    for line in block:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            rest = stripped[2:].strip()
            if not rest:
                raise ValueError("empty sequence item")
            if ":" in rest and not rest.startswith(("'", '"')):
                key, _, value = rest.partition(":")
                current = {key.strip(): _cff_scalar(value)}
                items.append(current)
            else:
                current = None
                items.append(_cff_scalar(rest))
        else:
            if current is None:
                raise ValueError(f"unsupported sequence continuation {stripped!r}")
            key, sep, value = stripped.partition(":")
            if not sep:
                raise ValueError(f"unsupported sequence line {stripped!r}")
            current[key.strip()] = _cff_scalar(value)
    return items


def parse_cff(path: Path) -> dict:
    """Parse the YAML subset used by ``CITATION.cff``.

    Supported: top-level block mappings, quoted/plain scalars, folded or literal
    block scalars, and sequences of scalars or single-level mappings. Tabs, flow
    style, anchors, nested mappings and multi-document files are rejected so a
    malformed CFF edit cannot pass silently. This deliberately avoids a PyYAML
    runtime dependency; the CFF is a flat metadata file.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    top: dict = {}
    i = 0
    while i < len(lines):
        raw = lines[i]
        if "\t" in raw:
            raise ValueError(f"line {i + 1}: tab indentation is not supported")
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        if raw[:1].isspace():
            raise ValueError(f"line {i + 1}: unexpected indentation")
        if stripped in ("---", "..."):
            raise ValueError("multi-document YAML is not supported")
        match = CFF_LINE.match(stripped)
        if not match:
            raise ValueError(f"line {i + 1}: unsupported syntax {stripped!r}")
        key, rest = match.group(1), match.group(2).strip()
        i += 1
        block: list[str] = []
        while i < len(lines):
            line = lines[i]
            if not line.strip() or line.strip().startswith("#"):
                i += 1
                continue
            if not line[:1].isspace():
                break
            if "\t" in line:
                raise ValueError(f"line {i + 1}: tab indentation is not supported")
            block.append(line)
            i += 1
        if rest in (">-", ">", "|", "|-"):
            joiner = " " if rest.startswith(">") else "\n"
            top[key] = joiner.join(line.strip() for line in block)
        elif rest == "":
            if not block:
                top[key] = None
            elif block[0].strip().startswith("- "):
                top[key] = _cff_sequence(block)
            else:
                raise ValueError(f"key {key!r}: nested mappings are not supported")
        else:
            if block:
                raise ValueError(f"key {key!r}: unexpected indented block")
            if rest.startswith(("{", "[")) or rest[0] in "&*!":
                raise ValueError(f"key {key!r}: flow/anchored scalars are not supported")
            top[key] = _cff_scalar(rest)
    return top


def _count_tests(root: Path) -> int:
    import ast

    total = 0
    for path in sorted((root / "tests").glob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
        total += sum(
            1 for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        )
    return total


def check_metadata_contract(root: Path) -> list[str]:
    """CFF/Zenodo/pyproject/licence agreement and the test-suite floor.

    The deposit licence is ``other-open`` because the tree mixes MIT code with
    ICEWS-derived aggregates that are not MIT; ``pyproject.toml`` and ``LICENSE``
    keep the explicit code-only MIT grant, the public title must match the
    manuscript, author metadata stays marked pending, and the test suite keeps
    its required modules and floor.
    """
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python < 3.11 is unsupported
        return ["tomllib is unavailable; Python 3.11+ is required"]

    errors: list[str] = []
    try:
        cff = parse_cff(root / "CITATION.cff")
    except Exception as exc:
        return [f"CITATION.cff does not parse: {exc}"]
    try:
        zenodo = strict_load(root / ".zenodo.json")
    except Exception as exc:
        return [f".zenodo.json does not parse: {exc}"]
    try:
        pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"pyproject.toml does not parse: {exc}"]

    cff_license = cff.get("license")
    zen_license = zenodo.get("license")
    if cff_license != zen_license:
        errors.append(
            f"CITATION.cff license {cff_license!r} != Zenodo license {zen_license!r}"
        )
    if cff_license != "other-open":
        errors.append(
            f"CITATION.cff license must be 'other-open' for the mixed deposit, got {cff_license!r}"
        )
    project = pyproject.get("project", {})
    code_license = project.get("license")
    if isinstance(code_license, dict):
        code_license = code_license.get("text")
    if code_license != "MIT":
        errors.append(f"pyproject.toml license must stay MIT, got {code_license!r}")

    license_text = (root / "LICENSE").read_text(encoding="utf-8")
    if "MIT License" not in license_text:
        errors.append("LICENSE is not the MIT text")
    if "covers the source code only" not in license_text.lower():
        errors.append("LICENSE lacks the code-only carve-out")
    if "not MIT-licensed" not in license_text:
        errors.append("LICENSE does not exclude the ICEWS-derived aggregates")

    notice = (root / "NOTICE").read_text(encoding="utf-8")
    if "not MIT-licensed" not in notice:
        errors.append("NOTICE does not exclude the ICEWS-derived aggregates")
    if "doi:10.7910/DVN/28075" not in notice:
        errors.append("NOTICE lacks the ICEWS v37 citation")

    if cff.get("title") != RELEASE_TITLE:
        errors.append(f"CITATION.cff title is not the manuscript title: {cff.get('title')!r}")
    if zenodo.get("title") != RELEASE_TITLE:
        errors.append(f".zenodo.json title is not the manuscript title: {zenodo.get('title')!r}")
    if str(project.get("description", "")).rstrip(".") != RELEASE_TITLE:
        errors.append(
            f"pyproject.toml description is not the manuscript title: {project.get('description')!r}"
        )
    old_framing = "portability " + "audit"
    for name in ("README.md", "CITATION.cff", ".zenodo.json", "pyproject.toml",
                 "notebooks/README.md"):
        path = root / name
        if not path.is_file():
            errors.append(f"{name} is missing")
            continue
        if old_framing in path.read_text(encoding="utf-8").lower():
            errors.append(f"{name} retains the superseded title framing")

    readme = (root / "README.md").read_text(encoding="utf-8")
    readme_flat = " ".join(readme.lower().split())
    if "near-collinearity, and a within-panel stability audit" not in readme_flat:
        errors.append("README.md subtitle does not carry the current title framing")
    versions = {
        "CITATION.cff": cff.get("version"),
        ".zenodo.json": zenodo.get("version"),
        "pyproject.toml": project.get("version"),
    }
    for name, value in versions.items():
        if value != "0.1.1":
            errors.append(f"{name} version is {value!r}, expected '0.1.1'")
    if "`v0.1.1` git tag" not in readme:
        errors.append("README.md does not state that no v0.1.1 tag exists")
    if "no doi has been minted" not in readme_flat:
        errors.append("README.md does not state that no DOI exists")
    if "do not cite this tree as an archival release" not in readme_flat:
        errors.append("README.md does not flag the review candidate as non-archival")

    cff_authors = [row.get("name") for row in cff.get("authors") or [] if isinstance(row, dict)]
    zen_authors = [row.get("name") for row in zenodo.get("creators") or [] if isinstance(row, dict)]
    if cff_authors != zen_authors:
        errors.append(f"CFF authors {cff_authors} != Zenodo creators {zen_authors}")
    if not cff_authors or not zen_authors:
        errors.append("CFF/Zenodo author lists are empty")
    if "pending" not in (cff.get("abstract") or "").lower():
        errors.append("CITATION.cff does not mark author metadata as pending")
    if "pending" not in (zenodo.get("notes") or "").lower():
        errors.append(".zenodo.json notes do not mark author metadata as pending")
    for name, payload in (("CITATION.cff", json.dumps(cff)),
                          (".zenodo.json", json.dumps(zenodo))):
        if ORCID_RE.search(payload):
            errors.append(f"{name} contains an ORCID; author metadata is pending")
        if DOI_RE.search(payload):
            errors.append(f"{name} contains a DOI; no DOI has been minted")

    expected_modules = sorted(rel for rel in REQUIRED if rel.startswith("tests/"))
    present = sorted(
        path.relative_to(root).as_posix()
        for path in (root / "tests").glob("test_*.py")
    )
    if present != expected_modules:
        errors.append(f"test modules changed: {present} != {expected_modules}")
    count = None
    try:
        count = _count_tests(root)
    except Exception as exc:
        errors.append(f"cannot count the public test suite: {exc}")
    if count is not None and count < MIN_TEST_COUNT:
        errors.append(
            f"public test suite shrank to {count} tests (minimum {MIN_TEST_COUNT})"
        )
    return errors


def check_forbidden(root: Path) -> list[str]:
    errors = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.is_dir() and path.name in FORBIDDEN_DIRS:
            errors.append(f"forbidden directory: {rel.as_posix()}")
        if path.is_file() and path.suffix in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden file type: {rel.as_posix()}")
    return errors


def _json_keys(obj) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            keys.add(str(key))
            keys |= _json_keys(value)
    elif isinstance(obj, list):
        for value in obj:
            keys |= _json_keys(value)
    return keys


def check_row_level_data(root: Path) -> list[str]:
    """Fail on row-level labels/predictions or named actor probability/rates.

    The world-line preview ships no row-level file: its two per-row CSVs are
    withheld, so their reappearance under any name is reported here. See
    DATA_EXCLUSIONS.md.
    """
    errors: list[str] = []
    results = root / "results"
    if not results.is_dir():
        return ["results/ is missing"]
    for rel in sorted(WORLDLINE_WITHHELD):
        if (root / rel).is_file():
            errors.append(f"row-level world-line export is present: {rel}")
    for path in sorted(results.rglob("*.csv")):
        rel = path.relative_to(root).as_posix()
        if "dyad_contributions" in path.name:
            errors.append(f"row-level dyad contribution file is present: {rel}")
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), [])
        columns = {str(column).strip().lower() for column in header}
        prediction = columns & PREDICTION_COLUMNS
        identifier = columns & IDENTIFIER_COLUMNS
        if {"y", "p"} <= columns:
            errors.append(f"row-level label+probability columns in {rel}")
        elif prediction and identifier:
            errors.append(
                f"prediction/label columns {sorted(prediction)} with unit "
                f"identifiers in {rel}"
            )
        elif prediction & {"mean_p", "p_sum", "y_sum", "mean_y"}:
            errors.append(
                f"actor-level predicted/observed rate columns "
                f"{sorted(prediction & RATE_JSON_KEYS)} in {rel}"
            )
    for path in sorted(results.rglob("*.json")):
        rel = path.relative_to(root).as_posix()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"cannot scan JSON {rel}: {exc}")
            continue
        bad = _json_keys(payload) & RATE_JSON_KEYS
        if bad:
            errors.append(f"actor-level predicted/observed rate keys {sorted(bad)} in {rel}")
    return errors


def check_strict_json(root: Path) -> list[str]:
    errors = []
    paths = sorted(list((root / "results").rglob("*.json")) + [root / ".zenodo.json"])
    for path in paths:
        rel = path.relative_to(root).as_posix()
        try:
            strict_load(path)
        except Exception as exc:
            errors.append(f"invalid strict JSON {rel}: {exc}")
    return errors


def parse_sums(path: Path) -> tuple[dict[str, str], list[str]]:
    entries: dict[str, str] = {}
    errors: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = SUM_LINE.match(line.rstrip())
        if not match:
            errors.append(f"{path.name}:{lineno}: malformed checksum line")
            continue
        digest, rel = match.groups()
        candidate = Path(rel)
        if candidate.is_absolute() or ".." in candidate.parts:
            errors.append(f"{path.name}:{lineno}: non-relative path {rel!r}")
            continue
        entries[rel] = digest
    return entries, errors


def check_tree_sums(root: Path, rel_dir: Path, label: str) -> list[str]:
    base = root / rel_dir
    sums_path = base / "SHA256SUMS"
    if not sums_path.is_file():
        return [f"{label}/SHA256SUMS missing"]
    entries, errors = parse_sums(sums_path)
    if "SHA256SUMS" in entries:
        errors.append(f"{label}/SHA256SUMS must not checksum itself")
    actual = {
        path.relative_to(root).as_posix()
        for path in base.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    missing = sorted(actual - set(entries))
    extra = sorted(set(entries) - actual)
    errors += [f"{label}/SHA256SUMS does not cover {rel}" for rel in missing]
    errors += [f"{label}/SHA256SUMS lists absent file {rel}" for rel in extra]
    for rel, digest in sorted(entries.items()):
        path = root / rel
        if path.is_file() and sha256_file(path) != digest:
            errors.append(f"{label} checksum mismatch: {rel}")
    return errors


def check_prereg_checksum(root: Path) -> list[str]:
    base = root / HOLDOUT_REL
    line = (base / "preregistration.json.sha256").read_text(encoding="utf-8").strip()
    parts = line.split()
    if len(parts) != 2:
        return ["preregistration.json.sha256 must be '<digest>  <repo-relative path>'"]
    digest, rel = parts
    if Path(rel).is_absolute() or ".." in Path(rel).parts:
        return [f"preregistration checksum path is not repo-relative: {rel!r}"]
    target = root / rel
    if not target.is_file():
        return [f"preregistration checksum target missing: {rel}"]
    if sha256_file(target) != digest:
        return ["preregistration checksum mismatch"]
    summary = strict_load(base / "summary.json")
    recorded = summary.get("holdout", {}).get("prereg_file_sha256")
    if recorded and recorded != digest:
        return ["summary.json prereg hash disagrees with preregistration.json.sha256"]
    return []


def check_manifest(root: Path) -> list[str]:
    base = root / HOLDOUT_REL
    manifest = strict_load(base / "manifest.json")
    outputs = manifest.get("outputs", {})
    current = manifest.get("release_annotations", {}).get("current_sha256", {})
    errors = []
    excluded = 0
    for rel, meta in outputs.items():
        path = root / rel
        if meta.get("redistributed", True) is False:
            excluded += 1
            if path.exists():
                errors.append(f"non-redistributed file is present: {rel}")
            if not meta.get("exclusion_reason"):
                errors.append(f"missing exclusion_reason for {rel}")
            continue
        if not path.is_file():
            errors.append(f"manifest output missing: {rel}")
            continue
        expected = current.get(rel, meta.get("sha256"))
        if expected and sha256_file(path) != expected:
            errors.append(f"manifest hash mismatch: {rel}")
    if excluded != 4:
        errors.append(f"expected 4 non-redistributed outputs, found {excluded}")
    annotations = manifest.get("release_annotations", {})
    if annotations.get("numbers_changed") is not False:
        errors.append("manifest release_annotations.numbers_changed must be false")
    inputs = manifest.get("inputs", {})
    for rel, digest in inputs.items():
        path = root / rel
        if "preds/" in rel:
            if path.exists():
                errors.append(f"row-level prediction input is present: {rel}")
            continue
        if not path.is_file():
            errors.append(f"manifest input missing: {rel}")
        elif sha256_file(path) != digest:
            errors.append(f"manifest input hash mismatch: {rel}")
    return errors


SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def check_manifest_code(root: Path) -> list[str]:
    """Recompute every manifest code hash against the shipped bytes.

    ``manifest.code`` records the digest of the bytes actually run at freeze
    time.  ``release_annotations.code_annotations.current_sha256`` records the
    bytes shipped at this release tag.  A file may legitimately differ between
    the two, but only when the difference is listed in
    ``code_annotations.changed_since_freeze``; every ``code`` entry must have a
    current-hash annotation and every annotated value must equal the shipped
    file.  The ``p_display`` block documents the post-freeze inference-code
    change and is checked against the file it names.
    """
    base = root / HOLDOUT_REL
    manifest = strict_load(base / "manifest.json")
    code = manifest.get("code") or {}
    annotations = (
        manifest.get("release_annotations", {}).get("code_annotations") or {}
    )
    current = annotations.get("current_sha256") or {}
    changed = annotations.get("changed_since_freeze") or {}
    errors: list[str] = []
    if not code:
        return ["manifest code block is missing or empty"]
    if not current:
        return ["manifest code_annotations.current_sha256 is missing or empty"]
    for rel, freeze_digest in sorted(code.items()):
        if not SHA256_HEX.fullmatch(str(freeze_digest)):
            errors.append(f"manifest code hash is not a sha256 digest: {rel}")
            continue
        path = root / rel
        if not path.is_file():
            errors.append(f"manifest code file missing: {rel}")
            continue
        shipped = sha256_file(path)
        if rel not in current:
            errors.append(
                f"manifest code entry has no current_sha256 annotation: {rel}")
            continue
        if current[rel] != shipped:
            errors.append(f"manifest current code hash mismatch: {rel}")
        if freeze_digest != shipped and rel not in changed:
            errors.append(
                f"manifest code changed since freeze without annotation: {rel}")
    for rel in sorted(set(current) - set(code)):
        errors.append(
            f"manifest code_annotations entry has no code hash: {rel}")
    p_display = annotations.get("p_display")
    if not isinstance(p_display, dict):
        errors.append("manifest code_annotations.p_display is missing")
    else:
        files = p_display.get("files") or []
        if not files:
            errors.append("manifest code_annotations.p_display has no files")
        for rel in files:
            if rel not in code:
                errors.append(
                    f"manifest p_display file is not a code entry: {rel}")
        for symbol in p_display.get("symbols") or []:
            found = False
            for rel in files:
                path = root / rel
                if path.is_file() and f"def {symbol}(" in path.read_text(
                        encoding="utf-8"):
                    found = True
            if not found:
                errors.append(
                    f"manifest p_display symbol is not defined: {symbol}")
    return errors


def check_text(root: Path) -> list[str]:
    errors = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in TEXT_NAMES:
            continue
        if path.stat().st_size > 4 * 1024 * 1024:
            errors.append(f"unexpectedly large text file: {rel.as_posix()}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in ABS_PATHS:
            if token in text:
                errors.append(f"absolute path {token!r} in {rel.as_posix()}")
        for pattern in SECRET_PATTERNS:
            match = pattern.search(text)
            if match:
                errors.append(
                    f"credential-like string in {rel.as_posix()}: "
                    f"{match.group(0)[:32]!r}"
                )
    return errors


def check_notebooks(root: Path) -> list[str]:
    errors = []
    paths = sorted((root / "notebooks").glob("*.ipynb"))
    if len(paths) != 3:
        return [f"expected 3 notebooks, found {len(paths)}"]
    for path in paths:
        rel = path.relative_to(root).as_posix()
        nb = strict_load(path)
        if nb.get("nbformat") != 4:
            errors.append(f"{rel}: nbformat is not 4")
        code = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        if not code:
            errors.append(f"{rel}: no code cells")
            continue
        image_outputs = 0
        for index, cell in enumerate(code):
            if not isinstance(cell.get("execution_count"), int):
                errors.append(f"{rel}: code cell {index} is not executed")
            outputs = cell.get("outputs", [])
            if not outputs:
                errors.append(f"{rel}: code cell {index} has no outputs")
            for output in outputs:
                if output.get("output_type") == "error":
                    errors.append(f"{rel}: code cell {index} has an error output")
                if "image/png" in output.get("data", {}):
                    image_outputs += 1
        if path.name in ("01_confirmatory_holdout.ipynb",
                         "03_worldline_preview.ipynb") and image_outputs == 0:
            errors.append(f"{rel}: expected an embedded PNG output")
    return errors


def check_stability_docs(root: Path) -> list[str]:
    errors = []
    variant = root / STABILITY_VARIANT_REL
    if not variant.is_file():
        errors.append("experiments/STABILITY_VARIANT.md is missing")
    else:
        text = variant.read_text(encoding="utf-8")
        for token in ("N_SEEDS = 5", "50", "_holdout_dids", "5,282", "2,790",
                      "C_grid=[1.0]", "N_BOOT = 1000", "rep0"):
            if token not in text:
                errors.append(f"STABILITY_VARIANT.md lacks {token!r}")
    for rel in STABILITY_DOCS:
        path = root / rel
        if not path.is_file():
            errors.append(f"required stability document missing: {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        if "provisional_no_fresh_holdout" in text:
            errors.append(f"{rel} retains the legacy provisional_no_fresh_holdout status")
        if re.search(r"\bN_SEEDS\s*=\s*5\b", text):
            errors.append(f"{rel} claims N_SEEDS=5; the shipped audit uses 50")
    return errors


def check_legacy_stability_label(root: Path) -> list[str]:
    """Fail if the superseded pre-holdout audit label is still shipped.

    The 50-refit grid was frozen at ``2026-10-23T06:31:40Z``, i.e. 64 minutes
    after the confirmatory label read began (``05:26:55.031577Z``) and before it
    completed (``07:45:19Z``).  The truthful literal is
    ``frozen_during_confirmatory_run_holdout_rows_excluded``; the legacy string
    is accepted by nothing.
    """
    base = root / STABILITY_REL
    legacy = "pre_holdout_holdout_excluded"
    errors: list[str] = []
    for rel in ("report.md", "summary.json", "summary_decompose.json",
                "summary_hgb.json", "summary_stability.json"):
        path = base / rel
        if not path.is_file():
            continue
        if legacy in path.read_text(encoding="utf-8"):
            errors.append(f"stability/{rel} retains the legacy audit label {legacy!r}")
    return errors


def check_stability(root: Path) -> list[str]:
    base = root / STABILITY_REL
    required = ("report.md", "summary.json", "summary_decompose.json",
                "summary_hgb.json", "summary_stability.json")
    missing = [name for name in required if not (base / name).is_file()]
    if missing:
        return [f"stability/{name} missing" for name in missing]
    errors = []
    report = (base / "report.md").read_text(encoding="utf-8").lower()
    for phrase in ("frozen at", "holdout completed after", "not integrated",
                   "available_at_audit", "provisional"):
        if phrase not in report:
            errors.append(f"stability report lacks phrase: {phrase!r}")
    summary = strict_load(base / "summary.json")
    if summary.get("status") != "provisional_holdout_not_integrated":
        errors.append(f"unexpected stability status: {summary.get('status')!r}")
    scope = summary.get("scope", {})
    expected_scope = {
        "audit_design": "frozen_during_confirmatory_run_holdout_rows_excluded",
        "n_seed_replicates": 50,
        "model_C": 1.0,
        "holdout_excluded_directed_dyads": 5282,
        "holdout_excluded_unordered_pairs": 2790,
    }
    for key, value in expected_scope.items():
        if scope.get(key) != value:
            errors.append(f"stability scope.{key} changed: {scope.get(key)!r}")
    # The superseded "pre-holdout" label must never come back: the grid was
    # frozen after the confirmatory read began (see consensus R2C2 M01).
    errors += check_legacy_stability_label(root)
    holdout = summary.get("holdout", {})
    if holdout.get("available_at_audit") is not False:
        errors.append("stability holdout.available_at_audit must be false")
    if holdout.get("completed_later") is not True:
        errors.append("stability holdout.completed_later must be true")
    if holdout.get("used") is not False:
        errors.append("stability holdout.used must be false")
    if holdout.get("completed_utc") != "2026-10-23T07:45:19Z":
        errors.append("stability holdout.completed_utc is stale or missing")
    if holdout.get("confirmatory_run_start_utc") != "2026-10-23T05:26:55.031577Z":
        errors.append(
            "stability holdout.confirmatory_run_start_utc is stale or missing"
        )
    if holdout.get("design_frozen_utc") != "2026-10-23T06:31:40Z":
        errors.append("stability holdout.design_frozen_utc is stale or missing")
    if holdout.get("audit_design") != expected_scope["audit_design"]:
        errors.append(
            "stability holdout.audit_design disagrees with scope.audit_design"
        )
    for name in ("summary_decompose.json", "summary_hgb.json",
                 "summary_stability.json"):
        sub = strict_load(base / name)
        if sub.get("status") != summary.get("status"):
            errors.append(
                f"{name} status {sub.get('status')!r} disagrees with summary.json"
            )
    decompose = strict_load(base / "summary_decompose.json")
    if decompose.get("protocol", {}).get("n_seed_replicates") != 50:
        errors.append("summary_decompose.json protocol.n_seed_replicates must be 50")
    stability = strict_load(base / "summary_stability.json")
    if stability.get("protocol", {}).get("n_seed_replicates") != 50:
        errors.append("summary_stability.json protocol.n_seed_replicates must be 50")
    bootstrap = stability.get("bootstrap", [])
    if not bootstrap:
        errors.append("summary_stability.json has no bootstrap rows")
    else:
        wrong = {row.get("n_boot") for row in bootstrap
                 if row.get("n_boot") != 50}
        if wrong:
            values = sorted(wrong, key=str)
            errors.append(
                "summary_stability.json bootstrap n_boot values changed: "
                f"{values}"
            )
    errors += check_stability_docs(root)
    return errors


def check_headlines(root: Path) -> list[str]:
    base = root / HOLDOUT_REL
    summary = strict_load(base / "summary.json")
    errors = []
    if summary.get("status") != "confirmatory_full_holdout_completed":
        errors.append("holdout summary status changed")
    holdout = summary.get("holdout", {})
    expected = {"rows": 311638, "positives": 32614, "dyads": 5282}
    for key, value in expected.items():
        if holdout.get(key) != value:
            errors.append(f"holdout.{key} changed: {holdout.get(key)!r}")
    metrics = summary.get("primary", {}).get("metrics", {})
    frozen = {
        "log_loss": 0.217875,
        "brier": 0.06401,
        "ece": 0.042112,
        "auc": 0.910152,
        "ap": 0.671863,
    }
    for key, value in frozen.items():
        if metrics.get(key) != value:
            errors.append(f"primary metric {key} changed: {metrics.get(key)!r}")
    readme = (root / "README.md").read_text(encoding="utf-8")
    if "Δ = model − comparator" not in readme:
        errors.append("README.md lacks the single sign-convention statement")
    if "negative = better" not in readme:
        errors.append("README.md lacks the loss-metric direction")
    extras = strict_load(base / "parts/holdout_extras.json").get("summary", {})
    if extras != summary:
        errors.append("holdout_extras.json summary block disagrees with summary.json")
    prose = json.dumps(summary, ensure_ascii=False).lower()
    for stale in ("within-dyad signal present", "calendar specification, not graph capacity, drives"):
        if stale in prose:
            errors.append(f"holdout summary retains stale claim string: {stale!r}")
    return errors


def check_worldline_notice(root: Path) -> list[str]:
    text = (root / "results/worldline_preview/README.md").read_text(encoding="utf-8")
    lowered = text.lower()
    errors = []
    if "not print-legible" in lowered:
        errors.append("worldline README still calls the shipped figure not print-legible")
    for phrase in ("print version", "vector", "150", "truetype"):
        if phrase not in lowered:
            errors.append(f"worldline README lacks {phrase!r}")
    if "no row-level" not in lowered:
        errors.append("worldline README does not state the no-row-level boundary")
    if "164" in lowered:
        errors.append("worldline README still advertises the superseded 164 mm preview")
    if "not a manuscript exhibit" not in lowered:
        errors.append(
            "worldline README does not state that the preview is not a manuscript exhibit")
    # m01 (R2C3): the two remaining docs that advertised the superseded
    # 164 mm / 465x380 pt geometry must now state the 150 mm paper-A1 geometry.
    docs = {}
    for rel in ("CODE_MAP.md", "notebooks/README.md",
                "DATA_EXCLUSIONS.md", "results/README.md"):
        path = root / rel
        if path.is_file():
            docs[rel] = path.read_text(encoding="utf-8")
        else:
            errors.append(f"{rel} missing")
    for rel in ("CODE_MAP.md", "notebooks/README.md"):
        doc = docs.get(rel)
        if doc is None:
            continue
        if "164" in doc:
            errors.append(f"{rel} still advertises the superseded 164 mm preview")
        if "465 × 380" in doc or "465x380" in doc:
            errors.append(f"{rel} still advertises the 465 x 380 pt preview")
        if "150 mm" not in doc:
            errors.append(f"{rel} lacks the 150 mm paper-A1 geometry")
    # m02 (R2C3): the retained aggregate probability distributions must be
    # named in the inventory, while no per-dyad value may ship.
    for rel in ("DATA_EXCLUSIONS.md", "results/README.md"):
        doc = docs.get(rel)
        if doc is None:
            continue
        for token in ("p1_summary", "p2_summary"):
            if token not in doc:
                errors.append(f"{rel} does not name the retained {token} aggregate")
    # The shipped preview must be a standalone 150 mm render: keep the digest
    # pin, but assert no manuscript membership (S01).
    for rel in ("results/worldline_preview/README.md", "results/README.md",
                "DATA_EXCLUSIONS.md", "notebooks/README.md", "CODE_MAP.md",
                "results/framework_pilot/holdout/CORRECTIONS.md"):
        doc = root / rel
        if not doc.is_file():
            continue
        body = doc.read_text(encoding="utf-8")
        for phrase in PAPER_EXHIBIT_ASSERTIONS:
            if phrase in body:
                errors.append(
                    f"{rel} still asserts paper-exhibit membership: {phrase!r}")
    base = root / "results/worldline_preview"
    pdf = base / "worldline_preview.pdf"
    png = base / "worldline_preview.png"
    if not pdf.is_file():
        errors.append("worldline_preview.pdf missing")
    else:
        blob = pdf.read_bytes()
        if b"465 380" in blob:
            errors.append("worldline PDF is the superseded 465 x 380 pt preview")
        if WG_PDF_MEDIABOX not in blob:
            errors.append(
                "worldline PDF is not 425.197 x 330 pt (150 mm) pinned preview geometry")
        if hashlib.sha256(blob).hexdigest() != WG_PDF_SHA256:
            errors.append("worldline PDF does not match the pinned preview digest")
        if b"DejaVu" in blob:
            errors.append("worldline PDF still uses the superseded DejaVu fonts")
    if not png.is_file():
        errors.append("worldline_preview.png missing")
    elif hashlib.sha256(png.read_bytes()).hexdigest() != WG_PNG_SHA256:
        errors.append("worldline PNG does not match the pinned 300 dpi preview digest")
    return errors


def check_git(root: Path) -> list[str] | None:
    """Return errors for a git tree, ``None`` when the check is not applicable.

    A ``None`` is reported as ``[SKIP] git index (not a git tree)``: an absent
    ``.git`` must not print ``[OK]`` for a check that did not run (m30).
    """
    if not (root / ".git").exists():
        return None
    try:
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True, capture_output=True, text=True,
        ).stdout.split("\0")
    except Exception as exc:
        return [f"git ls-files failed: {exc}"]
    errors = []
    for rel in filter(None, tracked):
        parts = Path(rel).parts
        if any(part in FORBIDDEN_DIRS for part in parts):
            errors.append(f"forbidden path is tracked: {rel}")
        if Path(rel).suffix in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden file type is tracked: {rel}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    root = args.root.resolve()

    checks = [
        ("required files & metadata", check_required(root) + check_metadata_contract(root)),
        ("forbidden paths", check_forbidden(root)),
        ("row-level data", check_row_level_data(root)),
        ("strict JSON", check_strict_json(root)),
        ("holdout checksums", check_tree_sums(root, HOLDOUT_REL, "holdout")),
        ("stability checksums", check_tree_sums(root, STABILITY_REL, "stability")),
        ("pre-registration checksum", check_prereg_checksum(root)),
        ("manifest exclusions", check_manifest(root)),
        ("manifest code hashes", check_manifest_code(root)),
        ("text hygiene", check_text(root)),
        ("notebooks", check_notebooks(root)),
        ("stability status", check_stability(root)),
        ("frozen headlines", check_headlines(root)),
        ("worldline notice", check_worldline_notice(root)),
        ("git index", check_git(root)),
    ]
    failures = 0
    for name, errors in checks:
        if errors is None:
            print(f"[SKIP] {name} (not a git tree)")
            continue
        if errors:
            failures += len(errors)
            print(f"[FAIL] {name}")
            for error in errors:
                print(f"       {error}")
        else:
            print(f"[OK] {name}")
    if failures:
        print(f"verify: FAILED ({failures} problem(s))")
        return 1
    print("verify: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
