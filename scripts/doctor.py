#!/usr/bin/env python3
"""Preflight checks for the public release (CPU only, no restricted data).

Checks
------
1. Python >= 3.12 (the pinned numpy/scipy have no cp311 wheels).
2. The analysis stack imports and reports its version.
3. Forbidden directories/files are absent from the working tree.

Run from anywhere::

    python3 scripts/doctor.py
    python3 scripts/doctor.py --root /path/to/extracted/release
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

MIN_PYTHON = (3, 12)
DEPS = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("duckdb", "duckdb"),
)
FORBIDDEN_DIRS = (
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
)
FORBIDDEN_SUFFIXES = (".pyc", ".pyo", ".npz", ".npy", ".parquet", ".log")
SKIP_DIRS = {".git", ".venv", "venv", "build", "dist", "node_modules",
             ".tox", ".nox", "site-packages"}


def check_python() -> bool:
    version = sys.version_info
    ok = version >= MIN_PYTHON
    status = "OK" if ok else "FAIL"
    print(f"[{status}] python {version.major}.{version.minor}.{version.micro} "
          f"(requires >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})")
    return ok


def check_deps() -> bool:
    ok = True
    for module, package in DEPS:
        try:
            mod = importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[FAIL] {package}: import failed ({exc})")
            ok = False
            continue
        version = getattr(mod, "__version__", "unknown")
        print(f"[OK] {package} {version}")
    return ok


def check_tree(root: Path) -> bool:
    ok = True
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.is_dir() and path.name in FORBIDDEN_DIRS:
            print(f"[FAIL] forbidden directory: {rel.as_posix()}")
            ok = False
        elif path.is_file() and path.suffix in FORBIDDEN_SUFFIXES:
            print(f"[FAIL] forbidden file type: {rel.as_posix()}")
            ok = False
    if ok:
        print("[OK] no forbidden input, cache or prediction paths")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="release root (default: the repository containing this script)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    print(f"release root: {root.name or root}")
    checks = [check_python(), check_deps(), check_tree(root)]
    if not all(checks):
        print("doctor: FAILED")
        return 1
    print("doctor: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
