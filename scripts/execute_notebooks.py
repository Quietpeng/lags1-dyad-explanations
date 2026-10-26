#!/usr/bin/env python3
"""Execute the three read-only notebooks from the repository root.

The notebooks read only shipped aggregate files. Execution is headless
(``MPLBACKEND=Agg``) and CPU only. Outputs are written back into the notebook
JSON so that the public release displays the frozen aggregates without a local
run. The script fails if an execution error, an absolute path or a
credential-like string appears in the committed notebook.

Usage::

    python3 scripts/execute_notebooks.py            # execute and write
    python3 scripts/execute_notebooks.py --check    # execute, do not write
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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


def scan(text: str, where: str) -> list[str]:
    problems = [f"absolute path {token!r} in {where}"
                for token in ABS_PATHS if token in text]
    for pattern in SECRET_PATTERNS:
        match = pattern.search(text)
        if match:
            problems.append(f"credential-like string in {where}: {match.group(0)[:40]!r}")
    return problems


def execute(path: Path, check_only: bool, root: Path) -> list[str]:
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
        allow_errors=False,
    )
    client.execute()

    nb.metadata.setdefault("language_info", {})["version"] = platform.python_version()
    nb.metadata.pop("execution", None)
    for cell in nb.cells:
        cell.metadata.pop("execution", None)
        if cell.cell_type != "code":
            continue
        if not cell.get("outputs"):
            return [f"{path.name}: code cell {cell.get('execution_count')} has no output"]
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                return [f"{path.name}: execution error {output.get('ename')}"]

    text = json.dumps(nb, ensure_ascii=False)
    problems = scan(text, path.name)
    if problems:
        return problems

    if not check_only:
        nbformat.write(nb, path)
    n_outputs = sum(len(c.get("outputs", [])) for c in nb.cells if c.cell_type == "code")
    print(f"[OK] {path.name}: {len(nb.cells)} cells, {n_outputs} outputs"
          f"{' (not written)' if check_only else ''}")
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="execute but do not write the notebooks")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    paths = sorted((root / "notebooks").glob("*.ipynb"))
    if len(paths) != 3:
        print(f"expected 3 notebooks, found {len(paths)}", file=sys.stderr)
        return 1

    problems = []
    for path in paths:
        try:
            problems.extend(execute(path, args.check, root))
        except Exception as exc:  # pragma: no cover - kernel/runtime dependent
            problems.append(f"{path.name}: execution failed: {exc}")
    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}", file=sys.stderr)
        return 1
    print("notebooks: OK" if not args.check else "notebooks: OK (check only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
