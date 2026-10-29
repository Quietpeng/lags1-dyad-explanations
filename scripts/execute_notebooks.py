#!/usr/bin/env python3
"""Execute the five read-only notebooks from the repository root.

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
EXPECTED_NOTEBOOKS = 5

FONT_CACHE_DIR = Path("build") / "mplconfig"
FONT_CACHE_NOTICE = "Matplotlib is building the font cache"

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


def warm_font_cache(root: Path) -> None:
    """Build the matplotlib font list once, before any kernel starts (S02).

    On a cold cache matplotlib writes ``Matplotlib is building the font cache``
    to the stderr *of the first process that renders text*.  Inside a kernel
    that stream becomes a notebook output, so the tracked notebook changes and
    the next ``make dist`` aborts.  Rendering one text artist here, in the
    driver process and in the directory the kernel will reuse, leaves the kernel
    a warm cache and no stream output.  ``MPLCONFIGDIR`` is kept if the caller
    set it, so a deliberately empty directory is warmed in place.
    """
    configured = os.environ.get("MPLCONFIGDIR")
    cache = Path(configured) if configured else root / FONT_CACHE_DIR
    try:
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache)
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(2.0, 2.0))
        FigureCanvasAgg(figure)
        figure.text(0.5, 0.5, "font cache warm-up")
        figure.canvas.draw()
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[note] matplotlib font-cache warm-up skipped: {exc}",
              file=sys.stderr)


def absorb_font_cache_notice(nb) -> int:
    """Drop a one-time matplotlib font-cache stderr stream if one appeared.

    Belt and braces: the warm-up above normally prevents it, but the executor
    also absorbs the known notice so a cold cache cannot dirty the tree.  A
    cell's last output is never removed; every code cell must keep an output.
    """
    absorbed = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        outputs = cell.get("outputs") or []
        kept = [output for output in outputs
                if not (output.get("output_type") == "stream"
                        and FONT_CACHE_NOTICE in str(output.get("text", "")))]
        if kept and len(kept) != len(outputs):
            cell["outputs"] = kept
            absorbed += len(outputs) - len(kept)
    return absorbed


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
    absorbed = absorb_font_cache_notice(nb)
    if absorbed:
        print(f"[note] {path.name}: absorbed {absorbed} matplotlib font-cache"
              " stream output", file=sys.stderr)
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
    warm_font_cache(root)

    paths = sorted((root / "notebooks").glob("*.ipynb"))
    if len(paths) != EXPECTED_NOTEBOOKS:
        print(f"expected {EXPECTED_NOTEBOOKS} notebooks, found {len(paths)}",
              file=sys.stderr)
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
