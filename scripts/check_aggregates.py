#!/usr/bin/env python3
"""Verify the shipped aggregates without re-running any experiment.

The public artifact distributes aggregate tables and JSON summaries only. The
ICEWS event panel, experiment configuration, caches and row-level predictions
are not redistributed, so this script deliberately checks the *shipped* files
and never touches restricted inputs.

Usage::

    python3 scripts/check_aggregates.py stability
    python3 scripts/check_aggregates.py worldline
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STABILITY_VARIANT = Path("experiments/STABILITY_VARIANT.md")
RATE_KEYS = {"mean_p", "p_sum", "y_sum", "mean_y", "p_model"}
IDENTIFIER_COLUMNS = {
    "did", "sid", "tid", "aid", "actor", "actor_label", "source", "target",
    "source_actor", "target_actor", "source_aid", "target_aid", "dyad",
    "dyad_label", "month",
}
RESTRICTED_NOTE = (
    "Restricted inputs (ICEWS event data, experiment configuration, caches and "
    "row-level predictions) are not redistributed. This target verifies the "
    "shipped aggregates only; it does not re-run the audit. The public "
    "experiments/ tree is a five-seed preview, not the 50-refit audit."
)

# The public world-line preview ships the paper's Figure A1 render itself (R6
# X01): 150 mm = 425.197 pt wide, Arimo, no baked key.  Digests are of
# ``paper/figures/figA1_worldline.{pdf,png}``.
WG_PDF_MEDIABOX = b"/MediaBox [ 0 0 425.1968503937 330 ]"
WG_PDF_SHA256 = "77081960c20db85ff3748cc06edd3d38ceb9cf69fb93687ff612f5deec8bb3b1"
WG_PNG_SHA256 = "3b6e0fb29706a5c8af1bf219120d4c125467bc37ecc9ad48f25bd3df046e02d4"


def strict_load(path: Path) -> dict:
    def reject(token: str):
        raise ValueError(f"non-finite JSON constant {token!r} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def header_of(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {str(c).strip().lower() for c in next(csv.reader(handle), [])}


def check_excluded_columns(base: Path, label: str) -> list[str]:
    errors: list[str] = []
    for path in sorted((base / "tables").glob("*.csv")):
        rel = path.relative_to(base.parent.parent.parent).as_posix()
        if "dyad_contributions" in path.name:
            errors.append(f"{label}: row-level dyad contribution file present: {rel}")
            continue
        columns = header_of(path)
        prediction = columns & (RATE_KEYS | {"y", "p"})
        if {"y", "p"} <= columns:
            errors.append(f"{label}: row-level label+probability columns in {rel}")
        elif prediction & RATE_KEYS:
            errors.append(
                f"{label}: actor-level predicted/observed rate columns "
                f"{sorted(prediction & RATE_KEYS)} in {rel}")
        elif prediction and (columns & IDENTIFIER_COLUMNS):
            errors.append(f"{label}: prediction with unit identifiers in {rel}")
    for name in ("summary.json", "summary_decompose.json", "summary_hgb.json",
                 "summary_stability.json"):
        payload = strict_load(base / name)
        text_keys = set()

        def walk(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    text_keys.add(str(key))
                    walk(value)
            elif isinstance(obj, list):
                for value in obj:
                    walk(value)

        walk(payload)
        bad = text_keys & RATE_KEYS
        if bad:
            errors.append(f"{label}: excluded rate keys {sorted(bad)} in {name}")
    return errors


def check_stability_sums(root: Path, base: Path) -> list[str]:
    sums = base / "SHA256SUMS"
    if not sums.is_file():
        return ["missing stability/SHA256SUMS"]
    entries: dict[str, str] = {}
    errors: list[str] = []
    for lineno, line in enumerate(sums.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2:
            errors.append(f"stability/SHA256SUMS:{lineno}: malformed checksum line")
            continue
        digest, rel = parts
        entries[rel] = digest
    if "SHA256SUMS" in entries:
        errors.append("stability/SHA256SUMS must not checksum itself")
    actual = {
        path.relative_to(root).as_posix()
        for path in base.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    errors += [f"stability/SHA256SUMS does not cover {rel}"
               for rel in sorted(actual - set(entries))]
    errors += [f"stability/SHA256SUMS lists absent file {rel}"
               for rel in sorted(set(entries) - actual)]
    for rel, digest in sorted(entries.items()):
        path = root / rel
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append(f"stability checksum mismatch: {rel}")
    return errors


def check_stability(root: Path) -> list[str]:
    base = root / "results" / "framework_pilot" / "stability"
    required = [
        "report.md",
        "summary.json",
        "summary_decompose.json",
        "summary_hgb.json",
        "summary_stability.json",
        "SHA256SUMS",
    ]
    errors = [f"missing {p}" for p in required if not (base / p).is_file()]
    if errors:
        return errors
    if not (root / STABILITY_VARIANT).is_file():
        errors.append(f"missing {STABILITY_VARIANT.as_posix()}")

    report = (base / "report.md").read_text(encoding="utf-8").lower()
    for phrase in ("frozen", "holdout completed after", "not used", "provisional"):
        if phrase not in report:
            errors.append(f"stability report lacks release-note phrase: {phrase!r}")

    summary = strict_load(base / "summary.json")
    if summary.get("status") != "provisional_holdout_not_integrated":
        errors.append(f"unexpected stability status {summary.get('status')!r}")
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
            errors.append(f"stability scope.{key} != {value!r}")
    # The superseded "pre-holdout" label must never come back.
    legacy_label = "pre_holdout_holdout_excluded"
    for name in ("report.md", "summary.json", "summary_decompose.json",
                 "summary_hgb.json", "summary_stability.json"):
        if legacy_label in (base / name).read_text(encoding="utf-8"):
            errors.append(
                f"stability/{name} retains the legacy audit label "
                f"{legacy_label!r}"
            )
    holdout = summary.get("holdout", {})
    expected = {
        "available_at_audit": False,
        "completed_later": True,
        "used": False,
    }
    for key, value in expected.items():
        if holdout.get(key) is not value:
            errors.append(f"stability summary holdout.{key} != {value!r}")
    if holdout.get("confirmatory_run_start_utc") != "2026-10-23T05:26:55.031577Z":
        errors.append(
            "stability summary holdout.confirmatory_run_start_utc stale or missing"
        )
    stability = strict_load(base / "summary_stability.json")
    if stability.get("protocol", {}).get("n_seed_replicates") != 50:
        errors.append("summary_stability.json protocol.n_seed_replicates != 50")
    boot = stability.get("bootstrap", [])
    if not boot:
        errors.append("summary_stability.json has no bootstrap rows")
    elif {row.get("n_boot") for row in boot} != {50}:
        errors.append("summary_stability.json bootstrap n_boot != 50")

    tables = sorted((base / "tables").glob("*.csv"))
    if len(tables) < 30:
        errors.append(f"only {len(tables)} stability tables found")
    if list((base / "preds").glob("*")):
        errors.append("row-level predictions are present in the stability tree")
    errors += check_excluded_columns(base, "stability")
    errors += check_stability_sums(root, base)
    return errors


def check_worldline(root: Path) -> list[str]:
    base = root / "results" / "worldline_preview"
    required = [
        "README.md",
        "run_manifest.json",
        "worldline_trace.json",
        "worldline_preview.pdf",
        "worldline_preview.png",
    ]
    errors = [f"missing {p}" for p in required if not (base / p).is_file()]
    if errors:
        return errors

    # No row-level probability file may ship from this directory.
    for name in ("worldline_trace.csv", "node_trace.csv"):
        if (base / name).is_file():
            errors.append(f"row-level world-line export is present: {name}")
    strays = sorted(path.name for path in base.glob("*.csv"))
    if strays:
        errors.append(f"worldline directory ships CSV data: {strays}")

    meta = strict_load(base / "worldline_trace.json")
    text = json.dumps(meta).lower()
    if "illustrative" not in text and "not validated" not in text:
        errors.append("worldline metadata does not mark the preview illustrative")
    if meta.get("no_row_level_probabilities") is not True:
        errors.append("worldline metadata does not assert the no-row-level rule")
    anchors = meta.get("anchors") or []
    if len(anchors) != 4:
        errors.append(f"worldline anchor summaries changed: {len(anchors)} != 4")
    t_idx = (meta.get("period") or {}).get("t_idx") or []
    if not t_idx or max(int(t) for t in t_idx) > 274:
        errors.append(f"worldline anchors reach beyond the inner period: {t_idx}")
    traces = meta.get("machine_readable_traces") or {}
    if traces.get("shipped") is not False:
        errors.append("worldline metadata still advertises shipped row-level traces")

    manifest = strict_load(base / "run_manifest.json")
    outputs = manifest.get("outputs", {})
    if any(str(rel).endswith(".csv") for rel in outputs):
        errors.append("run_manifest still lists a per-row CSV output")
    for rel, digest in outputs.items():
        path = base / rel
        if not path.is_file():
            errors.append(f"run_manifest output missing: {rel}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append(f"run_manifest hash mismatch: {rel}")
    scope = manifest.get("preview_scope", {})
    if scope.get("row_level_files_shipped") != 0:
        errors.append("run_manifest preview_scope.row_level_files_shipped != 0")
    if scope.get("holdout_rows") != 0:
        errors.append("run_manifest preview_scope.holdout_rows != 0")

    pdf = (base / "worldline_preview.pdf").read_bytes()
    # The public preview must be the paper's A1 render, not the superseded
    # 465x380 pt (164 mm) DejaVu print version (R6 X01).
    if b"465 380" in pdf:
        errors.append("worldline PDF is the superseded 465 x 380 pt print version")
    if WG_PDF_MEDIABOX not in pdf:
        errors.append(
            "worldline PDF is not 425.197 x 330 pt (150 mm) paper-A1 geometry")
    if b"/Type3" in pdf:
        errors.append("worldline PDF contains Type 3 fonts")
    if b"/FontFile2" not in pdf:
        errors.append("worldline PDF lacks embedded TrueType fonts")
    if hashlib.sha256(pdf).hexdigest() != WG_PDF_SHA256:
        errors.append(
            "worldline PDF is not byte-identical to paper Figure A1 "
            f"(expected {WG_PDF_SHA256})")
    png = (base / "worldline_preview.png").read_bytes()
    if hashlib.sha256(png).hexdigest() != WG_PNG_SHA256:
        errors.append(
            "worldline PNG is not the 300 dpi raster of paper Figure A1 "
            f"(expected {WG_PNG_SHA256})")
    provenance = manifest.get("figure_provenance") or {}
    if provenance.get("baked_key") is not False:
        errors.append("run_manifest figure_provenance.baked_key must be false")
    if provenance.get("regenerable_from_this_tree") is not False:
        errors.append(
            "run_manifest figure_provenance.regenerable_from_this_tree must be false")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=("stability", "worldline"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()

    errors = (
        check_stability(root)
        if args.group == "stability"
        else check_worldline(root)
    )
    print(RESTRICTED_NOTE)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    print(f"[OK] {args.group} aggregates verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
