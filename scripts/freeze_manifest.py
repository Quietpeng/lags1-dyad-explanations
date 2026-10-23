#!/usr/bin/env python3
"""Assemble the frozen-snapshot manifest: DOIs, versions, hashes, row counts,
schemas, split dates, software versions and blockers."""
import hashlib, json, os, platform, subprocess, sys
from datetime import datetime, timezone
import duckdb

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
META = os.path.join(ROOT, "data", "meta")
RES = os.path.join(ROOT, "results", "stage1")

def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def files_with_hash(d, pattern=None):
    out = []
    for base, _, fs in os.walk(d):
        for f in sorted(fs):
            if pattern and not f.endswith(pattern):
                continue
            p = os.path.join(base, f)
            out.append({"path": os.path.relpath(p, ROOT), "bytes": os.path.getsize(p),
                        "sha256": sha256(p),
                        "mtime_utc": datetime.fromtimestamp(os.path.getmtime(p), timezone.utc).isoformat(timespec="seconds")})
    return out

def main():
    con = duckdb.connect(); con.execute("SET threads=4;SET memory_limit='4GB'")
    icews = os.path.join(ROOT, "data", "icews", "icews_events", "*", "*.parquet")
    ungap = os.path.join(ROOT, "data", "unga", "unga_votes_all.parquet")
    anchp = os.path.join(ROOT, "data", "anchors", "actor_anchors.parquet")

    n_icews = con.execute(f"SELECT count(*) FROM read_parquet('{icews}')").fetchone()[0]
    n_unga = con.execute(f"SELECT count(*) FROM read_parquet('{ungap}')").fetchone()[0]
    n_anch = con.execute(f"SELECT count(*) FROM read_parquet('{anchp}')").fetchone()[0]
    st = con.execute(f"""SELECT min(event_date),max(event_date),count(DISTINCT source_iso3),
                                count(DISTINCT cameo_root) FROM read_parquet('{icews}')""").fetchone()
    schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{icews}')").df()
    mschema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{ungap}')").df()

    icews_hashes = []
    hp = os.path.join(META, "icews_sha256.txt")
    if os.path.exists(hp):
        for line in open(hp):
            h, name = line.strip().split("  ", 1)
            p = os.path.join(ROOT, "data", "icews", "raw", name)
            icews_hashes.append({"file": name, "sha256": h,
                                 "bytes": os.path.getsize(p) if os.path.exists(p) else None})

    man = {
        "project": "neurosim-q2 CPU stage 1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": {"platform": platform.platform(), "python": sys.version.split()[0],
                 "cpu": os.cpu_count(), "duckdb": duckdb.__version__,
                 "requested_host": "Windows 16GB, D:\\neurosim-q2",
                 "actual_host": "Linux; D: unavailable, layout mirrored at repo/"},
        "sources": {
            "icews": {"doi": "10.7910/DVN/28075", "version": "37.0", "released": "2023-05-01",
                      "coverage": "1995-01-01..2023-04-09", "access": "Harvard Dataverse API",
                      "license": "research/teaching only; NO redistribution",
                      "n_files": len(icews_hashes), "bytes": sum(x["bytes"] or 0 for x in icews_hashes),
                      "files": icews_hashes},
            "unga_votes": {"doi": "10.7910/DVN/7UBUUI", "version": "2.0", "released": "2016-01-25",
                           "coverage": "1946..2013", "files": files_with_hash(os.path.join(ROOT, "data", "unga", "raw"))},
            "unga_idealpoints": {"doi": "10.7910/DVN/LEJUQZ", "version": "39.0", "released": "2026-07-30",
                                 "coverage": "1946..2024", "use": "FROZEN ONLY, excluded from features (same-year leakage)",
                                 "files": [f for f in files_with_hash(os.path.join(ROOT, "data", "unga", "raw"))
                                           if "Ideal" in os.path.basename(f["path"]) or "codebook" in f["path"]]},
            "wikidata": {"transport": "https://qlever.dev/api/wikidata (WDQS unreachable)",
                         "properties": ["P571", "P576", "P1365", "P1366", "P463/Q1065/P580"],
                         "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            "polecat": {"doi": "10.7910/DVN/AJGVIT", "version": "62.0", "status": "NOT downloaded (optional)"}},
        "derived": {
            "icews_events": {"path": "data/icews/icews_events/year=*/", "rows": int(n_icews),
                             "date_range": [str(st[0]), str(st[1])],
                             "source_actors": int(st[2]), "cameo_roots": int(st[3]),
                             "compression": "zstd", "partition": "year",
                             "dedup_key": ["event_date", "source_name_raw", "target_name_raw",
                                           "cameo_code", "story_id", "sentence_number"]},
            "unga_votes": {"path": "data/unga/unga_votes_all.parquet", "rows": int(n_unga),
                           "compression": "zstd"},
            "actor_anchors": {"path": "data/anchors/actor_anchors.parquet", "rows": int(n_anch)},
            "files": files_with_hash(os.path.join(ROOT, "data", "icews", "icews_events"), ".parquet") +
                     [{"path": "data/unga/unga_votes_all.parquet", "bytes": os.path.getsize(ungap), "sha256": sha256(ungap)}] +
                     [{"path": "data/anchors/actor_anchors.parquet", "bytes": os.path.getsize(anchp), "sha256": sha256(anchp)}]},
        "schema": {"icews": schema.to_dict(orient="records"), "unga": mschema.to_dict(orient="records")},
        "meta_files": [f for f in files_with_hash(META) if not f["path"].endswith("freeze_manifest.json")],
        "splits": {"icews": {"train": "1995-01-01..2014-12-31", "val": "2015-01-01..2017-12-31",
                             "test": "2018-01-01..2022-12-31"},
                   "unga": {"train": "1946..2003", "val": "2004..2009", "test": "2010..2013"}},
        "blockers": [
            {"id": "HOST_MISMATCH", "detail": "No Windows/D: host reachable; Linux mirror used."},
            {"id": "WDQS_UNREACHABLE", "detail": "query.wikidata.org timeout; QLever mirror used."},
            {"id": "PLACEBO_INPUTS_MISSING", "detail": "results/stage0 + OWID/SALS not present under this repo."},
            {"id": "ICEWS_VINTAGE", "detail": "v37 is a 2023 compilation; 'test' period is retrospective, not real-time."}]
    }
    os.makedirs(META, exist_ok=True)
    p = os.path.join(META, "freeze_manifest.json")
    json.dump(man, open(p, "w"), indent=2)
    tot = sum(f["bytes"] for f in man["derived"]["files"]) + man["sources"]["icews"]["bytes"]
    print("manifest", p, "derived_files", len(man["derived"]["files"]), "approx_bytes", tot)

if __name__ == "__main__":
    main()
