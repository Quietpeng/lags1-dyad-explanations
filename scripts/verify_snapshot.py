#!/usr/bin/env python3
"""Post-build verification of the stage-1 snapshots."""
import json, os
import duckdb

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
IC = os.path.join(ROOT, "data", "icews", "icews_events", "*", "*.parquet")
UN = os.path.join(ROOT, "data", "unga", "unga_votes_all.parquet")
AN = os.path.join(ROOT, "data", "anchors", "actor_anchors.parquet")
con = duckdb.connect(); con.execute("SET memory_limit='6GB';SET threads=4")
checks = {}

dups = con.execute(f"""SELECT count(*) FROM (
  SELECT event_date, source_name_raw, target_name_raw, cameo_code, story_id, sentence_number
  FROM read_parquet('{IC}') GROUP BY ALL HAVING count(*) > 1)""").fetchone()[0]
checks["icews_duplicate_keys"] = int(dups)

bad_root = con.execute(f"""SELECT count(*) FROM read_parquet('{IC}')
  WHERE cameo_root IS NOT NULL AND (TRY_CAST(cameo_root AS INT) < 0 OR TRY_CAST(cameo_root AS INT) > 20)""").fetchone()[0]
checks["icews_roots_outside_01_20"] = int(bad_root)

yr = con.execute(f"""SELECT "year" y, count(*) n, round(avg(CASE WHEN source_iso3 IS NOT NULL THEN 1 ELSE 0 END),4) iso_cov,
  count(DISTINCT cameo_root) roots FROM read_parquet('{IC}') GROUP BY 1 ORDER BY 1""").fetchall()
checks["icews_years"] = {int(a): {"rows": int(b), "source_iso3_coverage": float(c), "roots": int(d)} for a, b, c, d in yr}
checks["icews_rows"] = int(sum(x[1] for x in yr))

u = con.execute(f"""SELECT split, count(*) n, count(DISTINCT country) countries,
  sum(CASE WHEN vote='yes' THEN 1 ELSE 0 END) AS "yes", sum(CASE WHEN vote='no' THEN 1 ELSE 0 END) AS "no",
  sum(CASE WHEN vote='abstain' THEN 1 ELSE 0 END) AS "abstain"
  FROM read_parquet('{UN}') GROUP BY 1 ORDER BY 1""").fetchall()
checks["unga_by_split"] = {a: {"rows": int(b), "countries": int(c), "yes": int(d), "no": int(e), "abstain": int(f)}
                           for a, b, c, d, e, f in u}
checks["unga_rows"] = int(sum(x[1] for x in u))

a = con.execute(f"""SELECT count(*) AS n_rows, count(DISTINCT actor_id) AS n_actors,
  sum(CASE WHEN is_inferred=0 THEN 1 ELSE 0 END) external_rows,
  sum(CASE WHEN is_inferred=1 THEN 1 ELSE 0 END) inferred_rows FROM read_parquet('{AN}')""").fetchone()
checks["anchors"] = {"rows": int(a[0]), "actors": int(a[1]), "external_rows": int(a[2]), "inferred_rows": int(a[3])}
checks["anchors_by_type"] = {k: int(v) for k, v in con.execute(
    f"SELECT anchor_type, count(*) FROM read_parquet('{AN}') GROUP BY 1 ORDER BY 2 DESC").fetchall()}

checks["pass"] = (checks["icews_duplicate_keys"] == 0 and checks["icews_roots_outside_01_20"] == 0
                  and checks["icews_rows"] > 20_000_000 and checks["unga_rows"] > 700_000)
json.dump(checks, open(os.path.join(ROOT, "results", "stage1", "verify_snapshot.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in checks.items() if k not in ("icews_years", "unga_by_split")}, indent=1))
print("years", {k: v["rows"] for k, v in checks["icews_years"].items()})
print("wrote results/stage1/verify_snapshot.json")
