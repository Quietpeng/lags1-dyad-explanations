"""Panel construction: actor-month node panel + dyad-month edge panel.

The SQL is a superset of the frozen G2 panel builder:

* dyad-month grid: all observed unordered dyads x all months 1995-01..2023-04
* labels: ``active_next`` (T2) and ``root_next`` (T3, later)
* actor-month node panel: 196 actors x 340 months, with in/out activity,
  partner counts and ``out_active_next`` (T1)
* split assignment is date-based and config-driven; the test split is tagged
  but never exported/fitted unless the caller explicitly allows it.

The panel build is deterministic and reads only ``data/icews/icews_events``.
It does not read any frozen result file.
"""

from __future__ import annotations

import glob
import os

import duckdb
import numpy as np
import pandas as pd

from . import SCHEMA
from .config import ensure_dirs
from .regions import assert_coverage, region_of
from .utils import jdump, log, sha256_file

PANEL_DYAD = "panel_dyad.parquet"
PANEL_ACTOR = "panel_actor.parquet"
ACTORS = "actors.parquet"
META = "panel_meta.json"


def connect(cfg: dict) -> duckdb.DuckDBPyConnection:
    ensure_dirs(cfg)
    con = duckdb.connect()
    threads = int(cfg["paths"].get("duckdb_threads", min(4, os.cpu_count() or 4)))
    mem = cfg["paths"].get("duckdb_memory", "6GB")
    con.execute(f"SET threads={threads}; SET memory_limit='{mem}';")
    con.execute("SET preserve_insertion_order=false;")
    tmp = os.path.join(cfg["paths"]["cache_dir"], "duckdb_tmp").replace("\\", "/")
    os.makedirs(tmp, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp}'")
    return con


def _p(path: str) -> str:
    return str(path).replace("\\", "/")


def source_files(cfg: dict) -> list[str]:
    files = sorted(glob.glob(cfg["paths"]["icews_glob"]))
    if not files:
        raise FileNotFoundError(f"no ICEWS parquet under {cfg['paths']['icews_glob']}")
    return files


def source_fingerprint(cfg: dict) -> dict:
    files = source_files(cfg)
    total = sum(os.path.getsize(f) for f in files)
    # Hash the concatenation of per-file SHA256 digests: stable, cheap, and
    # sufficient to detect a re-download/partition change.
    import hashlib

    h = hashlib.sha256()
    per_file = {}
    for f in files:
        d = sha256_file(f)
        per_file[os.path.relpath(f, cfg["paths"]["cache_dir"])] = d
        h.update(d.encode())
    return {
        "n_files": len(files),
        "total_bytes": total,
        "combined_sha256": h.hexdigest(),
        "first": os.path.basename(files[0]),
        "last": os.path.basename(files[-1]),
    }


def build_panels(cfg: dict, force: bool = False) -> dict:
    cache = cfg["paths"]["cache_dir"]
    ensure_dirs(cfg)
    dyad_path = os.path.join(cache, PANEL_DYAD)
    actor_path = os.path.join(cache, PANEL_ACTOR)
    actors_path = os.path.join(cache, ACTORS)
    meta_path = os.path.join(cache, META)

    con = connect(cfg)
    if (not force and os.path.exists(dyad_path) and os.path.exists(actor_path)
            and os.path.exists(actors_path) and os.path.exists(meta_path)):
        log("panel: cache present; loading")
        con.execute(f"CREATE OR REPLACE VIEW panel AS SELECT * FROM read_parquet('{_p(dyad_path)}')")
        con.execute(f"CREATE OR REPLACE VIEW actor_month AS SELECT * FROM read_parquet('{_p(actor_path)}')")
        con.execute(f"CREATE OR REPLACE VIEW actors AS SELECT * FROM read_parquet('{_p(actors_path)}')")
        meta = _load_meta(meta_path)
        meta["cached"] = True
        con.close()
        return meta

    ic = _p(cfg["paths"]["icews_glob"])
    split = cfg["split"]
    train_end = split["train_end"]
    val_end = split["val_end"]

    log("panel: building dyad-month aggregates from ICEWS")
    con.execute(f"""
      CREATE OR REPLACE TABLE dm AS
      SELECT s, t, m, sum(c) AS n_events, arg_max(root, c) AS root FROM (
        SELECT source_iso3 AS s, target_iso3 AS t,
               date_trunc('month', event_date) AS m,
               cameo_root AS root, count(*) AS c
        FROM read_parquet('{ic}')
        WHERE source_iso3 IS NOT NULL AND target_iso3 IS NOT NULL
          AND source_iso3 <> target_iso3 AND cameo_root IS NOT NULL
        GROUP BY 1, 2, 3, 4)
      GROUP BY 1, 2, 3
    """)
    con.execute("""
      CREATE OR REPLACE TABLE grid AS
      SELECT s, t, m FROM (SELECT DISTINCT s, t FROM dm),
        (SELECT unnest(generate_series((SELECT min(m) FROM dm),
                                       (SELECT max(m) FROM dm),
                                       INTERVAL 1 MONTH)) AS m)
    """)
    con.execute(f"""
      CREATE OR REPLACE TABLE panel0 AS
      SELECT g.s, g.t, g.m,
             coalesce(d.n_events, 0) AS n_events,
             d.root,
             (d.s IS NOT NULL) AS active,
             (nxt.s IS NOT NULL) AS active_next,
             nxt.root AS root_next,
             CASE WHEN g.m < DATE '{train_end}' THEN 'train'
                  WHEN g.m < DATE '{val_end}' THEN 'val'
                  ELSE 'test' END AS split,
             (year(g.m) - 1995) * 12 + (month(g.m) - 1) AS t_idx
      FROM grid g
      LEFT JOIN dm d ON g.s = d.s AND g.t = d.t AND g.m = d.m
      LEFT JOIN dm nxt ON nxt.s = g.s AND nxt.t = g.t
                       AND nxt.m = g.m + INTERVAL 1 MONTH
    """)
    con.execute("""
      CREATE OR REPLACE TABLE dyad_ids AS
      SELECT s, t, (row_number() OVER (ORDER BY s, t) - 1)::INTEGER AS did
      FROM (SELECT DISTINCT s, t FROM panel0)
    """)
    actors = sorted({r[0] for r in con.execute(
        "SELECT DISTINCT s FROM panel0 UNION SELECT DISTINCT t FROM panel0").fetchall()})
    assert_coverage(actors)
    actor_df = pd.DataFrame({
        "actor": actors,
        "aid": np.arange(len(actors), dtype=np.int16),
        "region": [region_of(a) for a in actors],
    })
    con.register("actor_df", actor_df)
    con.execute("CREATE OR REPLACE TABLE actors AS SELECT * FROM actor_df")
    con.execute("""
      CREATE OR REPLACE TABLE panel AS
      SELECT p.s, p.t, p.m, p.n_events, p.active, p.active_next, p.root, p.root_next,
             p.split, p.t_idx, d.did,
             a.aid AS sid, b.aid AS tid,
             a.region AS region_s, b.region AS region_t,
             (a.region = b.region) AS same_region
      FROM panel0 p
      JOIN dyad_ids d USING (s, t)
      JOIN actors a ON a.actor = p.s
      JOIN actors b ON b.actor = p.t
      ORDER BY d.did, p.t_idx
    """)

    log("panel: building actor-month node panel")
    con.execute("""
      CREATE OR REPLACE TABLE months AS
      SELECT unnest(generate_series((SELECT min(m) FROM dm),
                                    (SELECT max(m) FROM dm),
                                    INTERVAL 1 MONTH)) AS m
    """)
    con.execute("""
      CREATE OR REPLACE TABLE am_out AS
      SELECT s AS actor, m, sum(n_events) AS ev, count(*) AS partners
      FROM dm GROUP BY 1, 2
    """)
    con.execute("""
      CREATE OR REPLACE TABLE am_in AS
      SELECT t AS actor, m, sum(n_events) AS ev, count(*) AS partners
      FROM dm GROUP BY 1, 2
    """)
    con.execute(f"""
      CREATE OR REPLACE TABLE am0 AS
      SELECT a.actor, a.aid, a.region, mo.m,
             (year(mo.m) - 1995) * 12 + (month(mo.m) - 1) AS t_idx,
             coalesce(o.ev, 0) AS out_events,
             coalesce(i.ev, 0) AS in_events,
             (o.ev IS NOT NULL) AS out_active,
             (i.ev IS NOT NULL) AS in_active,
             coalesce(o.partners, 0) AS out_partners,
             coalesce(i.partners, 0) AS in_partners,
             CASE WHEN mo.m < DATE '{train_end}' THEN 'train'
                  WHEN mo.m < DATE '{val_end}' THEN 'val'
                  ELSE 'test' END AS split
      FROM actors a CROSS JOIN months mo
      LEFT JOIN am_out o ON o.actor = a.actor AND o.m = mo.m
      LEFT JOIN am_in i ON i.actor = a.actor AND i.m = mo.m
    """)
    con.execute("""
      CREATE OR REPLACE TABLE actor_month AS
      SELECT *,
             lead(out_active) OVER (PARTITION BY actor ORDER BY t_idx) AS out_active_next,
             lead(in_active)  OVER (PARTITION BY actor ORDER BY t_idx) AS in_active_next,
             lead(out_events) OVER (PARTITION BY actor ORDER BY t_idx) AS out_events_next
      FROM am0
      ORDER BY aid, t_idx
    """)

    # Persist only the public columns.
    con.execute(f"""
      COPY (SELECT s,t,m,n_events,active,active_next,root,root_next,split,t_idx,
                   did,sid,tid,region_s,region_t,same_region
            FROM panel ORDER BY did,t_idx)
      TO '{_p(dyad_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.execute(f"""
      COPY (SELECT actor,aid,region,m,t_idx,out_events,in_events,out_active,
                   in_active,out_partners,in_partners,split,
                   out_active_next,in_active_next,out_events_next
            FROM actor_month ORDER BY aid,t_idx)
      TO '{_p(actor_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.execute(f"COPY (SELECT * FROM actors ORDER BY aid) TO '{_p(actors_path)}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD)")

    counts = con.execute("""
      SELECT split, count(*) AS rows, sum(active_next::INT) AS positives
      FROM panel GROUP BY 1 ORDER BY 1
    """).fetchall()
    acounts = con.execute("""
      SELECT split, count(*) AS rows, sum(out_active_next::INT) AS positives
      FROM actor_month GROUP BY 1 ORDER BY 1
    """).fetchall()
    meta = {
        "schema": SCHEMA,
        "stage": "panel",
        "dyad": {"path": dyad_path, "sha256": sha256_file(dyad_path),
                 "rows": int(con.execute("SELECT count(*) FROM panel").fetchone()[0]),
                 "dyads": int(con.execute("SELECT count(DISTINCT did) FROM panel").fetchone()[0]),
                 "actors": len(actors),
                 "months": int(con.execute("SELECT count(DISTINCT t_idx) FROM panel").fetchone()[0]),
                 "split_counts": [{"split": r[0], "rows": int(r[1]), "positives": int(r[2])}
                                  for r in counts]},
        "actor_month": {"path": actor_path, "sha256": sha256_file(actor_path),
                        "rows": int(con.execute("SELECT count(*) FROM actor_month").fetchone()[0]),
                        "split_counts": [{"split": r[0], "rows": int(r[1]), "positives": int(r[2])}
                                         for r in acounts]},
        "actors_path": actors_path,
        "source": source_fingerprint(cfg),
        "split": split,
        "note": ("active_next is false at the right-censored final month "
                 "(2023-04); tail sensitivity is recorded in the manifest."),
    }
    jdump(meta_path, meta)
    con.close()
    log(f"panel: dyad={meta['dyad']['rows']:,} actor_month={meta['actor_month']['rows']:,}")
    return meta


def _load_meta(path: str) -> dict:
    from .utils import jload

    return jload(path)


def load_views(con: duckdb.DuckDBPyConnection, cfg: dict) -> None:
    cache = cfg["paths"]["cache_dir"]
    con.execute(f"CREATE OR REPLACE VIEW panel AS SELECT * FROM read_parquet('{_p(os.path.join(cache, PANEL_DYAD))}')")
    con.execute(f"CREATE OR REPLACE VIEW actor_month AS SELECT * FROM read_parquet('{_p(os.path.join(cache, PANEL_ACTOR))}')")
    con.execute(f"CREATE OR REPLACE VIEW actors AS SELECT * FROM read_parquet('{_p(os.path.join(cache, ACTORS))}')")


def tail_mask(t_idx: np.ndarray, cfg: dict) -> np.ndarray:
    from datetime import date

    y, m, _ = cfg["split"]["tail_start"].split("-")
    tail = (int(y) - 1995) * 12 + (int(m) - 1)
    return np.asarray(t_idx) >= tail
