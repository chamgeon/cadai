"""Step 8A: pipeline numbers from what the worker already recorded (no reruns).

    funnel    files seen -> loaded -> solids -> unique -> meshed -> passed the
              triangle gate -> rendered -> searchable, with the cause of every drop
    time      per stage (extract per file; mesh, render, embed per part) and per
              job wall clock, median / p95 / max
    storage   each generated artifact vs. the source STEP bytes
    flags     kept but suspect: invalid solids, non-watertight meshes, near the cap

Usage:
    python -m src.eval.pipeline_stats
"""
import json
import sqlite3
import subprocess
from datetime import datetime

import numpy as np

from src.config import DB_PATH, EVAL_RESULTS as RESULTS, FAILED, ROOT
from src.pipeline.mesh import MAX_TRIS
from src.pipeline.render import SOURCE_LEVEL



def dist(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    return {"n": len(v), "total": float(v.sum()), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def du(path):
    """Bytes on disk under path (Qdrant keeps its own files, there is no row to sum)."""
    out = subprocess.run(["du", "-sk", str(path)], capture_output=True, text=True, check=True).stdout
    return int(out.split()[0]) * 1024


def funnel(conn):
    one = lambda sql: conn.execute(sql).fetchone()[0]
    steps = [
        ("files seen", one("SELECT count(*) FROM files")),
        ("files loaded", one("SELECT count(*) FROM files WHERE status = 'ok'")),
        ("solids", one("SELECT count(*) FROM parts")),
        ("solids passed extract check", one("SELECT count(*) FROM parts WHERE status = 'ok'")),
        ("unique solids (= parts)", one("SELECT count(*) FROM parts WHERE status = 'ok' AND dup_of IS NULL")),
        ("meshed (coarse + medium)", one("SELECT count(*) FROM (SELECT part_id FROM meshes WHERE status = 'ok'"
                                         " GROUP BY part_id HAVING count(*) = 2)")),
        ("passed triangle gate", one(f"SELECT count(*) FROM meshes WHERE level = '{SOURCE_LEVEL}'"
                                     " AND status = 'ok' AND n_tris BETWEEN 4 AND " + str(MAX_TRIS))),
        ("rendered (6 views)", one(f"SELECT count(*) FROM renders r JOIN meshes m USING (part_id)"
                                   f" WHERE r.status = 'ok' AND m.level = '{SOURCE_LEVEL}'"
                                   f" AND m.n_tris BETWEEN 4 AND {MAX_TRIS}")),
        ("searchable (in Qdrant)", one("SELECT count(*) FROM embeddings")),
    ]
    return [{"stage": s, "count": n} for s, n in steps]


def failures(conn):
    """Every drop or flag, grouped by cause, with an example id to look at."""
    groups = [
        ("file", "dead job",
         "SELECT error, count(*), min(file_id) FROM jobs WHERE status = 'dead' GROUP BY error"),
        ("solid", "failed extract check",
         "SELECT error, count(*), min(part_id) FROM parts WHERE status = 'failed' GROUP BY error"),
        ("solid", "duplicate in same file (not searched, by design)",
         "SELECT 'identical to an earlier solid', count(*), min(part_id) FROM parts WHERE dup_of IS NOT NULL"),
        ("part", "mesh failed",
         "SELECT error, count(*), min(part_id) FROM meshes WHERE status = 'failed' GROUP BY error"),
        ("part", "rejected by triangle gate",
         f"SELECT 'medium mesh outside 4..{MAX_TRIS} triangles', count(*), min(part_id) FROM meshes"
         f" WHERE level = '{SOURCE_LEVEL}' AND status = 'ok' AND n_tris NOT BETWEEN 4 AND {MAX_TRIS}"),
        ("part", "render failed",
         "SELECT error, count(*), min(part_id) FROM renders WHERE status = 'failed' GROUP BY error"),
        ("part", "flag: invalid B-rep (kept, meshed fine)",
         "SELECT 'OpenCascade check failed', count(*), min(part_id) FROM parts"
         " WHERE status = 'ok' AND dup_of IS NULL AND is_valid = 0"),
        ("part", "flag: medium mesh not watertight (kept)",
         f"SELECT 'open or inconsistent surface', count(*), min(part_id) FROM meshes"
         f" WHERE level = '{SOURCE_LEVEL}' AND status = 'ok' AND watertight = 0"),
        ("part", "flag: medium mesh > 40% of triangle cap",
         f"SELECT 'over {int(MAX_TRIS * 0.4)} triangles', count(*), min(part_id) FROM meshes"
         f" WHERE level = '{SOURCE_LEVEL}' AND n_tris > {int(MAX_TRIS * 0.4)}"),
    ]
    rows = []
    # a job that succeeded on retry has its error cleared; the traceback stays in data/failed
    for (fid,) in conn.execute("SELECT file_id FROM jobs WHERE status = 'done' AND attempts > 1"):
        logs = sorted(FAILED.glob(f"{fid}.*.log"))
        cause = logs[0].read_text().strip().splitlines()[-1] if logs else "(log overwritten)"
        rows.append({"unit": "file", "kind": "retried, then done", "cause": cause, "count": 1, "example": fid})
    for unit, kind, sql in groups:
        for cause, n, example in conn.execute(sql):
            if n:
                rows.append({"unit": unit, "kind": kind, "cause": cause or "(no message)", "count": n,
                             "example": example})
    return rows


def timing(conn):
    col = lambda sql: [r[0] for r in conn.execute(sql)]
    secs = lambda a, b: (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()
    jobs = conn.execute("SELECT started_at, finished_at, n_indexed FROM jobs WHERE status = 'done'").fetchall()
    wall = [secs(a, b) for a, b, _ in jobs]
    t = {
        "extract (load + measure), per file": dist(col("SELECT load_seconds FROM files WHERE status = 'ok'")),
        "mesh (coarse + medium tessellation), per part": dist(col(
            "SELECT sum(tess_seconds) FROM meshes WHERE status = 'ok' GROUP BY part_id")),
        "render (6 views), per part": dist(col("SELECT render_seconds FROM renders WHERE status = 'ok'")),
        "embed (DINOv2, 6 views), per part": dist(col("SELECT embed_seconds FROM embeddings")),
    }
    t["other (export, DB, Qdrant, model load)"] = {"total": sum(wall) - sum(d["total"] for d in t.values())}
    # timestamps are whole seconds, so per-file wall clock is coarse; totals are fine
    t["job wall clock, per file (1 s resolution)"] = dist(wall)
    t["job wall clock, per searchable part"] = {"value": sum(wall) / max(1, sum(n for *_, n in jobs))}
    return t


def storage(conn):
    one = lambda sql: conn.execute(sql).fetchone()[0] or 0
    step = one("SELECT sum(file_bytes) FROM files WHERE status = 'ok'")
    items = {
        "coarse .glb (viewer)": one("SELECT sum(glb_bytes) FROM meshes WHERE level = 'coarse'"),
        "medium .glb (render source)": one("SELECT sum(glb_bytes) FROM meshes WHERE level = 'medium'"),
        "medium .stl (ML copy)": one("SELECT sum(stl_bytes) FROM meshes WHERE level = 'medium'"),
        "renders .png": one("SELECT sum(png_bytes) FROM renders"),
        "Qdrant parts collection": du(ROOT / "qdrant_data" / "collections" / "parts"),
        "SQLite registry": DB_PATH.stat().st_size,
    }
    total = sum(items.values())
    return {"step_bytes": step, "items": {k: {"bytes": v, "x_step": v / step} for k, v in items.items()},
            "generated_bytes": total, "multiplier": total / step,
            "not counted": "Qdrant docs collection (PLM text, step 6) "
                           f"{du(ROOT / 'qdrant_data' / 'collections' / 'docs') / 2**20:.0f} MB"}


def mb(n):
    return f"{n / 2**20:8.1f} MB"


def main():
    conn = sqlite3.connect(DB_PATH)
    result = {"funnel": funnel(conn), "timing": timing(conn), "storage": storage(conn), "failures": failures(conn)}

    print("Funnel")
    prev = result["funnel"][0]["count"]
    for row in result["funnel"]:
        delta = row["count"] - prev
        print(f"  {row['stage']:30} {row['count']:6}   {f'{delta:+d}' if delta else '':>6}")
        prev = row["count"]

    print("\nTime (s)                                        median     p95     max     total")
    for name, d in result["timing"].items():
        if "value" in d:
            print(f"  {name:44} {d['value']:7.2f}")
        elif "median" not in d:
            print(f"  {name:44} {'':31} {d['total']:9.0f}")
        else:
            print(f"  {name:44} {d['median']:7.2f} {d['p95']:7.2f} {d['max']:7.1f} {d['total']:9.0f}")

    s = result["storage"]
    print(f"\nStorage (source STEP {mb(s['step_bytes']).strip()})")
    for name, d in s["items"].items():
        print(f"  {name:30} {mb(d['bytes'])}  {d['x_step']:5.2f}x")
    print(f"  {'generated total':30} {mb(s['generated_bytes'])}  {s['multiplier']:5.2f}x   ({s['not counted']} not counted)")

    print("\nFailures and flags")
    for f in result["failures"]:
        print(f"  {f['unit']:5} {f['kind']:48} {f['count']:5}  {f['cause'][:60]}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "step8_pipeline.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
