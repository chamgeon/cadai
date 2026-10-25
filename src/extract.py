"""Day 1: load every STEP file in data/raw and record each solid as one part.

A STEP file can hold several bodies, but similar-part search works on single
shapes, so every solid becomes its own row in `parts`. The `files` table tracks
loading per file (hash skip, load errors).

Usage:
    python -m src.extract            # all files
    python -m src.extract --limit 20 # first 20 (quick test)
    python -m src.extract --force    # re-process even if the file hash is unchanged
"""
import argparse
import hashlib
import sqlite3
import time
import traceback
from datetime import datetime, timezone

import cadquery as cq

from src.config import DB_PATH, FAILED, RAW

MAX_BBOX = 1e5  # largest plausible part dimension (100 m in mm); beyond that the geometry is broken

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    file_id      TEXT PRIMARY KEY,   -- file name without extension
    file_name    TEXT NOT NULL,
    file_sha256  TEXT NOT NULL,      -- lets later runs skip unchanged files
    file_bytes   INTEGER,
    status       TEXT NOT NULL,      -- 'ok' (loaded) | 'failed' (could not load)
    error        TEXT,
    n_solids     INTEGER,
    load_seconds REAL,
    cadquery_ver TEXT,
    processed_at TEXT
);
CREATE TABLE IF NOT EXISTS parts (
    part_id      TEXT PRIMARY KEY,   -- {file_id}_s{solid_index:02d}
    file_id      TEXT NOT NULL REFERENCES files(file_id),
    solid_index  INTEGER NOT NULL,   -- position of the solid inside the file
    status       TEXT NOT NULL,      -- 'ok' | 'failed' (failed validation)
    error        TEXT,
    n_faces      INTEGER,
    n_edges      INTEGER,
    volume       REAL,               -- model units (usually mm^3)
    area         REAL,               -- model units (usually mm^2)
    bbox_x       REAL,
    bbox_y       REAL,
    bbox_z       REAL,
    is_valid     INTEGER,            -- OpenCascade validity check (1/0)
    dup_of       TEXT                -- part_id of an identical earlier solid in the same file
);
"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_solids(path):
    """Import a STEP file and return every solid in it."""
    shapes = cq.importers.importStep(str(path)).vals()
    return [solid for shape in shapes for solid in shape.Solids()]


def measure(solid):
    bb = solid.BoundingBox()
    return {
        "n_faces": len(solid.Faces()),
        "n_edges": len(solid.Edges()),
        "volume": solid.Volume(),
        "area": solid.Area(),
        "bbox_x": bb.xlen,
        "bbox_y": bb.ylen,
        "bbox_z": bb.zlen,
        "is_valid": int(solid.isValid()),
    }


def check(props):
    """Reject solids that loaded but aren't usable."""
    if props["volume"] <= 0:
        raise ValueError(f"non-positive volume: {props['volume']}")
    size = max(props["bbox_x"], props["bbox_y"], props["bbox_z"])
    if not 0 < size < MAX_BBOX:
        raise ValueError(f"implausible bounding box: {size:.3g}")


def signature(props):
    """Rotation-independent fingerprint for spotting repeated copies of a solid."""
    return (
        props["n_faces"],
        props["n_edges"],
        float(f"{props['volume']:.4g}"),
        float(f"{props['area']:.4g}"),
    )


def process(path):
    file_row = {
        "file_id": path.stem,
        "file_name": path.name,
        "file_sha256": sha256(path),
        "file_bytes": path.stat().st_size,
        "cadquery_ver": cq.__version__,
    }
    part_rows = []
    start = time.perf_counter()
    try:
        solids = load_solids(path)
        if not solids:
            raise ValueError("no solids (surfaces or wireframe only)")
        file_row["status"] = "ok"
        file_row["n_solids"] = len(solids)
    except Exception as e:
        file_row["status"] = "failed"
        file_row["error"] = f"{type(e).__name__}: {e}"
        (FAILED / f"{path.stem}.log").write_text(traceback.format_exc())
        solids = []

    seen = {}
    for i, solid in enumerate(solids):
        row = {"part_id": f"{path.stem}_s{i:02d}", "file_id": path.stem, "solid_index": i}
        try:
            props = measure(solid)
            row.update(props)
            check(props)
            row["status"] = "ok"
            sig = signature(props)
            row["dup_of"] = seen.get(sig)
            seen.setdefault(sig, row["part_id"])
        except Exception as e:
            row["status"] = "failed"
            row["error"] = f"{type(e).__name__}: {e}"
        part_rows.append(row)

    file_row["load_seconds"] = round(time.perf_counter() - start, 3)
    file_row["processed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return file_row, part_rows


def insert(conn, table, row):
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    conn.execute(f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({marks})", list(row.values()))


def save(conn, file_row, part_rows):
    # Replace the file's old parts: its solid count may have changed.
    conn.execute("DELETE FROM parts WHERE file_id = ?", (file_row["file_id"],))
    insert(conn, "files", file_row)
    for row in part_rows:
        insert(conn, "parts", row)
    conn.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="process at most N files")
    parser.add_argument("--force", action="store_true", help="ignore the hash skip")
    args = parser.parse_args()

    DB_PATH.parent.mkdir(exist_ok=True)
    FAILED.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    seen = dict(conn.execute("SELECT file_id, file_sha256 FROM files"))

    files = sorted(p for p in RAW.iterdir() if p.suffix.lower() in (".step", ".stp"))
    if args.limit:
        files = files[: args.limit]

    counts = {"files_failed": 0, "skipped": 0, "parts_ok": 0, "parts_failed": 0, "dups": 0}
    for i, path in enumerate(files, 1):
        if not args.force and seen.get(path.stem) == sha256(path):
            counts["skipped"] += 1
            continue
        file_row, part_rows = process(path)
        save(conn, file_row, part_rows)

        ok = [r for r in part_rows if r["status"] == "ok"]
        dups = [r for r in ok if r["dup_of"]]
        counts["files_failed"] += file_row["status"] == "failed"
        counts["parts_ok"] += len(ok)
        counts["parts_failed"] += len(part_rows) - len(ok)
        counts["dups"] += len(dups)
        note = file_row.get("error") or (
            f"{len(ok)}/{len(part_rows)} solid(s) ok" + (f", {len(dups)} duplicate(s)" if dups else "")
        )
        print(f"[{i}/{len(files)}] {file_row['status']:6} {file_row['load_seconds']:6.2f}s  {path.name}  {note}")

    conn.close()
    print(
        f"\nDone: {counts['parts_ok']} parts ok ({counts['dups']} duplicates), "
        f"{counts['parts_failed']} parts failed validation, "
        f"{counts['files_failed']} files failed to load, {counts['skipped']} files skipped (unchanged)"
    )
    print(f"Results: {DB_PATH}")


if __name__ == "__main__":
    main()
