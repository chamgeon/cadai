"""Step 1: load every STEP file in data/raw and record each solid as one part.

A STEP file can hold several bodies, but similar-part search works on single
shapes, so every solid becomes its own row in `parts`. The `files` table tracks
loading per file (hash skip, load errors) plus the STEP header (lineage).

A file is skipped only if both its hash and EXTRACT_VER are unchanged, so
bumping EXTRACT_VER after adding a measurement re-processes everything.

Usage:
    python -m src.pipeline.extract            # all files
    python -m src.pipeline.extract --limit 20 # first 20 (quick test)
    python -m src.pipeline.extract --force    # re-process even if hash and version are unchanged
"""
import argparse
import hashlib
import math
import re
import sqlite3
import time
import traceback
from collections import Counter
from datetime import datetime, timezone

import cadquery as cq
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

from src.config import DB_PATH, FAILED, RAW

EXTRACT_VER = 2  # bump when measure() or the header fields change
MAX_BBOX = 1e5  # largest plausible part dimension (100 m in mm); beyond that the geometry is broken

# cadquery geomType() -> column. Freeform surfaces (car body panels) land in n_bspline.
FACE_TYPES = {
    "PLANE": "n_plane",
    "CYLINDER": "n_cylinder",
    "CONE": "n_cone",
    "SPHERE": "n_sphere",
    "TORUS": "n_torus",
    "BSPLINE": "n_bspline",
    "BEZIER": "n_bspline",
}
SI_PREFIX = {"$": "m", ".MILLI.": "mm", ".CENTI.": "cm", ".MICRO.": "um", ".KILO.": "km"}

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
    processed_at TEXT,
    extract_ver  INTEGER,            -- EXTRACT_VER that produced this row
    step_schema  TEXT,               -- e.g. AUTOMOTIVE_DESIGN (AP214), AP242_...
    step_system  TEXT,               -- originating CAD system from FILE_NAME
    step_time    TEXT,               -- export timestamp from FILE_NAME
    product_name TEXT,               -- first PRODUCT name; often the part number in real exports
    length_unit  TEXT                -- unit declared in the file; OpenCascade converts geometry to mm on import
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
    dup_of       TEXT,               -- part_id of an identical earlier solid in the same file
    -- rotation-independent size: bbox dims and principal moments, sorted large to small
    dim_1        REAL,
    dim_2        REAL,
    dim_3        REAL,
    inertia_1    REAL,               -- about the center of mass, unit density (mm^5)
    inertia_2    REAL,
    inertia_3    REAL,
    com_x        REAL,               -- center of mass
    com_y        REAL,
    com_z        REAL,
    -- scale-independent shape
    fill_ratio   REAL,               -- volume / bbox volume: 1 = block, low = thin or hollow
    sphericity   REAL,               -- 1 = ball, toward 0 = flat or spindly
    -- face-type histogram
    n_plane      INTEGER,
    n_cylinder   INTEGER,
    n_cone       INTEGER,
    n_sphere     INTEGER,
    n_torus      INTEGER,
    n_bspline    INTEGER,
    n_other_face INTEGER
);
"""


def migrate(conn, schema=SCHEMA):
    """Add columns from `schema` that an older database is missing."""
    for table, body in re.findall(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", schema, re.S):
        have = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for line in body.splitlines():
            m = re.match(r"\s+(\w+)\s+(TEXT|INTEGER|REAL)\b", line)
            if m and m.group(1) not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {m.group(1)} {m.group(2)}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_header(path):
    """Pull lineage fields out of the STEP text without a full parse."""
    text = path.read_text(errors="replace")
    header = text[: text.find("ENDSEC;")]
    # FILE_NAME(name, time_stamp, (author), (organization), preprocessor, originating_system, authorization)
    file_name = re.search(r"FILE_NAME\s*\((.*?)\)\s*;", header, re.S)
    fields = re.findall(r"'((?:[^']|'')*)'", file_name.group(1)) if file_name else []
    schema = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^'{]*)", header)
    product = re.search(r"\bPRODUCT\s*\(\s*'((?:[^']|'')*)'", text)

    conv = re.search(r"CONVERSION_BASED_UNIT\s*\(\s*'(\w+)'", text)
    si = re.search(r"SI_UNIT\s*\(\s*(\.\w+\.|\$)\s*,\s*\.METRE\.\s*\)", text)
    unit = conv.group(1).lower() if conv else SI_PREFIX.get(si.group(1), si.group(1)) if si else None

    def field(i):
        return fields[i].strip() or None if len(fields) > i else None

    return {
        "step_schema": schema.group(1).strip(" .") if schema else None,
        "step_time": field(1),
        "step_system": field(-2) if len(fields) >= 6 else None,
        "product_name": product.group(1).strip() or None if product else None,
        "length_unit": unit,
    }


def load_solids(path):
    """Import a STEP file and return every solid in it."""
    shapes = cq.importers.importStep(str(path)).vals()
    return [solid for shape in shapes for solid in shape.Solids()]


def measure(solid):
    bb = solid.BoundingBox()
    faces = solid.Faces()
    volume, area = solid.Volume(), solid.Area()

    gprops = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid.wrapped, gprops)
    com = gprops.CentreOfMass()
    pm = gprops.PrincipalProperties().Moments()
    inertia = sorted(pm, reverse=True)
    dims = sorted((bb.xlen, bb.ylen, bb.zlen), reverse=True)

    face_counts = Counter(FACE_TYPES.get(f.geomType(), "n_other_face") for f in faces)
    props = {
        "n_faces": len(faces),
        "n_edges": len(solid.Edges()),
        "volume": volume,
        "area": area,
        "bbox_x": bb.xlen,
        "bbox_y": bb.ylen,
        "bbox_z": bb.zlen,
        "is_valid": int(solid.isValid()),
        "dim_1": dims[0],
        "dim_2": dims[1],
        "dim_3": dims[2],
        "inertia_1": inertia[0],
        "inertia_2": inertia[1],
        "inertia_3": inertia[2],
        "com_x": com.X(),
        "com_y": com.Y(),
        "com_z": com.Z(),
        "fill_ratio": volume / math.prod(dims) if math.prod(dims) > 0 else None,
        # surface area of a ball with this volume / actual area
        "sphericity": math.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / area if volume > 0 and area > 0 else None,
    }
    props.update({col: face_counts.get(col, 0) for col in [*dict.fromkeys(FACE_TYPES.values()), "n_other_face"]})
    return props


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
    """Load and measure one STEP file -> (file_row, part_rows, solids)."""
    file_row = {
        "file_id": path.stem,
        "file_name": path.name,
        "file_sha256": sha256(path),
        "file_bytes": path.stat().st_size,
        "cadquery_ver": cq.__version__,
        "extract_ver": EXTRACT_VER,
    }
    try:
        file_row.update(read_header(path))
    except Exception:
        pass  # header fields are lineage only; a bad header must not block the geometry
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
        (FAILED / f"{path.stem}.extract.log").write_text(traceback.format_exc())
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
    return file_row, part_rows, solids


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
    migrate(conn)
    seen = {fid: (h, v) for fid, h, v in conn.execute("SELECT file_id, file_sha256, extract_ver FROM files")}

    files = sorted(p for p in RAW.iterdir() if p.suffix.lower() in (".step", ".stp"))
    if args.limit:
        files = files[: args.limit]

    counts = {"files_failed": 0, "skipped": 0, "parts_ok": 0, "parts_failed": 0, "dups": 0}
    for i, path in enumerate(files, 1):
        if not args.force and seen.get(path.stem) == (sha256(path), EXTRACT_VER):
            counts["skipped"] += 1
            continue
        file_row, part_rows, _ = process(path)
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
        f"{counts['files_failed']} files failed to load, {counts['skipped']} files skipped (unchanged, v{EXTRACT_VER})"
    )
    print(f"Results: {DB_PATH}")


if __name__ == "__main__":
    main()
