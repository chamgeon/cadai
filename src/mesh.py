"""Step 2: tessellate every ok part at two tolerances and export meshes.

Tolerance scales with part size: the linear deflection (max distance between
the exact surface and a triangle) is a fraction of the bbox diagonal, so a
5 mm screw and a 2 m frame get the same relative quality.

    coarse  -> glb             (viewing)
    medium  -> glb + stl       (viewing / rendering for embeddings)

A third "fine" level was dropped in step 3: it took 71% of mesh storage and 80%
of meshing time, yet its renders differ from medium in ~0.01% of pixels
(src.render --compare). It stays in STUDY_LEVELS so that comparison reruns.

glTF is defined in metres, so the glb is scaled mm -> m. STL has no units and
stays in mm like the rest of the database.

Usage:
    python -m src.mesh --estimate    # tessellate + measure sizes in memory, write nothing
    python -m src.mesh               # write data/mesh/{part_id}/{level}.{glb,stl} + meshes table
    python -m src.mesh --limit 20 --force
"""
import argparse
import math
import sqlite3
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import trimesh
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.TopAbs import TopAbs_REVERSED
from OCP.TopLoc import TopLoc_Location

from src.config import DB_PATH, FAILED, MESH, RAW
from src.extract import load_solids, migrate, sha256

MESH_VER = 3  # bump when LEVELS or the export changes

# name: (linear deflection as a fraction of the bbox diagonal, angular deflection in radians, formats)
LEVELS = {
    "coarse": (0.005, 0.5, ("glb",)),
    "medium": (0.001, 0.2, ("glb", "stl")),
}
STUDY_LEVELS = {**LEVELS, "fine": (0.0002, 0.1, ("stl",))}  # tolerance comparison only, never written

SCHEMA = """
CREATE TABLE IF NOT EXISTS meshes (
    part_id      TEXT NOT NULL REFERENCES parts(part_id),
    level        TEXT NOT NULL,      -- coarse | medium | fine
    status       TEXT NOT NULL,      -- 'ok' | 'failed'
    error        TEXT,
    lin_tol      REAL,               -- absolute linear deflection used (mm)
    ang_tol      REAL,               -- angular deflection (rad)
    n_tris       INTEGER,
    n_verts      INTEGER,            -- after merging the per-face copies of shared vertices
    watertight   INTEGER,            -- closed, consistently oriented surface (1/0)
    open_edges   INTEGER,            -- edges used by only one triangle: 0 when watertight
    vol_err      REAL,               -- (mesh volume - exact volume) / exact volume
    tess_seconds REAL,               -- OpenCascade meshing only
    glb_bytes    INTEGER,
    stl_bytes    INTEGER,
    file_sha256  TEXT,               -- source STEP hash, for skipping
    mesh_ver     INTEGER,
    processed_at TEXT,
    PRIMARY KEY (part_id, level)
);
"""


def tessellate(solid, lin_tol, ang_tol):
    """Mesh the exact B-rep and return (vertices, faces) as numpy arrays, plus seconds spent meshing."""
    shape = solid.wrapped
    BRepTools.Clean_s(shape)  # drop any earlier triangulation, or OpenCascade reuses a finer one
    start = time.perf_counter()
    BRepMesh_IncrementalMesh(shape, lin_tol, False, ang_tol, True)
    seconds = time.perf_counter() - start

    verts, tris, offset = [], [], 0
    for face in solid.Faces():
        loc = TopLoc_Location()
        poly = BRep_Tool.Triangulation_s(face.wrapped, loc)
        if poly is None:
            continue
        trsf = loc.Transformation()
        nodes = (poly.Node(i).Transformed(trsf) for i in range(1, poly.NbNodes() + 1))
        verts.append(np.array([(p.X(), p.Y(), p.Z()) for p in nodes]))
        t = np.array([poly.Triangle(i).Get() for i in range(1, poly.NbTriangles() + 1)]) - 1 + offset
        if face.wrapped.Orientation() == TopAbs_REVERSED:
            t = t[:, [0, 2, 1]]
        tris.append(t)
        offset += poly.NbNodes()
    if not tris:
        raise ValueError("no triangles produced")
    return np.vstack(verts), np.vstack(tris), seconds


def export(verts, tris, formats):
    """Return {format: bytes} plus the merged mesh used for the quality checks."""
    # Vertices are duplicated per B-rep face: keeps sharp edges sharp in the glb normals.
    split = trimesh.Trimesh(verts, tris, process=False)
    merged = trimesh.Trimesh(verts, tris, process=True)  # welds shared vertices for watertight / volume
    # Sphere poles and cone tips give zero-area triangles once welded; they break watertightness.
    merged.update_faces(merged.nondegenerate_faces())
    merged.remove_unreferenced_vertices()
    out = {}
    if "glb" in formats:
        viewer = split.copy()
        viewer.apply_scale(0.001)
        out["glb"] = viewer.export(file_type="glb")
    if "stl" in formats:
        out["stl"] = merged.export(file_type="stl")
    return out, merged


def mesh_part(solid, diag, volume, write_dir=None):
    rows = []
    for level, (lin_frac, ang_tol, formats) in LEVELS.items():
        row = {"level": level, "lin_tol": lin_frac * diag, "ang_tol": ang_tol}
        try:
            verts, tris, row["tess_seconds"] = tessellate(solid, row["lin_tol"], ang_tol)
            data, merged = export(verts, tris, formats)
            row.update(
                status="ok",
                n_tris=len(merged.faces),
                n_verts=len(merged.vertices),
                watertight=int(merged.is_watertight),
                open_edges=int((np.unique(merged.edges_sorted, axis=0, return_counts=True)[1] == 1).sum()),
                vol_err=(merged.volume - volume) / volume if merged.is_watertight else None,
                glb_bytes=len(data["glb"]) if "glb" in data else None,
                stl_bytes=len(data["stl"]) if "stl" in data else None,
            )
            if write_dir:
                write_dir.mkdir(parents=True, exist_ok=True)
                for fmt, blob in data.items():
                    (write_dir / f"{level}.{fmt}").write_bytes(blob)
        except Exception as e:
            row.update(status="failed", error=f"{type(e).__name__}: {e}")
        rows.append(row)
    return rows


def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024


def summarize(rows):
    print(f"\n{'level':7} {'ok':>4} {'tris med':>9} {'tris max':>9} {'tess s tot':>10} {'tess s max':>10}"
          f" {'watertight':>10} {'|vol err| med':>13} {'glb total':>10} {'stl total':>10}")
    for level in LEVELS:
        ok = [r for r in rows if r["level"] == level and r["status"] == "ok"]
        if not ok:
            continue
        tris = [r["n_tris"] for r in ok]
        errs = [abs(r["vol_err"]) for r in ok if r["vol_err"] is not None]
        glb = sum(r["glb_bytes"] or 0 for r in ok)
        stl = sum(r["stl_bytes"] or 0 for r in ok)
        print(f"{level:7} {len(ok):4} {int(np.median(tris)):9} {max(tris):9}"
              f" {sum(r['tess_seconds'] for r in ok):10.2f} {max(r['tess_seconds'] for r in ok):10.2f}"
              f" {sum(r['watertight'] for r in ok):6}/{len(ok):<3}"
              f" {np.median(errs) * 100 if errs else float('nan'):12.3f}%"
              f" {fmt_bytes(glb) if glb else '-':>10} {fmt_bytes(stl) if stl else '-':>10}")
    total = sum((r.get("glb_bytes") or 0) + (r.get("stl_bytes") or 0) for r in rows)
    print(f"\nTotal output: {fmt_bytes(total)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate", action="store_true", help="measure sizes in memory, write nothing")
    parser.add_argument("--limit", type=int, help="process at most N parts")
    parser.add_argument("--force", action="store_true", help="ignore the hash/version skip")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    migrate(conn, SCHEMA)
    # Identical copies inside one file (dup_of) share their original's mesh.
    parts = conn.execute(
        """SELECT p.part_id, p.file_id, p.solid_index, p.volume, p.dim_1, p.dim_2, p.dim_3, f.file_name
           FROM parts p JOIN files f USING (file_id)
           WHERE p.status = 'ok' AND p.dup_of IS NULL ORDER BY p.part_id"""
    ).fetchall()
    done = {
        (pid, sha)
        for pid, sha in conn.execute(
            "SELECT part_id, file_sha256 FROM meshes WHERE mesh_ver = ? GROUP BY part_id"
            " HAVING count(*) = ?", (MESH_VER, len(LEVELS))
        )
    }
    if args.limit:
        parts = parts[: args.limit]

    by_file = {}
    for p in parts:
        by_file.setdefault(p[7], []).append(p)

    all_rows, skipped, n = [], 0, 0
    for file_name, file_parts in by_file.items():
        path = RAW / file_name
        file_sha = sha256(path)
        todo = [p for p in file_parts if args.force or args.estimate or (p[0], file_sha) not in done]
        skipped += len(file_parts) - len(todo)
        if not todo:
            continue
        solids = load_solids(path)
        for part_id, file_id, idx, volume, d1, d2, d3, _ in todo:
            n += 1
            diag = math.sqrt(d1 * d1 + d2 * d2 + d3 * d3)
            write_dir = None if args.estimate else MESH / part_id
            try:
                rows = mesh_part(solids[idx], diag, volume, write_dir)
            except Exception:
                (FAILED / f"{part_id}.mesh.log").write_text(traceback.format_exc())
                raise
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for r in rows:
                r.update(part_id=part_id, file_sha256=file_sha, mesh_ver=MESH_VER, processed_at=now)
            all_rows += rows
            print(f"[{n}] {part_id:55} " + "  ".join(
                f"{r['level']}={r.get('n_tris', 'FAIL')}" for r in rows))
            if not args.estimate:
                for r in rows:
                    cols = ", ".join(r)
                    conn.execute(f"INSERT OR REPLACE INTO meshes ({cols}) VALUES ({', '.join('?' * len(r))})",
                                 list(r.values()))
                conn.commit()

    conn.close()
    if skipped:
        print(f"\n{skipped} parts skipped (already meshed, v{MESH_VER})")
    if all_rows:
        summarize(all_rows)
        failed = [r for r in all_rows if r["status"] == "failed"]
        for r in failed:
            print(f"FAILED {r['part_id']} {r['level']}: {r['error']}")
    if args.estimate:
        print("(estimate only: nothing written)")


if __name__ == "__main__":
    main()
