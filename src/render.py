"""Step 3: render 6 fixed views of every part, headless, for image embeddings.

Similarity search compares pictures, so every picture must be taken the same
way: the part is first rotated into its canonical pose (src.canonical: own
axes from its faces, longest along X, shortest along Z), then centered on its
bbox and scaled so its bounding sphere has radius 1, the camera is orthographic (no perspective size cues), lights move
with the camera, and material, background and resolution never change.

Views (Z up): 0 front, 1 back, 2 left, 3 right, 4 top, 5 isometric.

Usage:
    python -m src.render                 # renders/{part_id}/view_0..5.png from medium.stl
    python -m src.render --limit 10 --force
    python -m src.render --compare 20    # same parts at coarse/medium/fine, pixel difference
"""
import argparse
import math
import random
import sqlite3
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import trimesh
import vtk
from PIL import Image
from vtk.util import numpy_support

from src.canonical import canonicalize
from src.config import DATA, DB_PATH, FAILED, MESH, RAW, RENDERS
from src.extract import load_solids, migrate
from src.mesh import STUDY_LEVELS, export, tessellate

RENDER_VER = 2  # bump when views, lighting, size or pose change
SIZE = 512
SOURCE_LEVEL = "medium"  # sub-pixel at embedding resolution; see the --compare run
COMPARE = DATA / "renders_compare"

# (camera direction from the part, view-up)
VIEWS = [
    ((0, -1, 0), (0, 0, 1)),  # front
    ((0, 1, 0), (0, 0, 1)),  # back
    ((-1, 0, 0), (0, 0, 1)),  # left
    ((1, 0, 0), (0, 0, 1)),  # right
    ((0, 0, 1), (0, 1, 0)),  # top
    ((1, -1, 1), (0, 0, 1)),  # isometric
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS renders (
    part_id        TEXT PRIMARY KEY REFERENCES parts(part_id),
    status         TEXT NOT NULL,    -- 'ok' | 'failed'
    error          TEXT,
    source_level   TEXT,             -- mesh level rendered
    pose_method    TEXT,             -- how the canonical axes were found: faces | face+minbox | inertia
    n_views        INTEGER,
    render_seconds REAL,
    png_bytes      INTEGER,          -- all views together
    file_sha256    TEXT,             -- source STEP hash, for skipping
    render_ver     INTEGER,
    processed_at   TEXT
);
"""


class Renderer:
    """One offscreen window reused for every part: creating windows is the slow part."""

    def __init__(self, size=SIZE):
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(1, 1, 1)
        self.ren.UseFXAAOn()
        self.ren.AutomaticLightCreationOff()
        vtk.vtkLightKit().AddLightsToRenderer(self.ren)  # camera lights: same lighting in every view

        self.win = vtk.vtkRenderWindow()
        self.win.SetOffScreenRendering(1)
        self.win.SetSize(size, size)
        self.win.AddRenderer(self.ren)

        self.actor = vtk.vtkActor()
        prop = self.actor.GetProperty()
        prop.SetColor(0.70, 0.72, 0.75)
        prop.SetAmbient(0.15)
        prop.SetDiffuse(0.8)
        prop.SetSpecular(0.2)
        prop.SetSpecularPower(20)
        self.ren.AddActor(self.actor)

        cam = self.ren.GetActiveCamera()
        cam.ParallelProjectionOn()
        cam.SetParallelScale(1.05)  # bounding sphere radius 1 plus a small margin

    def set_mesh(self, polydata):
        """Normalize to the unit bounding sphere and smooth normals within, not across, sharp edges."""
        x0, x1, y0, y1, z0, z1 = polydata.GetBounds()
        radius = math.dist((x0, y0, z0), (x1, y1, z1)) / 2
        t = vtk.vtkTransform()
        t.Scale(1 / radius, 1 / radius, 1 / radius)
        t.Translate(-(x0 + x1) / 2, -(y0 + y1) / 2, -(z0 + z1) / 2)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetTransform(t)
        tf.SetInputData(polydata)

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(tf.GetOutputPort())
        normals.SetFeatureAngle(30)  # holes and fillets look round, box edges stay crisp
        normals.SplittingOn()
        normals.ConsistencyOn()
        normals.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(normals.GetOutput())
        self.actor.SetMapper(mapper)

    def view(self, i):
        direction, up = VIEWS[i]
        d = np.array(direction, float)
        cam = self.ren.GetActiveCamera()
        cam.SetFocalPoint(0, 0, 0)
        cam.SetPosition(*(10 * d / np.linalg.norm(d)))
        cam.SetViewUp(*up)
        self.ren.ResetCameraClippingRange()
        self.win.Render()

        grab = vtk.vtkWindowToImageFilter()
        grab.SetInput(self.win)
        grab.ReadFrontBufferOff()
        grab.Update()
        img = grab.GetOutput()
        w, h, _ = img.GetDimensions()
        rgb = numpy_support.vtk_to_numpy(img.GetPointData().GetScalars()).reshape(h, w, -1)[:, :, :3]
        return np.flipud(rgb).copy()  # VTK images start at the bottom row


def to_polydata(verts, faces):
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(verts, dtype=np.float64), deep=True))
    cells = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64).ravel()
    polys = vtk.vtkCellArray()
    polys.ImportLegacyFormat(numpy_support.numpy_to_vtkIdTypeArray(cells, deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    pd.SetPolys(polys)
    return pd


def render_part(renderer, polydata, out_dir, prefix="view"):
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer.set_mesh(polydata)
    images, total = [], 0
    for i in range(len(VIEWS)):
        rgb = renderer.view(i)
        path = out_dir / f"{prefix}_{i}.png"
        Image.fromarray(rgb).save(path, optimize=True)
        total += path.stat().st_size
        images.append(rgb)
    return images, total


def run_pipeline(conn, args):
    conn.executescript(SCHEMA)
    migrate(conn, SCHEMA)
    parts = conn.execute(
        """SELECT m.part_id, m.file_sha256 FROM meshes m
           WHERE m.level = ? AND m.status = 'ok' ORDER BY m.part_id""", (SOURCE_LEVEL,)
    ).fetchall()
    done = set(conn.execute("SELECT part_id, file_sha256 FROM renders WHERE status = 'ok' AND render_ver = ?",
                            (RENDER_VER,)))
    if args.limit:
        parts = parts[: args.limit]

    renderer = Renderer()
    counts = {"ok": 0, "failed": 0, "skipped": 0}
    seconds = []
    for n, (part_id, file_sha) in enumerate(parts, 1):
        if not args.force and (part_id, file_sha) in done:
            counts["skipped"] += 1
            continue
        row = {"part_id": part_id, "source_level": SOURCE_LEVEL, "file_sha256": file_sha, "render_ver": RENDER_VER}
        start = time.perf_counter()
        try:
            mesh = trimesh.load(MESH / part_id / f"{SOURCE_LEVEL}.stl")
            if len(mesh.faces) == 0:
                raise ValueError("empty mesh")
            mesh, row["pose_method"] = canonicalize(mesh)
            _, row["png_bytes"] = render_part(renderer, to_polydata(mesh.vertices, mesh.faces), RENDERS / part_id)
            row.update(status="ok", n_views=len(VIEWS))
        except Exception as e:
            row.update(status="failed", error=f"{type(e).__name__}: {e}")
            (FAILED / f"{part_id}.render.log").write_text(traceback.format_exc())
        row["render_seconds"] = round(time.perf_counter() - start, 3)
        row["processed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        conn.execute(f"INSERT OR REPLACE INTO renders ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                     list(row.values()))
        conn.commit()
        counts[row["status"]] += 1
        seconds.append(row["render_seconds"])
        print(f"[{n}/{len(parts)}] {row['status']:6} {row['render_seconds']:5.2f}s  {part_id}  {row.get('error') or ''}")

    print(f"\nDone: {counts['ok']} rendered, {counts['failed']} failed, {counts['skipped']} skipped (v{RENDER_VER})")
    if seconds:
        print(f"Time per part: mean {np.mean(seconds):.2f}s, max {max(seconds):.2f}s, total {sum(seconds):.1f}s")


def run_compare(conn, n):
    """Render the same parts at every mesh level from one tessellation path, then diff the pixels.

    Answers: does a finer mesh change what the embedding model will see?
    """
    rows = conn.execute(
        """SELECT p.part_id, f.file_name, p.solid_index, p.volume, p.dim_1, p.dim_2, p.dim_3, p.n_bspline
           FROM parts p JOIN files f USING (file_id)
           WHERE p.status = 'ok' AND p.dup_of IS NULL ORDER BY p.part_id"""
    ).fetchall()
    # Your own part and the thread outlier always, then a fixed mix of freeform and prismatic parts.
    must = [r for r in rows if r[0] in ("my_bracket_s00", "00002307_4809f34e75cd44d393946d41_step_007_s00")]
    rng = random.Random(0)
    freeform = [r for r in rows if r[7] > 0 and r not in must]
    prismatic = [r for r in rows if r[7] == 0 and r not in must]
    rest = max(n - len(must), 0)
    chosen = must[:n] + rng.sample(freeform, rest // 2) + rng.sample(prismatic, rest - rest // 2)

    renderer = Renderer()
    diffs = {"coarse": [], "fine": []}  # each compared against medium
    print(f"{'part':48} {'coarse vs medium':>22} {'fine vs medium':>22}")
    print(f"{'':48} {'mean|d|':>9} {'px changed':>12} {'mean|d|':>9} {'px changed':>12}")
    for part_id, file_name, idx, volume, d1, d2, d3, _ in chosen:
        solid = load_solids(RAW / file_name)[idx]
        diag = math.sqrt(d1 * d1 + d2 * d2 + d3 * d3)
        imgs = {}
        for level, (lin_frac, ang_tol, _) in STUDY_LEVELS.items():
            verts, tris, _ = tessellate(solid, lin_frac * diag, ang_tol)
            _, merged = export(verts, tris, ())
            imgs[level], _ = render_part(renderer, to_polydata(merged.vertices, merged.faces),
                                         COMPARE / part_id, prefix=level)
        line = f"{part_id[:48]:48}"
        for level in ("coarse", "fine"):
            a = np.stack(imgs[level]).astype(int)
            b = np.stack(imgs["medium"]).astype(int)
            d = np.abs(a - b).max(axis=-1)  # per pixel, worst channel
            mean, changed = d.mean(), (d > 16).mean()  # >16/255: a visible change, not noise
            diffs[level].append((mean, changed))
            line += f" {mean:9.3f} {changed * 100:11.3f}%"
        print(line)

    print(f"\n{'median over ' + str(len(chosen)) + ' parts':48}", end="")
    for level in ("coarse", "fine"):
        means, changed = zip(*diffs[level])
        print(f" {np.median(means):9.3f} {np.median(changed) * 100:11.3f}%", end="")
    print(f"\n{'worst part':48}", end="")
    for level in ("coarse", "fine"):
        means, changed = zip(*diffs[level])
        print(f" {max(means):9.3f} {max(changed) * 100:11.3f}%", end="")
    print(f"\n\nImages: {COMPARE}/{{part_id}}/{{coarse,medium,fine}}_0..5.png (reused for the step 4 embedding check)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="render at most N parts")
    parser.add_argument("--force", action="store_true", help="ignore the hash/version skip")
    parser.add_argument("--compare", type=int, metavar="N", help="tolerance comparison on N parts instead")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    if args.compare:
        run_compare(conn, args.compare)
    else:
        run_pipeline(conn, args)
    conn.close()


if __name__ == "__main__":
    main()
