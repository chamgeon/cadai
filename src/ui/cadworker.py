"""Step 7: extract, mesh and render an uploaded STEP file, in a child process of the UI.

VTK's offscreen window on macOS is a Cocoa window, and Cocoa aborts the whole
process (NSException) when one is created off the main thread; Streamlit runs
every script on a worker thread. So the UI keeps this process alive: its main
thread owns the window, and cadquery and any OpenCascade crash stay out of the UI.

Nothing is written to the database or Qdrant: an upload is a question ("what do
we already have like this?"), not an ingest. Meshes and renders go to `out`.

Protocol: one JSON request per line on stdin, one JSON reply per line on the
original stdout (library prints are moved to stderr so they can't corrupt it):
    {"cmd": "warm"}                              -> {"ok": true, "seconds": s}
    {"cmd": "process", "step": path, "out": dir} -> {"ok": true, "parts": [...], "seconds": {stage: s}}
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import trimesh

from src.pipeline import extract, mesh, render
from src.pipeline.canonical import canonicalize
from src.pipeline.render import SOURCE_LEVEL, VIEWS


def warm(renderer):
    """First render creates the GL context and compiles shaders: do it before anyone waits on it."""
    start = time.perf_counter()
    box = trimesh.creation.box((1, 2, 3))
    renderer.set_mesh(render.to_polydata(box.vertices, box.faces))
    renderer.view(len(VIEWS) - 1)
    return {"ok": True, "seconds": round(time.perf_counter() - start, 2)}


def process(renderer, step, out):
    """Steps 1-3 for one file, the same stage functions and checks as the pipeline worker."""
    seconds = {"extract": 0.0, "mesh": 0.0, "render": 0.0}
    start = time.perf_counter()
    file_row, part_rows, solids = extract.process(Path(step))
    seconds["extract"] = round(time.perf_counter() - start, 2)
    if file_row["status"] == "failed":
        return {"ok": False, "error": file_row["error"], "seconds": seconds}

    parts = []
    for row in part_rows:
        part = {"part_id": row["part_id"], "status": row["status"], "error": row.get("error")}
        parts.append(part)
        if row["status"] != "ok":
            continue
        if row["dup_of"]:
            part.update(status="duplicate", error=f"same solid as {row['dup_of']}")
            continue
        dims = (row["dim_1"], row["dim_2"], row["dim_3"])
        part.update(dims=[round(d, 1) for d in dims], volume=row["volume"])
        part_dir = Path(out) / row["part_id"]

        t = time.perf_counter()
        levels = {r["level"]: r for r in mesh.mesh_part(solids[row["solid_index"]], math.dist((0, 0, 0), dims),
                                                         row["volume"], part_dir)}
        seconds["mesh"] += time.perf_counter() - t
        try:
            mesh.check_mesh(levels[SOURCE_LEVEL])
        except ValueError as e:
            part.update(status="rejected", error=str(e))
            continue

        t = time.perf_counter()
        m, _ = canonicalize(trimesh.load(part_dir / f"{SOURCE_LEVEL}.stl"))
        render.render_part(renderer, render.to_polydata(m.vertices, m.faces), part_dir)
        seconds["render"] += time.perf_counter() - t
        part.update(n_tris=levels[SOURCE_LEVEL]["n_tris"], glb=str(part_dir / "coarse.glb"),
                    renders=[str(part_dir / f"view_{i}.png") for i in range(len(VIEWS))])
    return {"ok": True, "parts": parts, "seconds": {k: round(v, 2) for k, v in seconds.items()}}


def main():
    reply_to = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)  # anything a library prints now goes to stderr
    renderer = render.Renderer()
    for line in sys.stdin:  # EOF when the UI exits: so does this process
        req = json.loads(line)
        try:
            if req["cmd"] == "warm":
                reply = warm(renderer)
            elif req["cmd"] == "process":
                reply = process(renderer, req["step"], req["out"])
            else:
                reply = {"ok": False, "error": f"unknown cmd {req['cmd']}"}
        except Exception as e:
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        reply_to.write(json.dumps(reply) + "\n")


if __name__ == "__main__":
    main()
