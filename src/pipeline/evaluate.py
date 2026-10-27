"""Step 4: score 12 search configurations on tests with known right answers.

    pose      world (as modeled) | canonical (src.pipeline.canonical)
    model     dinov2 | clip
    pooling   mean | max | maxsim (multi-vector: each query view takes its best-matching view)

Tests (the original part must come back; top-1 and top-5 hit rates):
    rotation-swap    part turned by a random 90-degree axis swap: the realistic case,
                     the same part modeled with its long side along Y instead of X
    rotation-random  part turned by a uniform random 3D rotation: the worst case
    duplicate        known identical copies (dup_of) find their original; 42 of 43
                     copies were only moved, so this is a sanity check, not a ranking
Plus the tolerance check from step 3: do coarse/fine renders embed like medium?

Usage:
    python -m src.pipeline.evaluate               # all tests, 60 parts per rotation test
    python -m src.pipeline.evaluate --n 187       # every part
"""
import argparse
import json
import math
import sqlite3
import time

import numpy as np
import trimesh
from PIL import Image

from src.pipeline.canonical import axis_swap_rotation, canonicalize, random_rotation
from src.config import DATA, DB_PATH, MESH, RAW
from src.pipeline.embed import MODELS, Embedder, load
from src.pipeline.extract import load_solids
from src.pipeline.mesh import LEVELS, export, tessellate
from src.pipeline.render import COMPARE, VIEWS, Renderer, to_polydata

POSES = {"world": "renders_world", "canonical": "renders"}
POOLINGS = ("mean", "max", "maxsim")
RESULTS = DATA / "eval"


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def scores(query, db, pooling):
    """query (6, D), db (N, 6, D), all L2-normalized -> similarity to each db part (N,)."""
    if pooling == "mean":
        return normalize(db.mean(1)) @ normalize(query.mean(0))
    if pooling == "max":
        return normalize(db.max(1)) @ normalize(query.max(0))
    sim = np.einsum("qd,nvd->nqv", query, db)  # every query view against every db view
    return sim.max(axis=2).mean(axis=1)  # MaxSim: best match per query view, averaged


def rank_of(target, query, ids, db, pooling):
    order = np.argsort(-scores(query, db, pooling))
    return int(np.where(ids[order] == target)[0][0]) + 1


def render_views(renderer, mesh):
    renderer.set_mesh(to_polydata(mesh.vertices, mesh.faces))
    return [Image.fromarray(renderer.view(i)) for i in range(len(VIEWS))]


def pose_mesh(mesh, pose):
    return canonicalize(mesh)[0] if pose == "canonical" else mesh


def build_queries(conn, n, seed):
    """Return [(test, target_part_id, mesh)] with meshes in their as-received orientation."""
    rng = np.random.default_rng(seed)
    ids = [r[0] for r in conn.execute("SELECT part_id FROM renders WHERE status = 'ok' ORDER BY part_id")]
    chosen = sorted(rng.choice(ids, size=min(n, len(ids)), replace=False))
    queries = []
    for pid in chosen:
        mesh = trimesh.load(MESH / pid / "medium.stl")
        for test, rot in (("rotation-swap", axis_swap_rotation), ("rotation-random", random_rotation)):
            m = mesh.copy()
            m.vertices = mesh.vertices @ rot(rng).T
            queries.append((test, pid, m))

    # Duplicates were never meshed (they share the original's mesh), so tessellate them here.
    dups = conn.execute(
        """SELECT p.dup_of, f.file_name, p.solid_index, p.dim_1, p.dim_2, p.dim_3
           FROM parts p JOIN files f USING (file_id) WHERE p.dup_of IS NOT NULL ORDER BY p.part_id"""
    ).fetchall()
    lin_frac, ang_tol, _ = LEVELS["medium"]
    solids = {}
    for target, file_name, idx, d1, d2, d3 in dups:
        if file_name not in solids:
            solids = {file_name: load_solids(RAW / file_name)}
        verts, tris, _ = tessellate(solids[file_name][idx], lin_frac * math.sqrt(d1 * d1 + d2 * d2 + d3 * d3), ang_tol)
        _, merged = export(verts, tris, ())
        queries.append(("duplicate", target, merged))
    return queries


def tolerance_check(embedders):
    """Cosine similarity of coarse/fine view embeddings to medium, on the step 3 comparison renders."""
    parts = sorted(p.name for p in COMPARE.iterdir() if (p / "medium_0.png").exists())
    out = {}
    for name, emb in embedders.items():
        vec = {lvl: emb([COMPARE / p / f"{lvl}_{i}.png" for p in parts for i in range(len(VIEWS))])
               for lvl in ("coarse", "medium", "fine")}
        out[name] = {lvl: float(np.median((vec[lvl] * vec["medium"]).sum(1))) for lvl in ("coarse", "fine")}
        out[name]["min_coarse"] = float((vec["coarse"] * vec["medium"]).sum(1).min())
    return out, len(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60, help="parts per rotation test")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    t0 = time.perf_counter()
    queries = build_queries(conn, args.n, args.seed)
    counts = {t: sum(q[0] == t for q in queries) for t in ("rotation-swap", "rotation-random", "duplicate")}
    print(f"queries: {counts}  ({time.perf_counter() - t0:.0f}s)")

    renderer = Renderer()
    embedders = {name: Embedder(name) for name in MODELS}
    ranks = {}  # (pose, model, pooling, test) -> [rank, ...]
    for pose, render_set in POSES.items():
        t0 = time.perf_counter()
        images = [img for _, _, mesh in queries for img in render_views(renderer, pose_mesh(mesh, pose))]
        for model, emb in embedders.items():
            ids, db = load(render_set, model)
            qv = emb(images).reshape(len(queries), len(VIEWS), -1)
            for pooling in POOLINGS:
                for (test, target, _), q in zip(queries, qv):
                    ranks.setdefault((pose, model, pooling, test), []).append(rank_of(target, q, ids, db, pooling))
        print(f"{pose:9} rendered + embedded {len(images)} query images ({time.perf_counter() - t0:.0f}s)")

    tests = ("rotation-swap", "rotation-random", "duplicate")
    print(f"\n{'pose':10} {'model':7} {'pooling':7}" + "".join(f" {t:>17}" for t in tests))
    print(f"{'':26}" + "".join(f" {'top1':>8} {'top5':>8}" for _ in tests))
    rows = []
    for pose in POSES:
        for model in MODELS:
            for pooling in POOLINGS:
                row = {"pose": pose, "model": model, "pooling": pooling}
                line = f"{pose:10} {model:7} {pooling:7}"
                for t in tests:
                    r = np.array(ranks[(pose, model, pooling, t)])
                    row[t] = {"top1": float((r == 1).mean()), "top5": float((r <= 5).mean()),
                              "median_rank": float(np.median(r))}
                    line += f" {row[t]['top1'] * 100:7.0f}% {row[t]['top5'] * 100:7.0f}%"
                rows.append(row)
                print(line)

    tol, n_tol = tolerance_check(embedders)
    print(f"\nTolerance check ({n_tol} parts x 6 views, world pose): median cosine similarity to medium")
    for name, v in tol.items():
        print(f"  {name:7} coarse {v['coarse']:.4f} (worst view {v['min_coarse']:.4f})   fine {v['fine']:.4f}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "step4_search.json"
    out.write_text(json.dumps({"n": args.n, "seed": args.seed, "queries": counts, "results": rows,
                               "tolerance": tol}, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
