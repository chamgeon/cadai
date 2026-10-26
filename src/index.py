"""Step 4: load the chosen embeddings into Qdrant and search similar parts.

Chosen in step 4 (src.evaluate): canonical pose + DINOv2. Canonical pose made
every model/pooling combination find rotated copies 100% of the time; DINOv2
won the manual check and spreads its scores (0.6-0.9 vs CLIP's 0.90-0.98), so
a "no good match" threshold is usable.

Each part is one Qdrant point with two named vectors:
    mean   one 768-d vector, the 6 views averaged: simplest, one comparison per part
    views  the 6 view vectors, compared with MaxSim (each query view takes its
           best-matching view): robust when two parts land in different but
           equivalent canonical poses
plus a payload of step 1 metadata, so results can be filtered or re-ranked by size.

Usage:
    python -m src.index build
    python -m src.index query my_bracket_s00 [--using mean] [--limit 5]
"""
import argparse
import sqlite3
import uuid

import numpy as np
from qdrant_client import QdrantClient, models

from src.config import DB_PATH, QDRANT_PATH
from src.embed import load

COLLECTION = "parts"
RENDER_SET, MODEL = "renders", "dinov2"
PAYLOAD_COLS = ("part_id", "file_id", "volume", "dim_1", "dim_2", "dim_3", "fill_ratio", "n_faces",
                "n_plane", "n_cylinder", "n_bspline")


def point_id(part_id):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, part_id))  # Qdrant ids must be ints or UUIDs


def client():
    return QdrantClient(path=str(QDRANT_PATH))


def build():
    ids, vecs = load(RENDER_SET, MODEL)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    meta = {r["part_id"]: dict(r) for r in conn.execute(f"SELECT {', '.join(PAYLOAD_COLS)} FROM parts")}
    mean = vecs.mean(axis=1)
    mean /= np.linalg.norm(mean, axis=1, keepdims=True)

    qc = client()
    if qc.collection_exists(COLLECTION):
        qc.delete_collection(COLLECTION)
    dim = vecs.shape[2]
    qc.create_collection(COLLECTION, vectors_config={
        "mean": models.VectorParams(size=dim, distance=models.Distance.COSINE),
        "views": models.VectorParams(size=dim, distance=models.Distance.COSINE,
                                     multivector_config=models.MultiVectorConfig(
                                         comparator=models.MultiVectorComparator.MAX_SIM)),
    })
    qc.upsert(COLLECTION, points=[
        models.PointStruct(id=point_id(pid), vector={"mean": m.tolist(), "views": v.tolist()},
                           payload={**meta[pid], "render_set": RENDER_SET, "model": MODEL})
        for pid, m, v in zip(ids, mean, vecs)
    ])
    print(f"Indexed {qc.count(COLLECTION).count} parts into '{COLLECTION}' at {QDRANT_PATH}")


def similar(qc, part_id, using="views", limit=5):
    """Parts most similar to an indexed part, excluding itself."""
    me = qc.retrieve(COLLECTION, [point_id(part_id)], with_vectors=[using])[0]
    hits = qc.query_points(COLLECTION, query=me.vector[using], using=using, limit=limit + 1,
                           query_filter=models.Filter(must_not=[
                               models.FieldCondition(key="part_id", match=models.MatchValue(value=part_id))]))
    return hits.points[:limit]


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    q = sub.add_parser("query")
    q.add_argument("part_id")
    q.add_argument("--using", choices=("views", "mean"), default="views")
    q.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    if args.cmd == "build":
        build()
    else:
        qc = client()
        me = qc.retrieve(COLLECTION, [point_id(args.part_id)])[0].payload
        print(f"query: {args.part_id}  ({me['dim_1']:.1f} x {me['dim_2']:.1f} x {me['dim_3']:.1f} mm)\n")
        print(f"{'similarity':>10}  {'part_id':50} size (bbox, largest side first)")
        for p in similar(qc, args.part_id, args.using, args.limit):
            d = p.payload
            # Qdrant's MaxSim adds up the best match of each of the 6 query views; /6 gives 0..1 like `mean`.
            score = p.score / 6 if args.using == "views" else p.score
            print(f"{score:10.3f}  {d['part_id']:50} {d['dim_1']:.1f} x {d['dim_2']:.1f} x {d['dim_3']:.1f} mm")


if __name__ == "__main__":
    main()
