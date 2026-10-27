"""Step 4: load the chosen embeddings into Qdrant and search similar parts.

Chosen in step 4 (src.pipeline.evaluate): canonical pose + DINOv2. Canonical pose made
every model/pooling combination find rotated copies 100% of the time; DINOv2
won the manual check and spreads its scores (0.6-0.9 vs CLIP's 0.90-0.98), so
a "no good match" threshold is usable.

Each part is one Qdrant point with two named vectors:
    mean   one 768-d vector, the 6 views averaged: simplest, one comparison per part
    views  the 6 view vectors, compared with MaxSim (each query view takes its
           best-matching view): robust when two parts land in different but
           equivalent canonical poses
plus a payload of step 1 metadata, so results can be filtered or re-ranked by size.

Two ways in:
    build    full rebuild from the batch .npz of src.pipeline.embed (drops the collection)
    pipeline src.pipeline upserts one part at a time via upsert_part() and
             records it in the `embeddings` table, its skip/lineage checkpoint

Qdrant runs as a server (Docker, see README) so the pipeline worker and the UI
can use it at the same time; local mode locks its folder to one process.

Usage:
    python -m src.pipeline.index build
    python -m src.pipeline.index query my_bracket_s00 [--using mean] [--limit 5]
"""
import argparse
import sqlite3
import uuid

import numpy as np
from qdrant_client import QdrantClient, models

from src.config import DB_PATH, QDRANT_URL
from src.pipeline.embed import load

COLLECTION = "parts"
RENDER_SET, MODEL = "renders", "dinov2"
PAYLOAD_COLS = ("part_id", "file_id", "volume", "dim_1", "dim_2", "dim_3", "fill_ratio", "n_faces",
                "n_plane", "n_cylinder", "n_bspline")

SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    part_id       TEXT PRIMARY KEY REFERENCES parts(part_id),  -- a row means: this part is in Qdrant
    model         TEXT NOT NULL,
    file_sha256   TEXT,              -- source STEP hash
    mesh_ver      INTEGER,           -- upstream versions: a new mesh or render makes the vectors stale
    render_ver    INTEGER,
    job_id        INTEGER,           -- pipeline job that wrote the point
    embed_seconds REAL,
    indexed_at    TEXT
);
"""


def point_id(part_id):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, part_id))  # Qdrant ids must be ints or UUIDs


def client():
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(qc, dim):
    if qc.collection_exists(COLLECTION):
        return
    qc.create_collection(COLLECTION, vectors_config={
        "mean": models.VectorParams(size=dim, distance=models.Distance.COSINE),
        "views": models.VectorParams(size=dim, distance=models.Distance.COSINE,
                                     multivector_config=models.MultiVectorConfig(
                                         comparator=models.MultiVectorComparator.MAX_SIM)),
    })
    for key in ("part_id", "file_id"):  # filtered deletes and self-exclusion in similar()
        qc.create_payload_index(COLLECTION, key, models.PayloadSchemaType.KEYWORD)


def make_point(part_id, views, payload):
    """views (6, D), L2-normalized per view."""
    mean = views.mean(axis=0)
    mean /= np.linalg.norm(mean)
    return models.PointStruct(id=point_id(part_id), vector={"mean": mean.tolist(), "views": views.tolist()},
                              payload={**payload, "render_set": RENDER_SET, "model": MODEL})


def part_payload(conn, part_id):
    row = conn.execute(f"SELECT {', '.join(PAYLOAD_COLS)} FROM parts WHERE part_id = ?", (part_id,)).fetchone()
    return dict(zip(PAYLOAD_COLS, row))


def upsert_part(qc, part_id, views, payload):
    ensure_collection(qc, views.shape[1])
    qc.upsert(COLLECTION, points=[make_point(part_id, views, payload)])


def delete_stale(qc, file_id, keep):
    """Remove a file's points whose part_id is not in `keep` (a revision can drop or reject solids)."""
    if not qc.collection_exists(COLLECTION):
        return
    must_not = [models.FieldCondition(key="part_id", match=models.MatchAny(any=sorted(keep)))] if keep else None
    qc.delete(COLLECTION, points_selector=models.FilterSelector(filter=models.Filter(
        must=[models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id))], must_not=must_not)))


def build():
    ids, vecs = load(RENDER_SET, MODEL)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    meta = {r["part_id"]: dict(r) for r in conn.execute(f"SELECT {', '.join(PAYLOAD_COLS)} FROM parts")}

    qc = client()
    if qc.collection_exists(COLLECTION):
        qc.delete_collection(COLLECTION)
    ensure_collection(qc, vecs.shape[2])
    qc.upsert(COLLECTION, points=[make_point(pid, v, meta[pid]) for pid, v in zip(ids, vecs)])
    # The pipeline's records no longer describe the collection: it re-embeds and re-upserts on its next run.
    conn.executescript(SCHEMA)
    conn.execute("DELETE FROM embeddings")
    conn.commit()
    print(f"Indexed {qc.count(COLLECTION).count} parts into '{COLLECTION}' at {QDRANT_URL}")


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
