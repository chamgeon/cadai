"""Step 8B: embed every part's 6 canonical renders with Voyage's hosted multimodal model.

The DINOv2 replacement experiment: same 512 px renders, same MaxSim scoring in
src.eval.search_quality, only the embedding model changes. Output has the format of
src.pipeline.embed: data/embeddings/renders__voyage.npz, part_ids (N,) and vectors
(N, 6, D) L2-normalized per view. Progress is saved after every batch, so a rate
limit or a crash costs one batch, not the run.

Usage:
    python -m src.eval.voyage_embed      # needs VOYAGE_API_KEY in .env
"""
import argparse
import sqlite3
import time

import numpy as np
import voyageai
from PIL import Image

from src.config import DB_PATH, RENDERS
from src.pipeline.embed import EMBEDDINGS
from src.pipeline.render import VIEWS

MODEL = "voyage-multimodal-3.5"
OUT = EMBEDDINGS / "renders__voyage.npz"
PARTIAL = EMBEDDINGS / "renders__voyage.partial.npz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="parts per request (x6 images)")
    args = parser.parse_args()
    conn = sqlite3.connect(DB_PATH)
    part_ids = [r[0] for r in conn.execute("SELECT part_id FROM embeddings ORDER BY part_id")]  # what is searchable
    done = {}
    if PARTIAL.exists():
        d = np.load(PARTIAL)
        done = dict(zip(d["part_ids"], d["vectors"]))
    todo = [p for p in part_ids if p not in done]
    print(f"{len(part_ids)} parts, {len(done)} already embedded, {len(todo)} to go")

    vo = voyageai.Client(max_retries=8)  # retries 429s with backoff
    start, tokens = time.perf_counter(), 0
    for b in range(0, len(todo), args.batch):
        batch = todo[b:b + args.batch]
        inputs = [[Image.open(RENDERS / p / f"view_{v}.png").convert("RGB")] for p in batch for v in range(len(VIEWS))]
        r = vo.multimodal_embed(inputs, model=MODEL, input_type="document")
        tokens += r.total_tokens
        vecs = np.array(r.embeddings, dtype=np.float32).reshape(len(batch), len(VIEWS), -1)
        vecs /= np.linalg.norm(vecs, axis=-1, keepdims=True)
        done.update(zip(batch, vecs))
        np.savez(PARTIAL, part_ids=np.array(list(done)), vectors=np.stack(list(done.values())))
        rate = (b + len(batch)) / (time.perf_counter() - start)
        print(f"  {len(done)}/{len(part_ids)}  {rate:.1f} parts/s  {tokens:,} tokens")

    np.savez(OUT, part_ids=np.array(part_ids), vectors=np.stack([done[p] for p in part_ids]))
    PARTIAL.unlink()
    print(f"Saved {OUT}  ({time.perf_counter() - start:.0f} s, {tokens:,} tokens this run)")


if __name__ == "__main__":
    main()
