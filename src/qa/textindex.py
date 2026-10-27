"""Step 6: join PLM, ERP and documents into part cards, index cards and notes for hybrid search.

The join is the unglamorous half of enterprise RAG: the four src.qa.plm tables
share only part number + revision, so each part's latest revision, its
supplier, its test history and its change notices are pulled together into one
`part_cards` row, plus step 1's measured geometry (the CAD side).

Chunks, one Qdrant point each in the `docs` collection:
    card  one per part: the joined record
    test  one per test report note
    eco   one per change-notice reason
Test and ECO chunks are embedded with a context header ("BR-19198 rev A | Bracket |
AL6061-T6 | fatigue | fail"), so "aluminum brackets that cracked" can match a note
that only says "crack at bolt hole". The note alone is what gets shown and cited.

Three vectors per chunk, so step 8 can compare them:
    dense      BGE-M3, multilingual: an English query finds a Korean note
    m3_sparse  BGE-M3's learned per-token weights
    bm25       term frequencies of Kiwi tokens (Korean particles split off, English
               stemmed); Qdrant multiplies in IDF at query time
Every payload carries `program`. search() puts the user's programs into the filter
of every prefetch, so another program's chunk never reaches ranking, let alone the LLM.

Usage:
    python -m src.qa.textindex build
    python -m src.qa.textindex search "크랙 난 브라켓" [--user normal] [--mode hybrid] [--keywords crack 균열]
"""
import argparse
import re
import sqlite3
import time
import uuid
import zlib
from collections import Counter

import numpy as np
import snowballstemmer
import torch
from huggingface_hub import hf_hub_download
from kiwipiepy import Kiwi
from qdrant_client import models

from src.config import DB_PATH
from src.pipeline.embed import device
from src.pipeline.index import client
from src.qa.plm import TESTS, USERS

COLLECTION = "docs"
TEXT_MODEL = "BAAI/bge-m3"
BATCH = 32
K1, B = 1.2, 0.75  # BM25 term-frequency saturation and length normalization
KEEP_TAGS = {"NNG", "NNP", "NR", "SL", "SN", "SH", "XR"}  # nouns, foreign words, numbers, roots
STOPWORDS = {"a", "an", "the", "at", "of", "in", "on", "to", "for", "and", "or", "by", "with", "after", "but",
             "no", "not", "is", "are", "was", "be", "near", "from", "than", "within", "below", "rev"}
FILTER_FIELDS = ("kind", "part_no", "part_class", "material", "status", "test_type", "result")

CARDS_SCHEMA = """
DROP TABLE IF EXISTS part_cards;
CREATE TABLE part_cards (
    part_no  TEXT PRIMARY KEY,
    part_id  TEXT,                   -- geometry of the latest revision
    rev      TEXT,
    program  TEXT,
    text     TEXT                    -- the joined record, as shown to the LLM
);
"""


# ---------------------------------------------------------------- tokens and vectors

_kiwi = None
_stem = snowballstemmer.stemmer("english")


def tokens(text):
    """Kiwi morphemes worth matching: particles and endings dropped, English lowercased and stemmed."""
    global _kiwi
    _kiwi = _kiwi or Kiwi()
    out = []
    for t in _kiwi.tokenize(text):
        if t.tag not in KEEP_TAGS:
            continue
        form = t.form.lower()
        if t.tag == "SL":
            if form in STOPWORDS:
                continue
            form = _stem.stemWord(form)
        out.append(form)
    return out


def term_id(token):
    return zlib.crc32(token.encode()) & 0x7FFFFFFF


def bm25_doc(toks, avg_len):
    """Saturated term frequencies; Qdrant's IDF modifier supplies the rest of BM25 at query time."""
    tf = Counter(term_id(t) for t in toks)
    norm = K1 * (1 - B + B * len(toks) / avg_len)
    return models.SparseVector(indices=list(tf), values=[c * (K1 + 1) / (c + norm) for c in tf.values()])


def bm25_query(text, keywords=()):
    ids = {term_id(t) for t in tokens(" ".join([text, *keywords]))}
    return models.SparseVector(indices=sorted(ids), values=[1.0] * len(ids))


class M3:
    """BGE-M3: CLS vector as the dense embedding, ReLU(linear(hidden)) per token as the sparse one."""

    def __init__(self, half=False):
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
        # half: queries only, ~1.9 GB less; build() stays fp32 so the stored vectors don't change.
        # Cast on the CPU: casting on MPS would leave the fp32-sized pool reserved. Eager attention:
        # the fused fp16 SDPA kernel fails to compile on M1 ("Failed to create function state object").
        dtype = torch.float16 if half else torch.float32
        self.model = AutoModel.from_pretrained(TEXT_MODEL, **({"attn_implementation": "eager"} if half else {}))
        self.model = self.model.eval().to(dtype).to(device())
        self.sparse = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.sparse.load_state_dict(torch.load(hf_hub_download(TEXT_MODEL, "sparse_linear.pt"), map_location="cpu"))
        self.sparse = self.sparse.to(dtype).to(device())
        self.special = set(self.tok.all_special_ids)

    @torch.no_grad()
    def __call__(self, texts):
        """-> (dense (n, 1024) L2-normalized, list of SparseVector), in the order of `texts`."""
        # Batch similar lengths together: a one-line note padded to a 7-line card costs as much as the card.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        texts = [texts[i] for i in order]
        dense, sparse = [], []
        for i in range(0, len(texts), BATCH):
            x = self.tok(texts[i : i + BATCH], padding=True, truncation=True, max_length=512,
                         return_tensors="pt").to(device())
            h = self.model(**x).last_hidden_state
            dense.append(torch.nn.functional.normalize(h[:, 0].float(), dim=-1).cpu().numpy())
            w = torch.relu(self.sparse(h)).squeeze(-1).float().cpu().numpy()
            for ids, ws, mask in zip(x["input_ids"].cpu().numpy(), w, x["attention_mask"].cpu().numpy()):
                best = {}
                for t, v, m in zip(ids, ws, mask):
                    if m and t not in self.special and v > 0:
                        best[int(t)] = max(best.get(int(t), 0.0), float(v))  # a repeated token keeps its max
                sparse.append(models.SparseVector(indices=list(best), values=list(best.values())))
        back = np.argsort(order)
        return np.concatenate(dense)[back], [sparse[i] for i in back]


# ---------------------------------------------------------------- join

def build_cards(conn):
    """One card per part from the four scattered tables; returns chunk dicts for cards, tests and ECOs."""
    conn.row_factory = sqlite3.Row
    q = lambda sql, *a: conn.execute(sql, a).fetchall()
    revs, reports, ecos = {}, {}, {}
    for r in q("SELECT * FROM plm_items ORDER BY part_no, rev"):
        revs.setdefault(r["part_no"], []).append(r)
    for r in q("SELECT * FROM doc_test_reports ORDER BY part_no, rev, test_type"):
        reports.setdefault(r["part_no"], []).append(r)
    for r in q("SELECT * FROM doc_change_notices ORDER BY part_no, from_rev"):
        ecos.setdefault(r["part_no"], []).append(r)
    erp = {r["part_no"]: r for r in q("SELECT * FROM erp_suppliers")}
    geo = {r["part_id"]: r for r in q("SELECT part_id, dim_1, dim_2, dim_3, volume, fill_ratio FROM parts")}

    conn.executescript(CARDS_SCHEMA)
    chunks = []
    for part_no, rs in revs.items():
        latest = rs[-1]
        g, s = geo[latest["part_id"]], erp.get(part_no)
        material_of = {r["rev"]: r["material"] for r in rs}
        lines = [
            f"{part_no} rev {latest['rev']}: {latest['name']}",
            f"{latest['part_class']}, {latest['material']}, {latest['mass_g']:g} g, {latest['status']}, "
            f"program {latest['program']}, owner {latest['owner_team']}.",
            f"Size {g['dim_1']:.1f} x {g['dim_2']:.1f} x {g['dim_3']:.1f} mm, volume {g['volume']:.0f} mm3, "
            f"fill ratio {g['fill_ratio']:.2f}.",
            f"Supplier: {s['supplier']}, {s['unit_cost_krw']:,} KRW, lead time {s['lead_time_wk']} weeks."
            if s else "Supplier: none in ERP.",
            "Revisions: " + "; ".join(f"{r['rev']} {r['status']} {r['rev_date']}" for r in rs) + ".",
        ]
        if part_no in reports:
            lines.append("Tests: " + "; ".join(f"{t['rev']} {TESTS[t['test_type']][0]} {t['result']} ({t['report_id']})"
                                               for t in reports[part_no]) + ".")
        for e in ecos.get(part_no, []):
            lines.append(f"Change {e['eco_id']} {e['from_rev']}→{e['to_rev']}: {e['reason']}")
        text = "\n".join(lines)
        conn.execute("INSERT INTO part_cards VALUES (?, ?, ?, ?, ?)",
                     (part_no, latest["part_id"], latest["rev"], latest["program"], text))

        base = {"part_no": part_no, "part_id": latest["part_id"], "program": latest["program"],
                "part_class": latest["part_class"], "status": latest["status"]}
        chunks.append({**base, "kind": "card", "doc_id": part_no, "rev": latest["rev"],
                       "material": latest["material"], "lang": "en", "header": "", "text": text})
        for t in reports.get(part_no, []):
            header = (f"{part_no} rev {t['rev']} | {latest['part_class']} | {material_of[t['rev']]} | "
                      f"{TESTS[t['test_type']][0]} | {t['result']}")
            chunks.append({**base, "kind": "test", "doc_id": t["report_id"], "rev": t["rev"],
                           "material": material_of[t["rev"]], "test_type": t["test_type"], "result": t["result"],
                           "lang": t["lang"], "date": t["test_date"], "header": header, "text": t["note"]})
        for e in ecos.get(part_no, []):
            header = f"{part_no} rev {e['from_rev']}→{e['to_rev']} | {latest['part_class']} | change notice"
            chunks.append({**base, "kind": "eco", "doc_id": e["eco_id"], "rev": e["to_rev"],
                           "material": material_of[e["to_rev"]], "lang": e["lang"], "date": e["eco_date"],
                           "header": header, "text": e["reason"]})
    conn.commit()
    return chunks


def embed_text(chunk):
    return f"{chunk['header']}\n{chunk['text']}" if chunk["header"] else chunk["text"]


# ---------------------------------------------------------------- index

def build():
    conn = sqlite3.connect(DB_PATH)
    chunks = build_cards(conn)
    conn.close()
    texts = [embed_text(c) for c in chunks]
    toks = [tokens(t) for t in texts]
    avg_len = sum(map(len, toks)) / len(toks)

    start = time.perf_counter()
    m3 = M3()
    loaded = time.perf_counter()
    dense, m3_sparse = m3(texts)
    took = time.perf_counter() - loaded

    qc = client()
    if qc.collection_exists(COLLECTION):
        qc.delete_collection(COLLECTION)
    qc.create_collection(
        COLLECTION,
        vectors_config={"dense": models.VectorParams(size=dense.shape[1], distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
                               "m3_sparse": models.SparseVectorParams()})
    for key in ("program", *FILTER_FIELDS):
        qc.create_payload_index(COLLECTION, key, models.PayloadSchemaType.KEYWORD)
    points = [models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["doc_id"])),
                                 vector={"dense": d.tolist(), "m3_sparse": s, "bm25": bm25_doc(t, avg_len)},
                                 payload=c)
              for c, d, s, t in zip(chunks, dense, m3_sparse, toks)]
    for i in range(0, len(points), 256):
        qc.upsert(COLLECTION, points=points[i : i + 256])
    kinds = Counter(c["kind"] for c in chunks)
    print(f"Indexed {qc.count(COLLECTION).count} chunks ({', '.join(f'{k} {n}' for k, n in kinds.items())}) "
          f"into '{COLLECTION}': model load {loaded - start:.0f}s, embed {took:.0f}s "
          f"({len(texts) / took:.0f} chunks/s), avg {avg_len:.1f} BM25 tokens")


_m3 = None


def load_model():
    """Load BGE-M3 now (~20 s) instead of inside the first search, where every queued tool call would wait on it."""
    global _m3
    _m3 = _m3 or M3(half=True)


def search(qc, query, user="normal", mode="hybrid", keywords=(), filters=None, limit=10):
    """Chunks for `query` that `user` may see. filters: {field: value or [values]} on FILTER_FIELDS."""
    must = [models.FieldCondition(key="program", match=models.MatchAny(any=list(USERS[user])))]
    for key, value in (filters or {}).items():
        if key not in FILTER_FIELDS:
            raise ValueError(f"unknown filter {key}")
        match = models.MatchAny(any=list(value)) if isinstance(value, (list, tuple)) else models.MatchValue(value=value)
        must.append(models.FieldCondition(key=key, match=match))
    flt = models.Filter(must=must)

    queries = {}
    if mode in ("hybrid", "bm25"):
        queries["bm25"] = bm25_query(query, keywords)
    if mode in ("hybrid", "dense", "m3_sparse"):
        load_model()
        dense, sparse = _m3([query])
        if mode != "m3_sparse":
            queries["dense"] = dense[0].tolist()
        if mode == "m3_sparse":
            queries["m3_sparse"] = sparse[0]

    if len(queries) == 1:
        (using, vec), = queries.items()
        return qc.query_points(COLLECTION, query=vec, using=using, query_filter=flt, limit=limit).points
    return qc.query_points(
        COLLECTION,
        prefetch=[models.Prefetch(query=v, using=u, filter=flt, limit=50) for u, v in queries.items()],
        query=models.FusionQuery(fusion=models.Fusion.RRF), query_filter=flt, limit=limit).points


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--user", choices=list(USERS), default="normal")
    s.add_argument("--mode", choices=("hybrid", "dense", "bm25", "m3_sparse"), default="hybrid")
    s.add_argument("--keywords", nargs="*", default=(), help="query expansion terms (what Claude will supply)")
    s.add_argument("--kind", choices=("card", "test", "eco"))
    s.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    if args.cmd == "build":
        build()
        return
    hits = search(client(), args.query, args.user, args.mode, args.keywords,
                  {"kind": args.kind} if args.kind else None, args.limit)
    print(f"[{args.mode}, user={args.user}: programs {', '.join(USERS[args.user])}]  {args.query}\n")
    for i, h in enumerate(hits, 1):
        p = h.payload
        first = re.sub(r"\s+", " ", p["text"])[:110]
        print(f"{i:2}. {h.score:.3f}  {p['kind']:4} {p['doc_id']:11} {p['program']}  {p['header'] or p['part_no']}\n"
              f"      {first}")


if __name__ == "__main__":
    main()
