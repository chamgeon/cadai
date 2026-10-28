"""Step 8B: similar-part search quality on 20 hand-labelled queries.

Step 4 only had synthetic tests (rotated copies must find their original). Here a
person grades real results, so the score means "would an engineer find this useful".

    pool    pick 20 query parts (spread over part classes, your own bracket included),
            run each search config, pool their top 5s into one candidate list per
            query -> step8_search_pool.json. Run again after adding a config: the
            queries, earlier runs and graded candidates stay; only new candidates join
    label   streamlit run src/eval/label_search.py   (blind: shuffled, config hidden)
    score   P@5 and nDCG@5 per config from step8_search_labels.json -> step8_search.json

Round 2 is a head-to-head of the embedding model: maxsim (DINOv2) vs voyage, both top 5s
pooled in a new order and graded from scratch in one sitting, so neither is judged against
grades from another day (src.eval.label_search -- --round 2; score --round 2). It also
reports how often round 2 repeats round 1's grade for the same pair: the grader's own noise.
Round 3 grades class_maxsim's top 5s alone and compares them with maxsim's round 2 grades;
the repeat rate on pairs both rounds saw shows how far grading drifted in between.

Configs (all as the app searches: admin sees every part, other solids of the query's file excluded):
    mean        DINOv2, one vector per part (6 views averaged), cosine
    maxsim      DINOv2, 6 vectors per part, MaxSim: what the Part page and the agent use
    maxsim_2x   maxsim, then only parts within 2x of the query's size (the UI's size filter)
    aligned     DINOv2, view i against view i only, averaged: the canonical pose makes front mean
                front, so MaxSim's any-view-to-any-view freedom only adds false matches
    maxsim_geo  maxsim's top 50, re-ranked half by image score, half by geometry distance
                (proportions, fill ratio, face-type mix, detail, size: GEO_GROUPS), each
                z-scored within the 50; weights fixed before looking at any grade
    voyage      maxsim with DINOv2 swapped for Voyage's hosted voyage-multimodal-3.5, on the same
                512 px renders (src.eval.voyage_embed): only the embedding model changes
    class_maxsim  maxsim among parts of the query's PLM part class only (here the class is
                src.qa.plm's guess from geometry; in a real PLM it is maintained metadata)

Grades: 0 not similar · 1 similar shape (same kind of part) · 2 could reuse it (same shape, similar size)

Usage:
    python -m src.eval.search_quality pool
    python -m src.eval.search_quality score
"""
import argparse
import json
import math
import random
import sqlite3

import numpy as np
from qdrant_client import models

from src.config import DB_PATH, EVAL_RESULTS as RESULTS
from src.pipeline import index
from src.pipeline.embed import EMBEDDINGS
from src.qa.tools import Tools

POOL = RESULTS / "step8_search_pool.json"
LABELS = {1: RESULTS / "step8_search_labels.json", **{r: RESULTS / f"step8_search_labels_r{r}.json" for r in (2, 3)}}
K = 5
MY_PART = "BR-66254"  # my_bracket.step, modelled in FreeCAD (step 2)
# queries per PLM part class (20, your bracket counted under Bracket); Nut and Gear have one part each
PER_CLASS = {"Bracket": 3, "Plate": 2, "Block": 2, "Beam": 2, "Body": 2, "Flange": 2, "Shaft": 2,
             "Cover": 1, "Washer": 1, "Housing": 1, "Pin": 1, "Bolt": 1}
CONFIGS = ("mean", "maxsim", "maxsim_2x", "aligned", "maxsim_geo", "voyage", "class_maxsim")
ROUND_CONFIGS = {1: CONFIGS[:5], 2: ("maxsim", "voyage"), 3: ("class_maxsim",)}
ROUND_PAIRS = {1: (("maxsim", "mean"), ("maxsim_2x", "maxsim"), ("aligned", "maxsim"), ("maxsim_geo", "maxsim")),
               2: (("voyage", "maxsim"),), 3: (("class_maxsim", "maxsim"),)}
BASELINES = {3: {"maxsim": 2}}  # config scored with an earlier round's grades: round -> {config: round}
RERANK_POOL = 50
# geometry from extract (step 1), in groups that count equally: no group outweighs another by having more columns
GEO_GROUPS = {
    "proportions": lambda g: [g["dim_2"] / g["dim_1"], g["dim_3"] / g["dim_1"]],  # dims sorted, dim_1 longest
    "fill": lambda g: [g["fill_ratio"]],  # volume / bbox volume: solid block ~1, thin bracket low
    "surfaces": lambda g: [g["n_plane"] / g["n_faces"], g["n_cylinder"] / g["n_faces"],
                           (g["n_cone"] + g["n_sphere"] + g["n_torus"]) / g["n_faces"],
                           (g["n_bspline"] + g["n_other_face"]) / g["n_faces"]],
    "detail": lambda g: [np.log1p(g["n_cylinder"]), np.log1p(g["n_faces"])],  # holes and bosses, complexity
    "size": lambda g: [np.log(g["dim_1"])],  # soft: renders erase scale, this brings it back as one vote
}
GEO_COLS = ("dim_1", "dim_2", "dim_3", "fill_ratio", "n_faces", "n_plane", "n_cylinder", "n_cone", "n_sphere",
            "n_torus", "n_bspline", "n_other_face")


def pick_queries(conn, seed):
    """One part per file, so no two queries are siblings from the same assembly."""
    rng = random.Random(seed)
    rows = conn.execute("SELECT c.part_no, c.part_id, p.file_id, i.part_class FROM part_cards c"
                        " JOIN parts p USING (part_id) JOIN plm_items i ON i.part_no = c.part_no AND i.rev = c.rev"
                        " ORDER BY c.part_no").fetchall()
    by_class = {}
    for r in rows:
        by_class.setdefault(r[3], []).append(r)
    mine = next(r for r in rows if r[0] == MY_PART)
    picked, files = [mine], {mine[2]}
    for cls, n in PER_CLASS.items():
        n -= cls == mine[3]
        for r in rng.sample(by_class[cls], len(by_class[cls])):
            if n == 0:
                break
            if r[2] not in files:
                picked.append(r)
                files.add(r[2])
                n -= 1
    return [{"part_no": p, "part_id": pid, "file_id": f, "class": c} for p, pid, f, c in picked]


def search_mean(tools, conn, q):
    me = tools.qc.retrieve(index.COLLECTION, [index.point_id(q["part_id"])], with_vectors=["mean"])[0]
    hits = tools.qc.query_points(index.COLLECTION, query=me.vector["mean"], using="mean", limit=K,
                                 query_filter=models.Filter(must_not=[models.FieldCondition(
                                     key="file_id", match=models.MatchValue(value=q["file_id"]))])).points
    ids = [h.payload["part_id"] for h in hits]
    return [conn.execute("SELECT part_no FROM part_cards WHERE part_id = ?", (i,)).fetchone()[0] for i in ids]


class Corpus:
    """Every indexed part's 6 view vectors and geometry, for the configs Qdrant can't run (aligned, maxsim_geo)."""

    def __init__(self, qc, conn):
        points, offset = [], None
        while True:
            batch, offset = qc.scroll(index.COLLECTION, limit=256, offset=offset, with_vectors=["views"],
                                      with_payload=["part_id", "file_id"])
            points += batch
            if offset is None:
                break
        self.part_ids = [p.payload["part_id"] for p in points]
        self.files = np.array([p.payload["file_id"] for p in points])
        views = np.array([p.vector["views"] for p in points], dtype=np.float32)
        self.views = views / np.linalg.norm(views, axis=-1, keepdims=True)  # (N, 6, D)
        self.row = {pid: i for i, pid in enumerate(self.part_ids)}
        voyage = EMBEDDINGS / "renders__voyage.npz"
        if voyage.exists():  # same rows as views, other model
            d = np.load(voyage)
            by_id = dict(zip(d["part_ids"], d["vectors"]))
            self.voyage = np.stack([by_id[pid] for pid in self.part_ids])
        self.part_no = dict(conn.execute("SELECT part_id, part_no FROM part_cards"))
        cls = dict(conn.execute("SELECT c.part_id, i.part_class FROM part_cards c"
                                " JOIN plm_items i ON i.part_no = c.part_no AND i.rev = c.rev"))  # current revision
        self.classes = np.array([cls[pid] for pid in self.part_ids])
        geo = {r[0]: dict(zip(GEO_COLS, r[1:])) for r in conn.execute(
            f"SELECT part_id, {', '.join(GEO_COLS)} FROM parts WHERE part_id IN"
            f" ({', '.join('?' * len(self.part_ids))})", self.part_ids)}
        self.groups = {}  # group -> (N, features), each feature z-scored over the corpus
        for name, fn in GEO_GROUPS.items():
            x = np.array([fn(geo[pid]) for pid in self.part_ids], dtype=float)
            self.groups[name] = (x - x.mean(0)) / (x.std(0) + 1e-9)

    def maxsim(self, i, views=None):
        views = self.views if views is None else views
        sim = np.einsum("qd,nvd->nqv", views[i], views)
        return sim.max(axis=2).mean(axis=1)

    def aligned(self, i):
        return np.einsum("vd,nvd->nv", self.views[i], self.views).mean(axis=1)

    def geo_distance(self, i, rows):
        return np.mean([np.abs(z[rows] - z[i]).mean(axis=1) for z in self.groups.values()], axis=0)

    def top(self, i, scores, k):
        scores = np.where(self.files == self.files[i], -np.inf, scores)  # the query's own file, itself included
        return np.argsort(-scores)[:k]

    def search(self, config, part_id):
        i = self.row[part_id]
        if config == "class_maxsim":
            rows = self.top(i, np.where(self.classes == self.classes[i], self.maxsim(i), -np.inf), K)
        elif config == "voyage":
            rows = self.top(i, self.maxsim(i, self.voyage), K)
        elif config == "aligned":
            rows = self.top(i, self.aligned(i), K)
        elif config == "maxsim_geo":
            img = self.maxsim(i)
            rows = self.top(i, img, RERANK_POOL)
            z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
            rows = rows[np.argsort(-(z(img[rows]) - z(self.geo_distance(i, rows))))][:K]
        else:  # maxsim: only to check this matches what Qdrant returned
            rows = self.top(i, self.maxsim(i), K)
        return [self.part_no[self.part_ids[r]] for r in rows]


def cmd_pool(args):
    conn = sqlite3.connect(DB_PATH)
    tools = Tools("admin")
    tools.qc = index.client()
    old = json.loads(POOL.read_text()) if POOL.exists() else None
    queries = old["queries"] if old else pick_queries(conn, args.seed)
    runs = old["runs"] if old else {}
    corpus = None
    for c in CONFIGS:
        if c in runs:
            continue
        runs[c] = {}
        for q in queries:
            if c == "mean":
                runs[c][q["part_no"]] = search_mean(tools, conn, q)
            elif c in ("maxsim", "maxsim_2x"):
                ratio = 2.0 if c == "maxsim_2x" else None
                runs[c][q["part_no"]] = [s["part_no"] for s in tools.similar_parts(q["part_no"], K, ratio)["similar"]]
            else:
                corpus = corpus or Corpus(tools.qc, conn)
                runs[c][q["part_no"]] = corpus.search(c, q["part_id"])
        print(f"ran {c}")
    if corpus:
        same = sum(corpus.search("maxsim", q["part_id"]) == runs["maxsim"][q["part_no"]] for q in queries)
        print(f"check: local MaxSim top {K} = Qdrant's for {same}/{len(queries)} queries")

    seed = old["seed"] if old else args.seed
    rng = random.Random(seed)
    candidates, added = {}, 0
    for q in queries:
        before = old["candidates"][q["part_no"]] if old else []
        new = sorted({p for c in ROUND_CONFIGS[1] for p in runs[c][q["part_no"]]} - set(before))
        rng.shuffle(new)  # no order hint about which config ranked it where
        candidates[q["part_no"]] = before + new  # graded ones keep their place
        added += len(new)
    out = {"seed": seed, "k": K, "queries": queries, "runs": runs, "candidates": candidates}
    n = sum(map(len, candidates.values()))
    print(f"{len(queries)} queries, round 1: {n} pairs, {added} new")
    # later rounds: graded from scratch, so a fresh order that position memory from earlier rounds can't use
    for rnd in sorted(ROUND_CONFIGS)[1:]:
        key = f"candidates_r{rnd}"
        cands = (old or {}).get(key)
        if not cands:  # made once: a round being graded never changes under the grader
            rng_r, cands = random.Random(seed + rnd), {}
            for q in queries:
                cands[q["part_no"]] = sorted({p for c in ROUND_CONFIGS[rnd] for p in runs[c][q["part_no"]]})
                rng_r.shuffle(cands[q["part_no"]])
        out[key] = cands
        print(f"round {rnd}: {sum(map(len, cands.values()))} pairs ({' vs '.join(ROUND_CONFIGS[rnd])})")
    POOL.write_text(json.dumps(out, indent=2))
    print(f"-> {POOL}")


def dcg(grades):
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def bootstrap_ci(per_query, n=5000, seed=0):
    """95% interval of the mean over queries: with 20 queries, differences under ~0.1 are noise."""
    v, rng = np.asarray(per_query, dtype=float), np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def agreement(r1, r2):
    """Same pair graded in two rounds: how often the grade repeats (exact, and similar-or-not)."""
    both = [(r1[q][c], g) for q, lq in r2.items() for c, g in lq.items() if c in r1.get(q, {})]
    if not both:
        return None
    a, b = np.array(both).T
    return {"n_pairs": len(both), "exact": float((a == b).mean()), "similar_or_not": float(((a >= 1) == (b >= 1)).mean()),
            "earlier_similar": int((a >= 1).sum()), "later_similar": int((b >= 1).sum())}


def cmd_score(args):
    rnd = args.round
    baselines = BASELINES.get(rnd, {})
    configs = (*ROUND_CONFIGS[rnd], *baselines)
    pool, labels = json.loads(POOL.read_text()), json.loads(LABELS[rnd].read_text())
    labels_of = {c: json.loads(LABELS[baselines[c]].read_text()) if c in baselines else labels for c in configs}
    known = {}  # every grade of the rounds compared, this round's last: the best possible top 5 for nDCG
    for c in configs:
        for q, lq in labels_of[c].items():
            known.setdefault(q, {}).update(lq)
    for q, lq in labels.items():
        known[q].update(lq)
    key = "candidates" if rnd == 1 else f"candidates_r{rnd}"
    missing = [(q, c) for q, cs in pool[key].items() for c in cs if labels.get(q, {}).get(c) not in (0, 1, 2)]
    if missing:
        raise SystemExit(f"{len(missing)} pairs not labelled yet, e.g. {missing[0]}")
    queries = [q["part_no"] for q in pool["queries"]]
    grades_of = {c: {q: [labels_of[c][q][p] for p in pool["runs"][c][q]] for q in queries} for c in configs}
    result = {"round": rnd, "n_queries": len(queries), "n_pairs": sum(map(len, labels.values())),
              "grade_counts": {g: sum(v == g for lq in labels.values() for v in lq.values()) for g in (0, 1, 2)},
              "configs": {}, "paired_p5": {}}
    print(f"{'config':10} {'P@5 (>=1)':>10} {'95% CI':>13} {'P@5 (=2)':>9} {'nDCG@5':>7} {'hit@1':>6}"
          f" {'returned':>9} {'precision':>10}")
    for c in configs:
        p1, p2, nd, top1 = [], [], [], []
        for q in queries:
            grades = grades_of[c][q] + [0] * (K - len(grades_of[c][q]))  # a short list counts as misses
            ideal = sorted(known[q].values(), reverse=True)[:K]  # best order of everything graded for q
            p1.append(np.mean([g >= 1 for g in grades]))
            p2.append(np.mean([g == 2 for g in grades]))
            nd.append(dcg(grades) / dcg(ideal) if dcg(ideal) else 0.0)
            top1.append(grades[0] >= 1)
        returned = sum(len(g) for g in grades_of[c].values())
        r = result["configs"][c] = {
            "p5_similar": float(np.mean(p1)), "p5_similar_ci": bootstrap_ci(p1), "p5_reusable": float(np.mean(p2)),
            "ndcg5": float(np.mean(nd)), "hit1_similar": float(np.mean(top1)), "returned": returned,
            # the size filter can return fewer than K: precision over what it did return
            "precision_returned": sum(g >= 1 for gs in grades_of[c].values() for g in gs) / returned,
            "per_query_p5": dict(zip(queries, map(float, p1))), "per_query_ndcg5": dict(zip(queries, map(float, nd)))}
        ci = r["p5_similar_ci"]
        print(f"{c:10} {r['p5_similar']:10.2f} {f'[{ci[0]:.2f}, {ci[1]:.2f}]':>13} {r['p5_reusable']:9.2f}"
              f" {r['ndcg5']:7.2f} {r['hit1_similar']:6.2f} {returned:9} {r['precision_returned']:10.2f}")

    print("\nPer query, P@5 (>=1):        wins  ties  losses   mean diff  95% CI")
    for a, b in ROUND_PAIRS[rnd]:
        d = np.array([result["configs"][a]["per_query_p5"][q] - result["configs"][b]["per_query_p5"][q]
                      for q in queries])
        ci = bootstrap_ci(d)
        result["paired_p5"][f"{a} vs {b}"] = {"wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
                                              "losses": int((d < 0).sum()), "mean_diff": float(d.mean()), "ci": ci}
        print(f"  {a + ' vs ' + b:26} {(d > 0).sum():5} {(d == 0).sum():5} {(d < 0).sum():7}"
              f"   {d.mean():+9.2f}  [{ci[0]:+.2f}, {ci[1]:+.2f}]")
    if baselines:
        print(f"  (baseline grades from round {', '.join(f'{c} {r}' for c, r in baselines.items())})")
    for earlier in range(1, rnd):
        agree = result[f"agreement_with_round{earlier}"] = agreement(json.loads(LABELS[earlier].read_text()), labels)
        if agree:
            print(f"Same pair in round {earlier} and {rnd} ({agree['n_pairs']}): same grade {agree['exact']:.0%},"
                  f" same similar-or-not {agree['similar_or_not']:.0%}"
                  f" (similar: {agree['earlier_similar']} then {agree['later_similar']})")
    out = RESULTS / ("step8_search.json" if rnd == 1 else f"step8_search_r{rnd}.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"\nSaved {out}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    pool = sub.add_parser("pool")
    pool.add_argument("--seed", type=int, default=8)
    score = sub.add_parser("score")
    score.add_argument("--round", type=int, choices=sorted(ROUND_CONFIGS), default=1)
    args = parser.parse_args()
    {"pool": cmd_pool, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
