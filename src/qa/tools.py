"""Step 6: the tools the Q&A agent may call, each enforcing the user's programs itself.

The LLM never writes SQL and never sees a record outside the user's programs:
    lookup_part    one part's card plus its test reports and change notices
    query_parts    count / list / group parts by fields; parameters only, the SQL is fixed here
    search_notes   hybrid search over cards, test notes and change notices (src.qa.textindex)
    similar_parts  step 4's image similarity, restricted to visible parts before ranking
A part in another program and a part that doesn't exist get the same "not found"
answer, so the existence of a hidden part never leaks. The database is opened
read-only: no tool can write.
"""
import json
import sqlite3

from qdrant_client import models

from src.pipeline import index
from src.qa import textindex
from src.config import DB_PATH
from src.qa.plm import USERS

GROUP_BY = ("part_class", "material", "status")
LIST_LIMIT = 50


def _json(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


class Tools:
    def __init__(self, user):
        self.user = user
        self.programs = USERS[user]
        self.conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row
        self.qc = index.client()
        marks = ", ".join("?" * len(self.programs))
        self.in_programs = f"program IN ({marks})"

    def run(self, name, args):
        """-> (result text, is_error). Unknown tools and bad arguments come back as tool errors, not exceptions."""
        fn = {"lookup_part": self.lookup_part, "query_parts": self.query_parts,
              "search_notes": self.search_notes, "similar_parts": self.similar_parts}.get(name)
        if fn is None:
            return f"unknown tool {name}", True
        try:
            return _json(fn(**args)), False
        except (TypeError, ValueError) as e:
            return f"bad arguments: {e}", True

    def _card(self, part_no):
        return self.conn.execute(f"SELECT * FROM part_cards WHERE part_no = ? AND {self.in_programs}",
                                 (part_no, *self.programs)).fetchone()

    # ------------------------------------------------------------ tools

    def lookup_part(self, part_no):
        card = self._card(part_no)
        if card is None:
            return {"error": f"no part {part_no} found"}
        tests = self.conn.execute("SELECT report_id, rev, test_type, result, test_date, note FROM doc_test_reports"
                                  " WHERE part_no = ? ORDER BY rev, test_type", (part_no,)).fetchall()
        ecos = self.conn.execute("SELECT eco_id, from_rev, to_rev, eco_date, reason FROM doc_change_notices"
                                 " WHERE part_no = ? ORDER BY from_rev", (part_no,)).fetchall()
        return {"card": card["text"], "test_reports": [dict(t) for t in tests],
                "change_notices": [dict(e) for e in ecos]}

    def query_parts(self, aggregate="list", group_by=None, part_class=None, material=None, status=None,
                    test_type=None, result=None, test_revision="any", min_size_mm=None, max_size_mm=None):
        """Latest revision of each part; size = longest bbox side from step 1's measured geometry."""
        where, args = [f"i.{self.in_programs}"], list(self.programs)
        for col, value in (("part_class", part_class), ("material", material), ("status", status)):
            if value is not None:
                where.append(f"i.{col} = ?")
                args.append(value)
        if min_size_mm is not None:
            where.append("p.dim_1 >= ?")
            args.append(min_size_mm)
        if max_size_mm is not None:
            where.append("p.dim_1 <= ?")
            args.append(max_size_mm)
        if test_type is not None or result is not None:
            sub = ["t.part_no = i.part_no"] + (["t.rev = i.rev"] if test_revision == "latest" else [])
            for col, value in (("test_type", test_type), ("result", result)):
                if value is not None:
                    sub.append(f"t.{col} = ?")
                    args.append(value)
            where.append(f"EXISTS (SELECT 1 FROM doc_test_reports t WHERE {' AND '.join(sub)})")
        base = f"FROM plm_items i JOIN parts p ON p.part_id = i.part_id WHERE {' AND '.join(where)}"

        if aggregate == "count":
            return {"count": self.conn.execute(f"SELECT count(*) {base}", args).fetchone()[0]}
        if aggregate == "group":
            if group_by not in GROUP_BY:
                raise ValueError(f"group_by must be one of {GROUP_BY}")
            rows = self.conn.execute(f"SELECT i.{group_by} AS key, count(*) AS n {base} GROUP BY 1 ORDER BY 2 DESC",
                                     args).fetchall()
            return {"groups": {r["key"]: r["n"] for r in rows}}
        rows = self.conn.execute(
            f"SELECT i.part_no, i.rev, i.name, i.part_class, i.material, i.status, i.mass_g,"
            f" round(p.dim_1, 1) AS d1, round(p.dim_2, 1) AS d2, round(p.dim_3, 1) AS d3 {base}"
            f" ORDER BY i.part_no LIMIT {LIST_LIMIT + 1}", args).fetchall()
        out = {"parts": [dict(r) for r in rows[:LIST_LIMIT]]}
        if len(rows) > LIST_LIMIT:
            out["truncated"] = f"more than {LIST_LIMIT} parts match; narrow the filters or use aggregate=count"
        return out

    def search_notes(self, query, keywords=(), kind=None, result=None, test_type=None, material=None,
                     part_class=None, part_nos=None, limit=10):
        filters = {k: v for k, v in (("kind", kind), ("result", result), ("test_type", test_type),
                                     ("material", material), ("part_class", part_class),
                                     ("part_no", part_nos or None)) if v is not None}
        hits = textindex.search(self.qc, query, self.user, "hybrid", keywords, filters, min(limit, 20))
        keep = ("doc_id", "kind", "part_no", "rev", "lang", "header", "text")
        return {"results": [{k: h.payload.get(k) for k in keep} for h in hits]}

    def similar_parts(self, part_no, limit=5, max_size_ratio=None):
        """Parts that look like `part_no`, among those the user may see; other solids of its own file excluded."""
        card = self._card(part_no)
        if card is None:
            return {"error": f"no part {part_no} found"}
        me = self.qc.retrieve(index.COLLECTION, [index.point_id(card["part_id"])], with_vectors=["views"])[0]
        size = self.conn.execute("SELECT dim_1 FROM parts WHERE part_id = ?", (card["part_id"],)).fetchone()[0]
        found = self.similar_to(me.vector["views"], size, limit, max_size_ratio, exclude_file=me.payload["file_id"])
        return {"query": {"part_no": part_no, "size_mm": round(size, 1)}, **found}

    def similar_to(self, views, size, limit=5, max_size_ratio=None, exclude_file=None):
        """Parts that look like a shape given by its 6 view vectors, among those the user may see.

        For a shape with no part number yet (an uploaded file, not in the index); size is its longest bbox
        side, for size_ratio. Not offered to the LLM: it can't produce view vectors.
        """
        visible = {r["part_id"]: r for r in self.conn.execute(
            f"SELECT i.part_id, i.part_no, i.name, i.material, i.status, p.dim_1, p.dim_2, p.dim_3"
            f" FROM plm_items i JOIN parts p ON p.part_id = i.part_id WHERE i.{self.in_programs}", self.programs)}
        flt = models.Filter(
            must=[models.FieldCondition(key="part_id", match=models.MatchAny(any=list(visible)))],
            must_not=[models.FieldCondition(key="file_id", match=models.MatchValue(value=exclude_file))]
            if exclude_file else None)
        hits = self.qc.query_points(index.COLLECTION, query=views, using="views", query_filter=flt,
                                    limit=min(limit, 20) * (4 if max_size_ratio else 1)).points
        out = []
        for h in hits:
            v = visible[h.payload["part_id"]]
            ratio = v["dim_1"] / size
            if max_size_ratio and not 1 / max_size_ratio <= ratio <= max_size_ratio:
                continue
            tests = self.conn.execute("SELECT result, count(*) FROM doc_test_reports WHERE part_no = ? GROUP BY 1",
                                      (v["part_no"],)).fetchall()
            out.append({"part_no": v["part_no"], "name": v["name"], "material": v["material"], "status": v["status"],
                        "similarity": round(h.score / 6, 3),  # MaxSim over 6 views, /6 -> 0..1
                        "size_mm": [round(v["dim_1"], 1), round(v["dim_2"], 1), round(v["dim_3"], 1)],
                        "size_ratio": round(ratio, 2), "test_results": dict(tests)})
            if len(out) == limit:
                break
        return {"similar": out}
