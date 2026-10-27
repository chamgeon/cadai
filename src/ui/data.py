"""Step 7: what the pages read, permission first.

A part is shown only if src.qa.tools says this user may see it (lookup_part,
similar_parts), or if the text search returned it (filtered by the user's
programs inside Qdrant); its other rows are read for display after that. A
hidden part and a missing one come back the same: None.
"""
import sqlite3
import time

import streamlit as st

from src.config import DB_PATH
from src.qa import textindex
from src.qa.tools import Tools
from src.ui import resources


def _conn():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(ttl=60, show_spinner=False)
def part(user, part_no):
    found = Tools(user).lookup_part(part_no)
    if "error" in found:
        return None
    conn = _conn()
    card = dict(conn.execute("SELECT * FROM part_cards WHERE part_no = ?", (part_no,)).fetchone())
    revisions = [dict(r) for r in conn.execute(
        "SELECT rev, status, rev_date, name, part_class, material, mass_g, program, owner_team FROM plm_items"
        " WHERE part_no = ? ORDER BY rev", (part_no,))]
    geometry = conn.execute("SELECT p.*, f.file_name, f.step_system FROM parts p JOIN files f USING (file_id)"
                            " WHERE p.part_id = ?", (card["part_id"],)).fetchone()
    meshes = {r["level"]: r["n_tris"] for r in conn.execute(
        "SELECT level, n_tris FROM meshes WHERE part_id = ? AND status = 'ok'", (card["part_id"],))}
    conn.close()
    return {"card": card, "latest": revisions[-1], "revisions": revisions, "tests": found["test_reports"],
            "ecos": found["change_notices"], "geometry": dict(geometry) if geometry else None, "meshes": meshes}


def _with_part_ids(out):
    """Each similar hit's part_id, for its renders (the agent's tools don't return it: it doesn't need it)."""
    conn = _conn()
    for s in out.get("similar", []):
        s["part_id"] = conn.execute("SELECT part_id FROM part_cards WHERE part_no = ?", (s["part_no"],)).fetchone()[0]
    conn.close()
    return out


@st.cache_data(ttl=60, show_spinner=False)
def similar(user, part_no, limit=10, max_size_ratio=None):
    return _with_part_ids(Tools(user).similar_parts(part_no, limit, max_size_ratio))


def process_upload(step_path, out_dir):
    """Steps 1-4 for a file that is not in the index: extract, mesh and render in the CAD worker (renders go to
    out_dir), then DINOv2 on the renders. Nothing is stored. -> the worker's reply, each ok part with "views"."""
    res = resources.cad().call(cmd="process", step=str(step_path), out=str(out_dir))
    if not res["ok"]:
        return res
    start = time.perf_counter()
    for p in res["parts"]:
        if p["status"] == "ok":
            p["views"] = resources.on_tool_thread(resources.models().embedder, p["renders"]).tolist()
    res["seconds"]["embed"] = round(time.perf_counter() - start, 2)
    return res


@st.cache_data(ttl=300, show_spinner=False)
def similar_upload(user, upload_key, _views, size, limit=10, max_size_ratio=None):
    """Tools.similar_to for an uploaded part; upload_key (file hash + part) stands in for the views in the cache."""
    return _with_part_ids(Tools(user).similar_to(_views, size, limit, max_size_ratio))


@st.cache_data(show_spinner=False)
def filter_options():
    conn = _conn()
    out = {col: [r[0] for r in conn.execute(f"SELECT DISTINCT {col} FROM plm_items ORDER BY 1")]
           for col in ("part_class", "material", "status")}
    out.update({col: [r[0] for r in conn.execute(f"SELECT DISTINCT {col} FROM doc_test_reports ORDER BY 1")]
                for col in ("test_type", "result")})
    conn.close()
    return out


@st.cache_data(ttl=300, show_spinner=False)
def search(user, query, filters=(), limit=40):
    """Hybrid text search (BM25 + BGE-M3 dense) grouped by part, best-ranked part first.

    filters: ((field, (values...)), ...), hashable for the cache. -> {"parts": [...], "n_chunks", "seconds"}
    """
    start = time.perf_counter()
    hits = resources.on_tool_thread(textindex.search, resources.qdrant(), query, user, "hybrid", (),
                                    {k: list(v) for k, v in filters}, limit)
    parts = {}
    for h in hits:
        c = h.payload
        p = parts.setdefault(c["part_no"], {k: c[k] for k in ("part_no", "part_id", "part_class", "status")}
                             | {"chunks": []})
        p["chunks"].append(c)
    conn = _conn()
    for p in parts.values():  # latest revision's: a test chunk carries the material of the revision tested
        p["name"], p["material"] = conn.execute("SELECT name, material FROM plm_items WHERE part_no = ?"
                                                " ORDER BY rev DESC LIMIT 1", (p["part_no"],)).fetchone()
    conn.close()
    return {"parts": list(parts.values()), "n_chunks": len(hits), "seconds": round(time.perf_counter() - start, 2)}
