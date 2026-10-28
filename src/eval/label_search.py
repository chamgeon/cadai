"""Step 8B: grade pooled similar-part results, one query at a time.

    python -m src.eval.search_quality pool      # first: builds the candidate pool
    streamlit run src/eval/label_search.py
    streamlit run src/eval/label_search.py -- --round 2

Blind: candidates are shuffled, and neither the config that found them nor their part
number or class is shown (BR-, PL- ... prefixes would hint at the answer). Every click
is saved, so you can stop and come back.

Round 2 (`-- --round 2`) is a head-to-head: only maxsim's and voyage's top 5s, in a new
order, none pre-graded, saved to their own file. Both models are graded in one sitting,
so a shift in how strictly you grade can't favour either.
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # `streamlit run` puts src/eval on the path, not the root

import streamlit as st

from src.config import DB_PATH, EVAL_RESULTS, RENDERS

POOL = EVAL_RESULTS / "step8_search_pool.json"  # same files as src.eval.search_quality, not imported: it loads torch
_parser = argparse.ArgumentParser()
_parser.add_argument("--round", type=int, default=1)  # streamlit passes what follows `--`
ROUND = _parser.parse_args().round
LABELS = EVAL_RESULTS / ("step8_search_labels.json" if ROUND == 1 else f"step8_search_labels_r{ROUND}.json")
CANDIDATES = "candidates" if ROUND == 1 else f"candidates_r{ROUND}"

GRADES = {0: "0 · not similar", 1: "1 · similar shape", 2: "2 · could reuse"}
HELP = ("**0** not similar: a different kind of part.  \n"
        "**1** similar shape: same kind of part (both L-brackets, both stepped shafts), but proportions, "
        "features or size differ enough that you'd still design your own.  \n"
        "**2** could reuse: same shape and a similar size; an engineer would check this part before making a new one.")
QUERY_VIEWS = (5, 0, 2, 4)  # iso, front, left, top
TILE_VIEWS = (5, 0, 4)

st.set_page_config(page_title=f"label search, round {ROUND}", layout="wide")


@st.cache_data
def pool():
    return json.loads(POOL.read_text())


@st.cache_data
def size(part_no):
    conn = sqlite3.connect(DB_PATH)
    return conn.execute("SELECT p.dim_1, p.dim_2, p.dim_3, c.part_id FROM part_cards c JOIN parts p USING (part_id)"
                        " WHERE c.part_no = ?", (part_no,)).fetchone()


def load_labels():
    return json.loads(LABELS.read_text()) if LABELS.exists() else {}


def save(query, cand):
    labels = load_labels()
    labels.setdefault(query, {})[cand] = st.session_state[f"g_{query}_{cand}"]
    LABELS.write_text(json.dumps(labels, indent=2, sort_keys=True))


def views(part_id, which, width):
    for col, v in zip(st.columns(len(which)), which):
        col.image(str(RENDERS / part_id / f"view_{v}.png"), width=width)


p, labels = pool(), load_labels()
queries = [q["part_no"] for q in p["queries"]]
done = {q: sum(c in labels.get(q, {}) for c in p[CANDIDATES][q]) for q in queries}
total, n_done = sum(len(c) for c in p[CANDIDATES].values()), sum(done.values())

with st.sidebar:
    st.progress(n_done / total, text=f"{n_done} / {total} pairs graded")
    # fixed labels and a key: a label that changes (e.g. with a progress count) makes it a new widget, back at query 1
    i = st.radio("Query", range(len(queries)), format_func=lambda i: f"query {i + 1}", key="query")
    left = [str(j + 1) for j, q in enumerate(queries) if done[q] < len(p[CANDIDATES][q])]
    st.caption("Not finished: " + (", ".join(left) if left else "none"))
    st.markdown(HELP)

query = queries[i]
qd = size(query)
st.subheader(f"Query {i + 1} of {len(queries)}   ·   {done[query]}/{len(p[CANDIDATES][query])} graded")
st.caption(f"{qd[0]:.0f} × {qd[1]:.0f} × {qd[2]:.0f} mm   ·   views: iso, front, left, top")
views(qd[3], QUERY_VIEWS, 220)
st.divider()

cands = p[CANDIDATES][query]
for row in range(0, len(cands), 3):
    for col, cand in zip(st.columns(3), cands[row:row + 3]):
        with col.container(border=True):
            cd = size(cand)
            views(cd[3], TILE_VIEWS, 120)
            st.caption(f"#{row + cands[row:row + 3].index(cand) + 1}   {cd[0]:.0f} × {cd[1]:.0f} × {cd[2]:.0f} mm"
                       f"   ({cd[0] / qd[0]:.2g}× the query)")
            key = f"g_{query}_{cand}"
            if key not in st.session_state and cand in labels.get(query, {}):
                st.session_state[key] = labels[query][cand]
            st.radio("grade", list(GRADES), format_func=GRADES.get, key=key, index=None, horizontal=True,
                     label_visibility="collapsed", on_change=save, args=(query, cand))
