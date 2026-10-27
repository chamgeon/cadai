"""Search page: hybrid text search over part records, test reports and change notices, grouped by part.

The query and filters are remembered for the session, so coming back from a part shows the same results.
"""
import re

import streamlit as st

from src.ui import data, widgets
from src.ui.widgets import RESULT_COLOR

PART_NO = re.compile(r"\b[A-Z]{2}-\d{5}\b")
EXAMPLES = ("crack near bolt hole", "볼트 구멍 균열", "salt spray corrosion", "wall thickness increased after failure")
FILTERS = {"part_class": "Class", "material": "Material", "status": "Status", "result": "Test result"}
KINDS = {"all": "All documents", "test": "Test reports", "eco": "Change notices", "card": "Part records"}
KIND_LABEL = {"card": "part record", "test": "test report", "eco": "change notice"}
MAX_PARTS = 15
SNIPPETS = 2


def remember(key, default):
    """Streamlit drops a widget's state when its page isn't shown; restore it from a copy that survives."""
    if key not in st.session_state:
        st.session_state[key] = st.session_state.get(f"saved_{key}", default)
    return key


def set_query(q):
    st.session_state.search_q = q


user = st.session_state.user
query = st.text_input("Search", key=remember("search_q", ""),
                      placeholder="e.g. crack near bolt hole · 볼트 구멍 균열",
                      help="Searches part records, test reports and change notices in English and Korean: "
                           "keywords (BM25) and meaning (BGE-M3) combined.").strip()

options = data.filter_options()
cols = st.columns([1, 1, 1, 1, 1.2])
chosen = {field: col.multiselect(label, options[field], key=remember(f"search_{field}", []))
          for col, (field, label) in zip(cols, FILTERS.items())}
kind = cols[-1].selectbox("Documents", list(KINDS), format_func=KINDS.get, key=remember("search_kind", "all"))
for key in ("search_q", "search_kind", *(f"search_{f}" for f in FILTERS)):
    st.session_state[f"saved_{key}"] = st.session_state[key]

if not query:
    st.caption("Try:")
    for col, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        col.button(example, on_click=set_query, args=(example,), width="stretch")
    st.stop()

for part_no in PART_NO.findall(query.upper()):
    widgets.part_link(part_no, label=f"Open {part_no}")

filters = tuple((f, tuple(v)) for f, v in chosen.items() if v) + ((("kind", (kind,)),) if kind != "all" else ())
with st.spinner("Searching…"):
    res = data.search(user, query, filters)
parts = res["parts"]
if not parts:
    st.info("Nothing found. Fewer filters, or other words, may help.")
    st.stop()
st.caption(f"{len(parts)} parts from {res['n_chunks']} matching documents · {res['seconds']} s"
           + (f" · showing the first {MAX_PARTS}" if len(parts) > MAX_PARTS else ""))

for p in parts[:MAX_PARTS]:
    with st.container(border=True):
        img, body = st.columns([1, 6], vertical_alignment="center")
        img.image(str(widgets.render(p["part_id"])), width="stretch")
        with body:
            head, badges = st.columns([3, 2], vertical_alignment="center")
            with head:
                widgets.part_link(p["part_no"], label=f"{p['part_no']} · {p['name']}")
            badges.markdown(f"{p['part_class']} · {p['material']} {widgets.status_badge(p['status'])}")
            for c in p["chunks"][:SNIPPETS]:
                if c["kind"] == "card":
                    text = c["text"].split("\n", 2)[-1].replace("\n", " ")  # skip the name lines
                else:
                    text = c["text"]
                tag = f"**{KIND_LABEL[c['kind']]}** {c['doc_id'] if c['kind'] != 'card' else ''} rev {c['rev']}"
                if c.get("result"):
                    tag += f" · {c['test_type']} :{RESULT_COLOR[c['result']]}-badge[{c['result']}]"
                st.markdown(f"{tag}  \n{text[:240] + ('…' if len(text) > 240 else '')}")
            if len(p["chunks"]) > SNIPPETS:
                n = len(p["chunks"]) - SNIPPETS
                st.caption(f"and {n} more matching document{'s' if n > 1 else ''}")
