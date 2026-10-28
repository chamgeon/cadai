"""Search page: hybrid text search over part records, test reports and change notices, grouped by part.

The query and filters are remembered for the session, so coming back from a part shows the same results.
"""
import html
import re

import streamlit as st

from src.ui import data, widgets
from src.ui.widgets import bi, bi_lines, bi_plain

PART_NO = re.compile(r"\b[A-Z]{2}-\d{5}\b")
EXAMPLES = ("crack near bolt hole", "볼트 구멍 균열", "salt spray corrosion", "wall thickness increased after failure")
FILTERS = {"part_class": ("분류", "Class"), "material": ("재질", "Material"), "status": ("상태", "Status"),
           "result": ("시험 결과", "Test result")}
KINDS = {"all": bi_plain("전체 문서", "All documents"), "test": bi_plain("시험 성적서", "Test reports"),
         "eco": bi_plain("변경 통지", "Change notices"), "card": bi_plain("부품 기록", "Part records")}
KIND_LABEL = {"card": ("부품 기록", "part record"), "test": ("시험 성적서", "test report"),
              "eco": ("변경 통지", "change notice")}
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
query = st.text_input(bi("검색", "Search"), key=remember("search_q", ""),
                      placeholder="e.g. crack near bolt hole",
                      help="부품 기록, 시험 성적서, 변경 통지를 한국어·영어로 검색: 키워드(BM25) + 의미(BGE-M3).  \n"
                           "Part records, test reports and change notices in Korean and English: "
                           "keywords (BM25) + meaning (BGE-M3).").strip()

options = data.filter_options()
cols = st.columns([1, 1, 1, 1, 1.2])
chosen = {field: col.multiselect(bi(*label), options[field], key=remember(f"search_{field}", []),
                                 placeholder="Choose")
          for col, (field, label) in zip(cols, FILTERS.items())}
kind = cols[-1].selectbox(bi("문서", "Documents"), list(KINDS), format_func=KINDS.get, key=remember("search_kind", "all"))
for key in ("search_q", "search_kind", *(f"search_{f}" for f in FILTERS)):
    st.session_state[f"saved_{key}"] = st.session_state[key]

if not query:
    st.caption(bi("예시", "Try"))
    for col, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        col.button(example, on_click=set_query, args=(example,), width="stretch")
    st.stop()

for part_no in PART_NO.findall(query.upper()):
    widgets.part_link(part_no, label=f"{part_no} {bi('열기', 'open')}")

filters = tuple((f, tuple(v)) for f, v in chosen.items() if v) + ((("kind", (kind,)),) if kind != "all" else ())
with st.spinner(bi("검색 중…", "Searching…")):
    res = data.search(user, query, filters)
parts = res["parts"]
if not parts:
    st.info(bi_lines("결과 없음. 필터를 줄이거나 다른 단어로 검색해 보세요.", "Nothing found. Fewer filters or other words may help."))
    st.stop()
st.caption(f"{len(parts)} parts from {res['n_chunks']} matching documents · {res['seconds']} s"
           + (f" · first {MAX_PARTS} shown" if len(parts) > MAX_PARTS else ""))

for p in parts[:MAX_PARTS]:
    with st.container(border=True):
        img, body = st.columns([1, 6], vertical_alignment="center")
        img.image(str(widgets.render(p["part_id"])), width="stretch")
        with body:
            head, badges = st.columns([3, 2], vertical_alignment="center")
            with head:
                widgets.part_link(p["part_no"], label=f"{p['part_no']} · {p['name']}")
            badges.markdown(f"{p['part_class']} · {p['material']} {widgets.status_badge(p['status'])}",
                            unsafe_allow_html=True)
            for c in p["chunks"][:SNIPPETS]:
                if c["kind"] == "card":
                    text = c["text"].split("\n", 2)[-1].replace("\n", " ")  # skip the name lines
                else:
                    text = c["text"]
                ko, en = KIND_LABEL[c["kind"]]
                tag = f"**{ko}** :gray[:small[{en}]] {c['doc_id'] if c['kind'] != 'card' else ''} rev {c['rev']}"
                if c.get("result"):
                    tag += f" · {c['test_type']} {widgets.result_chip(c['result'])}"
                text = html.escape(text[:240] + ("…" if len(text) > 240 else ""))  # shown as HTML for the chip
                st.markdown(f"{tag}  \n{text}", unsafe_allow_html=True)
            if len(p["chunks"]) > SNIPPETS:
                n = len(p["chunks"]) - SNIPPETS
                st.caption(bi(f"일치 문서 {n}개 더", f"and {n} more matching document{'s' if n > 1 else ''}"))
