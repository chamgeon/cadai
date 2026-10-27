"""Step 7: the engineer-facing app: search, part detail with similar parts, Q&A, upload.

    streamlit run src/ui/app.py

The sidebar's user decides what every page may show (src.qa.plm USERS: user ->
programs); the tools enforce it, the pages only pass it on.
"""
import logging
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # `streamlit run` puts src/ui on the path, not the root
# Models load from the local cache only: startup must not depend on the network (demo Wi-Fi), nor wait on it.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import streamlit as st

# The file watcher (runOnSave) probes every loaded module; transformers' lazy image processors then try to
# import torchvision, which isn't installed, and each logs a traceback (~100). Harmless: the import fails first.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

from src.config import DB_PATH
from src.qa.plm import USERS
from src.ui import resources

st.set_page_config(page_title="cadai · part search", page_icon=":material/precision_manufacturing:", layout="wide")


@st.cache_data(ttl=30)
def index_counts():
    try:
        qc = resources.qdrant()
        return {c: qc.count(c).count for c in ("parts", "docs")}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


@st.cache_data
def visible_parts(user):
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    programs = USERS[user]
    n = conn.execute(f"SELECT count(*) FROM part_cards WHERE program IN ({', '.join('?' * len(programs))})",
                     programs).fetchone()[0]
    total = conn.execute("SELECT count(*) FROM part_cards").fetchone()[0]
    conn.close()
    return n, total


def user_changed():
    st.session_state.pop("chat", None)  # an answer shown to admin must not stay on screen for normal


cad = resources.cad()  # first: it starts loading in its own process without waiting
models = resources.models()

with st.sidebar:
    st.selectbox("User", list(USERS), key="user", on_change=user_changed,
                 format_func=lambda u: f"{u}  ·  programs {', '.join(USERS[u])}")
    n, total = visible_parts(st.session_state.user)
    st.caption(f"Sees {n:,} of {total:,} part numbers")

    st.divider()
    counts = index_counts()
    if "error" in counts:
        st.error(f"Qdrant unreachable: start it with `docker start qdrant`\n\n{counts['error']}")
    else:
        st.caption(f"Index: {counts['parts']:,} shapes · {counts['docs']:,} text chunks")
    warm = cad.ready()
    st.caption(f"Models ready in {models.seconds} s · CAD worker "
               + (f"ready in {warm['seconds']} s" if warm.get("ok") else f"failed: {warm.get('error')}"))

pages = [
    st.Page("views/search.py", title="Search", icon=":material/search:", default=True),
    st.Page("views/part.py", title="Part", icon=":material/category:"),
    st.Page("views/ask.py", title="Ask", icon=":material/chat:"),
    st.Page("views/upload.py", title="Upload", icon=":material/upload_file:"),
]
st.navigation(pages).run()
