"""Upload page: a STEP file with no part number yet -> the existing parts it looks like, and their history.

The file goes through the pipeline's own stage functions (extract, mesh, render, DINOv2), so its renders
match the indexed ones pixel for pixel, but nothing is stored: an upload is a question, not an ingest.
Results are kept for the session by file hash; the renders live in a temp folder.
"""
import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from src.config import RAW
from src.ui import data, widgets

SAMPLE = RAW / "my_bracket.step"  # the bracket modeled in step 2: it is in the index, as BR-66254
SAME_PART = 0.97  # similarity above which the upload is almost surely an existing part (DINOv2: same renders -> 1.0)
STAGES = ("extract", "mesh", "render", "embed")


def use_sample():
    st.session_state.upload_sample = True


def ask_about(question):
    st.session_state.ask_prefill = question


if st.session_state.get("ask_prefill"):  # "Ask about these parts" was clicked: the Ask page runs the question
    st.switch_page("views/ask.py")

user = st.session_state.user
uploads = st.session_state.setdefault("uploads", {})  # sha256 -> processed result

st.caption("A new part has no part number yet, only a shape. Drop its STEP file to find the parts it looks like. "
           "It is read and rendered like the pipeline does, but not stored.")
file = st.file_uploader("STEP file", type=["step", "stp"])
if file is None:
    st.button(f"Try with {SAMPLE.name}", on_click=use_sample, help="The bracket modeled in step 2")
    if not st.session_state.get("upload_sample"):
        st.stop()
    name, content = SAMPLE.name, SAMPLE.read_bytes()
else:
    st.session_state.upload_sample = False
    name, content = Path(file.name).name, file.getvalue()

sha = hashlib.sha256(content).hexdigest()
if sha not in uploads:
    work = Path(tempfile.mkdtemp(prefix="cadai-upload-"))
    (work / name).write_bytes(content)
    with st.status(f"Reading {name}…") as status:
        res = data.process_upload(work / name, work / "parts")
        state = "complete" if res["ok"] else "error"
        status.update(label=f"Read {name} in {sum(res['seconds'].values()):.1f} s", state=state)
    uploads[sha] = res
res = uploads[sha]
if not res["ok"]:
    st.error(f"Could not read {name}: {res['error']}")
    st.stop()

parts = [p for p in res["parts"] if p["status"] == "ok"]
skipped = [p for p in res["parts"] if p["status"] != "ok"]
st.caption(" · ".join(f"{stage} {res['seconds'][stage]} s" for stage in STAGES if stage in res["seconds"])
           + (f" · skipped {len(skipped)} solid{'s' if len(skipped) > 1 else ''}: "
              + "; ".join(f"{p['part_id']} {p['status']} ({p['error']})" for p in skipped) if skipped else ""))
if not parts:
    st.error("No solid in this file passed validation.")
    st.stop()
if len(parts) > 1:
    part = st.selectbox(f"Solid ({len(parts)} in this file)", parts,
                        format_func=lambda p: f"{p['part_id']} · {'×'.join(f'{d:.0f}' for d in p['dims'])} mm")
else:
    part = parts[0]

dims = "×".join(f"{d:.0f}" for d in part["dims"])
st.subheader(f"{name} · {dims} mm")
left, right = st.columns([3, 2], gap="large")
with left:
    widgets.render_views(part["renders"], part["glb"], key="upload_3d")
with right:
    a, b = st.columns(2)
    a.metric("Volume (cm³)", f"{part['volume'] / 1000:,.1f}")
    b.metric("Triangles", f"{part['n_tris']:,}")
    st.caption("No part number, no PLM record: everything below comes from the shape alone.")

st.subheader("Existing parts that look like it")
limit, max_size_ratio = widgets.similar_controls("upload")
sim = data.similar_upload(user, f"{sha}/{part['part_id']}", part["views"], part["dims"][0], limit, max_size_ratio)
hits = sim["similar"]
if not hits:
    st.caption("No similar parts found among the parts you may see.")
    st.stop()
top = hits[0]
if top["similarity"] >= SAME_PART:
    st.success(f"This looks like an existing part: **{top['part_no']}** ({top['name']}, similarity "
               f"{top['similarity']:.2f}, size ×{top['size_ratio']:.2f}). Reuse it rather than release a new number?")
names = ", ".join(h["part_no"] for h in hits[:5])
st.button("Ask about these parts", icon=":material/chat:", type="primary", on_click=ask_about, args=(
    f"I have a new part with no part number yet, {dims} mm. By shape it is most similar to {names}. "
    "Have any of these failed tests, and how were the failures fixed?",))
widgets.similar_grid(hits)
