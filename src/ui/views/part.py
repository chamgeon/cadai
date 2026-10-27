"""Part page: renders, the PLM card, test reports, change notices, geometry, similar parts.

The part comes from ?part=BR-66254, so links from search, chat and similar parts land here.
"""
import pandas as pd
import streamlit as st

from src.ui import data, widgets
from src.ui.widgets import RESULT_COLOR

DEMO_PARTS = ("BR-66254", "BK-11639", "BD-28744")  # the bracket modeled in step 2; parts with failed tests


def typed():
    """The box changed: that is the part now. Runs before the page, so the page reads it from the URL."""
    st.query_params["part"] = st.session_state.part_box = st.session_state.part_box.strip().upper()
    if not st.session_state.part_box:
        st.session_state.last_part = ""  # cleared on purpose: don't bring the last part back


# ?part= is the source of truth (without one, the last part shown); the box only mirrors it. A box with
# value=part_no and no key would be
# matched to its old state by Streamlit whenever part_no repeats, and hand back stale text (e.g. "" after
# clearing it, so a link to that part would open the empty page).
user = st.session_state.user
part_no = st.query_params.get("part") or st.session_state.get("last_part", "")  # sidebar links carry no ?part=
if "part_box" not in st.session_state or st.session_state.get("part_box_for") != part_no:
    st.session_state.part_box = part_no  # arrived by a link or the URL: show it in the box
st.session_state.part_box_for = part_no
st.text_input("Part number", key="part_box", on_change=typed, placeholder="e.g. BR-66254")
# Write it back every run: a page_link into this same page (similar parts, demo links) hands the new part to the
# server but leaves the old one in the address bar, and the next rerun (any widget) would send that one back.
if part_no:
    st.query_params["part"] = part_no
else:
    st.query_params.pop("part", None)

if not part_no:
    st.caption("Open a part from Search, or try one of these:")
    for demo in DEMO_PARTS:
        widgets.part_link(demo)
    st.stop()

p = data.part(user, part_no)
if p is None:
    st.warning(f"No part {part_no} found.")  # also for parts in other programs: their existence doesn't leak
    st.stop()

st.session_state.last_part = part_no
if st.session_state.get("shown_part") != part_no:
    st.session_state.shown_part = part_no
    widgets.scroll_to_top(part_no)

card, latest, geo = p["card"], p["latest"], p["geometry"]
pid = card["part_id"]
size = " · " + "×".join(f"{geo[k]:.0f}" for k in ("dim_1", "dim_2", "dim_3")) + " mm" if geo else ""
st.title(f"{part_no} · {latest['name']}")
st.markdown(f"rev **{card['rev']}** {widgets.status_badge(latest['status'])} · {latest['part_class']} · "
            f"{latest['material']}{size} · program {card['program']} · {latest['owner_team']}")

left, right = st.columns([3, 2], gap="large")
with left:
    widgets.render_views([widgets.render(pid, view) for view in range(6)], widgets.glb(pid), key="part_3d")
with right:
    st.caption("PLM record · the Q&A agent reads this same record as plain text")
    with st.container(border=True):
        st.markdown(widgets.card_markdown(card["text"]))
    if geo:
        a, b, c = st.columns(3)
        a.metric("Mass (g)", f"{latest['mass_g']:g}")
        b.metric("Volume (cm³)", f"{geo['volume'] / 1000:,.1f}")
        c.metric("Faces", f"{geo['n_faces']:,}")

tests, ecos = pd.DataFrame(p["tests"]), pd.DataFrame(p["ecos"])
n_fail = sum(t["result"] == "fail" for t in p["tests"])
tab_tests, tab_ecos, tab_revs, tab_geo = st.tabs([
    f"Test reports ({len(tests)}{f', {n_fail} failed' if n_fail else ''})", f"Change notices ({len(ecos)})",
    f"Revisions ({len(p['revisions'])})", "Geometry"])
with tab_tests:
    if tests.empty:
        st.caption("No test reports.")
    else:
        styled = tests.style.map(lambda r: f"color: {RESULT_COLOR.get(r, 'inherit')}; font-weight: 600",
                                 subset=["result"])
        st.dataframe(styled, hide_index=True, width="stretch",
                     column_config={"note": st.column_config.TextColumn("note", width="large")})
with tab_ecos:
    if ecos.empty:
        st.caption("No change notices.")
    else:
        st.dataframe(ecos, hide_index=True, width="stretch",
                     column_config={"reason": st.column_config.TextColumn("reason", width="large")})
with tab_revs:
    st.dataframe(pd.DataFrame(p["revisions"]), hide_index=True, width="stretch")
with tab_geo:
    if geo:
        faces = {k[2:]: geo[k] for k in ("n_plane", "n_cylinder", "n_cone", "n_sphere", "n_torus", "n_bspline",
                                         "n_other_face") if geo[k]}
        num = lambda v, f: "?" if v is None else format(v, f)
        st.dataframe(pd.DataFrame([
            ("source file", geo["file_name"]), ("exported by", geo["step_system"] or "?"),
            ("solid", f"{geo['solid_index']} · valid {'yes' if geo['is_valid'] else 'no'}"),
            ("area (mm²)", num(geo["area"], ",.0f")), ("fill ratio", num(geo["fill_ratio"], ".2f")),
            ("sphericity", num(geo["sphericity"], ".2f")), ("edges", num(geo["n_edges"], ",")),
            ("face types", ", ".join(f"{k} {v}" for k, v in faces.items())),
            ("mesh triangles", ", ".join(f"{k} {v:,}" for k, v in p["meshes"].items())),
        ], columns=["property", "value"]), hide_index=True)

st.subheader("Similar parts")
st.caption("By shape: 6 renders per part compared with DINOv2. Only parts you may see are ranked.")
limit, max_size_ratio = widgets.similar_controls("part")
sim = data.similar(user, part_no, limit, max_size_ratio)
if "error" in sim or not sim["similar"]:
    st.caption("No similar parts found.")
else:
    widgets.similar_grid(sim["similar"])
