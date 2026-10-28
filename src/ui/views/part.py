"""Part page: renders, the PLM card, test reports, change notices, geometry, similar parts.

The part comes from ?part=BR-66254, so links from search, chat and similar parts land here.
"""
import pandas as pd
import streamlit as st

from src.ui import data, widgets
from src.ui.widgets import RESULT_COLOR, bi, bi_plain

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
st.text_input(bi("부품번호", "Part number"), key="part_box", on_change=typed, placeholder="e.g. BR-66254")
# Write it back every run: a page_link into this same page (similar parts, demo links) hands the new part to the
# server but leaves the old one in the address bar, and the next rerun (any widget) would send that one back.
if part_no:
    st.query_params["part"] = part_no
else:
    st.query_params.pop("part", None)

if not part_no:
    st.caption(bi("검색에서 부품을 열거나, 아래 부품을 열어 보세요", "open a part from Search, or try one of these"))
    for demo in DEMO_PARTS:
        widgets.part_link(demo)
    st.stop()

p = data.part(user, part_no)
if p is None:
    st.warning(bi(f"부품 없음: {part_no}", f"no part {part_no} found"))  # also for parts in other programs: their existence doesn't leak
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
            f"{latest['material']}{size} · {bi('프로그램', 'program')} {card['program']} · {latest['owner_team']}")

left, right = st.columns([3, 2], gap="large")
with left:
    widgets.render_views([widgets.render(pid, view) for view in range(6)], widgets.glb(pid), key="part_3d")
with right:
    st.caption(bi("PLM 기록 · 질문 에이전트가 읽는 것과 같은 기록", "PLM record, as the Q&A agent reads it"))
    with st.container(border=True):
        st.markdown(widgets.card_markdown(card["text"]))
    if geo:
        a, b, c = st.columns(3)
        a.metric(bi("질량", "Mass"), f"{latest['mass_g']:g} g")
        b.metric(bi("부피", "Volume"), f"{geo['volume'] / 1000:,.1f} cm³")
        c.metric(bi("면 수", "Faces"), f"{geo['n_faces']:,}")

tests, ecos = pd.DataFrame(p["tests"]), pd.DataFrame(p["ecos"])
n_fail = sum(t["result"] == "fail" for t in p["tests"])
tab_tests, tab_ecos, tab_revs, tab_geo = st.tabs([
    bi(f"시험 성적서 ({len(tests)}{f', 실패 {n_fail}' if n_fail else ''})", "Test reports"),
    bi(f"변경 통지 ({len(ecos)})", "Change notices"), bi(f"리비전 ({len(p['revisions'])})", "Revisions"),
    bi("형상", "Geometry")])
with tab_tests:
    if tests.empty:
        st.caption(bi("시험 성적서 없음", "no test reports"))
    else:
        styled = tests.style.map(lambda r: f"color: {RESULT_COLOR.get(r, 'inherit')}; font-weight: 600",
                                 subset=["result"])
        st.dataframe(styled, hide_index=True, width="stretch",
                     column_config={"note": st.column_config.TextColumn("note", width="large")})
with tab_ecos:
    if ecos.empty:
        st.caption(bi("변경 통지 없음", "no change notices"))
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
            ("원본 파일 · source file", geo["file_name"]), ("작성 시스템 · exported by", geo["step_system"] or "?"),
            ("솔리드 · solid", f"{geo['solid_index']} · valid {'yes' if geo['is_valid'] else 'no'}"),
            ("면적 · area (mm²)", num(geo["area"], ",.0f")), ("채움 비율 · fill ratio", num(geo["fill_ratio"], ".2f")),
            ("구형도 · sphericity", num(geo["sphericity"], ".2f")), ("모서리 · edges", num(geo["n_edges"], ",")),
            ("면 종류 · face types", ", ".join(f"{k} {v}" for k, v in faces.items())),
            ("메시 삼각형 · mesh triangles", ", ".join(f"{k} {v:,}" for k, v in p["meshes"].items())),
        ], columns=[bi_plain("항목", "property"), bi_plain("값", "value")]), hide_index=True)

st.subheader(bi("유사 부품", "Similar parts"))
st.caption(bi("형상 기준: 부품당 렌더 6장을 DINOv2로 비교, 열람 가능한 부품만 순위에 포함",
              "by shape: 6 renders per part, DINOv2; only parts you may see are ranked"))
limit, max_size_ratio = widgets.similar_controls("part")
sim = data.similar(user, part_no, limit, max_size_ratio)
if "error" in sim or not sim["similar"]:
    st.caption(bi("유사 부품 없음", "no similar parts found"))
else:
    widgets.similar_grid(sim["similar"])
