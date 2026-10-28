"""Step 7: pieces shared by the pages: render paths, part links, badges, part tiles, similar-parts grid, 3D view.

Every label is bilingual, Korean first: bi() where Streamlit renders markdown (titles, captions, labels,
buttons, tabs, status), bi_lines() for whole sentences, bi_plain() where it shows plain text (menu,
placeholders, dropdown options)."""
import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import trimesh
from trimesh.visual.material import PBRMaterial

from src.config import MESH, RENDERS

PART_PAGE = "views/part.py"
VIEW_NAMES = ("정면 · front", "후면 · back", "좌측 · left", "우측 · right", "평면 · top", "iso")
ISO = 5
STATUS_COLOR = {"Released": "green", "In Work": "orange", "Obsolete": "gray"}
RESULT_COLOR = {"fail": "red", "conditional": "orange", "pass": "green"}
SIZE_FILTERS = {"전체 · any size": None, "1.5배 이내 · within 1.5×": 1.5, "2배 이내 · within 2×": 2.0}


def bi(ko, en):
    """Korean, then English small and grey (markdown)."""
    return f"{ko} :gray[:small[{en}]]"


def bi_lines(ko, en):
    """A sentence: Korean, then the English on the next line, small and grey."""
    return f"{ko}  \n:gray[:small[{en}]]"


def bi_plain(ko, en):
    return f"{ko} · {en}"


def render(part_id, view=ISO):
    return RENDERS / part_id / f"view_{view}.png"


def glb(part_id):
    return MESH / part_id / "coarse.glb"


def part_link(part_no, label=None):
    st.page_link(PART_PAGE, label=label or part_no, query_params={"part": part_no},
                 icon=":material/arrow_forward:")


def scroll_to_top(token):
    """Streamlit keeps the scroll position across page changes: a part opened from the bottom of another
    part's similar-parts grid would open at its own bottom. Call it only when the content changed; `token`
    (e.g. the part number) makes the element differ from the last run's, or Streamlit keeps the old one
    and the script doesn't run again."""
    st.html(f"<script>/* {token} */ (document.querySelector('[data-testid=\"stMain\"]') || window)"
            ".scrollTo(0, 0)</script>", unsafe_allow_javascript=True)


def scroll_to_last_message(token):
    """Put the newest chat message at the top of the view: Streamlit doesn't follow a growing chat, so a new
    question and its answer would appear below what's on screen. `token` as in scroll_to_top()."""
    st.html(f"<script>/* {token} */ [...document.querySelectorAll('[data-testid=\"stChatMessage\"]')].at(-1)"
            "?.scrollIntoView({block: 'start', behavior: 'smooth'})</script>", unsafe_allow_javascript=True)


def status_badge(status):
    return f":{STATUS_COLOR.get(status, 'gray')}-badge[{status}]"


def results_badges(counts):
    """{"fail": 2, "pass": 3} -> red/orange/green badges, worst first; "no tests" if empty."""
    shown = [f":{RESULT_COLOR[r]}-badge[{counts[r]} {r}]" for r in RESULT_COLOR if counts.get(r)]
    return " ".join(shown) or ":gray-badge[시험 없음] :gray[:small[no tests]]"


def card_markdown(text):
    """The PLM card (src.qa.textindex.build_cards, the text the agent reads) for people: one field per line with
    its name in bold, one revision or test per line. Display only; the agent keeps the plain text."""
    lines = text.split("\n")
    head, _, name = lines[0].partition(": ")
    rows, blocks = [f"**{head}** · {name}"], []
    for line in lines[1:]:
        label, _, rest = line.partition(": ")
        fields = line.rstrip(".").split(", ")
        if label in ("Revisions", "Tests"):
            blocks += ["  \n".join(rows), f"**{label}**", "\n".join(f"- {item}" for item in rest.rstrip(".").split("; "))]
            rows = []
        elif label == "Supplier" or label.startswith("Change "):
            rows.append(f"**{label}** {rest}")
        elif line.startswith("Size ") and len(fields) == 3:  # Size .. mm, volume .. mm3, fill ratio ..
            rows += [f"**Size** {fields[0][5:]}", f"**Volume** {fields[1][7:]}", f"**Fill ratio** {fields[2][11:]}"]
        elif len(fields) == 6:  # class, material, mass, status, program P1, owner Team
            values = fields[:4] + [fields[4].removeprefix("program "), fields[5].removeprefix("owner ")]
            rows += [f"**{k}** {v}" for k, v in zip(("Class", "Material", "Mass", "Status", "Program", "Owner"), values)]
        else:
            rows.append(line)
    return "\n\n".join(b for b in blocks + ["  \n".join(rows)] if b)


def part_tile(part_id, part_no, lines):
    """Iso render, link to the part, then caption lines (markdown)."""
    with st.container(border=True):
        st.image(str(render(part_id)), width="stretch")
        part_link(part_no)
        for line in lines:
            st.markdown(line)


@st.cache_data(max_entries=200, show_spinner=False)
def viewer_glb(glb_path, mtime):
    """The pipeline's glb as a viewer needs it: src.pipeline.mesh exports it with normals but without a material,
    so it would show plain white. Loaded without welding, its vertices stay split per B-rep face: the normals
    stay smooth within a face and sharp across edges. ~3 ms; mtime keys the cache."""
    mesh = trimesh.load(glb_path, force="mesh", process=False)
    mesh.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(  # matte grey: darker than the renders'
        baseColorFactor=[97, 102, 110, 255], metallicFactor=0.0, roughnessFactor=0.6))  # (0.70), brighter light
    return mesh.export(file_type="glb", include_normals=True)


def model_view(glb_path, height=440):
    """Interactive 3D view of a coarse glTF: drag to rotate, scroll to zoom. The mesh goes inline (median 11 KB);
    the <model-viewer> script (@google/model-viewer 4.0.0) comes from src/ui/static, served by Streamlit."""
    data = base64.b64encode(viewer_glb(str(glb_path), Path(glb_path).stat().st_mtime)).decode()
    components.html(
        '<script type="module" src="/app/static/model-viewer.min.js"></script>'
        f'<model-viewer src="data:model/gltf-binary;base64,{data}" camera-controls auto-rotate '
        'tone-mapping="neutral" shadow-intensity="0.6" '
        f'style="width: 100%; height: {height - 16}px; background: #fff"></model-viewer>'
        # Lighting: src/ui/make_studio_hdr.py. model-viewer resolves its URL against this iframe's address,
        # about:srcdoc, so a plain "/app/static/..." is invalid; set it absolute before model-viewer starts.
        '<script>document.querySelector("model-viewer").setAttribute("environment-image",'
        ' new URL("/app/static/studio.hdr", window.parent.location.href).href)</script>',
        height=height)


def three_d_toggle(key):
    """The "3D view" toggle, its value kept under key + "_on". Set into the widget every run: when a page ran
    without the toggle (the empty Part page), Streamlit kept its old True but drew it off in the browser,
    and the 3D view spun next to an "off" toggle. Setting it through session_state syncs the browser too."""
    saved = f"{key}_on"
    st.session_state[key] = st.session_state.get(saved, False)
    return st.toggle(bi("3D 보기", "3D view"), key=key, on_change=lambda: st.session_state.update({saved: st.session_state[key]}))


def render_views(paths, glb_path=None, key=None):
    """The 6 renders of a part: iso large, the other five below it. With a glb_path, a toggle swaps the iso
    render for the 3D view (off by default: WebGL costs the browser 100-300 MB)."""
    if glb_path and Path(glb_path).exists() and three_d_toggle(key):
        model_view(glb_path)
    else:
        st.image(str(paths[ISO]), width=440)
    for col, view in zip(st.columns(5), range(5)):
        col.image(str(paths[view]), caption=VIEW_NAMES[view], width="stretch")


def similar_controls(key):
    """Size filter and count for a similar-parts grid -> (limit, max_size_ratio)."""
    size_col, limit_col, _ = st.columns([1, 1, 3])
    size = size_col.selectbox(bi("크기", "Size"), list(SIZE_FILTERS), key=f"{key}_size",
                              help="이미지 유사도는 크기를 무시합니다: 크기가 비슷한 부품만 남깁니다.  \n"
                                   "Image similarity ignores scale: this keeps parts of comparable size.")
    limit = limit_col.selectbox(bi("표시", "Show"), (5, 10), index=1, key=f"{key}_limit")
    return limit, SIZE_FILTERS[size]


def similar_grid(hits):
    """Tiles for Tools.similar_parts / similar_to hits that carry a part_id (src.ui.data adds it)."""
    for row in range(0, len(hits), 5):
        for col, s in zip(st.columns(5), hits[row:row + 5]):
            with col:
                part_tile(s["part_id"], s["part_no"], [
                    s["name"],
                    f"유사도 **{s['similarity']:.2f}** :gray[:small[similarity]]",
                    f"크기 ×{s['size_ratio']:.2f} :gray[:small[size]]",
                    f"{s['material']} {status_badge(s['status'])}",
                    results_badges(s["test_results"]),
                ])
