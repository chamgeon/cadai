"""Ask page: the step 6 agent, with its tool calls shown as they happen and the cited parts as tiles.

The chat is one conversation: each question goes to the agent with the transcript
so far (st.session_state.convo), so follow-ups can refer to earlier answers.
It is cleared when the user changes, so nothing seen as admin stays on screen or
in the agent's context. An example button hands over a question (and files) in
st.session_state.ask_prefill; it runs at once, in a new conversation.

Files can be attached to a question:
    STEP   the pipeline's own stage functions (extract, mesh, render, DINOv2) in the CAD worker, so its renders
           match the indexed ones pixel for pixel, but nothing is stored: an upload is a question, not an ingest.
           Each solid gets a handle (NEW-1) the agent's similar_parts accepts; the similar parts are shown at once.
    image  sent to Claude, scaled down to MAX_IMAGE_SIDE (no image search: photos are not renders)
    PDF, text   sent to Claude as documents
"""
import asyncio
import hashlib
import io
import re
import tempfile
from pathlib import Path

import anthropic
import streamlit as st
from PIL import Image

from src.config import RAW
from src.qa import agent
from src.ui import data, widgets
from src.ui.widgets import bi, bi_lines

PART_NO = re.compile(r"\b(?!TR-|ECO-)[A-Z]{2}-\d{5}\b")
MAX_TILES = 10
SAMPLE = RAW / "my_bracket.step"  # the bracket modeled in step 2: it is in the index, as BR-66254
SAME_PART = 0.97  # similarity above which an upload is almost surely an existing part (DINOv2: same renders -> 1.0)
SIMILAR_SHOWN = 5
MAX_SHAPES_SHOWN = 4
MAX_IMAGE_SIDE = 1568  # larger images are scaled down by the API anyway, at the same token cost
MAX_IMAGE_BYTES = 5_000_000  # the API's limit per image
STEP_TYPES = ("step", "stp")
IMAGE_TYPES = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp", "gif": "image/gif"}
TEXT_TYPES = ("txt", "md", "csv")
STAGES = ("extract", "mesh", "render", "embed")
EXAMPLES = (
    ("Have parts similar to BR-66254 failed, and how were they fixed?", ()),
    ("Which aluminum parts failed durability tests?", ()),
    ("볼트 구멍에서 피로 균열이 난 알루미늄 브래킷은 몇 개야?", ()),
    ("What happened to BR-10087 in testing?", ()),
)
SAMPLE_QUESTION = "I have a new part with no part number yet. Have parts like it failed tests, and how were they fixed?"


def ask(question, files=()):
    st.session_state.ask_prefill = {"text": question, "files": [(Path(f).name, Path(f).read_bytes()) for f in files]}


def new_chat():
    st.session_state.pop("chat", None)
    st.session_state.pop("convo", None)


def attach(name, content, user, convo):
    """One attached file -> (what the agent gets or None, what the chat shows)."""
    ext = Path(name).suffix.lower().lstrip(".")
    if ext in STEP_TYPES:
        return attach_step(name, content, user, convo)
    if ext in IMAGE_TYPES:
        image = Image.open(io.BytesIO(content))
        image.seek(0)  # an animated gif: its first frame
        media_type = IMAGE_TYPES[ext]
        if max(image.size) > MAX_IMAGE_SIDE or ext == "gif" or len(content) > MAX_IMAGE_BYTES:
            image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            content, media_type = encode(image, "PNG" if image.mode in ("RGBA", "LA", "P") else "JPEG")
            if len(content) > MAX_IMAGE_BYTES:  # a big PNG with alpha: flattened, as JPEG
                content, media_type = encode(image, "JPEG")
        return ({"kind": "image", "name": name, "media_type": media_type, "data": content},
                {"kind": "image", "name": name, "data": content})
    if ext == "pdf":
        return {"kind": "pdf", "name": name, "data": content}, {"kind": "doc", "name": name, "bytes": len(content)}
    if ext in TEXT_TYPES:
        return ({"kind": "text", "name": name, "text": content.decode("utf-8", errors="replace")},
                {"kind": "doc", "name": name, "bytes": len(content)})
    return None, {"kind": "error", "text": f"{name}: " + bi("읽을 수 없는 파일 형식", "this file type can't be read")}


def encode(image, fmt):
    buf = io.BytesIO()
    if fmt == "JPEG":
        image.convert("RGB").save(buf, fmt, quality=90)
    else:
        image.save(buf, fmt)
    return buf.getvalue(), f"image/{fmt.lower()}"


def attach_step(name, content, user, convo):
    """Extract, mesh, render and embed (once per file and conversation), a handle per solid, similar parts."""
    sha = hashlib.sha256(content).hexdigest()
    if sha not in convo["files"]:
        work = Path(tempfile.mkdtemp(prefix="cadai-upload-"))
        (work / name).write_bytes(content)
        with st.status(f"{name} " + bi("읽는 중…", "reading…")) as status:
            res = data.process_upload(work / name, work / "parts")
            seconds = sum(res.get("seconds", {}).values())
            status.update(label=f"{name} " + bi(f"{seconds:.1f}초", f"read in {seconds:.1f} s"),
                          state="complete" if res["ok"] else "error")
        if not res["ok"]:
            return None, {"kind": "error", "text": f"{name}: " + bi("읽기 실패", "could not read") + f" ({res['error']})"}
        shapes = []
        for p in res["parts"]:
            if p["status"] != "ok":
                continue
            handle = f"NEW-{len(convo['shapes']) + 1}"
            convo["shapes"][handle] = {"views": p["views"], "size": p["dims"][0]}
            shapes.append({"handle": handle, "part_id": p["part_id"], "dims": p["dims"], "volume": p["volume"],
                           "n_tris": p["n_tris"], "renders": p["renders"], "glb": p["glb"],
                           "similar": data.similar_upload(user, f"{sha}/{p['part_id']}", p["views"], p["dims"][0],
                                                          SIMILAR_SHOWN)["similar"]})
        convo["files"][sha] = {
            "shapes": shapes, "skipped": [f"{p['part_id']} {p['status']} ({p['error']})"
                                          for p in res["parts"] if p["status"] != "ok"],
            "timing": " · ".join(f"{s} {res['seconds'][s]} s" for s in STAGES if s in res["seconds"])}
    found = convo["files"][sha]
    shown = {"kind": "step", "name": name, **found}
    if not found["shapes"]:
        return None, shown | {"kind": "error", "text": f"{name}: " + bi("검증을 통과한 솔리드 없음",
                                                                         "no solid passed validation")}
    return {"kind": "step", "name": name, "shapes": [
        {"handle": s["handle"], "dims": s["dims"], "volume": s["volume"], "render": s["renders"][widgets.ISO]}
        for s in found["shapes"]]}, shown


def show_attachment(a):
    if a["kind"] == "error":
        st.warning(a["text"])
    elif a["kind"] == "image":
        st.image(a["data"], caption=a["name"], width=320)
    elif a["kind"] == "doc":
        st.markdown(f":material/description: **{a['name']}** · {a['bytes'] / 1024:,.0f} KB")
    else:
        for shape in a["shapes"][:MAX_SHAPES_SHOWN]:
            show_shape(a["name"], shape)
        more = len(a["shapes"]) - MAX_SHAPES_SHOWN
        st.caption(a["timing"] + (" · " + bi(f"솔리드 {more}개 더 (생략)", f"{more} more solids not shown") if more > 0 else "")
                   + (" · " + bi("제외", "skipped") + f": {'; '.join(a['skipped'])}" if a["skipped"] else ""))


def show_shape(name, shape):
    """An attached solid: its iso render, handle and size, the existing parts it looks like, views + 3D on demand."""
    dims = "×".join(f"{d:.0f}" for d in shape["dims"])
    with st.container(border=True):
        left, right = st.columns([1, 4])
        left.image(shape["renders"][widgets.ISO], width="stretch")
        right.markdown(f"**{shape['handle']}** · {name} · {dims} mm · {shape['volume'] / 1000:,.1f} cm³")
        right.caption(bi_lines("부품번호·PLM 기록 없음: 아래 부품은 형상만으로 찾았습니다",
                               "No part number, no PLM record: found by shape alone"))
        hits = shape["similar"]
        if hits and hits[0]["similarity"] >= SAME_PART:
            right.success(bi_lines(f"기존 부품으로 보입니다: **{hits[0]['part_no']}** ({hits[0]['name']}, 유사도 "
                                   f"{hits[0]['similarity']:.2f}, 크기 ×{hits[0]['size_ratio']:.2f}). "
                                   "새 번호를 받는 대신 재사용할까요?",
                                   "Looks like an existing part: reuse it rather than release a new number?"))
        if hits:
            widgets.similar_grid(hits)
        else:
            st.caption(bi("열람 가능한 부품 중 유사 부품 없음", "no similar parts among the parts this user may see"))
        with st.expander(bi("모든 뷰", "All views")):
            widgets.render_views(shape["renders"], shape["glb"], key=f"view3d_{shape['handle']}")


def describe(tool, args):
    """One line per tool call, in words an engineer reads."""
    if tool == "lookup_part":
        return f"**{args['part_no']}** " + bi("조회", "looked up")
    if tool == "similar_parts":
        ratio = args.get("max_size_ratio")
        return (f"**{args['part_no']}** " + bi("와 비슷한 형상 검색", "parts shaped like it")
                + (f" · {bi('크기', 'size')} ×{ratio}" if ratio else ""))
    if tool == "search_notes":
        line = bi("노트 검색", "searched notes") + f" “{args['query']}”"
        if args.get("keywords"):
            line += f" · {bi('키워드', 'keywords')} {', '.join(args['keywords'])}"
        filters = {k: v for k, v in args.items() if k not in ("query", "keywords", "limit") and v}
        return line + "".join(f" · {k}={v}" for k, v in filters.items())
    if tool == "query_parts":
        filters = {k: v for k, v in args.items() if k != "aggregate" and v not in (None, "any")}
        return bi("부품 테이블 조회", "queried the parts table") + f" ({args.get('aggregate', 'list')})" + "".join(
            f" · {k}={v}" for k, v in filters.items())
    return f"{tool}({args})"


def plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def took(seconds):
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.1f} s"


def show_answer(out, user):
    if "error" in out:
        st.error(out["error"])
        return
    st.markdown(out["answer"])
    if out["unverified_ids"]:
        ids = ", ".join(out["unverified_ids"])
        st.warning(bi_lines(f"어떤 기록에도 없는 번호: {ids}. 사용 전에 확인하세요.",
                            f"Not found in any record the tools returned: {ids}. Check them before relying on them."))
    cited = []
    for part_no in dict.fromkeys(PART_NO.findall(out["answer"])):  # first mention order, no repeats
        p = data.part(user, part_no)
        if p is not None:
            cited.append(p)
    if cited:
        st.caption(bi(f"인용된 부품 {len(cited)}개", f"parts cited ({len(cited)})")
                   + (" · " + bi(f"처음 {MAX_TILES}개 표시", f"first {MAX_TILES} shown") if len(cited) > MAX_TILES else ""))
        cited = cited[:MAX_TILES]
        for row in range(0, len(cited), 5):
            for col, p in zip(st.columns(5), cited[row:row + 5]):
                with col:
                    widgets.part_tile(p["card"]["part_id"], p["card"]["part_no"], [
                        p["latest"]["name"],
                        f"{p['latest']['material']} {widgets.status_badge(p['latest']['status'])}",
                    ])
    u = out["usage"]
    tokens, n_calls = u.get("output_tokens", 0), len(out["trace"])
    st.caption(bi(f"{out['seconds']}초 · 도구 라운드 {out['rounds']} · 도구 호출 {n_calls} · 출력 토큰 {tokens:,}",
                  f"{plural(out['rounds'], 'tool round')} · {plural(n_calls, 'tool call')} · output tokens")
               + (" · " + bi("번호 사후 수정됨", "ids corrected by the post-check") if out["repaired"] else ""))


def run(question, user, convo, attachments):
    status = st.status(bi("생각 중…", "Thinking…"), expanded=True)  # neutral: a greeting is answered without reading records
    done = []

    def on_event(kind, info):
        if kind == "request" and info["kind"] == "id_repair":
            status.update(label=bi("기록에 없는 번호 수정 중…", "correcting ids the records don't contain…"))
        elif kind == "request" and done:
            status.update(label=bi(f"결과 읽는 중 (도구 호출 {len(done)})…",
                                   f"reading the results ({plural(len(done), 'tool call')} so far)…"))
        elif kind == "tool":
            status.update(label=f"{info['tool']} " + bi("실행 중…", "running…"))
        elif kind == "result":
            done.append(info)
            status.markdown(f"{describe(info['tool'], info['input'])}"
                            + (" · :red[error]" if info["is_error"] else f" · {took(info['seconds'])}"))

    try:
        out = asyncio.run(agent.ask_async(question, user, anthropic.AsyncAnthropic(max_retries=3),
                                          on_event=on_event, history=convo["messages"], seen=convo["seen"],
                                          attachments=attachments, shapes=convo["shapes"]))
    except anthropic.APIError as e:
        status.update(label=bi("실패", "failed"), state="error")
        return {"error": bi("모델 API 실패", "the model API failed") + f": {type(e).__name__}: {e}"}
    n_calls = len(out["trace"])
    status.update(label=bi(f"{out['seconds']}초 만에 답변" + (f" · 도구 호출 {n_calls}" if n_calls else ""),
                           f"answered in {out['seconds']} s"
                           + (f" · {plural(n_calls, 'tool call')}" if n_calls else "")),
                  state="complete", expanded=False)
    convo["messages"], convo["seen"] = out.pop("messages"), out.pop("seen")
    return out


user = st.session_state.user
prefill = st.session_state.pop("ask_prefill", None)
if prefill:  # a handoff or an example starts a new topic
    new_chat()
chat = st.session_state.setdefault("chat", [])
convo = st.session_state.setdefault("convo", {"messages": [], "seen": set(), "shapes": {}, "files": {}})
left, right = st.columns([4, 1], vertical_alignment="center")
left.caption(bi_lines("열람 가능한 기록으로만 답하고 모든 사실에 출처를 답니다. 후속 질문, STEP·이미지·문서 첨부 가능.",
                      "Answers only from records you may see, every fact cited. Follow-ups and attachments work."))
if chat:
    right.button(bi("새 대화", "New chat"), icon=":material/add_comment:", on_click=new_chat, width="stretch")

for msg in chat:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            for a in msg.get("attachments", ()):
                show_attachment(a)
            st.markdown(msg["content"])
        else:
            if msg["out"].get("trace"):
                n_calls = len(msg["out"]["trace"])
                with st.expander(bi(f"도구 호출 {n_calls}", plural(n_calls, "tool call"))):
                    for t in msg["out"]["trace"]:
                        st.markdown(f"{describe(t['tool'], t['input'])} · {took(t['seconds'])}")
            show_answer(msg["out"], user)

typed = st.chat_input("Ask about parts, tests and changes, or attach a file",  # always drawn, or it vanishes
                      accept_file="multiple", max_upload_size=30,                  # during a prefill
                      file_type=[*STEP_TYPES, *IMAGE_TYPES, "pdf", *TEXT_TYPES])
if prefill:
    question, files = prefill["text"], prefill["files"]
elif typed:
    question, files = typed.text.strip(), [(Path(f.name).name, f.getvalue()) for f in typed.files]
else:
    question, files = "", []

if not chat and not question and not files:  # not while answering: a click would stop the run and lose the answer
    st.caption(bi("예시", "Try"))
    for col, (example, example_files) in zip(st.columns(2), EXAMPLES[:2]):
        col.button(example, on_click=ask, args=(example, example_files), width="stretch")
    for col, (example, example_files) in zip(st.columns(2), EXAMPLES[2:]):
        col.button(example, on_click=ask, args=(example, example_files), width="stretch")
    st.button(bi(f"{SAMPLE.name} 첨부해서 질문하기", f"attach {SAMPLE.name} and ask"), icon=":material/attach_file:",
              on_click=ask, args=(SAMPLE_QUESTION, (SAMPLE,)), width="stretch")

if question or files:
    with st.chat_message("user"):
        attachments, shown = [], []
        for name, content in files:
            sent, view = attach(name, content, user, convo)
            show_attachment(view)
            shown.append(view)
            if sent:
                attachments.append(sent)
        if not question:
            question = (SAMPLE_QUESTION if any(a["kind"] == "step" for a in attachments)
                        else "What does this show?" if attachments else "")
        st.markdown(question)
    widgets.scroll_to_last_message(len(chat))
    out = None
    if question:  # nothing to ask when every attachment failed and no text came with them
        with st.chat_message("assistant"):
            out = run(question, user, convo, attachments)
            show_answer(out, user)
    chat.append({"role": "user", "content": question, "attachments": shown})
    if out is not None:
        chat.append({"role": "assistant", "out": out})
    # Redraw in the history layout now. Otherwise this run's elements (live status, answer) stay on the page,
    # and during the next question's run Streamlit shows those it hasn't overwritten yet, faded: the previous
    # answer appeared under the new question for the 10-20 s the agent takes.
    st.rerun()
