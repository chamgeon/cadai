"""Ask page: the step 6 agent, with its tool calls shown as they happen and the cited parts as tiles.

Each question is answered on its own (the agent keeps no conversation); the chat
is cleared when the user changes, so nothing seen as admin stays on screen.
Another page can hand over a question in st.session_state.ask_prefill; it runs at once.
"""
import asyncio
import re

import anthropic
import streamlit as st

from src.qa import agent
from src.ui import data, widgets

PART_NO = re.compile(r"\b(?!TR-|ECO-)[A-Z]{2}-\d{5}\b")
MAX_TILES = 10
EXAMPLES = (
    "Have parts similar to BR-66254 failed, and how were they fixed?",
    "Which aluminum parts failed durability tests?",
    "볼트 구멍에서 피로 균열이 난 알루미늄 브래킷은 몇 개야?",
    "What happened to BR-10087 in testing?",
)


def ask(question):
    st.session_state.ask_prefill = question


def describe(tool, args):
    """One line per tool call, in words an engineer reads."""
    if tool == "lookup_part":
        return f"Looked up **{args['part_no']}**"
    if tool == "similar_parts":
        ratio = args.get("max_size_ratio")
        return f"Found parts shaped like **{args['part_no']}**" + (f", size within ×{ratio}" if ratio else "")
    if tool == "search_notes":
        line = f"Searched notes for “{args['query']}”"
        if args.get("keywords"):
            line += f" · keywords {', '.join(args['keywords'])}"
        filters = {k: v for k, v in args.items() if k not in ("query", "keywords", "limit") and v}
        return line + "".join(f" · {k}={v}" for k, v in filters.items())
    if tool == "query_parts":
        filters = {k: v for k, v in args.items() if k != "aggregate" and v not in (None, "any")}
        return f"Queried the parts table ({args.get('aggregate', 'list')})" + "".join(
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
        st.warning(f"Not found in any record the tools returned: {', '.join(out['unverified_ids'])}. "
                   "Check these before relying on them.")
    cited = []
    for part_no in dict.fromkeys(PART_NO.findall(out["answer"])):  # first mention order, no repeats
        p = data.part(user, part_no)
        if p is not None:
            cited.append(p)
    if cited:
        st.caption(f"Parts cited ({len(cited)})" + (f", first {MAX_TILES} shown" if len(cited) > MAX_TILES else ""))
        cited = cited[:MAX_TILES]
        for row in range(0, len(cited), 5):
            for col, p in zip(st.columns(5), cited[row:row + 5]):
                with col:
                    widgets.part_tile(p["card"]["part_id"], p["card"]["part_no"], [
                        p["latest"]["name"],
                        f"{p['latest']['material']} {widgets.status_badge(p['latest']['status'])}",
                    ])
    u = out["usage"]
    st.caption(f"{out['seconds']} s · {plural(out['rounds'], 'tool round')} · "
               f"{plural(len(out['trace']), 'tool call')} · {u.get('output_tokens', 0):,} output tokens"
               + (" · ids corrected by the post-check" if out["repaired"] else ""))


def run(question, user):
    status = st.status("Claude is planning which records to read…", expanded=True)
    done = []

    def on_event(kind, info):
        if kind == "request" and info["kind"] == "id_repair":
            status.update(label="Correcting ids the records don't contain…")
        elif kind == "request" and done:
            status.update(label=f"Claude is reading the results ({plural(len(done), 'tool call')} so far)…")
        elif kind == "tool":
            status.update(label=f"Running {info['tool']}…")
        elif kind == "result":
            done.append(info)
            status.markdown(f"{describe(info['tool'], info['input'])}"
                            + (" · :red[error]" if info["is_error"] else f" · {took(info['seconds'])}"))

    try:
        out = asyncio.run(agent.ask_async(question, user, anthropic.AsyncAnthropic(max_retries=3),
                                          on_event=on_event))
    except anthropic.APIError as e:
        status.update(label="Failed", state="error")
        return {"error": f"The model API failed: {type(e).__name__}: {e}"}
    status.update(label=f"Answered in {out['seconds']} s · {plural(len(out['trace']), 'tool call')}",
                  state="complete", expanded=False)
    return out


user = st.session_state.user
chat = st.session_state.setdefault("chat", [])
st.caption("Answers come only from records this user may see, with every fact cited. "
           "Each question is answered on its own.")

for msg in chat:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            if msg["out"].get("trace"):
                with st.expander(plural(len(msg["out"]["trace"]), "tool call")):
                    for t in msg["out"]["trace"]:
                        st.markdown(f"{describe(t['tool'], t['input'])} · {took(t['seconds'])}")
            show_answer(msg["out"], user)

typed = st.chat_input("Ask about parts, tests and changes…")  # always drawn, or it vanishes during a prefill
question = st.session_state.pop("ask_prefill", None) or typed

if not chat and not question:  # not while answering: a click would stop the run and lose the answer
    st.caption("Try:")
    for col, example in zip(st.columns(2), EXAMPLES[:2]):
        col.button(example, on_click=ask, args=(example,), width="stretch")
    for col, example in zip(st.columns(2), EXAMPLES[2:]):
        col.button(example, on_click=ask, args=(example,), width="stretch")

if question:
    with st.chat_message("user"):
        st.markdown(question)
    widgets.scroll_to_last_message(len(chat))
    with st.chat_message("assistant"):
        out = run(question, user)
        show_answer(out, user)
    chat += [{"role": "user", "content": question}, {"role": "assistant", "out": out}]
    # Redraw in the history layout now. Otherwise this run's elements (live status, answer) stay on the page,
    # and during the next question's run Streamlit shows those it hasn't overwritten yet, faded: the previous
    # answer appeared under the new question for the 10-20 s the agent takes.
    st.rerun()
