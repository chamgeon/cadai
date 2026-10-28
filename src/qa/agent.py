"""Step 6: the Q&A agent: Claude plans, calls src.qa.tools, answers with citations.

A bounded tool loop: at most MAX_ROUNDS rounds of tool calls, then one last
request with tools switched off, so every question ends in an answer. Claude
decides the route itself (lookup, structured query, text search, similar
parts, or a chain of them); the tools enforce permissions, so the system
prompt doesn't even know who is asking and stays byte-identical, which keeps
it in the prompt cache.

After the answer, every part number and report id in it is checked against
the tool results; an unknown id triggers one rewrite with tools off, and any
that survive are returned as `unverified_ids`.

A question can carry attachments (see _content): solids of a STEP file, which
get handles (NEW-1) that similar_parts accepts, images and documents.

A conversation continues by passing the previous result's `messages` and `seen`
back in: the whole transcript, tool results included, so a follow-up ("which of
those are steel?") can reuse what was already read. The request is cached
automatically, so earlier turns are read from the cache. Once the old tool
results grow past HISTORY_CHARS, those before the last KEEP_TURNS questions
are replaced by a stub.

Every call is traced (tool, input, size of result, seconds; each LLM request's
seconds and output tokens) for step 8: routing can be scored separately from the answer.
ask_many() runs a whole question set concurrently for evaluation.

Requests use Claude Opus 5.5 at effort medium, with server-side refusal fallback
(`fallbacks="default"`): a safety-classifier decline is retried on another
model inside the same call instead of coming back as a refusal.

Usage:
    python -m src.qa.agent "Has a part like BR-66254 ever failed, and how was it fixed?" [--user normal]
"""
import argparse
import asyncio
import base64
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic

from src.qa import textindex
from src.qa.plm import USERS
from src.qa.tools import GROUP_BY, Tools

MODEL = "claude-opus-5-5"
EFFORT = "medium"
MAX_ROUNDS = 4
MAX_TOKENS = 16000
HISTORY_CHARS = 200_000  # tool result text kept from earlier questions, ~60k tokens
KEEP_TURNS = 2  # questions whose tool results survive compaction
OMITTED = "[earlier result omitted; call the tool again if it's needed]"
# part numbers, test report and change-notice ids, as src.qa.plm writes them
ID = re.compile(r"\b(?:TR|ECO)-\d{6}\b|\b[A-Z]{2}-\d{5}\b")
REPAIR = ("Automatic check: these ids in your answer appear in no tool result: {ids}. Rewrite the whole answer using "
          "only ids the tools returned, correcting or dropping the others. Don't mention this check.")

SYSTEM = """You answer questions from vehicle R&D engineers about parts, using a parts database (PLM items, ERP supplier data, test reports, change notices) and a shape-similarity index built from the parts' CAD geometry. Notes are in Korean or English.

The tools only return what this engineer is allowed to see. If the tools find nothing, say the records don't show it; never guess a part number, value or cause.

Searching well:
- Put hard facts into tool filters, not only into the query text: result (fail, conditional, pass), test_type, material, part_class. Text search ignores negation: "no crack" looks like "crack".
- search_notes: give `keywords` with 2-4 synonyms per key term in both English and Korean (crack: crack, fracture, 균열, 크랙, 파단) and the material code when a material is named.
- Counting or listing by fields: query_parts. A named part: lookup_part. Parts that look like a given part: similar_parts, then read their history with search_notes (part_nos) or lookup_part.
- Materials: AL6061-T6 (aluminum), ADC12 (die-cast aluminum), SPCC and SPFH590 (steel sheet), S45C (carbon steel), SUS304 (stainless), PA66-GF30, PP-T20, ABS (plastics).
- Test types: fatigue (durability), salt_spray, heat_aging, cold_impact, vibration, wear, static, dimensional, nvh.
- You have at most 4 rounds of tool calls. Make independent calls in parallel within one round.
- Attached STEP files: each solid has a handle (NEW-1, NEW-2…), no part number and no records. To find existing parts that look like it, call similar_parts with the handle. Its render is attached so you can see it.
- Attached images and documents come from the engineer, not the database: use them as context, but cite only database ids as evidence.
- Follow-up questions may refer to earlier answers. Reuse earlier tool results when they already answer it; otherwise call the tools again.

Answering:
- Answer in the language of the question. Lead with the answer, then the evidence; keep it short.
- Cite every fact with the part number and the report or change-notice id in brackets, e.g. BR-19198 [TR-381830]. Cite only ids a tool returned, copied exactly.
- When many parts match, give the count, group them (by class, status or revision), name at most 10 as examples, and offer the full list.
- Say which revision a result belongs to when it matters: a rev A failure fixed in rev B is not an open problem."""


def _opt(kind, **extra):
    """A nullable property: strict tool use needs every property listed in `required`."""
    return {"anyOf": [{"type": kind, **extra}, {"type": "null"}]}


RESULTS = ["pass", "conditional", "fail"]
TEST_TYPES = ["fatigue", "salt_spray", "heat_aging", "cold_impact", "vibration", "wear", "static", "dimensional", "nvh"]
CLASSES = ["Bracket", "Plate", "Block", "Beam", "Body", "Flange", "Cover", "Shaft", "Washer", "Housing", "Pin",
           "Bolt", "Nut", "Gear"]
MATERIALS = ["AL6061-T6", "ADC12", "SPCC", "SPFH590", "S45C", "SUS304", "PA66-GF30", "PP-T20", "ABS"]


def _tool(name, description, properties):
    return {"name": name, "description": description, "strict": True,
            "input_schema": {"type": "object", "properties": properties, "required": list(properties),
                             "additionalProperties": False}}


TOOLS = [
    _tool("lookup_part", "One part's full record: card (class, material, mass, status, size, supplier, revisions), "
          "every test report note and every change notice.",
          {"part_no": {"type": "string", "description": "e.g. BR-19198"}}),
    _tool("query_parts", "Count, list or group parts (latest revision) by fields. Size is the longest bounding-box "
          "side in mm, measured from the CAD geometry. test_type/result match parts having such a test report.",
          {"aggregate": {"type": "string", "enum": ["count", "list", "group"]},
           "group_by": _opt("string", enum=list(GROUP_BY)),
           "part_class": _opt("string", enum=CLASSES), "material": _opt("string", enum=MATERIALS),
           "status": _opt("string", enum=["In Work", "Released"]),
           "test_type": _opt("string", enum=TEST_TYPES), "result": _opt("string", enum=RESULTS),
           "test_revision": {"type": "string", "enum": ["any", "latest"],
                             "description": "latest: only tests on the part's current revision (open issues)"},
           "min_size_mm": _opt("number"), "max_size_mm": _opt("number")}),
    _tool("search_notes", "Hybrid keyword + semantic search over part cards, test report notes and change notices "
          "(Korean and English). Filters are exact.",
          {"query": {"type": "string"},
           "keywords": {"type": "array", "items": {"type": "string"},
                        "description": "synonyms in English and Korean for the key terms"},
           "kind": _opt("string", enum=["card", "test", "eco"]),
           "result": _opt("string", enum=RESULTS), "test_type": _opt("string", enum=TEST_TYPES),
           "material": _opt("string", enum=MATERIALS), "part_class": _opt("string", enum=CLASSES),
           "part_nos": _opt("array", items={"type": "string"}),
           "limit": {"type": "integer", "description": "1-20"}}),
    _tool("similar_parts", "Parts whose shape looks like the given part (6-view image similarity from CAD renders), "
          "with each one's test result counts. Similarity ignores size; max_size_ratio keeps parts within that "
          "factor of the query's size.",
          {"part_no": {"type": "string", "description": "a part number, or an attached shape's handle (NEW-1)"},
           "limit": {"type": "integer", "description": "1-20"},
           "max_size_ratio": _opt("number")}),
]


# Tools share one SQLite connection per user, BGE-M3 on the GPU and Kiwi: run them one at a time on one
# thread. They take well under a second; the LLM calls are what runs concurrently.
_tool_thread = ThreadPoolExecutor(max_workers=1)


MAX_SHAPE_RENDERS = 4  # iso renders sent per question; a file can hold dozens of solids


def _content(question, attachments):
    """The question as content blocks: one text block per STEP file (its solids' handles and sizes) plus their iso
    renders, then images and documents, then the question. attachments: dicts with "kind":
        step   {"name", "shapes": [{"handle", "dims", "volume", "render"}]}
        image  {"name", "media_type", "data"}      pdf {"name", "data"}      text {"name", "text"}
    """
    if not attachments:
        return question
    blocks, renders = [], 0
    for a in attachments:
        if a["kind"] == "step":
            lines = [f"{s['handle']}: {'×'.join(f'{d:.0f}' for d in s['dims'])} mm, volume {s['volume'] / 1000:,.1f} cm³"
                     for s in a["shapes"]]
            blocks.append({"type": "text", "text": f"Attached STEP file {a['name']}, {len(lines)} solid(s):\n"
                                                   + "\n".join(lines)})
            for shape in a["shapes"]:
                if renders < MAX_SHAPE_RENDERS:
                    renders += 1
                    blocks.append({"type": "text", "text": f"{shape['handle']} (iso view):"})
                    blocks.append(_image("image/png", open(shape["render"], "rb").read()))
        elif a["kind"] == "image":
            blocks += [{"type": "text", "text": f"Attached image {a['name']}:"}, _image(a["media_type"], a["data"])]
        elif a["kind"] == "pdf":
            blocks.append({"type": "document", "title": a["name"], "source": {
                "type": "base64", "media_type": "application/pdf", "data": base64.b64encode(a["data"]).decode()}})
        else:
            blocks.append({"type": "document", "title": a["name"],
                           "source": {"type": "text", "media_type": "text/plain", "data": a["text"]}})
    return blocks + [{"type": "text", "text": question}]


def _image(media_type, data):
    return {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                        "data": base64.b64encode(data).decode()}}


def _is_question(message):
    return message["role"] == "user" and (isinstance(message["content"], str) or not any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in message["content"]))


def _compact(history):
    """Stub out tool results before the last KEEP_TURNS questions once they pass HISTORY_CHARS.

    Attachments are kept: the engineer's own files, and the question may still be about them.
    Changes the cached prefix, so it runs rarely: only past the budget.
    """
    results = [(i, j) for i, m in enumerate(history) if isinstance(m["content"], list)
               for j, b in enumerate(m["content"]) if isinstance(b, dict) and b.get("type") == "tool_result"]
    if sum(len(history[i]["content"][j]["content"]) for i, j in results) <= HISTORY_CHARS:
        return history
    questions = [i for i, m in enumerate(history) if _is_question(m)]
    cut = questions[-KEEP_TURNS] if len(questions) >= KEEP_TURNS else 0
    history = list(history)
    for i in sorted({i for i, _ in results if i < cut}):
        history[i] = {"role": "user", "content": [
            {**b, "content": OMITTED} if b.get("type") == "tool_result" else b for b in history[i]["content"]]}
    return history


async def ask_async(question, user="normal", client=None, verbose=False, on_event=None, history=None, seen=(),
                    attachments=(), shapes=None):
    """-> {"answer", "trace", "calls", "rounds", "usage", "stop_reason", "seconds", "repaired", "unverified_ids",
           "messages", "seen"}.

    trace: one entry per tool call; calls: one per LLM request (seconds, output tokens).
    history, seen: the previous result's "messages" and "seen" to continue that conversation (same user only:
        the transcript holds what that user's tools returned). A refused question leaves them unchanged.
    attachments: files sent with this question (_content). shapes: every attached solid of the conversation so
        far, handle -> {"views", "size"}, for similar_parts (src.qa.tools.Tools).
    on_event(kind, info): progress for a UI, called on the caller's event loop as it happens:
        "request" {round, kind}               an LLM request starts (kind: plan | id_repair)
        "tool"    {round, tool, input}        a tool call starts
        "result"  {...the trace entry}        it finished
    """
    emit = on_event or (lambda kind, info: None)
    client = client or anthropic.AsyncAnthropic()
    loop = asyncio.get_running_loop()
    tools = await loop.run_in_executor(_tool_thread, Tools, user, shapes)
    history = _compact(history or [])
    messages = history + [{"role": "user", "content": _content(question, attachments)}]
    trace, calls, usage = [], [], {}
    seen = set(seen) | set(ID.findall(question))  # ids the answer may cite: questions' and every tool result's
    for a in attachments:  # and those in an attached text file: the engineer's own (a PDF's text isn't read here)
        if a["kind"] == "text":
            seen.update(ID.findall(a["text"]))
    start = time.perf_counter()

    async def request(tools_off, kind):
        emit("request", {"round": len(calls) + 1, "kind": kind})
        t = time.perf_counter()
        response = await client.beta.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM, tools=TOOLS, messages=messages,
            tool_choice={"type": "none"} if tools_off else {"type": "auto"},
            output_config={"effort": EFFORT},
            cache_control={"type": "ephemeral"},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
        calls.append({"round": len(calls) + 1, "kind": kind, "seconds": round(time.perf_counter() - t, 1),
                      "output_tokens": response.usage.output_tokens,
                      "text_chars": sum(len(b.text) for b in response.content if b.type == "text")})
        for k, v in response.usage.model_dump().items():
            if isinstance(v, int):
                usage[k] = usage.get(k, 0) + v
        if response.stop_reason != "refusal":
            messages.append({"role": "assistant", "content": response.content})  # thinking blocks included, unedited
        return response

    def result(answer, response, messages=messages, **extra):
        return {"answer": answer, "trace": trace, "calls": calls, "rounds": len({c["round"] for c in trace}),
                "usage": usage, "stop_reason": response.stop_reason,
                "seconds": round(time.perf_counter() - start, 1), "messages": messages, "seen": seen, **extra}

    for round_ in range(MAX_ROUNDS + 1):
        response = await request(tools_off=round_ == MAX_ROUNDS, kind="plan")  # out of rounds: answer now
        if response.stop_reason == "refusal":
            return result("The request was declined.", response, history, repaired=False, unverified_ids=[])
        uses = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not uses:
            break
        results = []
        for call in uses:
            emit("tool", {"round": round_ + 1, "tool": call.name, "input": call.input})
            t = time.perf_counter()
            text, is_error = await loop.run_in_executor(_tool_thread, tools.run, call.name, call.input)
            seen.update(ID.findall(text))
            trace.append({"round": round_ + 1, "tool": call.name, "input": call.input, "is_error": is_error,
                          "result_chars": len(text), "seconds": round(time.perf_counter() - t, 2)})
            emit("result", trace[-1])
            if verbose:
                print(f"  [{round_ + 1}] {call.name}({json.dumps(call.input, ensure_ascii=False)})"
                      f"{'  ERROR ' + text if is_error else ''}")
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": text,
                            **({"is_error": True} if is_error else {})})
        messages.append({"role": "user", "content": results})  # all results of a round in one message
    answer = "".join(b.text for b in response.content if b.type == "text")
    if uses:  # cut off mid tool call (max_tokens): a tool_use without its result would break the next turn
        messages[-1] = {"role": "assistant", "content": [b for b in response.content if b.type != "tool_use"]
                        + ([] if answer else [{"type": "text", "text": "(answer cut off)"}])}

    # Id post-check: a part number or report id no tool returned is a copying slip or an invention.
    unverified = sorted(set(ID.findall(answer)) - seen)
    if not unverified:
        return result(answer, response, repaired=False, unverified_ids=[])
    if verbose:
        print(f"  [check] ids not in any tool result: {', '.join(unverified)}; asking for a corrected answer")
    messages.append({"role": "user", "content": REPAIR.format(ids=", ".join(unverified))})
    fixed = await request(tools_off=True, kind="id_repair")
    if fixed.stop_reason == "refusal":
        return result(answer, response, messages[:-1], repaired=False, unverified_ids=unverified)
    answer = "".join(b.text for b in fixed.content if b.type == "text")
    # The transcript keeps the corrected answer in place of the wrong one and the check message. Its thinking
    # was about the check, so it is left out (thinking from finished turns may be omitted).
    kept = messages[:-3] + [{"role": "assistant", "content": [b for b in fixed.content
                                                            if b.type not in ("thinking", "redacted_thinking")]}]
    return result(answer, fixed, kept, repaired=True, unverified_ids=sorted(set(ID.findall(answer)) - seen))


async def ask_many(items, concurrency=8):
    """items: [(question, user)] -> results in the same order; a failed question carries "error" instead."""
    client = anthropic.AsyncAnthropic(max_retries=6)  # backs off on 429 / 529 by itself
    gate = asyncio.Semaphore(concurrency)
    await asyncio.get_running_loop().run_in_executor(_tool_thread, textindex.load_model)

    async def one(question, user):
        async with gate:
            try:
                return await ask_async(question, user, client)
            except anthropic.APIError as e:
                return {"error": f"{type(e).__name__}: {e}"}

    return await asyncio.gather(*(one(q, u) for q, u in items))


def ask(question, user="normal", verbose=False):
    return asyncio.run(ask_async(question, user, verbose=verbose))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--user", choices=list(USERS), default="normal")
    args = parser.parse_args()
    print(f"[{args.user}: programs {', '.join(USERS[args.user])}]  {args.question}\n")
    out = ask(args.question, args.user, verbose=True)
    u = out["usage"]
    if out["unverified_ids"]:
        print(f"\nWARNING: ids not found in any tool result: {', '.join(out['unverified_ids'])}")
    print(f"\n{out['answer']}\n\n-- {out['rounds']} tool rounds, {len(out['trace'])} calls, {out['seconds']}s, "
          f"tokens in {u.get('input_tokens', 0)} (cache read {u.get('cache_read_input_tokens', 0)}, "
          f"write {u.get('cache_creation_input_tokens', 0)}) out {u.get('output_tokens', 0)}, stop {out['stop_reason']}")


if __name__ == "__main__":
    main()
