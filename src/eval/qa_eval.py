"""Step 8C: run the Q&A gold set (src.eval.qa_gold) through the agent and score it by code.

    run     every question through src.qa.agent.ask_async, per setup (model x effort), 4 at a
            time; each answer, its tool calls and tool results, cost and time go to
            step8_qa_runs/{setup}.jsonl. A rerun skips what is done; no new question starts
            once the spend in all runs reaches --budget minus a reserve.
    score   checks every answer against facts.json, the "why" answers also by an LLM judge
            (Claude Sonnet 5.5, effort low; cached in step8_qa_runs/judge.jsonl) -> step8_qa.json

Scored separately, so a wrong answer says where it went wrong:
    routing     an expected tool was called (smalltalk: none)
    retrieval   the gold parts were in what the tools returned
    answer      the gold number / parts / fields are in the answer (see qa_gold's table)
    leaks       an id from a program the user can't see, in the answer   (must be 0)
    invented    an id in the answer that exists nowhere and wasn't in the question
    repaired    the agent's id post-check had to rewrite the answer

The hard set (--set hard, src.eval.qa_gold build_hard) has its own runs and results; the
budget counts the spend of both sets. Two checks only it uses: rates (each a/b given, or
its percentage) and conclusion (an LLM judge answers the question's criterion).

Usage:
    python -m src.eval.qa_eval run [--set hard] [--setups opus_medium,...] [--budget 15] [--ids count-01,...]
    python -m src.eval.qa_eval score [--set hard]
"""
import argparse
import asyncio
import json
import re
import time

import anthropic
import numpy as np

from src.config import EVAL_RESULTS as RESULTS
from src.eval.common import cost
from src.eval.qa_gold import GOLD, HARD_GOLD, Facts
from src.qa import agent, textindex
from src.qa.plm import USERS

SETS = {"main": (GOLD, RESULTS / "step8_qa_runs", RESULTS / "step8_qa.json"),
        "hard": (HARD_GOLD, RESULTS / "step8_qa_hard_runs", RESULTS / "step8_qa_hard.json")}
GOLD_PATH, RUNS, OUT = SETS["main"]  # main() switches them for --set hard
SETUPS = {"opus_medium": ("claude-opus-5-5", "medium"),  # the app's setting until step 8
          "opus_low": ("claude-opus-5-5", "low"),  # the app's setting since
          "sonnet_medium": ("claude-sonnet-5-5", "medium")}
JUDGE = ("claude-sonnet-5-5", "low")
RESERVE = 0.6  # $ kept back: ~4 questions in flight, plus the judge
CONCURRENCY = 4
TOOL_TEXT = 30_000  # chars of tool results kept per turn in the run file
PART = re.compile(r"\b[A-Z]{2}-\d{5}\b")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}
RESULT_WORDS = {"pass": ("pass", "합격", "통과"), "conditional": ("conditional", "조건부"),
                "fail": ("fail", "불합격")}


# ------------------------------------------------------------------ run

def spent():
    """Every run of both sets, judge included: the budget is one."""
    return sum(r.get("usd", 0) for _, runs, _ in SETS.values() for path in runs.glob("*.jsonl")
               for r in map(json.loads, path.open()))


async def one_turn(client, model, effort, question, user, history=None, seen=()):
    start = len(history or [])
    out = await agent.ask_async(question, user, client, history=history, seen=seen, model=model, effort=effort)
    tool_text = "\n".join(b["content"] for m in out["messages"][start:] if m["role"] == "user"
                          and isinstance(m["content"], list) for b in m["content"]
                          if isinstance(b, dict) and b.get("type") == "tool_result")
    turn = {"question": question, "answer": out["answer"], "trace": out["trace"], "calls": out["calls"],
            "usage": out["usage"], "usd": cost(out["usage"], model), "seconds": out["seconds"],
            "repaired": out["repaired"], "unverified_ids": out["unverified_ids"], "stop_reason": out["stop_reason"],
            "tool_ids": sorted(set(agent.ID.findall(tool_text))), "tool_text": tool_text[:TOOL_TEXT]}
    return turn, out


async def run_setup(name, gold, budget):
    model, effort = SETUPS[name]
    path = RUNS / f"{name}.jsonl"
    done = {json.loads(line)["id"] for line in path.open()} if path.exists() else set()
    todo = [q for q in gold if q["id"] not in done]
    print(f"{time.strftime('%H:%M:%S')} {name}: {len(done)} done, {len(todo)} to run  (spent so far ${spent():.2f})")
    client = anthropic.AsyncAnthropic(max_retries=6)
    gate, lock = asyncio.Semaphore(CONCURRENCY), asyncio.Lock()
    state = {"spent": spent(), "stopped": False}

    async def item(q):
        async with gate:
            if state["spent"] >= budget - RESERVE:
                state["stopped"] = True
                return
            rec = {"id": q["id"], "setup": name, "turns": []}
            try:
                history, seen = None, ()
                for question in [t["question"] for t in q["turns"]] if "turns" in q else [q["question"]]:
                    turn, out = await one_turn(client, model, effort, question, q["user"], history, seen)
                    rec["turns"].append(turn)
                    history, seen = out["messages"], out["seen"]
            except anthropic.APIError as e:
                rec["error"] = f"{type(e).__name__}: {e}"
            rec["usd"] = sum(t["usd"] for t in rec["turns"])
            async with lock:
                state["spent"] += rec["usd"]
                with path.open("a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"  {time.strftime('%H:%M:%S')} {name} {q['id']:14} {sum(t['seconds'] for t in rec['turns']):5.1f}s ${rec['usd']:.3f}"
                      f"  total ${state['spent']:.2f}{'  ERROR ' + rec['error'][:80] if 'error' in rec else ''}")

    await asyncio.gather(*(item(q) for q in todo))
    if state["stopped"]:
        print(f"{name}: stopped at the budget (${state['spent']:.2f} of ${budget})")
    return not state["stopped"]


def cmd_run(args):
    gold = json.loads(GOLD_PATH.read_text())
    if args.ids:
        gold = [q for q in gold if q["id"] in args.ids.split(",")]
    RUNS.mkdir(parents=True, exist_ok=True)
    asyncio.run(_load())  # BGE-M3 before the first search_notes, not inside it
    for name in args.setups.split(","):
        if not asyncio.run(run_setup(name, gold, args.budget)):
            break
    print(f"spent ${spent():.2f}")


async def _load():
    await asyncio.get_running_loop().run_in_executor(agent._tool_thread, textindex.load_model)


# ------------------------------------------------------------------ score

def numbers(text):
    text = agent.ID.sub(" ", text)  # digits inside part numbers and report ids aren't answers
    found = {float(n.replace(",", "")) for n in NUMBER.findall(text)}
    return found | {float(WORDS[w]) for w in re.findall(r"[a-z]+", text.lower()) if w in WORDS}  # "Four parts ..."


def has_number(text, n, tol=0.0):
    return any(abs(x - n) <= tol * max(abs(n), 1) for x in numbers(text))


def set_scores(cited, found, gold):
    """recall against at most 10 (the agent names at most 10 examples), precision of what it cited."""
    gold, cited = set(gold), set(cited)
    cap = min(len(gold), 10) or 1
    return {"recall": min(len(cited & gold) / cap, 1.0), "precision": len(cited & gold) / len(cited) if cited else 0.0,
            "retrieval": min(len(set(found) & gold) / cap, 1.0)}


def judge(client, reference, record, question, answer, criterion=None):
    """why: is the stated reason right, given the part's record. conclusion: the question's own criterion."""
    model, effort = JUDGE
    if criterion:
        prompt = ("You grade an answer from a parts-database assistant.\n\n"
                  f"Question: {question}\n\nThe truth, from the records: {reference}\n\nAnswer: {answer}\n\n"
                  f"{criterion} Reply with JSON only: " '{"correct": true or false, "why": "<one sentence>"}')
    else:
        prompt = ("You grade an answer from a parts-database assistant against the record it should be based on.\n\n"
                  f"Question: {question}\n\nThe reason, from the records: {reference}\n\n"
                  f"The part's full record (other facts the answer may correctly use): {record}\n\nAnswer: {answer}\n\n"
                  "Grade only the reason it gives for the revision: is it correct? Wording and language may differ. "
                  "Other details count against it only if they contradict the records. Reply with JSON only: "
                  '{"correct": true or false, "why": "<one sentence>"}')
    msg = client.messages.create(model=model, max_tokens=2000, output_config={"effort": effort},
                                 messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in msg.content if b.type == "text")
    verdict = json.loads(text[text.index("{"):text.rindex("}") + 1])
    return verdict, cost(msg.usage.model_dump(), model)


def score_turn(kind, gold, turn, question, user, facts, prev=None, judged=None):
    a = turn["answer"]
    ids = set(agent.ID.findall(a))
    cited = set(PART.findall(a))
    s = {}
    if kind in ("count", "negation"):
        s["correct"] = has_number(a, gold["number"])
    elif kind == "group":
        s["correct"] = all(has_number(a, n) for n in gold["groups"].values())
    elif kind in ("list", "notes", "similar"):
        s.update(set_scores(cited, turn["tool_ids"], gold["parts"]))
        # Precision alone misjudges answers that also name parts to exclude ("not fixed: ..."): stating the right
        # count passes too. similar: other look-alikes named as context are fine.
        s["count_stated"] = has_number(a, len(gold["parts"]))
        s["correct"] = s["recall"] >= 0.8 and (kind == "similar" or s["precision"] >= 0.8 or s["count_stated"])
        if gold.get("forbidden"):
            s["correct"] = s["correct"] and not cited & set(gold["forbidden"])
        if gold.get("ids"):
            s["ids_cited"] = len(ids & set(gold["ids"])) / len(gold["ids"])
            s["correct"] = s["correct"] and s["ids_cited"] >= 0.5
    elif kind == "rest":  # both turns together should cover the list
        both = cited | set(PART.findall(prev["answer"]))
        s.update(set_scores(both, set(turn["tool_ids"]) | set(prev["tool_ids"]), gold["parts"]))
        s["recall_full"] = len(both & set(gold["parts"])) / len(gold["parts"])
        s["correct"] = s["recall_full"] >= 0.8 and len(cited - set(gold["parts"])) == 0
    elif kind == "lookup":
        ok = [f.lower() in a.lower() if len(f) > 1 else re.search(rf"(?<![A-Za-z]){f}(?![A-Za-z])", a) is not None
              for f in gold.get("fields", [])]
        ok += [has_number(a, n) for n in gold.get("numbers", [])]
        if "mass_g" in gold:
            ok.append(has_number(a, gold["mass_g"], 0.01) or has_number(a, gold["mass_g"] / 1000, 0.01))
        if "result" in gold:
            ok.append(any(w in a.lower() for w in RESULT_WORDS[gold["result"]]))
            ok.append(all(i in ids for i in gold["ids"]))
        s["correct"] = all(ok) if ok else None  # turn 1 of a conversation has no gold of its own
    elif kind == "rates":  # each rate as its a/b counts, or as a percentage within a point; either reading
        rate_ok = lambda rates: all((has_number(a, k) and has_number(a, n)) or has_number(a, round(100 * k / n), 0.08)
                                    for k, n in rates)
        s["correct"] = rate_ok(gold["rates"]) or rate_ok(gold.get("rates_alt", []) or [[0, 1]])
    elif kind == "conclusion":
        if gold.get("parts"):  # how many of the true parts it found: shown, not scored
            s["recall_full"] = len(cited & set(gold["parts"])) / len(gold["parts"])
        s["judge"] = judged
        s["correct"] = bool(judged and judged["correct"])
    elif kind == "why":
        s["ids_cited"] = all(i in ids for i in gold["ids"])
        s["judge"] = judged
        s["correct"] = s["ids_cited"] and bool(judged and judged["correct"])
    elif kind == "permission":
        together = gold.get("hidden_together")
        s["correct"] = not (ids & set(gold["hidden_ids"])) and not (
            together and together[0] in a and has_number(a, together[1]))
    elif kind == "not_found":  # alternatives may be offered ("no crash test; cold impact failures are ...")
        s["correct"] = True  # wrong only through an invented id, below
    elif kind == "smalltalk":
        s["correct"] = not turn["trace"] and not ids and not turn["repaired"]

    asked = set(agent.ID.findall(question))  # repeating the asked part number gives nothing away
    s["leaks"] = sorted(i for i in ids - asked if facts.program.get(i) and facts.program[i] not in USERS[user])
    s["invented"] = sorted(i for i in ids if i not in facts.program and i not in question)
    if s["leaks"] or (kind == "not_found" and s["invented"]):
        s["correct"] = False
    return s


def routing(expected, turn, kind):
    called = {t["tool"] for t in turn["trace"]}
    if kind == "smalltalk":
        return not called
    return bool(called & set(expected)) if expected else None


def cmd_score(args):
    gold = {q["id"]: q for q in json.loads(GOLD_PATH.read_text())}
    facts = Facts()
    judge_path = RUNS / "judge.jsonl"
    judged = {(r["setup"], r["id"]): r for r in map(json.loads, judge_path.open())} if judge_path.exists() else {}
    client = anthropic.Anthropic(max_retries=6)
    result = {}
    for path in sorted(RUNS.glob("*.jsonl")):
        if path.stem not in SETUPS:
            continue
        rows = []
        for rec in map(json.loads, path.open()):
            q = gold[rec["id"]]
            if "error" in rec or not rec["turns"]:
                rows.append({"id": q["id"], "kind": q["kind"], "lang": q["lang"], "error": rec.get("error")})
                continue
            if "turns" in q:  # score the last turn; the first only sets it up
                spec, turn = q["turns"][-1], rec["turns"][-1]
                kind, g, expected, prev = spec["kind"], spec["gold"], spec.get("tools", []), rec["turns"][0]
            else:
                kind, g, expected, turn, prev = q["kind"], q["gold"], q["tools"], rec["turns"][0], None
            verdict = None
            if kind in ("why", "conclusion"):
                key = (rec["setup"], rec["id"])
                if key not in judged:
                    v, usd = judge(client, g["reference"], g.get("record"), turn["question"], turn["answer"],
                                   g.get("criterion"))
                    judged[key] = {"setup": rec["setup"], "id": rec["id"], **v, "usd": usd}
                    with judge_path.open("a") as f:
                        f.write(json.dumps(judged[key], ensure_ascii=False) + "\n")
                verdict = judged[key]
            s = score_turn(kind, g, turn, turn["question"], q["user"], facts, prev, verdict)
            s["leaks"] += [i for t in rec["turns"][:-1] for i in set(agent.ID.findall(t["answer"]))
                           - set(agent.ID.findall(t["question"]))
                           if facts.program.get(i) and facts.program[i] not in USERS[q["user"]]]
            rows.append({"id": q["id"], "kind": q["kind"], "lang": q["lang"], **s,
                         "routing": routing(expected, turn, kind),
                         "repaired": any(t["repaired"] for t in rec["turns"]),
                         "unverified": sum(len(t["unverified_ids"]) for t in rec["turns"]),
                         "seconds": sum(t["seconds"] for t in rec["turns"]), "usd": rec["usd"],
                         "tool_calls": sum(len(t["trace"]) for t in rec["turns"])})
        result[path.stem] = summarize(rows)
        result[path.stem]["rows"] = rows

    kinds = sorted({r["kind"] for s in result.values() for r in s["rows"]})
    names = [n for n in SETUPS if n in result]
    print(f"{'':22}" + "".join(f"{n:>15}" for n in names))
    lines = [("answers correct", lambda s: f"{s['correct']:.0%} ({s['n_correct']}/{s['n']})"),
             ("  English", lambda s: f"{s['by_lang'].get('en', float('nan')):.0%}"),
             ("  Korean", lambda s: f"{s['by_lang'].get('ko', float('nan')):.0%}"),
             ("routing", lambda s: f"{s['routing']:.0%}"),
             ("retrieval (lists)", lambda s: f"{s['retrieval']:.0%}"),
             ("leaks", lambda s: str(s["leaks"])),
             ("invented ids", lambda s: str(s["invented"])),
             ("id repairs", lambda s: str(s["repaired"])),
             ("errors", lambda s: str(s["errors"])),
             ("seconds median", lambda s: f"{s['seconds_median']:.1f}"),
             ("seconds p95", lambda s: f"{s['seconds_p95']:.1f}"),
             ("$ per question", lambda s: f"{s['usd_mean']:.3f}"),
             ("$ total", lambda s: f"{s['usd_total']:.2f}")]
    for label, fn in lines:
        print(f"{label:22}" + "".join(f"{fn(result[n]):>15}" for n in names))
    print("\nby kind (correct)")
    for k in kinds:
        print(f"  {k:20}" + "".join(f"{result[n]['by_kind'].get(k, float('nan')):>15.0%}" for n in names))
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"\nSaved {OUT}  (judge ${sum(r.get('usd', 0) for r in judged.values()):.2f})")


def summarize(rows):
    ok = [r for r in rows if "error" not in r and r.get("correct") is not None]
    mean = lambda rs, k: float(np.mean([bool(r[k]) for r in rs])) if rs else float("nan")
    by = lambda key: {v: mean([r for r in ok if r[key] == v], "correct") for v in sorted({r[key] for r in ok})}
    routed = [r for r in ok if r["routing"] is not None]
    listed = [r for r in ok if "retrieval" in r]
    secs = [r["seconds"] for r in ok]
    return {"n": len(ok), "n_correct": sum(bool(r["correct"]) for r in ok), "correct": mean(ok, "correct"),
            "by_kind": by("kind"), "by_lang": by("lang"), "routing": mean(routed, "routing"),
            "retrieval": float(np.mean([r["retrieval"] for r in listed])) if listed else float("nan"),
            "leaks": sum(len(r["leaks"]) for r in ok), "invented": sum(len(r["invented"]) for r in ok),
            "repaired": sum(r["repaired"] for r in ok), "unverified": sum(r["unverified"] for r in ok),
            "errors": sum("error" in r for r in rows),
            "seconds_median": float(np.median(secs)), "seconds_p95": float(np.percentile(secs, 95)),
            "usd_mean": float(np.mean([r["usd"] for r in ok])), "usd_total": float(sum(r.get("usd", 0) or 0 for r in rows))}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--setups", default=",".join(SETUPS))
    run.add_argument("--budget", type=float, default=15.0)
    run.add_argument("--ids", help="only these question ids (a smoke test)")
    score = sub.add_parser("score")
    for p in (run, score):
        p.add_argument("--set", choices=list(SETS), default="main")
    args = parser.parse_args()
    global GOLD_PATH, RUNS, OUT
    GOLD_PATH, RUNS, OUT = SETS[args.set]
    {"run": cmd_run, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
