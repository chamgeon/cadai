"""Step 8D: what each UI page costs in time and memory, measured on the calls the pages make.

Runs outside Streamlit (the st.cache_* decorators fall back to an in-process
cache), so it times the backend, not the browser drawing the page.

    cold      imports, CAD worker warm-up, models() (BGE-M3, Kiwi, DINOv2)
    search    data.search, 20 new queries, then the same 20 again (cache hit)
    part      data.part + data.similar for 20 parts the user may see
    attach    data.process_upload + data.similar_upload for 5 STEP files
    ask       agent.ask_async for 10 questions (Opus 5.5, effort medium): LLM vs tool time, $
    soak      --rounds x (10 searches, 5 parts, 1 upload, 1 ask), memory footprint after each
              round: does it level off?

Usage:
    python -m src.eval.system_bench [--rounds 20]      # ~$2 of API calls at 20 rounds
    python -m src.eval.system_bench --soak-only        # rerun the soak, keep the saved latencies
"""
import time

T0 = time.perf_counter()

import argparse
import asyncio
import json
import random
import shutil
import sqlite3
import tempfile
from pathlib import Path

import anthropic
import numpy as np
import torch

from src.config import DB_PATH, EVAL_RESULTS as RESULTS, RAW
from src.eval.common import cost, footprint_mb
from src.qa import agent
from src.ui import data, resources

T_IMPORT = time.perf_counter() - T0
USER = "normal"

SEARCHES = [
    "aluminum bracket durability failure", "salt spray corrosion on steel cover", "fatigue crack near hole",
    "PA66 housing heat aging", "bolt torque loss vibration", "gear wear after endurance test",
    "flange dimensional out of tolerance", "cold impact plastic cracked", "SUS304 washer", "shaft NVH",
    "브래킷 피로 균열", "염수분무 부식 발생", "알루미늄 하우징 열노화", "볼트 진동 풀림", "기어 마모",
    "커버 치수 불량", "저온 충격 파손", "ADC12 die cast porosity", "beam static load pass", "pin press fit",
]
QUESTIONS = [
    "Which aluminum parts failed durability tests?",
    "알루미늄 부품 중 피로 시험에서 불합격한 것은?",
    "What changed in the latest revision of the most recently released bracket?",
    "How many parts use PA66-GF30, by status?",
    "Any corrosion problems reported on steel covers?",
    "Find parts similar to the first housing you can see and tell me if any failed a test.",
    "Which supplier has the most conditional results in salt spray?",
    "저온 충격 시험에서 문제가 된 플라스틱 부품 알려줘",
    "Show me released shafts heavier than 500 g.",
    "hi",
]
CLASSES = ["bracket", "cover", "housing", "shaft", "gear", "bolt", "flange", "plate", "beam", "pin"]
PROBLEMS = ["crack", "corrosion", "wear", "noise", "deformation", "loosening", "porosity", "fracture"]


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start


def dist(values):
    v = np.array(values, dtype=float)
    return {"n": len(v), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max())}


def mps_mb():
    return torch.mps.driver_allocated_memory() / 2**20 if torch.backends.mps.is_available() else 0.0


def memory():
    return {**footprint_mb(), "mps_driver": mps_mb()}


def visible_parts(n, rng):
    conn = sqlite3.connect(DB_PATH)
    ids = [r[0] for r in conn.execute("SELECT part_no FROM part_cards WHERE program = 'P1' ORDER BY part_no")]
    return rng.sample(ids, n)


def step_files(n, rng):
    """Your own bracket plus random pipeline files with 1-3 solids (uploads are single parts, mostly)."""
    conn = sqlite3.connect(DB_PATH)
    names = [r[0] for r in conn.execute("SELECT file_name FROM files WHERE status = 'ok' AND n_solids <= 3"
                                        " AND file_id != 'my_bracket' ORDER BY file_name")]
    return [RAW / "my_bracket.step", *(RAW / f for f in rng.sample(names, n - 1))]


def attach(path):
    work = Path(tempfile.mkdtemp(prefix="cadai-bench-"))
    try:
        shutil.copy(path, work / path.name)
        start = time.perf_counter()
        res = data.process_upload(work / path.name, work / "parts")
        stages = dict(res.get("seconds", {}))
        t = time.perf_counter()
        for p in res.get("parts", []):
            if p["status"] == "ok":
                data.similar_upload(USER, f"bench/{path.name}/{p['part_id']}/{t}", p["views"], p["dims"][0])
        stages["similar"] = time.perf_counter() - t
        return {"file": path.name, "ok": res["ok"], "seconds": time.perf_counter() - start, "stages": stages}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def ask(client, question):
    out = asyncio.run(agent.ask_async(question, USER, client))
    llm = sum(c["seconds"] for c in out["calls"])
    tools = sum(t["seconds"] for t in out["trace"])
    return {"question": question, "seconds": out["seconds"], "llm_seconds": llm, "tool_seconds": tools,
            "n_tool_calls": len(out["trace"]), "rounds": out["rounds"], "repaired": out["repaired"],
            "output_tokens": out["usage"].get("output_tokens", 0), "usd": cost(out["usage"], agent.MODEL)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--soak-only", action="store_true")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    out = RESULTS / "step8_system.json"
    saved = json.loads(out.read_text()) if args.soak_only else {}
    result = {"memory": {"after_import": memory()}}

    # cold start, in app.py's order: CAD worker first (it loads in its own process), then the models
    cad, t_cad_start = timed(resources.cad)
    models, t_models = timed(resources.models)
    _, t_cad_wait = timed(cad.ready)  # what is still left of the worker's warm-up once the models are in
    result["cold"] = {"imports": T_IMPORT, "cad_worker_start": t_cad_start, "models": t_models,
                      "cad_worker_wait_after_models": t_cad_wait,
                      "total": time.perf_counter() - T0}
    result["memory"]["after_cold_start"] = memory()
    print(f"cold start {result['cold']['total']:.1f} s  (imports {T_IMPORT:.1f}, models {t_models:.1f}, "
          f"CAD worker wait {t_cad_wait:.1f})")
    client = anthropic.AsyncAnthropic(max_retries=3)
    if args.soak_only:
        result = {**saved, "memory": result["memory"]}
        asks = saved["ask"]["runs"]
    else:
        asks = latency(result, client, rng)

    soak(result, client, rng, args)
    result["usd_total"] = sum(a["usd"] for a in asks) + sum(s["usd"] for s in result["soak"])

    RESULTS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved {out}  (API ${result['usd_total']:.2f})")


def latency(result, client, rng):
    new = [timed(data.search, USER, q)[1] for q in SEARCHES]
    hit = [timed(data.search, USER, q)[1] for q in SEARCHES]
    result["search"] = {"new": dist(new), "cached": dist(hit)}
    print(f"search     median {np.median(new):.2f} s  p95 {np.percentile(new, 95):.2f}   cached {np.median(hit):.3f}")

    part_t, sim_t = [], []
    for pn in visible_parts(20, rng):
        part_t.append(timed(data.part, USER, pn)[1])
        sim_t.append(timed(data.similar, USER, pn, 10)[1])
    both = [a + b for a, b in zip(part_t, sim_t)]
    result["part"] = {"record": dist(part_t), "similar": dist(sim_t), "page": dist(both)}
    print(f"part       median {np.median(both):.2f} s  p95 {np.percentile(both, 95):.2f}  "
          f"(record {np.median(part_t):.2f}, similar {np.median(sim_t):.2f})")

    uploads = [attach(p) for p in step_files(5, rng)]
    result["attach"] = {"runs": uploads, "total": dist([u["seconds"] for u in uploads])}
    print(f"attach     median {result['attach']['total']['median']:.2f} s  max {result['attach']['total']['max']:.2f}"
          f"  ({sum(u['ok'] for u in uploads)}/{len(uploads)} ok)")

    asks = [ask(client, q) for q in QUESTIONS]
    result["ask"] = {"runs": asks, "total": dist([a["seconds"] for a in asks]),
                     "llm": dist([a["llm_seconds"] for a in asks]), "tools": dist([a["tool_seconds"] for a in asks]),
                     "usd_per_question": float(np.mean([a["usd"] for a in asks]))}
    print(f"ask        median {result['ask']['total']['median']:.1f} s  p95 {result['ask']['total']['p95']:.1f}  "
          f"(LLM {result['ask']['llm']['median']:.1f}, tools {result['ask']['tools']['median']:.2f})  "
          f"${result['ask']['usd_per_question']:.3f}/question")
    result["memory"]["after_latency"] = memory()
    return asks


def soak(result, client, rng, args):
    # soak: new inputs every round, so the caches keep growing the way a long session's would
    rounds, files = [], step_files(args.rounds + 1, random.Random(args.seed + 1))[1:]
    for i in range(args.rounds):
        start = time.perf_counter()
        for _ in range(10):
            data.search(USER, f"{rng.choice(CLASSES)} {rng.choice(PROBLEMS)} {rng.randint(1, 10**6)}")
        for pn in visible_parts(5, rng):
            data.part(USER, pn)
            data.similar(USER, pn, 10)
        attach(files[i])
        a = ask(client, QUESTIONS[i % len(QUESTIONS)])
        rounds.append({"round": i + 1, "seconds": time.perf_counter() - start, "usd": a["usd"], **memory()})
        r = rounds[-1]
        print(f"soak {i + 1:2}/{args.rounds}  self {r['self']:6.0f} MB  CAD worker {r['children']:5.0f} MB"
              f"  MPS {r['mps_driver']:6.0f} MB")
    result["soak"] = rounds


if __name__ == "__main__":
    main()
