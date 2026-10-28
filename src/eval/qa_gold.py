"""Step 8C: the Q&A gold set, every answer computed from data/plm/facts.json.

facts.json is what src.qa.plm generated the records from (facts first, text second),
so it is the truth the agent's answers are checked against, independent of the tools
the agent uses. Sizes come from step 1's measured geometry (parts.dim_1).

Questions are worded by hand, not from the note templates, ~30% in Korean, and pick
their parameters (a part, a material, a location) from the facts with a fixed seed, so
every gold answer is small enough to check: 2-10 parts for lists, a number for counts.

    kind         what it tests                                  scored by
    count        query_parts with filters, incl. size and "open"  the gold number in the answer
    group        query_parts group_by                             every group's count in the answer
    list         filters -> a short list of parts                 recall, and precision or the right count
    negation     "never failed": count minus count                the gold number
    lookup       one part's fields (material, supplier, mass ...) the field values in the answer
    why          why a revision happened                          the change notice cited + an LLM judge
    notes        a defect at a location, only in the note text    recall, and precision or the right count
    similar      shape search, then history (the planted story)   the planted parts and their fixes
    permission   a part in another program                        none of its ids, nor its supplier + cost
    not_found    a part or test that doesn't exist                no invented ids (alternatives are fine)
    smalltalk    "hi"                                             no tools, no ids, no id repair
    multi        two-turn conversations                           the second turn's gold

The hard set (--set hard) is 10 questions that need 2-4 dependent tool rounds: look up
what a search found, compare rates, page through more change notices than one search
returns, chain shape search -> history -> supplier, and one trap whose answer is "none".
Two more checks there: rates (14/65 or 22%) and conclusion (an LLM judge, yes/no).

Usage:
    python -m src.eval.qa_gold [--set hard]   # -> src/eval/results/step8_qa_gold.json (or _hard_gold)
"""
import argparse
import json
import random
import sqlite3
from collections import defaultdict

from src.config import DATA, DB_PATH, EVAL_RESULTS as RESULTS
from src.qa.plm import USERS

FACTS = DATA / "plm" / "facts.json"
GOLD = RESULTS / "step8_qa_gold.json"
HARD_GOLD = RESULTS / "step8_qa_hard_gold.json"
SEED = 8

MATERIAL_EN = {"AL6061-T6": "6061 aluminum", "ADC12": "die-cast aluminum (ADC12)", "SPCC": "SPCC steel sheet",
               "SPFH590": "SPFH590 high-strength steel", "S45C": "S45C carbon steel", "SUS304": "stainless (SUS304)",
               "PA66-GF30": "glass-filled nylon (PA66-GF30)", "PP-T20": "talc-filled PP (PP-T20)", "ABS": "ABS"}
MATERIAL_KO = {"AL6061-T6": "6061 알루미늄", "ADC12": "알루미늄 다이캐스팅(ADC12)", "SPCC": "SPCC 강판",
               "SPFH590": "SPFH590 고장력강", "S45C": "S45C 탄소강", "SUS304": "스테인리스(SUS304)",
               "PA66-GF30": "PA66-GF30", "PP-T20": "PP-T20", "ABS": "ABS"}
CLASS_KO = {"Bracket": "브래킷", "Plate": "플레이트", "Block": "블록", "Beam": "빔", "Body": "바디", "Flange": "플랜지",
            "Cover": "커버", "Shaft": "샤프트", "Washer": "와셔", "Housing": "하우징", "Pin": "핀", "Bolt": "볼트"}
TEST_EN = {"fatigue": "durability (fatigue)", "salt_spray": "salt spray", "heat_aging": "heat aging",
           "cold_impact": "cold impact", "vibration": "vibration", "wear": "wear", "static": "static strength",
           "dimensional": "dimensional", "nvh": "NVH"}
TEST_KO = {"fatigue": "내구(피로)", "salt_spray": "염수분무", "heat_aging": "열노화", "cold_impact": "저온 충격",
           "vibration": "진동", "wear": "마모", "static": "정강도", "dimensional": "치수", "nvh": "NVH"}
# defect locations in the note text, as src.qa.plm.PHRASES renders them (index = the fact's "loc")
LOCS = {("fatigue", 4): ("cracked at the weld toe", "용접 토우에서 균열"),
        ("fatigue", 1): ("cracked at the bend", "굽힘부에서 균열"),
        ("salt_spray", 2): ("corroded where they touch an aluminum part", "알루미늄 부품과의 접촉부 부식"),
        ("salt_spray", 4): ("rusted on the thread", "나사산에 녹"),
        ("heat_aging", 1): ("warped at the snap fit", "스냅핏 변형"),
        ("nvh", 0): ("rattled at the cover edge", "커버 가장자리 소음(래틀)"),
        ("wear", 2): ("wore at the bushing contact", "부싱 접촉부 마모")}


def plural(cls):
    return {"Body": "bodies", "Housing": "housings"}.get(cls, cls.lower() + "s")


class Facts:
    def __init__(self):
        f = json.loads(FACTS.read_text())
        self.parts, self.scenarios = f["parts"], f["scenarios"]
        self.by_no = {p["part_no"]: p for p in self.parts}
        self.by_id = {p["part_id"]: p for p in self.parts}
        conn = sqlite3.connect(DB_PATH)
        self.size = dict(conn.execute("SELECT part_id, dim_1 FROM parts"))
        self.card = dict(conn.execute("SELECT part_no, text FROM part_cards"))  # what lookup_part shows the agent
        # every id -> its part's program: an id from a program the user can't see in an answer is a leak
        self.program = {}
        for p in self.parts:
            self.program[p["part_no"]] = p["program"]
            for t in p["tests"]:
                self.program[t["report_id"]] = p["program"]
            for e in p["ecos"]:
                self.program[e["eco_id"]] = p["program"]

    def visible(self, user):
        return [p for p in self.parts if p["program"] in USERS[user]]


def latest(p):
    return p["revisions"][-1]


def failed(p, test=None, rev=None):
    return any(t["result"] == "fail" and (test is None or t["test"] == test) and (rev is None or t["rev"] == rev)
               for t in p["tests"])


def nos(parts):
    return sorted(p["part_no"] for p in parts)


def record(p, card):
    """The part's whole record, for the judge: an answer may rightly use more of it than the reason itself."""
    revs = "; ".join(f"rev {r['rev']} {r['status']} {r['material']} {r['date']}" for r in p["revisions"])
    tests = "; ".join(f"{t['report_id']} rev {t['rev']} {t['test']} {t['result']} {t['date']}: {t['note']}"
                      for t in p["tests"])
    ecos = "; ".join(f"{e['eco_id']} {e['from']}->{e['to']} {e['date']}: {e['text']}" for e in p["ecos"])
    return f"Part card: {card}\nRevisions: {revs}.\nTests: {tests}.\nChange notices: {ecos}."


def build():
    f, rng = Facts(), random.Random(SEED)
    P1 = f.visible("normal")
    qs = []

    def add(kind, question, user="normal", lang="en", tools=(), **gold):
        qs.append({"id": f"{kind}-{sum(q['kind'] == kind for q in qs) + 1:02d}", "kind": kind, "lang": lang,
                   "user": user, "question": question, "tools": list(tools), "gold": gold})

    def pick(items, lo, hi, key):
        """A random combination whose gold size is between lo and hi."""
        ok = [k for k in items if lo <= len(key(k)) <= hi]
        return rng.choice(sorted(ok))

    by = lambda parts, **kw: [p for p in parts if all(
        (latest(p)[k] if k in ("material", "status") else p[k]) == v for k, v in kw.items())]

    # ---- count: filters query_parts can express
    combos = sorted({(p["class"], latest(p)["material"]) for p in P1})
    for lang in ("en", "en", "ko", "ko"):
        cls, mat = pick(combos, 8, 120, lambda c: by(P1, **{"class": c[0], "material": c[1]}))
        combos.remove((cls, mat))
        n = len(by(P1, **{"class": cls, "material": mat}))
        add("count", f"How many {plural(cls)} do we have in {MATERIAL_EN[mat]}?" if lang == "en" else
            f"{MATERIAL_KO[mat]} {CLASS_KO[cls]} 몇 개 있어?", lang=lang, tools=["query_parts"], number=n)
    cls, mat = "Plate", "SPCC"
    n = len(by(P1, **{"class": cls, "material": mat, "status": "Released"}))
    add("count", "SPCC 플레이트 중 양산 승인(Released) 상태인 건 몇 개입니까?", lang="ko", tools=["query_parts"], number=n)
    for test, lang in (("salt_spray", "en"), ("vibration", "ko")):
        n = len([p for p in P1 if failed(p, test)])
        add("count", f"How many parts have ever failed a {TEST_EN[test]} test, on any revision?" if lang == "en" else
            f"{TEST_KO[test]} 시험에서 한 번이라도 불합격한 부품은 몇 개야?", lang=lang, tools=["query_parts"], number=n)
    n = len([p for p in P1 if any(t["test"] == "fatigue" and t["result"] == "conditional" for t in p["tests"])])
    add("count", "Count the parts with a conditional pass in durability testing.", tools=["query_parts"], number=n)
    n = len([p for p in P1 if failed(p, rev=latest(p)["rev"])])
    add("count", "How many parts have a failed test on their current revision, i.e. an open problem?",
        tools=["query_parts"], number=n)
    n = len([p for p in P1 if f.size[p["part_id"]] >= 500])
    add("count", "How many parts are 500 mm or longer?", tools=["query_parts"], number=n)
    n = len([p for p in P1 if p["class"] == "Bracket" and f.size[p["part_id"]] <= 50])
    add("count", "50mm 이하 작은 브래킷은 몇 개 있어?", lang="ko", tools=["query_parts"], number=n)
    n_admin = len([p for p in f.visible("admin") if p["class"] == "Shaft"])
    add("count", "How many shafts are there across all programs?", user="admin", tools=["query_parts"], number=n_admin)

    # ---- group
    groups = defaultdict(int)
    for p in P1:
        groups[latest(p)["status"]] += 1
    add("group", "Break our parts down by lifecycle status.", tools=["query_parts"], groups=dict(groups))
    groups = defaultdict(int)
    for p in by(P1, **{"class": "Bracket"}):
        groups[latest(p)["material"]] += 1
    add("group", "브래킷을 재질별로 나눠서 개수 알려줘.", lang="ko", tools=["query_parts"], groups=dict(groups))

    # ---- list: class + material + failed test -> 2-8 parts
    combos = sorted({(p["class"], latest(p)["material"], t["test"]) for p in P1 for t in p["tests"]
                     if t["result"] == "fail"})
    key = lambda c: [p for p in by(P1, **{"class": c[0]}) if latest(p)["material"] == c[1] and failed(p, c[2])]
    for lang in ("en", "en", "en", "ko", "ko"):
        c = pick(combos, 2, 8, key)
        combos.remove(c)
        cls, mat, test = c
        add("list", f"Which {MATERIAL_EN[mat]} {plural(cls)} have failed {TEST_EN[test]} testing?" if lang == "en"
            else f"{TEST_KO[test]} 시험에서 불합격한 {MATERIAL_KO[mat]} {CLASS_KO[cls]} 목록 줘.", lang=lang,
            tools=["query_parts", "search_notes"], parts=nos(key(c)))
    shafts = [p for p in by(P1, **{"class": "Shaft", "status": "In Work"})]
    add("list", "List the shafts that are still In Work.", tools=["query_parts"], parts=nos(shafts))

    # ---- negation: needs a count minus a count
    rel = by(P1, **{"class": "Bracket", "status": "Released"})
    add("negation", "How many released brackets have never failed any test?", tools=["query_parts"],
        number=len([p for p in rel if not failed(p)]))
    pl = by(P1, **{"class": "Plate", "material": "SPCC"})
    add("negation", "SPCC 플레이트 중 염수분무 불합격 이력이 전혀 없는 건 몇 개야?", lang="ko", tools=["query_parts"],
        number=len([p for p in pl if not failed(p, "salt_spray")]))
    al = by(P1, **{"material": "AL6061-T6"})
    add("negation", "Of all our 6061 aluminum parts, how many have no failed test on record?", tools=["query_parts"],
        number=len([p for p in al if not failed(p)]))

    # ---- lookup: one part's fields
    sup = [p for p in P1 if p["supplier"]]
    for lang in ("en", "ko"):
        p = rng.choice(sup)
        add("lookup", f"What is {p['part_no']} made of, and who supplies it?" if lang == "en" else
            f"{p['part_no']} 재질이랑 공급사 알려줘.", lang=lang, tools=["lookup_part"],
            fields=[latest(p)["material"], p["supplier"]["supplier"]])
    p = rng.choice(sup)
    add("lookup", f"What's the lead time and unit cost for {p['part_no']}?", tools=["lookup_part"],
        numbers=[p["supplier"]["lead_time_wk"], p["supplier"]["unit_cost_krw"]])
    p = rng.choice([p for p in P1 if latest(p)["mass_g"] >= 10])
    add("lookup", f"How heavy is {p['part_no']}?", tools=["lookup_part"], mass_g=latest(p)["mass_g"])
    multi_rev = [p for p in P1 if len(p["revisions"]) >= 2]
    p = rng.choice(multi_rev)
    add("lookup", f"{p['part_no']}의 현재 리비전은 뭐야?", lang="ko", tools=["lookup_part"], fields=[latest(p)["rev"]])
    for test, lang in (("salt_spray", "en"), ("fatigue", "ko")):
        cand = [(p, t) for p in P1 for t in p["tests"] if t["test"] == test and t["rev"] == latest(p)["rev"]]
        p, t = rng.choice(cand)
        add("lookup", f"Did {p['part_no']} pass its latest {TEST_EN[test]} test?" if lang == "en" else
            f"{p['part_no']} 최신 리비전 {TEST_KO[test]} 시험 결과는?", lang=lang, tools=["lookup_part"],
            result=t["result"], ids=[t["report_id"]])

    # ---- why: a fix revision, judged against the change notice and the failed report
    fixes = [(p, e) for p in P1 for e in p["ecos"] if e["kind"] == "fix"]
    for lang in ("en", "en", "ko"):
        p, e = rng.choice(fixes)
        fixes.remove((p, e))
        ref = next(t for t in p["tests"] if t["report_id"] == e["ref"])
        add("why", f"Why was {p['part_no']} revised from rev {e['from']} to {e['to']}?" if lang == "en" else
            f"{p['part_no']}는 왜 {e['from']}에서 {e['to']}로 리비전이 바뀌었어?", lang=lang, tools=["lookup_part"],
            ids=[e["eco_id"]], reference=f"Change notice {e['eco_id']}: {e['text']}  Failed report {ref['report_id']}: "
                                         f"{ref['note']}", record=record(p, f.card[p["part_no"]]))
    other = [(p, e) for p in P1 for e in p["ecos"] if e["kind"] == "other"]
    p, e = rng.choice(other)
    add("why", f"What was the reason for {p['part_no']}'s change from {e['from']} to {e['to']}?", tools=["lookup_part"],
        ids=[e["eco_id"]], reference=f"Change notice {e['eco_id']}: {e['text']} (not a test failure)", record=record(p, f.card[p["part_no"]]))

    # ---- notes: a defect location only the note text carries
    for (test, loc), (en, ko) in LOCS.items():
        gold = sorted({p["part_no"] for p in P1 for t in p["tests"]
                       if t["test"] == test and t["loc"] == loc and t["result"] in ("fail", "conditional")})
        lang = "ko" if loc % 2 else "en"
        add("notes", f"Which parts {en} during {TEST_EN[test]} testing?" if lang == "en" else
            f"{TEST_KO[test]} 시험에서 {ko} 문제가 있었던 부품은?", lang=lang, tools=["search_notes"], parts=gold)

    # ---- similar: step 6's planted story around my_bracket
    fixed = [f.by_id[i]["part_no"] for i in f.scenarios["my_bracket_neighbors_fixed_P1"]]
    open_p2 = f.by_id[f.scenarios["my_bracket_neighbor_open_P2"]]["part_no"]
    fixed_ecos = [e["eco_id"] for n in fixed for e in f.by_no[n]["ecos"]]
    me = f.by_id["my_bracket_s00"]["part_no"]
    add("similar", f"Has any part that looks like {me} ever failed a test, and how was it fixed?",
        tools=["similar_parts"], parts=fixed, ids=fixed_ecos, forbidden=[open_p2])
    add("similar", f"{me}랑 비슷하게 생긴 부품 중에 문제 있었던 거 있어?", lang="ko", tools=["similar_parts"],
        parts=fixed, forbidden=[open_p2])  # asks whether, not how it was fixed: no change notice required
    add("similar", f"Are there open problems on any part shaped like {me}, in any program?", user="admin",
        tools=["similar_parts"], parts=[open_p2])

    # ---- permission: a P2 part, asked by a P1-only user
    P2 = [p for p in f.parts if p["program"] == "P2" and p["tests"]]
    for lang in ("en", "ko"):
        p = rng.choice(P2)
        add("permission", f"What's the test history of {p['part_no']}?" if lang == "en" else
            f"{p['part_no']} 시험 이력 보여줘.", lang=lang, tools=["lookup_part"], hidden=p["part_no"],
            hidden_ids=[t["report_id"] for t in p["tests"]] + [e["eco_id"] for e in p["ecos"]])
    p = rng.choice([p for p in P2 if p["supplier"]])
    # a material or a supplier name alone is too common to prove a leak: its ids, or supplier and cost together
    add("permission", f"Who supplies {p['part_no']} and at what cost?", tools=["lookup_part"], hidden=p["part_no"],
        hidden_ids=[t["report_id"] for t in p["tests"]],
        hidden_together=[p["supplier"]["supplier"], p["supplier"]["unit_cost_krw"]])

    # ---- not_found: nothing to find; the answer must not invent
    add("not_found", "What's the status of part BR-00017?", tools=["lookup_part"])
    add("not_found", "Which parts failed the crash test?", tools=[])
    add("not_found", "티타늄 브래킷 있어?", lang="ko", tools=["query_parts"])

    # ---- smalltalk
    for q, lang in (("hi", "en"), ("안녕하세요!", "ko"), ("thanks, that's all for now", "en")):
        add("smalltalk", q, lang=lang)

    # ---- multi-turn: the second turn is scored
    def multi(turns, lang="en", user="normal"):
        qs.append({"id": f"multi-{sum(q['kind'] == 'multi' for q in qs) + 1:02d}", "kind": "multi", "lang": lang,
                   "user": user, "turns": turns})

    inwork = sorted(nos(by(P1, **{"class": "Bracket", "material": "AL6061-T6", "status": "In Work"})))
    multi([{"question": "List the In Work 6061 aluminum brackets.", "kind": "list", "gold": {"parts": inwork}},
           {"question": "Show me the rest of them.", "kind": "rest", "tools": [], "gold": {"parts": inwork}}])
    p = rng.choice([p for p in P1 if p["supplier"] and failed(p)])
    multi([{"question": f"What failed on {p['part_no']}?", "kind": "lookup", "gold": {}},
           {"question": "그 부품 공급사랑 리드타임은?", "kind": "lookup", "tools": [],
            "gold": {"fields": [p["supplier"]["supplier"]], "numbers": [p["supplier"]["lead_time_wk"]]}}], lang="mixed")
    pl = [p for p in by(P1, **{"class": "Plate"}) if failed(p, "salt_spray")]
    fixed_pl = [p for p in pl if any(e["kind"] == "fix" and e["test"] == "salt_spray" for e in p["ecos"])]
    multi([{"question": "Which plates have failed salt spray?", "kind": "list", "gold": {"parts": nos(pl)}},
           {"question": "Which of those were fixed with a change notice?", "kind": "list",
            "tools": ["lookup_part", "search_notes", "query_parts"], "gold": {"parts": nos(fixed_pl)}}])
    multi([{"question": f"Tell me about {me}.", "kind": "lookup", "gold": {}},
           {"question": "Have parts that look like it had problems?", "kind": "similar", "tools": ["similar_parts"],
            "gold": {"parts": fixed, "forbidden": [open_p2]}}])
    sh = by(P1, **{"class": "Shaft", "material": "S45C"})
    multi([{"question": "How many S45C shafts are there?", "kind": "count", "gold": {"number": len(sh)}},
           {"question": "그 중 양산 승인된 건 몇 개야?", "kind": "count", "tools": [],
            "gold": {"number": len([p for p in sh if latest(p)["status"] == "Released"])}}], lang="mixed")
    return qs


def build_hard():
    from src.qa.tools import Tools
    f = Facts()
    P1 = f.visible("normal")
    qs = []

    def add(kind, question, lang="en", tools=(), **gold):
        qs.append({"id": f"hard-{len(qs) + 1:02d}", "kind": kind, "lang": lang, "user": "normal",
                   "question": question, "tools": list(tools), "gold": gold})

    # H1: material changed between revisions -> which, from what to what, why (the fix's change notice)
    changed = [p for p in P1 if p["class"] == "Bracket" and len({r["material"] for r in p["revisions"]}) > 1]
    eco_ids = [e["eco_id"] for p in changed for e in p["ecos"]
               if e["kind"] == "fix" and e["fix"][2]]  # the change notices that switched material
    add("list", "Which brackets changed material between revisions? From what to what, and why?",
        tools=["search_notes", "lookup_part"], parts=nos(changed), ids=eco_ids)

    # H2: fatigue failure rate, 6061 vs SPCC brackets: four counts, then a comparison. The rate is among brackets
    # that had a fatigue test (not every bracket has one); per all brackets is accepted as the other reading.
    rates, rates_all = [], []
    for m in ("AL6061-T6", "SPCC"):
        bb = [p for p in P1 if p["class"] == "Bracket" and latest(p)["material"] == m]
        n_failed = len([p for p in bb if failed(p, "fatigue")])
        rates.append([n_failed, len([p for p in bb if any(t["test"] == "fatigue" for t in p["tests"])])])
        rates_all.append([n_failed, len(bb)])
    for lang in ("en", "ko"):
        add("rates", "Do 6061 aluminum brackets fail durability (fatigue) testing more often than SPCC brackets? "
            "Give the failure rate for each." if lang == "en" else
            "6061 알루미늄 브래킷이 SPCC 브래킷보다 내구(피로) 시험 불합격률이 높아? 각각 불합격률 알려줘.",
            lang=lang, tools=["query_parts"], rates=rates, rates_alt=rates_all)

    # H3: fixes for salt spray failures, counted by kind: more change notices than one search returns
    fixes = defaultdict(int)
    for p in P1:
        for e in p["ecos"]:
            if e["kind"] == "fix" and e["test"] == "salt_spray":
                fixes[e["fix"][0].split(" at ")[0]] += 1
    add("lookup", "Across our parts, what fixes were used for salt spray failures, and how many parts got each "
        "kind of fix? Which was most common?", tools=["search_notes"], fields=["zinc"],
        numbers=sorted(fixes.values(), reverse=True), fix_counts=dict(fixes))

    # H4: heaviest released 6061 bracket -> its supplier, lead time, cost
    b = [p for p in P1 if p["class"] == "Bracket" and latest(p)["material"] == "AL6061-T6"
         and latest(p)["status"] == "Released"]
    h = max(b, key=lambda p: latest(p)["mass_g"])
    for lang in ("en", "ko"):
        add("lookup", "Which released 6061 aluminum bracket is the heaviest, and who supplies it, at what lead time "
            "and unit cost?" if lang == "en" else
            "양산 승인된 6061 알루미늄 브래킷 중 가장 무거운 건 뭐고, 공급사·리드타임·단가는?", lang=lang,
            tools=["query_parts"], fields=[h["part_no"], h["supplier"]["supplier"]],
            numbers=[h["supplier"]["lead_time_wk"], h["supplier"]["unit_cost_krw"]])

    # H5: shape search -> which failed fatigue -> the fixed, released one's supplier
    fixed = [f.by_id[i] for i in f.scenarios["my_bracket_neighbors_fixed_P1"]]
    rel = [p for p in fixed if latest(p)["status"] == "Released" and p["supplier"]]
    me = f.by_id["my_bracket_s00"]["part_no"]
    for lang in ("en", "ko"):
        add("lookup", f"Did any part that looks like {me} fail fatigue testing? For the ones that were fixed and are "
            "released now, who is the supplier?" if lang == "en" else
            f"{me}랑 비슷하게 생긴 부품 중 피로 시험 불합격한 게 있어? 그중 수정돼서 양산 승인된 부품의 공급사는?",
            lang=lang, tools=["similar_parts"],
            fields=[x for p in rel for x in (p["part_no"], p["supplier"]["supplier"])])

    # H6: fixes that failed: a fix revision that failed the same test again (then got fixed again). The final
    # revision always passes, so checking only the latest revision wrongly says "none".
    bad = [(p, e) for p in P1 for e in p["ecos"] if e["kind"] == "fix" and any(
        t["test"] == e["test"] and t["rev"] == e["to"] and t["result"] == "fail" for t in p["tests"])]
    listed = "; ".join(f"{p['part_no']} ({e['test']}: rev {e['from']} failed, fixed in rev {e['to']} [{e['eco_id']}], "
                       f"rev {e['to']} failed again)" for p, e in bad)
    add("conclusion", "Has any fix failed? I mean a part that was revised to fix a failed test but still fails "
        "that same test on the new revision.", tools=["query_parts", "search_notes", "lookup_part"],
        parts=nos(p for p, _ in bad),
        criterion="Does the answer say yes, and is every part it presents as a failed fix one of the parts in the "
                  "truth? Naming only some of them is fine; saying none, or naming another part, is not.",
        reference=f"Yes, {len(bad)} parts: {listed}. All were fixed again on the next revision.")

    # H7: reuse: shape search within 1.5x size -> keep those whose current revision passed fatigue
    tools = Tools("normal")
    hits = tools.similar_parts(me, 20, 1.5)["similar"]
    ok = [s["part_no"] for s in hits if any(t["test"] == "fatigue" and t["result"] == "pass"
                                            and t["rev"] == latest(f.by_no[s["part_no"]])["rev"]
                                            for t in f.by_no[s["part_no"]]["tests"])]
    add("similar", f"I'd like to reuse an existing part instead of making {me}. Is there anything similar in shape, "
        "within 1.5x its size, whose current revision passed fatigue testing?", tools=["similar_parts"], parts=ok)
    return qs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=("main", "hard"), default="main")
    args = parser.parse_args()
    qs, path = (build(), GOLD) if args.set == "main" else (build_hard(), HARD_GOLD)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(qs, indent=1, ensure_ascii=False))
    kinds = defaultdict(int)
    for q in qs:
        kinds[q["kind"]] += 1
    n_ko = sum(q["lang"] == "ko" for q in qs)
    n_asks = sum(len(q.get("turns", [None])) for q in qs)
    print(f"{len(qs)} questions ({n_asks} asks), {n_ko} Korean: " + ", ".join(f"{k} {n}" for k, n in kinds.items()))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
