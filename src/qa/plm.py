"""Step 6: generate fake PLM, ERP and test records for every searchable part.

Facts first, text second: each part gets a seeded set of facts (class, material,
revisions, test outcomes, fixes), and the tables and short notes are rendered
from them. The facts are saved as data/plm/facts.json, the truth that step 8's
gold labels are computed from.

The records lean on the real geometry so the text side serves the CAD side:
    class     STEP product name for single-part files, else rules on step 1's
              measured dims and face types
    material  chosen per class; mass = measured volume x density
    failures  conditioned on material and shape (thin aluminum cracks, uncoated
              steel rusts, PP warps); a failure on rev A is usually fixed in
              rev B by a change notice, and a part never ships with an open fail

Scattered on purpose, as in a real company, joined only by part number + rev:
    plm_items           one row per revision: name, class, material, mass, status, program
    erp_suppliers       released parts only (a few have none)
    doc_test_reports    structured result + a 1-3 sentence note, 40% in Korean
    doc_change_notices  rev A -> B with a one-sentence reason

Planted for the demo: two parts similar to my_bracket (step 4 search) in program
P1 cracked at a bolt hole on rev A and were fixed on rev B; a third in P2 has an
open crack, so a P1-only user must not see it.

Usage:
    python -m src.qa.plm             # regenerate (deterministic for a given set of parts)
"""
import argparse
import hashlib
import json
import random
import re
import sqlite3
from datetime import date, timedelta

from src.config import DATA, DB_PATH

PLM = DATA / "plm"
SEED = 6
PROGRAMS = ("P1", "P2")
USERS = {"normal": ("P1",), "admin": PROGRAMS}  # user -> programs they may see
KO_SHARE = 0.4  # share of notes written in Korean

# controlled vocabulary, as picked from a PLM material library: name -> density g/cm^3, KRW per kg
MATERIALS = {
    "AL6061-T6": (2.70, 6000), "ADC12": (2.74, 4500), "SPCC": (7.85, 1500), "SPFH590": (7.85, 1800),
    "S45C": (7.85, 2000), "SUS304": (7.93, 5000), "PA66-GF30": (1.36, 5500), "PP-T20": (1.04, 2500),
    "ABS": (1.05, 3000),
}
PLASTICS = {"PA66-GF30", "PP-T20", "ABS"}
# a fix may switch material only within its family: a casting doesn't become a steel sheet
UPGRADES = {"SPFH590": {"SPCC"}, "SUS304": {"SPCC", "S45C", "SPFH590"}, "PA66-GF30": {"PP-T20", "ABS"}}

# class -> (part number prefix, owner team, {material: weight}, tests it goes through)
CLASSES = {
    "Bracket": ("BR", "Chassis Design", {"AL6061-T6": 5, "SPCC": 5}, ("fatigue", "static", "salt_spray")),
    "Plate": ("PL", "Body Design", {"SPCC": 5, "AL6061-T6": 3, "SUS304": 2}, ("fatigue", "salt_spray", "dimensional")),
    "Bolt": ("FS", "Standard Parts", {"S45C": 8, "SUS304": 2}, ("vibration", "salt_spray")),
    "Nut": ("FS", "Standard Parts", {"S45C": 8, "SUS304": 2}, ("vibration", "salt_spray")),
    "Washer": ("FS", "Standard Parts", {"SUS304": 5, "S45C": 5}, ("salt_spray", "dimensional")),
    "Pin": ("PN", "Chassis Design", {"S45C": 8, "SUS304": 2}, ("wear", "fatigue")),
    "Shaft": ("SH", "Powertrain Mounts", {"S45C": 1}, ("wear", "fatigue", "dimensional")),
    "Gear": ("GR", "Powertrain Mounts", {"S45C": 1}, ("wear", "fatigue")),
    "Flange": ("FL", "Powertrain Mounts", {"S45C": 5, "ADC12": 5}, ("dimensional", "salt_spray")),
    "Housing": ("HS", "Powertrain Mounts", {"ADC12": 6, "PA66-GF30": 4}, ("heat_aging", "dimensional", "vibration")),
    "Cover": ("CV", "Interior Trim", {"PP-T20": 5, "PA66-GF30": 3, "ABS": 2}, ("heat_aging", "cold_impact", "nvh")),
    "Beam": ("BM", "Body Design", {"AL6061-T6": 7, "S45C": 3}, ("fatigue", "static")),
    "Block": ("BK", "Chassis Design", {"S45C": 5, "AL6061-T6": 5}, ("static", "dimensional")),
    "Body": ("BD", "Body Design", {"AL6061-T6": 3, "S45C": 3, "PA66-GF30": 2, "ADC12": 2}, ("dimensional", "static")),
}
# product-name keyword -> class, first match wins ("X Nut Mount" is a mount)
KEYWORDS = [
    (r"bracket|mount|holder|support|clamp", "Bracket"), (r"beam|profile|rail|extrusion", "Beam"), (r"plate|panel|sheet", "Plate"),
    (r"bolt|screw", "Bolt"), (r"nut", "Nut"), (r"washer|spacer|ring", "Washer"), (r"pin|rod", "Pin"),
    (r"shaft|axle|leadscrew", "Shaft"), (r"gear|pinion|sprocket|pulley", "Gear"), (r"flange|hub", "Flange"),
    (r"housing|case|casing|enclosure|box", "Housing"), (r"cover|cap|lid|shield", "Cover"), (r"block|base", "Block"),
]
# CAD defaults and feature names that say nothing about the part
PLACEHOLDER = re.compile(r"(part|body|solid|surface|curve|rib|sketch|pad|pocket|extrude|fillet|component|"
                         r"untitled|unnamed|shape|feature|default|product|model|object|item)[\s_\-]*\d*", re.I)

TESTS = {  # key -> (English name, Korean name)
    "fatigue": ("durability (fatigue)", "내구(피로)"), "salt_spray": ("salt spray", "염수분무"),
    "heat_aging": ("heat aging", "열노화"), "cold_impact": ("cold impact", "저온 충격"),
    "vibration": ("vibration", "진동"), "wear": ("wear", "마모"), "static": ("static strength", "정강도"),
    "dimensional": ("dimensional", "치수"), "nvh": ("NVH", "NVH"),
}
TEAMS_KO = {"Chassis Design": "샤시설계", "Body Design": "차체설계", "Standard Parts": "표준부품",
            "Powertrain Mounts": "PT마운트", "Interior Trim": "내장트림"}


# ---------------------------------------------------------------- note templates
# Per test: locations and defect words as (English, Korean) synonym pools, how the
# measurement is drawn, note templates per result and language, and the fixes a
# change notice can apply (a fix may change the material).

def _p_fatigue(r):
    target = r.choice((200, 500, 1000))
    return {"target": target, "at": int(target * r.uniform(0.3, 0.9))}


def _p_salt(r):
    target = r.choice((240, 480, 720))
    return {"target": target, "at": int(target * r.uniform(0.3, 0.8) / 24) * 24}


def _p_heat(r):
    return {"T": r.choice((85, 105, 120)), "hours": r.choice((500, 1000)), "mm": round(r.uniform(0.8, 3.0), 1)}


def _p_cold(r):
    return {"T": r.choice((-30, -40))}


def _p_vib(r):
    return {"hours": r.choice((8, 24, 50)), "pct": r.randint(40, 72)}


def _p_wear(r):
    return {"target": r.choice((100, 300)), "mm": round(r.uniform(0.15, 0.6), 2)}


def _p_static(r):
    target = r.choice((5, 10, 20, 30))
    return {"target": target, "at": round(target * r.uniform(0.5, 0.9), 1)}


def _p_dim(r):
    tol = r.choice((0.1, 0.2))
    return {"tol": tol, "mm": round(tol * r.uniform(1.5, 4), 2), "tol2": round(tol * 2, 1)}


def _p_nvh(r):
    return {"hz": r.choice((18, 25, 32, 45))}


PHRASES = {
    "fatigue": {
        "loc": [("bolt hole", "볼트 체결부"), ("the bend", "벤딩부"), ("fillet radius", "필렛부"),
                ("mounting hole", "장착 홀 주변"), ("weld toe", "용접부")],
        "defect": (["crack", "fracture", "cracking"], ["균열", "크랙", "파단"]),
        "params": _p_fatigue,
        "fail": (["{Defect} at {loc} after {at}k cycles (target {target}k).",
                  "Failed at {at}k of {target}k cycles: {defect} initiated near {loc}.",
                  "Durability test stopped at {at}k cycles, {defect} found at {loc}."],
                 ["{at}k cycle 후 {loc} {defect} 발생 (목표 {target}k).",
                  "내구시험 {at}k cycle에서 {loc} {defect} 확인, 시험 중단.",
                  "목표 {target}k cycle 미달: {at}k에서 {loc} {defect}."]),
        "conditional": (["Completed {target}k cycles; minor {defect} indication at {loc}, recheck next build.",
                         "Passed {target}k cycles but hairline {defect} at {loc} under dye penetrant."],
                        ["목표 {target}k cycle 완료, {loc} 미세 {defect} 징후. 차기 빌드 재확인.",
                         "{target}k cycle 통과했으나 침투탐상 시 {loc} 미세 {defect} 관찰."]),
        "pass": (["Completed {target}k cycles, no {defect} found.", "Passed: {target}k cycles without damage."],
                 ["목표 {target}k cycle 완료, 이상 없음.", "{target}k cycle 내구 합격, {defect} 없음."]),
        "fix": [("added rib near {loc}", "{loc} 리브 추가", None),
                ("wall thickness 2.0 → 3.0 mm", "두께 2.0 → 3.0 mm 증대", None),
                ("fillet radius R2 → R5", "필렛 R2 → R5 확대", None)],
    },
    "salt_spray": {
        "loc": [("cut edges", "절단면"), ("edges", "모서리"), ("contact with the aluminum part", "알루미늄 접촉부"),
                ("weld bead", "용접 비드"), ("thread", "나사부")],
        "defect": (["red rust", "corrosion", "rust spots"], ["적청", "부식", "녹"]),
        "params": _p_salt,
        "fail": (["{Defect} on {loc} after {at} h salt spray (spec {target} h).",
                  "Salt spray: {defect} at {loc} at {at} h, below the {target} h requirement."],
                 ["염수분무 {at}h 후 {loc} {defect} 발생 (기준 {target}h).",
                  "{loc}에 {defect} 발생, {at}h에서 기준 {target}h 미달."]),
        "conditional": (["{target} h completed; slight {defect} at {loc}, within appearance limit."],
                        ["{target}h 완료, {loc} 경미한 {defect} (외관 기준 이내)."]),
        "pass": (["No {defect} after {target} h salt spray.", "Passed {target} h salt spray."],
                 ["염수분무 {target}h 후 {defect} 없음.", "염수분무 {target}h 합격."]),
        "fix": [("zinc-nickel coating added", "아연-니켈 도금 추가", None),
                ("material changed to SUS304", "재질 SUS304로 변경", "SUS304"),
                ("insulating washer added at {loc}", "{loc} 절연 와셔 추가", None)],
    },
    "heat_aging": {
        "loc": [("flange", "플랜지"), ("snap fit", "스냅핏"), ("side wall", "측벽"), ("boss", "보스부")],
        "defect": (["warpage", "deformation", "distortion"], ["변형", "뒤틀림", "휨"]),
        "params": _p_heat,
        "fail": (["{Defect} of {mm} mm at {loc} after {hours} h at {T} °C (limit 0.5 mm).",
                  "Heat aging {T} °C / {hours} h: {loc} {defect} {mm} mm, exceeds 0.5 mm."],
                 ["{T}°C {hours}h 열노화 후 {loc} {defect} {mm} mm (기준 0.5 mm 초과).",
                  "열노화 시험에서 {loc} {defect} {mm} mm 발생."]),
        "conditional": (["{Defect} 0.4 mm at {loc} after {hours} h at {T} °C, close to the 0.5 mm limit."],
                        ["{loc} {defect} 0.4 mm, 기준 0.5 mm 근접."]),
        "pass": (["{hours} h at {T} °C, {defect} below 0.2 mm."], ["{T}°C {hours}h 열노화 합격, {defect} 0.2 mm 이하."]),
        "fix": [("material changed to PA66-GF30", "재질 PA66-GF30으로 변경", "PA66-GF30"),
                ("ribs added at {loc}", "{loc} 리브 추가", None)],
    },
    "cold_impact": {
        "loc": [("clip", "클립부"), ("corner", "코너부"), ("mounting tab", "장착 탭")],
        "defect": (["brittle fracture", "cracking", "breakage"], ["취성 파괴", "깨짐", "파손"]),
        "params": _p_cold,
        "fail": (["{Defect} at {loc} in the {T} °C impact test.", "Cold impact at {T} °C: {loc} {defect}."],
                 ["{T}°C 저온 충격 시험에서 {loc} {defect} 발생.", "저온({T}°C) 충격 후 {loc} {defect}."]),
        "conditional": (["Whitening at {loc} after {T} °C impact, no through-crack."],
                        ["{T}°C 충격 후 {loc} 백화 발생, 관통 균열 없음."]),
        "pass": (["No {defect} in the {T} °C impact test."], ["{T}°C 저온 충격 합격."]),
        "fix": [("changed to impact-modified grade", "내충격 그레이드로 변경", None),
                ("corner radius increased at {loc}", "{loc} R 확대", None)],
    },
    "vibration": {
        "loc": [("the joint", "체결부"), ("the mounting bolt", "장착 볼트")],
        "defect": (["loosening", "torque loss", "preload loss"], ["풀림", "토크 저하", "체결력 저하"]),
        "params": _p_vib,
        "fail": (["{Defect} after {hours} h vibration: residual torque {pct}% (min 80%).",
                  "Vibration {hours} h: {defect} at {loc}, {pct}% torque remaining."],
                 ["진동 {hours}h 후 {loc} {defect}, 잔류 토크 {pct}% (기준 80%).",
                  "진동시험 중 {loc} {defect} 발생 (잔류 {pct}%)."]),
        "conditional": (["Residual torque 82% after {hours} h, marginal."], ["진동 {hours}h 후 잔류 토크 82%, 경계 수준."]),
        "pass": (["No {defect} after {hours} h vibration."], ["진동 {hours}h 후 {defect} 없음."]),
        "fix": [("thread locker applied", "나사 고정제 적용", None),
                ("changed to flange bolt", "플랜지 볼트로 변경", None)],
    },
    "wear": {
        "loc": [("the sliding surface", "습동면"), ("the outer diameter", "외경부"), ("the bushing contact", "부싱 접촉부")],
        "defect": (["wear", "galling", "excessive play"], ["마모", "긁힘", "유격 증가"]),
        "params": _p_wear,
        "fail": (["{Defect} of {mm} mm at {loc} after {target}k cycles (limit 0.1 mm).",
                  "Wear test {target}k cycles: {loc} shows {defect}, {mm} mm."],
                 ["{target}k cycle 후 {loc} {defect} {mm} mm (기준 0.1 mm).", "마모시험 결과 {loc} {defect} {mm} mm 발생."]),
        "conditional": (["{target}k cycles, {defect} 0.09 mm at {loc}, near the limit."],
                        ["{target}k cycle 후 {loc} {defect} 0.09 mm, 기준 근접."]),
        "pass": (["{target}k cycles, {defect} below 0.05 mm."], ["{target}k cycle 마모 합격."]),
        "fix": [("induction hardening added", "고주파 열처리 추가", None),
                ("surface coating added at {loc}", "{loc} 표면 코팅 추가", None)],
    },
    "static": {
        "loc": [("the bend", "벤딩부"), ("the mounting hole", "장착 홀"), ("the web", "웨브")],
        "defect": (["yielding", "permanent deformation", "plastic deformation"], ["항복", "영구 변형", "소성 변형"]),
        "params": _p_static,
        "fail": (["{Defect} at {loc} at {at} kN (requirement {target} kN).",
                  "Static load test: {defect} at {loc} at {at} kN, below {target} kN."],
                 ["{at} kN에서 {loc} {defect} 발생 (요구 {target} kN).", "정하중 시험 {at} kN에서 {loc} {defect}."]),
        "conditional": (["Held {target} kN, permanent set 0.3 mm (limit 0.5 mm)."],
                        ["{target} kN 하중 후 영구 변형 0.3 mm (기준 0.5 mm 이내)."]),
        "pass": (["Held {target} kN without {defect}."], ["{target} kN 정하중 합격."]),
        "fix": [("material changed to SPFH590", "재질 SPFH590으로 변경", "SPFH590"),
                ("wall thickness 2.0 → 3.0 mm", "두께 2.0 → 3.0 mm 증대", None)],
    },
    "dimensional": {
        "loc": [("hole position", "홀 위치"), ("flatness", "평면도"), ("flange height", "플랜지 높이")],
        "defect": (["out of tolerance", "dimension deviation", "OOT"], ["공차 이탈", "치수 불량", "치수 편차"]),
        "params": _p_dim,
        "fail": (["{Loc} {mm} mm vs tolerance ±{tol} mm ({defect}).", "{Defect}: {loc} off by {mm} mm (±{tol} mm)."],
                 ["{loc} {mm} mm, 공차 ±{tol} mm {defect}.", "{loc} {defect} {mm} mm (공차 ±{tol} mm)."]),
        "conditional": (["{Loc} at the tolerance limit (±{tol} mm), process capability low."],
                        ["{loc} 공차 한계(±{tol} mm), 공정능력 부족."]),
        "pass": (["All dimensions within tolerance."], ["전 치수 공차 이내."]),
        "fix": [("tooling corrected", "금형 수정", None),
                ("tolerance relaxed to ±{tol2} mm", "공차 ±{tol2} mm로 완화", None)],
    },
    "nvh": {
        "loc": [("the cover edge", "커버 가장자리"), ("the clip", "클립부")],
        "defect": (["rattle", "buzz noise", "squeak"], ["래틀", "잡음", "떨림 소음"]),
        "params": _p_nvh,
        "fail": (["{Defect} at {loc} at {hz} Hz in rough-road simulation."], ["험로 가진 {hz} Hz에서 {loc} {defect} 발생."]),
        "conditional": (["Faint {defect} at {hz} Hz, subjective rating 6/10."], ["{hz} Hz에서 약한 {defect}, 주관평가 6/10."]),
        "pass": (["No {defect} in rough-road simulation."], ["험로 가진 시 {defect} 없음."]),
        "fix": [("damping pad added at {loc}", "{loc} 방진 패드 추가", None),
                ("extra clip added", "클립 추가", None)],
    },
}
ECO_FIX = (["{Fix} after {test} failure ({report}).", "{Fix} to address {defect} at {loc}.",
            "Rev {src} → {dst}: {fix}."],
           ["{test} 불합격({report}) 대책: {fix}.", "{loc} {defect} 대책으로 {fix}.", "Rev {src} → {dst}: {fix}."])
ECO_OTHER = [("Cost reduction: hole count reduced.", "원가절감: 홀 수 축소."),
             ("Supplier change for tooling capacity.", "금형 대응을 위한 협력사 변경."),
             ("Weight reduction: pocket added.", "경량화: 포켓 추가."),
             ("Drawing update: tolerance clarification.", "도면 갱신: 공차 명확화.")]
LABS = ("Lab A", "Lab B", "Supplier lab")


def p_fail(test, material, part):
    """Chance of a fail on one test, from material and shape."""
    thin = (part["fill_ratio"] or 1) < 0.3
    return {
        "fatigue": 0.4 if material == "AL6061-T6" and thin else 0.18 if material.startswith(("AL", "ADC")) else 0.06,
        "salt_spray": 0.25 if material in ("SPCC", "S45C", "SPFH590") else 0.12 if material.startswith(("AL", "ADC")) else 0.03,
        "heat_aging": {"PP-T20": 0.35, "ABS": 0.3, "PA66-GF30": 0.1}.get(material, 0.02),
        "cold_impact": {"PP-T20": 0.3, "ABS": 0.25, "PA66-GF30": 0.15}.get(material, 0.02),
        "vibration": 0.15, "wear": 0.15, "nvh": 0.15,
        "static": 0.12 if material in ("AL6061-T6", "SPCC") else 0.05,
        "dimensional": 0.25 if material == "ADC12" else 0.08,
    }[test]


P_CONDITIONAL = 0.06


# ---------------------------------------------------------------- classification

def classify(part, product_name, single):
    """Class from the product name of a single-part file, else from the measured geometry."""
    name = (product_name or "").strip()
    meaningful = bool(name) and not PLACEHOLDER.fullmatch(name) and len(re.findall(r"[A-Za-z]", name)) >= 3
    words = " ".join(re.split(r"[\s_\-]+|(?<=[a-z])(?=[A-Z])", name)).lower()
    if meaningful and single:
        for pattern, cls in KEYWORDS:
            if re.search(rf"\b({pattern})s?\b", words):
                return cls, clean(name)
    cls = geometry_class(part)
    label = f"{cls} {part['dim_1']:.0f}x{part['dim_2']:.0f}"
    if meaningful:
        label += f" ({clean(name)})"  # the file it came from, as assembly context
    return cls, label


def clean(name):
    return re.sub(r"\s+", " ", re.sub(r"[_]+", " ", name)).strip(" -")


def geometry_class(p):
    d1, d2, d3 = p["dim_1"], p["dim_2"], p["dim_3"]
    n = max(p["n_faces"], 1)
    share = lambda *cols: sum(p[c] or 0 for c in cols) / n
    round_ = share("n_cylinder", "n_cone", "n_torus")
    prismatic = share("n_plane", "n_cylinder")  # machined or sheet: flat faces plus holes
    fill = p["fill_ratio"] or 1
    if d1 > 4 * d2:  # long: a round cross-section is a pin, a rectangular one a profile
        if abs(d2 - d3) < 0.15 * d2 and round_ >= 0.3:
            return "Bolt" if p["n_bspline"] else "Pin" if d1 < 60 else "Shaft"
        return "Beam"
    if d3 < 0.25 * d1 and abs(d1 - d2) < 0.15 * d1 and round_ > 0.3:
        return "Washer" if d1 < 40 else "Flange"
    if share("n_bspline") > 0.3:
        return "Cover" if d3 < 0.25 * d1 or fill < 0.3 else "Housing"
    if d3 < 0.15 * d1 and prismatic > 0.7:
        return "Plate"
    if prismatic > 0.7:
        return "Bracket" if fill < 0.35 else "Block"
    return "Body"


# ---------------------------------------------------------------- generation

def weighted(r, options):
    keys = list(options)
    return r.choices(keys, weights=[options[k] for k in keys])[0]


def doc_id(prefix, key, used):
    n = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 900000 + 100000
    while f"{prefix}-{n}" in used:
        n = n % 999999 + 1
    used.add(f"{prefix}-{n}")
    return f"{prefix}-{n}"


def part_number(part_id, prefix, used):
    n = int(hashlib.sha1(part_id.encode()).hexdigest(), 16) % 90000 + 10000
    while f"{prefix}-{n}" in used:
        n = n % 99999 + 1
    used.add(f"{prefix}-{n}")
    return f"{prefix}-{n}"


def outcome(r, test, material, part):
    x = r.random()
    pf = p_fail(test, material, part)
    return "fail" if x < pf else "conditional" if x < pf + P_CONDITIONAL else "pass"


def gen_part(part, cls, label, part_no, story=None):
    """All facts of one part. `story` forces a planted history (see plant())."""
    r = random.Random(f"{SEED}:{part['part_id']}")
    prefix, team, mats, plan = CLASSES[cls]
    material = weighted(r, mats)
    n_revs = weighted(r, {1: 7, 2: 2.5, 3: 0.5})
    program = r.choice(PROGRAMS)
    tests = r.sample(plan, k=min(len(plan), r.randint(1, 3)))
    if story:
        program, material, n_revs, tests = story["program"], story.get("material", material), story["n_revs"], story["tests"]

    revs, reports, ecos = [], [], []
    day = date(2024, 1, 1) + timedelta(days=r.randint(0, 500))
    for i in range(n_revs):
        revs.append({"rev": "ABC"[i], "material": material, "date": day.isoformat()})
        if i == n_revs - 1:
            break
        # Half of all revisions exist to fix a failed test on the one before.
        fix_test = story["fail_test"] if story else r.choice(tests)
        if story or r.random() < 0.5:
            reports.append(report(r, part_no, revs[i], fix_test, "fail", day, forced=story))
            fixes = [f for f in PHRASES[fix_test]["fix"] if f[2] is None or material in UPGRADES[f[2]]]
            fix = r.choice(fixes[:2] if story else fixes)
            material = fix[2] or material
            ecos.append({"from": revs[i]["rev"], "to": "ABC"[i + 1], "kind": "fix", "test": fix_test,
                         "fix": fix, "ref": reports[-1], "date": (day + timedelta(days=r.randint(20, 60))).isoformat()})
        else:
            ecos.append({"from": revs[i]["rev"], "to": "ABC"[i + 1], "kind": "other", "other": r.choice(ECO_OTHER),
                         "date": (day + timedelta(days=r.randint(20, 60))).isoformat()})
        day += timedelta(days=r.randint(60, 240))

    latest = revs[-1]
    for test in tests:
        fixed = any(e["kind"] == "fix" and e["test"] == test for e in ecos)
        if fixed:
            result = "pass"
        elif story:  # planted: only the story's test may fail
            result = story["latest_result"] if test == story["fail_test"] else "pass"
        else:
            result = outcome(r, test, latest["material"], part)
        reports.append(report(r, part_no, latest, test, result, date.fromisoformat(latest["date"]), forced=story))

    open_fail = any(t["rev"] == latest["rev"] and t["result"] == "fail" for t in reports)
    for rv in revs[:-1]:
        rv["status"] = "Obsolete"
    # A part never ships with an open fail.
    latest["status"] = "In Work" if open_fail or (story and story.get("in_work")) or r.random() < 0.25 else "Released"
    for rv in revs:
        rv["mass_g"] = round(part["volume"] * MATERIALS[rv["material"]][0] / 1000, 1)

    supplier = None
    if latest["status"] == "Released" and r.random() > 0.05:  # a few released parts have no ERP record
        supplier = erp(r, cls, latest)
    return {"part_id": part["part_id"], "part_no": part_no, "class": cls, "name": label, "program": program,
            "owner_team": team, "revisions": revs, "tests": reports, "ecos": ecos, "supplier": supplier}


def report(r, part_no, rev, test, result, day, forced=None):
    ph = PHRASES[test]
    # Same targets on every revision of a part: a rev A fail at 584k can't sit beside a rev B pass of a 200k target.
    params = ph["params"](random.Random(f"{SEED}:{part_no}:{test}"))
    loc_i = 0 if forced and test == forced["fail_test"] else r.randrange(len(ph["loc"]))  # planted: the bolt hole
    return {"part_no": part_no, "rev": rev["rev"], "test": test, "result": result, "params": params,
            "loc": loc_i, "lab": r.choice(LABS), "ko": r.random() < KO_SHARE,
            "defect_i": r.randrange(len(ph["defect"][0])), "template_i": r.randrange(10),
            "date": (day + timedelta(days=r.randint(10, 50))).isoformat()}


def erp(r, cls, rev):
    m = rev["material"]
    specialty = ("plastic" if m in PLASTICS else "casting" if m == "ADC12" else
                 "fastener" if cls in ("Bolt", "Nut", "Washer") else
                 "sheet" if cls in ("Bracket", "Plate") or m in ("SPCC", "SPFH590") else "machining")
    base = {"fastener": 1, "sheet": 3, "casting": 5, "machining": 7, "plastic": 9}[specialty]
    mass_kg = rev["mass_g"] / 1000 if "mass_g" in rev else 0
    return {"supplier": f"Supplier {base + r.randint(0, 1):02d}", "specialty": specialty,
            "unit_cost_krw": int(round(mass_kg * MATERIALS[m][1] * 1.3 + r.randint(200, 3000), -1)),
            "lead_time_wk": r.randint(2, 12)}


def render(rep):
    """A test report's note, from its facts."""
    ph = PHRASES[rep["test"]]
    lang = 1 if rep["ko"] else 0
    pool = ph[rep["result"]][lang]
    defect = ph["defect"][lang][rep["defect_i"] % len(ph["defect"][lang])]
    loc = ph["loc"][rep["loc"]][lang]
    fields = {**rep["params"], "defect": defect, "Defect": defect[0].upper() + defect[1:],
              "loc": loc, "Loc": loc[0].upper() + loc[1:]}
    return pool[rep["template_i"] % len(pool)].format(**fields)


def render_eco(eco, rng):
    ko = rng.random() < KO_SHARE
    if eco["kind"] == "other":
        return eco["other"][1 if ko else 0], ko
    ph, ref = PHRASES[eco["test"]], eco["ref"]
    lang = 1 if ko else 0
    loc = ph["loc"][ref["loc"]][lang]
    fix = eco["fix"][lang].format(loc=loc, tol2=ref["params"].get("tol2", ""))
    fields = {"fix": fix, "Fix": fix[0].upper() + fix[1:], "loc": loc, "src": eco["from"], "dst": eco["to"],
              "test": TESTS[eco["test"]][lang], "defect": ph["defect"][lang][ref["defect_i"] % 3],
              "report": ref["report_id"]}
    pool = ECO_FIX[lang]
    return pool[rng.randrange(len(pool))].format(**fields), ko


def plant(parts):
    """Stories around my_bracket's nearest neighbors in step 4's image search."""
    from src.pipeline import index
    qc = index.client()
    neighbors = [p.payload["part_id"] for p in index.similar(qc, "my_bracket_s00", "views", 10)
                 if p.payload["file_id"] != "my_bracket"]
    seen_files, picked = set(), []
    for pid in neighbors:  # one per file: sibling solids would tell the same story twice
        f = pid.rsplit("_s", 1)[0]
        if f not in seen_files and pid in parts:
            seen_files.add(f)
            picked.append(pid)
    fixed = {"program": "P1", "material": "AL6061-T6", "n_revs": 2, "tests": ["fatigue", "static"],
             "fail_test": "fatigue", "latest_result": "pass"}
    stories = {
        "my_bracket_s00": {"program": "P1", "material": "AL6061-T6", "n_revs": 1, "tests": [], "in_work": True,
                           "fail_test": None, "latest_result": None, "class": "Bracket",
                           "name": "L-Bracket, engine mount"},
        picked[0]: fixed,
        picked[1]: fixed,
        # same crack, still open, in the other program: invisible to a P1-only user
        picked[2]: {"program": "P2", "material": "AL6061-T6", "n_revs": 1, "tests": ["fatigue"],
                    "fail_test": "fatigue", "latest_result": "fail", "in_work": True},
    }
    for pid, st in stories.items():
        st["class"] = st.get("class", "Bracket")
    return stories, picked[:3]


def generate():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT p.*, f.product_name,
                  (SELECT count(*) FROM parts q WHERE q.file_id = p.file_id AND q.status = 'ok'
                   AND q.dup_of IS NULL) AS n_unique
           FROM embeddings e JOIN parts p USING (part_id) JOIN files f USING (file_id)
           ORDER BY p.part_id""").fetchall()
    parts = {r["part_id"]: dict(r) for r in rows}
    classified = {pid: classify(p, p["product_name"], p["n_unique"] == 1) for pid, p in parts.items()}
    stories, neighbors = plant(parts)

    used, facts = set(), []
    for pid, p in parts.items():
        cls, label = classified[pid]
        story = stories.get(pid)
        if story:
            cls, label = story["class"], story.get("name", f"{story['class']} {p['dim_1']:.0f}x{p['dim_2']:.0f}")
        facts.append(gen_part(p, cls, label, part_number(pid, CLASSES[cls][0], used), story))

    # Opaque ids derived from what each document is about, so adding parts never renumbers the others.
    used = set()
    reports = [t for f in facts for t in f["tests"]]
    for t in reports:
        t["report_id"] = doc_id("TR", f"{t['part_no']}:{t['rev']}:{t['test']}", used)
    for f in facts:
        for e in f["ecos"]:
            e["eco_id"] = doc_id("ECO", f"{f['part_no']}:{e['from']}", used)
            e["text"], e["ko"] = render_eco(e, random.Random(f"{SEED}:{e['eco_id']}"))
    for t in reports:
        t["note"] = render(t)

    write_tables(conn, facts)
    PLM.mkdir(parents=True, exist_ok=True)
    scenarios = {"my_bracket_neighbors_fixed_P1": neighbors[:2], "my_bracket_neighbor_open_P2": neighbors[2]}
    for f in facts:
        for e in f["ecos"]:
            if "ref" in e:
                e["ref"] = e["ref"]["report_id"]  # facts.json: link by id, not a nested copy
    (PLM / "facts.json").write_text(json.dumps(
        {"seed": SEED, "users": USERS, "scenarios": scenarios, "parts": facts}, ensure_ascii=False, indent=1))
    summarize(conn, facts, scenarios)
    conn.close()


SCHEMA = """
DROP TABLE IF EXISTS plm_items;
DROP TABLE IF EXISTS erp_suppliers;
DROP TABLE IF EXISTS doc_test_reports;
DROP TABLE IF EXISTS doc_change_notices;
CREATE TABLE plm_items (
    part_no     TEXT NOT NULL,
    rev         TEXT NOT NULL,
    part_id     TEXT,                -- CAD geometry (parts table); only the latest rev has it
    name        TEXT,
    part_class  TEXT,
    material    TEXT,
    mass_g      REAL,
    status      TEXT,                -- In Work | Released | Obsolete
    program     TEXT,                -- access control: users see only their programs
    owner_team  TEXT,
    rev_date    TEXT,
    PRIMARY KEY (part_no, rev)
);
CREATE TABLE erp_suppliers (
    part_no       TEXT PRIMARY KEY,
    supplier      TEXT,
    unit_cost_krw INTEGER,
    lead_time_wk  INTEGER
);
CREATE TABLE doc_test_reports (
    report_id  TEXT PRIMARY KEY,
    part_no    TEXT NOT NULL,
    rev        TEXT NOT NULL,
    test_type  TEXT,                 -- fatigue | salt_spray | heat_aging | cold_impact | vibration | wear | static | dimensional | nvh
    result     TEXT,                 -- pass | conditional | fail
    test_date  TEXT,
    lab        TEXT,
    lang       TEXT,                 -- en | ko
    note       TEXT                  -- free text: what happened, where, how much
);
CREATE TABLE doc_change_notices (
    eco_id     TEXT PRIMARY KEY,
    part_no    TEXT NOT NULL,
    from_rev   TEXT,
    to_rev     TEXT,
    eco_date   TEXT,
    lang       TEXT,
    reason     TEXT                  -- free text; may cite the failed report
);
"""


def write_tables(conn, facts):
    conn.executescript(SCHEMA)
    for f in facts:
        latest = f["revisions"][-1]["rev"]
        for rv in f["revisions"]:
            conn.execute("INSERT INTO plm_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         (f["part_no"], rv["rev"], f["part_id"] if rv["rev"] == latest else None, f["name"],
                          f["class"], rv["material"], rv["mass_g"], rv["status"], f["program"], f["owner_team"],
                          rv["date"]))
        if f["supplier"]:
            s = f["supplier"]
            conn.execute("INSERT INTO erp_suppliers VALUES (?, ?, ?, ?)",
                         (f["part_no"], s["supplier"], s["unit_cost_krw"], s["lead_time_wk"]))
        for t in f["tests"]:
            conn.execute("INSERT INTO doc_test_reports VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         (t["report_id"], t["part_no"], t["rev"], t["test"], t["result"], t["date"], t["lab"],
                          "ko" if t["ko"] else "en", t["note"]))
        for e in f["ecos"]:
            conn.execute("INSERT INTO doc_change_notices VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (e["eco_id"], f["part_no"], e["from"], e["to"], e["date"], "ko" if e["ko"] else "en",
                          e["text"]))
    conn.commit()


def summarize(conn, facts, scenarios):
    q = lambda sql: conn.execute(sql).fetchall()
    print(f"{len(facts)} parts -> {q('SELECT count(*) FROM plm_items')[0][0]} revisions, "
          f"{q('SELECT count(*) FROM erp_suppliers')[0][0]} ERP records, "
          f"{q('SELECT count(*) FROM doc_test_reports')[0][0]} test reports, "
          f"{q('SELECT count(*) FROM doc_change_notices')[0][0]} change notices")
    print("classes:", ", ".join(f"{c} {n}" for c, n in q(
        "SELECT part_class, count(*) FROM plm_items WHERE part_id IS NOT NULL GROUP BY 1 ORDER BY 2 DESC")))
    print("results:", ", ".join(f"{r} {n / sum(x[1] for x in q('SELECT result, count(*) FROM doc_test_reports GROUP BY 1')):.0%}"
                                for r, n in q("SELECT result, count(*) FROM doc_test_reports GROUP BY 1")))
    print("Korean notes:", q("SELECT round(avg(lang = 'ko') * 100) FROM doc_test_reports")[0][0], "%")
    print("status (latest):", ", ".join(f"{s} {n}" for s, n in q(
        "SELECT status, count(*) FROM plm_items WHERE part_id IS NOT NULL GROUP BY 1")))
    print("scenarios:", json.dumps(scenarios))
    print(f"facts: {PLM / 'facts.json'}")


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    generate()


if __name__ == "__main__":
    main()
