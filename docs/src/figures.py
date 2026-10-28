"""Diagrams for docs/pipeline.pdf and docs/query-flow.pdf (drawn in inches, saved as 300 dpi PNG).

Used by build_docs.py; draw(out_dir, names) writes the named figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Arial", "Arial Unicode MS"]  # Arial Unicode MS: Hangul and arrows
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = None  # set by draw()
INK, SUB, LINE = "#1d2b44", "#3a4760", "#5b6577"
BLUE = ("#eaf1fb", "#3b6fb6")
GRAY = ("#f3f4f6", "#9aa1ad")
RED = ("#fdecec", "#c0392b")
GREEN = ("#e8f6ee", "#23935a")
AMBER = ("#fff6e0", "#c28a00")
PURPLE = ("#f1ecfb", "#6c4bb6")


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, body="", colors=BLUE, tsize=8.2, bsize=6.9, align="center", lw=1.1, dash=False):
    fc, ec = colors
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec=ec, lw=lw,
                                ls="--" if dash else "-"))
    tx = x + w / 2 if align == "center" else x + 0.08
    if title:
        ax.text(tx, y + h - 0.07, title, ha=align, va="top", fontsize=tsize, weight="bold", color=INK)
    if body:
        ax.text(tx, y + h - (0.27 if title else 0.08), body, ha=align, va="top", fontsize=bsize, color=SUB,
                linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, label=None, color=LINE, dash=False, rad=0.0, lsize=6.4, loff=(0, 0.05), ha="center"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, color=color, lw=1.0,
                                 ls="--" if dash else "-", connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        ax.text((x1 + x2) / 2 + loff[0], (y1 + y2) / 2 + loff[1], label, ha=ha, va="bottom", fontsize=lsize,
                color=color, style="italic")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ pipeline: overview
def pipeline_overview():
    fig, ax = canvas(7.3, 9.25)  # drawn in inches: the PDF places it at full width
    # intake
    box(ax, 0.05, 8.35, 1.45, 0.8, "data/raw", "STEP files dropped in\nor revised")
    box(ax, 1.75, 8.35, 2.45, 0.8, "Watcher  (poll every 10 s)",
        "waits until size is stable, sha256\n(cached by size + mtime), enqueue")
    box(ax, 4.45, 8.35, 2.8, 0.8, "jobs table  (SQLite queue)",
        "UNIQUE (file, sha256, pipeline_ver):\nINSERT OR IGNORE, so rescans are no-ops")
    arrow(ax, 1.5, 8.75, 1.75, 8.75)
    arrow(ax, 4.2, 8.75, 4.45, 8.75)

    # worker container
    ax.add_patch(FancyBboxPatch((0.05, 1.45), 7.2, 6.67, boxstyle="round,pad=0,rounding_size=0.08",
                                fc="#fafbfc", ec="#7d8796", lw=1.0))
    arrow(ax, 5.85, 8.35, 5.85, 8.12)
    ax.text(5.75, 8.235, "claim one job (atomic UPDATE … RETURNING), lease 10 min", fontsize=5.8, color=LINE,
            style="italic", va="center", ha="right")
    ax.text(0.17, 7.98, "Worker: one job = one STEP file, its solids one by one", fontsize=8.4, weight="bold",
            color=INK, va="top")
    ax.text(0.17, 7.78, "each stage is skipped when its table already holds output for this sha256 and its own + "
            "upstream versions", fontsize=6.3, color=SUB, va="top", style="italic")
    for x, label in ((0.2, "Stage"), (2.75, "Validation / canonicalization"), (5.45, "Writes")):
        ax.text(x, 7.5, label.upper(), fontsize=6.2, color="#7d8796", weight="bold", va="top")

    rows = [  # (height, stage title, stage body, detail boxes, writes)
        (0.78, "1  Extract", "load STEP → one part per solid\nmeasure volume, bbox, inertia,\nface types; flag in-file copies",
         [("GATE 1  per solid", "volume > 0, bbox < 100 m\n(no solids → job dead)", RED)],
         "SQLite: files, parts"),
        (0.78, "2  Mesh", "tessellate; tolerance scales with\nthe bbox diagonal\ncoarse .glb · medium .glb + .stl",
         [("GATE 2  per part", "medium mesh has\n4 – 500,000 triangles", RED)],
         "data/mesh/{part}/\nSQLite: meshes"),
        (1.36, "3  Render", "from medium.stl\n\n6 orthographic views,\nsame lights, 512 px, white bg",
         [("CANONICALIZE  (before any view)", "axes from flat-face normals,\ninertia if none; longest → X,\ncenter + scale to unit sphere", GREEN),
          ("GATE 3  per part", "rendering must succeed", RED)],
         "data/renders/{part}/\nview_0..5.png\nSQLite: renders"),
        (0.6, "4  Embed", "DINOv2 per view → 6 × 768,\nL2-normalized", [], "(in memory)"),
        (0.78, "5  Index", "Qdrant 'parts': mean vector +\n6-view MaxSim; payload: size,\nface types, sha256, job_id",
         [], "Qdrant: parts\nSQLite: embeddings"),
        (0.6, "6  Cleanup", "drop points + folders of solids a\nrevision removed or now rejected", [], "Qdrant, data/*"),
    ]
    y = 7.3
    prev = None
    for h, title, body, details, writes in rows:
        y -= h
        box(ax, 0.2, y, 2.35, h, title, body, BLUE, align="left")
        if prev is not None:
            arrow(ax, 1.375, prev, 1.375, y + h)
        dy = y + h
        for dt, db, col in details:
            dh = (h - 0.08 * (len(details) - 1)) / len(details)
            dy -= dh
            box(ax, 2.75, dy, 2.5, dh, dt, db, col, align="left", tsize=7.2, bsize=6.4)
            arrow(ax, 2.55, dy + dh / 2, 2.75, dy + dh / 2, color=col[1])
            dy -= 0.08
        box(ax, 5.45, y, 1.65, h, "", writes, GRAY, align="left", bsize=6.4)
        prev = y
        y -= 0.13
    ax.text(2.78, 2.72, "a part failing a gate is rejected (log in data/failed);\nthe job continues with the "
            "other solids", fontsize=6.0, color=RED[1], style="italic", va="top")

    # failure handling
    box(ax, 0.05, 0.05, 4.55, 1.25, "When a job fails",
        "STEP won't load / no solids   →  dead at once (retry can't help)\n"
        "other error (Qdrant down, DB locked)   →  requeued: 30 s, 2 min, then dead\n"
        "worker crashed   →  lease expires → requeued; 3rd crash → dead\n"
        "Ctrl-C   →  handed back, attempt not counted\n"
        "retry resumes at the failed stage: finished stages are skipped", AMBER, align="left", bsize=6.5)
    box(ax, 4.8, 0.05, 2.45, 1.25, "data/failed  (dead letter)",
        "{file}.{stage}.log  tracebacks\n{part}.{stage}.log  rejected parts\nSTEP file of every dead job\n"
        "`pipeline retry` moves it back", RED, align="left", bsize=6.5)
    arrow(ax, 3.9, 1.45, 3.9, 1.3, color="#7d8796")
    arrow(ax, 6.0, 1.45, 6.0, 1.3, color=RED[1])
    save(fig, "pipeline_overview")


# ------------------------------------------------------------------ pipeline: job states
def job_states():
    fig, ax = canvas(7.3, 2.75)
    S = {"queued": (0.35, 1.75), "running": (2.95, 1.75), "done": (5.6, 1.75), "dead": (5.6, 0.25),
         "superseded": (0.35, 0.25)}
    col = {"queued": BLUE, "running": PURPLE, "done": GREEN, "dead": RED, "superseded": GRAY}
    W, H = 1.35, 0.5
    for s, (x, y) in S.items():
        box(ax, x, y, W, H, "", "", col[s])
        ax.text(x + W / 2, y + H / 2, s, ha="center", va="center", fontsize=8.6, weight="bold", color=INK)
    c = lambda s, dx=0.5, dy=0.5: (S[s][0] + W * dx, S[s][1] + H * dy)
    arrow(ax, *c("queued", 1, 0.7), *c("running", 0, 0.7), "claim")
    arrow(ax, *c("running", 1, 0.5), *c("done", 0, 0.5), "all stages finished")
    arrow(ax, *c("running", 0, 0.3), *c("queued", 1, 0.3))
    ax.text(2.33, 1.68, "transient error (retry after 30 s, 2 min)\nlease expired · Ctrl-C", ha="center", va="top",
            fontsize=6.2, color=LINE, style="italic")
    arrow(ax, *c("running", 0.7, 0), *c("dead", 0.2, 1), color=RED[1])
    ax.text(4.95, 1.25, "STEP unloadable\n3rd failure · 3rd crash", ha="left", va="center", fontsize=6.2,
            color=RED[1], style="italic")
    arrow(ax, *c("queued", 0.5, 0), *c("superseded", 0.5, 1))
    ax.text(1.1, 1.25, "newer revision\nof the file queued", ha="left", va="center", fontsize=6.2, color=LINE,
            style="italic")
    arrow(ax, *c("dead", 0, 0.5), *c("queued", 0.85, 0), dash=True, color="#7d8796", rad=-0.32)
    ax.text(3.6, 0.22, "`pipeline retry`: STEP moved back, attempts reset", ha="center", va="center",
            fontsize=6.2, color="#7d8796", style="italic")
    ax.text(7.25, 2.7, "rows are never deleted: the table is also the history", ha="right", va="top", fontsize=6.4,
            color=SUB)
    save(fig, "job_states")


# ------------------------------------------------------------------ query flow: overview
def query_overview():
    fig, ax = canvas(7.3, 6.2)
    box(ax, 0.05, 4.7, 2.3, 1.45, "Ask page  (src/ui)",
        "question in any language, as user\n`normal` (P1) or `admin` (P1, P2)\n"
        "+ attachments: STEP → CAD worker\n   → NEW-1 (6 views, size), ~1 s;\n   image · PDF · txt / md / csv\n"
        "+ the conversation so far", GRAY, bsize=6.3)
    box(ax, 2.65, 4.25, 4.6, 1.9, "Agent loop  (src/qa/agent.py)",
        "Claude Opus 5.5, effort medium, adaptive thinking\n"
        "system prompt + 4 tool schemas: identical for every user → cached\n"
        "messages = transcript so far + question (+ attachment blocks)\n"
        "rounds 1–4: reads all results so far, then calls tools (parallel\n"
        "                    within a round) or answers; a greeting needs no tool\n"
        "round 5: tools switched off → must answer\n"
        "id post-check: an id no tool returned → one rewrite, tools off\n"
        "returns the transcript + citable ids for the next question\n"
        "refusal → server-side fallback model (fallbacks=\"default\")", PURPLE, align="left", bsize=6.4)
    arrow(ax, 2.35, 5.5, 2.65, 5.5)
    box(ax, 0.05, 3.5, 2.3, 1.05, "Answer",
        "every fact cited: BR-19198 [TR-381830]\ncited parts: part numbers in the text\n"
        "→ lookup_part (permission) → tiles\n+ trace (tools, seconds, tokens)", GREEN, bsize=6.3)
    arrow(ax, 2.65, 4.4, 2.35, 4.4)

    ax.add_patch(FancyBboxPatch((0.05, 1.6), 7.2, 1.8, boxstyle="round,pad=0,rounding_size=0.08", fc="#fafbfc",
                                ec="#7d8796", lw=1.0))
    ax.text(0.17, 3.31, "Tools  (src/qa/tools.py): the permission filter lives here, inside every tool",
            fontsize=8.3, weight="bold", color=INK, va="top")
    ax.text(0.17, 3.1, "programs come from the user, never from the model · hidden and missing parts both answer "
            "\"not found\" · DB opened read-only", fontsize=6.3, color=SUB, va="top", style="italic")
    arrow(ax, 4.2, 4.25, 4.2, 3.4, "tool_use", loff=(0.08, -0.06), ha="left")
    arrow(ax, 5.6, 3.4, 5.6, 4.25, "tool_result (JSON)", loff=(0.08, -0.06), ha="left")
    tools = [
        ("lookup_part", "one part's card + all its\ntest reports and change\nnotices\nuse: a named part"),
        ("query_parts", "count / list / group by\nfields + measured size;\nfixed SQL, model gives\nonly parameters"),
        ("search_notes", "hybrid search over cards,\ntest notes, change notices;\nexact filters + keywords\n"
                         "use: meaning in text"),
        ("similar_parts", "6-view MaxSim (DINOv2) for\na part number or an\nattached shape (NEW-1);\n"
                          "visible parts only; size ratio"),
    ]
    for i, (t, b) in enumerate(tools):
        box(ax, 0.2 + i * 1.77, 1.75, 1.62, 1.1, t, b, BLUE, tsize=7.8, bsize=6.3)

    box(ax, 0.05, 0.05, 3.05, 0.9, "SQLite  (read-only for tools)",
        "part_cards: the join of plm_items, erp_suppliers,\ndoc_test_reports, doc_change_notices + step 1\n"
        "geometry; plus parts (measured geometry)", GRAY, align="left", bsize=6.3)
    box(ax, 3.25, 0.05, 2.15, 0.9, "Qdrant 'docs'",
        "5,475 chunks: 1 card per part,\n1 per test note, 1 per change\nnotice; dense + bm25 + m3_sparse", GRAY,
        bsize=6.3)
    box(ax, 5.55, 0.05, 1.7, 0.9, "Qdrant 'parts'", "1,625 parts, DINOv2\nmean + 6-view MaxSim\n"
        "(steps 1–5); uploads\nare never added", GRAY, bsize=6.3)
    for x1, x2 in ((1.0, 1.0), (2.75, 2.6), (4.5, 4.3), (6.3, 6.4)):
        arrow(ax, x1, 1.75, x2, 0.95, color="#7d8796", dash=True)
    save(fig, "query_overview")


# ------------------------------------------------------------------ query flow: hybrid search
def hybrid_search():
    fig, ax = canvas(7.3, 3.55)
    box(ax, 0.05, 1.45, 1.55, 1.0, "search_notes call",
        "query: \"볼트 체결부 균열\"\nkeywords (from Claude):\ncrack, 균열, 크랙,\nbolt hole, AL6061-T6", GRAY, bsize=6.2)
    box(ax, 1.9, 2.25, 2.0, 1.15, "Keyword side",
        "Kiwi splits particles (균열이 → 균열),\nEnglish stemmed; query + keywords\n→ BM25 sparse vector\n(IDF applied by Qdrant)",
        BLUE, bsize=6.2)
    box(ax, 1.9, 0.45, 2.0, 1.15, "Meaning side",
        "BGE-M3 dense, 1024-d,\nmultilingual: an English query\nfinds a Korean note\n(query text only; fp16)", BLUE,
        bsize=6.2)
    arrow(ax, 1.6, 2.15, 1.9, 2.8)
    arrow(ax, 1.6, 1.75, 1.9, 1.05)
    box(ax, 4.2, 0.45, 1.45, 2.95, "Filter in\neach prefetch",
        "\nprogram ∈ user's\n(always)\n\n+ exact filters\nfrom the model:\nkind, result,\ntest_type, material,\npart_class, part_nos",
        RED, bsize=6.2)
    arrow(ax, 3.9, 2.8, 4.2, 2.8, "top 50", lsize=5.8)
    arrow(ax, 3.9, 1.05, 4.2, 1.05, "top 50", lsize=5.8)
    box(ax, 5.95, 1.45, 1.3, 1.0, "RRF fusion", "score = Σ 1/(60+rank)\n→ top k chunks", GREEN, bsize=6.2)
    arrow(ax, 5.65, 1.95, 5.95, 1.95)
    ax.text(0.05, 0.3, "chunk = header (embedded, never cited):  BR-19198 rev A | Bracket | AL6061-T6 | durability "
            "(fatigue) | fail", fontsize=6.3, color=SUB, va="top")
    ax.text(0.05, 0.12, "             + text (shown and cited):      Crack at bolt hole after 509k cycles (target 1000k).  "
            "[TR-381830]", fontsize=6.3, color=SUB, va="top")
    save(fig, "hybrid_search")


# ------------------------------------------------------------------ query flow: STEP retrieval
def step_search():
    fig, ax = canvas(7.3, 3.3)
    box(ax, 0.05, 1.85, 1.55, 1.3, "similar_parts(\"BR-66254\")",
        "an indexed part: its 6\nview vectors are read\nfrom Qdrant 'parts'\n(made by the pipeline)",
        GRAY, tsize=7.4, bsize=6.2)
    box(ax, 0.05, 0.35, 1.55, 1.3, "Attached STEP  (NEW-1)",
        "Ask page, CAD worker:\nextract → mesh → canonical\npose → 6 renders → DINOv2\n~1 s, nothing stored",
        AMBER, tsize=7.4, bsize=6.2)
    box(ax, 1.85, 1.15, 1.1, 1.3, "6 view vectors",
        "6 × 768,\nL2-normalized\nfront, back, left,\nright, top, iso", BLUE, tsize=7.4, bsize=6.2)
    arrow(ax, 1.6, 2.5, 1.85, 2.1)
    arrow(ax, 1.6, 1.0, 1.85, 1.5)
    box(ax, 3.2, 0.35, 1.1, 2.8, "Filter in\nthe query",
        "\npart_id ∈ parts\nthe user may see\n(always)\n\nnot the query\npart's own file\n(its other solids)",
        RED, tsize=7.4, bsize=6.2)
    arrow(ax, 2.95, 1.8, 3.2, 1.8)
    box(ax, 4.55, 1.85, 1.25, 1.3, "MaxSim",
        "each query view takes\nits best-matching view\nof the candidate;\nsum / 6 → 0..1\n(equivalent poses\nstill match)",
        BLUE, tsize=7.4, bsize=6.2)
    box(ax, 4.55, 0.35, 1.25, 1.3, "Size filter",
        "optional max_size_ratio:\nlongest side within ×r\n(images ignore scale);\nfetch 4 × k, then filter",
        AMBER, tsize=7.4, bsize=6.2)
    arrow(ax, 4.3, 2.5, 4.55, 2.5)
    arrow(ax, 5.175, 1.85, 5.175, 1.65)
    box(ax, 6.05, 0.35, 1.2, 2.8, "Top k parts",
        "similarity, size ratio,\ntest result counts\n\n→ the agent, or\nthe Ask page card\n\n≥ 0.97:\n'existing part,\nreuse it?'",
        GREEN, tsize=7.4, bsize=6.2)
    arrow(ax, 5.8, 1.0, 6.05, 1.0)
    ax.text(0.05, 0.2, "One signal, not hybrid: image similarity only. Step 1's geometry numbers (volume, face types, "
            "fill ratio) are in the payload; fusing them by RRF is a step 8 experiment.", fontsize=6.3, color=SUB,
            va="top")
    save(fig, "step_search")


FIGURES = {"pipeline": (pipeline_overview, job_states), "query": (query_overview, hybrid_search, step_search)}


def draw(out_dir, doc):
    global OUT
    OUT = out_dir
    for fn in FIGURES[doc]:
        fn()
