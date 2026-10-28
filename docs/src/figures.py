"""Diagrams for the PDFs in docs/ (drawn in inches, saved as 300 dpi PNG).

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
        "Claude Opus 5.5, effort low, adaptive thinking\n"
        "system prompt + 5 tool schemas: identical for every user → cached\n"
        "messages = transcript so far + question (+ attachment blocks)\n"
        "rounds 1–4: reads all results so far, then calls tools (parallel\n"
        "                    within a round) or answers; a greeting needs no tool\n"
        "round 5: tools off, told so → answers, says what is missing\n"
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
        ("lookup_part", "one part's card + all\nits test reports and\nchange notices\nuse: a named part"),
        ("query_parts", "count / list / group\nby fields + size;\nfixed SQL, model\ngives parameters"),
        ("fix_history", "every failed test →\nits change notice →\nthe retest; counts\n(added after step 8)"),
        ("search_notes", "hybrid search over\ncards, notes, change\nnotices; filters +\nkeywords (≤ 20)"),
        ("similar_parts", "6-view MaxSim for a\npart or an attached\nshape (NEW-1);\nvisible only; size"),
    ]
    for i, (t, b) in enumerate(tools):
        box(ax, 0.2 + i * 1.42, 1.75, 1.32, 1.1, t, b, BLUE, tsize=7.6, bsize=6.1)

    box(ax, 0.05, 0.05, 3.05, 0.9, "SQLite  (read-only for tools)",
        "part_cards: the join of plm_items, erp_suppliers,\ndoc_test_reports, doc_change_notices + step 1\n"
        "geometry; plus parts (measured geometry)", GRAY, align="left", bsize=6.3)
    box(ax, 3.25, 0.05, 2.15, 0.9, "Qdrant 'docs'",
        "5,475 chunks: 1 card per part,\n1 per test note, 1 per change\nnotice; dense + bm25 + m3_sparse", GRAY,
        bsize=6.3)
    box(ax, 5.55, 0.05, 1.7, 0.9, "Qdrant 'parts'", "1,625 parts, DINOv2\nmean + 6-view MaxSim\n"
        "(steps 1–5); uploads\nare never added", GRAY, bsize=6.3)
    for x1, x2 in ((0.86, 0.86), (2.28, 2.1), (3.7, 2.9), (5.12, 4.3), (6.54, 6.4)):
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
            "fill ratio) are in the payload; re-ranking with them did not help (step 8).", fontsize=6.3, color=SUB,
            va="top")
    save(fig, "step_search")


# ------------------------------------------------------------------ evaluation: what was measured where
def eval_overview():
    fig, ax = canvas(7.3, 3.95)
    ax.set_ylim(0.5, 4.45)  # drawn on 0..4.45; the strip below the lowest boxes is empty
    # the system, left to right: CAD side on top, text side below, the app on the right
    box(ax, 0.05, 2.95, 1.0, 0.75, "STEP files", "500 ABC files\n+ my bracket", GRAY, bsize=6.3)
    box(ax, 1.3, 2.95, 1.9, 0.75, "Pipeline  (steps 1–5)", "extract · mesh · render ·\nDINOv2 embed · index", BLUE,
        bsize=6.3)
    box(ax, 3.45, 2.95, 1.15, 0.75, "Qdrant 'parts'", "1,625 parts,\n6 view vectors", GRAY, bsize=6.3)
    box(ax, 4.85, 2.95, 1.15, 0.75, "Shape search", "MaxSim,\nsize filter", BLUE, bsize=6.3)
    box(ax, 0.05, 1.5, 1.0, 0.75, "Fake PLM", "facts.json →\nrecords, notes", GRAY, bsize=6.3)
    box(ax, 1.3, 1.5, 1.9, 0.75, "Text index  (step 6)", "part cards + 5,475 chunks,\nBM25 + BGE-M3", GRAY, bsize=6.3)
    box(ax, 3.45, 1.5, 2.55, 0.75, "Q&A agent  (step 6)", "Claude Opus 5.5 (effort low) + 4 tools;\n"
        "permissions inside every tool", PURPLE, bsize=6.3)
    box(ax, 6.25, 1.5, 1.0, 2.2, "App  (step 7)", "Streamlit\n\nSearch\nPart\nAsk", GRAY, bsize=6.3)
    for x1, x2, y in ((1.05, 1.3, 3.325), (3.2, 3.45, 3.325), (4.6, 4.85, 3.325), (1.05, 1.3, 1.875),
                      (3.2, 3.45, 1.875), (6.0, 6.25, 3.325), (6.0, 6.25, 1.875)):
        arrow(ax, x1, y, x2, y)
    arrow(ax, 5.425, 2.95, 5.425, 2.25, "similar_parts", loff=(0.06, -0.08), ha="left", lsize=5.8)

    # what step 8 measured, and the headline
    box(ax, 0.05, 3.85, 3.15, 0.55, "A  Pipeline", "98.0% of files load · 99.94% of parts searchable ·\n"
        "0.91 s per part · output 1.49× the STEP size", AMBER, align="left", tsize=7.6, bsize=6.2)
    box(ax, 3.45, 3.85, 2.55, 0.55, "B  Search quality", "precision@5 0.35 on 20 hand-graded queries;\n"
        "6 other setups tried, none better; reusable: 0", GREEN, align="left", tsize=7.6, bsize=6.2)
    box(ax, 3.45, 0.55, 2.55, 0.8, "C  Q&A", "58 questions: 97–98% correct · 10 hard: 8–10\n"
        "0 leaks, 0 invented ids in 204 answers\n3 setups: Opus medium / low, Sonnet medium", PURPLE,
        align="left", tsize=7.6, bsize=6.2)
    box(ax, 6.25, 0.55, 1.0, 0.8, "D  System", "start 22–25 s\nsearch 0.04 s\nask 7–17 s\n4.3 GB, flat",
        AMBER, align="left", tsize=7.6, bsize=6.0)
    for x, y1, y2 in ((2.25, 3.85, 3.7), (4.0, 3.85, 3.7), (5.4, 3.85, 3.7), (4.7, 1.35, 1.5), (6.75, 1.35, 1.5)):
        arrow(ax, x, y1, x, y2, color="#7d8796", dash=True)
    ax.text(0.05, 1.3, "How it was graded", fontsize=7.4, weight="bold", color=INK, va="top")
    ax.text(0.05, 1.1, "A, D  measured: the pipeline's own tables, timers,\n       memory footprint (not RSS)\n"
            "B  graded by hand, blind, 3 rounds (repeat rate 84–91%)\n"
            "C  checked by code against facts.json; an LLM judge\n     only for 'why' answers and one yes/no",
            fontsize=6.2, color=SUB, va="top", linespacing=1.35)
    save(fig, "eval_overview")



# ------------------------------------------------------------------ requirements: production architecture
def production():
    fig, ax = canvas(7.3, 5.35)
    ax.set_ylim(0.25, 5.6)  # drawn on 0..5.6; nothing below the demo note

    def tag(x, y, ids):
        ax.text(x, y, ids, fontsize=6.0, color=RED[1], weight="bold", va="top", ha="right")

    # query side
    ax.text(0.05, 5.55, "QUERY SIDE", fontsize=6.4, color="#7d8796", weight="bold", va="top")
    box(ax, 0.05, 4.45, 1.25, 0.95, "Engineers", "≈1,000 active\n~100 sessions at peak\n3–6 asks in flight", GRAY,
        bsize=6.1)
    box(ax, 1.5, 4.45, 1.2, 0.95, "SSO + load\nbalancer", "\n\nsticky sessions\n(Streamlit websockets)", GRAY,
        bsize=6.1)
    tag(2.68, 4.53, "S2 P5")
    box(ax, 2.9, 4.45, 1.35, 0.95, "App replicas", "UI only, stateless,\n< 2 GB each; no\nmodels inside", BLUE,
        bsize=6.1)
    tag(4.23, 4.53, "O1 O2")
    box(ax, 4.45, 4.45, 1.5, 0.95, "Search + agent API", "tools enforce PLM\npermissions; audit\nlog of every answer",
        PURPLE, bsize=6.1)
    tag(5.93, 4.53, "S1 S4 S5")
    box(ax, 6.15, 4.45, 1.1, 0.95, "LLM gateway", "private endpoint,\nquotas, rate limit,\ncost per user", PURPLE,
        bsize=6.1)
    tag(7.23, 4.53, "S3 O4")
    for x1, x2 in ((1.3, 1.5), (2.7, 2.9), (4.25, 4.45), (5.95, 6.15)):
        arrow(ax, x1, 4.92, x2, 4.92)

    # shared services and stores
    ax.text(0.05, 4.2, "SHARED SERVICES AND STORES  (read by the query side, written by ingest)", fontsize=6.4,
            color="#7d8796", weight="bold", va="top")
    box(ax, 0.05, 3.05, 1.6, 0.95, "Embedding service", "GPU: DINOv2 (views),\nBGE-M3 (text), batched;\n"
        "uploads and ingest", BLUE, bsize=6.1)
    tag(1.63, 3.13, "P3 O2")
    box(ax, 1.85, 3.05, 1.75, 0.95, "Vector DB", "parts: 2M × 6 × 768\n≈ 37 GB raw, ~9 GB int8\n"
        "docs: notes, cards", GRAY, bsize=6.1)
    tag(3.58, 3.13, "P1 P2")
    box(ax, 3.8, 3.05, 1.75, 0.95, "Postgres", "registry, jobs, PLM\nmirror, permissions,\naudit log", GRAY,
        bsize=6.1)
    tag(5.53, 3.13, "S1 S4")
    box(ax, 5.75, 3.05, 1.5, 0.95, "Object storage", "meshes, renders;\n70% alert, 3-2-1\nbackups", GRAY, bsize=6.1)
    tag(7.23, 3.13, "O3")
    arrow(ax, 5.2, 4.45, 5.2, 4.0, "reads", color="#7d8796", dash=True, loff=(0.06, -0.09), ha="left", lsize=5.8)

    # ingest side
    ax.text(0.05, 2.8, "INGEST SIDE  (workers embed through the shared service, write every store)", fontsize=6.4,
            color="#7d8796", weight="bold", va="top")
    box(ax, 0.05, 1.55, 1.3, 1.05, "PLM + CAD vault", "CATIA parts and\nassemblies; release,\nlifecycle, "
        "permissions", GRAY, bsize=6.1)
    box(ax, 1.55, 1.55, 1.25, 1.05, "Change feed", "release events +\nnightly reconcile;\nobsolete → removed", AMBER,
        bsize=6.1)
    tag(2.78, 1.63, "I1 I2 I5")
    box(ax, 3.0, 1.55, 1.3, 1.05, "Translation", "CATIA → STEP AP242,\nassemblies resolved;\nlicensed seats",
        AMBER, bsize=6.1)
    tag(4.28, 1.63, "F1 F5")
    box(ax, 4.5, 1.55, 1.2, 1.05, "Queue", "jobs by revision;\nDLQ + alert;\nbackfill lane", AMBER, bsize=6.1)
    tag(5.68, 1.63, "I6 I7")
    box(ax, 5.9, 1.55, 1.35, 1.05, "CPU workers", "extract, mesh,\nrender (scaled out);\nembed via service",
        AMBER, bsize=6.1)
    tag(7.23, 1.63, "I3 I4")
    for x1, x2 in ((1.35, 1.55), (2.8, 3.0), (4.3, 4.5), (5.7, 5.9)):
        arrow(ax, x1, 2.07, x2, 2.07)
    arrow(ax, 6.6, 2.6, 6.6, 3.05, "writes", color="#7d8796", dash=True, loff=(0.06, -0.09), ha="left", lsize=5.8)
    ax.text(0.05, 1.32, "permissions and PLM records are mirrored into Postgres, so tools filter before retrieval",
            fontsize=6.2, color=SUB, style="italic", va="top")

    # what the demo used instead
    ax.text(0.05, 1.0, "In the demo:", fontsize=7.0, weight="bold", color=INK, va="top")
    ax.text(0.05, 0.8, "one laptop · folder polling instead of the change feed · SQLite instead of Postgres · local "
            "folders instead of object storage · models inside the app (4.3 GB)\nuser picked in a sidebar instead of SSO "
            "· public Claude API instead of a private endpoint · fixed program table instead of PLM permissions",
            fontsize=6.2, color=SUB, va="top", linespacing=1.4)
    ax.text(7.25, 1.0, "red: requirement ids (table below)", fontsize=6.0, color=RED[1], ha="right", va="top")
    save(fig, "production")


# ------------------------------------------------------------------ requirements: what each format keeps
def catia_loss():
    fig, ax = canvas(7.3, 2.75)
    ax.set_ylim(0.45, 3.2)  # drawn on 0..3.2; nothing below the bracket note
    stages = [  # title, format note, kept, lost, colors
        ("CATIA V5", "CATPart · CATProduct", "feature tree, parameters,\nconstraints, design intent;\n"
         "PMI; links to child parts;\nPLM revision + permissions", "", BLUE),
        ("STEP AP242", "licensed translator", "exact B-rep geometry;\nPMI (if exported);\nassembly structure",
         "feature tree, parameters,\nconstraints, design intent", AMBER),
        ("Mesh", "size-scaled tolerance", "triangles: medium\n0.06% volume error;\ncap 500k per part",
         "exact surfaces,\nface identities", AMBER),
        ("Renders + vectors", "6 views → 6 × 768", "shape, seen from\n6 fixed directions", "size (scaled to fit),\n"
         "hidden faces, material", AMBER),
    ]
    for i, (title, note, kept, lost, col) in enumerate(stages):
        x = 0.05 + i * 1.85
        box(ax, x, 1.0, 1.6, 2.15, title, "", col, tsize=8.2)
        ax.text(x + 0.8, 2.86, note, ha="center", va="top", fontsize=6.2, color=SUB, style="italic")
        ax.text(x + 0.1, 2.62, "keeps", fontsize=6.2, color=GREEN[1], weight="bold", va="top")
        ax.text(x + 0.1, 2.46, kept, fontsize=6.2, color=SUB, va="top", linespacing=1.3)
        if lost:
            ax.text(x + 0.1, 1.72, "loses", fontsize=6.2, color=RED[1], weight="bold", va="top")
            ax.text(x + 0.1, 1.56, lost, fontsize=6.2, color=SUB, va="top", linespacing=1.3)
        if i:
            arrow(ax, x - 0.25, 2.07, x, 2.07)
    ax.text(0.05, 0.85, "My bracket (FreeCAD, parametric): exported to STEP AP242 and read back, the feature tree and its "
            "parameters were gone,\nonly the geometry was left, and it was exact: volume 13,967.3 mm³, as calculated by "
            "hand. So the pipeline keeps PLM metadata\nnext to the STEP: design intent is not in the file the AI reads.",
            fontsize=6.4, color=INK, va="top", linespacing=1.4)
    save(fig, "catia_loss")


FIGURES = {"pipeline": (pipeline_overview, job_states), "query": (query_overview, hybrid_search, step_search),
           "eval": (eval_overview,), "requirements": (production, catia_loss)}


def draw(out_dir, doc):
    global OUT
    OUT = out_dir
    for fn in FIGURES[doc]:
        fn()
