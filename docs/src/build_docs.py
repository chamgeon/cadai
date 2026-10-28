"""Build docs/pipeline.pdf and docs/query-flow.pdf (ReportLab), figures from figures.py.

    python docs/src/build_docs.py [pipeline] [query]     (both when none is named)

The figures are drawn into a temporary folder; only the PDFs are kept.
"""
import sys
import tempfile
from pathlib import Path

import figures

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

OUT = Path(__file__).resolve().parents[1]  # docs/
FIG = None  # set per run
F = "/System/Library/Fonts/Supplemental/"
pdfmetrics.registerFont(TTFont("Arial", F + "Arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", F + "Arial Bold.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", F + "Arial Italic.ttf"))
pdfmetrics.registerFont(TTFont("AU", F + "Arial Unicode.ttf"))  # Hangul
pdfmetrics.registerFont(TTFont("Mono", F + "Courier New.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-Bold")

INK = colors.HexColor("#1d2b44")
MUTED = colors.HexColor("#5b6577")
RULE = colors.HexColor("#c9ced6")
HEAD = colors.HexColor("#eaf1fb")
W = letter[0] - 1.3 * inch

S = {
    "title": ParagraphStyle("title", fontName="Arial-Bold", fontSize=19, leading=23, textColor=INK, spaceAfter=2),
    "sub": ParagraphStyle("sub", fontName="Arial", fontSize=9.5, leading=13, textColor=MUTED, spaceAfter=8),
    "h1": ParagraphStyle("h1", fontName="Arial-Bold", fontSize=13, leading=16, textColor=INK, spaceBefore=12,
                         spaceAfter=5),
    "h2": ParagraphStyle("h2", fontName="Arial-Bold", fontSize=10.5, leading=13, textColor=INK, spaceBefore=8,
                         spaceAfter=3),
    "body": ParagraphStyle("body", fontName="Arial", fontSize=9.2, leading=13, textColor=INK, spaceAfter=5,
                           alignment=TA_LEFT),
    "cap": ParagraphStyle("cap", fontName="Arial-Italic", fontSize=8.2, leading=11, textColor=MUTED, spaceBefore=4,
                          spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="Arial", fontSize=7.9, leading=10.2, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Arial-Bold", fontSize=7.9, leading=10.2, textColor=INK),
}


def p(text, style="body"):
    return Paragraph(text, S[style])


def code(text):
    return f'<font name="Mono" size="8.4">{text}</font>'


def ko(text):
    return f'<font name="AU">{text}</font>'


def bullets(items, style="body"):
    return ListFlowable([ListItem(p(t, style), leftIndent=12, value="•") for t in items], bulletType="bullet",
                        start="•", leftIndent=12, bulletFontName="Arial", bulletFontSize=8)


def numbered(items):
    return ListFlowable([ListItem(p(t), leftIndent=14) for t in items], bulletType="1", leftIndent=14,
                        bulletFontName="Arial", bulletFontSize=8.5)


def table(rows, widths, header=True):
    data = [[p(c, "cellb" if header and i == 0 else "cell") for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1 if header else 0)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("BACKGROUND", (0, 0), (-1, 0), HEAD if header else colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def figure(name, caption, max_h):
    from PIL import Image as PImage

    path = FIG / f"{name}.png"
    w, h = PImage.open(path).size
    width = W
    height = width * h / w
    if height > max_h:
        height, width = max_h, max_h * w / h
    return KeepTogether([Image(str(path), width=width, height=height), p(caption, "cap")])


def build(path, doc_title, story):
    def footer(canv, doc):
        canv.saveState()
        canv.setFont("Arial", 7.5)
        canv.setFillColor(MUTED)
        canv.drawString(0.65 * inch, 0.45 * inch, f"cadai · {doc_title}")
        canv.drawRightString(letter[0] - 0.65 * inch, 0.45 * inch, f"{doc.page}")
        canv.restoreState()

    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.65 * inch, rightMargin=0.65 * inch,
                            topMargin=0.55 * inch, bottomMargin=0.7 * inch, title=doc_title, author="cadai",
                            subject=doc_title)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


# ======================================================================= pipeline.pdf
def pipeline_doc():
    s = [
        p("CAD Ingestion Pipeline", "title"),
        p("Steps 1–5 · " + code("src/pipeline/") + " · " + code("python -m src.pipeline run") + " · 2026-10-27",
          "sub"),
        figure("pipeline_overview",
               "Figure 1. One STEP file through the pipeline. Red: validation gates (a failing part is rejected, the "
               "job continues). Green: canonicalization, inside render, before any view.", 8.2 * inch),
        PageBreak(),
        p("Stages", "h1"),
        table([
            ["Stage", "Output", "Skipped when its table has"],
            ["1 Extract", "parts rows: volume, bbox, inertia, face types, duplicate flag", "sha256 + EXTRACT_VER"],
            ["2 Mesh", "coarse.glb, medium.glb + .stl", "sha256 + MESH_VER"],
            ["3 Render", "view_0..5.png, canonical pose", "sha256 + MESH_VER + RENDER_VER"],
            ["4–5 Embed, index", "6 × 768 DINOv2 → Qdrant 'parts' (mean + MaxSim)", "model + sha256 + MESH/RENDER_VER"],
            ["6 Cleanup", "drops what a revision removed", "—"],
        ], [0.2, 0.48, 0.32]),
        p("Skip keys include upstream versions, so a retry resumes at the failed stage and a version bump "
          "backfills only the changed stage and those after it.", "cap"),

        p("Validation and canonicalization", "h1"),
        bullets([
            "<b>Gate 1</b> (extract, per solid): volume > 0, largest bbox side < 100 m. No solids or unloadable STEP "
            "→ job dead.",
            "<b>Gate 2</b> (mesh, per part): 4 – 500,000 triangles on the medium mesh. Caught a 2-triangle part.",
            "<b>Gate 3</b> (render, per part): all six views must render.",
            "<b>Canonicalize</b> (render, before views): axes from flat-face normals (inertia if none), longest → X, "
            "never mirrored; center + unit scale. Random-rotation search hits: 23–70% without, 100% with.",
        ]),

        p("Jobs", "h1"),
        figure("job_states", "Figure 2. A job is one file at one sha256 and one pipeline version; "
               "UNIQUE (file, sha256, pipeline_ver) makes rescans no-ops.", 2.6 * inch),

        p("Measured (500 files)", "h1"),
        bullets([
            "492 done, 10 dead; 1,625 searchable parts (67% of new solids were in-file duplicates).",
            "Drop → searchable ~16 s; 396 new files in 22.5 min; unchanged rerun 5 s.",
            "1.6 GB total: mesh 699 MB, Qdrant 151 MB, renders 88 MB for 686 MB of STEP.",
        ]),
        p("Known gaps", "h1"),
        bullets([
            "Deleted files stay searchable; new parts get no PLM records until step 6 data is rebuilt.",
            "Batch " + code("render") + " skips gate 2; one worker on SQLite.",
        ]),
    ]
    build(OUT / "pipeline.pdf", "CAD Ingestion Pipeline", s)


# ======================================================================= query-flow.pdf
def query_doc():
    s = [
        p("How a Question Is Answered", "title"),
        p("Steps 6–7 · " + code("src/qa/") + ", " + code("src/ui/views/ask.py") + " · "
          + code("streamlit run src/ui/app.py") + " · 2026-10-28", "sub"),
        figure("query_overview",
               "Figure 1. Claude chooses the tools; the tools decide what the user may see; the answer cites every "
               "fact.", 6.6 * inch),
        p("Prompt rules", "h1"),
        p("Hard facts go into filters (search can't see negation); bilingual synonyms as keywords; cite only ids a "
          "tool returned; name the revision a result belongs to; never guess. Attached files are context, not "
          "evidence; follow-ups may reuse earlier tool results."),
        PageBreak(),
        p("Tools", "h1"),
        table([
            ["Tool", "Model controls", "Permission"],
            ["lookup_part", "part number", "hidden = missing → 'not found'"],
            ["query_parts", "parameters only; the SQL is fixed", "program condition added by code"],
            ["search_notes", "query, keywords, exact filters", "filter inside each prefetch"],
            ["similar_parts", "part number or attached shape (NEW-1), size ratio", "only visible parts are ranked"],
        ], [0.18, 0.42, 0.4]),

        p("Attachments and conversation", "h1"),
        table([
            ["Attached", "Done in the app", "Claude gets"],
            ["STEP", "CAD worker: extract, mesh, canonical pose, 6 renders; DINOv2. Not stored. Each solid → NEW-n; "
                     "its vectors stay in the session for similar_parts", "handle + size per solid, iso render (≤ 4)"],
            ["Image", "scaled to ≤ 1568 px, ≤ 5 MB (JPEG fallback)", "image block; no image search"],
            ["PDF, txt / md / csv", "—", "document block"],
        ], [0.16, 0.54, 0.3]),
        bullets([
            "Each question resends the whole transcript, tool results included (cached), so a follow-up "
            "(\"of those three…\") can answer without tools. Citable ids carry over.",
            "An id repair keeps only the corrected answer; a refused or failed question is dropped. Past 200k chars "
            "of old tool results, those before the last 2 questions become a stub.",
            "New chat, an example, or a user switch starts a new conversation: nothing seen as admin stays.",
        ]),

        p("Data", "h1"),
        p("Four scattered tables (plm_items, erp_suppliers, doc_test_reports, doc_change_notices; 39% of notes "
          "Korean) are joined into one part card per part. 5,475 chunks: cards + one per note; each note embedded "
          "with a header (part, class, material, test, result) that is never cited."),

        p("Measured", "h1"),
        bullets([
            "Permissions: 14 vs 33 parts for normal vs admin; a direct question about a P2 part leaked nothing.",
            "Ask 5–18 s per question, of which tools ~0.2 s: the latency is the LLM. A greeting: no tools, 7–17 s.",
            "Follow-up from history: 0 tools, 2.3 s. STEP attachment ~1 s (extract 0.2, mesh 0.05, render 0.2, "
            "embed 0.35 s); the sample bracket finds itself (BR-66254) at 1.0.",
            "~$0.07 per question; one 60-question evaluation pass ≈ $4; concurrent runs via " + code("ask_many") + ".",
        ]),
        PageBreak(),
        p("Retrieval", "h1"),
        figure("hybrid_search", "Figure 2. Text retrieval (search_notes, and the Search page without the LLM): "
               "keyword and meaning rankings, both restricted before ranking, merged by RRF.", 3.3 * inch),
        table([
            ["Test", "Result"],
            [ko("크랙 난 브라켓") + " ('cracked bracket')", "3 of top 5 said 'no crack'; fixed by result = fail filter"],
            ["English query, BM25 / dense", "BM25 matched wrong literal words; dense ignored the material"],
            ["BM25 + bilingual expansion", "4 of 4 correct, all from Korean notes"],
        ], [0.4, 0.6]),
        Spacer(1, 10),
        figure("step_search", "Figure 3. STEP retrieval (similar_parts, and the attachment card): one image signal, "
               "restricted before ranking, then filtered by size.", 3.1 * inch),

    ]
    build(OUT / "query-flow.pdf", "How a Question Is Answered", s)


if __name__ == "__main__":
    docs = {"pipeline": pipeline_doc, "query": query_doc}
    for name in sys.argv[1:] or list(docs):
        with tempfile.TemporaryDirectory() as tmp:
            FIG = Path(tmp)
            figures.draw(tmp, name)
            docs[name]()
        print("written", OUT / ("pipeline.pdf" if name == "pipeline" else "query-flow.pdf"))
