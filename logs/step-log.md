# Step Log

One short entry per plan step. Paths are current (`src/pipeline/` steps 1–5, `src/qa/` step 6, `src/ui/` step 7).

## Step 1 — Setup + read CAD (2026-10-25)

**Built:** conda env `cadai` (`environment.yml`); ABC chunk 0000 (10,000 STEP), 100 files in `data/raw`; `extract.py`: sha256 skip → CadQuery load → one row per solid (faces, edges, volume, area, bbox, validity, in-file duplicate flag) → `db/parts.sqlite` (`files`, `parts`).

**Broke → fix**
- NYU archive direct link returned HTTP 500 (curl saved a 0-byte file) → found the chunk by searching the archive.
- A file loaded "fine" with an infinite bbox (non-solid junk) → per-solid extraction + bbox < 100 m check.
- 1 file surfaces only (no solid); 3 solids invalid but sensible → kept, flagged `is_valid=0` (all meshed fine in step 2).

**Numbers:** 99/100 loaded; 229 solids, 186 unique; 0.57 s/file avg, 18.6 s max; rerun skips all by hash.

**Decisions:** one record per solid (search works on single shapes); flag duplicates, don't delete; `parts` is the registry every later stage reads, the vector index is rebuildable from it.

**Learned:** STEP keeps exact geometry (B-rep), not design intent (feature tree, parameters). Invalid solids come from kernel-to-kernel translation → at scale: validation, repair, failed queue.

## Step 2 — Own part + meshes (2026-10-26)

**Built:** parametric L-bracket in FreeCAD → `my_bracket.step` (volume matches hand calc, 13,967.3 mm³); `extract.py` v2 (sorted dims, inertia, fill ratio, face types, STEP lineage); `mesh.py`: size-scaled tolerance → coarse `.glb` (view) + medium `.glb`/`.stl`.

**Broke → fix:** 29/187 meshes not watertight (zero-area triangles at sphere poles, cone tips) → removed; cadquery `tessellate()` tolerance is per edge → call the OpenCascade mesher directly. (2026-10-28) glbs written without normals → `include_normals=True`, 3,252 glbs rewritten in place, no `MESH_VER` bump.

| level | tolerance | median tris | volume err | DINOv2 cos vs medium |
|---|---|---|---|---|
| coarse | 0.5%, 29° | 530 | 0.37% | 0.998 |
| medium | 0.1%, 11° | 1,292 | 0.06% | — |
| fine | 0.02%, 6° | 2,612 | 0.015% | 0.9996 |

**Decision:** medium = render source; fine dropped (71% of storage, no visible change).

**Learned:** tolerance must scale with size (0.1 mm – 9.7 m parts) and needs a triangle cap (an M3 thread hit 331k).

## Step 3 — Render (2026-10-26)

**Built:** `render.py`: 6 orthographic views, 512 px, unit sphere, same lights/material everywhere.

**Broke → fix:** pyrender can't run headless on macOS → VTK offscreen, one reused window.

**Decision:** orthographic: no perspective size cues, only shape differs.

## Step 4 — Similar-part search (2026-10-26)

**Built:** `canonical.py` (pose before render), `embed.py` (DINOv2, CLIP), `evaluate.py` (12 configs), `index.py` (Qdrant). Report: `docs/step4-search-configuration-report.pdf`.

**Broke → fix:** inertia (PCA) axes fail on CAD (114/187 undefined) → axes from flat-face normals; symmetric parts flipped → sign by center-of-mass offset; pip torch OMP crash → conda-forge torch; DINOv2 preprocessing cropped edges → resize to 224.

**Numbers:** rotated-copy top-1: 23–70% without canonical pose, 100% with.

**Decisions:** DINOv2 over CLIP (better matches, wider score spread 0.6–0.9); store mean + 6-view MaxSim.

**Open:** labeled similar pairs needed to compare models fairly; mirror parts share the duplicate fingerprint.

## Step 5 — Pipeline (2026-10-27)

**Built:** `python -m src.pipeline run|status|retry`: poll `data/raw` → extract → mesh → render → embed → index. SQLite `jobs` = queue + history (`UNIQUE file, sha, pipeline_ver`); per-stage skip on own + upstream versions; retries 30 s / 2 min → dead → `data/failed/`; 10 min lease for crashes. Qdrant moved to Docker. Doc: `docs/pipeline.pdf`.

**Broke → fix:** `INSERT OR IGNORE` burned AUTOINCREMENT ids → check first; Ctrl-C left jobs `running` → release in the worker; mesh change didn't re-render → skip key includes upstream versions.

**Numbers:** drop → searchable ~16 s; 500 files (one per ABC source doc): 492 done, 10 dead, 1,625 parts, 22.5 min, 1.6 GB; unchanged rerun 5 s. Search: shape good, size ignored, scores not comparable across queries.

**Decisions:** SQLite queue over Redis/RQ (no server, no forking with VTK/MPS); polling over OS events.

**Mini vs production**

| Mini | Production |
|---|---|
| Poll `data/raw` | PLM change feed (release events or "released since cursor") + nightly reconciliation |
| `file_id` + sha | Item id + revision (+ sha); new revision replaces old |
| Deleted file stays searchable | Lifecycle "obsolete" removes it from the index |
| Native → STEP not needed | PLM translation service (e.g. Teamcenter Dispatcher), licensed translators |
| SQLite jobs, 1 worker | Postgres `SKIP LOCKED` / Kafka / SQS; CPU and GPU workers per stage |
| `data/failed` | DLQ + alerts + owner; failure-rate dashboard |
| Local folders | On-prem object storage |
| Version bump → backfill | Same, in a low-priority lane |
| No access control | PLM permissions in payload, filtered at query |

**Open:** deleted files stay searchable; batch `render` skips the triangle gate.

## Step 6 — Fake PLM + Q&A agent (2026-10-27)

**Built:** `qa/plm.py` (seeded, facts first: 2,214 revisions, 953 suppliers, 3,261 test reports 39% Korean, 589 change notices, `data/plm/facts.json` = truth); `qa/textindex.py` (join → `part_cards`; 5,475 chunks; BGE-M3 dense + sparse + BM25 on Kiwi tokens, RRF); `qa/tools.py` (`lookup_part`, `query_parts`, `search_notes`, `similar_parts`, permissions inside each); `qa/agent.py` (Opus 5.5, effort medium, ≤ 4 tool rounds then tools off, id post-check). Doc: `docs/query-flow.pdf` (rebuilt 2026-10-28 via `docs/src/build_docs.py`).

**Broke → fix:** fail rate 26% → 19%; "Body" fallback 29% → 12%; ids renumbered on regen → content-hash ids; index 8 chunks/s → length-sorted batches 2.5×; strict schemas rejected nullable enums → `anyOf`; BGE-M3 lazy-loaded in first search (22.7 s) → preload; a listed part number didn't exist → id post-check (unknown id → one rewrite).

**Numbers:** Korean count question matched `facts.json` exactly; normal 14 vs admin 33 parts, no leak; 10–27 s/question, ~$0.07.

**Decisions:** text serves the CAD side ("has a part like mine failed?" = `similar_parts` → history); model never writes SQL; hidden = missing; facts as filters (search can't see negation); Korean via Kiwi + query expansion, not translation.

**Changed 2026-10-28:**
- Multi-turn: `ask_async(history=, seen=)` resends the full transcript (tool results included, cached) so follow-ups reuse data; cited ids carry over; repair keeps only the corrected answer; refused turn dropped; old tool results stubbed past 200k chars. 3-turn check: follow-up answered with 0 tools, 2.3 s.
- Attachments: `ask_async(attachments=, shapes=)`. STEP solids get handles `NEW-1` that `similar_parts` accepts; images, PDF, txt/md/csv go to Claude as blocks; "attached files are context, not evidence".
- Small talk ("hi") already works: no tools, 7–17 s.

**→ step 8:** gold from `facts.json`; wording unlike the templates (incl. KO); score routing / retrieval / answer / citations / leaks separately; Opus medium vs low vs Sonnet (~$4 per 60 questions); count repairs; ~5 two-turn cases (pointer back, "show the rest", EN→KO, needs new tools, user switch).

**Open:** new pipeline parts get no PLM records until `plm` + `textindex build` rerun; text index has no incremental update.

## Step 7 — UI (2026-10-27/28)

**Built:** `streamlit run src/ui/app.py --server.fileWatcherType none` (README). Pages: Search (hybrid text, filters), Part (renders, PLM card, tests/ECOs/revisions/geometry, similar parts, 3D), Ask (multi-turn chat, live tool progress, cited parts, attachments). Upload page merged into Ask (STEP card: render, size, top-5 similar, "reuse it?" at ≥ 0.97). `cadworker.py` runs extract/mesh/render in a child process.

**Look:** bilingual, Korean first (`widgets.bi`, `bi_lines`, `bi_plain`); black sidebar with the pages; black accent only; action fields with a thick black border; part numbers as black buttons; chat with your turn right in a black bubble, no avatars; results as colourless chips (✕ fail solid, ! conditional outline, ✓ pass grey); robot-arm logo (`static/icon.svg`); loading screen. Theme in `.streamlit/config.toml`, extra CSS in `widgets.STYLE`.

**Broke → fix:** VTK window aborts off the main thread → CAD child process; fp16 attention fails on M1 → eager; new session 4.9 s (file watcher) → watcher off, 0.6 s; Part page stale URL/box → URL is the source of truth; old answer faded under the next → `st.rerun()`; edited agent not picked up → restart needed for code outside `views/`; splash blank for 5 s → drawn before heavy imports.

**Numbers:** cold start ~20 s (models 12–14 s); Search 0.5 s; Part < 0.5 s; STEP attach ~1 s; Ask 5–18 s (tools ~0.2 s, the rest is the LLM). Memory: app 3.15 → 4.2 GB with use, CAD worker 0.3 GB, Qdrant 0.4 GB.

**Decisions:** memory first on 8 GB (BGE-M3 fp16 queries, 3D off by default); an upload is a question, not an ingest; all model calls on one tool thread; images to Claude only (no photo → render search).

**Open:** full 2-minute demo run moved to after step 8 (evaluation); latency table → step 8; answer streaming; does memory plateau; a real photo / large PDF / Word files untested; tile "similarity" wraps in narrow tiles.
