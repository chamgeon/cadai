# Step Log

One entry per plan step; the date is when the step was done. Module paths are the current ones (`src/pipeline/`, `src/qa/` since step 6).

## Step 1 — Setup + read CAD (2026-10-25)

**Goal:** set up the environment, data, and project skeleton; load every STEP file, extract properties, store in SQLite, list failures and why.

**What was built**
- Conda env `cadai` (Python 3.11, CadQuery 2.8, trimesh, pyrender, qdrant-client, streamlit); `environment.yml` to recreate it.
- Downloaded ABC chunk 0000 (STEP, ~1.6 GB compressed, 10,000 models) and copied 100 files into `data/raw`.
- Project layout: `data/{raw,step_ok,failed,mesh,renders}`, `db/`, `src/`, `logs/`; `src/config.py` holds all paths; git repo with `data/` ignored. (`step_ok` dropped in step 5.)
- Installed FreeCAD and opened sample STEP files to see what the data looks like.
- `src/pipeline/extract.py`: for each STEP file in `data/raw`
  1. SHA-256 hash → skip file if unchanged since last run.
  2. Load with CadQuery, split into individual solids.
  3. Per solid: faces, edges, volume, area, bounding box, OpenCascade validity.
  4. Validate (volume > 0, bounding box < 100 m).
  5. Flag repeated copies of the same solid within a file (`dup_of`).
  6. Save to `db/parts.sqlite`; write full traceback to `data/failed/<file>.extract.log` on load failure.
- Two tables: `files` (one row per STEP file: hash, load status, error, solid count, load time, tool version, timestamp) and `parts` (one row per solid: geometry properties, validation status, duplicate flag).

**What broke (and why)**
- First download produced a 0-byte `.7z`: the NYU archive's file endpoint returned HTTP 500 (server-side). curl saved the empty error response as the file.
- 1 file has surfaces only, no solid → can't be a part.
- 1 file loaded **without any error** but its whole-file bounding box was 2×10¹⁰⁰ (infinite). Cause: non-solid junk in the file (likely a construction plane or stray curve). "Loaded" ≠ "valid."
- 3 solids fail OpenCascade's validity check but have sensible volumes (likely translation tolerance gaps from Parasolid → STEP).

**Fixes / workarounds**
- Download: a GitHub discussion on the ABC repo reported that the network (direct link) download never works. Found the chunk by searching the NYU archive and downloaded it from there instead.
- Added a bounding-box plausibility check (caught the infinite-bbox file).
- Switching to per-solid extraction dropped the non-solid junk, so that file's 39 solids now load correctly. The bbox check stays as a safety net.
- Invalid-but-sensible solids are kept with `is_valid=0`; step 2 (meshing) will show whether they still mesh. Repair option for later: OpenCascade `ShapeFix`.

**Numbers measured**
- 100 files → 99 loaded, 1 failed.
- 229 parts (solids) total, 186 unique, 43 duplicate copies within files.
- 31 of 100 files contain more than one solid; one file has 39.
- Load time: 0.57 s average per file, 18.6 s max (parts with thousands of faces). ~1 min for 100 files.
- Re-run with no changes: all 100 files skipped by hash.

**Decisions made and trade-offs**
- **One ABC chunk, start with 100 files.** Plan needs 300–2,000 parts; 100 files keeps reruns fast while building. Scale to ~1,000 once meshing and rendering are stable (hash skip means only new files get processed).
- **One record per solid, not per file.** Similar-part search works on single shapes; engineers reuse a handle or bracket, not a whole assembly. Trade-off: a solid is *usually* one physical part, but that's a modeling convention (designers can fuse several parts into one body, or split one part into several bodies).
- **Flag in-file duplicates instead of deleting them.** Keeps the data complete while letting search show each shape once (otherwise 4 identical screws would fill the top-5). Signature = face count, edge count, volume, area (rotation-independent).
- **Keep `data/raw` untouched; record failures in the DB + log files.** Moving files belongs to the step 5 pipeline.
- **Keep invalid solids** (flagged) rather than dropping them, until meshing shows whether they're usable.
- **Role of this data:** the `parts` table is the **part registry**, not the search index. Every later stage reads from it (which parts to mesh/render, size-scaled mesh tolerance, joining PLM records, UI detail pages, step 8 metrics). The embedding index (step 4) is separate and rebuildable from it. Size/volume can also filter or re-rank search results to fix image search's "same shape, different size" weakness.

**What was learned**
- **Design intent** = the "how and why" of a model (parameters, constraints, feature order, rules like "hole stays centered"). Native CAD keeps it; STEP keeps only the final exact geometry. In FreeCAD, an imported STEP shows bare bodies with no feature tree.
- **B-rep (boundary representation):** a solid described by its skin with exact math. Geometry (planes, cylinders, NURBS surfaces, curves) + topology (faces, edges, vertices → shells → solid). `n_faces` / `n_edges` are B-rep counts. A mesh (step 2) approximates the same surfaces with triangles.
- **Solid:** one closed, watertight body. A door may be 3 solids (frame, glass, handle) or 1 if fused; it depends on how it was modeled.
- **Valid solid:** closed (every edge shared by exactly 2 faces), consistent outward orientation, no self-intersection, edges lie on their surfaces within tolerance, no degenerate edges/faces. Invalid solids mostly come from translation between CAD kernels with different tolerances.
- **At scale:** CATIA already stores one physical part per `CATPart`, assembled in a `CATProduct`, so splitting assemblies matters less there, but CATIA → STEP translation will produce invalid solids, so the pipeline needs validation, automatic repair, and a failed queue.
- ABC formats: only `step` is needed; meshes and renders will be generated by my own pipeline.

**Open questions**
- Should multi-body parts (one physical part modeled as several solids) be merged back into one part? How could that be detected?
- How many of the 3 invalid solids will mesh cleanly in step 2? (Answered in step 2: all, after the degenerate-triangle fix.)
- How does real CATIA → STEP AP242 export handle assemblies with missing referenced parts?

## Step 2 — Own part + meshes (2026-10-26)

**Goal:** model one part by hand; mesh every part at a few tolerances and compare.

**What was built**
- Parametric L-bracket in FreeCAD (spreadsheet parameters) → `data/raw/my_bracket.step`. Extracted volume matches the hand calculation exactly (13,967.3 mm³).
- `extract.py` v2: rotation-independent size (sorted dims, principal moments), shape (fill ratio, sphericity), face-type counts, STEP header lineage. Bumping `EXTRACT_VER` reprocesses all files; old rows get the new columns added.
- `mesh.py`: tolerance scaled to part size → coarse `.glb` (viewing) + medium `.glb`/`.stl`. 104 MB.

**What broke → fix**
- 29/187 meshes not watertight: zero-area triangles at sphere poles and cone tips → remove them (now 182–185/187).
- cadquery's `tessellate()` treats tolerance as relative to each edge → call the OpenCascade mesher directly.

**Tessellation tolerance test** (tolerance = fraction of bbox diagonal; pixel and embedding columns measured in steps 3–4 on 20 parts)

| level | tolerance | median triangles | storage, 187 parts | median volume error | pixels changed vs medium | DINOv2 cosine vs medium |
|---|---|---|---|---|---|---|
| coarse | 0.5%, 29° | 530 | 8 MB | 0.37% | 0.063% | 0.998 (worst view 0.93) |
| medium | 0.1%, 11° | 1,292 | 95 MB | 0.06% | — | — |
| fine | 0.02%, 6° | 2,612 | 258 MB | 0.015% | 0.012% | 0.9996 |

Medium is already sub-pixel at the model's 224 px. Fine used 71% of storage and changed nothing visible → dropped. Coarse shows facets on round parts → viewing only.

**Decisions**
- Medium mesh is the render source; fine dropped from the pipeline (kept only for the comparison).

**What was learned**
- Three levels of information loss: feature tree (`.FCStd`) → exact geometry (`.step`, parameters gone) → mesh (faces and exact curves gone).
- Tolerance must scale with size: parts range from 0.1 mm to 9.7 m. Size scaling alone isn't enough: an M3 screw's thread became 331k triangles → also needs a triangle cap.
- ABC files are declared in metres; OpenCascade converts to mm on import.
- Step 1 question answered: the 3 invalid solids mesh watertight after the degenerate-triangle fix.

## Step 3 — Render (2026-10-26)

**Goal:** consistent renders, `renders/{part_id}/view_0..5.png`.

**What was built**
- `render.py`: 6 orthographic views per part, headless; part centered and scaled to a unit bounding sphere, lights move with the camera, same material, background and 512 px everywhere.
- `render --compare`: same parts at coarse/medium/fine, pixel difference (the tolerance table in step 2).

**What broke → fix**
- pyrender can't render headless on macOS → VTK offscreen, one window reused for every part.

**Decisions**
- Orthographic camera: no perspective size cues, so only shape differs between renders.

## Step 4 — Similar-part search (2026-10-26)

**Goal:** embed the renders, store them in Qdrant, and pick a search setup by testing it.

**What was built**
- `canonical.py`: turns each part onto its own axes before rendering (render v2).
- `embed.py` (DINOv2, CLIP), `evaluate.py` (12 configs), `index.py` (Qdrant local; moved to a Docker server in step 5).
- Report: `docs/step4-search-configuration-report.pdf`.

**What broke → fix**
- Inertia-axis (PCA) alignment fails on CAD: bracket axes tilted 24°, 114/187 parts have no defined axes → take axes from flat-face normals.
- First canonical pose flipped symmetric parts upside down → sign every axis by center-of-mass offset.
- pip torch crashed on import (OMP Error #15, two `libomp` copies) → torch from conda-forge.
- DINOv2's default preprocessing crops ~6% off every edge → resize straight to 224 px.

**Search test** (top-1, 60 rotated parts): without alignment, random rotations find the original 23–70% of the time. With canonical pose, every config hits 100%.

**Decisions**
- Canonical pose from faces, not inertia.
- DINOv2 over CLIP: better manual matches, and its score spread (0.6–0.9 vs 0.90–0.98) allows a "nothing similar" cutoff.
- Store both mean and 6-view MaxSim vectors.

**What was learned**
- Mirror-image parts share the duplicate fingerprint, so left and right brackets would merge.

**Open questions**
- Search ignores size: the 60 mm bracket's top match is 20 mm → re-rank by dims? (Step 6: `similar_parts` takes a size ratio.)
- Rotation test is maxed out → need labeled similar pairs to compare models fairly.
- Where to set the "nothing similar" cutoff (washer query)? (Step 5 grid: scores aren't comparable across queries, so one fixed cutoff won't work.)

## Step 5 — Pipeline (2026-10-27)

**Goal:** a new STEP file in `data/raw` becomes searchable with no manual steps.

**What was built**
- `pipeline/worker.py` (`python -m src.pipeline`): one process that polls `data/raw` (watcher) and runs extract → mesh → render → embed → index per file (worker). `run`, `status`, `retry`.
- SQLite `jobs` table = queue + history. `UNIQUE (file_id, file_sha256, pipeline_ver)` → rescans are no-ops, an edited file gets a new job, a stage-version bump re-enqueues everything (backfill).
- Per-stage skip on (sha, own + upstream versions): retries and backfills redo only what changed.
- Validation: triangle count 4–500k on the rendered mesh (plus step 1's volume/bbox checks).
- Failures: rejected part → log, job still done; unloadable STEP → dead at once; other errors → retry 30 s, 2 min, then dead. Dead STEP moves to `data/failed/` next to its log.
- Lease (10 min, renewed per stage) → a crashed worker's job is requeued; 3 crashes → dead.
- Lineage: tool versions per job; `file_sha256` + `job_id` in every Qdrant payload.
- Qdrant moved to a Docker server (local mode locks its folder to one process; the UI will need it too). `data/step_ok` dropped: files stay in `raw`.
- Doc: `docs/pipeline.pdf`.

**What broke → fix**
- Job ids jumped 101 → 302 → 1009: `INSERT OR IGNORE` burns an AUTOINCREMENT id on every ignored row → check before insert.
- Ctrl-C left jobs stuck `running`: `finally` cleared the job before the handler released it → release inside the worker.
- Renders only checked their own version, so a mesh change wouldn't re-render → skip key includes upstream versions.

**Numbers measured**
- Drop → searchable: ~16 s (10 s poll + settle, ~10 s job incl. model load).
- Full first pass, 101 files: 2 min. Unchanged rerun: 5 s, nothing redone.
- 108 files: 104 done, 2 dead (no solids; deliberately corrupt file), 1 part rejected (2 triangles), 196 parts searchable.
- Qdrant stopped mid-job → retry resumed at `index`, earlier stages skipped (checked by timestamps).
- Scaled to 500 files (one per ABC source document, seeded): 492 done, 10 dead (no solids), 1,625 searchable parts; 67% of the new files' solids were in-file duplicates. 22.5 min for 396 new files; 1.6 GB total.
- Top-5 grid of 10 queries (`docs/img/step5-top5-matches.png`): shape matching convincing (sheets, flanges, rods); size ignored (21 mm query → 4,980 mm match); scores not comparable across queries (rods 0.99, complex housing 0.55); my bracket's matches are blocks with holes, not L-brackets.

**Decisions**
- SQLite queue over Redis + RQ: no extra server, and RQ forks per job (crash risk with VTK/MPS on macOS, model reloaded each job).
- Polling over OS events: ingest needs completeness, not sub-second latency.
- User uploads → separate synchronous path (same stage code, not indexed); step 7.
- Sample one file per ABC source document: the chunk's 10,000 files come from 3,173 documents, so consecutive files are versions of the same design.

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

**What was learned**
- Idempotency is what makes at-least-once safe: jobs will run twice (crash, retry).
- Rows aren't deleted from the jobs table: it's how the watcher knows what's done, plus audit and step 8 metrics.
- Hard crash (lease expiry, counts as an attempt) vs Ctrl-C (job handed back, doesn't count).
- Background processes started without job control ignore SIGINT: a harness quirk, not a pipeline bug.

**Open questions**
- Deleted files (and obsolete parts) not removed from the index yet.
- Upload path `query_file()` designed, not built (step 7).
- Batch `render` command skips the triangle gate; only the worker applies it.

## Step 6 — Text side: fake PLM + Q&A agent (2026-10-27)

**Goal:** ask "Which aluminum parts failed durability tests?" and get a cited answer; a restricted user can't see other programs' parts.

**What was built**
- `qa/plm.py`: seeded generator, facts first → `plm_items` (2,214 revisions), `erp_suppliers` (953), `doc_test_reports` (3,261, 39% Korean), `doc_change_notices` (589) + `data/plm/facts.json` (the truth).
- `qa/textindex.py`: join → `part_cards`; 5,475 chunks (card per part, one per note) with BGE-M3 dense + BGE-M3 sparse + BM25 over Kiwi tokens.
- `qa/tools.py`: `lookup_part`, `query_parts`, `search_notes`, `similar_parts`, each enforcing the user's programs.
- `qa/agent.py`: bounded loop on Claude Opus 5.5; `ask_many` runs questions concurrently.
- Refactor: `src/pipeline/` (steps 1–5) and `src/qa/` (step 6); commands unchanged via `__main__`. Docs: `docs/pipeline.pdf`, `docs/query-flow.pdf`.

**Decisions**
- **Text serves the CAD side.** Cards carry step 1's measured geometry; class comes from the STEP name or geometry rules; mass = volume × density; failures depend on material and shape. Demo centerpiece: "has a part like mine failed?" = `similar_parts` → history.
- **Facts first, text second:** every answer is known exactly → step 8 gold labels need no hand labeling.
- **Scattered tables + an explicit join**, as in a real PLM/ERP/test-system landscape; the join is the unglamorous half of enterprise RAG.
- **Templates + synonym pools (EN/KO), no LLM-written notes:** free, deterministic, and the synonyms already make keyword search insufficient on its own.
- **One chunk per document + one card per part;** a context header (part, class, material, test, result) is embedded but never shown or cited.
- **Hybrid = parallel fusion (RRF), not a keyword-first cascade:** a cascade loses every synonym in its first stage.
- **Korean: Kiwi tokenization + Claude query expansion,** not document translation (cost, non-determinism, risk to part numbers).
- **Bounded agent (≤ 4 tool rounds, then tools off) over a fixed router:** the best demo question needs two rounds.
- **The model never writes SQL; permissions live inside the tools,** programs come from the user, hidden = missing ("not found").
- **Ids derived from content hashes,** so regenerating or adding parts never renumbers reports or ECOs.
- **Opus 5.5 at effort medium, `fallbacks="default"`, cached prompt;** Sonnet / lower effort left for step 8 to decide.

**What broke → fix**
- Fail rate 26% (revision fixes added historical fails) → fewer, rarer fixes → 19% fail, 5% conditional.
- "Body" fallback took 29%: rods missed the round-face rule, profiles had no class → cross-section test + Beam class → 12%.
- Rev A failed at 584k cycles beside a rev B "pass" of a 200k target → test targets fixed per part and test.
- ADC12 casting "fixed" into SPFH590 steel sheet → material changes only within a family.
- Report ids renumbered on every regeneration → hash-based ids.
- Index build at 8 chunks/s (short notes padded to card length) → length-sorted batches, 2.5× faster.
- Strict tool schemas rejected `type: [string, null]` with enums → `anyOf`.
- BGE-M3 loaded lazily inside the first search (22.7 s) and every queued tool call waited on it → preload.

**Numbers measured**
- Korean question ("aluminum brackets with a bolt-hole fatigue crack, how many?"): 3 / 2 / 14 matched `facts.json` exactly.
- Permissions: same count 14 (normal) vs 33 (admin); direct question about a P2 part leaked nothing.
- 10–27 s per question once warm (output tokens are the bottleneck: thinking + long answers); $0.03–0.12, avg ~$0.07; 4 concurrent questions 50 s vs 159 s serial.

**For step 7 (UI)**
- Load BGE-M3, Kiwi and the tools once at startup; stream the answer.
- Upload path: run the step 5 stage functions on a dropped STEP without indexing, then `similar_parts` on its vector.
- Show the renders of cited parts next to the answer.

**For step 8 (measure)**
- Gold labels from `facts.json`: write each question with a structured spec (what it means), compute the answer by code.
- Questions must use wording the note templates don't (incl. Korean and cross-language), or the eval tests the generator against itself.
- Score separately: routing (tool chosen), retrieval (recall@k / MRR: BM25, BM25 + expansion, dense, m3_sparse, hybrid; free, no LLM), answer (exact / list F1 / key facts), citations valid, leaks = 0, abstention on unanswerable.
- Rows to compare: Opus 5.5 medium vs low, Sonnet 5.5 (Haiku 5.5 optional); ~$4 per 60-question pass; repeat 2–3× for spread.
- Acceptance question run verbatim after the refactor: counts exact (45 aluminum parts with a fatigue fail, 32 open, 13 fixed), but one listed part number didn't exist (BM-93093 for BM-93975; the model corrected itself at the end). → Built an id post-check: every part number / report id in the answer must appear in the question or a tool result; otherwise one rewrite with tools off, and survivors are returned as `unverified_ids`. Plus a prompt rule to group long lists. Rerun: no unknown ids, list grouped, but still ~2,600 output tokens (mostly thinking) → effort is the lever, not the prompt. Step 8: count how often the repair fires.

**For step 9 (story)**
- "Text RAG over PLM is a commodity; grounding it in geometry is the differentiator: a new part has no part number yet, only a shape."
- Permission filtering before retrieval, inside tools; negation fools search → facts as filters.

**Open questions**
- Parts added by the pipeline get no PLM records or chunks until `src.qa.plm` + `src.qa.textindex build` rerun (~5 min); the index has no incremental update.
- 39% of latest revisions are "In Work" (open fails force it); fine for the demo, high for a real program.
- BR-63592's "hole count reduced" change right before its crack is a generator coincidence, not a planted story.

## Step 7 — UI (2026-10-27) · TEMPORARY, step not finished

**Goal:** a Streamlit app where a 2-minute demo flows from search to similar parts to Q&A.

**What was built**
- `src/ui/`: `app.py` (sidebar: user = programs, index status), pages `views/search.py`, `part.py`, `ask.py` (`upload.py` until it merged into Ask); `data.py` (reads, permission first), `widgets.py`, `resources.py` (models loaded once), `cadworker.py`.
- Search: hybrid text search grouped by part, filters, EN/KO. Part: renders, PLM record, tests / ECOs / revisions / geometry, similar parts + size filter, 3D view. Ask: the agent with live tool progress and cited parts as tiles. Upload: STEP → renders → similar existing parts → "Ask about these parts".
- Attachments in Ask (Upload page removed): STEP → extract/mesh/render/embed in the CAD worker, each solid a handle `NEW-1` that `similar_parts` accepts (`Tools(user, shapes)`), shown at once as a card (render, size, top-5 similar, "reuse it?" at ≥ 0.97, views + 3D in an expander), its iso render sent to Claude (max 4 per question); images to Claude (≤ 1568 px, ≤ 5 MB, JPEG fallback); PDF and txt/md/csv as documents. A sample-STEP example button replaces the Upload page's.
- Multi-turn Ask: `agent.ask_async(history=, seen=)` continues a conversation with the full transcript (tool results included); the page keeps it in `session_state.convo`, "New chat" clears it, a handoff or example starts a new one.
- Step 6 code touched: `agent.ask_async(on_event=)` progress hook; `Tools.similar_parts` split into `similar_to(views, size)` (output identical, checked); BGE-M3 fp16 for queries.

**Decisions**
- **Memory first (8 GB M1):** BGE-M3 fp16 for queries (−1.6 GB, same top-10 on 8 queries); one process holds the models; 3D view off by default (WebGL 100–300 MB).
- **Extract / mesh / render run in a child process:** VTK's Cocoa window aborts the process when created off the main thread, and Streamlit runs scripts off it. Its renders are pixel-identical to the pipeline's.
- **An upload is a question, not an ingest:** nothing is written to SQLite or Qdrant; similarity ≥ 0.97 is flagged as "existing part, reuse it?".
- **All model calls on the agent's single tool thread;** models warmed on startup (incl. a 6-image DINOv2 batch).
- **Multi-turn = full transcript, not Q + answer text:** follow-ups mostly point at the last turn's data ("of those three…"), so keeping tool results lets Claude answer without calling tools again; automatic caching reads earlier turns from cache. Cited ids (`seen`) carry over or the id check flags follow-ups. An id repair keeps only the corrected answer; a refused or failed turn is dropped. Past 200k chars of old tool results, those before the last 2 questions become a stub.
- **Upload merged into the chat:** two entry points for one job confused the demo; the page's detail (views, 3D) lives in the card's expander. Images go to Claude only, no photo → render search (domain gap, untested). Attached files are the engineer's: context, not evidence; ids in a text file count as citable, a PDF's don't (not parsed here).
- **Demo hardening:** server bound to localhost (the app spends the API key), HF offline, model-viewer + lighting served from `src/ui/static` (no network).

**What broke → fix**
- VTK in Streamlit's thread: NSException abort → CAD child process.
- Fused fp16 attention fails on M1 → eager attention for the fp16 path.
- New browser session took 4.9 s: Streamlit's file watcher scans 6,226 modules per session → demo with `--server.fileWatcherType none` (0.6 s).
- Links within the Part page left the old part in the address bar, and the part box (no key) returned stale text → the next click reopened the empty page; the 3D toggle drew "off" while the viewer spun → URL is the source of truth, box and toggle synced explicitly.
- 3D view blurry: the pipeline's glbs have no normals (`mesh.export` lacks `include_normals=True`) and no material, and model-viewer's built-in lighting is near uniform → normals + material added at view time; HDR studio light (`src/ui/make_studio_hdr.py`).
- Ask: the previous run's live answer lingered, faded, under the next question → `st.rerun()` after each answer; examples hidden while answering.

**Numbers measured**
- Cold start ~20 s (models 12–14 s); Search 0.4–0.5 s; Part < 0.5 s; Upload ~1 s (extract 0.15, mesh 0.03, render 0.17, embed 0.6 s); Ask 5–18 s, of which tools ~0.2 s: the latency is the LLM.
- Memory: Streamlit 3.15 GB at start, 4.2 GB after use; CAD worker 0.29 GB; Qdrant 0.44 GB.

**For step 8 (measure)**
- Multi-turn cases (~5, two turns each, run with `ask_async(history=, seen=)`): a pointer back ("of those three…"), "show the rest" after a grouped answer, a language switch (EN → KO), a follow-up that must call tools again (new fact not in the earlier results), and a user switch (new user = no history; must not answer from the other user's results).
- Score turn 2 on its own: answer correct, tool calls made (0 when history suffices, > 0 when not), cited ids valid, repair rate not higher than single-turn.
- Cost per turn: cache read vs. fresh input tokens; check compaction (past 200k chars) once with a long conversation.

**Open issues**
1. ~~No multi-turn chat~~ → done (3-turn check: turn 3 "그 부품 재질 다시 알려줘" answered from history, 0 tool calls, 2.3 s). Eval cases: see "For step 8" above.
2. ~~Upload is not part of the chat~~ → done (sample bracket: NEW-1 → BR-66254 at 1.0, agent called `similar_parts(NEW-1)` itself; image + PDF + text in one question: 7.1 s). Not yet: a real photo, a large PDF's token cost, Word files.
3. ~~Page title~~ → "cadai".
4. ~~Pipeline glbs without normals~~ → `mesh.export(include_normals=True)`; the 3,252 existing glbs rewritten in place (vertices and faces checked identical, `glb_bytes` updated), no `MESH_VER` bump. glbs +47% (251 → 359 MB). The UI still adds the material.
5. Streamlit process grows 3.15 → 4.2 GB with use (likely the MPS cache); not yet checked whether it plateaus.
6. ~~No Korean UI~~ → bilingual, Korean first (no toggle): `widgets.bi()` = Korean + small grey English for labels, buttons, tabs, status; `bi_lines()` puts a sentence's English on the next line (inline mixing read badly in sentences); `bi_plain()` "한국어 · English" where Streamlit shows plain text (menu, placeholders, options). Checked in Chrome on all three pages. Data, PLM cards, answers and example questions stay as they are.
7. **Design too generic:** default Streamlit look; needs its own visual identity.
8. Not done: full 2-minute demo run, latency table for step 8, README demo command, answer streaming.
