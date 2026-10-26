# Daily Log

## 2026-10-25 — Step 1: Setup + Read CAD

**Goal:** set up the environment, data, and project skeleton; load every STEP file, extract properties, store in SQLite, list failures and why.

**What was built**
- Conda env `cadai` (Python 3.11, CadQuery 2.8, trimesh, pyrender, qdrant-client, streamlit); `environment.yml` to recreate it.
- Downloaded ABC chunk 0000 (STEP, ~1.6 GB compressed, 10,000 models) and copied 100 files into `data/raw`.
- Project layout: `data/{raw,step_ok,failed,mesh,renders}`, `db/`, `src/`, `logs/`; `src/config.py` holds all paths; git repo with `data/` ignored.
- Installed FreeCAD and opened sample STEP files to see what the data looks like.
- `src/extract.py`: for each STEP file in `data/raw`
  1. SHA-256 hash → skip file if unchanged since last run.
  2. Load with CadQuery, split into individual solids.
  3. Per solid: faces, edges, volume, area, bounding box, OpenCascade validity.
  4. Validate (volume > 0, bounding box < 100 m).
  5. Flag repeated copies of the same solid within a file (`dup_of`).
  6. Save to `db/parts.sqlite`; write full traceback to `data/failed/<file>.log` on load failure.
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
- Invalid-but-sensible solids are kept with `is_valid=0`; Step 2 (meshing) will show whether they still mesh. Repair option for later: OpenCascade `ShapeFix`.

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
- How many of the 3 invalid solids will mesh cleanly in step 2? (Answered 2026-10-26: all, after the degenerate-triangle fix.)
- How does real CATIA → STEP AP242 export handle assemblies with missing referenced parts?

## 2026-10-26 — Steps 2–4: Own part, meshes, renders, similar-part search

**Goal:** model one part by hand, mesh and render every part, and pick a similar-part search setup by testing it.

**What was built**
- Parametric L-bracket in FreeCAD (spreadsheet parameters) → `data/raw/my_bracket.step`. Extracted volume matches the hand calculation exactly (13,967.3 mm³).
- `extract.py` v2: rotation-independent size (sorted dims, principal moments), shape (fill ratio, sphericity), face-type counts, STEP header lineage. Bumping `EXTRACT_VER` reprocesses all files; old rows get the new columns added.
- `mesh.py`: tolerance scaled to part size → coarse `.glb` (viewing) + medium `.glb`/`.stl`. 104 MB.
- `render.py`: 6 orthographic views per part, headless via VTK.
- `canonical.py`: turns each part onto its own axes before rendering.
- `embed.py` (DINOv2, CLIP), `evaluate.py` (12 configs), `index.py` (Qdrant local).
- Report: `docs/step4-search-configuration-report.pdf`.

**What broke → fix**
- 29/187 meshes not watertight: zero-area triangles at sphere poles and cone tips → remove them (now 182–185/187).
- cadquery's `tessellate()` treats tolerance as relative to each edge → call the OpenCascade mesher directly.
- pyrender can't render headless on macOS → VTK offscreen.
- Inertia-axis (PCA) alignment fails on CAD: bracket axes tilted 24°, 114/187 parts have no defined axes → take axes from flat-face normals.
- First canonical pose flipped symmetric parts upside down → sign every axis by center-of-mass offset.
- pip torch crashed on import (OMP Error #15, two `libomp` copies) → torch from conda-forge.
- DINOv2's default preprocessing crops ~6% off every edge → resize straight to 224 px.

**Tessellation tolerance test** (tolerance = fraction of bbox diagonal)

| level | tolerance | median triangles | storage, 187 parts | median volume error | pixels changed vs medium | DINOv2 cosine vs medium |
|---|---|---|---|---|---|---|
| coarse | 0.5%, 29° | 530 | 8 MB | 0.37% | 0.063% | 0.998 (worst view 0.93) |
| medium | 0.1%, 11° | 1,292 | 95 MB | 0.06% | — | — |
| fine | 0.02%, 6° | 2,612 | 258 MB | 0.015% | 0.012% | 0.9996 |

Medium is already sub-pixel at the model's 224 px. Fine used 71% of storage and changed nothing visible → dropped. Coarse shows facets on round parts → viewing only. (Pixel and embedding checks on 20 parts.)

**Search test** (top-1, 60 rotated parts): without alignment, random rotations find the original 23–70% of the time. With canonical pose, every config hits 100%.

**Decisions**
- Render from the medium mesh; fine dropped from the pipeline.
- Canonical pose from faces, not inertia.
- DINOv2 over CLIP: better manual matches, and its score spread (0.6–0.9 vs 0.90–0.98) allows a "nothing similar" cutoff.
- Store both mean and 6-view MaxSim vectors.

**What was learned**
- Three levels of information loss: feature tree (`.FCStd`) → exact geometry (`.step`, parameters gone) → mesh (faces and exact curves gone).
- Tolerance must scale with size: parts range from 0.1 mm to 9.7 m. Size scaling alone isn't enough: an M3 screw's thread became 331k triangles → also needs a triangle cap.
- ABC files are declared in metres; OpenCascade converts to mm on import.
- Mirror-image parts share the duplicate fingerprint, so left and right brackets would merge.
- Step 1 question answered: the 3 invalid solids mesh watertight after the degenerate-triangle fix.

**Open questions**
- Search ignores size: the 60 mm bracket's top match is 20 mm → re-rank by dims?
- Rotation test is maxed out → need labeled similar pairs to compare models fairly.
- Where to set the "nothing similar" cutoff (washer query)?
