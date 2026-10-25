# Daily Log

## Day 1 — Setup + Read CAD (2026-10-25)

**Goal today:** set up the environment, data, and project skeleton; load every STEP file, extract properties, store in SQLite, list failures and why.

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
- Invalid-but-sensible solids are kept with `is_valid=0`; Day 2 will show whether they still mesh. Repair option for later: OpenCascade `ShapeFix`.

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
- **Keep `data/raw` untouched; record failures in the DB + log files.** Moving files belongs to the Day 5 pipeline.
- **Keep invalid solids** (flagged) rather than dropping them, until meshing shows whether they're usable.
- **Role of this data:** the `parts` table is the **part registry**, not the search index. Every later stage reads from it (which parts to mesh/render, size-scaled mesh tolerance, joining PLM records, UI detail pages, Day 8 metrics). The embedding index (Day 4) is separate and rebuildable from it. Size/volume can also filter or re-rank search results to fix image search's "same shape, different size" weakness.

**What was learned**
- **Design intent** = the "how and why" of a model (parameters, constraints, feature order, rules like "hole stays centered"). Native CAD keeps it; STEP keeps only the final exact geometry. In FreeCAD, an imported STEP shows bare bodies with no feature tree.
- **B-rep (boundary representation):** a solid described by its skin with exact math. Geometry (planes, cylinders, NURBS surfaces, curves) + topology (faces, edges, vertices → shells → solid). `n_faces` / `n_edges` are B-rep counts. A mesh (Day 2) approximates the same surfaces with triangles.
- **Solid:** one closed, watertight body. A door may be 3 solids (frame, glass, handle) or 1 if fused; it depends on how it was modeled.
- **Valid solid:** closed (every edge shared by exactly 2 faces), consistent outward orientation, no self-intersection, edges lie on their surfaces within tolerance, no degenerate edges/faces. Invalid solids mostly come from translation between CAD kernels with different tolerances.
- **At scale:** CATIA already stores one physical part per `CATPart`, assembled in a `CATProduct`, so splitting assemblies matters less there, but CATIA → STEP translation will produce invalid solids, so the pipeline needs validation, automatic repair, and a failed queue.
- ABC formats: only `step` is needed; meshes and renders will be generated by my own pipeline.

**Open questions**
- Should multi-body parts (one physical part modeled as several solids) be merged back into one part? How could that be detected?
- How many of the 3 invalid solids will mesh cleanly on Day 2?
- How does real CATIA → STEP AP242 export handle assemblies with missing referenced parts?
