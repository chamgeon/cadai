"""Shared paths for every pipeline stage."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW = DATA / "raw"            # incoming STEP files
STEP_OK = DATA / "step_ok"    # loaded and validated
FAILED = DATA / "failed"      # dead-letter folder + error logs
MESH = DATA / "mesh"          # glTF (viewing) and STL (ML)
RENDERS = DATA / "renders"    # renders/{part_id}/view_0..5.png

DB_PATH = ROOT / "db" / "parts.sqlite"
QDRANT_PATH = ROOT / "qdrant_storage"  # Qdrant local mode (no server); gitignored
