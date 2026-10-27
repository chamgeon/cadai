"""Shared paths for every pipeline stage."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW = DATA / "raw"            # incoming STEP files; they stay here once processed
FAILED = DATA / "failed"      # dead-letter folder: error logs + STEP files of dead jobs
MESH = DATA / "mesh"          # glTF (viewing) and STL (ML)
RENDERS = DATA / "renders"    # renders/{part_id}/view_0..5.png

DB_PATH = ROOT / "db" / "parts.sqlite"
# Qdrant server in Docker (see README); storage is bind-mounted to qdrant_data/, gitignored
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
