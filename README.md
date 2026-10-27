# cadai

Mini CAD-to-AI pipeline: STEP files → metadata → meshes → renders → embeddings → similar-part search + RAG Q&A.
See `docs/cad-ai-interview-prep-plan.md` for the step-by-step plan and `logs/daily-log.md` for progress.

## Setup
    conda env create -f environment.yml
    conda activate cadai

Qdrant runs as a server in Docker (first run downloads the official image):

    docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
      -v "$(pwd)/qdrant_data:/qdrant/storage" qdrant/qdrant:v1.20.0
    docker start qdrant        # after a reboot

Dashboard: http://localhost:6333/dashboard

## Run
    python -m src.pipeline run       # watch data/raw: drop a STEP file in, it becomes searchable
    python -m src.pipeline status    # job counts, queued / dead jobs with errors
    python -m src.pipeline retry --dead
    python -m src.index query my_bracket_s00

## Layout
    data/raw        incoming STEP files (they stay here once processed)
    data/failed     dead-letter: error logs + STEP files of dead jobs
    data/mesh       glTF / STL
    data/renders    6 views per part
    db/             SQLite (parts metadata + jobs queue)
    qdrant_data/    Qdrant server storage
    src/            pipeline code
    logs/           daily log
