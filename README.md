# cadai

Mini CAD-to-AI pipeline: STEP files → metadata → meshes → renders → embeddings → similar-part search + RAG Q&A.
See `docs/cad-ai-interview-prep-plan.md` for the 10-day plan and `logs/daily-log.md` for progress.

## Setup
    conda env create -f environment.yml
    conda activate cadai

## Layout
    data/raw        incoming STEP files
    data/step_ok    loaded + validated
    data/failed     failed files + error logs
    data/mesh       glTF / STL
    data/renders    6 views per part
    db/             SQLite
    src/            pipeline code
    logs/           daily log
