# cadai

Mini CAD-to-AI pipeline: STEP files → metadata → meshes → renders → embeddings → similar-part search + RAG Q&A.
See `docs/cad-ai-interview-prep-plan.md` for the plan, `docs/pipeline.pdf` and `docs/query-flow.pdf` for how
the system works (rebuilt by `python docs/src/build_docs.py [pipeline] [query]`), and `logs/step-log.md`
for progress per step.

## Setup
    conda env create -f environment.yml
    conda activate cadai
    cp .env.example .env       # then put your ANTHROPIC_API_KEY in it (gitignored)

Qdrant runs as a server in Docker (first run downloads the official image):

    docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
      -v "$(pwd)/qdrant_data:/qdrant/storage" qdrant/qdrant:v1.20.0
    docker start qdrant        # after a reboot

Dashboard: http://localhost:6333/dashboard

## Run
CAD pipeline (steps 1–5):

    python -m src.pipeline run       # watch data/raw: drop a STEP file in, it becomes searchable
    python -m src.pipeline status    # job counts, queued / dead jobs with errors
    python -m src.pipeline retry --dead
    python -m src.pipeline.index query my_bracket_s00

Q&A (step 6):

    python -m src.qa.plm                     # (re)generate the fake PLM / ERP / test records
    python -m src.qa.textindex build         # part cards + text index (~5 min)
    python -m src.qa "Have parts similar to BR-66254 failed?" --user normal

Demo app (step 7): Qdrant running, then

    streamlit run src/ui/app.py --server.fileWatcherType none

Opens http://localhost:8501 (bound to localhost: the app spends the API key). The first load after a start
takes ~15 s (models); later page loads are instant. `--server.fileWatcherType none` keeps each new browser
session fast (0.6 s instead of ~5 s); restart the app after changing code outside `src/ui/views/`.
Search, Part and Ask pages; in Ask, attach a STEP file (or click the sample-bracket example) to find similar
parts and ask about them.

## Layout
    src/config.py   shared paths and .env loading
    src/pipeline/   steps 1–5: extract, mesh, render, canonical, embed, index, evaluate, worker
    src/qa/         step 6: plm (fake records), textindex, tools, agent
    src/ui/         step 7: Streamlit app (app.py, views/, widgets, CAD worker); theme in .streamlit/config.toml
    data/raw        incoming STEP files (they stay here once processed)
    data/failed     dead-letter: error logs + STEP files of dead jobs
    data/mesh       glTF / STL
    data/renders    6 views per part
    data/plm        facts.json: the truth behind the fake records (step 8 gold labels)
    db/             SQLite (parts, jobs, PLM tables, part cards)
    qdrant_data/    Qdrant server storage (collections: parts, docs)
    docs/           plan, reports, system documents
    logs/           step log
