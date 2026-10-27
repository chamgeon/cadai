"""Step 5: drop a STEP file into data/raw and it becomes searchable, no manual steps.

One process, two roles, a SQLite `jobs` table between them:
    watcher  polls data/raw every --interval seconds; a file whose size held
             still since the last scan is hashed and enqueued
    worker   claims queued jobs one at a time and runs, per file,
             extract -> mesh (+ triangle gate) -> render -> embed + index -> cleanup

Polling, not OS file events: release/revision ingest tolerates minutes of
latency but not a missed file. A scan can't miss one; events get dropped while
the process is down and are unreliable on network shares.

The jobs table is both the queue and the history (rows are never deleted):
    UNIQUE (file_id, file_sha256, pipeline_ver)
        an unchanged file is ignored however often it is scanned; an edited file
        (new hash) or a bumped stage version (new pipeline_ver) gets a new job,
        so a version bump backfills every file by itself
    each stage is skipped when its table already holds output for this hash and
    these upstream versions, so a backfill or a retry redoes only what changed

Failures:
    part fails validation         part rejected (log in data/failed), job still done
    STEP won't load / no solids   job dead at once: retrying gives the same result
    anything else (Qdrant down,   retried after 30 s, then 2 min, then dead
      DB locked, worker crashed)
A dead job's STEP file moves to data/failed next to its log; `retry` moves it back.

Usage:
    python -m src.pipeline run [--interval 10] [--once]
    python -m src.pipeline status
    python -m src.pipeline retry JOB_ID | --dead
"""
import argparse
import importlib.metadata
import json
import shutil
import sqlite3
import time
import traceback
from datetime import datetime, timedelta, timezone
from functools import cached_property

import OCP

from src import extract, index, mesh, render
from src.config import DB_PATH, FAILED, MESH, RAW, RENDERS
from src.embed import MODELS, Embedder, embed_part
from src.extract import EXTRACT_VER, sha256
from src.index import MODEL
from src.mesh import MESH_VER
from src.render import RENDER_VER, SOURCE_LEVEL

PIPELINE_VER = f"e{EXTRACT_VER}.m{MESH_VER}.r{RENDER_VER}.{MODEL}"
MAX_ATTEMPTS = 3
RETRY_BASE = 30  # seconds before the first retry, x4 for each one after
LEASE = 600  # a running job not renewed within this many seconds is presumed crashed
SETTLE = 30  # a file first seen this long after its last write counts as fully copied
TOOLS = ("cadquery", "trimesh", "vtk", "torch", "transformers", "qdrant-client")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id      TEXT NOT NULL,
    file_name    TEXT NOT NULL,
    file_sha256  TEXT NOT NULL,
    pipeline_ver TEXT NOT NULL,      -- every stage version joined: a bump re-enqueues every file
    status       TEXT NOT NULL,      -- queued | running | done | dead | superseded
    stage        TEXT,               -- last stage reached: extract | mesh | render | index | cleanup
    attempts     INTEGER NOT NULL DEFAULT 0,
    error        TEXT,
    n_parts      INTEGER,            -- solids that passed extract validation (duplicates excluded)
    n_indexed    INTEGER,            -- of those, searchable after the job
    next_run_at  TEXT,               -- not claimable before this (retry backoff)
    lease_until  TEXT,               -- a running job past this is presumed crashed
    enqueued_at  TEXT,
    started_at   TEXT,
    finished_at  TEXT,
    versions     TEXT,               -- JSON: tool versions of the worker that ran it (lineage)
    UNIQUE (file_id, file_sha256, pipeline_ver)
);
"""


class Permanent(Exception):
    """A failure that retrying cannot fix."""


def now(offset=0):
    return (datetime.now(timezone.utc) + timedelta(seconds=offset)).isoformat(timespec="seconds")


def connect():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=5)  # wait up to 5 s for another process's write lock
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # readers (status, the UI) don't block the worker's writes
    for schema in (extract.SCHEMA, mesh.SCHEMA, render.SCHEMA, index.SCHEMA, SCHEMA):
        conn.executescript(schema)
        extract.migrate(conn, schema)
    return conn


def tool_versions():
    versions = {name: importlib.metadata.version(name) for name in TOOLS}
    versions["OCP"] = OCP.__version__  # conda-forge's OCP has no package metadata
    versions["model"] = MODELS[MODEL]
    return versions


def owned_by(part_id, file_id):
    return part_id.rsplit("_s", 1)[0] == file_id  # part_id = {file_id}_s{solid_index:02d}


# ---------------------------------------------------------------- watcher

class Watcher:
    """Polls RAW. Hashes a file only when its size or mtime changed since it was last hashed."""

    def __init__(self):
        self.sizes = {}  # path -> size at the previous scan
        self.hashes = {}  # path -> ((size, mtime_ns), sha256)

    def scan(self, conn, settle=True):
        sizes = {}
        for path in sorted(RAW.iterdir()):
            if path.suffix.lower() not in (".step", ".stp") or path.name.startswith("."):
                continue
            st = path.stat()
            sizes[path] = st.st_size
            # Still being copied in? Wait until its size holds still for one scan.
            if path in self.sizes:
                settled = self.sizes[path] == st.st_size
            else:
                settled = time.time() - st.st_mtime > SETTLE
            if settle and not settled:
                continue
            key = (st.st_size, st.st_mtime_ns)
            if self.hashes.get(path, (None,))[0] != key:
                self.hashes[path] = (key, sha256(path))
            enqueue(conn, path, self.hashes[path][1])
        self.sizes = sizes
        self.hashes = {p: h for p, h in self.hashes.items() if p in sizes}


def enqueue(conn, path, sha):
    # Look first: an INSERT OR IGNORE that hits the UNIQUE key still burns an AUTOINCREMENT id, every scan.
    if conn.execute("SELECT 1 FROM jobs WHERE file_id = ? AND file_sha256 = ? AND pipeline_ver = ?",
                    (path.stem, sha, PIPELINE_VER)).fetchone():
        return
    cur = conn.execute(
        "INSERT OR IGNORE INTO jobs (file_id, file_name, file_sha256, pipeline_ver, status, next_run_at, enqueued_at)"
        " VALUES (?, ?, ?, ?, 'queued', ?, ?)", (path.stem, path.name, sha, PIPELINE_VER, now(), now()))
    if cur.rowcount:
        # An older revision still waiting would only redo work this job replaces.
        conn.execute("UPDATE jobs SET status = 'superseded', finished_at = ?"
                     " WHERE file_id = ? AND status = 'queued' AND job_id != ?", (now(), path.stem, cur.lastrowid))
        print(f"queued   job {cur.lastrowid:<4} {path.name}")
    conn.commit()


# ---------------------------------------------------------------- queue

def claim(conn, versions):
    """Take the oldest claimable job. One UPDATE, so two workers can never take the same job."""
    job = conn.execute(
        """UPDATE jobs SET status = 'running', attempts = attempts + 1, started_at = ?, lease_until = ?,
                          versions = ?, error = NULL
           WHERE job_id = (SELECT job_id FROM jobs WHERE status = 'queued' AND next_run_at <= ?
                           ORDER BY job_id LIMIT 1)
           RETURNING *""", (now(), now(LEASE), versions, now())).fetchone()
    conn.commit()
    return job


def recover(conn):
    """Requeue running jobs whose worker died; a job that keeps killing its worker goes dead."""
    for job in conn.execute("SELECT * FROM jobs WHERE status = 'running' AND lease_until < ?", (now(),)).fetchall():
        if job["attempts"] >= MAX_ATTEMPTS:
            kill(conn, job, f"worker died during this job {job['attempts']} times (crash or kill)")
        else:
            conn.execute("UPDATE jobs SET status = 'queued', next_run_at = ?, lease_until = NULL WHERE job_id = ?",
                         (now(), job["job_id"]))
            conn.commit()
            print(f"requeued job {job['job_id']:<4} {job['file_name']}  (lease expired)")


def finish(conn, job_id, status, **fields):
    fields.update(status=status, finished_at=now(), lease_until=None)
    conn.execute(f"UPDATE jobs SET {', '.join(f'{k} = ?' for k in fields)} WHERE job_id = ?",
                 [*fields.values(), job_id])
    conn.commit()


def kill(conn, job, error):
    """Dead-letter: mark the job dead and move its STEP file next to the logs in data/failed."""
    finish(conn, job["job_id"], "dead", error=error)
    path = RAW / job["file_name"]
    if path.exists() and sha256(path) == job["file_sha256"]:  # not a newer revision dropped in meanwhile
        shutil.move(path, FAILED / path.name)
    print(f"dead     job {job['job_id']:<4} {job['file_name']}  {error}")


# ---------------------------------------------------------------- worker

class Worker:
    """Runs one job at a time. The VTK window, model and Qdrant client are created once, on first use."""

    def __init__(self, conn):
        self.conn = conn
        self.versions = json.dumps(tool_versions())
        self.job = None  # the job in progress (stage() renews its lease)
        self.current = None  # its current stage, for the failure log

    @cached_property
    def renderer(self):
        return render.Renderer()

    @cached_property
    def embedder(self):
        return Embedder(MODEL)

    @cached_property
    def qc(self):
        return index.client()

    def process(self, job):
        self.job = job
        start = time.perf_counter()
        try:
            n_parts, n_indexed = self.run(job)
        except Permanent as e:
            self.fail(job, str(e), permanent=True)
        except Exception as e:
            self.fail(job, f"{type(e).__name__}: {e}", permanent=False)
        except KeyboardInterrupt:
            self.release(job)  # here, before `finally` forgets the job
            raise
        else:
            if n_parts is not None:
                finish(self.conn, job["job_id"], "done", n_parts=n_parts, n_indexed=n_indexed)
                print(f"done     job {job['job_id']:<4} {job['file_name']}  "
                      f"{n_indexed}/{n_parts} parts searchable  {time.perf_counter() - start:.1f}s")
        finally:
            self.job = None

    def stage(self, name):
        """Record progress and renew the lease: a live worker's job is never taken for a crashed one."""
        self.current = name
        self.conn.execute("UPDATE jobs SET stage = ?, lease_until = ? WHERE job_id = ?",
                          (name, now(LEASE), self.job["job_id"]))
        self.conn.commit()

    def run(self, job):
        conn, fid, sha = self.conn, job["file_id"], job["file_sha256"]
        path = RAW / job["file_name"]
        if not path.exists() or sha256(path) != sha:
            finish(conn, job["job_id"], "superseded", error="file changed or removed after it was queued")
            return None, None

        self.stage("extract")
        solids = None
        f = conn.execute("SELECT status, error FROM files WHERE file_id = ? AND file_sha256 = ? AND extract_ver = ?",
                         (fid, sha, EXTRACT_VER)).fetchone()
        if f is None:
            file_row, part_rows, solids = extract.process(path)
            extract.save(conn, file_row, part_rows)
            f = file_row
        if f["status"] == "failed":
            raise Permanent(f["error"])  # extract already wrote data/failed/{file_id}.extract.log

        def get_solids():  # only loaded if a stage below actually has to run
            nonlocal solids
            if solids is None:
                solids = extract.load_solids(path)
            return solids

        parts = conn.execute("SELECT part_id, solid_index, volume, dim_1, dim_2, dim_3 FROM parts"
                             " WHERE file_id = ? AND status = 'ok' AND dup_of IS NULL ORDER BY part_id",
                             (fid,)).fetchall()
        indexed = [p["part_id"] for p in parts if self.run_part(p, sha, get_solids)]

        self.stage("cleanup")
        self.cleanup(fid, indexed)
        return len(parts), len(indexed)

    def run_part(self, p, sha, get_solids):
        """mesh -> render -> embed + index for one part. True if it ends up searchable."""
        conn, pid = self.conn, p["part_id"]

        self.stage("mesh")
        n_levels = conn.execute("SELECT count(*) FROM meshes WHERE part_id = ? AND file_sha256 = ? AND mesh_ver = ?",
                                (pid, sha, MESH_VER)).fetchone()[0]
        if n_levels < len(mesh.LEVELS):
            mesh.mesh_and_save(conn, pid, get_solids()[p["solid_index"]], p["volume"],
                               (p["dim_1"], p["dim_2"], p["dim_3"]), sha)
        source = conn.execute("SELECT * FROM meshes WHERE part_id = ? AND level = ?", (pid, SOURCE_LEVEL)).fetchone()
        try:
            mesh.check_mesh(source)
        except ValueError as e:
            (FAILED / f"{pid}.mesh.log").write_text(f"rejected by mesh validation: {e}\n")
            print(f"rejected {pid}  {e}")
            return False

        self.stage("render")
        r = conn.execute("SELECT status, error FROM renders"
                         " WHERE part_id = ? AND file_sha256 = ? AND mesh_ver = ? AND render_ver = ?",
                         (pid, sha, MESH_VER, RENDER_VER)).fetchone()
        if r is None:
            r = render.render_and_save(conn, self.renderer, pid, sha)
        if r["status"] != "ok":  # render_and_save wrote data/failed/{part_id}.render.log
            print(f"rejected {pid}  render: {r['error']}")
            return False

        self.stage("index")
        done = conn.execute("SELECT 1 FROM embeddings WHERE part_id = ? AND model = ? AND file_sha256 = ?"
                            " AND mesh_ver = ? AND render_ver = ?", (pid, MODEL, sha, MESH_VER, RENDER_VER)).fetchone()
        if done is None:
            start = time.perf_counter()
            views = embed_part(self.embedder, pid)
            seconds = round(time.perf_counter() - start, 3)
            payload = {**index.part_payload(conn, pid), "file_sha256": sha, "job_id": self.job["job_id"]}
            index.upsert_part(self.qc, pid, views, payload)  # Qdrant first: a row must never claim a missing point
            conn.execute("INSERT OR REPLACE INTO embeddings (part_id, model, file_sha256, mesh_ver, render_ver,"
                         " job_id, embed_seconds, indexed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (pid, MODEL, sha, MESH_VER, RENDER_VER, self.job["job_id"], seconds, now()))
            conn.commit()
        return True

    def cleanup(self, fid, indexed):
        """Drop what an earlier revision of this file left behind: solids it no longer has, parts now rejected."""
        conn = self.conn
        current = {r[0] for r in conn.execute("SELECT part_id FROM parts WHERE file_id = ?", (fid,))}
        index.delete_stale(self.qc, fid, indexed)
        for table, keep in (("meshes", current), ("renders", current), ("embeddings", set(indexed))):
            for (pid,) in conn.execute(f"SELECT DISTINCT part_id FROM {table}").fetchall():
                if owned_by(pid, fid) and pid not in keep:
                    conn.execute(f"DELETE FROM {table} WHERE part_id = ?", (pid,))
        conn.commit()
        for root in (MESH, RENDERS):
            for d in root.iterdir():
                if d.is_dir() and owned_by(d.name, fid) and d.name not in current:
                    shutil.rmtree(d)

    def fail(self, job, error, permanent):
        self.conn.rollback()  # drop any half-written stage
        if not permanent:
            (FAILED / f"{job['file_id']}.{self.current}.log").write_text(traceback.format_exc())
        if permanent or job["attempts"] >= MAX_ATTEMPTS:
            kill(self.conn, job, f"[{self.current}] {error}")
            return
        delay = RETRY_BASE * 4 ** (job["attempts"] - 1)
        self.conn.execute("UPDATE jobs SET status = 'queued', error = ?, next_run_at = ?, lease_until = NULL"
                          " WHERE job_id = ?", (f"[{self.current}] {error}", now(delay), job["job_id"]))
        self.conn.commit()
        print(f"retry    job {job['job_id']:<4} {job['file_name']}  in {delay}s "
              f"(attempt {job['attempts']}/{MAX_ATTEMPTS}): [{self.current}] {error}")

    def release(self, job):
        """Ctrl-C mid-job: hand the job back as if this attempt never happened."""
        self.conn.rollback()
        self.conn.execute("UPDATE jobs SET status = 'queued', attempts = attempts - 1, lease_until = NULL"
                          " WHERE job_id = ?", (job["job_id"],))
        self.conn.commit()
        print(f"\nreleased job {job['job_id']} back to the queue")


# ---------------------------------------------------------------- commands

def cmd_run(args):
    conn = connect()
    FAILED.mkdir(parents=True, exist_ok=True)
    watcher, worker = Watcher(), Worker(conn)
    print(f"pipeline {PIPELINE_VER}: watching {RAW} every {args.interval}s (Ctrl-C to stop)")
    try:
        while True:
            recover(conn)
            watcher.scan(conn, settle=not args.once)  # --once: assumes nothing is mid-copy
            while job := claim(conn, worker.versions):
                worker.process(job)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass  # a job in progress was already handed back by Worker.process
    conn.close()


def cmd_status(args):
    conn = connect()
    counts = dict(conn.execute("SELECT status, count(*) FROM jobs GROUP BY status").fetchall())
    print(f"pipeline {PIPELINE_VER}   jobs: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print(f"searchable parts: {conn.execute('SELECT count(*) FROM embeddings').fetchone()[0]}")
    rows = conn.execute("SELECT * FROM jobs WHERE status IN ('queued', 'running', 'dead') ORDER BY job_id").fetchall()
    for j in rows:
        when = f"next try {j['next_run_at']}" if j["status"] == "queued" and j["attempts"] else ""
        print(f"  {j['job_id']:<4} {j['status']:8} {j['file_name']:52} attempts {j['attempts']}  {when}"
              + (f"\n       {j['error']}" if j["error"] else ""))
    conn.close()


def cmd_retry(args):
    conn = connect()
    if args.dead:
        jobs = conn.execute("SELECT * FROM jobs WHERE status = 'dead'").fetchall()
    else:
        jobs = conn.execute("SELECT * FROM jobs WHERE job_id = ? AND status = 'dead'", (args.job_id,)).fetchall()
        if not jobs:
            raise SystemExit(f"no dead job {args.job_id}")
    for job in jobs:
        moved, path = FAILED / job["file_name"], RAW / job["file_name"]
        if moved.exists() and not path.exists():
            shutil.move(moved, path)
        conn.execute("UPDATE jobs SET status = 'queued', attempts = 0, error = NULL, next_run_at = ?,"
                     " finished_at = NULL WHERE job_id = ?", (now(), job["job_id"]))
        print(f"requeued job {job['job_id']:<4} {job['file_name']}")
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="watch data/raw and process jobs until Ctrl-C")
    run.add_argument("--interval", type=float, default=10, help="seconds between scans")
    run.add_argument("--once", action="store_true", help="scan once, drain the queue, exit")
    sub.add_parser("status")
    retry = sub.add_parser("retry", help="requeue dead jobs (moves their STEP files back to data/raw)")
    which = retry.add_mutually_exclusive_group(required=True)
    which.add_argument("job_id", nargs="?", type=int)
    which.add_argument("--dead", action="store_true", help="every dead job")
    args = parser.parse_args()
    {"run": cmd_run, "status": cmd_status, "retry": cmd_retry}[args.cmd](args)


if __name__ == "__main__":
    main()
