"""Step 7: what the UI loads once per server start, shared by every session.

    models()  BGE-M3 (fp16) + Kiwi for text search, DINOv2 for uploaded parts,
              each run once on dummy input: the first GPU pass builds its kernels,
              and it should not be an engineer's first click that pays for it
    cad()     the child process that extracts, meshes and renders uploads (src.ui.cadworker)
    qdrant()  one client; it is thread-safe

Model calls go through on_tool_thread(): the agent's single tool thread, so two
browser tabs never run BGE-M3, Kiwi or the GPU at the same time (see src.qa.agent).
"""
import json
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import streamlit as st
from PIL import Image

from src.config import ROOT
from src.pipeline.embed import Embedder
from src.pipeline.index import MODEL, client
from src.qa import agent, textindex


_models_loaded = False  # set once models() has run: app.py shows its loading screen only before that


def models_loaded():
    return _models_loaded


def on_tool_thread(fn, *args, **kwargs):
    return agent._tool_thread.submit(fn, *args, **kwargs).result()


class CadWorker:
    """Client of src.ui.cadworker: one request at a time, restarted if it died.

    Every (re)start sends a warm-up request without waiting for it: the process imports cadquery and VTK
    and renders once while the caller does something else. The reply is read before the next request.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.warmup = None  # the warm-up reply, once read

    def start(self):
        with self.lock:
            self._spawn()

    def _spawn(self):
        self.proc = subprocess.Popen([sys.executable, "-m", "src.ui.cadworker"], cwd=ROOT, text=True, bufsize=1,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.proc.stdin.write(json.dumps({"cmd": "warm"}) + "\n")
        self.warmup = None

    def _read(self):
        line = self.proc.stdout.readline()
        if not line:  # it crashed mid-request (OpenCascade can segfault on a bad file)
            return {"ok": False, "error": f"CAD worker exited with code {self.proc.wait()}"}
        return json.loads(line)

    def ready(self):
        """-> the warm-up reply, waiting for it if it hasn't come yet."""
        with self.lock:
            if self.warmup is None:
                self.warmup = self._read()
            return self.warmup

    def call(self, **request):
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._spawn()
            if self.warmup is None:
                self.warmup = self._read()
            self.proc.stdin.write(json.dumps(request) + "\n")
            return self._read()


@st.cache_resource(show_spinner=False)  # app.py draws widgets.splash() instead
def models():
    global _models_loaded
    start = time.perf_counter()

    def load():
        textindex.load_model()
        textindex.tokens("워밍업 warm-up")
        textindex._m3(["warm-up"])
        embedder = Embedder(MODEL)
        embedder([Image.new("RGB", (224, 224), "white")] * 6)  # an upload's batch: 6 views (kernels are per shape)
        return embedder

    embedder = on_tool_thread(load)
    _models_loaded = True
    return SimpleNamespace(embedder=embedder, seconds=round(time.perf_counter() - start, 1))


@st.cache_resource(show_spinner=False)
def cad():
    """Started before models() in app.py: its imports then run while the models load."""
    worker = CadWorker()
    worker.start()
    return worker


@st.cache_resource
def qdrant():
    return client()
