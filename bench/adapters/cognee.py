"""Cognee 1.6.2 under test (bench/tools/cognee_beam.py, run in bench/external/cognee-venv).

Ingestion follows Cognee's own BEAM pipeline and runs before the track, one Cognee root per
conversation (<store>/beam-<id>, marker file "ingested"); `ingest_session` only checks that
it happened. Retrieval: Cognee's reported 100K setting (HybridRetriever, chunks_top_k=20,
entities_top_k=20), served per conversation by a subprocess speaking JSON lines.

Modes:
- "native": Cognee's whole hybrid context (about 25k tokens per question);
- "1k":     the head of that ranked context, cut at paragraph boundaries to the budget.
"""
import json
import os
import subprocess
import threading
from pathlib import Path

from .base import MemorySystem

BENCH = Path(__file__).resolve().parents[1]
PYTHON = BENCH / "external/cognee-venv/bin/python"
TOOLS = BENCH / "tools"


class Cognee(MemorySystem):
    def __init__(self, store_dir, name="cognee", mode="1k"):
        self.name, self.mode = name, mode
        self.unbounded = mode == "native"
        self.dir = Path(store_dir)
        self.bridges = {}
        self.lock = threading.Lock()

    def _root(self, ns):
        return self.dir / ns

    def ingest_session(self, ns, session_id, messages, timestamp, project=None):
        if not (self._root(ns) / "ingested").exists():
            raise RuntimeError(f"Cognee store for {ns} is not ingested (run cognee_beam.py ingest first)")

    def _bridge(self, ns):
        with self.lock:
            bridge = self.bridges.get(ns)
            if bridge is None:
                env = dict(os.environ)
                env_script = TOOLS / "cognee.env.sh"
                exported = subprocess.run(["bash", "-c", f"source {env_script} && env -0"], env=env,
                                          capture_output=True, check=True).stdout
                env = dict(item.split("=", 1) for item in exported.decode().split("\0") if "=" in item)
                process = subprocess.Popen([str(PYTHON), str(TOOLS / "cognee_beam.py"), "serve", "--root", str(self._root(ns))],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                           env=env, text=True, bufsize=1)
                ready = json.loads(process.stdout.readline())
                assert ready.get("ready"), ready
                bridge = self.bridges[ns] = (process, threading.Lock())
        return bridge

    def search(self, ns, query, project=None, limit=20, now=None):
        process, lock = self._bridge(ns)
        with lock:
            process.stdin.write(json.dumps({"query": query}) + "\n")
            process.stdin.flush()
            reply = json.loads(process.stdout.readline())
        if "error" in reply:
            raise RuntimeError(reply["error"])
        context = reply["context"]
        if self.mode == "native":
            return [context]
        # Paragraphs in Cognee's order; beam.py keeps them while they fit the budget.
        return [block.strip() for block in context.split("\n\n") if block.strip()]

    def release(self, ns):
        with self.lock:
            bridge = self.bridges.pop(ns, None)
        if bridge:
            bridge[0].stdin.close()
            bridge[0].wait(timeout=60)

    def teardown(self):
        for ns in list(self.bridges):
            self.release(ns)
