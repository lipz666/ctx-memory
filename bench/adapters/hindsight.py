"""Hindsight (vectorize-io/hindsight 0.10.2) under test, through its HTTP API.

The server runs separately (bench/external/hindsight-venv, see docs/ctx-m-design.md) with
the benchmark's gateway model; HINDSIGHT_URL points at it. Settings follow Hindsight's own
BEAM configuration in the Agent Memory Benchmark (vectorize-io/agent-memory-benchmark,
src/memory_bench/memory/hindsight.py): one bank per conversation, observations off, the
BEAM retain mission, sessions retained with their date, recall with budget "high".

Modes:
- "1k":  recall max_tokens=1000, no raw chunks (same injection budget as ctx-m);
- "amb": recall max_tokens=12288 plus raw chunks up to 8192 tokens (AMB's BEAM setting).
Both modes share the banks: a bank finished in an earlier run is reused.
"""
import os
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from .base import MemorySystem

URL = os.environ.get("HINDSIGHT_URL", "http://127.0.0.1:8890")

# Verbatim from AMB (_BEAM_RETAIN_MISSION).
BEAM_RETAIN_MISSION = (
    "Extract ALL factual claims the user makes about themselves, their project, "
    "and their experience — including NEGATIVE statements (e.g. 'I have never done X', "
    "'I don't know Y', 'I haven't used Z'). Negative self-assessments and denials "
    "are as important as positive ones. Also preserve contradictions: if the user "
    "says opposite things at different points, extract BOTH statements as separate facts. "
    "Preserve specific numbers, dates, versions, and quantities exactly as stated."
)

MODES = {"1k": {"max_tokens": 1000, "include_chunks": False},
         "amb": {"max_tokens": 12288, "include_chunks": True, "max_chunk_tokens": 8192}}


def format_result(r, chunks, seen):
    """AMB's rendering of one recall result (type, text, dates, chunk text on first use)."""
    lines = [f"**[{r.type}]** {r.text}" if r.type else r.text]
    meta = []
    if r.occurred_start and r.occurred_end and r.occurred_start != r.occurred_end:
        meta.append(f"occurred: {r.occurred_start} – {r.occurred_end}")
    elif r.occurred_start:
        meta.append(f"occurred: {r.occurred_start}")
    if r.mentioned_at:
        meta.append(f"mentioned: {r.mentioned_at}")
    if meta:
        lines.append("_" + " · ".join(meta) + "_")
    if chunks and r.chunk_id and r.chunk_id in chunks and r.chunk_id not in seen:
        lines.append(f"> {chunks[r.chunk_id].text}")
        seen.add(r.chunk_id)
    return "\n".join(lines)


class Hindsight(MemorySystem):
    def __init__(self, store_dir, name="hindsight", mode="1k"):
        self.name, self.mode = name, mode
        self.unbounded = mode == "amb"
        self.dir = Path(store_dir)
        self.created, self.counters = set(), {"retain_seconds": 0.0, "sessions": 0}
        self.lock = threading.Lock()
        self.local = threading.local()

    @property
    def client(self):
        """One client per thread: the sync client's aiohttp session is bound to the event
        loop of the thread that created it."""
        if not hasattr(self.local, "client"):
            from hindsight_client import Hindsight as Client
            self.local.client = Client(base_url=URL, timeout=3600)
        return self.local.client

    def setup(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        for _ in range(600):
            try:
                with urllib.request.urlopen(f"{URL}/health", timeout=5) as response:
                    if response.status == 200:
                        break
            except Exception:  # noqa: BLE001 - server still starting
                time.sleep(2)
        else:
            raise RuntimeError(f"Hindsight at {URL} is not up")

    def _bank(self, ns):
        return f"hs-{ns}"

    def _done(self, ns):
        return self.dir / f"{ns}.done"

    def ingest_session(self, ns, session_id, messages, timestamp, project=None):
        if self._done(ns).exists():
            return
        bank = self._bank(ns)
        with self.lock:
            fresh = bank not in self.created
            self.created.add(bank)
        if fresh:
            try:
                self.client.delete_bank(bank)
            except Exception:  # noqa: BLE001 - no such bank yet
                pass
            self.client.create_bank(bank_id=bank, name=f"BEAM {ns}", retain_mission=BEAM_RETAIN_MISSION,
                                    enable_observations=False)
        date = datetime.strptime(timestamp, "%Y/%m/%d") if timestamp else None
        anchor = date.strftime("%B-%d-%Y") if date else ""
        content = "\n\n".join(f"[{anchor}] {m['role'].capitalize()}: {m['content']}" for m in messages)
        started = time.time()
        self.client.retain(bank_id=bank, content=content, timestamp=date, context=f"Conversation {ns}",
                           document_id=f"{ns}-{session_id}")
        with self.lock:
            self.counters["retain_seconds"] += time.time() - started
            self.counters["sessions"] += 1

    def search(self, ns, query, project=None, limit=20, now=None):
        self._done(ns).touch()
        kwargs = dict(MODES[self.mode])
        if now:
            kwargs["query_timestamp"] = f"{now}T23:59:59"
        response = self.client.recall(bank_id=self._bank(ns), query=query[:1900], budget="high", **kwargs)
        chunks = getattr(response, "chunks", None) or {}
        seen, out, keys = set(), [], set()
        for r in response.results:
            key = r.chunk_id or r.id
            if key in keys:
                continue
            keys.add(key)
            out.append(format_result(r, chunks, seen))
        return out

    def usage(self):
        return dict(self.counters)
