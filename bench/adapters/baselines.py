"""Baselines: no memory, full history in context, and naive chunk RAG."""
import threading

import numpy as np

from .base import MemorySystem
from .ctx import CtxService


class NoMemory(MemorySystem):
    name = "no-memory"

    def ingest_session(self, ns, session_id, messages, timestamp, project=None):
        pass

    def add_memory(self, ns, text, project=None):
        pass

    def search(self, ns, query, project=None, limit=20):
        return []


def render_session(messages, timestamp):
    lines = [f"[Session date: {timestamp}]"]
    lines += [f"{m['role']}: {m['content']}" for m in messages]
    return "\n".join(lines)


class FullContext(MemorySystem):
    """Every past session verbatim, oldest first. Upper-bound reference, not a memory system."""
    name = "full-context"
    unbounded = True

    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()

    def ingest_session(self, ns, session_id, messages, timestamp, project=None):
        with self.lock:
            self.sessions.setdefault(ns, []).append(render_session(messages, timestamp))

    def add_memory(self, ns, text, project=None):
        self.ingest_session(ns, None, [{"role": "note", "content": text}], "unknown")

    def search(self, ns, query, project=None, limit=20):
        return list(self.sessions.get(ns, []))

    def release(self, ns):
        with self.lock:
            self.sessions.pop(ns, None)


class NaiveRag(MemorySystem):
    """Each user turn with the assistant reply that follows is one chunk; cosine top-k with
    the same embedding model ctx uses (served by a ctx instance)."""
    name = "naive-rag"

    def __init__(self, embed_service: CtxService):
        self.embed = embed_service
        self.chunks = {}
        self.lock = threading.Lock()

    def _store(self, ns, texts):
        vectors = np.asarray(self.embed.embed(texts), dtype=np.float32)
        with self.lock:
            entry = self.chunks.setdefault(ns, {"texts": [], "vectors": np.zeros((0, vectors.shape[1]), np.float32)})
            entry["texts"] += texts
            entry["vectors"] = np.vstack([entry["vectors"], vectors])

    def ingest_session(self, ns, session_id, messages, timestamp, project=None):
        chunks, current = [], []
        for message in messages:
            if message["role"] == "user" and current:
                chunks.append(current)
                current = []
            current.append(message)
        if current:
            chunks.append(current)
        texts = [f"[{timestamp}] " + "\n".join(f"{m['role']}: {m['content']}" for m in chunk)[:6000] for chunk in chunks]
        if texts:
            self._store(ns, texts)

    def add_memory(self, ns, text, project=None):
        self._store(ns, [text])

    def search(self, ns, query, project=None, limit=20):
        entry = self.chunks.get(ns)
        if not entry:
            return []
        query_vector = np.asarray(self.embed.embed([query])[0], dtype=np.float32)
        scores = entry["vectors"] @ query_vector
        order = np.argsort(-scores)[:limit]
        return [entry["texts"][i] for i in order]

    def search_scored(self, ns, query, project=None, limit=20):
        entry = self.chunks.get(ns)
        if not entry:
            return []
        query_vector = np.asarray(self.embed.embed([query])[0], dtype=np.float32)
        scores = entry["vectors"] @ query_vector
        order = np.argsort(-scores)[:limit]
        return [(entry["texts"][i], float(scores[i])) for i in order]

    def release(self, ns):
        with self.lock:
            self.chunks.pop(ns, None)
