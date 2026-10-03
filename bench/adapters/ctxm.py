"""ctx-m prototype under test (bench/ctxm, docs/ctx-m-design.md). Memory per namespace is
kept in <store>/<ns>.json; a store finished in an earlier run is reused (answering again
with another mode or budget does not extract again). An isolated ctx instance serves the
local embeddings."""
import threading
from pathlib import Path

from ctxm.read import Index, Reader
from ctxm.store import Store
from ctxm import write

from .base import MemorySystem
from .ctx import CtxService


class CtxM(MemorySystem):
    def __init__(self, store_dir, name="ctxm", mode="render", budget=1000):
        self.name, self.mode, self.budget = name, mode, budget
        self.dir = Path(store_dir)
        self.stores, self.complete = {}, set()
        self.stats = {}
        self.lock = threading.Lock()
        self.service = None

    def setup(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        self.service = CtxService(self.dir / "embedder-home", embed_workers=2).start()
        self.index = Index(self._embed)
        self.reader = Reader(self.index, self.mode, self.stats)

    def teardown(self):
        if self.service:
            self.service.stop()

    def _embed(self, texts, kind):
        vectors = []
        for start in range(0, len(texts), 128):
            reply = self.service.request("/api/v1/embeddings", {"input": texts[start:start + 128], "kind": kind})
            vectors += [item["embedding"] for item in sorted(reply["data"], key=lambda d: d["index"])]
        return vectors

    def _path(self, ns):
        return self.dir / f"{ns}.json"

    def _store(self, ns):
        with self.lock:
            if ns not in self.stores:
                path = self._path(ns)
                done = path.with_suffix(".done")
                if path.exists() and done.exists():
                    self.stores[ns] = Store.load(path)
                    self.complete.add(ns)
                else:
                    self.stores[ns] = Store()
            return self.stores[ns]

    def ingest_session(self, ns, session_id, messages, timestamp, project=None):
        store = self._store(ns)
        if ns in self.complete:
            return
        date = (timestamp or "").replace("/", "-") or None
        write.ingest(store, session_id, date, messages, self.index, self.stats)
        store.save(self._path(ns))

    def search(self, ns, query, project=None, limit=20, now=None):
        store = self._store(ns)
        if ns not in self.complete:
            store.save(self._path(ns))
            self._path(ns).with_suffix(".done").touch()
            self.complete.add(ns)
        return self.reader.recall(store, query, self.budget, tag=f"ctxm-compose-{self.budget}")

    def release(self, ns):
        with self.lock:
            self.stores.pop(ns, None)

    def usage(self):
        cards = records = 0
        for path in self.dir.glob("*.json"):
            store = Store.load(path)
            cards += len(store.cards)
            records += sum(len(c[k]) for c in store.cards.values() for k in ("values", "items", "events", "notes", "conflicts"))
        return {"engine_llm": dict(self.stats), "cards": cards, "records": records}
