"""Mem0 open source (mem0ai), default extraction and update pipeline.

One Memory instance per namespace with its own local Qdrant path, so namespaces run in
parallel without sharing state. LLM: the benchmark model via the same gateway. Embedder:
the same EmbeddingGemma model as ctx, through ctx's OpenAI-compatible endpoint.
"""
import os
import shutil
import threading
from pathlib import Path

from .base import MemorySystem
from .ctx import CtxService, namespace


def share_bm25_encoder():
    """Mem0's Qdrant store loads fastembed's Qdrant/bm25 model per instance, checking the
    Hugging Face Hub each time; many instances at once get rate-limited and silently fall
    back to vector-only search. Load it once and share it across instances."""
    import fastembed
    if getattr(fastembed, "_bench_shared", None) is None:
        original = fastembed.SparseTextEmbedding
        encoder = original(model_name="Qdrant/bm25")

        class Shared(original):
            def __new__(cls, *args, **kwargs):
                return encoder

            def __init__(self, *args, **kwargs):
                pass

        fastembed._bench_shared = encoder
        fastembed.SparseTextEmbedding = Shared


class Mem0(MemorySystem):
    name = "mem0"

    def __init__(self, workdir, embed_service: CtxService, reuse=False):
        self.workdir = Path(workdir)
        self.embed = embed_service
        self.instances = {}
        self.lock = threading.Lock()
        # reuse: answer from the stores an earlier run built (no ingestion).
        self.reuse = reuse
        self.tokens = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        self.failed_sessions = []

    def setup(self):
        if not self.reuse:
            if self.workdir.exists():
                shutil.rmtree(self.workdir)
            self.workdir.mkdir(parents=True)
        os.environ.setdefault("MEM0_TELEMETRY", "False")
        share_bm25_encoder()

    def _memory(self, ns):
        with self.lock:
            if ns in self.instances:
                return self.instances[ns]
        from mem0 import Memory
        path = self.workdir / namespace(ns)
        config = {
            "llm": {"provider": "openai", "config": {
                "model": os.environ.get("BENCH_MODEL", "gemini-3.8-flash-high"), "temperature": 0,
                "api_key": os.environ[os.environ.get("BENCH_KEY_ENV", "CTX_GW_KEY")],
                "openai_base_url": os.environ.get("BENCH_BASE_URL", "https://vps.lpzproxy.xyz/v1"), "max_tokens": 2000}},
            "embedder": {"provider": "openai", "config": {
                "model": "embeddinggemma", "api_key": self.embed.token,
                "openai_base_url": self.embed.embeddings_base_url(), "embedding_dims": 768}},
            "vector_store": {"provider": "qdrant", "config": {
                "collection_name": "mem", "path": str(path / "qdrant"), "embedding_model_dims": 768, "on_disk": True}},
            "history_db_path": str(path / "history.db"),
        }
        memory = Memory.from_config(config)
        self._count_tokens(memory)
        with self.lock:
            self.instances.setdefault(ns, memory)
            return self.instances[ns]

    def _count_tokens(self, memory):
        """Record the usage of every model call Mem0 makes (they bypass the harness client)."""
        completions = memory.llm.client.chat.completions
        create = completions.create

        def counted(*args, **kwargs):
            response = create(*args, **kwargs)
            usage = getattr(response, "usage", None)
            with self.lock:
                self.tokens["calls"] += 1
                self.tokens["input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
                self.tokens["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0
            return response

        completions.create = counted

    def ingest_session(self, ns, session_id, messages, timestamp, project=None):
        if self.reuse:
            return None
        memory = self._memory(project or ns)
        clean = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] in ("user", "assistant")]
        # Mem0 OSS has no timestamp parameter (platform only) and dates facts with today's
        # date; stating the conversation date in the text is the usual workaround.
        if clean and timestamp:
            clean[0] = dict(clean[0], content=f"(Conversation date: {timestamp})\n{clean[0]['content']}")
        from common.llm import wait_for_gateway
        for attempt in range(5):
            try:
                return memory.add(clean, user_id="u", metadata={"session_date": timestamp, "session_id": session_id})
            except Exception as error:  # noqa: BLE001 - retried after the gateway recovers
                if attempt == 4:
                    # As for ctx: a session whose extraction keeps failing is left out and
                    # the next sessions are still ingested.
                    with self.lock:
                        self.failed_sessions.append(f"{ns}:{session_id}: {type(error).__name__}: {error}"[:300])
                    return None
                wait_for_gateway()

    def add_memory(self, ns, text, project=None):
        self._memory(project or ns).add([{"role": "user", "content": text}], user_id="u", infer=False)

    def search(self, ns, query, project=None, limit=20):
        return [text for text, _ in self.search_scored(ns, query, project, limit)]

    def search_scored(self, ns, query, project=None, limit=20):
        result = self._memory(project or ns).search(query, filters={"user_id": "u"}, limit=limit)
        items = result.get("results", result) if isinstance(result, dict) else result
        return [(f"[{(item.get('metadata') or {}).get('session_date', '')}] {item['memory']}", item.get("score") or 0.0) for item in items]

    def release(self, ns):
        with self.lock:
            memory = self.instances.pop(ns, None)
        client = getattr(getattr(memory, "vector_store", None), "client", None)
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass

    def usage(self):
        return {"namespaces": len(self.instances), "engine_llm_calls": self.tokens["calls"],
                "engine_input_tokens": self.tokens["input_tokens"], "engine_output_tokens": self.tokens["output_tokens"],
                "failed_sessions": len(self.failed_sessions)}
