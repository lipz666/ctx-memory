"""ctx under test: an isolated instance per variant, driven through its REST API."""
import json
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .base import MemorySystem

ROOT = Path(__file__).resolve().parents[2]
BIN = Path(os.environ.get("CTX_BIN", ROOT / "target/release/ctx"))


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class CtxService:
    """One isolated ctx daemon. Also serves embeddings for the other systems."""

    def __init__(self, home, prompt_file=None, embed_workers=3, fresh=True, embedding=True, agents=(), recall=None):
        self.agents = agents
        self.recall = recall or {}
        self.home = Path(home)
        self.embedding = embedding
        self.prompt_file = prompt_file
        self.embed_workers = embed_workers
        self.fresh = fresh
        self.port = None
        self.token = None
        self.process = None

    def start(self):
        env = self._env()
        if self.fresh and self.home.exists():
            shutil.rmtree(self.home)
        if not (self.home / "config.yaml").exists():
            subprocess.run([BIN, "init"], env=env, check=True, capture_output=True)
            subprocess.run([BIN, "model", "set", os.environ.get("BENCH_BASE_URL", "https://vps.lpzproxy.xyz/v1"),
                            os.environ.get("BENCH_MODEL", "gemini-3.8-flash-high"), "--credential-ref",
                            "env:" + os.environ.get("BENCH_KEY_ENV", "CTX_GW_KEY"), "--upstream-user-agent", "curl/8.0"],
                           env=env, check=True, capture_output=True)
            for agent in self.agents:
                subprocess.run([BIN, "connect", agent], env=env, check=True, capture_output=True)
        config = (self.home / "config.yaml").read_text()
        self.port = int(re.search(r"^port: (\d+)", config, re.M).group(1))
        if self.fresh:
            self.port = free_port()
            config = re.sub(r"^port: \d+", f"port: {self.port}", config, flags=re.M)
            config = config.replace("history: true", "history: false")
            config = config.replace("global_scope: true", "global_scope: false")
            config = config.replace("daily_llm_calls: 100", "daily_llm_calls: 1000000")
            config = config.replace("idle_minutes: 10", "idle_minutes: 100000")
            config = re.sub(r"workers: \d+", f"workers: {self.embed_workers}", config)
            for key, value in self.recall.items():
                config = re.sub(rf"^  {key}: .*$", f"  {key}: {value}", config, flags=re.M)
            if not self.embedding:
                config = config.replace("embedding:\n  enabled: true", "embedding:\n  enabled: false")
            if self.prompt_file:
                config = config.replace("prompt_file: null", f"prompt_file: {json.dumps(str(self.prompt_file))}")
            (self.home / "config.yaml").write_text(config)
        self.token = (self.home / "token").read_text().strip()
        self.process = subprocess.Popen([BIN, "serve"], env=env, stdout=subprocess.DEVNULL,
                                        stderr=open(self.home / "ctxd.err.log", "a"))
        for _ in range(600):
            try:
                status = self.request("/api/v1/status")
                if status["embedding"]["loaded"] or not self.embedding:
                    return self
            except (urllib.error.URLError, ConnectionError, KeyError):
                pass
            time.sleep(0.5)
        raise RuntimeError(f"ctx at {self.home} did not become ready")

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=30)
            self.process = None

    def _env(self):
        return dict(os.environ, CTX_HOME=str(self.home))

    def request(self, path, data=None, method=None, timeout=600):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", method=method or ("POST" if data is not None else "GET"),
                                     data=json.dumps(data).encode() if data is not None else None,
                                     headers={"X-Ctx-Token": self.token, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)

    def embed(self, texts):
        vectors = []
        for start in range(0, len(texts), 128):
            reply = self.request("/api/v1/embeddings", {"input": texts[start:start + 128]})
            vectors += [item["embedding"] for item in sorted(reply["data"], key=lambda d: d["index"])]
        return vectors

    def embeddings_base_url(self):
        return f"http://127.0.0.1:{self.port}/api/v1"


def render(hit):
    """One retrieved item as the answer model sees it: when it was said, the content, when
    the event happened, and earlier values of a fact that changed."""
    label = {"preference": "(the user's preference) ",
             "instruction": "(the user's standing instruction for the assistant; apply it to this answer) ",
             "reflection": "(summary of what the user has shared about this topic) "}.get(hit.get("type"), "")
    if hit.get("type") in ("timeline", "conflict", "brief"):
        return hit["content"]
    if hit.get("status") == "superseded":
        label = "(earlier statement, changed later) " + label
    said = hit.get('observed_at') or hit.get('created_at') or ''
    # The date of the event itself leads; "said" is when the conversation took place.
    dates = f"event date {hit['event_at']}; said {said}" if hit.get("event_at") else said
    text = f"[{dates}] {label}{hit['content']}"
    if hit.get("mentioned"):
        text = text.replace("] ", f"; mention #{hit['mentioned']}] ", 1)
    again = hit.get("mentioned_at") or []
    if again:
        text += f" (the user brought this up {len(again) + 1} times: first as dated, again on {', '.join(again)})"
    conflicts = hit.get("conflicts") or []
    if conflicts:
        text += " (CONFLICTS with what the user also said: " + "; ".join(f"[{c['date']}] {c['content']}" for c in conflicts) + ")"
    history = hit.get("history") or []
    if history:
        text += " (previously: " + "; ".join(f"[{h['date']}] {h['content']}" for h in history) + ")"
    return text


def namespace(ns):
    return re.sub(r"[^a-z0-9_.-]", "-", ns.lower())


class Ctx(MemorySystem):
    """ctx with its built-in extraction prompt (tuned for coding work), or a variant prompt."""

    def __init__(self, home, name="ctx", prompt_file=None, embed_workers=3, embedding=True, recall=None, mode="search", episodes=0, reuse_from=None, deep=False, budget=None, brief=False):
        self.name = name
        self.failed_sessions = []
        self.mode = mode
        self.episodes = episodes
        self.deep = deep  # plan sub-queries and a date window (one model call per search)
        # Ask ctx to pack results for the reader (memories first, excerpts trimmed); a
        # margin below the harness budget covers the rendering added here.
        self.budget = budget
        # One model call turns a wide retrieval into a brief for the question.
        self.brief = brief
        # Reuse another variant's memory store (same memories, no new extraction) and only
        # build the conversation excerpts; ingestion then does nothing.
        self.reuse_from = reuse_from
        # BENCH_RESUME=1 continues an interrupted run on the same store: finished sessions
        # are skipped, ingested but unfinished ones are only extracted again.
        self.resume = os.environ.get("BENCH_RESUME") == "1" and reuse_from is None and (Path(home) / "config.yaml").exists()
        self.done = {}
        self.service = CtxService(home, prompt_file=prompt_file, embed_workers=embed_workers, embedding=embedding, recall=recall,
                                  fresh=reuse_from is None and not self.resume)

    def setup(self):
        if self.resume:
            import sqlite3
            db = sqlite3.connect(f"file:{self.service.home / 'state/events.db'}?mode=ro", uri=True)
            self.done = dict(db.execute("SELECT key, status FROM sessions"))
            db.close()
        if self.reuse_from:
            if self.service.home.exists():
                shutil.rmtree(self.service.home)
            shutil.copytree(self.reuse_from, self.service.home)
            subprocess.run([BIN, "reindex"], env=self.service._env(), check=True, capture_output=True)
        self.service.start()

    def teardown(self):
        self.service.stop()

    def ingest_session(self, ns, session_id, messages, timestamp, project=None):
        if self.reuse_from:
            return None
        key = f"{namespace(ns)}:{session_id}"
        status = self.done.get(key)
        if status in ("done", "skipped", "failed"):
            return None
        if status is None:
            self.service.request("/api/v1/sessions/ingest", {"session": key, "agent": "bench", "project": namespace(project or ns),
                                                             "observed_at": timestamp, "messages": messages})
        from common.llm import wait_for_gateway
        for attempt in range(5):
            try:
                return self.service.request(f"/api/v1/sessions/{urllib.parse.quote(key, safe='')}/extract?wait=true", {})
            except urllib.error.HTTPError as error:
                if attempt == 4:
                    # Like the daemon, a session whose extraction keeps failing is left out
                    # and the next sessions are still ingested (its excerpts are kept).
                    self.failed_sessions.append(f"{key}: {error.read()[:200]!r}")
                    return None
                wait_for_gateway()

    def add_memory(self, ns, text, project=None):
        self.service.request("/api/v1/memories", {"content": text, "type": "fact", "scope": namespace(project or ns)})

    accepts_now = True  # the question date anchors relative dates in deep search

    def search(self, ns, query, project=None, limit=20, now=None):
        return [text for text, _ in self.search_scored(ns, query, project, limit, now=now)]

    def search_scored(self, ns, query, project=None, limit=20, now=None):
        args = {"q": query, "project": namespace(project or ns), "limit": min(limit, 50), "mode": self.mode,
                "episodes": self.episodes, "deep": str(self.deep).lower()}
        if now:
            args["now"] = now
        if self.budget:
            args["budget"] = int(self.budget * 0.9)
        if self.brief:
            args["brief"] = "true"
        hits = self.service.request(f"/api/v1/recall?{urllib.parse.urlencode(args)}")
        return [(render(h), h["score"]) for h in hits]

    def usage(self):
        stats = self.service.request("/api/v1/stats")
        return {k: stats.get(k) for k in ("memories", "episodes", "engine_llm_calls", "engine_input_tokens",
                                           "engine_output_tokens", "engine_failed_calls", "extracted_sessions")} | {
            "failed_sessions": len(self.failed_sessions)}
