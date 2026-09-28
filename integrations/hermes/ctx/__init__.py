"""ctx memory provider for Hermes Agent.

Long-term memory from the local ctx daemon: every turn is
recorded in ctx, which distils memories when the session goes idle or ends and keeps
the raw conversation as searchable excerpts. Before each turn the confident few
memories for the user's message are injected; the agent can search deeper with
``ctx_recall`` (memories and conversation excerpts), save with ``ctx_remember`` and
archive with ``ctx_forget``. Rules you approved in ctx go into the system prompt.

Use either this provider or ``ctx connect hermes`` (the proxy), not both: each records
the conversation and injects memories on its own.

Config (``memory.ctx`` in config.yaml, all optional):
  url:      ctx API base, default from $CTX_HOME/config.yaml (http://127.0.0.1:<port>)
  project:  memory scope, default "hermes-<profile>"
  inject:   memories injected per turn (0 disables), default 3
Environment: CTX_HOME (default ~/.ctx), CTX_URL, CTX_TOKEN.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_TIMEOUT = 5  # seconds; recall is local and usually well under 100 ms
_MIN_QUERY_LEN = 4


def _ctx_home() -> Path:
    return Path(os.environ.get("CTX_HOME") or Path.home() / ".ctx")


def _plugin_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        section = (load_config().get("memory") or {}).get("ctx") or {}
        return dict(section) if isinstance(section, dict) else {}
    except Exception:
        return {}


def _default_url() -> str:
    try:
        text = (_ctx_home() / "config.yaml").read_text()
        match = re.search(r"^port:\s*(\d+)", text, re.M)
        if match:
            return f"http://127.0.0.1:{match.group(1)}"
    except OSError:
        pass
    return "http://127.0.0.1:7788"


def _token() -> Optional[str]:
    if os.environ.get("CTX_TOKEN"):
        return os.environ["CTX_TOKEN"]
    try:
        return (_ctx_home() / "token").read_text().strip() or None
    except OSError:
        return None


def _text(content: Any) -> str:
    """Plain text of an OpenAI-style message content (string or content parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") in ("text", "input_text", "output_text")
        )
    return ""


RECALL_SCHEMA = {
    "name": "ctx_recall",
    "description": (
        "Search long-term memory from earlier conversations: facts about the user and "
        "their projects, past decisions, lessons, and excerpts of the original "
        "conversations (for exact details such as names, numbers and what was "
        "recommended). Results about one topic are grouped and in chronological order; "
        "prefer the latest value when they conflict."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What you need to know, in natural language."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 30, "description": "Memories to return (default 10)."},
        },
        "required": ["query"],
    },
}

REMEMBER_SCHEMA = {
    "name": "ctx_remember",
    "description": (
        "Save one durable fact, preference, lesson or procedure to long-term memory. "
        "Write one self-contained statement with specifics (names, numbers, dates). "
        "Most memories are extracted automatically when the session ends; use this for "
        "things the user explicitly asks you to remember."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The statement to remember."},
            "type": {"type": "string", "enum": ["fact", "lesson", "skill"], "description": "Default fact."},
        },
        "required": ["content"],
    },
}

FORGET_SCHEMA = {
    "name": "ctx_forget",
    "description": "Archive a memory that is wrong or no longer true (by id from ctx_recall).",
    "parameters": {
        "type": "object",
        "properties": {"id": {"type": "string", "description": "Memory id, e.g. mem_..."}},
        "required": ["id"],
    },
}


class CtxMemoryProvider(MemoryProvider):
    """Hermes memory provider backed by the local ctx daemon's REST API."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = dict(config) if config is not None else _plugin_config()
        self._url = (self._config.get("url") or os.environ.get("CTX_URL") or _default_url()).rstrip("/")
        self._inject = int(self._config.get("inject", 3))
        self._project = self._config.get("project") or ""
        self._session_id = ""
        self._writes = True
        # Turns are sent in order by one background worker so a slow daemon never
        # blocks the agent.
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None

    # -- plumbing -------------------------------------------------------------

    def _request(self, method: str, path: str, data: Any = None, timeout: float = _TIMEOUT) -> Any:
        headers = {"Content-Type": "application/json"}
        token = _token()
        if token:
            headers["X-Ctx-Token"] = token
        body = json.dumps(data).encode() if data is not None else None
        request = urllib.request.Request(self._url + path, data=body, method=method, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def _recall(self, query: str, limit: int, mode: str) -> List[Dict[str, Any]]:
        params = {"q": query[:4000], "project": self._project, "limit": limit, "mode": mode}
        return self._request("GET", "/api/v1/recall?" + urllib.parse.urlencode(params))

    def _run_worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            kind, payload = item
            try:
                if kind == "ingest":
                    self._request("POST", "/api/v1/sessions/ingest", payload, timeout=30)
                elif kind == "extract":
                    key = urllib.parse.quote(payload, safe="")
                    self._request("POST", f"/api/v1/sessions/{key}/extract", {}, timeout=30)
            except Exception as error:  # the daemon may be down; memory is best effort
                logger.debug("ctx %s failed: %s", kind, error)

    def _send(self, kind: str, payload: Any) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run_worker, daemon=True, name="ctx-memory")
            self._worker.start()
        self._queue.put((kind, payload))

    def _session_key(self, session_id: str = "") -> str:
        return f"hermes:{session_id or self._session_id}"

    # -- MemoryProvider -------------------------------------------------------

    @property
    def name(self) -> str:
        return "ctx"

    def is_available(self) -> bool:
        return bool(_token()) or bool(os.environ.get("CTX_URL"))

    def get_config_schema(self):
        return [
            {"key": "url", "description": "ctx API base URL (default: from $CTX_HOME/config.yaml)"},
            {"key": "project", "description": "Memory scope (default: hermes-<profile>)"},
            {"key": "inject", "description": "Memories injected per turn (0 disables)", "default": "3"},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        # Stored under memory.ctx in config.yaml by `hermes memory setup`; nothing else.
        return None

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        # Cron, flush and subagent contexts would pollute the memory of the conversation.
        self._writes = kwargs.get("agent_context", "primary") == "primary"
        if not self._project:
            self._project = f"hermes-{kwargs.get('agent_identity') or 'default'}"

    def system_prompt_block(self) -> str:
        rules = []
        try:
            for memory in self._request("GET", "/api/v1/memories"):
                active = memory.get("status") in ("active", "contested")
                in_scope = memory.get("scope") in ("global", self._project)
                if active and in_scope and (memory.get("type") == "rule" or memory.get("pinned")):
                    rules.append(f"- {(memory.get('body') or '').strip()}")
        except Exception as error:
            logger.debug("ctx rules unavailable: %s", error)
        block = (
            "# Long-term memory (ctx)\n"
            "Relevant memories from earlier conversations are added to user messages automatically. "
            "Use ctx_recall to search memories and original conversation excerpts when you need "
            "details from the past, ctx_remember for things the user asks you to remember, and "
            "ctx_forget for memories that are wrong."
        )
        if rules:
            block += "\n\nStanding rules from the user:\n" + "\n".join(rules)
        return block

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._inject <= 0 or len(query.strip()) < _MIN_QUERY_LEN:
            return ""
        try:
            hits = self._recall(query, self._inject, "inject")
        except Exception as error:
            logger.debug("ctx prefetch failed: %s", error)
            return ""
        lines = []
        for hit in hits:
            date = (hit.get("observed_at") or hit.get("created_at") or "")[:10]
            lines.append(f"- [{date}] {hit.get('content', '').strip()}" if date else f"- {hit.get('content', '').strip()}")
        return "## Long-term memory (ctx)\n" + "\n".join(lines) if lines else ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "",
                  messages: Optional[List[Dict[str, Any]]] = None) -> None:
        if not self._writes:
            return
        turn = [{"role": "user", "content": _text(user_content)}, {"role": "assistant", "content": _text(assistant_content)}]
        turn = [m for m in turn if m["content"].strip()]
        if turn:
            self._send("ingest", {"session": self._session_key(session_id), "agent": "hermes",
                                  "project": self._project, "messages": turn})

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if self._writes:
            self._send("extract", self._session_key())

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False, **kwargs) -> None:
        if reset and self._writes:
            self._send("extract", self._session_key())
        self._session_id = new_session_id

    def on_memory_write(self, action: str, target: str, content: str, metadata=None) -> None:
        """Mirror Hermes' built-in memory additions so they are searchable in ctx too."""
        if self._writes and action in ("add", "replace") and content and content.strip():
            try:
                self._request("POST", "/api/v1/memories", {"content": content.strip(), "type": "fact", "scope": self._project})
            except Exception as error:
                logger.debug("ctx mirror failed: %s", error)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RECALL_SCHEMA, REMEMBER_SCHEMA, FORGET_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            if tool_name == "ctx_recall":
                query = str(args.get("query") or "").strip()
                if not query:
                    return json.dumps({"error": "query is required"})
                hits = self._recall(query, int(args.get("limit") or 10), "search")
                return json.dumps({"results": [
                    {"id": h.get("id"), "type": h.get("type"), "date": (h.get("observed_at") or h.get("created_at") or "")[:19],
                     "group": h.get("group"), "content": h.get("content")} for h in hits
                ]}, ensure_ascii=False)
            if tool_name == "ctx_remember":
                content = str(args.get("content") or "").strip()
                if not content:
                    return json.dumps({"error": "content is required"})
                memory = self._request("POST", "/api/v1/memories",
                                       {"content": content, "type": args.get("type") or "fact", "scope": self._project})
                return json.dumps({"id": memory.get("id"), "status": memory.get("status")})
            if tool_name == "ctx_forget":
                memory_id = str(args.get("id") or "")
                result = self._request("DELETE", "/api/v1/memories/" + urllib.parse.quote(memory_id, safe=""))
                return json.dumps(result)
        except urllib.error.HTTPError as error:
            return json.dumps({"error": f"ctx returned {error.code}: {error.read()[:200].decode(errors='replace')}"})
        except Exception as error:
            return json.dumps({"error": f"ctx unavailable: {error}"})
        return json.dumps({"error": f"unknown tool {tool_name}"})

    def shutdown(self) -> None:
        if self._worker and self._worker.is_alive():
            self._queue.put(None)
            self._worker.join(timeout=10)


def register(ctx) -> None:
    """Hermes plugin entry point."""
    ctx.register_memory_provider(CtxMemoryProvider())
