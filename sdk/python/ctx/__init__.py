"""Dependency-free Python client for the local ctx API."""

import json
import os
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


class Client:
    def __init__(self, agent_id: str, project: str | None = None,
                 base_url: str = "http://127.0.0.1:7788", token: str | None = None):
        self.agent_id = agent_id
        self.project = project
        self.base_url = base_url.rstrip("/")
        root = Path(os.environ.get("CTX_HOME", Path.home() / ".ctx"))
        self.token = token or (root / "token").read_text().strip()

    def _request(self, method: str, path: str, data: dict | None = None):
        raw = json.dumps(data, ensure_ascii=False).encode() if data is not None else None
        request = Request(self.base_url + path, data=raw, method=method,
                          headers={"X-Ctx-Token": self.token, "Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            return json.load(response)

    def event(self, kind: str, data: dict, session_id: str | None = None):
        return self._request("POST", "/api/v1/events", {
            "agent_id": self.agent_id, "project": self.project,
            "session_id": session_id, "type": kind, "data": data,
        })

    def task_start(self, task_id: str, description: str, session_id: str | None = None):
        return self.event("task_start", {"task_id": task_id, "description": description}, session_id)

    def task_end(self, task_id: str, result: str, session_id: str | None = None):
        """With a session id, the session's memories are extracted right away."""
        return self.event("task_end", {"task_id": task_id, "result": result}, session_id)

    def remember(self, content: str, type: str = "fact", scope: str | None = None):
        """Saved as agent-sourced; a rule waits for user review before it takes effect."""
        return self._request("POST", "/api/v1/memories", {
            "content": content, "type": type, "scope": scope or self.project or "global",
        })

    def recall(self, query: str, limit: int = 5, mode: str = "search", episodes: int | None = None):
        """Hits: id, title, content, type, scope, score, channel (search, trigger, rule or
        episode), observed_at, created_at, group (search mode: hits about one topic share a
        group and come in chronological order).

        mode "search" returns up to `limit` memories by relevance plus raw conversation
        excerpts (type "episode", at most `episodes`, default from config); "inject" returns
        only the confident few the proxy would inject."""
        args = {"q": query, "limit": limit, "mode": mode}
        if episodes is not None:
            args["episodes"] = episodes
        if self.project:
            args["project"] = self.project
        return self._request("GET", "/api/v1/recall?" + urlencode(args))

    def sessions(self):
        return self._request("GET", "/api/v1/sessions")

    def extract_session(self, key: str):
        return self._request("POST", "/api/v1/sessions/" + quote(key, safe="") + "/extract")

    def stats(self):
        return self._request("GET", "/api/v1/stats")

    def expand(self, id: str):
        if id.startswith("mem_"):
            return self._request("GET", "/api/v1/memories/" + id)
        if id.startswith("evt_"):
            return self._request("GET", "/api/v1/events/" + id)
        raise ValueError("Expected a memory or event id")

    def forget(self, id: str):
        return self._request("DELETE", "/api/v1/memories/" + id)

    def feedback(self, step_event: str, memory_id: str, *, cited: bool | None = None,
                 action_consistent: bool | None = None, task_result: str | None = None):
        return self._request("POST", "/api/v1/feedback", {
            "step_event": step_event, "memory_id": memory_id, "cited": cited,
            "action_consistent": action_consistent, "task_result": task_result,
        })

    def missed_recall(self, event_id: str, memory_id: str, reason: str):
        return self._request("POST", "/api/v1/feedback/miss", {
            "event_id": event_id, "memory_id": memory_id, "reason": reason,
        })

    def debug_step(self, step_event: str):
        return self._request("GET", "/api/v1/debug/steps/" + step_event)
