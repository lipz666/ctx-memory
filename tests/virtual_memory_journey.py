"""Synthetic, isolated memory journey: scoping, expiry, review, miss repair, sticky
injection, deduplication and rollback, with real embeddings.

Run with: cargo build && python3 tests/virtual_memory_journey.py
All records live in a TemporaryDirectory; no real gateway or ~/.ctx state is used.
"""

import http.server
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request


BIN = Path(__file__).resolve().parents[1] / "target/debug/ctx"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class EchoModel(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


def call(port, token, path, data=None, headers=None):
    headers = {"X-Ctx-Token": token, "Content-Type": "application/json", **(headers or {})}
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(data).encode() if data is not None else None,
        headers=headers,
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=8) as response:
        return json.load(response), response.headers


def set_field(root, memory_id, old, new):
    path = root / "memory" / f"{memory_id}.md"
    raw = path.read_text()
    assert old in raw, (memory_id, old)
    path.write_text(raw.replace(old, new, 1))


def wait_for(port, token, path, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value, _ = call(port, token, path)
        except urllib.error.URLError:
            time.sleep(0.05)
            continue
        if predicate(value):
            return value
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path}")


def main():
    assert BIN.exists(), "run cargo build first"
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), EchoModel)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = dict(os.environ, CTX_HOME=directory)

            def run(*args):
                return subprocess.run([BIN, *map(str, args)], env=env, check=True,
                                      capture_output=True, text=True).stdout.strip()

            run("init")
            port = free_port()
            config = root / "config.yaml"
            config.write_text(config.read_text().replace("port: 7788", f"port: {port}"))
            run("connect", "demo", f"http://127.0.0.1:{upstream.server_port}/v1")

            payments = run("remember", "Run the schema migration before every payments deploy", "--type", "lesson",
                           "--scope", "payments", "--trigger-text", "deploy payments")
            atlas = run("remember", "Run the schema migration before every atlas deploy", "--type", "lesson",
                        "--scope", "atlas", "--trigger-text", "deploy atlas")
            queue = run("remember", "When the payments queue backlog grows, restart the worker pods", "--type", "fact",
                        "--scope", "payments")
            stale = run("remember", "payments staging endpoint is old.example", "--type", "fact", "--scope", "payments")
            set_field(root, stale, "created_at:", "expires: '2020-01-01T00:00:00Z'\ncreated_at:")
            run("remember", "marketing newsletter publish cadence is biweekly", "--type", "fact")
            archived = run("remember", "retired legacy oauth endpoint at auth-old", "--type", "fact")
            run("forget", archived)
            repair = run("remember", "Inspect the schema mismatch report before retrying a release", "--type", "lesson",
                         "--scope", "payments")
            duplicate_a = run("remember", "Check the release checklist before shipping alpha", "--type", "fact")
            duplicate_b = run("remember", "check the release checklist  before shipping alpha", "--type", "fact")
            assert len(run("memories").splitlines()) == 9

            token = (root / "token").read_text().strip()
            daemon = subprocess.Popen([BIN, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                wait_for(port, token, "/api/v1/status", lambda s: s["embedding"]["loaded"], timeout=120)
                high, _ = call(port, token, "/api/v1/memories", {
                    "content": "Never deploy without a migration check", "type": "lesson", "scope": "payments",
                    "title": "Action guard lesson", "triggers": [{"kind": "tool", "pattern": "deploy", "before_action": True}],
                })
                assert high["status"] == "pending_review"
                trace = root / "trace.jsonl"
                trace.write_text(json.dumps({"query": None, "project": "payments", "moment": "pre_action",
                                             "features": {"tool": "deploy", "args": {}}, "expected_memory_ids": []}) + "\n")
                run("eval", "replay", trace, "--output", root / "before.json")
                assert json.loads((root / "before.json").read_text())["hit_rate"] is None
                reviewed, _ = call(port, token, f"/api/v1/memories/{high['id']}/review", {"decision": "approve"})
                assert reviewed["reviewed"]

                event, _ = call(port, token, "/api/v1/events", {
                    "agent_id": "demo", "project": "payments", "type": "error",
                    "data": {"error_sig": "schema mismatch in orders table", "tool": "deploy"},
                })
                miss, _ = call(port, token, "/api/v1/feedback/miss", {
                    "event_id": event["id"], "memory_id": repair,
                    "reason": "schema mismatch lesson was absent before failed deploy",
                })
                assert miss["id"].startswith("miss_")
                repaired = wait_for(port, token, f"/api/v1/memories/{repair}",
                                    lambda row: any(t["origin"] == "repair" for t in row.get("triggers", [])))
                assert repaired["triggers"][0] == {**repaired["triggers"][0], "kind": "error", "pattern": "schema mismatch in orders table"}

                rows = [
                    {"query": "deploy payments", "project": "payments", "expected_memory_ids": [payments]},
                    {"query": "deploy atlas", "project": "atlas", "expected_memory_ids": [atlas]},
                    {"query": "the payments queue is backing up", "project": "payments", "expected_memory_ids": [queue]},
                    {"query": "payments staging endpoint", "project": "payments", "expected_memory_ids": []},
                    {"query": "retired legacy oauth endpoint", "expected_memory_ids": []},
                    {"query": "weekly hiring meeting", "expected_memory_ids": []},
                    {"query": None, "project": "payments", "moment": "pre_action",
                     "features": {"tool": "deploy", "args": {}}, "expected_memory_ids": [high["id"]]},
                    {"query": None, "project": "payments",
                     "features": {"error_sig": "Error: schema mismatch in orders table at <line>"}, "expected_memory_ids": [repair]},
                    {"query": "deploy payments", "project": "atlas", "expected_memory_ids": []},
                ]
                trace.write_text("\n".join(map(json.dumps, rows)) + "\n")
                run("eval", "replay", trace, "--output", root / "report.json")
                replay = json.loads((root / "report.json").read_text())
                assert replay["memory_recall"] == 1.0, replay
                assert replay["injection_precision"] >= 0.85, replay

                # One injection per session, kept on later steps; a new session gets it again.
                for session in ("s-1", "s-2"):
                    messages = [{"role": "user", "content": "deploy payments"}]
                    body, headers = call(port, token, "/a/demo/v1/chat/completions", {"messages": messages},
                                         headers={"X-Ctx-Project": "payments", "X-Ctx-Session": session})
                    step = headers["X-Ctx-Step"]
                    assert payments in body["messages"][-1]["content"]
                    debug, _ = call(port, token, f"/api/v1/debug/steps/{step}")
                    assert payments in [item["memory_id"] for item in debug["recalls"]]
                    call(port, token, "/api/v1/feedback", {"step_event": step, "memory_id": payments, "cited": True,
                                                              "action_consistent": True, "task_result": "success"})
                    messages += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "go on"}]
                    body, headers = call(port, token, "/a/demo/v1/chat/completions", {"messages": messages},
                                         headers={"X-Ctx-Project": "payments", "X-Ctx-Session": session})
                    assert headers["X-Ctx-Recalls"] == "0" and payments in body["messages"][0]["content"]

                consolidated = json.loads(run("maintenance", "run"))
                assert consolidated["changed"] == 2 and consolidated["batch"], consolidated
                duplicate = wait_for(port, token, f"/api/v1/memories/{duplicate_b}", lambda row: row["status"] == "superseded")
                assert duplicate["superseded_by"] == duplicate_a
                wait_for(port, token, f"/api/v1/memories/{stale}", lambda row: row["status"] == "archived")
                run("maintenance", "rollback", consolidated["batch"])
                wait_for(port, token, f"/api/v1/memories/{duplicate_b}", lambda row: row["status"] == "active")
                stats, _ = call(port, token, "/api/v1/stats")
                assert stats["missed_recalls"] == 1 and stats["labeled_used"] >= 2
                print(json.dumps({"fixture_memories": 10, "labeled_trace_steps": replay["steps"],
                                  "memory_recall": replay["memory_recall"],
                                  "injection_precision": replay["injection_precision"],
                                  "sticky_injection": True, "expiry_and_dedup": True, "batch_rollback": True}))
            finally:
                daemon.terminate()
                daemon.wait(timeout=5)
    finally:
        upstream.shutdown()


if __name__ == "__main__":
    main()
