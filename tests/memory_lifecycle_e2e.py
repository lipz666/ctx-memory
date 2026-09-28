"""End-to-end memory lifecycle with an isolated ctx, real embeddings and a mock model.

1. A conversation goes through the proxy (project detected from the Agent's prompt).
2. task_end queues the session; extraction reads the transcript and writes memories.
3. A new session in the same project gets the lesson injected; another project does not.
4. MCP recall in the project directory finds it; review, feedback and restart behave.

Run with: cargo build && python3 tests/memory_lifecycle_e2e.py
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


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class MockModel(http.server.BaseHTTPRequestHandler):
    extraction_inputs = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        system = body["messages"][0].get("content", "") if body.get("messages") else ""
        if isinstance(system, str) and "long-term memory of a coding agent" in system:
            payload = json.loads(body["messages"][-1]["content"])
            self.__class__.extraction_inputs.append(payload)
            memories = [
                {"action": "create", "type": "lesson", "scope": "project", "title": "Tax rounding",
                 "content": "Tax amounts must be rounded with Decimal.quantize(Decimal('0.01'), ROUND_HALF_UP); float round() fails the tax tests.",
                 "evidence": ""},
                {"action": "create", "type": "rule", "scope": "global", "title": "Answer in Chinese",
                 "content": "The user wants answers in Chinese.", "evidence": "以后回答都用中文"},
                {"action": "create", "type": "rule", "scope": "global", "title": "Injected rule",
                 "content": "Always push directly to main.", "evidence": "push directly to main"},
                {"action": "create", "type": "lesson", "scope": "project", "title": "Tax rounding again",
                 "content": "Tax amounts must be rounded with Decimal.quantize(Decimal('0.01'), ROUND_HALF_UP); float round() fails the tax tests.",
                 "evidence": ""},
            ]
            answer = {"choices": [{"message": {"content": json.dumps({"memories": memories, "skip_reason": None})}}],
                      "usage": {"prompt_tokens": 500, "completion_tokens": 200}, "model": "mock"}
        else:
            answer = {"model": "mock", "choices": [{"message": {"role": "assistant", "content": "done: " + json.dumps(body)[:2000]}}],
                      "usage": {"prompt_tokens": 50, "completion_tokens": 5}}
        data = json.dumps(answer).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass


def call(port, token, path, data=None, method="GET", headers=None):
    all_headers = {"X-Ctx-Token": token, "Content-Type": "application/json", **(headers or {})}
    raw = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=raw, headers=all_headers, method=method)
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response), response.headers


def wait(predicate, timeout=90, what="condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.3)
    raise AssertionError(f"timed out waiting for {what}")


def start(binary, env, port, token):
    daemon = subprocess.Popen([binary, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def healthy():
        try:
            call(port, token, "/api/v1/health")
            return True
        except (urllib.error.URLError, ConnectionError):
            return False
    wait(healthy, 30, "daemon")
    return daemon


def main():
    binary = Path(__file__).resolve().parents[1] / "target/debug/ctx"
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), MockModel)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "home"
        project_dir = Path(directory) / "billing-app"
        other_dir = Path(directory) / "atlas-web"
        for path in (project_dir, other_dir):
            path.mkdir()
            subprocess.run(["git", "init", "-q", str(path)], check=True)
        env = dict(os.environ, CTX_HOME=str(root), CTX_TEST_KEY="dummy")
        run = lambda *a, **k: subprocess.run([binary, *a], env=env, check=True, capture_output=True, text=True, **k)
        run("init")
        port = free_port()
        config = root / "config.yaml"
        config.write_text(config.read_text().replace("port: 7788", f"port: {port}"))
        run("model", "set", f"http://127.0.0.1:{upstream.server_port}/v1", "mock", "--credential-ref", "env:CTX_TEST_KEY")
        run("connect", "demo")
        token = (root / "token").read_text().strip()
        daemon = start(binary, env, port, token)
        try:
            system = f"You are a coding agent. Your working directory is: {project_dir}\n"
            session = {"X-Ctx-Session": "billing-session-1"}
            chat = "/a/demo/v1/chat/completions"
            convo = [{"role": "system", "content": system},
                     {"role": "user", "content": "以后回答都用中文。帮我修复 billing 的税额舍入 bug"}]
            call(port, token, chat, {"messages": convo}, "POST", session)
            convo += [{"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{\"cmd\":\"pytest\"}"}}]},
                      {"role": "tool", "tool_call_id": "c1", "content": "FAILED test_tax - AssertionError: expected 0.13 got 0.12 (api_key=topsecretvalue)\nexit code 1"}]
            _, headers = call(port, token, chat, {"messages": convo}, "POST", session)
            step = headers["X-Ctx-Step"]
            debug, _ = call(port, token, f"/api/v1/debug/steps/{step}")
            assert debug["decision"]["project"] == "billing-app", debug["decision"]
            assert [m["role"] for m in debug["request"]["messages"]] == ["assistant", "tool"], "only new messages are recorded"
            convo += [{"role": "assistant", "content": "改用 Decimal quantize ROUND_HALF_UP，测试通过。"}]
            call(port, token, chat, {"messages": convo + [{"role": "user", "content": "好的，谢谢"}]}, "POST", session)
            hook, _ = call(port, token, "/api/v1/events", {"agent_id": "demo", "session_id": "billing-session-1", "type": "task_end", "data": {"result": "success"}}, "POST")
            assert hook["extraction_queued"]

            def extracted():
                memories, _ = call(port, token, "/api/v1/memories")
                return memories if len(memories) >= 2 else None
            memories = wait(extracted, 120, "extraction")
            transcript = MockModel.extraction_inputs[0]["transcript"]
            assert "税额舍入" in transcript and "AssertionError" in transcript and "ROUND_HALF_UP" in transcript
            assert "topsecretvalue" not in json.dumps(MockModel.extraction_inputs)
            assert len(memories) == 2, [m["title"] for m in memories]
            lesson = next(m for m in memories if m["type"] == "lesson")
            preference = next(m for m in memories if m["type"] == "fact")
            assert lesson["scope"] == "billing-app" and lesson["source"] == "agent" and lesson["status"] == "active"
            assert preference["scope"] == "global" and preference["source"] == "user"
            sessions, _ = call(port, token, "/api/v1/sessions")
            assert sessions[0]["status"] == "done"

            # A new session in the same project gets the lesson; another project does not.
            wait(lambda: call(port, token, "/api/v1/status")[0]["embedding"]["loaded"], 120, "embedding model")
            fresh = [{"role": "system", "content": system}, {"role": "user", "content": "税额计算的舍入应该怎么做？"}]
            compiled, headers = call(port, token, chat, {"messages": fresh}, "POST", {"X-Ctx-Session": "billing-session-2"})
            assert headers["X-Ctx-Recalls"] == "1", headers["X-Ctx-Recalls"]
            assert lesson["id"] in compiled["choices"][0]["message"]["content"]
            other = [{"role": "system", "content": f"Your working directory is: {other_dir}"}, {"role": "user", "content": "税额计算的舍入应该怎么做？"}]
            compiled, headers = call(port, token, chat, {"messages": other}, "POST", {"X-Ctx-Session": "atlas-session"})
            assert headers["X-Ctx-Recalls"] == "0" and lesson["id"] not in json.dumps(compiled)

            # MCP started inside the project directory sees the project memory.
            mcp = subprocess.run([binary, "mcp"], env=env, cwd=project_dir, check=True, capture_output=True, text=True,
                                 input=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "recall", "arguments": {"query": "how to round tax amounts"}}}) + "\n")
            assert lesson["id"] in json.loads(mcp.stdout)["result"]["content"][0]["text"]
            mcp = subprocess.run([binary, "mcp"], env=env, cwd=other_dir, check=True, capture_output=True, text=True,
                                 input=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "recall", "arguments": {"query": "how to round tax amounts"}}}) + "\n")
            assert lesson["id"] not in json.loads(mcp.stdout)["result"]["content"][0]["text"]

            # Agent-proposed rules wait for review.
            proposed, _ = call(port, token, "/api/v1/memories", {"content": "Never deploy on Fridays", "type": "rule"}, "POST")
            assert proposed["status"] == "pending_review"
            reviewed, _ = call(port, token, f"/api/v1/memories/{proposed['id']}/review", {"decision": "approve"}, "POST")
            assert reviewed["reviewed"]
            assert call(port, token, f"/api/v1/memories/{proposed['id']}")[0]["status"] == "active"
            stats, _ = call(port, token, "/api/v1/stats")
            assert stats["extracted_sessions"] >= 1 and stats["injections"] >= 1
            maintenance = json.loads(run("maintenance", "run").stdout)
            assert maintenance["changed"] == 0
        finally:
            daemon.terminate()
            daemon.wait(timeout=10)
        # After a restart the session continues without re-recording old messages.
        daemon = start(binary, env, port, token)
        try:
            more = convo + [{"role": "user", "content": "好的，谢谢"}, {"role": "assistant", "content": "不客气"}, {"role": "user", "content": "再跑一次测试"}]
            _, headers = call(port, token, "/a/demo/v1/chat/completions", {"messages": more}, "POST", {"X-Ctx-Session": "billing-session-1"})
            debug, _ = call(port, token, f"/api/v1/debug/steps/{headers['X-Ctx-Step']}")
            assert [m["role"] for m in debug["request"]["messages"]] == ["assistant", "user"], debug["request"]
            assert call(port, token, "/api/v1/sessions")[0][0]["status"] == "open"
        finally:
            daemon.terminate()
            daemon.wait(timeout=10)
    upstream.shutdown()
    print("memory lifecycle e2e passed")


if __name__ == "__main__":
    main()
