"""Exercise nonstream ActionGuard recheck and SSE fail-open with a mock provider."""

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


class Provider(http.server.BaseHTTPRequestHandler):
    calls = 0
    fail_recheck = False

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.__class__.calls += 1
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"deploy"}}]}}]}\n\ndata: [DONE]\n\n')
            return
        is_recheck = "Before executing the proposed tool call" in body["messages"][-1]["content"]
        if is_recheck and self.__class__.fail_recheck:
            self.send_response(503)
            self.end_headers()
            return
        target = "staging" if is_recheck else "production"
        response = {"choices": [{"message": {"tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": "deploy", "arguments": json.dumps({"service": "payments", "environment": target})
        }}]}}], "usage": {"prompt_tokens": 12, "completion_tokens": 8}}
        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_):
        pass


def call(port, token, path, data=None, method="GET"):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(data).encode() if data is not None else None,
                                 method=method, headers={"X-Ctx-Token": token, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as response:
        return response.read(), response.headers


def main():
    binary = Path(__file__).resolve().parents[1] / "target/debug/ctx"
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), Provider)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        env = dict(os.environ, CTX_HOME=directory)
        def run(*args):
            return subprocess.run([binary, *args], env=env, check=True, capture_output=True, text=True)
        run("init")
        port = free_port()
        config = root / "config.yaml"
        config.write_text(config.read_text().replace("port: 7788", f"port: {port}"))
        run("connect", "demo", f"http://127.0.0.1:{upstream.server_port}/v1")
        run("automation", "guard", "recheck")
        token = (root / "token").read_text().strip()
        daemon = subprocess.Popen([binary, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            for _ in range(50):
                try:
                    call(port, token, "/api/v1/health")
                    break
                except urllib.error.URLError:
                    time.sleep(0.1)
            else:
                raise AssertionError("daemon failed to start")
            memory, _ = call(port, token, "/api/v1/memories", {
                "content": "Check the target before deploying payments", "type": "lesson", "scope": "global",
                "triggers": [{"kind": "tool", "pattern": "deploy", "before_action": True}],
            }, "POST")
            memory = json.loads(memory)
            assert memory["triggers"][0]["before_action"] is True
            assert memory["status"] == "pending_review"
            call(port, token, "/api/v1/memories/" + memory["id"] + "/review", {"decision": "approve"}, "POST")
            body = {"model": "test", "messages": [{"role": "user", "content": "Deploy payments"}],
                    "tools": [{"type": "function", "function": {"name": "deploy", "parameters": {"type": "object"}}}]}
            raw, headers = call(port, token, "/a/demo/v1/chat/completions", body, "POST")
            result = json.loads(raw)
            args = json.loads(result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
            assert args["environment"] == "staging", result
            assert headers["X-Ctx-Recheck"] == "changed"
            assert Provider.calls == 2
            Provider.fail_recheck = True
            raw, headers = call(port, token, "/a/demo/v1/chat/completions", body, "POST")
            fallback = json.loads(raw)
            fallback_args = json.loads(fallback["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
            assert fallback_args["environment"] == "production" and "X-Ctx-Recheck" not in headers
            assert Provider.calls == 4
            Provider.fail_recheck = False
            body["stream"] = True
            raw, headers = call(port, token, "/a/demo/v1/chat/completions", body, "POST")
            assert raw.startswith(b"data:") and "X-Ctx-Recheck" not in headers
            assert Provider.calls == 5
        finally:
            daemon.terminate()
            daemon.wait(timeout=5)
            upstream.shutdown()
    print("action guard smoke test passed")


if __name__ == "__main__":
    main()
