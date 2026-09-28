"""Idle zero-overhead check (ADR 0006). Run with: cargo build && python3 tests/idle_overhead_smoke.py

With no memories and nothing to evict, the upstream must receive the Agent's request
byte for byte, including key order, whitespace and escapes. A meter-only agent must
forward byte-identical requests even when memories exist, and still record usage.
"""
import http.server
import json
import os
import pathlib
import socket
import sqlite3
import subprocess
import tempfile
import threading
import time
import urllib.request

RECEIVED = []


class Upstream(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        RECEIVED.append(body)
        reply = json.dumps({"model": "m", "choices": [{"message": {"content": "ok"}}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 1}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *_):
        pass


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(url, raw, token):
    req = urllib.request.Request(url, raw, headers={"Content-Type": "application/json", "X-Ctx-Token": token})
    with urllib.request.urlopen(req, timeout=5) as res:
        return res.read(), res.headers


def main():
    binary = pathlib.Path(__file__).resolve().parents[1] / "target/debug/ctx"
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{upstream.server_port}/v1"
    history = [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "fix the bug"}]
    for i in range(12):
        history.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "read", "arguments": "{}"}}]})
        history.append({"role": "tool", "tool_call_id": f"c{i}", "content": "log line é\n" * 300})
    requests = [
        b'{"stream":false,  "model":"m","messages":[{"role":"user","content":"hi \\u00e9 \xe2\x9c\x93"}],"z":1,"a":2}',
        json.dumps({"model": "m", "messages": history, "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]}, indent=1).encode(),
    ]
    with tempfile.TemporaryDirectory() as directory:
        env = dict(os.environ, CTX_HOME=directory)
        run = lambda *a: subprocess.run([binary, *a], env=env, check=True, capture_output=True, text=True)
        run("init")
        run("connect", "idle", base)
        run("connect", "meter", base, "--meter-only")
        port = free_port()
        config = pathlib.Path(directory) / "config.yaml"
        config.write_text(config.read_text().replace("port: 7788", f"port: {port}"))
        token = (pathlib.Path(directory) / "token").read_text().strip()
        process = subprocess.Popen([binary, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            for _ in range(50):
                try:
                    urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/api/v1/health", headers={"X-Ctx-Token": token}), timeout=1)
                    break
                except OSError:
                    time.sleep(.1)
            for raw in requests:
                RECEIVED.clear()
                _, headers = post(f"http://127.0.0.1:{port}/a/idle/v1/chat/completions", raw, token)
                assert RECEIVED == [raw], "idle request was not forwarded byte for byte"
                assert headers["X-Ctx-Recalls"] == "0"
            run("remember", "Always run migration before deploy", "--kind", "rule")
            time.sleep(.5)
            RECEIVED.clear()
            post(f"http://127.0.0.1:{port}/a/idle/v1/chat/completions", requests[0], token)
            assert RECEIVED[0] != requests[0], "control: a rule memory should change the request"
            for raw in requests:
                RECEIVED.clear()
                post(f"http://127.0.0.1:{port}/a/meter/v1/chat/completions", raw, token)
                assert RECEIVED == [raw], "meter-only agent modified the request"
            time.sleep(.3)
            with sqlite3.connect(pathlib.Path(directory) / "state/events.db") as db:
                metered = db.execute("SELECT COUNT(*) FROM usage WHERE agent_id='meter' AND actual_input_tokens=10").fetchone()[0]
                flagged = db.execute("SELECT COUNT(*) FROM steps s JOIN events e ON e.id=s.event_id WHERE e.agent_id='meter' AND json_extract(s.decision,'$.meter_only')=1").fetchone()[0]
            assert metered == 2 and flagged == 2, (metered, flagged)
        finally:
            process.terminate()
            process.wait(timeout=5)
    upstream.shutdown()
    print("idle overhead smoke test passed")


if __name__ == "__main__":
    main()
