"""Isolated real-gateway smoke for the Gate (ADR 0007).

CTX_TEST_GATE_MODE=deferred (default): the first decision point returns without waiting,
and the next step of the same session applies a real Gate decision with a boolean label.
CTX_TEST_GATE_MODE=sync: the 300 ms hard timeout falls back and passes the request through.
"""

import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


BIN = Path(__file__).resolve().parents[1] / "target/debug/ctx"


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    base = os.environ["CTX_TEST_LIVE_BASE_URL"]
    model = os.environ["CTX_TEST_LIVE_MODEL"]
    reference = os.environ["CTX_TEST_LIVE_CREDENTIAL_REF"]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        env = dict(os.environ, CTX_HOME=directory)

        def ctx(*args):
            return subprocess.run([BIN, *args], env=env, check=True,
                                  capture_output=True, text=True).stdout.strip()

        ctx("init")
        listen_port = port()
        config = root / "config.yaml"
        config.write_text(config.read_text().replace("port: 7788", f"port: {listen_port}"))
        ctx("model", "set", base, model, "--credential-ref", reference,
            "--upstream-user-agent", "curl/8.0")
        ctx("connect", "demo")
        ctx("remember", "Run migration before deploy payments", "--kind", "lesson",
            "--scope", "payments", "--trigger-text", "deploy payments")
        ctx("automation", "budget", "20")
        ctx("automation", "enable", "gate")
        mode = os.environ.get("CTX_TEST_GATE_MODE", "deferred")
        config.write_text(config.read_text().replace("gate_mode: auto", f"gate_mode: {mode}"))
        token = (root / "token").read_text().strip()
        headers = {"X-Ctx-Token": token, "Content-Type": "application/json",
                   "X-Ctx-Project": "payments"}
        daemon = subprocess.Popen([BIN, "serve"], env=env, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.PIPE)
        try:
            for _ in range(50):
                try:
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{listen_port}/api/v1/health",
                        headers={"X-Ctx-Token": token})
                    with urllib.request.urlopen(request, timeout=2):
                        break
                except urllib.error.URLError:
                    time.sleep(0.1)
            else:
                raise RuntimeError("isolated ctx did not start")
            session = dict(headers, **{"X-Ctx-Session": "gate-live"})

            def send(text):
                body = {"model": model, "messages": [{"role": "user", "content": text}],
                        "max_tokens": 8}
                request = urllib.request.Request(
                    f"http://127.0.0.1:{listen_port}/a/demo/v1/chat/completions",
                    data=json.dumps(body).encode(), headers=session)
                with urllib.request.urlopen(request, timeout=60) as response:
                    outcome = json.load(response)
                    step = response.headers["X-Ctx-Step"]
                    hot = float(response.headers["X-Ctx-Hot-Path-Ms"])
                    status = response.status
                debug_request = urllib.request.Request(
                    f"http://127.0.0.1:{listen_port}/api/v1/debug/steps/{step}",
                    headers={"X-Ctx-Token": token})
                with urllib.request.urlopen(debug_request, timeout=3) as response:
                    debug = json.load(response)
                gate = (debug.get("decision") or {}).get("gate") or {}
                return {"status": status, "step": step, "hot_path_ms": hot,
                        "actual_model": outcome.get("model"), "gate": gate}

            first = send("deploy payments")
            print(json.dumps({k: first[k] for k in ("status", "hot_path_ms", "gate")},
                             ensure_ascii=False))
            assert first["status"] == 200
            if mode == "sync":
                gate = first["gate"]
                assert gate.get("called") is True
                assert gate.get("reason") == "timeout" and gate.get("latency_ms", 1000) <= 500
                return
            assert first["gate"].get("reason") == "deferred_pending"
            assert first["hot_path_ms"] < 300, "deferred Gate must not block the request"
            for _ in range(60):
                if json.loads(ctx("automation", "status"))["llm_calls_today"] >= 1:
                    break
                time.sleep(0.5)
            second = send("continue the payments deploy")
            gate = second["gate"]
            print(json.dumps({k: second[k] for k in ("status", "hot_path_ms", "actual_model", "gate")},
                             ensure_ascii=False))
            assert second["status"] == 200 and gate.get("mode") == "deferred"
            assert gate.get("source_features", {}).get("text") == "deploy payments"
            assert isinstance(gate.get("decision"), bool), "no valid Gate label produced"
        finally:
            daemon.terminate()
            daemon.wait(timeout=5)


if __name__ == "__main__":
    main()
