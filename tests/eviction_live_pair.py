"""Freeze one long tool-history request and compare direct vs ctx on the real gateway.

This isolates the eviction mechanism from Agent trajectory variance. It is a
synthetic protocol workload and does not prove per-successful-task savings.
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


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def credential(reference):
    if reference.startswith("env:"):
        return os.environ[reference[4:]]
    if reference.startswith("keychain:"):
        return subprocess.run(["security", "find-generic-password", "-a", "default",
                               "-s", reference[9:], "-w"], check=True,
                              capture_output=True, text=True).stdout.strip()
    raise ValueError("credential must be env: or keychain:")


def call(url, headers, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **headers})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=90) as response:
        value = json.load(response)
        return value, response.headers, round(time.monotonic() - started, 2)


def main():
    base = os.environ["CTX_TEST_LIVE_BASE_URL"].rstrip("/")
    model = os.environ["CTX_TEST_LIVE_MODEL"]
    reference = os.environ["CTX_TEST_LIVE_CREDENTIAL_REF"]
    key = credential(reference)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        env = dict(os.environ, CTX_HOME=directory)

        def ctx(*args):
            return subprocess.run([BIN, *args], env=env, check=True,
                                  capture_output=True, text=True).stdout.strip()

        ctx("init")
        port = free_port()
        config = root / "config.yaml"
        config.write_text(config.read_text().replace("port: 7788", f"port: {port}"))
        ctx("model", "set", base, model, "--credential-ref", reference,
            "--upstream-user-agent", "curl/8.0")
        ctx("connect", "demo")
        token = (root / "token").read_text().strip()
        daemon = subprocess.Popen([BIN, "serve"], env=env, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
        try:
            for _ in range(50):
                try:
                    health_request = urllib.request.Request(
                        f"http://127.0.0.1:{port}/api/v1/health",
                        headers={"X-Ctx-Token": token})
                    with urllib.request.urlopen(health_request, timeout=2):
                        pass
                    break
                except urllib.error.URLError:
                    time.sleep(0.1)
            else:
                raise RuntimeError("isolated ctx did not start")
            noise = "".join(f"[CI] test shard {i:05d}: setup completed; no failure in this line.\n"
                            for i in range(2000))
            messages = [
                {"role": "system", "content": "Answer the final user instruction concisely."},
                {"role": "user", "content": "Inspect the CI log."},
                {"role": "assistant", "tool_calls": [{"id": "call_log", "type": "function",
                     "function": {"name": "read_log", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call_log", "content": noise},
            ]
            for index in range(5):
                messages.extend([{"role": "assistant", "content": f"Log check {index} complete."},
                                 {"role": "user", "content": f"Continue to check stage {index + 1}."}])
            messages.append({"role": "user", "content": "The earlier log is background only. Reply exactly OK."})
            body = {"model": model, "messages": messages, "max_tokens": 16}
            direct, _, direct_seconds = call(f"{base}/chat/completions",
                                             {"Authorization": f"Bearer {key}",
                                              "User-Agent": "curl/8.0"}, body)
            proxy, headers, proxy_seconds = call(
                f"http://127.0.0.1:{port}/a/demo/v1/chat/completions",
                {"X-Ctx-Token": token, "User-Agent": "curl/8.0"}, body)
            step = headers.get("X-Ctx-Step")
            debug_request = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/v1/debug/steps/{step}",
                headers={"X-Ctx-Token": token})
            with urllib.request.urlopen(debug_request, timeout=3) as response:
                debug = json.load(response)
            decision = debug.get("decision") or {}
            report = {"source": "synthetic frozen long CI log", "log_chars": len(noise),
                      "direct_model": direct.get("model"), "proxy_model": proxy.get("model"),
                      "direct_input_tokens": (direct.get("usage") or {}).get("prompt_tokens"),
                      "proxy_input_tokens": (proxy.get("usage") or {}).get("prompt_tokens"),
                      "direct_output_tokens": (direct.get("usage") or {}).get("completion_tokens"),
                      "proxy_output_tokens": (proxy.get("usage") or {}).get("completion_tokens"),
                      "direct_seconds": direct_seconds, "proxy_seconds": proxy_seconds,
                      "evictions": len(decision.get("evictions") or []),
                      "request_modified": decision.get("modified"),
                      "direct_answer_ok": str(direct.get("choices", [{}])[0].get("message", {}).get("content", "")).strip() == "OK",
                      "proxy_answer_ok": str(proxy.get("choices", [{}])[0].get("message", {}).get("content", "")).strip() == "OK"}
            print(json.dumps(report, ensure_ascii=False))
            assert report["evictions"] >= 1 and report["request_modified"] is True
            assert report["direct_answer_ok"] and report["proxy_answer_ok"]
        finally:
            daemon.terminate()
            daemon.wait(timeout=5)


if __name__ == "__main__":
    main()
