"""Optional concurrent gateway pilot with an isolated ctx instance.

Environment: CTX_TEST_LIVE_BASE_URL, CTX_TEST_LIVE_MODEL,
CTX_TEST_LIVE_CREDENTIAL_REF. Each arm sends N short model requests; this
measures real gateway traffic but is not a replacement for a task A/B trial.
"""

import concurrent.futures
import json
import os
from pathlib import Path
import random
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
        return subprocess.run(["security", "find-generic-password", "-a", "default", "-s",
                               reference[9:], "-w"], check=True, capture_output=True,
                              text=True).stdout.strip()
    raise ValueError("credential reference must be env: or keychain:")


def percentile(values, proportion):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * proportion))] if values else None


def main():
    base = os.environ["CTX_TEST_LIVE_BASE_URL"].rstrip("/")
    model = os.environ["CTX_TEST_LIVE_MODEL"]
    reference = os.environ["CTX_TEST_LIVE_CREDENTIAL_REF"]
    count = int(os.environ.get("CTX_TEST_LIVE_COUNT_PER_ARM", "8"))
    workers = int(os.environ.get("CTX_TEST_LIVE_WORKERS", "4"))
    seeded_memories = int(os.environ.get("CTX_TEST_LIVE_SEED_MEMORIES", "0"))
    assert 1 <= count <= 100 and 1 <= workers <= 16
    assert 0 <= seeded_memories <= 100
    key = credential(reference)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        env = dict(os.environ, CTX_HOME=directory)

        def run(*args):
            return subprocess.run([BIN, *args], env=env, check=True,
                                  capture_output=True, text=True)

        run("init")
        port = free_port()
        config = root / "config.yaml"
        config.write_text(config.read_text().replace("port: 7788", f"port: {port}"))
        run("model", "set", base, model, "--credential-ref", reference,
            "--upstream-user-agent", "curl/8.0")
        run("connect", "demo")
        for index in range(seeded_memories):
            run("remember", f"For reply task {index}, keep the final answer concise.",
                "--trigger-text", "Reply")
        token = (root / "token").read_text().strip()
        daemon = subprocess.Popen([BIN, "serve"], env=env, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.PIPE)
        try:
            for _ in range(50):
                try:
                    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/v1/health",
                                                 headers={"X-Ctx-Token": token})
                    with urllib.request.urlopen(req, timeout=2):
                        break
                except urllib.error.URLError:
                    time.sleep(0.1)
            else:
                raise AssertionError("isolated ctx daemon did not start")

            def probe(arm):
                if arm == "direct":
                    url = f"{base}/chat/completions"
                    headers = {"Authorization": f"Bearer {key}"}
                else:
                    url = f"http://127.0.0.1:{port}/a/demo/v1/chat/completions"
                    headers = {"X-Ctx-Token": token}
                headers.update({"Content-Type": "application/json", "User-Agent": "curl/8.0"})
                body = json.dumps({"model": model, "messages": [{"role": "user",
                                 "content": "Reply exactly OK."}], "max_tokens": 8}).encode()
                request = urllib.request.Request(url, data=body, headers=headers)
                started = time.perf_counter()
                try:
                    with urllib.request.urlopen(request, timeout=45) as response:
                        payload = json.load(response)
                        hot = response.headers.get("X-Ctx-Hot-Path-Ms")
                        recalls = response.headers.get("X-Ctx-Recalls")
                        return {"arm": arm, "status": response.status,
                                "elapsed_ms": (time.perf_counter() - started) * 1000,
                                "hot_path_ms": float(hot) if hot else None,
                                "recalls": int(recalls) if recalls else 0,
                                "actual_model": payload.get("model")}
                except urllib.error.HTTPError as error:
                    return {"arm": arm, "status": error.code,
                            "elapsed_ms": (time.perf_counter() - started) * 1000,
                            "hot_path_ms": None}
                except urllib.error.URLError:
                    return {"arm": arm, "status": "network_error",
                            "elapsed_ms": (time.perf_counter() - started) * 1000,
                            "hot_path_ms": None}

            arms = ["direct", "proxy"] * count
            random.Random(20260926).shuffle(arms)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(probe, arms))
            report = {"requests_per_arm": count, "max_concurrency": workers,
                      "seeded_memories": seeded_memories,
                      "note": "Pilot: end-to-end differences include gateway jitter. Proxy hot-path header is measured before upstream dispatch."}
            for arm in ("direct", "proxy"):
                items = [item for item in results if item["arm"] == arm]
                good = [item for item in items if item["status"] == 200]
                report[arm] = {"successful": len(good), "statuses": [item["status"] for item in items],
                               "elapsed_p50_ms": percentile([item["elapsed_ms"] for item in good], 0.50),
                               "elapsed_p95_ms": percentile([item["elapsed_ms"] for item in good], 0.95),
                               "actual_models": sorted({item["actual_model"] for item in good if item["actual_model"]})}
                if arm == "proxy":
                    report[arm]["hot_path_p95_ms"] = percentile(
                        [item["hot_path_ms"] for item in good if item["hot_path_ms"] is not None], 0.95)
                    report[arm]["total_recalls"] = sum(item["recalls"] for item in good)
            print(json.dumps(report, ensure_ascii=False))
            assert report["proxy"]["successful"] > 0 and report["direct"]["successful"] > 0
        finally:
            daemon.terminate()
            daemon.wait(timeout=5)


if __name__ == "__main__":
    main()
