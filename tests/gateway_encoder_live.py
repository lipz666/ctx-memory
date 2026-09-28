"""Optional live Encoder check with a synthetic event and an isolated CTX_HOME.

Set CTX_TEST_LIVE_BASE_URL, CTX_TEST_LIVE_MODEL and CTX_TEST_LIVE_CREDENTIAL_REF.
The credential reference points to Keychain or an environment variable; no key is logged.
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


def request(port, token, path, data=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json", "X-Ctx-Token": token},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.load(response)


def main():
    base = os.environ["CTX_TEST_LIVE_BASE_URL"]
    model = os.environ["CTX_TEST_LIVE_MODEL"]
    credential = os.environ["CTX_TEST_LIVE_CREDENTIAL_REF"]
    assert BIN.exists(), "run cargo build first"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        env = dict(os.environ, CTX_HOME=directory)

        def run(*args, check=True):
            return subprocess.run([BIN, *map(str, args)], env=env, check=check,
                                  capture_output=True, text=True, timeout=90)

        run("init")
        port = free_port()
        config = root / "config.yaml"
        config.write_text(config.read_text().replace("port: 7788", f"port: {port}"))
        run("model", "set", base, model, "--credential-ref", credential,
            "--upstream-user-agent", "curl/8.0")
        run("automation", "enable", "encoder")
        token = (root / "token").read_text().strip()
        daemon = subprocess.Popen([BIN, "serve"], env=env, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.PIPE)
        try:
            for _ in range(50):
                try:
                    request(port, token, "/api/v1/health")
                    break
                except urllib.error.URLError:
                    time.sleep(0.1)
            else:
                raise AssertionError("isolated daemon failed to start")
            event = request(port, token, "/api/v1/events", {
                "agent_id": "synthetic", "project": "payments-demo", "type": "user_correction",
                "data": {"text": "When deploying payments-demo, run database migrations first. "
                                 "The previous attempt failed with schema mismatch."},
            })
            result = run("automation", "run", check=False)
            jobs = request(port, token, "/api/v1/automation")["jobs"]
            memories = request(port, token, "/api/v1/memories")
            summary = {"event_recorded": event["id"].startswith("evt_"),
                       "encoder_exit_code": result.returncode,
                       "encoder_result": result.stdout.strip() if result.returncode == 0 else None,
                       "encoder_error": result.stderr.strip()[-500:] if result.returncode else None,
                       "job_statuses": [job["status"] for job in jobs],
                       "memory_count": len(memories),
                       "memory_statuses": [memory["status"] for memory in memories]}
            print(json.dumps(summary, ensure_ascii=False))
            if result.returncode != 0:
                raise AssertionError("live Encoder failed; inspect isolated job status")
            if not any(job["status"] == "done" for job in jobs):
                raise AssertionError("live Encoder did not finish the synthetic job")
            if not memories:
                raise AssertionError("live Encoder produced no memory from a clear user correction")
            memory = memories[0]
            if memory["status"] == "pending_review":
                reviewed = request(port, token, f"/api/v1/memories/{memory['id']}/review",
                                   {"decision": "approve"})
                assert reviewed["reviewed"]
            trace = root / "trace.jsonl"
            trace.write_text(json.dumps({"query": "deploy payments-demo", "project": "payments-demo",
                                         "expected_memory_ids": [memory["id"]]}) + "\n")
            report_path = root / "report.json"
            run("eval", "replay", trace, "--output", report_path)
            report = json.loads(report_path.read_text())
            assert report["memory_recall"] == 1.0, report
            print(json.dumps({"approved_memory_recalled": True,
                              "generated_trigger_count": len(memory["triggers"])},
                             ensure_ascii=False))
        finally:
            daemon.terminate()
            daemon.wait(timeout=5)


if __name__ == "__main__":
    main()
