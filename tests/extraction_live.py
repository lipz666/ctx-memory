"""Live memory lifecycle on a real gateway with a real Agent (OpenClaw), isolated ctx.

1. OpenClaw fixes the noisy billing fixture through the ctx proxy. The user message also
   states a preference and project knowledge that is not in the repository (release branch
   and reviewer). 2. task_end triggers real extraction. 3. A fresh OpenClaw session in a new
   copy of the same project asks about that knowledge; it must be injected and answered.

Environment: CTX_TEST_LIVE_BASE_URL, CTX_TEST_LIVE_MODEL, CTX_TEST_LIVE_CREDENTIAL_REF.
Prints memories and a JSON summary; no credentials are printed.
"""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target/release/ctx"
FIXTURE = ROOT / "tests/fixtures/openclaw_ab/billing"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def call(port, token, path, data=None, method="GET"):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method,
                                 data=json.dumps(data).encode() if data is not None else None,
                                 headers={"X-Ctx-Token": token, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def openclaw(state, workspace, port, token, model, session, message, timeout):
    state.mkdir(parents=True, exist_ok=True)
    config = {"gateway": {"mode": "local"},
              "agents": {"defaults": {"workspace": str(workspace), "model": {"primary": f"ctx/{model}"}}},
              "models": {"mode": "merge", "providers": {"ctx": {
                  "baseUrl": f"http://127.0.0.1:{port}/a/openclaw/v1", "api": "openai-completions",
                  "apiKey": "${CTX_PROXY_TOKEN}", "request": {"headers": {"X-Ctx-Session": session}},
                  "models": [{"id": model, "name": model}]}}}}
    (state / "openclaw.json").write_text(json.dumps(config))
    env = dict(os.environ, OPENCLAW_STATE_DIR=str(state), OPENCLAW_CONFIG_PATH=str(state / "openclaw.json"),
               CTX_PROXY_TOKEN=token)
    started = time.time()
    result = subprocess.run([shutil.which("openclaw"), "agent", "--local", "--agent", "main", "--session-id", session,
                             "--message", message, "--json", "--thinking", "off", "--timeout", str(timeout)],
                            cwd=workspace, env=env, capture_output=True, text=True, timeout=timeout + 60)
    return result.returncode, result.stdout, round(time.time() - started, 1)


def main():
    base, model, reference = (os.environ[k] for k in ("CTX_TEST_LIVE_BASE_URL", "CTX_TEST_LIVE_MODEL", "CTX_TEST_LIVE_CREDENTIAL_REF"))
    summary = {"model": model}
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        home = tmp / "ctx"
        env = dict(os.environ, CTX_HOME=str(home))
        run = lambda *a: subprocess.run([BIN, *a], env=env, check=True, capture_output=True, text=True).stdout
        run("init")
        port = free_port()
        config = home / "config.yaml"
        config.write_text(config.read_text().replace("port: 7788", f"port: {port}").replace("idle_minutes: 10", "idle_minutes: 60"))
        run("model", "set", base, model, "--credential-ref", reference, "--upstream-user-agent", "curl/8.0")
        run("connect", "openclaw")
        token = (home / "token").read_text().strip()
        daemon = subprocess.Popen([BIN, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=open(tmp / "ctxd.log", "w"))
        try:
            for _ in range(100):
                try:
                    call(port, token, "/api/v1/health")
                    break
                except (urllib.error.URLError, ConnectionError):
                    time.sleep(0.2)
            first = tmp / "run1" / "billing"
            shutil.copytree(FIXTURE, first, ignore=shutil.ignore_patterns("__pycache__"))
            subprocess.run(["git", "init", "-q", str(first)], check=True)
            code, _, seconds = openclaw(tmp / "oc1", first, port, token, model, "live-billing-1",
                                        "以后回答我请用中文。另外说明一下：这个项目修好的代码要合到 release/2026-q4 分支，"
                                        "合并前必须找 @wang-li 做代码审核。现在的任务：CI is red for this billing package: read README.md, run "
                                        "`python3 -m unittest discover -s tests`, and fix the code until the suite passes. "
                                        "Do not edit tests or silence logging. You do not need to create branches or PRs.", 600)
            verify = subprocess.run([sys.executable, str(ROOT / "tests/verify_ab_workloads.py"), "billing"], cwd=first, capture_output=True)
            summary["task1"] = {"agent_exit": code, "verified": verify.returncode == 0, "seconds": seconds}
            session = next(s for s in call(port, token, "/api/v1/sessions") if s["key"] == "live-billing-1")
            summary["task1"].update(project=session["project"], steps=session["steps"])
            call(port, token, "/api/v1/events", {"agent_id": "openclaw", "session_id": "live-billing-1", "type": "task_end",
                                                 "data": {"result": "success" if verify.returncode == 0 else "failure"}}, "POST")
            started = time.time()
            while time.time() - started < 600:
                session = next(s for s in call(port, token, "/api/v1/sessions") if s["key"] == "live-billing-1")
                if session["status"] in ("done", "failed", "skipped"):
                    break
                time.sleep(3)
            summary["extraction"] = {"status": session["status"], "error": session["error"], "seconds": round(time.time() - started, 1)}
            memories = call(port, token, "/api/v1/memories")
            summary["memories"] = [{k: m[k] for k in ("type", "scope", "source", "status", "title", "body")} for m in memories]
            for m in memories:
                print(f"- [{m['type']}/{m['scope']}/{m['source']}] {m['title']}\n    {m['body']}")
            # A new session in a new copy of the same project.
            second = tmp / "run2" / "billing"
            shutil.copytree(first, second, ignore=shutil.ignore_patterns("__pycache__", ".git"))
            subprocess.run(["git", "init", "-q", str(second)], check=True)
            code, stdout, seconds = openclaw(tmp / "oc2", second, port, token, model, "live-billing-2",
                                             "修完 bug 之后，代码应该合到哪个分支？合并前要找谁审核？简短回答，不要修改文件。", 300)
            steps = [e for e in call(port, token, "/api/v1/sessions") if e["key"] == "live-billing-2"]
            injected = []
            stats = call(port, token, "/api/v1/stats")
            db_steps = subprocess.run(["sqlite3", str(home / "state/events.db"),
                                       "select s.decision from steps s join events e on e.id=s.event_id where e.session_id='live-billing-2'"],
                                      capture_output=True, text=True).stdout.splitlines()
            for row in db_steps:
                injected += json.loads(row).get("injected", [])
            sys.path.insert(0, str(ROOT / "tools"))
            from openclaw_dogfood import reply_texts
            try:
                answer = " ".join(reply_texts(json.loads(stdout)))
            except json.JSONDecodeError:
                answer = ""
            summary["task2"] = {"agent_exit": code, "seconds": seconds, "steps": steps[0]["steps"] if steps else 0,
                                "injected_memories": len(set(injected)),
                                "answer_has_branch": "release/2026-q4" in answer, "answer_has_reviewer": "wang-li" in answer,
                                "answer": answer[:600]}
            summary["stats"] = {k: stats[k] for k in ("memories", "injections", "extracted_sessions", "engine_llm_calls")}
            summary["extraction_log"] = [line[:1500] for line in (tmp / "ctxd.log").read_text().splitlines() if "extraction" in line]
        finally:
            daemon.terminate()
            daemon.wait(timeout=10)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
