"""Run a real OpenClaw task through the local ctx dogfood route.

Uses a persistent, separate OpenClaw profile under ~/.ctx/dogfood. The user's
normal OpenClaw model/config remain unchanged. Pass --verify-command to produce
a task success/failure hook; without it the outcome is unknown.
"""

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid


def reply_texts(value):
    if isinstance(value, dict):
        payloads = value.get("payloads")
        if isinstance(payloads, list):
            for payload in payloads:
                if isinstance(payload, dict) and isinstance(payload.get("text"), str):
                    yield payload["text"].strip()
        for child in value.values():
            yield from reply_texts(child)
    elif isinstance(value, list):
        for child in value:
            yield from reply_texts(child)


def post_event(port, token, project, session, event_type, data):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/v1/events",
        data=json.dumps({"agent_id": "openclaw", "project": project,
                         "session_id": session, "type": event_type, "data": data}).encode(),
        headers={"X-Ctx-Token": token, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--message", required=True)
    parser.add_argument("--verify-command", help="Independent verification command; no shell")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--participant", default=os.environ.get("CTX_DOGFOOD_PARTICIPANT")
                        or getpass.getuser(),
                        help="Short handle counted by tools/dogfood_verdict.py")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        raise ValueError(f"workspace is not a directory: {workspace}")
    binary = shutil.which("openclaw")
    if not binary:
        raise RuntimeError("openclaw is not on PATH")
    root = Path(os.environ.get("CTX_HOME", Path.home() / ".ctx"))
    token = (root / "token").read_text().strip()
    port = 7788
    for line in (root / "config.yaml").read_text().splitlines():
        if line.startswith("port:"):
            port = int(line.split(":", 1)[1].strip())
            break
    health = urllib.request.Request(f"http://127.0.0.1:{port}/api/v1/health",
                                    headers={"X-Ctx-Token": token})
    with urllib.request.urlopen(health, timeout=3) as response:
        if response.status != 200:
            raise RuntimeError("ctx is unhealthy")
    model = "gemini-3.8-flash-high"
    project = workspace.name
    task_id = uuid.uuid4().hex
    session = f"ctx-dogfood-{task_id}"
    profile = root / "dogfood" / hashlib.sha256(str(workspace).encode()).hexdigest()[:12]
    profile.mkdir(mode=0o700, parents=True, exist_ok=True)
    config_path = profile / "openclaw.json"
    config = {"gateway": {"mode": "local"},
              "agents": {"defaults": {"workspace": str(workspace),
                                      "model": {"primary": f"ctx/{model}"}}},
              "models": {"mode": "merge", "providers": {"ctx": {
                  "baseUrl": f"http://127.0.0.1:{port}/a/openclaw/v1",
                  "api": "openai-completions", "apiKey": "${CTX_PROXY_TOKEN}",
                  "request": {"headers": {"X-Ctx-Task": task_id,
                                          "X-Ctx-Project": project,
                                          "X-Ctx-Session": session}},
                  "models": [{"id": model, "name": model}]}}}}
    config_path.write_text(json.dumps(config))
    config_path.chmod(0o600)
    env = dict(os.environ, OPENCLAW_STATE_DIR=str(profile),
               OPENCLAW_CONFIG_PATH=str(config_path), CTX_PROXY_TOKEN=token)
    post_event(port, token, project, session, "task_start",
               {"task_id": task_id, "participant": args.participant})
    started = time.monotonic()
    try:
        result = subprocess.run([binary, "agent", "--local", "--agent", "main",
                                 "--session-id", session, "--message", args.message,
                                 "--json", "--thinking", "off", "--timeout",
                                 str(args.timeout_seconds)], cwd=workspace, env=env,
                                capture_output=True, text=True,
                                timeout=args.timeout_seconds + 30)
        agent_exit = result.returncode
        try:
            agent_reply = json.loads(result.stdout)
        except json.JSONDecodeError:
            agent_reply = {}
    except subprocess.TimeoutExpired:
        agent_exit = 124
        agent_reply = {}
    verify_exit = None
    if args.verify_command:
        command = shlex.split(args.verify_command)
        if not command:
            raise ValueError("verify command is empty")
        try:
            verification = subprocess.run(command, cwd=workspace, timeout=180,
                                          capture_output=True, text=True)
            verify_exit = verification.returncode
        except FileNotFoundError:
            verify_exit = 127
        except subprocess.TimeoutExpired:
            verify_exit = 124
    outcome = "unknown" if verify_exit is None else (
        "success" if agent_exit == 0 and verify_exit == 0 else "failure")
    post_event(port, token, project, session, "task_end",
               {"task_id": task_id, "result": outcome, "participant": args.participant,
                "agent_exit": agent_exit, "verification_exit": verify_exit})
    for answer in reply_texts(agent_reply):
        print(answer)
    print(json.dumps({"task_id": task_id, "result": outcome,
                      "agent_exit": agent_exit, "verification_exit": verify_exit,
                      "elapsed_seconds": round(time.monotonic() - started, 2)},
                     ensure_ascii=False), file=sys.stderr)
    raise SystemExit(0 if agent_exit == 0 and verify_exit in {None, 0} else 1)


if __name__ == "__main__":
    main()
