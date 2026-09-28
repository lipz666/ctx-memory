"""Live test of the OpenClaw memory plugin (integrations/openclaw/ctx-memory) against an
isolated ctx and an isolated OpenClaw state directory.

  CTX_GW_KEY=... python3 tests/openclaw_plugin_live.py

Session 1 states a project convention; ctx records it through the plugin's agent_end
hook and distils it. Session 2 (a new session) asks about it: the plugin must inject the
memory before the prompt, and the agent must answer with it. Nothing outside the
temporary directories is touched.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target/release/ctx"
PLUGIN = ROOT / "integrations/openclaw/ctx-memory"
GW_URL = os.environ.get("CTX_TEST_BASE_URL", "https://vps.lpzproxy.xyz/v1")
MODEL = os.environ.get("CTX_TEST_MODEL", "gemini-3.8-flash-high")


def main():
    key = os.environ["CTX_GW_KEY"]
    work = Path(tempfile.mkdtemp(prefix="ctx-openclaw-"))
    ctx_home, state, repo = work / "ctx", work / "openclaw", work / "ledger"
    env = dict(os.environ, CTX_HOME=str(ctx_home))
    subprocess.run([BIN, "init"], env=env, check=True, capture_output=True)
    subprocess.run([BIN, "model", "set", GW_URL, MODEL, "--credential-ref", "env:CTX_GW_KEY", "--upstream-user-agent", "curl/8.0"],
                   env=env, check=True, capture_output=True)
    port = 20000 + os.getpid() % 20000
    config = (ctx_home / "config.yaml").read_text().replace("history: true", "history: false")
    (ctx_home / "config.yaml").write_text(re.sub(r"^port: \d+", f"port: {port}", config, flags=re.M))
    server = subprocess.Popen([BIN, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=open(work / "ctx.err", "w"))
    token = None
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/health", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        token = (ctx_home / "token").read_text().strip()

        def api(path, data=None, method=None):
            req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(data).encode() if data is not None else None,
                                         headers={"X-Ctx-Token": token, "Content-Type": "application/json"},
                                         method=method or ("POST" if data is not None else "GET"))
            with urllib.request.urlopen(req, timeout=300) as response:
                return json.load(response)

        repo.mkdir()
        (repo / "README.md").write_text("# ledger\n\nA small ledger service.\n")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        state.mkdir()
        (state / "openclaw.json").write_text(json.dumps({
            "gateway": {"mode": "local"},
            "agents": {"defaults": {"workspace": str(repo), "model": {"primary": f"bench/{MODEL}"}}},
            "models": {"mode": "merge", "providers": {"bench": {
                "baseUrl": GW_URL, "api": "openai-completions", "apiKey": key, "models": [{"id": MODEL, "name": MODEL}]}}},
            "plugins": {
                "load": {"paths": [str(PLUGIN)]},
                "slots": {"memory": "ctx-memory"},
                "entries": {"ctx-memory": {"enabled": True, "hooks": {"allowConversationAccess": True},
                                           "config": {"project": "ledger"}}},
            },
        }))
        oc_env = dict(os.environ, CTX_HOME=str(ctx_home), OPENCLAW_STATE_DIR=str(state),
                      OPENCLAW_CONFIG_PATH=str(state / "openclaw.json"))

        inspect = subprocess.run(["openclaw", "plugins", "inspect", "ctx-memory", "--runtime", "--json"], cwd=repo, env=oc_env,
                                 capture_output=True, text=True, timeout=120)
        print("inspect:", inspect.returncode, inspect.stdout[:600].replace("\n", " "), inspect.stderr[-400:].replace("\n", " "))

        def agent(session, message):
            proc = subprocess.run(["openclaw", "agent", "--local", "--agent", "main", "--session-id", session, "--message", message,
                                   "--json", "--thinking", "off", "--timeout", "240"], cwd=repo, env=oc_env,
                                  capture_output=True, text=True, timeout=360)
            try:
                payload = json.loads(proc.stdout)
                text = " ".join(p.get("text", "") for p in payload.get("payloads", []) if isinstance(p, dict))
            except ValueError:
                text = proc.stdout[-800:]
            return proc.returncode, text, proc.stderr[-800:]

        code, reply, err = agent("plugin-s1", "Team convention for this repo that is not written down anywhere: changelog lines "
                                 "must use the format `[LEDG-<number>] <description>` and the release branch is always called "
                                 "`release/ledger-next`. Please just acknowledge.")
        print("session 1:", code, reply[:200], err[-300:] if code else "")
        sessions = api("/api/v1/sessions")
        print("ctx sessions:", [(s["key"], s["status"], s["user_turns"]) for s in sessions])
        captured = [s for s in sessions if s["key"].startswith("openclaw:")]
        assert captured, "agent_end did not record the conversation"
        for s in captured:
            result = api(f"/api/v1/sessions/{urllib.parse.quote(s['key'], safe='')}/extract?wait=true", {})
            print("extract:", json.dumps(result)[:300])
        memories = [m for m in api("/api/v1/memories") if m["status"] == "active"]
        for m in memories:
            print(f"  [{m['scope']}] {m['body']}")
        assert any("LEDG" in m["body"] for m in memories), memories

        code, reply, err = agent("plugin-s2", "I'm about to add a changelog line for fixing the rounding bug (ticket 812). "
                                 "Write the exact line I should add, nothing else.")
        print("session 2:", code, reply[:300], err[-300:] if code else "")
        steps = api("/api/v1/stats")
        assert "[LEDG-812]" in reply.replace(" ]", "]"), reply
        print(json.dumps({"memories": len(memories), "session2_reply": reply[:120], "sessions": steps.get("sessions")}))
        print("openclaw plugin live test passed")
    finally:
        server.terminate()
        server.wait(timeout=30)
        if os.environ.get("KEEP"):  # keep the temp dirs to inspect transcripts
            print("kept", work)
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    import urllib.parse  # noqa: F401  (used in api paths)
    main()
