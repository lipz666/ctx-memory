"""Live test of the Hermes memory provider (integrations/hermes/ctx) against an isolated ctx.

  CTX_GW_KEY=... HERMES_SRC=~/hermes ~/.hermes/venvs/hermes-dev/bin/python tests/hermes_provider_live.py

Loads the provider through Hermes' own plugin discovery from a temporary HERMES_HOME,
runs a two-session conversation through its hooks, and checks: turns are recorded,
the session is distilled at session end, the next session's turn gets the memory
injected, the tools work, and a cron context writes nothing.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target/release/ctx"
HERMES_SRC = Path(os.environ.get("HERMES_SRC", Path.home() / "hermes")).expanduser()


def main():
    work = Path(tempfile.mkdtemp(prefix="ctx-hermes-"))
    ctx_home, hermes_home = work / "ctx", work / "hermes"
    env = dict(os.environ, CTX_HOME=str(ctx_home))
    subprocess.run([BIN, "init"], env=env, check=True, capture_output=True)
    subprocess.run([BIN, "model", "set", os.environ.get("CTX_TEST_BASE_URL", "https://vps.lpzproxy.xyz/v1"),
                    os.environ.get("CTX_TEST_MODEL", "gemini-3.8-flash-high"), "--credential-ref", "env:CTX_GW_KEY",
                    "--upstream-user-agent", "curl/8.0"], env=env, check=True, capture_output=True)
    config = (ctx_home / "config.yaml").read_text().replace("history: true", "history: false")
    port = 20000 + os.getpid() % 20000
    import re
    config = re.sub(r"^port: \d+", f"port: {port}", config, flags=re.M)
    (ctx_home / "config.yaml").write_text(config)
    server = subprocess.Popen([BIN, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=open(work / "ctx.err", "w"))
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/health", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        (hermes_home / "plugins").mkdir(parents=True)
        shutil.copytree(ROOT / "integrations/hermes/ctx", hermes_home / "plugins/ctx")
        os.environ.update(CTX_HOME=str(ctx_home), HERMES_HOME=str(hermes_home))
        sys.path.insert(0, str(HERMES_SRC))
        from plugins.memory import discover_memory_providers, load_memory_provider

        names = [entry[0] for entry in discover_memory_providers()]
        assert "ctx" in names, names
        provider = load_memory_provider("ctx")
        assert provider is not None and provider.name == "ctx" and provider.is_available()

        # Session 1: the user mentions facts; the session ends.
        provider.initialize("s1", hermes_home=str(hermes_home), platform="cli", agent_context="primary", agent_identity="tester")
        assert "Long-term memory (ctx)" in provider.system_prompt_block()
        provider.sync_turn("My sister Mira is moving to Lisbon on 2026-11-02, and I promised to help her pack.",
                           "That's exciting! I can help you plan the packing.", session_id="s1")
        provider.sync_turn("Also, I'm allergic to penicillin, keep that in mind for anything medical.",
                           "Noted, I'll keep your penicillin allergy in mind.", session_id="s1")
        provider.on_session_end([])
        provider.shutdown()  # drains the queue (ingest, then extract request)

        token = (ctx_home / "token").read_text().strip()

        def api(path, data=None):
            req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(data).encode() if data is not None else None,
                                         headers={"X-Ctx-Token": token, "Content-Type": "application/json"},
                                         method="POST" if data is not None else "GET")
            with urllib.request.urlopen(req, timeout=300) as response:
                return json.load(response)

        # Extraction was queued by on_session_end; wait for it (the daemon wakes up on request).
        for _ in range(120):
            sessions = [s for s in api("/api/v1/sessions") if s["key"] == "hermes:s1"]
            if sessions and sessions[0]["status"] in ("done", "failed", "skipped"):
                break
            time.sleep(2)
        assert sessions and sessions[0]["status"] == "done", sessions
        # Personal facts may be stored globally (extraction.global_scope) or in the profile scope.
        memories = [m for m in api("/api/v1/memories") if m["scope"] in ("hermes-tester", "global")]
        assert any("penicillin" in m["body"].lower() for m in memories), memories
        # Hermes sessions use the general (personal assistant) extraction prompt.
        assert any("lisbon" in m["body"].lower() for m in memories), memories

        # Session 2: the memory is injected before the turn; tools work.
        provider = load_memory_provider("ctx")
        provider.initialize("s2", hermes_home=str(hermes_home), platform="cli", agent_context="primary", agent_identity="tester")
        injected = provider.prefetch("Can you suggest an antibiotic for my sore throat? I'm allergic to something, I forget what.")
        assert "penicillin" in injected.lower(), injected
        found = json.loads(provider.handle_tool_call("ctx_recall", {"query": "When is Mira moving?"}))
        assert any("2026-11-02" in (r["content"] or "") or "November" in (r["content"] or "") for r in found["results"]), found
        assert any(r["type"] == "episode" for r in found["results"]), found  # raw excerpt tier
        saved = json.loads(provider.handle_tool_call("ctx_remember", {"content": "The user prefers aisle seats on flights."}))
        assert saved["id"].startswith("mem_"), saved
        forgotten = json.loads(provider.handle_tool_call("ctx_forget", {"id": saved["id"]}))
        assert forgotten["archived"] is True, forgotten
        provider.shutdown()

        # A cron context records nothing.
        cron = load_memory_provider("ctx")
        cron.initialize("cron1", hermes_home=str(hermes_home), platform="cron", agent_context="cron", agent_identity="tester")
        cron.sync_turn("Daily digest: nothing new.", "OK.", session_id="cron1")
        cron.shutdown()
        assert not [s for s in api("/api/v1/sessions") if s["key"] == "hermes:cron1"]
        for m in memories:
            print(f"  [{m['scope']}] {m['body']}")
        print(json.dumps({"memories_extracted": len(memories), "injected": injected[:160], "recall_results": len(found["results"])},
                         ensure_ascii=False))
        print("hermes provider live test passed")
    finally:
        server.terminate()
        server.wait(timeout=30)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
