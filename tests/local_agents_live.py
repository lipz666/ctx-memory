"""Optional live acceptance: installed OpenClaw and Hermes against a running ctx.

Run after `ctx init` and `ctx connect openclaw` / `ctx connect hermes`.
This uses temporary Agent homes and never changes the user's Agent settings.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request


CTX_HOME = Path(os.environ.get("CTX_HOME", Path.home() / ".ctx"))
CTX_BIN = CTX_HOME / "bin" / "ctx"
PROXY_TOKEN = (CTX_HOME / "token").read_text().strip()
PORT = 7788


def command(args, env, *, input=None, timeout=30):
    return subprocess.run(
        [str(part) for part in args],
        env=env,
        input=input,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=True,
    )


def assert_health():
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/api/v1/health",
        headers={"X-Ctx-Token": PROXY_TOKEN},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 200


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


def check_openclaw(home):
    binary = shutil.which("openclaw")
    if not binary:
        print("OpenClaw: skipped (not on PATH)")
        return False
    state = home / "openclaw"
    state.mkdir()
    config = state / "openclaw.json"
    config.write_text(
        json.dumps(
            {
                "gateway": {"mode": "local"},
                "agents": {
                    "defaults": {
                        "workspace": str(state / "workspace"),
                        "model": {"primary": "ctx/gemini-3.8-flash-high"},
                    }
                },
                "models": {
                    "mode": "merge",
                    "providers": {
                        "ctx": {
                            "baseUrl": f"http://127.0.0.1:{PORT}/a/openclaw/v1",
                            "api": "openai-completions",
                            "apiKey": "${CTX_PROXY_TOKEN}",
                            "models": [
                                {"id": "gemini-3.8-flash-high", "name": "Gemini 3.8 Flash High"}
                            ],
                        }
                    },
                },
            }
        )
    )
    env = dict(
        os.environ,
        OPENCLAW_STATE_DIR=str(state),
        OPENCLAW_CONFIG_PATH=str(config),
        CTX_PROXY_TOKEN=PROXY_TOKEN,
    )
    command([binary, "config", "validate"], env)
    command(
        [
            binary,
            "mcp",
            "set",
            "ctx",
            json.dumps({"command": str(CTX_BIN), "args": ["mcp"]}),
        ],
        env,
    )
    probe = json.loads(command([binary, "mcp", "probe", "ctx", "--json"], env).stdout)
    assert set(probe["tools"]) == {
        "ctx__recall",
        "ctx__remember",
        "ctx__forget",
        "ctx__expand",
    }
    result = command(
        [
            binary,
            "agent",
            "--local",
            "--agent",
            "main",
            "--message",
            "Reply with exactly OK. Do not use tools.",
            "--json",
            "--thinking",
            "off",
            "--timeout",
            "60",
        ],
        env,
        timeout=75,
    )
    payload = json.loads(result.stdout)
    assert "OK" in set(reply_texts(payload)), "OpenClaw did not return the expected reply"
    print("OpenClaw: Agent turn and four MCP tools passed")
    return True


def check_hermes_mcp(home):
    binary = shutil.which("hermes")
    if not binary:
        print("Hermes: skipped (not on PATH)")
        return False
    state = home / "hermes"
    state.mkdir()
    env = dict(os.environ, HERMES_HOME=str(state))
    added = command(
        [binary, "mcp", "add", "ctx", "--command", CTX_BIN, "--args", "mcp"],
        env,
        input="y\n",
    )
    assert "4/4 tools enabled" in added.stdout
    tested = command([binary, "mcp", "test", "ctx"], env)
    assert "Tools discovered: 4" in tested.stdout
    print("Hermes: four MCP tools passed; model turn not exercised here")
    return True


def main():
    assert_health()
    with tempfile.TemporaryDirectory() as directory:
        home = Path(directory)
        check_openclaw(home)
        check_hermes_mcp(home)


if __name__ == "__main__":
    main()
