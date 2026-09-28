"""Run with: cargo build && python3 tests/adapter_smoke.py"""
import json
import os
import pathlib
import socket
import subprocess
import tempfile
import time
import tomllib
import urllib.error
import urllib.request


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    binary = pathlib.Path(__file__).resolve().parents[1] / "target/debug/ctx"
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        fake_bin = root / "originals"
        fake_bin.mkdir()
        for name in ("claude", "codex"):
            path = fake_bin / name
            path.write_text("#!/bin/sh\npython3 -c 'import json,os,sys;print(json.dumps({\"args\":sys.argv[1:],\"base\":os.getenv(\"ANTHROPIC_BASE_URL\")}))' \"$@\"\n")
            path.chmod(0o755)
        env = dict(os.environ, CTX_HOME=directory, CTX_TEST_KEY="dummy", PATH=f"{fake_bin}:{os.environ['PATH']}")
        def run(*args):
            return subprocess.run([binary, *args], env=env, check=True, text=True, capture_output=True)
        run("init")
        port = free_port()
        config = root / "config.yaml"
        config.write_text(config.read_text().replace("port: 7788", f"port: {port}"))
        run("model", "set", "http://127.0.0.1:9999/v1", "test-model", "--credential-ref", "env:CTX_TEST_KEY")
        run("connect", "claude-code")
        run("connect", "codex")
        claude = root / "bin/claude"
        codex = root / "bin/codex"
        fallback_claude = json.loads(subprocess.check_output([claude, "--version"], env=env))["args"]
        assert fallback_claude == ["--mcp-config", str(root / "bin/ctx-mcp.json"), "--version"]
        fallback_codex = json.loads(subprocess.check_output([codex, "--version"], env=env))["args"]
        assert fallback_codex[0] == "-c" and fallback_codex[2] == "--version"
        mcp = tomllib.loads(fallback_codex[1])["mcp_servers"]["ctx"]
        assert mcp["args"] == ["mcp"] and pathlib.Path(mcp["command"]).exists()
        claude_mcp = json.loads((root / "bin/ctx-mcp.json").read_text())
        assert claude_mcp["mcpServers"]["ctx"]["args"] == ["mcp"]
        daemon = subprocess.Popen([binary, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            token = (root / "token").read_text().strip()
            for _ in range(50):
                try:
                    urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/api/v1/health", headers={"X-Ctx-Token": token}), timeout=0.1)
                    break
                except urllib.error.URLError:
                    time.sleep(0.1)
            else:
                raise AssertionError("daemon did not start")
            result = json.loads(subprocess.check_output([claude, "--version"], env=env))
            assert result["base"] == f"http://127.0.0.1:{port}/a/claude-code"
            assert result["args"][:4] == ["--mcp-config", str(root / "bin/ctx-mcp.json"), "--model", "test-model"]
            result = json.loads(subprocess.check_output([codex, "--version"], env=env))
            assert result["args"][0] == "-c" and result["args"][2:4] == ["-c", "model_provider=ctx"]
            provider = tomllib.loads(result["args"][5])
            assert provider["model_providers"]["ctx"]["base_url"] == f"http://127.0.0.1:{port}/a/codex/v1"
        finally:
            daemon.terminate()
            daemon.wait(timeout=5)
    print("adapter smoke test passed")


if __name__ == "__main__":
    main()
