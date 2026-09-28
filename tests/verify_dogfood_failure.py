"""Black-box check that dogfood tasks always record an outcome hook."""

import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading


ENTRY = Path(__file__).resolve().parents[1] / "tools/openclaw_dogfood.py"


class FakeCtx(http.server.BaseHTTPRequestHandler):
    events = []

    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.events.append(json.loads(body))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")


def main():
    FakeCtx.events = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        binary_dir = root / "bin"
        binary_dir.mkdir()
        fake = binary_dir / "openclaw"
        fake.write_text("#!/bin/sh\nprintf '%s\\n' '{\"payloads\":[{\"text\":\"done\"}]}'\n")
        fake.chmod(0o700)
        workspace = root / "work"
        workspace.mkdir()
        server = http.server.HTTPServer(("127.0.0.1", 0), FakeCtx)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            (root / "token").write_text("test-token")
            (root / "config.yaml").write_text(f"port: {server.server_port}\n")
            env = dict(os.environ, CTX_HOME=directory,
                       PATH=f"{binary_dir}:{os.environ.get('PATH', '')}")
            result = subprocess.run(
                [sys.executable, str(ENTRY), "--workspace", str(workspace),
                 "--message", "probe", "--verify-command", "missing-verifier-executable"],
                env=env, capture_output=True, text=True, timeout=10)
            task_end = [event for event in FakeCtx.events if event.get("type") == "task_end"]
            assert result.returncode == 1, result.returncode
            assert len(task_end) == 1, task_end
            assert task_end[0]["data"]["result"] == "failure", task_end
            assert task_end[0]["data"]["verification_exit"] == 127, task_end
            print("dogfood failure hook passed")
        finally:
            server.shutdown()


if __name__ == "__main__":
    main()
