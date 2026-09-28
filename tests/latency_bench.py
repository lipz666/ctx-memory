"""Measure local proxy overhead: cargo build && python3 tests/latency_bench.py."""
import http.server
import json
import os
import pathlib
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request


class Upstream(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        data = b'{"choices":[],"usage":{"prompt_tokens":8,"completion_tokens":1}}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    binary = pathlib.Path(__file__).resolve().parents[1] / "target/debug/ctx"
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as directory:
        env = dict(os.environ, CTX_HOME=directory)
        subprocess.run([binary, "init"], env=env, check=True, capture_output=True)
        subprocess.run([binary, "connect", "bench", f"http://127.0.0.1:{upstream.server_port}/v1"], env=env, check=True, capture_output=True)
        port = free_port()
        config = pathlib.Path(directory) / "config.yaml"
        config.write_text(config.read_text().replace("port: 7788", f"port: {port}"))
        token = (pathlib.Path(directory) / "token").read_text().strip()
        process = subprocess.Popen([binary, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            url = f"http://127.0.0.1:{port}/a/bench/v1/chat/completions"
            for _ in range(50):
                try:
                    urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/api/v1/health", headers={"X-Ctx-Token": token}), timeout=.1)
                    break
                except urllib.error.URLError:
                    time.sleep(.1)
            else:
                raise RuntimeError("daemon did not start")
            payload = json.dumps({"messages":[{"role":"user","content":"hello"}]}).encode()
            def timed(target, auth=False):
                headers = {"Content-Type":"application/json"}
                if auth:
                    headers["X-Ctx-Token"] = token
                started = time.perf_counter()
                with urllib.request.urlopen(urllib.request.Request(target, data=payload, headers=headers), timeout=5) as response:
                    response.read()
                return (time.perf_counter()-started)*1000
            direct_url = f"http://127.0.0.1:{upstream.server_port}/v1/chat/completions"
            for _ in range(10):
                timed(direct_url)
                timed(url, True)
            direct = [timed(direct_url) for _ in range(100)]
            proxied = [timed(url, True) for _ in range(100)]
            p95 = lambda samples: sorted(samples)[int(len(samples)*.95)-1]
            report = {"sample_count":100,"direct_p95_ms":round(p95(direct),2),"proxy_p95_ms":round(p95(proxied),2),"overhead_p95_ms":round(p95(proxied)-p95(direct),2),"environment":"local mock upstream, one process, no concurrent load"}
            print(json.dumps(report,ensure_ascii=False))
        finally:
            process.terminate()
            process.wait(timeout=5)
    upstream.shutdown()


if __name__ == "__main__":
    main()
