"""Hot-path latency with a large memory store (default 1,000 memories), isolated ctx.

Writes synthetic memory files, embeds them (`ctx reindex`), starts the daemon against a
mock upstream, and sends new user turns in new sessions. Reports the proxy's own time
before forwarding (X-Ctx-Hot-Path-Ms, which includes query embedding and recall).

Run with: cargo build --release && python3 tests/recall_latency.py [--memories 1000]
"""
import argparse
import http.server
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request

BIN = Path(__file__).resolve().parents[1] / "target/release/ctx"


class Echo(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = b'{"choices":[{"message":{"content":"ok"}}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


PROJECTS = [f"svc-{i}" for i in range(10)]
TOPICS = ["数据库迁移", "缓存失效", "部署流程", "单元测试", "日志格式", "权限校验", "重试策略", "配置加载",
          "database migration", "cache invalidation", "deploy pipeline", "unit tests", "log format",
          "auth checks", "retry policy", "config loading", "rate limiting", "feature flags"]
ACTIONS = ["必须先运行 make check", "要用 UTC 时间", "不要直接改生产配置", "先在 staging 验证", "需要加幂等键",
           "must run the linter first", "uses the shared client in lib/http", "requires the VPN", "is owned by the platform team",
           "breaks if the env var is missing", "should log at info level"]


def write_memories(root, count, rng):
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    for i in range(count):
        mid = "mem_" + "".join(rng.choice(alphabet) for _ in range(26))
        topic, action = rng.choice(TOPICS), rng.choice(ACTIONS)
        scope = rng.choice(PROJECTS + ["global"])
        body = f"{scope} 的{topic}：{action}。备注 {i}" if rng.random() < 0.5 else f"In {scope}, {topic} {action} (note {i})."
        (root / "memory" / f"{mid}.md").write_text(
            f"---\nid: {mid}\ntype: {rng.choice(['fact', 'lesson', 'skill'])}\ntitle: '{topic} {i}'\nscope: {scope}\n"
            f"status: active\nsource: user\nconfidence: 0.85\ncreated_at: '2026-09-01T00:00:00Z'\nupdated_at: '2026-09-01T00:00:00Z'\n---\n{body}\n")


def percentile(values, p):
    values = sorted(values)
    return round(values[min(len(values) - 1, int((len(values) - 1) * p))], 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memories", type=int, default=1000)
    parser.add_argument("--requests", type=int, default=60)
    args = parser.parse_args()
    rng = random.Random(7)
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), Echo)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        env = dict(os.environ, CTX_HOME=directory)
        run = lambda *a: subprocess.run([BIN, *a], env=env, check=True, capture_output=True, text=True).stdout
        run("init")
        port = free_port()
        config = root / "config.yaml"
        config.write_text(config.read_text().replace("port: 7788", f"port: {port}"))
        run("connect", "demo", f"http://127.0.0.1:{upstream.server_port}/v1")
        write_memories(root, args.memories, rng)
        started = time.time()
        reindex = run("reindex").strip()
        reindex_seconds = time.time() - started
        token = (root / "token").read_text().strip()
        started = time.time()
        daemon = subprocess.Popen([BIN, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            while True:
                try:
                    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/v1/status", headers={"X-Ctx-Token": token})
                    with urllib.request.urlopen(req, timeout=2) as response:
                        if json.load(response)["embedding"]["loaded"]:
                            break
                except (urllib.error.URLError, ConnectionError):
                    pass
                time.sleep(0.1)
            ready_seconds = time.time() - started
            hot, injected = [], 0
            for i in range(args.requests):
                query = f"{rng.choice(TOPICS)} 在 {rng.choice(PROJECTS)} 里要注意什么？ request {i}"
                body = {"messages": [{"role": "system", "content": "agent"}, {"role": "user", "content": query}]}
                req = urllib.request.Request(f"http://127.0.0.1:{port}/a/demo/p/{rng.choice(PROJECTS)}/v1/chat/completions",
                                             data=json.dumps(body).encode(),
                                             headers={"X-Ctx-Token": token, "Content-Type": "application/json", "X-Ctx-Session": f"lat-{i}"})
                with urllib.request.urlopen(req, timeout=30) as response:
                    response.read()
                    hot.append(float(response.headers["X-Ctx-Hot-Path-Ms"]))
                    injected += int(response.headers["X-Ctx-Recalls"])
        finally:
            daemon.terminate()
            daemon.wait(timeout=10)
    upstream.shutdown()
    print(json.dumps({"memories": args.memories, "reindex": reindex, "reindex_seconds": round(reindex_seconds, 1),
                      "daemon_ready_seconds": round(ready_seconds, 1), "requests": len(hot),
                      "hot_path_ms_p50": percentile(hot, 0.5), "hot_path_ms_p95": percentile(hot, 0.95),
                      "hot_path_ms_max": round(max(hot), 2), "memories_injected": injected,
                      "environment": "local mock upstream, sequential requests, new user turn in a new session each time"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
