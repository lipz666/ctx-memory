"""Proxy, API, MCP and SDK smoke test against a mock upstream (keyword recall only).

Run with: cargo build && python3 tests/proxy_smoke.py
"""
import http.client
import http.server
import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request


class Upstream(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        value = json.loads(body)
        if value.get("force_error"):
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"rate limited"}}')
            return
        if value.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in (b"data: {\"part\":1}\n\n", b"data: [DONE]\n\n"):
                self.wfile.write(chunk)
                self.wfile.flush()
            return
        if value.get("force_actual_model"):
            value["model"] = "actual-model"
            value["usage"] = {"prompt_tokens": 100, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 20}}
            body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def request(url, body, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Ctx-Token"] = token
    req = urllib.request.Request(url, json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=10) as res:
        return json.loads(res.read()), res.headers


def get(url, token):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"X-Ctx-Token": token}), timeout=10) as res:
        return json.load(res)


def main():
    binary = pathlib.Path(__file__).resolve().parents[1] / "target/debug/ctx"
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as directory:
        env = dict(os.environ, CTX_HOME=directory)
        run = lambda *a: subprocess.run([binary, *a], env=env, check=True, capture_output=True, text=True)
        run("init")
        config = pathlib.Path(directory) / "config.yaml"
        port = free_port()
        text = config.read_text().replace("port: 7788", f"port: {port}")
        config.write_text(text.replace("embedding:\n  enabled: true", "embedding:\n  enabled: false"))
        run("connect", "demo", f"http://127.0.0.1:{upstream.server_port}/v1")
        rule_id = run("remember", "Always run migration before deploy", "--kind", "rule").stdout.strip()
        trigger_id = run("remember", "Run the database migration first", "--kind", "lesson", "--trigger-text", "rollback").stdout.strip()
        trace = pathlib.Path(directory) / "trace.jsonl"
        trace.write_text(json.dumps({"query": "rollback", "expected_memory_ids": [trigger_id]}) + "\n")
        assert json.loads(run("replay", str(trace)).stderr)["hit_rate"] == 1.0
        report_path = pathlib.Path(directory) / "eval-report.json"
        run("eval", "replay", str(trace), "--output", str(report_path))
        assert json.loads(report_path.read_text())["memory_recall"] == 1.0
        mcp_requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "expand", "arguments": {"id": rule_id}}},
        ]
        mcp = subprocess.run([binary, "mcp"], env=env, check=True, input="\n".join(map(json.dumps, mcp_requests)) + "\n", text=True, capture_output=True)
        mcp_responses = [json.loads(line) for line in mcp.stdout.splitlines()]
        assert mcp_responses[0]["result"]["protocolVersion"] == "2025-11-25"
        assert [t["name"] for t in mcp_responses[1]["result"]["tools"]] == ["recall", "remember", "forget", "expand"]
        assert rule_id in mcp_responses[2]["result"]["content"][0]["text"]
        token = (pathlib.Path(directory) / "token").read_text().strip()
        process = subprocess.Popen([binary, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            url = f"http://127.0.0.1:{port}/a/demo/v1/chat/completions"
            for _ in range(100):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/memories", timeout=0.2)
                except urllib.error.HTTPError as error:
                    if error.code == 401:
                        break
                except urllib.error.URLError:
                    time.sleep(0.1)
            else:
                raise AssertionError("server did not start")
            # The rule goes into the system prompt on every step.
            body = {"messages": [{"role": "system", "content": "original"}, {"role": "user", "content": "deploy now"}]}
            result, headers = request(url, body, token)
            assert result["messages"][0]["content"].startswith("original")
            assert rule_id in result["messages"][0]["content"]
            assert headers["X-Ctx-Recalls"] == "0" and headers["X-Ctx-Step"]
            # A trigger memory attaches to the user's message and stays there on later steps.
            convo = [{"role": "system", "content": "s"}, {"role": "user", "content": "please rollback the release"}]
            first, headers = request(url, {"messages": convo}, token)
            assert headers["X-Ctx-Recalls"] == "1" and trigger_id in first["messages"][1]["content"]
            later = convo + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "continue"}]
            second, headers = request(url, {"messages": later}, token)
            assert headers["X-Ctx-Recalls"] == "0"
            assert trigger_id in second["messages"][1]["content"], "memory must stay attached"
            assert trigger_id not in second["messages"][3]["content"]
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            chunked = json.dumps(body).encode()
            connection.request("POST", "/a/demo/v1/chat/completions", body=(chunked[i:i + 10] for i in range(0, len(chunked), 10)), headers={"Content-Type": "application/json", "X-Ctx-Token": token}, encode_chunked=True)
            chunked_response = connection.getresponse()
            assert chunked_response.status == 200 and chunked_response.getheader("X-Ctx-Step")
            assert "ctx-memory" in json.loads(chunked_response.read())["messages"][0]["content"]
            connection.close()
            sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "sdk/python"))
            from ctx import Client
            sdk = Client("smoke", base_url=f"http://127.0.0.1:{port}", token=token)
            event_id = sdk.event("task_start", {"task_id": "smoke"})["id"]
            assert sdk.expand(event_id)["type"] == "task_start"
            assert sdk.expand(rule_id)["id"] == rule_id
            assert sdk.recall("database migration rollback")[0]["id"] == trigger_id
            edit_req = urllib.request.Request(f"http://127.0.0.1:{port}/api/v1/memories/{trigger_id}", data=json.dumps({"content": "Run the database migration before rollback"}).encode(), headers={"Content-Type": "application/json", "X-Ctx-Token": token}, method="PATCH")
            with urllib.request.urlopen(edit_req, timeout=5) as response:
                edited = json.load(response)
            assert edited["body"] == "Run the database migration before rollback"
            assert sdk.expand(trigger_id)["body"] == edited["body"]
            request(url, {"model": "requested-model", "force_actual_model": True, "messages": [{"role": "user", "content": "usage test"}]}, token)
            for _ in range(40):
                stats = get(f"http://127.0.0.1:{port}/api/v1/stats", token)
                if stats["input_tokens"] >= 100:
                    break
                time.sleep(.05)
            assert stats["cached_input_tokens"] >= 20 and stats["model_mismatch_steps"] >= 1
            assert stats["sessions"] >= 2
            ticket_req = urllib.request.Request(f"http://127.0.0.1:{port}/api/v1/ui-ticket", data=b"", headers={"X-Ctx-Token": token})
            with urllib.request.urlopen(ticket_req, timeout=5) as response:
                ticket = json.load(response)["url"]
            with urllib.request.urlopen(ticket, timeout=5) as response:
                assert "ctxSession" in response.read().decode()
            try:
                urllib.request.urlopen(ticket, timeout=5)
            except urllib.error.HTTPError as error:
                assert error.code == 401
            else:
                raise AssertionError("UI ticket was reusable")
            responses, _ = request(f"http://127.0.0.1:{port}/a/demo/v1/responses", {"input": "deploy now"}, token)
            assert "ctx-memory" in responses["instructions"]
            anthropic, _ = request(f"http://127.0.0.1:{port}/a/demo/v1/messages", {"messages": [{"role": "user", "content": "deploy now"}]}, token)
            assert "ctx-memory" in anthropic["system"]
            # Project routes scope memories.
            run("remember", "billing deploys use make ship", "--scope", "billing", "--trigger-text", "ship it")
            time.sleep(0.5)
            routed, _ = request(f"http://127.0.0.1:{port}/a/demo/p/billing/v1/chat/completions", {"messages": [{"role": "user", "content": "ship it"}]}, token)
            unrouted, _ = request(url, {"messages": [{"role": "user", "content": "ship it now"}]}, token)
            assert "make ship" in routed["messages"][-1]["content"], routed
            assert "make ship" not in json.dumps(unrouted)
            for route, payload in [
                ("chat/completions", {"messages": [{"role": "user", "content": "deploy now"}], "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}], "parallel_tool_calls": True}),
                ("responses", {"input": [{"role": "user", "content": "deploy now"}], "tools": [{"type": "function", "name": "read_file", "parameters": {"type": "object"}}]}),
                ("messages", {"messages": [{"role": "user", "content": "deploy now"}], "tools": [{"name": "read_file", "input_schema": {"type": "object"}}]}),
            ]:
                target = f"http://127.0.0.1:{port}/a/demo/v1/{route}"
                result, _ = request(target, payload, token)
                assert result["tools"] == payload["tools"]
                if "parallel_tool_calls" in payload:
                    assert result["parallel_tool_calls"] is True
                req = urllib.request.Request(target, json.dumps(dict(payload, stream=True)).encode(), headers={"Content-Type": "application/json", "X-Ctx-Token": token})
                with urllib.request.urlopen(req, timeout=5) as response:
                    assert response.read() == b"data: {\"part\":1}\n\ndata: [DONE]\n\n"
                try:
                    request(target, {"force_error": True}, token)
                except urllib.error.HTTPError as error:
                    assert error.code == 429 and json.load(error)["error"]["message"] == "rate limited"
                else:
                    raise AssertionError("upstream error was not preserved")
            result, headers = request(url, {"unexpected": "shape"}, token)
            assert result == {"unexpected": "shape"} and headers["X-Ctx-Recalls"] == "0"
            try:
                request(url, body)
            except urllib.error.HTTPError as error:
                assert error.code == 401
            else:
                raise AssertionError("request without token succeeded")
        finally:
            process.terminate()
            process.wait(timeout=5)
    upstream.shutdown()
    print("proxy smoke test passed")


if __name__ == "__main__":
    main()
