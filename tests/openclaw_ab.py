"""Run paired OpenClaw coding tasks: direct arm vs ctx arm, both metered by one layer.

By default the direct arm goes through a meter-only ctx agent that forwards requests
byte for byte, so both arms' steps, cache and output tokens come from the same
gateway responses. Cost is split into uncached input, cached input, output and
engine LLM calls; USD is computed only when prices are passed explicitly.
See docs/evidence-plan.md for manifest format.
"""

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from token_attribution import attribute  # noqa: E402

CTX_BIN = ROOT / "target/release/ctx"


def credential(reference):
    if reference.startswith("env:"):
        return os.environ[reference[4:]]
    if reference.startswith("keychain:"):
        return subprocess.run(["security", "find-generic-password", "-a", "default", "-s",
                               reference[9:], "-w"], check=True, capture_output=True,
                              text=True).stdout.strip()
    raise ValueError("credential reference must be env: or keychain:")


def free_port():
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def copied_workspace(source, target):
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(
        ".git", ".ctx", ".env", ".env.*", "__pycache__", "node_modules", "target"))


def proxy_usage(db_path, marks, prices):
    with sqlite3.connect(db_path) as db:
        rows = db.execute("SELECT actual_model,actual_input_tokens,output_tokens,cached_input_tokens "
                          "FROM usage WHERE rowid>? ORDER BY rowid", (marks["usage"],)).fetchall()
        decisions = [json.loads(row[0]) for row in db.execute(
            "SELECT decision FROM steps WHERE rowid>? ORDER BY rowid", (marks["steps"],))]
        engine = db.execute("SELECT COUNT(*),COALESCE(SUM(input_tokens),0),COALESCE(SUM(output_tokens),0) "
                            "FROM llm_calls WHERE rowid>?", (marks["llm"],)).fetchone()
    total_input = sum(row[1] or 0 for row in rows)
    cached = sum(row[3] or 0 for row in rows)
    output = sum(row[2] or 0 for row in rows)
    cost = {"uncached_input_tokens": total_input - cached, "cached_input_tokens": cached,
            "output_tokens": output, "engine_llm_calls": engine[0],
            "engine_input_tokens": engine[1], "engine_output_tokens": engine[2]}
    if prices:
        cost["usd"] = round(((total_input - cached + engine[1]) * prices["input"]
                             + cached * prices["cached_input"]
                             + (output + engine[2]) * prices["output"]) / 1_000_000, 6)
    return {"steps": len(rows), "actual_models": sorted({row[0] for row in rows if row[0]}),
            "input_tokens": total_input, "output_tokens": output, "cached_input_tokens": cached,
            "cache_hit_rate": round(cached / total_input, 4) if total_input else None,
            "modified_requests": sum(bool(row.get("modified")) for row in decisions),
            "evicted_tool_results": sum(len(row.get("evictions") or []) for row in decisions),
            "cost": cost}


def marks(db_path):
    with sqlite3.connect(db_path) as db:
        return {"usage": db.execute("SELECT COALESCE(MAX(rowid),0) FROM usage").fetchone()[0],
                "steps": db.execute("SELECT COALESCE(MAX(rowid),0) FROM steps").fetchone()[0],
                "llm": db.execute("SELECT COALESCE(MAX(rowid),0) FROM llm_calls").fetchone()[0],
                "events": db.execute("SELECT COALESCE(MAX(rowid),0) FROM events").fetchone()[0]}


def run_task(task, arm, task_index, root, proxy_url, token, direct_url, key, model, binary, db_path,
             ctx_home=None, metered_direct=True, prices=None):
    task_id = task["id"]
    # Same path length and shape in both arms so the system prompt differs only by one letter.
    workspace = root / f"w{task_index}{arm[0]}" / "workspace"
    workspace.parent.mkdir()
    copied_workspace(task["source_dir"], workspace)
    state = root / f"openclaw-{task_index}-{arm}"
    state.mkdir()
    metered = arm == "ctx" or metered_direct
    api_key = "${CTX_PROXY_TOKEN}" if metered else "${AB_MODEL_KEY}"
    config = {"gateway": {"mode": "local"},
              "agents": {"defaults": {"workspace": str(workspace),
                                      "model": {"primary": f"ab/{model}"}}},
              "models": {"mode": "merge", "providers": {"ab": {
                  "baseUrl": proxy_url if arm == "ctx" else direct_url,
                  "api": "openai-completions", "apiKey": api_key,
                  "models": [{"id": model, "name": model}]}}}}
    config_path = state / "openclaw.json"
    config_path.write_text(json.dumps(config))
    env = dict(os.environ, OPENCLAW_STATE_DIR=str(state), OPENCLAW_CONFIG_PATH=str(config_path),
               CTX_PROXY_TOKEN=token, AB_MODEL_KEY=key)
    timeout = int(task.get("timeout_seconds", 180))
    prompt = f"Work only in your configured workspace. {task['prompt']}"
    before = marks(db_path) if metered else None
    started = time.perf_counter()
    try:
        agent = subprocess.run([binary, "agent", "--local", "--agent", "main",
                                "--message", prompt, "--json", "--thinking", "off",
                                "--timeout", str(timeout)], env=env, capture_output=True,
                               text=True, timeout=timeout + 20)
        agent_exit = agent.returncode
        try:
            payload = json.loads(agent.stdout) if agent_exit == 0 else {}
        except json.JSONDecodeError:
            payload = {}
    except subprocess.TimeoutExpired:
        agent_exit = 124
        payload = {}
    elapsed = time.perf_counter() - started
    try:
        verification = subprocess.run(task["verify"], cwd=workspace,
                                      capture_output=True, text=True,
                                      timeout=int(task.get("verify_timeout_seconds", 30)))
        verification_exit = verification.returncode
    except subprocess.TimeoutExpired:
        verification_exit = 124
    except OSError:
        verification_exit = 127
    meta = payload.get("meta", {}).get("agentMeta", {})
    usage = meta.get("usage", {})
    last_usage = meta.get("lastCallUsage", {})
    cache_read = usage.get("cacheRead")
    if cache_read is None:
        cache_read = last_usage.get("cacheRead")
    cache_write = usage.get("cacheWrite")
    if cache_write is None:
        cache_write = last_usage.get("cacheWrite")
    total_input = (usage.get("input") or 0) + (cache_read or 0) + (cache_write or 0)
    result = {"task_id": task_id, "arm": arm, "agent_exit": agent_exit,
              "verification_exit": verification_exit,
              "success": agent_exit == 0 and verification_exit == 0,
              "elapsed_seconds": round(elapsed, 3),
              "reported_usage": {"input_tokens": usage.get("input"),
                                 "output_tokens": usage.get("output"),
                                 "cache_read_tokens": cache_read,
                                 "cache_write_tokens": cache_write,
                                 "total_input_tokens": total_input if usage.get("input") is not None else None,
                                 "cache_hit_rate": (cache_read or 0) / total_input if total_input else None}}
    if metered:
        result["proxy_usage"] = proxy_usage(db_path, before, prices)
        attribution = attribute(ctx_home, agent="openclaw" if arm == "ctx" else "openclaw-direct",
                                after_rowid=before["events"])
        result["attribution"] = {k: v for k, v in attribution.items() if k != "per_step"}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--direct-via-gateway", action="store_true",
                        help="old behavior: direct arm bypasses the meter (no per-step usage)")
    parser.add_argument("--price-input", type=float, help="USD per million uncached input tokens")
    parser.add_argument("--price-cached-input", type=float)
    parser.add_argument("--price-output", type=float)
    args = parser.parse_args()
    prices = None
    if args.price_input is not None:
        if args.price_cached_input is None or args.price_output is None:
            raise ValueError("pass all three prices or none")
        prices = {"input": args.price_input, "cached_input": args.price_cached_input,
                  "output": args.price_output}
    manifest_path = args.manifest.resolve()
    spec = json.loads(manifest_path.read_text())
    tasks = spec["tasks"]
    if not tasks or any(not isinstance(task.get("verify"), list) or not task["verify"] for task in tasks):
        raise ValueError("each task needs a nonempty verify argv list")
    for task in tasks:
        source = Path(task["source_dir"])
        task["source_dir"] = source if source.is_absolute() else (manifest_path.parent / source).resolve()
        task["verify"] = [part.replace("${MANIFEST_DIR}", str(manifest_path.parent))
                          for part in task["verify"]]
        if not task["source_dir"].is_dir():
            raise FileNotFoundError(task["source_dir"])
    direct_url = os.environ["CTX_TEST_LIVE_BASE_URL"].rstrip("/")
    model = os.environ["CTX_TEST_LIVE_MODEL"]
    reference = os.environ["CTX_TEST_LIVE_CREDENTIAL_REF"]
    key = credential(reference)
    binary = shutil.which("openclaw")
    if not binary or not CTX_BIN.exists():
        raise RuntimeError("build ctx --release and put openclaw on PATH")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ctx_root = root / "ctx"
        env = dict(os.environ, CTX_HOME=str(ctx_root))

        def ctx(*argv):
            return subprocess.run([CTX_BIN, *argv], env=env, capture_output=True,
                                  text=True, check=True)

        ctx("init")
        port = free_port()
        config = ctx_root / "config.yaml"
        config.write_text(config.read_text().replace("port: 7788", f"port: {port}"))
        ctx("model", "set", direct_url, model, "--credential-ref", reference,
            "--upstream-user-agent", "curl/8.0")
        ctx("connect", "openclaw")
        ctx("connect", "openclaw-direct", "--meter-only")
        token = (ctx_root / "token").read_text().strip()
        daemon = subprocess.Popen([CTX_BIN, "serve"], env=env, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.PIPE)
        try:
            for _ in range(50):
                try:
                    request = urllib.request.Request(f"http://127.0.0.1:{port}/api/v1/health",
                                                     headers={"X-Ctx-Token": token})
                    with urllib.request.urlopen(request, timeout=2):
                        break
                except urllib.error.URLError:
                    time.sleep(0.1)
            else:
                raise RuntimeError("isolated ctx did not start")
            results = []
            meter_url = f"http://127.0.0.1:{port}/a/openclaw-direct/v1"
            for repeat in range(args.repeats):
                for index, task in enumerate(tasks):
                    arms = ["direct", "ctx"]
                    random.Random(index + 20260926 + 1000 * repeat).shuffle(arms)
                    for arm in arms:
                        result = run_task(task, arm, repeat * len(tasks) + index, root,
                                          f"http://127.0.0.1:{port}/a/openclaw/v1", token,
                                          direct_url if args.direct_via_gateway else meter_url,
                                          key, model, binary, ctx_root / "state/events.db",
                                          ctx_root, not args.direct_via_gateway, prices)
                        result["repeat"] = repeat
                        result["order"] = arms.index(arm)
                        results.append(result)
                        print(json.dumps({k: result.get(k) for k in
                                          ("task_id", "repeat", "arm", "success", "elapsed_seconds")}),
                              file=sys.stderr, flush=True)
            report = {"kind": "paired_openclaw_metered" if not args.direct_via_gateway
                      else "paired_openclaw_pilot", "model": model, "tasks": len(tasks),
                      "repeats": args.repeats, "results": results,
                      "note": "Success uses each task's verifier. Both arms are metered by ctx "
                              "unless --direct-via-gateway. USD appears only with explicit prices."}
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if args.output:
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        finally:
            daemon.terminate()
            daemon.wait(timeout=5)


if __name__ == "__main__":
    main()
