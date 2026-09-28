"""Track B2: multi-session coding tasks with a real Agent (OpenClaw).

  python -m tracks.b2_coding --configs no-memory,openclaw-native,ctx,ctx-mcp,mem0-mcp --runs 1 --out results/trackB2

Each scenario is a sequence of independent Agent sessions on the same repositories. The
knowledge needed by the "apply" session is stated only in an earlier session's message.
Checks run on the repository after each apply session. Configurations differ only in the
memory available across sessions:

  no-memory        nothing; OpenClaw's own memory files are removed after every session
  openclaw-native  OpenClaw's built-in workspace memory (MEMORY.md, memory/) kept
  ctx              ctx proxy (automatic extraction and injection) + ctx MCP tools
  ctx-mcp          ctx MCP tools only (the Agent decides to remember/recall)
  ctx-plugin       ctx as OpenClaw's memory plugin (integrations/openclaw/ctx-memory): hooks
                   record and inject, memory_search/memory_store tools; no proxy
  mem0-mcp         Mem0 OSS via MCP (add_memory, search_memories)
"""
import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
ROOT = BENCH.parent
sys.path.insert(0, str(BENCH))
from adapters.ctx import BIN, CtxService  # noqa: E402
from common import memguard  # noqa: E402

GW_URL = os.environ.get("BENCH_BASE_URL", "https://vps.lpzproxy.xyz/v1")
MODEL = os.environ.get("BENCH_MODEL", "gemini-3.8-flash-high")
OPENCLAW_FILES = ["AGENTS.md", "HEARTBEAT.md", "IDENTITY.md", "MEMORY.md", "SOUL.md", "TOOLS.md", "USER.md",
                  "BOOTSTRAP.md", "memory", "openclaw-workspace-state.json", ".openclaw"]
REPOS = BENCH / "b2/repos"


def gw_key():
    return os.environ[os.environ.get("BENCH_KEY_ENV", "CTX_GW_KEY")]


# ---------- checks ----------

def run_tests(repo):
    cmd = ["node", "--test"] if (repo / "package.json").exists() else [sys.executable, "-m", "unittest", "discover", "-s", "tests"]
    return subprocess.run(cmd, cwd=repo, capture_output=True, timeout=120).returncode == 0


def check(repo, spec):
    kind = spec["type"]
    path = repo / spec.get("file", "")
    text = path.read_text(errors="replace") if spec.get("file") and path.is_file() else None
    if kind == "tests":
        return run_tests(repo)
    if kind == "exists":
        return text is not None
    if kind == "contains":
        return text is not None and all(s in text for s in spec["all"])
    if kind == "not_contains":
        return text is None or not any(s in text for s in spec["any"])
    if kind == "regex":
        return text is not None and re.search(spec["pattern"], text) is not None
    if kind in ("cjk", "no_cjk"):
        has = text is not None and re.search(r"[一-鿿]", text) is not None
        return has if kind == "cjk" else (text is not None and not has)
    if kind == "python":
        return subprocess.run([sys.executable, "-c", spec["code"]], cwd=repo, capture_output=True, timeout=60).returncode == 0
    if kind == "node":
        return subprocess.run(["node", "--input-type=module", "-e", spec["code"]], cwd=repo, capture_output=True, timeout=60).returncode == 0
    if kind == "shell":
        result = subprocess.run(spec["cmd"], shell=True, cwd=repo, capture_output=True, text=True, timeout=60)
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if spec.get("stdout_json_lines"):
            try:
                return result.returncode == 0 and bool(lines) and all(isinstance(json.loads(line), dict) for line in lines)
            except ValueError:
                return False
        if spec.get("stdout_not_regex"):
            return result.returncode in (0, 1) and bool(lines) and not any(re.search(spec["stdout_not_regex"], line) for line in lines)
        return result.returncode == 0
    raise ValueError(kind)


# ---------- memory systems ----------

class Services:
    """Long-running services for one configuration."""

    def __init__(self, config, workdir):
        self.config = config
        self.ctx = None
        self.embed = None
        if config in ("ctx", "ctx-mcp", "ctx-plugin"):
            self.ctx = CtxService(workdir / f"b2-{config}-home", embed_workers=2, agents=("openclaw",)).start()
        if config == "mem0-mcp":
            self.embed = CtxService(workdir / "b2-embed-home", embed_workers=2).start()

    def stop(self):
        for service in (self.ctx, self.embed):
            if service:
                service.stop()


def ctx_mcp(services, project):
    return {"command": str(BIN), "args": ["mcp"], "env": {"CTX_HOME": str(services.ctx.home), "CTX_PROJECT": project,
                                                         os.environ.get("BENCH_KEY_ENV", "CTX_GW_KEY"): gw_key()}}


def openclaw_config(config, services, workspace, project, session_key, mem0_dir):
    if config == "ctx":
        provider = {"baseUrl": f"http://127.0.0.1:{services.ctx.port}/a/openclaw/v1", "api": "openai-completions",
                    "apiKey": services.ctx.token, "request": {"headers": {"X-Ctx-Session": session_key, "X-Ctx-Project": project}}}
    else:
        provider = {"baseUrl": GW_URL, "api": "openai-completions", "apiKey": gw_key()}
    provider["models"] = [{"id": MODEL, "name": MODEL}]
    config_json = {"gateway": {"mode": "local"},
                   "agents": {"defaults": {"workspace": str(workspace), "model": {"primary": f"bench/{MODEL}"}}},
                   "models": {"mode": "merge", "providers": {"bench": provider}}}
    servers = {}
    if config in ("ctx", "ctx-mcp"):
        servers["ctx"] = ctx_mcp(services, project)
    if config == "mem0-mcp":
        servers["mem0"] = {"command": str(BENCH / ".venv/bin/python"), "args": [str(BENCH / "mcp/mem0_mcp.py")],
                           "env": {"MEM0_NS": project, "MEM0_DIR": str(mem0_dir), "GW_KEY": gw_key(), "GW_URL": GW_URL,
                                   "EMBED_URL": services.embed.embeddings_base_url(), "EMBED_TOKEN": services.embed.token,
                                   "MEM0_TELEMETRY": "False"}}
    if servers:
        config_json["mcp"] = {"servers": servers}
    if config == "ctx-plugin":
        config_json["plugins"] = {
            "load": {"paths": [str(BENCH.parent / "integrations/openclaw/ctx-memory")]},
            "slots": {"memory": "ctx-memory"},
            "entries": {"ctx-memory": {"enabled": True, "hooks": {"allowConversationAccess": True}, "config": {"project": project}}},
        }
    return config_json


def strip_openclaw_files(repo):
    for name in OPENCLAW_FILES:
        path = repo / name
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()


# ---------- scenario execution ----------

def run_scenario(config, services, scenario, distractors, run, workroot, out):
    root = workroot / config / f"{scenario['id']}-r{run}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    repos, projects = {}, {}
    for name in dict.fromkeys(s["repo"] for s in scenario["sessions"]):
        project = f"{name}-{scenario['id']}-r{run}-{config}".lower()
        target = root / project
        shutil.copytree(REPOS / name, target)
        subprocess.run(["git", "init", "-q"], cwd=target, check=True)
        subprocess.run(["git", "add", "-A"], cwd=target, check=True)
        subprocess.run(["git", "-c", "user.email=b@b", "-c", "user.name=bench", "commit", "-qm", "base"], cwd=target, check=True)
        repos[name], projects[name] = target, project
    first = scenario["sessions"][0]
    plan = [first] + [{"repo": first["repo"], "role": "distract", "message": m} for m in distractors[first["repo"]]] + scenario["sessions"][1:]
    native_state = root / "openclaw-state-native"
    results = []
    for index, session in enumerate(plan):
        memguard.wait()
        repo, project = repos[session["repo"]], projects[session["repo"]]
        key = f"{project}-s{index}"
        state = native_state if config == "openclaw-native" else Path(tempfile.mkdtemp(prefix="oc-", dir=root))
        state.mkdir(parents=True, exist_ok=True)
        (state / "openclaw.json").write_text(json.dumps(openclaw_config(config, services, repo, project, key, root / f"mem0-{session['repo']}")))
        env = dict(os.environ, OPENCLAW_STATE_DIR=str(state), OPENCLAW_CONFIG_PATH=str(state / "openclaw.json"))
        if services.ctx:
            env["CTX_HOME"] = str(services.ctx.home)  # the ctx-memory plugin finds the daemon here
        started = time.time()
        try:
            proc = subprocess.run(["openclaw", "agent", "--local", "--agent", "main", "--session-id", key, "--message", session["message"],
                                   "--json", "--thinking", "off", "--timeout", "300"], cwd=repo, env=env, capture_output=True, text=True, timeout=400)
            exit_code, stdout = proc.returncode, proc.stdout
        except subprocess.TimeoutExpired:
            exit_code, stdout = 124, ""
        elapsed = round(time.time() - started, 1)
        usage = {}
        try:
            meta = json.loads(stdout).get("meta", {}).get("agentMeta", {})
            usage = {k: meta.get("usage", {}).get(k) for k in ("input", "output", "cacheRead")}
        except ValueError:
            pass
        extraction = None
        if config in ("ctx", "ctx-plugin"):
            # The plugin records sessions as openclaw:<session id>; the proxy under the session header.
            ctx_key = f"openclaw:{key}" if config == "ctx-plugin" else key
            try:
                extraction = services.ctx.request(f"/api/v1/sessions/{urllib.parse.quote(ctx_key, safe='')}/extract?wait=true", {})
                extraction = {k: len(extraction.get(k, [])) for k in ("created", "updated", "skipped")}
            except Exception as error:  # noqa: BLE001
                extraction = {"error": str(error)[:200]}
        row = {"config": config, "scenario": scenario["id"], "kind": scenario["kind"], "run": run, "index": index,
               "role": session["role"], "repo": session["repo"], "exit": exit_code, "seconds": elapsed, "usage": usage, "extraction": extraction}
        if session.get("checks"):
            outcomes = [check(repo, spec) for spec in session["checks"]]
            row.update(checks=outcomes, passed=all(outcomes))
        if config != "openclaw-native":
            for target in repos.values():
                strip_openclaw_files(target)
        if config != "openclaw-native":
            shutil.rmtree(state, ignore_errors=True)
        results.append(row)
        with open(out / "sessions.jsonl", "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[{config}] {scenario['id']} r{run}: " + " ".join(f"{r['role']}={'ok' if r.get('passed') else ('fail' if 'passed' in r else r['exit'])}" for r in results), flush=True)
    return results


def summarize(rows, config):
    applies = [r for r in rows if r["role"] == "apply"]
    others = [r for r in rows if r["role"] == "apply_other"]
    kinds = {}
    for r in applies:
        kinds.setdefault(r["kind"], []).append(r["passed"])
    return {"config": config, "scenario_runs": len({(r["scenario"], r["run"]) for r in rows}),
            "apply_success": round(sum(r["passed"] for r in applies) / max(1, len(applies)), 3),
            "apply_by_kind": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(kinds.items())},
            "isolation_success": f"{sum(r['passed'] for r in others)}/{len(others)}",
            "sessions": len(rows), "agent_failures": sum(r["exit"] != 0 for r in rows),
            "mean_session_seconds": round(sum(r["seconds"] for r in rows) / max(1, len(rows)), 1),
            "input_tokens": sum((r["usage"] or {}).get("input") or 0 for r in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="no-memory,openclaw-native,ctx,ctx-mcp,mem0-mcp")
    parser.add_argument("--scenarios", default="all")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--out", type=Path, default=BENCH / "results/trackB2")
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads((BENCH / "b2/scenarios.json").read_text())
    scenarios = [s for s in spec["scenarios"] if args.scenarios == "all" or s["id"] in args.scenarios.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    memguard.start_monitor(args.out / "memory.log")
    summaries = []
    for config in args.configs.split(","):
        out = args.out / config
        if (out / "summary.json").exists():
            summaries.append(json.loads((out / "summary.json").read_text()))
            continue
        out.mkdir(parents=True, exist_ok=True)
        (out / "sessions.jsonl").unlink(missing_ok=True)
        services = Services(config, args.workdir)
        try:
            jobs = [(s, run) for run in range(args.runs) for s in scenarios]
            with concurrent.futures.ThreadPoolExecutor(args.parallel) as pool:
                rows = [row for result in pool.map(lambda job: run_scenario(config, services, job[0], spec["distractors"], job[1], args.workdir, out), jobs) for row in result]
        finally:
            services.stop()
        summary = summarize(rows, config)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        summaries.append(summary)
    (args.out / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
