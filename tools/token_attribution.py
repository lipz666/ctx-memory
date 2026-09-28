"""Break each proxied step's input tokens into attributable parts.

Reads steps from a local ctx instance (debug API + usage table) and splits every
request into: agent system prompt, agent tool schemas, ctx MCP tool schemas,
history (with the part the default eviction policy could archive), and what ctx
added (L0 note, memories) or removed (eviction). Estimates use chars/4 and are
scaled to the gateway-reported input tokens of each step so the parts sum to the
actual number. Payloads stay local; only aggregate numbers are printed.

  python3 tools/token_attribution.py --session SESSION_ID [--ctx-home ~/.ctx]
  python3 tools/token_attribution.py --after-rowid N --agent openclaw
"""

import argparse
import json
from pathlib import Path
import sqlite3
import urllib.request

CTX_MCP_TOOLS = {"recall", "expand", "remember", "forget", "flag_memory", "set_intent"}


def estimate(value):
    if value is None:
        return 0
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return (len(text) + 3) // 4


def history_items(body):
    items = body.get("messages", body.get("input"))
    return items if isinstance(items, list) else []


def tool_result_text(item):
    if item.get("role") == "tool" and isinstance(item.get("content"), str):
        return [item["content"]]
    if item.get("type") == "function_call_output" and isinstance(item.get("output"), str):
        return [item["output"]]
    if isinstance(item.get("content"), list):
        return [part["content"] for part in item["content"]
                if part.get("type") == "tool_result" and isinstance(part.get("content"), str)]
    return []


def split_request(body, min_tokens, threshold):
    """Estimated token parts of one request as the agent sent it."""
    items = history_items(body)
    system = estimate(body.get("system")) + estimate(body.get("instructions"))
    history = 0
    for item in items:
        if item.get("role") in ("system", "developer"):
            system += estimate(item.get("content"))
        else:
            history += estimate(item)
    tools = body.get("tools") or []
    ctx_tools = sum(estimate(t) for t in tools if tool_name(t) in CTX_MCP_TOOLS
                    or tool_name(t).startswith("ctx__") or tool_name(t).startswith("mcp__ctx__"))
    agent_tools = sum(estimate(t) for t in tools) - ctx_tools
    eligible = 0
    for index, item in enumerate(items):
        if index + 8 >= len(items):
            continue
        for text in tool_result_text(item):
            tokens = (len(text) + 3) // 4
            if tokens >= min_tokens:
                eligible += tokens
    total = estimate(body)
    other = max(total - system - history - agent_tools - ctx_tools, 0)
    return {"system": system, "agent_tools": agent_tools, "ctx_mcp_tools": ctx_tools,
            "history": history, "other_fields": other, "total": total,
            "evictable_old_tool_results": eligible,
            "eviction_would_fire": eligible >= threshold}


def tool_name(tool):
    return tool.get("name") or (tool.get("function") or {}).get("name") or ""


def fetch(port, token, step):
    request = urllib.request.Request(f"http://127.0.0.1:{port}/api/v1/debug/steps/{step}",
                                     headers={"X-Ctx-Token": token})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def read_policy(home):
    port, min_tokens, threshold = 7788, 500, 20_000
    for line in (home / "config.yaml").read_text().splitlines():
        key, _, value = line.partition(":")
        if key == "port":
            port = int(value)
        elif key == "tool_result_min_tokens":
            min_tokens = int(value)
        elif key == "evict_threshold_tokens":
            threshold = int(value)
    return port, min_tokens, threshold


def attribute(home, session=None, agent=None, after_rowid=0):
    """Aggregate attribution for request steps matching the filters."""
    home = Path(home).expanduser()
    token = (home / "token").read_text().strip()
    port, min_tokens, threshold = read_policy(home)
    where, params = ["e.kind='request'", "e.rowid>?"], [after_rowid]
    if session:
        where.append("e.session_id=?")
        params.append(session)
    if agent:
        where.append("e.agent_id=?")
        params.append(agent)
    with sqlite3.connect(home / "state/events.db") as db:
        rows = db.execute(
            "SELECT e.id,u.actual_input_tokens,u.cached_input_tokens,u.output_tokens "
            "FROM events e JOIN usage u ON u.step_event=e.id WHERE " + " AND ".join(where)
            + " ORDER BY e.rowid", params).fetchall()
    steps = []
    for step, actual, cached, output in rows:
        debug = fetch(port, token, step)
        original = debug["original"]
        compiled = debug.get("compiled") or original
        parts = split_request(original, min_tokens, threshold)
        added = max(estimate(compiled) - estimate(original), 0)
        removed = max(estimate(original) - estimate(compiled), 0)
        sent = parts["total"] + added - removed
        scale = (actual / sent) if actual and sent else 1.0
        scaled = {key: round(value * scale) for key, value in parts.items()
                  if isinstance(value, int) and key != "total"}
        scaled.update({"ctx_added": round(added * scale), "ctx_evicted": round(removed * scale),
                       "actual_input": actual, "cached_input": cached or 0,
                       "uncached_input": (actual or 0) - (cached or 0), "output": output,
                       "eviction_would_fire": parts["eviction_would_fire"],
                       "modified": bool((debug.get("decision") or {}).get("modified"))})
        steps.append(scaled)
    keys = ["system", "agent_tools", "ctx_mcp_tools", "history", "other_fields", "ctx_added",
            "ctx_evicted", "evictable_old_tool_results", "actual_input", "cached_input",
            "uncached_input", "output"]
    totals = {key: sum(step[key] or 0 for step in steps) for key in keys}
    actual = totals["actual_input"] or 1
    return {"steps": len(steps),
            "totals": totals,
            "share_of_input": {key: round(totals[key] / actual, 4) for key in
                               ["system", "agent_tools", "ctx_mcp_tools", "history",
                                "ctx_added", "evictable_old_tool_results"]},
            "fixed_prefix_per_step": round((totals["system"] + totals["agent_tools"]
                                            + totals["ctx_mcp_tools"]) / max(len(steps), 1)),
            "steps_where_default_eviction_fires": sum(s["eviction_would_fire"] for s in steps),
            "modified_steps": sum(s["modified"] for s in steps),
            "policy": {"tool_result_min_tokens": min_tokens, "evict_threshold_tokens": threshold},
            "per_step": steps,
            "note": "Parts are chars/4 estimates scaled to gateway-reported input per step."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx-home", type=Path, default=Path.home() / ".ctx")
    parser.add_argument("--session")
    parser.add_argument("--agent")
    parser.add_argument("--after-rowid", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = attribute(args.ctx_home, args.session, args.agent, args.after_rowid)
    summary = {k: v for k, v in report.items() if k != "per_step"}
    print(json.dumps(summary if args.output else report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
