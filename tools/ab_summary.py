"""Summarize metered OpenClaw A/B results (ADR 0005).

  python3 tools/ab_summary.py result.json [more.json ...]

Per arm: success rate, steps, input per step, the four cost components per task and per
successful task (failures stay in the numerator). Per pair where both arms succeeded:
ctx/direct ratio of total input, of input per step and of steps, so a step-count
difference is not mistaken for engine overhead.
"""

import json
import statistics
import sys
from collections import defaultdict

PARTS = ["uncached_input_tokens", "cached_input_tokens", "output_tokens", "engine_input_tokens",
         "engine_output_tokens"]


def load(paths):
    rows = []
    for path in paths:
        rows.extend(r for r in json.load(open(path))["results"] if r.get("proxy_usage"))
    return rows


def arm_summary(rows):
    tasks = len(rows)
    successes = sum(r["success"] for r in rows)
    totals = {p: sum(r["proxy_usage"]["cost"][p] for r in rows) for p in PARTS}
    steps = sum(r["proxy_usage"]["steps"] for r in rows)
    inputs = sum(r["proxy_usage"]["input_tokens"] for r in rows)
    usd = [r["proxy_usage"]["cost"].get("usd") for r in rows]
    return {"tasks": tasks, "successes": successes, "steps": steps,
            "input_per_step": round(inputs / steps) if steps else None,
            "cache_hit_rate": round(totals["cached_input_tokens"] / inputs, 4) if inputs else None,
            "modified_requests": sum(r["proxy_usage"]["modified_requests"] for r in rows),
            "evicted_tool_results": sum(r["proxy_usage"]["evicted_tool_results"] for r in rows),
            "per_task": {p: round(v / tasks) for p, v in totals.items()} if tasks else None,
            "per_successful_task": ({p: round(v / successes) for p, v in totals.items()}
                                    if successes else None),
            "usd_per_successful_task": (round(sum(usd) / successes, 6)
                                        if successes and all(u is not None for u in usd) else None)}


def main():
    rows = load(sys.argv[1:])
    by_task = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_task[r["task_id"]][r["arm"]].append(r)
    report = {"tasks": {}, "overall": {}}
    pair_ratios = []
    for task, arms in by_task.items():
        entry = {arm: arm_summary(items) for arm, items in arms.items()}
        pairs = []
        for c, d in zip(arms.get("ctx", []), arms.get("direct", [])):
            if not (c["success"] and d["success"]):
                pairs.append({"repeat": c.get("repeat"), "both_succeeded": False})
                continue
            cu, du = c["proxy_usage"], d["proxy_usage"]
            ratio = {"repeat": c.get("repeat"), "both_succeeded": True,
                     "input_ratio": round(cu["input_tokens"] / du["input_tokens"], 3),
                     "uncached_input_ratio": round(cu["cost"]["uncached_input_tokens"]
                                                   / max(du["cost"]["uncached_input_tokens"], 1), 3),
                     "steps": [cu["steps"], du["steps"]],
                     "input_per_step_ratio": round((cu["input_tokens"] / cu["steps"])
                                                   / (du["input_tokens"] / du["steps"]), 3)}
            pairs.append(ratio)
            pair_ratios.append(ratio)
        entry["pairs"] = pairs
        report["tasks"][task] = entry
    for arm in ("ctx", "direct"):
        report["overall"][arm] = arm_summary([r for r in rows if r["arm"] == arm])
    if pair_ratios:
        report["overall"]["paired_both_succeeded"] = {
            "n": len(pair_ratios),
            "median_input_ratio": statistics.median(p["input_ratio"] for p in pair_ratios),
            "median_input_per_step_ratio": statistics.median(p["input_per_step_ratio"]
                                                             for p in pair_ratios)}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
