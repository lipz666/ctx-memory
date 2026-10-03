"""Run the sealed ccx acceptance tasks (tasks.py) through an OpenAI-compatible endpoint,
normally ccx (`ccx serve`), and check the replies.

  python bench/ccx_accept/run.py --base-url http://127.0.0.1:7789/v1 --arm ccx-on --out DIR
  python bench/ccx_accept/run.py --oracle ...    # scripted replies, no model: size estimate only

The API key is a placeholder: ccx adds the real one (`--upstream-key-stdin`). Every request
carries `X-Ccx-Tag: <arm>/<task>` so `ccx report --tag` can split the numbers.
"""
import argparse
import json
import pathlib
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from tasks import SYSTEM, TOOLS, by_name  # noqa: E402

MAX_STEPS_PER_TURN = 30


def post(base_url: str, model: str, messages: list, tag: str, timeout: int = 600) -> dict:
    body = json.dumps({"model": model, "messages": messages, "tools": TOOLS}).encode()
    request = urllib.request.Request(f"{base_url}/chat/completions", body, {
        "content-type": "application/json", "authorization": "Bearer via-ccx", "x-ccx-tag": tag,
        # The gateway's CDN rejects urllib's default User-Agent (Cloudflare error 1010).
        "user-agent": "ccx-accept/1.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:
            if attempt == 3:
                raise
            print(f"  retry after error: {e}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def run_task(task, base_url: str, model: str, arm: str, oracle: bool) -> dict:
    tag = f"{arm}/{task.name}"
    messages = [{"role": "system", "content": SYSTEM}]
    replies, steps, calls, usage = [], 0, [], {"input": 0, "output": 0}
    for turn, text in enumerate(task.turns):
        messages.append({"role": "user", "content": text})
        script = list(task.oracle[turn][0]) if oracle else []
        for _ in range(MAX_STEPS_PER_TURN):
            steps += 1
            response = post(base_url, model, messages, tag)
            if oracle:
                # Scripted step: one planned tool call, or the planned reply when none are left.
                if script:
                    name, args = script.pop(0)
                    message = {"role": "assistant", "content": None, "tool_calls": [
                        {"id": f"o{steps}", "type": "function",
                         "function": {"name": name, "arguments": json.dumps(args)}}]}
                else:
                    message = {"role": "assistant", "content": task.oracle[turn][1]}
            else:
                message = response["choices"][0]["message"]
                u = response.get("usage") or {}
                usage["input"] += u.get("prompt_tokens") or 0
                usage["output"] += (u.get("completion_tokens") or 0) + \
                    ((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
            kept = {k: message[k] for k in ("role", "content", "tool_calls") if message.get(k) is not None}
            kept.setdefault("content", None)
            messages.append(kept)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                replies.append(message.get("content") or "")
                break
            for call in tool_calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append(name)
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": task.call(name, args)})
        else:
            replies.append("")  # turn ran out of steps
    checks = {str(i): {"what": what, "ok": bool(fn(replies[i])) if i < len(replies) else False}
              for i, (what, fn) in task.checks.items()}
    return {"task": task.name, "arm": arm, "passed": all(c["ok"] for c in checks.values()),
            "checks": checks, "steps": steps, "tool_calls": calls, "usage": usage,
            "replies": [r[-600:] for r in replies]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:7789/v1")
    ap.add_argument("--model", default="gemini-3.8-flash-high")
    ap.add_argument("--arm", required=True, help="label, e.g. baseline or ccx-on")
    ap.add_argument("--tasks", default=",".join(by_name()))
    ap.add_argument("--out", required=True)
    ap.add_argument("--oracle", action="store_true")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    makers = by_name()
    for name in args.tasks.split(","):
        result = run_task(makers[name](), args.base_url, args.model, args.arm, args.oracle)
        with open(out / "results.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"{args.arm:10} {name:14} {'PASS' if result['passed'] else 'FAIL'}  steps {result['steps']:3}"
              f"  input {result['usage']['input']:>8,}  calls {len(result['tool_calls'])}", flush=True)


if __name__ == "__main__":
    main()
