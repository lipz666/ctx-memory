"""Measure gateway latency for Gate-shaped requests (P50/P95) before choosing a budget.

Environment: CTX_TEST_LIVE_BASE_URL, CTX_TEST_LIVE_MODEL (comma-separated to compare),
CTX_TEST_LIVE_CREDENTIAL_REF. Sends the production Gate prompt with 20 synthetic
memory cards, sequentially, and prints latency, parse rate and reported model.
Does not print credentials or request bodies.
"""

import json
import os
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gateway_latency_live import credential, percentile  # noqa: E402

PROMPT = (Path(__file__).resolve().parents[1] / "prompts/gate-v1.txt").read_text()
CARDS = [{"id": f"mem_{i:02d}", "title": title} for i, title in enumerate([
    "Run DB migrations before deploy", "payments-svc uses Decimal for money",
    "CI flaky test: test_retry_timeout", "Use uv, not pip, in this repo",
    "Staging DB is read-only on Fridays", "Ledger dedup key is (source, external_id)",
    "Never force-push main", "Inventory reservations must be atomic",
    "Prefer pytest -x for quick checks", "Release notes live in docs/changes",
    "Gateway rejects UA without curl prefix", "Hermes needs mcp extra installed",
    "clamp() must handle lo > hi", "Use UTC in all timestamps",
    "Retry budget is 3 with jitter", "Secrets come from Keychain",
    "OpenClaw profile is isolated for dogfood", "Encoder batch size is 20",
    "SQLite WAL must be enabled", "Docs are written in Chinese"])]
STEPS = ["部署 payments-svc 到 staging", "pytest failed: AssertionError in test_ledger_dedup",
         "Refactor reservations to be atomic", "What's the weather like?",
         "exit code 1: ModuleNotFoundError: decimal_utils"]


def main():
    base = os.environ["CTX_TEST_LIVE_BASE_URL"].rstrip("/")
    key = credential(os.environ["CTX_TEST_LIVE_CREDENTIAL_REF"])
    count = int(os.environ.get("CTX_TEST_LIVE_COUNT", "20"))
    report = {}
    for model in os.environ["CTX_TEST_LIVE_MODEL"].split(","):
        latencies, parsed, failures, models, reasoning = [], 0, 0, set(), []
        for i in range(count):
            features = {"text": STEPS[i % len(STEPS)], "role": "user"}
            body = {"model": model, "temperature": 0, "max_tokens": 256,
                    "messages": [{"role": "system", "content": PROMPT},
                                 {"role": "user", "content": json.dumps(
                                     {"features": features, "cards": CARDS}, ensure_ascii=False)}]}
            request = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Authorization": f"Bearer {key}",
                                                      "Content-Type": "application/json",
                                                      "User-Agent": "curl/8.0"})
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = json.load(response)
            except Exception:  # noqa: BLE001 - record any transport failure as a failure sample
                failures += 1
                continue
            latencies.append((time.perf_counter() - started) * 1000)
            models.add(payload.get("model"))
            details = (payload.get("usage") or {}).get("completion_tokens_details") or {}
            reasoning.append(details.get("reasoning_tokens") or 0)
            text = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            try:
                json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
                parsed += 1
            except ValueError:
                pass
        report[model] = {"samples": len(latencies), "failures": failures,
                         "p50_ms": round(percentile(latencies, 0.5) or 0),
                         "p95_ms": round(percentile(latencies, 0.95) or 0),
                         "max_ms": round(max(latencies, default=0)),
                         "within_300ms": sum(x <= 300 for x in latencies),
                         "within_1500ms": sum(x <= 1500 for x in latencies),
                         "json_parsed": parsed, "reported_models": sorted(m for m in models if m),
                         "mean_reasoning_tokens": round(sum(reasoning) / len(reasoning), 1) if reasoning else None}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
