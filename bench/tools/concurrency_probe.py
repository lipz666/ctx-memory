"""Probe how many concurrent chat completions the gateway sustains.

  CTX_BENCH_BASE_URL=... CTX_BENCH_KEY_ENV=NAME python3 bench/tools/concurrency_probe.py 50 100 200

For each level, sends that many requests at once (a ~2K-token extraction-like prompt, 300
output tokens) and reports success rate, error codes and latency percentiles.
"""
import collections
import concurrent.futures
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("CTX_BENCH_BASE_URL", "https://vps.lpzproxy.xyz/v1").rstrip("/")
KEY = os.environ[os.environ.get("CTX_BENCH_KEY_ENV", "CTX_GW_KEY")]
MODEL = os.environ.get("CTX_BENCH_MODEL", "gemini-3.8-flash-high")
FILLER = " ".join(f"user said item {i} is on shelf {i % 17};" for i in range(400))


def one(i):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 300, "messages": [
        {"role": "system", "content": "Extract durable facts as a JSON list of short strings."},
        {"role": "user", "content": f"Request {i}. {FILLER}"}]}
    req = urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "User-Agent": "curl/8.0"})
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            payload = json.load(r)
            ok = bool(payload.get("choices", [{}])[0].get("message", {}).get("content"))
            return ("ok" if ok else "empty", time.time() - started)
    except urllib.error.HTTPError as e:
        return (f"http_{e.code}", time.time() - started)
    except Exception as e:  # noqa: BLE001
        return (type(e).__name__, time.time() - started)


def pct(values, p):
    values = sorted(values)
    return round(values[min(len(values) - 1, int((len(values) - 1) * p))], 1) if values else None


for level in map(int, sys.argv[1:]):
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(level) as pool:
        results = list(pool.map(one, range(level)))
    wall = time.time() - started
    outcomes = collections.Counter(r[0] for r in results)
    ok_lat = [r[1] for r in results if r[0] == "ok"]
    print(json.dumps({"concurrency": level, "outcomes": dict(outcomes), "success_rate": round(outcomes["ok"] / level, 3),
                      "p50_s": pct(ok_lat, .5), "p95_s": pct(ok_lat, .95), "max_s": pct(ok_lat, 1.0),
                      "wall_s": round(wall, 1), "throughput_per_min": round(outcomes["ok"] / wall * 60, 1)}), flush=True)
    time.sleep(5)
