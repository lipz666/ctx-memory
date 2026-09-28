"""Chat client for the benchmark: one model, bounded concurrency, retries and a disk cache.

Every call in the benchmark (answers, judging, and systems that accept an injected client)
goes through the same gateway and model. Cached replies make interrupted runs resumable.
"""
import hashlib
import json
import os
import random
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = os.environ.get("BENCH_BASE_URL", "https://vps.lpzproxy.xyz/v1").rstrip("/")
MODEL = os.environ.get("BENCH_MODEL", "gemini-3.8-flash-high")
CONCURRENCY = int(os.environ.get("BENCH_CONCURRENCY", "250"))
CACHE = Path(os.environ.get("BENCH_CACHE", Path(__file__).resolve().parents[1] / "results/llm-cache.sqlite"))

_slots = threading.BoundedSemaphore(CONCURRENCY)
_local = threading.local()
_stats_lock = threading.Lock()
STATS = {"calls": 0, "cached": 0, "retries": 0, "failures": 0, "input_tokens": 0, "output_tokens": 0}


def api_key():
    return os.environ[os.environ.get("BENCH_KEY_ENV", "CTX_GW_KEY")]


def _db():
    if not hasattr(_local, "db"):
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        _local.db = sqlite3.connect(CACHE, timeout=60)
        _local.db.execute("PRAGMA journal_mode=WAL")
        _local.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, reply TEXT NOT NULL)")
    return _local.db


def _bump(**kwargs):
    with _stats_lock:
        for key, value in kwargs.items():
            STATS[key] += value


_gateway_lock = threading.Lock()
_gateway_ok_at = [0.0]


def wait_for_gateway(max_wait=7200):
    """Block until a tiny request succeeds (checked every 30 s). Workers that hit
    persistent failures wait here instead of failing their question."""
    with _gateway_lock:
        if time.time() - _gateway_ok_at[0] < 30:
            return True
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                request = urllib.request.Request(f"{BASE_URL}/chat/completions", data=json.dumps({
                    "model": MODEL, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5}).encode(), headers={
                    "Authorization": f"Bearer {api_key()}", "Content-Type": "application/json", "User-Agent": "curl/8.0"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status == 200:
                        _gateway_ok_at[0] = time.time()
                        return True
            except Exception:  # noqa: BLE001
                pass
            time.sleep(30)
        return False


def chat(messages, max_tokens=1024, temperature=0, tag="", use_cache=True, timeout=300, cycles=3):
    """Return the assistant text. After repeated failures, wait for the gateway and try
    again (`cycles` times) before raising."""
    for cycle in range(cycles):
        try:
            return _chat_once(messages, max_tokens, temperature, tag, use_cache, timeout)
        except RuntimeError:
            if cycle == cycles - 1:
                raise
            wait_for_gateway()


def _chat_once(messages, max_tokens, temperature, tag, use_cache, timeout):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    key = hashlib.sha256(json.dumps([body, tag], sort_keys=True).encode()).hexdigest()
    if use_cache:
        row = _db().execute("SELECT reply FROM cache WHERE key=?", (key,)).fetchone()
        if row:
            _bump(cached=1)
            return row[0]
    data = json.dumps(body).encode()
    for attempt in range(8):
        request = urllib.request.Request(f"{BASE_URL}/chat/completions", data=data, headers={
            "Authorization": f"Bearer {api_key()}", "Content-Type": "application/json", "User-Agent": "curl/8.0"})
        try:
            with _slots:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = json.load(response)
            text = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content")
            if not text:
                raise ValueError("empty reply")
            usage = payload.get("usage") or {}
            _bump(calls=1, input_tokens=usage.get("prompt_tokens") or 0, output_tokens=usage.get("completion_tokens") or 0)
            if use_cache:
                _db().execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (key, text))
                _db().commit()
            return text
        except urllib.error.HTTPError as error:
            if error.code not in (408, 409, 429, 500, 502, 503, 504):
                _bump(failures=1)
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError, json.JSONDecodeError):
            pass
        _bump(retries=1)
        time.sleep(min(60, 2 ** attempt) * (0.5 + random.random()))
    _bump(failures=1)
    raise RuntimeError("LLM call failed after retries")
