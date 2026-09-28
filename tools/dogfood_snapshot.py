"""Write one aggregate-only daily snapshot of the local ctx dogfood instance."""

import datetime as dt
import getpass
import json
import os
from pathlib import Path
import sqlite3
import urllib.error
import urllib.request


ROOT = Path(os.environ.get("CTX_HOME", Path.home() / ".ctx"))
PARTICIPANT = os.environ.get("CTX_DOGFOOD_PARTICIPANT") or getpass.getuser()
OUTPUT = Path(__file__).resolve().parents[1] / "docs/dogfood-snapshots.jsonl"


def health():
    try:
        port = 7788
        for line in (ROOT / "config.yaml").read_text().splitlines():
            if line.startswith("port:"):
                port = int(line.split(":", 1)[1].strip())
                break
        token = (ROOT / "token").read_text().strip()
        request = urllib.request.Request(f"http://127.0.0.1:{port}/api/v1/health",
                                         headers={"X-Ctx-Token": token})
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status == 200
    except (OSError, ValueError, urllib.error.URLError):
        return False


def count(db, sql):
    return db.execute(sql).fetchone()[0]


def snapshot():
    db_path = ROOT / "state/events.db"
    today = dt.datetime.now().astimezone().date()
    start = dt.datetime.combine(today, dt.time(), dt.datetime.now().astimezone().tzinfo)
    window = (start.astimezone(dt.timezone.utc).isoformat(),
              (start + dt.timedelta(days=1)).astimezone(dt.timezone.utc).isoformat())
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
        metrics = {
            "events": count(db, "SELECT COUNT(*) FROM events"),
            "steps": count(db, "SELECT COUNT(*) FROM steps"),
            "agent_usage_steps": count(db, "SELECT COUNT(*) FROM usage"),
            "engine_llm_calls": count(db, "SELECT COUNT(*) FROM llm_calls"),
            "engine_llm_failures": count(db, "SELECT COUNT(*) FROM llm_calls WHERE outcome IN ('error','timeout','network_error','invalid_response','upstream_error')"),
            "encode_jobs_pending": count(db, "SELECT COUNT(*) FROM encode_jobs WHERE status='pending'"),
            "encode_jobs_failed": count(db, "SELECT COUNT(*) FROM encode_jobs WHERE status='failed'"),
            "reviewed_memories": count(db, "SELECT COUNT(*) FROM memory_reviews"),
            "model_mismatch_steps": count(db, "SELECT COUNT(*) FROM usage WHERE requested_model IS NOT NULL AND actual_model IS NOT NULL AND requested_model<>actual_model"),
            "model_mismatch_pairs": count(db, "SELECT COUNT(DISTINCT requested_model || ' -> ' || actual_model) FROM usage WHERE requested_model IS NOT NULL AND actual_model IS NOT NULL AND requested_model<>actual_model"),
            "priced_steps": count(db, "SELECT COUNT(*) FROM usage WHERE actual_usd IS NOT NULL"),
            "known_model_steps": count(db, "SELECT COUNT(*) FROM usage WHERE actual_model IS NOT NULL"),
            "task_success_hooks": count(db, "SELECT COUNT(*) FROM events WHERE kind='hook' AND json_extract(features,'$.result')='success'"),
            "task_failure_hooks": count(db, "SELECT COUNT(*) FROM events WHERE kind='hook' AND json_extract(features,'$.result')='failure'"),
        }
        verified = db.execute(
            "SELECT json_extract(features,'$.result'),COUNT(*) FROM events WHERE kind='hook' "
            "AND json_extract(features,'$.result') IN ('success','failure') AND ts>=? AND ts<? "
            "GROUP BY 1", window).fetchall()
        metrics["verified_tasks_today"] = sum(n for _, n in verified)
        metrics["verified_successes_today"] = sum(n for r, n in verified if r == "success")
        metrics["gate_timeouts"] = count(db, "SELECT COUNT(*) FROM llm_calls WHERE role='gate' AND outcome='timeout'")
        metrics["average_review_wait_seconds"] = db.execute(
            "SELECT AVG(wait_seconds) FROM memory_reviews").fetchone()[0]
    metrics["pending_review_memories"] = sum(
        "\nstatus: pending_review\n" in path.read_text()
        for path in (ROOT / "memory").glob("mem_*.md")
    )
    metrics["service_healthy"] = health()
    metrics["date"] = today.isoformat()
    metrics["participant"] = PARTICIPANT
    return metrics


def main():
    value = snapshot()
    existing = []
    if OUTPUT.exists():
        existing = [json.loads(line) for line in OUTPUT.read_text().splitlines() if line.strip()]
    key = (value["date"], value["participant"])
    existing = [row for row in existing if (row["date"], row.get("participant")) != key]
    existing.append(value)
    existing.sort(key=lambda row: (row["date"], row.get("participant") or ""))
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in existing))
    print(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
