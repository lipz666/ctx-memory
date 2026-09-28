"""Aggregate verified dogfood tasks without reading raw event payloads."""

import json
import os
from pathlib import Path
import sqlite3


ROOT = Path(os.environ.get("CTX_HOME", Path.home() / ".ctx"))


def main():
    db_path = ROOT / "state/events.db"
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
        rows = db.execute("""
            SELECT json_extract(h.features,'$.result') AS result,
                   COUNT(u.step_event) AS steps,
                   COALESCE(SUM(u.actual_input_tokens),0) AS input_tokens,
                   COALESCE(SUM(u.output_tokens),0) AS output_tokens,
                   COALESCE(SUM(u.cached_input_tokens),0) AS cached_input_tokens,
                   SUM(CASE WHEN r.id IS NOT NULL AND u.actual_input_tokens IS NULL THEN 1 ELSE 0 END) AS missing_usage_steps
            FROM events h
            LEFT JOIN events r ON r.session_id=h.session_id AND r.kind='request'
            LEFT JOIN usage u ON u.step_event=r.id
            WHERE h.kind='hook' AND json_extract(h.features,'$.result') IN ('success','failure')
            GROUP BY h.id
        """).fetchall()
    by_result = {}
    for result in ("success", "failure"):
        selected = [row for row in rows if row[0] == result]
        steps = sum(row[1] or 0 for row in selected)
        inputs = sum(row[2] or 0 for row in selected)
        outputs = sum(row[3] or 0 for row in selected)
        cached = sum(row[4] or 0 for row in selected)
        missing = sum(row[5] or 0 for row in selected)
        by_result[result] = {"tasks": len(selected), "steps": steps,
                             "input_tokens": inputs, "output_tokens": outputs,
                             "cached_input_tokens": cached,
                             "cache_hit_rate": cached / inputs if inputs and not missing else None,
                             "input_tokens_per_task": inputs / len(selected) if selected and not missing else None,
                             "missing_usage_steps": missing}
    print(json.dumps({"source": "verified dogfood hooks", "by_result": by_result},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
