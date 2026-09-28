"""Three-way verdict for the two-week dogfood window: pass / fail / inconclusive.

Reads aggregate rows from docs/dogfood-snapshots.jsonl (one row per participant per day,
written by tools/dogfood_snapshot.py on each participant's machine) and an optional
docs/dogfood-incidents.jsonl ({"date","participant","blocking":true,"summary"}).

- fail: any blocking incident, or a day where a participant's ctx service was unhealthy
  on a day they ran verified tasks. A real negative is reported even on a small sample.
- inconclusive: fewer than --min-participants distinct participants with verified tasks,
  or fewer than --min-day-share of workdays reaching --min-tasks-per-day verified tasks
  (team total). "Nobody used it" must not read as "the product failed".
- pass: otherwise.
"""

import argparse
import datetime as dt
import json
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"


def load(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verdict(rows, incidents, start, end, min_participants=3, min_tasks_per_day=5,
            min_day_share=0.8):
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    workdays = [d.isoformat() for d in days if d.weekday() < 5]
    in_window = [r for r in rows if start.isoformat() <= r["date"] <= end.isoformat()]
    per_day = {day: 0 for day in workdays}
    participants = set()
    unhealthy = []
    for row in in_window:
        tasks = row.get("verified_tasks_today") or 0
        if row["date"] in per_day:
            per_day[row["date"]] += tasks
        if tasks:
            participants.add(row.get("participant") or "unknown")
            if row.get("service_healthy") is False:
                unhealthy.append({"date": row["date"], "participant": row.get("participant")})
    blocking = [i for i in incidents if i.get("blocking")
                and start.isoformat() <= i.get("date", "") <= end.isoformat()]
    days_met = sum(n >= min_tasks_per_day for n in per_day.values())
    elapsed = [d for d in workdays if d <= dt.date.today().isoformat()]
    summary = {"window": [start.isoformat(), end.isoformat()], "workdays": len(workdays),
               "workdays_elapsed": len(elapsed), "participants_with_tasks": len(participants),
               "verified_tasks": sum(per_day.values()), "workdays_meeting_minimum": days_met,
               "tasks_per_workday": per_day, "blocking_incidents": len(blocking),
               "unhealthy_service_days": unhealthy,
               "rules": {"min_participants": min_participants,
                         "min_tasks_per_day": min_tasks_per_day,
                         "min_day_share": min_day_share}}
    if blocking or unhealthy:
        summary["verdict"] = "fail"
        summary["reason"] = "blocking incident or unhealthy ctx service on a working day"
    elif len(participants) < min_participants or days_met < min_day_share * len(workdays):
        summary["verdict"] = "inconclusive"
        summary["reason"] = "sample below the agreed minimum; not evidence for or against ctx"
    else:
        summary["verdict"] = "pass"
        summary["reason"] = "minimum participation met with no blocking incident"
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-09-27")
    parser.add_argument("--end", default="2026-10-10")
    parser.add_argument("--min-participants", type=int, default=3)
    parser.add_argument("--min-tasks-per-day", type=int, default=5)
    parser.add_argument("--min-day-share", type=float, default=0.8)
    args = parser.parse_args()
    result = verdict(load(DOCS / "dogfood-snapshots.jsonl"), load(DOCS / "dogfood-incidents.jsonl"),
                     dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end),
                     args.min_participants, args.min_tasks_per_day, args.min_day_share)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
