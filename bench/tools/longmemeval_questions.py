"""Write <out>/questions.json for tracks.a_longmemeval (it reads that file when present).

  python tools/longmemeval_questions.py --out results/longmemeval-unused167            # never used before
  python tools/longmemeval_questions.py --out results/longmemeval-all500 --all

"Never used": not in any committed results/**/sample.json or subset-*.json of LongMemEval
or track A (the questions every earlier run saw), i.e. untouched by development.
"""
import argparse
import collections
import json
import subprocess
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]


def used_ids():
    files = subprocess.run(["git", "ls-files", "results"], cwd=BENCH, capture_output=True, text=True, check=True).stdout.split()
    used = set()
    for name in files:
        if (name.endswith("/sample.json") or "/subset-" in name) and "beam" not in name and "loca" not in name:
            used |= set(json.loads((BENCH / name).read_text()))
    return used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--all", action="store_true", help="all 500 questions")
    args = parser.parse_args()
    data = json.loads((BENCH / "data/longmemeval_s_cleaned.json").read_text())
    used = set() if args.all else used_ids()
    chosen = sorted((q for q in data if q["question_id"] not in used), key=lambda q: q["question_id"])
    out = args.out if args.out.is_absolute() else BENCH / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "questions.json").write_text(json.dumps(chosen))
    mix = collections.Counter("abstention" if q["question_id"].endswith("_abs") else q["question_type"] for q in chosen)
    print(f"{len(chosen)} questions ({len(used)} excluded), {sum(len(q['haystack_sessions']) for q in chosen)} sessions -> {out}")
    print(dict(mix))


if __name__ == "__main__":
    main()
