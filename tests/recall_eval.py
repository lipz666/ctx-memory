"""Recall quality eval on tests/fixtures/recall_eval (isolated ctx instance).

  python3 tests/recall_eval.py [--binary target/release/ctx] [--split dev|test|all] [--output r.json]

Imports every memory with `ctx remember`, then asks each query. Uses the batch command
`ctx eval recall` when the binary has it, otherwise one `ctx recall` call per query
(so older binaries can be measured as a baseline). Always-on rules are not counted.

Metrics over positive queries: hit@1 (top result is expected), recall@3 (an expected
memory is in the top 3), MRR. Over negative queries: false-positive rate (anything
returned). Negatives include questions about another project's memory.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/recall_eval"


def load(name):
    return [json.loads(line) for line in (FIXTURE / name).read_text().splitlines() if line.strip()]


def rankings(binary, env, queries):
    batch = subprocess.run([binary, "eval", "recall", "--help"], env=env, capture_output=True)
    if batch.returncode == 0:
        path = Path(env["CTX_HOME"]) / "queries.jsonl"
        path.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in queries))
        out = subprocess.run([binary, "eval", "recall", str(path)], env=env, check=True,
                             capture_output=True, text=True).stdout
        return {row["id"]: row["ids"] for row in map(json.loads, out.splitlines())}
    result = {}
    for q in queries:
        args = [binary, "recall", q["query"]] + (["--project", q["project"]] if q["project"] else [])
        out = subprocess.run(args, env=env, check=True, capture_output=True, text=True).stdout
        result[q["id"]] = [line.split()[0] for line in out.splitlines()
                           if line.startswith("mem_") and "[pinned]" not in line]
    return result


def score(queries, ranked, key_of):
    rows, by_cat = [], defaultdict(list)
    for q in queries:
        got = [key_of.get(i, i) for i in ranked.get(q["id"], [])][:3]
        expected = set(q["expected"])
        row = {"id": q["id"], "category": q["category"], "query": q["query"], "got": got,
               "expected": q["expected"]}
        if expected:
            rank = next((i + 1 for i, k in enumerate(got) if k in expected), None)
            row.update(hit1=bool(got) and got[0] in expected, recall3=rank is not None,
                       rr=1 / rank if rank else 0.0)
        else:
            row.update(false_positive=bool(got))
        rows.append(row)
        by_cat[q["category"]].append(row)
    pos = [r for r in rows if r["expected"]]
    neg = [r for r in rows if not r["expected"]]
    summary = {"positives": len(pos), "negatives": len(neg),
               "hit@1": round(sum(r["hit1"] for r in pos) / len(pos), 3) if pos else None,
               "recall@3": round(sum(r["recall3"] for r in pos) / len(pos), 3) if pos else None,
               "mrr": round(sum(r["rr"] for r in pos) / len(pos), 3) if pos else None,
               "negative_fp_rate": round(sum(r["false_positive"] for r in neg) / len(neg), 3) if neg else None}
    categories = {}
    for cat, items in sorted(by_cat.items()):
        if items[0]["expected"]:
            categories[cat] = f'{sum(r["recall3"] for r in items)}/{len(items)} recall@3'
        else:
            categories[cat] = f'{sum(r["false_positive"] for r in items)}/{len(items)} false positives'
    return summary, categories, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default=str(ROOT / "target/release/ctx"))
    parser.add_argument("--split", default="all", choices=["dev", "test", "all"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show-misses", action="store_true")
    parser.add_argument("--no-embedding", action="store_true", help="measure the keyword-only fallback")
    args = parser.parse_args()
    memories = load("memories.jsonl")
    queries = [q for q in load("queries.jsonl") if args.split == "all" or q["split"] == args.split]
    with tempfile.TemporaryDirectory() as home:
        env = dict(os.environ, CTX_HOME=home)
        subprocess.run([args.binary, "init"], env=env, check=True, capture_output=True)
        if args.no_embedding:
            config = Path(home) / "config.yaml"
            config.write_text(config.read_text().replace("embedding:\n  enabled: true", "embedding:\n  enabled: false"))
        key_of = {}
        for m in memories:
            out = subprocess.run([args.binary, "remember", m["content"], "--kind", m["type"],
                                  "--scope", m["scope"]], env=env, check=True,
                                 capture_output=True, text=True).stdout.split()
            key_of[out[0]] = m["key"]
        ranked = rankings(args.binary, env, queries)
    summary, categories, rows = score(queries, ranked, key_of)
    report = {"binary": args.binary, "split": args.split, "embedding": not args.no_embedding, "summary": summary,
              "categories": categories}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.show_misses:
        for r in rows:
            if r.get("recall3") is False or r.get("false_positive"):
                print(r["id"], r["category"], r["query"][:60], "got", r["got"], "want", r["expected"])
    if args.output:
        args.output.write_text(json.dumps(dict(report, rows=rows), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
