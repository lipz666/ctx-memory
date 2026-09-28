"""Track B1: retrieval quality on identical memory text.

  python -m tracks.b1_retrieval --systems ctx,ctx-keyword,mem0,naive-rag --out results/trackB1

Every system stores the same memories verbatim (no extraction), one namespace per project
plus one global namespace; a query searches its project and the global namespace and the
two lists are merged by the system's own score. Top-3 is scored against the expected
memory keys (memories are matched by their text).
"""
import argparse
import collections
import concurrent.futures
import json
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from common import memguard  # noqa: E402


def make_system(name, workdir, embed):
    from adapters import baselines
    from adapters.ctx import Ctx
    if name == "ctx":
        return Ctx(workdir / "b1-ctx", name="ctx", mode="inject", embed_workers=1)
    if name == "ctx-wide":
        # Same a-priori question-answering operating point as Track A's ctx-general-wide.
        return Ctx(workdir / "b1-ctx-wide", name="ctx-wide", mode="search", embed_workers=1)
    if name == "ctx-keyword":
        return Ctx(workdir / "b1-ctx-keyword", name="ctx-keyword", embedding=False, mode="inject")
    if name == "mem0":
        from adapters.mem0_adapter import Mem0
        system = Mem0(workdir / "b1-mem0", embed)
        system.add_memory = lambda ns, text, project=None: system._memory(ns).add(
            [{"role": "user", "content": text}], user_id="u", infer=False)
        return system
    if name == "naive-rag":
        return baselines.NaiveRag(embed)
    raise ValueError(name)


def percentile(values, p):
    values = sorted(values)
    return round(values[min(len(values) - 1, int((len(values) - 1) * p))], 1) if values else None


def run(name, memories, queries, workdir, embed, out):
    system = make_system(name, workdir, embed)
    system.setup()
    try:
        started = time.time()
        by_text = {}
        for m in memories:  # chronological: updated facts come after the originals
            system.add_memory(f"b1-{m['scope']}", m["content"])
            by_text[m["content"].strip()] = m["key"]
        ingest_seconds = time.time() - started

        def key_of(text):
            body = text.split("] ", 1)[1] if text.startswith("[") and "] " in text else text
            return by_text.get(body.strip(), "?")

        def one(q):
            memguard.wait()
            t = time.time()
            hits = system.search_scored(f"b1-{q['project']}", q["query"], limit=10)
            hits += system.search_scored("b1-global", q["query"], limit=10)
            ms = (time.time() - t) * 1000
            hits.sort(key=lambda h: -h[1])
            got = [key_of(text) for text, _ in hits][:3]
            row = {"id": q["id"], "category": q["category"], "split": q["split"], "got": got, "returned": len(hits), "ms": round(ms, 1)}
            if q["expected"]:
                rank = next((i + 1 for i, k in enumerate(got) if k in q["expected"]), None)
                row.update(hit1=bool(got) and got[0] in q["expected"], recall3=rank is not None, rr=1 / rank if rank else 0.0)
                if q.get("stale"):
                    stale_rank = next((i + 1 for i, k in enumerate(got) if k in q["stale"]), None)
                    row["newer_first"] = rank is not None and (stale_rank is None or rank < stale_rank)
            else:
                row["returned_any"] = len(hits) > 0
            return row
        with concurrent.futures.ThreadPoolExecutor(16) as pool:
            rows = list(pool.map(one, queries))
    finally:
        system.teardown()
    pos = [r for r in rows if "recall3" in r]
    neg = [r for r in rows if "returned_any" in r]
    cats = collections.defaultdict(list)
    for r in pos:
        cats[r["category"]].append(r["recall3"])
    updated = [r for r in rows if "newer_first" in r]
    summary = {"system": name, "memories": len(memories), "queries": len(rows),
               "recall@1": round(sum(r["hit1"] for r in pos) / len(pos), 3),
               "recall@3": round(sum(r["recall3"] for r in pos) / len(pos), 3),
               "mrr": round(sum(r["rr"] for r in pos) / len(pos), 3),
               "test_split_recall@3": round(sum(r["recall3"] for r in pos if r["split"] == "test") / max(1, sum(r["split"] == "test" for r in pos)), 3),
               "by_category_recall@3": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(cats.items())},
               "updated_fact_newer_first": f"{sum(r['newer_first'] for r in updated)}/{len(updated)}",
               "negatives_returned_anything": {c: f"{sum(r['returned_any'] for r in neg if r['category'] == c)}/{sum(r['category'] == c for r in neg)}" for c in ("negative", "negative_cross_project")},
               "search_ms_p50": percentile([r["ms"] for r in rows], .5), "search_ms_p95": percentile([r["ms"] for r in rows], .95),
               "ingest_seconds": round(ingest_seconds, 1)}
    (out / name).mkdir(parents=True, exist_ok=True)
    with open(out / name / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / name / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", default="ctx,ctx-keyword,naive-rag,mem0")
    parser.add_argument("--data", type=Path, default=BENCH / "data/b1")
    parser.add_argument("--out", type=Path, default=BENCH / "results/trackB1")
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    memories = [json.loads(l) for l in open(args.data / "memories.jsonl")]
    queries = [json.loads(l) for l in open(args.data / "queries.jsonl")]
    args.out.mkdir(parents=True, exist_ok=True)
    memguard.start_monitor(args.out / "memory.log")
    from adapters.ctx import CtxService
    for name in args.systems.split(","):
        if (args.out / name / "summary.json").exists():
            continue
        embed = CtxService(args.workdir / "b1-embed", embed_workers=2).start() if name in ("naive-rag", "mem0") else None
        try:
            run(name, memories, queries, args.workdir, embed, args.out)
        finally:
            if embed:
                embed.stop()


if __name__ == "__main__":
    main()
