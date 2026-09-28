"""Track A: LongMemEval_S, stratified sample, one answer model and one judge for everyone.

  python -m tracks.a_longmemeval --n 20 --systems ctx,ctx-general,mem0,naive-rag,full-context,no-memory --out results/trackA-pilot

Per system and question: ingest the question's history sessions in order (sessions of one
question are sequential, questions run in parallel), retrieve with the question, keep at most
`--budget` tokens of retrieved text, answer with the shared prompt, grade with the official
LongMemEval judge prompts. Everything is written to <out>/<system>/rows.jsonl.
"""
import argparse
import collections
import gc
import concurrent.futures
import json
import random
import sys
import time
import traceback
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "data"))

from adapters.base import fit_budget  # noqa: E402
from common import llm, memguard  # noqa: E402
from evaluate_qa_official import get_anscheck_prompt  # noqa: E402

TYPES = ["single-session-user", "single-session-assistant", "single-session-preference",
         "multi-session", "knowledge-update", "temporal-reasoning"]
ANSWER_PROMPT = """Below are memories retrieved from earlier conversations with the user. They may be incomplete or irrelevant.

{memories}

Current date: {date}
Question: {question}

Answer the question using the memories. If they do not contain the information needed, say that you don't know. Be concise; for questions about time, reason with the dates given."""


def sample(data, n, seed=20260928):
    rng = random.Random(seed)
    abstention = [q for q in data if q["question_id"].endswith("_abs")]
    regular = [q for q in data if not q["question_id"].endswith("_abs")]
    n_abs = max(1, round(n * 0.15))
    per_type = (n - n_abs) // len(TYPES)
    chosen = rng.sample(abstention, n_abs)
    for qtype in TYPES:
        pool = [q for q in regular if q["question_type"] == qtype]
        chosen += rng.sample(pool, min(per_type, len(pool)))
    extra = [q for q in regular if q not in chosen and q["question_type"] in ("multi-session", "temporal-reasoning")]
    chosen += rng.sample(extra, n - len(chosen))
    return sorted(chosen, key=lambda q: q["question_id"])


def subset(questions, n, seed=20260929):
    """Half of the stratified sample: 9 abstention questions, 9 each of the three
    multi-step types and 8 of each single-session type (60 in total)."""
    if n != 60:
        raise ValueError("only the 60-question subset is defined")
    rng = random.Random(seed)
    take = {"abstention": 9, "multi-session": 9, "temporal-reasoning": 9, "knowledge-update": 9,
            "single-session-user": 8, "single-session-assistant": 8, "single-session-preference": 8}
    chosen = []
    for category, count in take.items():
        pool = sorted((q for q in questions if ("abstention" if q["question_id"].endswith("_abs") else q["question_type"]) == category),
                      key=lambda q: q["question_id"])
        chosen += rng.sample(pool, count)
    return sorted(chosen, key=lambda q: q["question_id"])


def make_system(name, workdir, embed):
    from adapters import baselines
    if name == "ctx":
        from adapters.ctx import Ctx
        return Ctx(workdir / "ctx-home", name="ctx", mode="inject")
    if name == "ctx-general":
        from adapters.ctx import Ctx
        return Ctx(workdir / "ctx-general-home", name="ctx-general", prompt_file=BENCH / "prompts/ctx-general.txt", mode="inject")
    if name == "ctx-general-wide":
        # Question-answering operating point, fixed a priori: return every memory with some
        # semantic support instead of the few best (the default injection setting).
        from adapters.ctx import Ctx
        return Ctx(workdir / "ctx-general-wide-home", name="ctx-general-wide", prompt_file=BENCH / "prompts/ctx-general.txt",
                   recall={"relative_cutoff": 0.0, "min_similarity": 0.25, "max_injected": 20})
    if name == "ctx-atomic":
        # Atomic, detail-preserving extraction (prompt v2) and the explicit search mode.
        from adapters.ctx import Ctx
        return Ctx(workdir / "ctx-atomic-home", name="ctx-atomic", prompt_file=BENCH / "prompts/ctx-general.txt")
    if name == "ctx-episodic":
        # ctx-atomic plus raw conversation excerpts ranked with the memories.
        from adapters.ctx import Ctx
        return Ctx(workdir / "ctx-episodic-home", name="ctx-episodic", prompt_file=BENCH / "prompts/ctx-general.txt", episodes=5,
                   reuse_from=workdir / "ctx-atomic-home")
    if name == "mem0":
        from adapters.mem0_adapter import Mem0
        return Mem0(workdir / "mem0", embed)
    if name == "naive-rag":
        return baselines.NaiveRag(embed)
    if name == "full-context":
        return baselines.FullContext()
    if name == "no-memory":
        return baselines.NoMemory()
    raise ValueError(name)


def ingest(system, question):
    started = time.time()
    for session_id, date, session in zip(question["haystack_session_ids"], question["haystack_dates"], question["haystack_sessions"]):
        messages = [{"role": m["role"], "content": m["content"]} for m in session]
        system.ingest_session(question["question_id"], session_id, messages, date)
    return time.time() - started


def answer_and_grade(system, question, budget):
    started = time.time()
    retrieved = system.search(question["question_id"], question["question"], limit=20)
    search_ms = (time.time() - started) * 1000
    kept = retrieved if system.unbounded else fit_budget(retrieved, budget)
    memories = "\n".join(f"- {item}" for item in kept) or "(no memories)"
    reply = llm.chat([{"role": "user", "content": ANSWER_PROMPT.format(memories=memories, date=question["question_date"], question=question["question"])}],
                     max_tokens=1500, tag="answer")
    prompt = get_anscheck_prompt(question["question_type"], question["question"], question["answer"], reply,
                                 abstention=question["question_id"].endswith("_abs"))
    verdict = llm.chat([{"role": "user", "content": prompt}], max_tokens=200, tag="judge")
    return {"retrieved_items": len(retrieved), "kept_items": len(kept), "kept_tokens": sum(len(k) for k in kept) // 4,
            "search_ms": round(search_ms, 1), "hypothesis": reply, "judge": verdict.strip()[:200],
            "correct": verdict.strip().lower().startswith("yes") or "yes" in verdict.strip().lower()[:10]}


def percentile(values, p):
    values = sorted(values)
    return round(values[min(len(values) - 1, int((len(values) - 1) * p))], 1) if values else None


def run_system(name, questions, out, workdir, embed, budget, workers):
    system = make_system(name, workdir, embed)
    directory = out / name
    directory.mkdir(parents=True, exist_ok=True)
    system.setup()
    rows, started = [], time.time()
    try:
        def one(question):
            row = {"question_id": question["question_id"], "question_type": question["question_type"],
                   "abstention": question["question_id"].endswith("_abs")}
            memguard.wait()
            try:
                row["ingest_seconds"] = round(ingest(system, question), 1)
                row.update(answer_and_grade(system, question, budget))
            except Exception as error:  # noqa: BLE001 - a failure counts as wrong
                row.update(error=f"{type(error).__name__}: {error}"[:500], correct=False, trace=traceback.format_exc()[-800:])
            finally:
                system.release(question["question_id"])
            return row
        with concurrent.futures.ThreadPoolExecutor(min(workers, len(questions))) as pool:
            for row in pool.map(one, questions):
                rows.append(row)
                with open(directory / "rows.jsonl", "a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[{name}] {len(rows)}/{len(questions)} {row['question_id']} correct={row['correct']} {row.get('error', '')[:80]}", flush=True)
        usage = system.usage()
    finally:
        system.teardown()
    by_type = collections.defaultdict(list)
    for row in rows:
        by_type["abstention" if row["abstention"] else row["question_type"]].append(row["correct"])
    summary = {"system": name, "questions": len(rows), "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 3),
               "by_type": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_type.items())},
               "errors": sum("error" in r for r in rows), "wall_seconds": round(time.time() - started),
               "ingest_seconds_per_question_p50": percentile([r["ingest_seconds"] for r in rows if "ingest_seconds" in r], .5),
               "search_ms_p50": percentile([r["search_ms"] for r in rows if "search_ms" in r], .5),
               "search_ms_p95": percentile([r["search_ms"] for r in rows if "search_ms" in r], .95),
               "kept_tokens_mean": round(sum(r.get("kept_tokens", 0) for r in rows) / len(rows)),
               "usage": usage, "budget_tokens": None if system.unbounded else budget}
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--systems", default="no-memory,full-context,naive-rag,ctx,ctx-general,mem0")
    parser.add_argument("--out", type=Path, default=BENCH / "results/trackA")
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/ctx-bench"))
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=250)
    parser.add_argument("--subset", type=int, default=0,
                        help="run on the stratified 60-question subset of the cached sample")
    args = parser.parse_args()
    cached = args.out / "questions.json"
    if cached.exists():
        questions = json.loads(cached.read_text())
    else:
        data = json.load(open(BENCH / "data/longmemeval_s_cleaned.json"))
        questions = sample(data, args.n)
        del data
        gc.collect()
        args.out.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(questions))
    args.out.mkdir(parents=True, exist_ok=True)
    if args.subset:
        questions = subset(questions, args.subset)
        (args.out / f"subset-{args.subset}.json").write_text(json.dumps([q["question_id"] for q in questions]))
    else:
        (args.out / "sample.json").write_text(json.dumps([q["question_id"] for q in questions]))
    from adapters.ctx import CtxService
    memguard.start_monitor(args.out / "memory.log")
    summaries = []
    for name in args.systems.split(","):
        if (args.out / name / "summary.json").exists():
            summaries.append(json.loads((args.out / name / "summary.json").read_text()))
            continue
        (args.out / name / "rows.jsonl").unlink(missing_ok=True)
        # Only systems that need the shared embedder start it, so two ctx processes never
        # hold embedding models at the same time.
        embed = CtxService(args.workdir / "embed-home", embed_workers=3).start() if name in ("naive-rag", "mem0") else None
        try:
            summaries.append(run_system(name, questions, args.out, args.workdir, embed, args.budget, args.workers))
        finally:
            if embed:
                embed.stop()
    (args.out / "summary.json").write_text(json.dumps({"summaries": summaries, "llm": llm.STATS}, ensure_ascii=False, indent=2))
    for s in summaries:
        print(f"{s['system']:14} acc {s['accuracy']:.3f} {s['by_type']}")


if __name__ == "__main__":
    main()
