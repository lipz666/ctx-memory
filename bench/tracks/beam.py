"""Track BEAM: long-term memory over 100K-10M-token conversations (ICLR 2026).

  python -m tracks.beam --split 100K --systems ctx-beam --out results/beam/100K

Data: bench/data/beam/<split>.parquet (Hugging Face Mohammadta/BEAM, Mohammadta/BEAM-10M).
Each conversation is written into its own memory namespace in order: every batch of the
chat (one time anchor) is cut at user turns into sessions of at most --session-chars
characters, each extracted before the next. Then each of its 20 probing questions (2 per
ability) is answered with the official BEAM answer prompt from the retrieved memories
(within --budget tokens) and scored with the official rubric judge; event ordering uses
the official LLM alignment and normalized Kendall tau. The reported score per ability is
what BEAM's report_results.py averages (tau_norm for event ordering, the rubric judge
score otherwise), and the overall score is the mean of the ten abilities.
"""
import argparse
import ast
import collections
import concurrent.futures
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "tracks"))

from adapters.base import fit_budget  # noqa: E402
from common import llm, memguard  # noqa: E402
import beam_official as official  # noqa: E402

ABILITIES = ["abstention", "contradiction_resolution", "event_ordering", "information_extraction",
             "instruction_following", "knowledge_update", "multi_session_reasoning", "preference_following",
             "summarization", "temporal_reasoning"]
MARKER = re.compile(r"\s*->->\s*\d+\s*,\s*\d+\s*$")
SALT = os.environ.get("BENCH_ANSWER_SALT", "")


def load(split):
    import pyarrow.parquet as pq
    return pq.read_table(BENCH / f"data/beam/{split}.parquet").to_pylist()


def anchor_date(anchor):
    """'March-15-2024' -> '2024/03/15' (the anchor as given when it does not parse)."""
    try:
        return datetime.strptime(anchor, "%B-%d-%Y").strftime("%Y/%m/%d")
    except (TypeError, ValueError):
        return anchor or None


def sessions(conversation, max_chars):
    """(session id, date, messages) in order: each batch cut at user turns so that no
    session exceeds max_chars (ctx reads at most ~30k characters of one session)."""
    for b, batch in enumerate(conversation["chat"]):
        date = anchor_date(batch[0].get("time_anchor")) if batch else None
        current, size, part = [], 0, 0
        for message in batch:
            content = MARKER.sub("", message.get("content") or "").strip()
            if message.get("role") not in ("user", "assistant") or not content:
                continue
            if current and message["role"] == "user" and size + len(content) > max_chars:
                yield f"b{b + 1}-{part}", date, current
                current, size, part = [], 0, part + 1
            current.append({"role": message["role"], "content": content})
            size += len(content)
        if current:
            yield f"b{b + 1}-{part}", date, current


def judge(rubric, response):
    """Official rubric judge: mean of per-item scores (1.0 / 0.5 / 0.0)."""
    items = []
    for item in rubric:
        prompt = official.JUDGE_PROMPT.replace("<rubric_item>", item).replace("<llm_response>", response)
        reply = llm.chat([{"role": "user", "content": prompt}], max_tokens=600, tag="beam-judge" + SALT)
        match = re.search(r'"score"\s*:\s*"?([0-9.]+)', reply)
        score = float(match.group(1)) if match else 0.0
        items.append({"item": item, "score": score, "reply": reply[:300]})
    return sum(i["score"] for i in items) / max(1, len(items)), items


def order_events(rubric, response):
    """Official event ordering: align answer lines to reference events with the LLM
    classifier (first match wins), then normalized Kendall tau-b."""
    used, system = set(), []
    for line in response.split("\n"):
        matched = None
        if line.strip():
            for index, reference in enumerate(rubric):
                if index in used:
                    continue
                reply = llm.chat(official.equivalence_messages(reference, line), max_tokens=10, tag="beam-align" + SALT)
                if "yes" in reply.lower():
                    matched = index
                    break
        if matched is not None:
            system.append(rubric[matched])
            used.add(matched)
        else:
            system.append(line)
    return official.event_ordering_score(list(rubric), system)


def make_system(name, workdir, budget):
    from adapters.ctx import Ctx
    prompt = BENCH / "prompts/ctx-general.txt"
    if "-reanswer" in name:
        # The memory store of the named run (ctx-beam-v3-reanswer-8k reuses ctx-beam-v3),
        # answered again (e.g. with another --budget); no extraction.
        base = name.split("-reanswer")[0]
        # ...-reanswer-brief: ctx writes a brief for each question (one model call);
        # ...-reanswer-agent: the brief writer may call tools over the memory first.
        agent = name.endswith("-agent")
        return Ctx(workdir / f"{name}-home", name=name, prompt_file=prompt, episodes=max(5, budget // 800),
                   budget=budget, reuse_from=workdir / f"{base}-home", brief=name.endswith("-brief") or agent, agent=agent)
    if name == "ctx-beam" or name.startswith("ctx-beam-v"):
        # ctx-beam-vN: the same configuration on a newer engine (run with CTX_BIN), kept apart;
        # ...-nodossier: without topic dossiers (extraction.dossiers off).
        settings = {"dossiers": "false"} if "-nodossier" in name else None
        return Ctx(workdir / f"{name}-home", name=name, prompt_file=prompt, episodes=5, budget=budget, recall=settings)
    raise ValueError(name)


def run_conversation(system, conversation, budget, session_chars, rows_path, lock, max_sessions=0, abilities=ABILITIES):
    ns = f"beam-{conversation['conversation_id']}"
    started = time.time()
    count = 0
    for session_id, date, messages in sessions(conversation, session_chars):
        if max_sessions and count >= max_sessions:
            break
        memguard.wait()
        system.ingest_session(ns, session_id, messages, date)
        count += 1
    ingest_seconds = time.time() - started
    probing = ast.literal_eval(conversation["probing_questions"])
    rows = []
    for ability in abilities:
        for index, question in enumerate(probing.get(ability, [])):
            row = {"conversation": conversation["conversation_id"], "ability": ability, "index": index,
                   "question": question["question"], "sessions": count, "ingest_seconds": round(ingest_seconds, 1)}
            try:
                t = time.time()
                # Candidates scale with the budget so a larger budget can actually be filled.
                retrieved = system.search(ns, question["question"], limit=max(20, budget // 100))
                row["search_ms"] = round((time.time() - t) * 1000, 1)
                kept = retrieved if system.unbounded else fit_budget(retrieved, budget)
                context = "\n".join(f"- {item}" for item in kept) or "(no memories)"
                row.update(kept_items=len(kept), kept_tokens=sum(len(k) for k in kept) // 4)
                prompt = official.ANSWER_PROMPT.replace("<context>", context).replace("<question>", question["question"])
                response = llm.chat([{"role": "user", "content": prompt}], max_tokens=1500, tag="beam-answer" + SALT)
                row["response"] = response
                if ability == "event_ordering":
                    ordering = order_events(question["rubric"], response)
                    row.update(ordering)
                    row["score"] = 0.0 if ordering["tau_norm"] != ordering["tau_norm"] else ordering["tau_norm"]
                    row["tau_nan"] = ordering["tau_norm"] != ordering["tau_norm"]
                else:
                    row["score"], row["judged"] = judge(question["rubric"], response)
            except Exception as error:  # noqa: BLE001 - a failure scores 0 and is reported
                row.update(score=0.0, error=f"{type(error).__name__}: {error}"[:300], trace=traceback.format_exc()[-600:])
            rows.append(row)
            with lock, open(rows_path, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    system.release(ns)
    return rows


def summarize(name, rows, usage, wall, budget):
    by_ability = collections.defaultdict(list)
    for row in rows:
        by_ability[row["ability"]].append(row["score"])
    ability_scores = {a: round(sum(v) / len(v), 4) for a, v in by_ability.items()}
    overall = round(sum(ability_scores.values()) / max(1, len(ability_scores)), 4)
    return {"system": name, "questions": len(rows), "conversations": len({r["conversation"] for r in rows}),
            "overall": overall, "abilities": ability_scores, "errors": sum("error" in r for r in rows),
            "tau_nan": sum(r.get("tau_nan", False) for r in rows), "budget_tokens": budget,
            "kept_tokens_mean": round(sum(r.get("kept_tokens", 0) for r in rows) / max(1, len(rows))),
            "search_ms_p50": sorted(r.get("search_ms", 0) for r in rows)[len(rows) // 2] if rows else None,
            "wall_seconds": round(wall), "usage": usage}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="100K")
    parser.add_argument("--systems", default="ctx-beam")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--session-chars", type=int, default=24000)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--conversations", default="", help="comma-separated conversation ids (default: all)")
    parser.add_argument("--max-sessions", type=int, default=0, help="smoke tests: ingest only the first N sessions")
    parser.add_argument("--abilities", default="", help="comma-separated abilities to answer (default: all ten)")
    parser.add_argument("--shard", default="", help="i/n: only conversations whose position % n == i-1")
    args = parser.parse_args()
    conversations = load(args.split)
    if args.conversations:
        wanted = set(args.conversations.split(","))
        conversations = [c for c in conversations if c["conversation_id"] in wanted]
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        conversations = [c for k, c in enumerate(conversations) if k % n == i - 1]
    abilities = [a for a in ABILITIES if not args.abilities or a in args.abilities.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    memguard.start_monitor(args.out / "memory.log")
    import threading
    for name in args.systems.split(","):
        directory = args.out / name
        directory.mkdir(parents=True, exist_ok=True)
        rows_path = directory / "rows.jsonl"
        rows_path.unlink(missing_ok=True)
        system = make_system(name, args.workdir, args.budget)
        system.setup()
        lock = threading.Lock()
        started, before = time.time(), dict(llm.STATS)
        rows = []
        try:
            with concurrent.futures.ThreadPoolExecutor(min(args.workers, len(conversations))) as pool:
                futures = [pool.submit(run_conversation, system, c, args.budget, args.session_chars, rows_path, lock,
                                       args.max_sessions, abilities)
                           for c in conversations]
                for future in concurrent.futures.as_completed(futures):
                    done = future.result()
                    rows += done
                    mean = sum(r["score"] for r in done) / max(1, len(done))
                    print(f"[{name}] conversation {done[0]['conversation'] if done else '?'}: {len(done)} questions, mean {mean:.3f} "
                          f"({len(rows)}/{2 * len(abilities) * len(conversations)})", flush=True)
            usage = system.usage()
            usage["reader_judge"] = {k: llm.STATS[k] - before[k] for k in ("calls", "input_tokens", "output_tokens")}
        finally:
            system.teardown()
        summary = summarize(name, rows, usage, time.time() - started, args.budget)
        (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
