"""BEAM with perfect retrieval: each probing question is answered from its gold source
messages only (the dataset's source_chat_ids, in conversation order and with their dates),
with the official answer prompt and judge. The per-ability scores are the ceiling a memory
system can reach with the same answer and judge models, and the gap between them and a
system's scores is what its memory loses.

  python -m tracks.beam_oracle --split 100K --out results/beam/100K
"""
import argparse
import ast
import concurrent.futures
import json
import sys
import time
import traceback
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "tracks"))

from common import llm  # noqa: E402
import beam  # noqa: E402
import beam_official as official  # noqa: E402


def ids(value):
    """Every message id in a source_chat_ids field (a list, or a dict of lists)."""
    out = []
    if isinstance(value, int):
        out.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            out += ids(item)
    elif isinstance(value, dict):
        for item in value.values():
            out += ids(item)
    return out


def gold_context(conversation, question):
    messages = {}
    for batch in conversation["chat"]:
        date = beam.anchor_date(batch[0].get("time_anchor")) if batch else None
        for message in batch:
            content = beam.MARKER.sub("", message.get("content") or "").strip()
            if content:
                messages[message.get("id")] = (date, message.get("role"), content)
    sources = sorted(i for i in dict.fromkeys(ids(question.get("source_chat_ids"))) if i in messages)
    return "\n".join(f"- [{messages[i][0]}] [{messages[i][1]}] {messages[i][2]}" for i in sources), len(sources)


def answer(conversation, ability, index, question):
    row = {"conversation": conversation["conversation_id"], "ability": ability, "index": index,
           "question": question["question"]}
    try:
        context, sources = gold_context(conversation, question)
        row.update(sources=sources, kept_tokens=len(context) // 4)
        prompt = official.ANSWER_PROMPT.replace("<context>", context or "(no memories)").replace("<question>", question["question"])
        response = llm.chat([{"role": "user", "content": prompt}], max_tokens=1500, tag="beam-answer-oracle" + beam.SALT)
        row["response"] = response
        if ability == "event_ordering":
            ordering = beam.order_events(question["rubric"], response)
            row.update(ordering)
            row["score"] = 0.0 if ordering["tau_norm"] != ordering["tau_norm"] else ordering["tau_norm"]
        else:
            row["score"], row["judged"] = beam.judge(question["rubric"], response)
    except Exception as error:  # noqa: BLE001 - a failure scores 0 and is reported
        row.update(score=0.0, error=f"{type(error).__name__}: {error}"[:300], trace=traceback.format_exc()[-600:])
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="100K")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default="oracle-evidence")
    parser.add_argument("--workers", type=int, default=40)
    args = parser.parse_args()
    directory = args.out / args.name
    directory.mkdir(parents=True, exist_ok=True)
    jobs = []
    for conversation in beam.load(args.split):
        probing = ast.literal_eval(conversation["probing_questions"])
        for ability in beam.ABILITIES:
            for index, question in enumerate(probing.get(ability, [])):
                jobs.append((conversation, ability, index, question))
    started, before = time.time(), dict(llm.STATS)
    rows = []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool, open(directory / "rows.jsonl", "w") as out:
        for row in pool.map(lambda job: answer(*job), jobs):
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    usage = {"reader_judge": {k: llm.STATS[k] - before[k] for k in ("calls", "input_tokens", "output_tokens")}}
    summary = beam.summarize(args.name, rows, usage, time.time() - started, None)
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
