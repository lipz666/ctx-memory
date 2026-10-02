"""Where BEAM answers go wrong: for each question a run scored below --below, the same
brief search is repeated on that run's memory store and the failure is placed in one stage.

  material  the gathered material (what the brief writer reads) lacks what the answer needs
  brief     the material has it, the brief does not carry it (or gets it wrong)
  answer    the brief has it, the final answer loses it
  judge     the answer has it, the score does not show it

Each stage's text is checked against the reference answer and rubric by one model call
("full", "partial" or "none"). The question is also answered and scored again from the
new brief, which shows how often the miss repeats.

  python -m tracks.beam_errors --run ctx-beam-v8b-reanswer-8k-brief \
      --stores 1-10=WORK/beam-100k-v8 11-20=WORK/beam-100k-v9 --out results/beam/100K/errors-v8b
"""
import argparse
import ast
import collections
import concurrent.futures
import json
import re
import sys
import time
import traceback
import urllib.parse
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "tracks"))

from adapters.base import fit_budget  # noqa: E402
from adapters.ctx import Ctx, namespace, render  # noqa: E402
from common import llm  # noqa: E402
import beam  # noqa: E402
import beam_official as official  # noqa: E402
from beam_oracle import reference  # noqa: E402

CHECK = """You check whether a text holds what a correct answer to a question needs.

Question: {question}

What a correct answer must contain (reference answer and grading points):
{reference}

Text:
<<<
{text}
>>>

Does the text hold the information needed to give the correct answer, with the right values, dates, items and order? It may hold much else; only what is needed counts. For a question that should be declined because the information was never given, "full" means the text gives no such information or says it is missing. For an instruction or preference the answer must follow, "full" means the text states that instruction or preference.

Reply with one word on the first line: full, partial or none. Then one sentence on what is missing or wrong."""


def check(question, ref, text, tag):
    reply = llm.chat([{"role": "user", "content": CHECK.format(question=question, reference=ref, text=text)}],
                     max_tokens=300, tag=tag)
    word = (re.findall(r"[a-z]+", reply.lower()) or ["none"])[0]
    return (word if word in ("full", "partial", "none") else "none"), reply.strip()[:300]


def expected(ability, question):
    parts = []
    text = reference(ability, question)
    if text:
        parts.append("Reference answer: " + text)
    rubric = question.get("rubric") or []
    if rubric:
        parts.append("Grading points:\n" + "\n".join(f"- {r}" for r in rubric))
    return "\n".join(parts)


def stage(material, brief, answer):
    if material == "none" or material == "partial" and brief != "full":
        return "material"
    if brief != "full":
        return "brief"
    if answer != "full":
        return "answer"
    return "judge"


def diagnose(system, conversation, ability, index, question, budget, now, old):
    ns = f"beam-{conversation['conversation_id']}"
    row = {"conversation": conversation["conversation_id"], "ability": ability, "index": index,
           "question": question["question"], "old_score": old["score"], "old_response": old.get("response")}
    try:
        ref = expected(ability, question)
        args = {"q": question["question"], "project": namespace(ns), "limit": min(max(20, budget // 100), 50), "mode": "search",
                "episodes": system.episodes, "budget": int(budget * 0.9), "brief": "true", "deep": "false", "dry_run": "true"}
        if now:
            args["now"] = now
        material = "\n".join(f"- {render(h)}" for h in system.service.request(f"/api/v1/recall?{urllib.parse.urlencode(args)}"))
        retrieved = system.search(ns, question["question"], limit=max(20, budget // 100), now=now)
        brief = retrieved[0] if retrieved else ""
        context = "\n".join(f"- {item}" for item in fit_budget(retrieved, budget)) or "(no memories)"
        prompt = official.ANSWER_PROMPT.replace("<context>", context).replace("<question>", question["question"])
        response = llm.chat([{"role": "user", "content": prompt}], max_tokens=1500, tag="beam-answer-errors")
        if ability == "event_ordering":
            ordering = beam.order_events(question["rubric"], response)
            score = 0.0 if ordering["tau_norm"] != ordering["tau_norm"] else ordering["tau_norm"]
        else:
            score, _ = beam.judge(question["rubric"], response)
        verdicts = {}
        for name, text in (("material", material), ("brief", brief), ("answer", response)):
            verdicts[name], verdicts[name + "_why"] = check(question["question"], ref, text, "beam-errors-" + name)
        row.update(material_tokens=len(material) // 4, brief_text=brief, response=response, score=score, **verdicts,
                   stage=stage(verdicts["material"], verdicts["brief"], verdicts["answer"]))
    except Exception as error:  # noqa: BLE001 - reported, not fatal
        row.update(error=f"{type(error).__name__}: {error}"[:300], trace=traceback.format_exc()[-600:])
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="100K")
    parser.add_argument("--run", required=True, help="the run whose wrong answers are diagnosed (its rows and stores)")
    parser.add_argument("--rows", nargs="+", type=Path, required=True)
    parser.add_argument("--stores", nargs="+", required=True, help="first-last=WORKDIR: the run's store for those conversations")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--below", type=float, default=0.75)
    parser.add_argument("--budget", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--subset", type=Path, help="only the questions of an earlier diagnosis (its rows.jsonl)")
    parser.add_argument("--subset-stage", default="", help="with --subset: only those placed in this stage")
    parser.add_argument("--tag", default="errors", help="name of the store copy (two diagnoses may run at once)")
    args = parser.parse_args()
    old = {}
    for path in args.rows:
        for line in open(path):
            r = json.loads(line)
            old[(r["conversation"], r["ability"], r["index"])] = r
    subset = None
    if args.subset:
        subset = {(r["conversation"], r["ability"], r["index"]) for r in map(json.loads, open(args.subset))
                  if not args.subset_stage or r.get("stage") == args.subset_stage}
    args.out.mkdir(parents=True, exist_ok=True)
    conversations = {c["conversation_id"]: c for c in beam.load(args.split)}
    started, before = time.time(), dict(llm.STATS)
    rows = []
    with open(args.out / "rows.jsonl", "w") as out:
        for spec in args.stores:
            span, workdir = spec.split("=", 1)
            first, last = map(int, span.split("-"))
            system = Ctx(Path(workdir) / f"{args.run}-{args.tag}-home", name=args.run, prompt_file=BENCH / "prompts/ctx-general.txt",
                         episodes=max(5, args.budget // 800), budget=args.budget, reuse_from=Path(workdir) / f"{args.run}-home",
                         brief=True, recall={"time_chains": "false", "entity_hops": "false"})
            system.setup()
            try:
                jobs = []
                for cid in map(str, range(first, last + 1)):
                    conversation = conversations[cid]
                    last_day = beam.anchor_date(conversation["chat"][-1][0].get("time_anchor")) if conversation["chat"][-1] else None
                    probing = ast.literal_eval(conversation["probing_questions"])
                    for ability in beam.ABILITIES:
                        for index, question in enumerate(probing.get(ability, [])):
                            r = old.get((cid, ability, index))
                            if r and r["score"] < args.below and (subset is None or (cid, ability, index) in subset):
                                jobs.append((conversation, ability, index, question, last_day.replace("/", "-") if last_day else None, r))
                with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
                    for row in pool.map(lambda j: diagnose(system, j[0], j[1], j[2], j[3], args.budget, j[4], j[5]), jobs):
                        rows.append(row)
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out.flush()
            finally:
                system.teardown()
    table = collections.defaultdict(collections.Counter)
    for r in rows:
        table[r["ability"]][r.get("stage", "error")] += 1
        table["all"][r.get("stage", "error")] += 1
    stages = ["material", "brief", "answer", "judge", "error"]
    summary = {"run": args.run, "questions": len(rows), "below": args.below, "seconds": round(time.time() - started),
               "repeat_fail": sum(r.get("score", 0) < args.below for r in rows),
               "stages": {a: {s: c[s] for s in stages if c[s]} for a, c in sorted(table.items())},
               "usage": {k: llm.STATS[k] - before[k] for k in ("calls", "input_tokens", "output_tokens")}}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
