"""Generate the Track B1 retrieval set (coding memories and natural queries).

  python -m tracks.b1_generate --out data/b1

Per project the model writes realistic memories a coding agent would keep, then queries in
fixed styles for a subset of them. Filters drop queries that copy long spans from their
memory. Cross-project and off-topic negatives and updated-fact pairs are added. The set is
new: no system (ctx included) is tuned on it.
"""
import argparse
import concurrent.futures
import json
import random
import re
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from common import llm  # noqa: E402

PROJECTS = {
    "orderflow": "Java 21 / Spring Boot order management service, PostgreSQL, Kafka, Gradle, deployed on Kubernetes via Argo CD",
    "pixelcraft": "React Native mobile app with Expo, TypeScript, Redux Toolkit, Detox e2e tests, App Store / Play Store releases",
    "tidewatch": "Python 3.12 data pipeline: Airflow DAGs, dbt models on Snowflake, pandas, great_expectations checks",
    "kilnrs": "Rust CLI and library for image processing, cargo workspace, criterion benchmarks, cross-compiled for Linux ARM",
    "harbor-api": "Go 1.23 REST API with chi router, sqlc, Redis cache, gRPC to an internal auth service, deployed on AWS ECS",
    "lumen-docs": "Next.js 15 documentation site with MDX, Tailwind, Algolia search, Vercel preview deployments, i18n (zh/en)",
}
STYLES = {
    "paraphrase_zh": "a Chinese question that asks about it in different words (no copied phrases)",
    "paraphrase_en": "an English question that asks about it in different words (no copied phrases)",
    "cross_lingual": "a question in the OTHER language than the memory (Chinese if the memory is English, English if Chinese)",
    "conversational": "a long, natural developer message (2-3 sentences) describing what they are about to do, where this memory would help; it must not name the memory's key terms directly",
    "error": "the raw error output or log line a developer would paste when this memory applies (only if the memory is about an error or failure; otherwise write a short symptom description)",
    "lexical": "a short query of 1-4 words using the exact identifier, command, or name from the memory",
}

MEMORY_PROMPT = """Invent {count} realistic, specific memories a coding agent would want to keep across sessions for this project:
Project "{name}": {stack}

Mix: conventions and commands (fact), mistakes and their fixes with exact error messages (lesson), multi-step procedures (skill). Each is one self-contained statement, 1-2 sentences, specific (names, paths, commands, versions). Write about half in Chinese and half in English. Make them distinct from each other.
Return only JSON: {{"memories":[{{"type":"fact|lesson|skill","language":"zh|en","content":"..."}}]}}"""
GLOBAL_PROMPT = """Invent {count} realistic memories about a developer's personal preferences and cross-project habits that a coding agent should remember (tools, style, communication, workflow, machine setup). Specific, one statement each, about half Chinese and half English.
Return only JSON: {{"memories":[{{"type":"fact","language":"zh|en","content":"..."}}]}}"""
QUERY_PROMPT = """A coding agent has this stored memory (project: {project}):
"{content}"

Write {style_desc}. The query must be something a developer would really type when this memory is needed, and a reader should be able to tell the memory answers it. Do not copy any phrase of 4 or more words from the memory.
Return only JSON: {{"query":"..."}}"""
NEGATIVE_PROMPT = """Write {count} questions a developer might ask a coding assistant that have NOTHING to do with any specific project knowledge: general programming trivia, casual chat, math, unrelated tasks. Half Chinese, half English. Return only JSON: {{"queries":["..."]}}"""
UPDATE_PROMPT = """This is a stored memory for project {project}:
"{content}"

Write an UPDATED version stating that this has changed (a new value, command, path or rule replaces the old one), as a memory written later. Then write a question whose correct answer is the NEW value. Same language as the memory.
Return only JSON: {{"updated":"...","query":"..."}}"""


def ask(prompt, tag):
    reply = llm.chat([{"role": "user", "content": prompt}], max_tokens=4000, temperature=0, tag=tag)
    text = reply.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


def copies_memory(query, content, n=4):
    words = re.findall(r"\w+", content.lower())
    grams = {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}
    text = " ".join(re.findall(r"\w+", query.lower()))
    if any(g in text for g in grams):
        return True
    cjk = re.sub(r"[^一-鿿]", "", content)
    return any(cjk[i:i + 8] in query for i in range(max(0, len(cjk) - 7)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=BENCH / "data/b1")
    parser.add_argument("--per-project", type=int, default=40)
    parser.add_argument("--global-count", type=int, default=30)
    args = parser.parse_args()
    rng = random.Random(20260929)
    args.out.mkdir(parents=True, exist_ok=True)
    memories = []
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        jobs = {pool.submit(ask, MEMORY_PROMPT.format(count=args.per_project, name=p, stack=s), f"b1-mem-{p}"): p for p, s in PROJECTS.items()}
        jobs[pool.submit(ask, GLOBAL_PROMPT.format(count=args.global_count), "b1-mem-global")] = "global"
        for job, project in jobs.items():
            for i, m in enumerate(job.result()["memories"]):
                memories.append({"key": f"{project}-{i:02d}", "scope": project, "type": m["type"], "language": m["language"], "content": m["content"].strip()})
    by_scope = {}
    for m in memories:
        by_scope.setdefault(m["scope"], []).append(m)
    # Queries: each project memory gets 1 style, cycling so styles are balanced.
    plans = []
    styles = list(STYLES)
    for scope, items in by_scope.items():
        for i, m in enumerate(items):
            plans.append((m, styles[(i + len(scope)) % len(styles)]))
    # Updated facts: 3 per project; the update is stored as a newer memory.
    updates = [(m, scope) for scope, items in by_scope.items() if scope != "global" for m in rng.sample(items, 3)]
    queries = []
    with concurrent.futures.ThreadPoolExecutor(64) as pool:
        query_jobs = [(m, style, pool.submit(ask, QUERY_PROMPT.format(project=m["scope"], content=m["content"], style_desc=STYLES[style]), f"b1-q-{m['key']}-{style}")) for m, style in plans]
        update_jobs = [(m, pool.submit(ask, UPDATE_PROMPT.format(project=scope, content=m["content"]), f"b1-upd-{m['key']}")) for m, scope in updates]
        negative_job = pool.submit(ask, NEGATIVE_PROMPT.format(count=40), "b1-neg")
        dropped = 0
        for m, style, job in query_jobs:
            try:
                query = job.result()["query"].strip()
            except Exception:  # noqa: BLE001
                dropped += 1
                continue
            if style != "lexical" and copies_memory(query, m["content"]):
                dropped += 1
                continue
            project = None if m["scope"] == "global" else m["scope"]
            if project is None:
                project = rng.choice(list(PROJECTS))
            queries.append({"project": project, "query": query, "expected": [m["key"]], "category": style})
        for m, job in update_jobs:
            result = job.result()
            new_key = m["key"] + "-v2"
            memories.append({"key": new_key, "scope": m["scope"], "type": m["type"], "language": m["language"], "content": result["updated"].strip(), "supersedes": m["key"]})
            queries.append({"project": m["scope"], "query": result["query"].strip(), "expected": [new_key], "stale": [m["key"]], "category": "updated_fact"})
        for q in negative_job.result()["queries"]:
            queries.append({"project": rng.choice(list(PROJECTS)), "query": q, "expected": [], "category": "negative"})
    # Cross-project negatives: ask another project's question in a different project.
    project_queries = [q for q in queries if q["category"] in ("paraphrase_en", "paraphrase_zh", "cross_lingual") and not q["expected"][0].startswith("global")]
    for q in rng.sample(project_queries, 40):
        other = rng.choice([p for p in PROJECTS if p != q["project"]])
        queries.append({"project": other, "query": q["query"], "expected": [], "category": "negative_cross_project"})
    rng.shuffle(queries)
    for i, q in enumerate(queries):
        q["id"] = f"b1q{i:03d}"
        q["split"] = "dev" if i % 2 == 0 else "test"
    with open(args.out / "memories.jsonl", "w") as f:
        for m in memories:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(args.out / "queries.jsonl", "w") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    counts = {}
    for q in queries:
        counts[q["category"]] = counts.get(q["category"], 0) + 1
    print(json.dumps({"memories": len(memories), "queries": len(queries), "dropped_copying_queries": dropped, "by_category": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
