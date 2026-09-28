"""Exercise the M0 labeling/replay pipeline on fictional, isolated trajectories.

This is a procedure check, never a substitute for consented internal traces or
10% human review. It writes only fictional step text and aggregate results.
"""

import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "target/debug/ctx"
OUT = ROOT / "docs/m0-synthetic-2026-09-27.json"
REVIEW = ROOT / "docs/m0-review-sample-2026-09-27.json"


def credential(reference):
    if reference.startswith("keychain:"):
        return subprocess.run(["security", "find-generic-password", "-a", "default",
                               "-s", reference[9:], "-w"], check=True,
                              capture_output=True, text=True).stdout.strip()
    if reference.startswith("env:"):
        return os.environ[reference[4:]]
    raise ValueError("expected keychain: or env: credential reference")


def main():
    base = os.environ["CTX_TEST_LIVE_BASE_URL"].rstrip("/")
    model = os.environ["CTX_TEST_LIVE_MODEL"]
    key = credential(os.environ["CTX_TEST_LIVE_CREDENTIAL_REF"])
    with tempfile.TemporaryDirectory() as directory:
        env = dict(os.environ, CTX_HOME=directory)

        def ctx(*args):
            return subprocess.run([BIN, *args], env=env, check=True,
                                  capture_output=True, text=True).stdout.strip()

        ctx("init")
        cards = {
            "inventory": "Reservation batches must commit atomically; retries with the same order ID must not consume stock twice.",
            "ledger": "Payment reconciliation must use decimal arithmetic, deduplicate row IDs, and reject refunds above the paid amount.",
            "deploy": "Before deploying payments, run the schema migration and verify the queue worker is healthy.",
        }
        scopes = {"inventory": "inventory", "ledger": "ledger", "deploy": "payments"}
        memory = {
            "inventory": ctx("remember", cards["inventory"],
                             "--kind", "lesson", "--scope", "inventory", "--trigger-text", "reserve_batch"),
            "ledger": ctx("remember", cards["ledger"],
                          "--kind", "lesson", "--scope", "ledger", "--trigger-text", "reconcile payment"),
            "deploy": ctx("remember", cards["deploy"],
                          "--kind", "lesson", "--scope", "payments", "--trigger-text", "deploy payments"),
        }
        moments = [
            ("inventory", "The reserve_batch test fails after an OutOfStock exception; inspect partial writes.", ["inventory"]),
            ("inventory", "Fix reserve_batch so retries of an existing order ID are idempotent.", ["inventory"]),
            ("inventory", "Review the checkout page colors.", []),
            ("ledger", "The reconcile payment report differs by one cent after three exports.", ["ledger"]),
            ("ledger", "Investigate duplicate payment IDs in reconcile payment rows.", ["ledger"]),
            ("ledger", "Rename the dashboard navigation heading.", []),
            ("payments", "Prepare to deploy payments to staging.", ["deploy"]),
            ("payments", "A schema mismatch happened during deploy payments.", ["deploy"]),
            ("inventory", "Prepare to deploy payments to staging in another project.", []),
            ("payments", "Check weekly support ticket volume.", []),
            ("inventory", "The reserve_batch API is mentioned in release notes, no code change needed.", []),
            ("ledger", "Check whether refunds exceed their original payment during reconcile payment.", ["ledger"]),
        ]
        prompt = {
            "memories": [{"id": value, "scope": scopes[name], "content": cards[name]}
                         for name, value in memory.items()],
            "steps": [{"index": i, "project": project, "text": query}
                      for i, (project, query, _) in enumerate(moments)],
            "instruction": "For each step, select memory IDs that would materially help the Agent's next action. Respect project scope. Return only JSON: {\"labels\":[{\"index\":0,\"ids\":[\"memory-id\"]},...]}. Include every index, including empty ids.",
        }
        body = {"model": model, "messages": [{"role": "user", "content": json.dumps(prompt)}],
                "temperature": 0, "max_tokens": 1600}
        request = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Authorization": f"Bearer {key}",
                                                  "Content-Type": "application/json",
                                                  "User-Agent": "curl/8.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            answer = json.load(response)
        content = answer["choices"][0]["message"]["content"].strip()
        if content.startswith("```json"):
            content = content[7:].removesuffix("```").strip()
        elif content.startswith("```"):
            content = content[3:].removesuffix("```").strip()
        raw_labels = json.loads(content)["labels"]
        labels = {entry["index"]: entry["ids"] for entry in raw_labels}
        valid_ids = set(memory.values())
        if set(labels) != set(range(len(moments))) or any(
                not isinstance(ids, list) or not set(ids) <= valid_ids for ids in labels.values()):
            raise ValueError("LLM labels missing steps or containing invalid IDs")
        trace = Path(directory) / "trace.jsonl"
        rows = []
        agreement = 0
        for index, (project, query, names) in enumerate(moments):
            expected = [memory[name] for name in names]
            agreement += set(labels[index]) == set(expected)
            rows.append({"query": query, "project": project,
                         "expected_memory_ids": labels[index]})
        trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        output = Path(directory) / "report.json"
        ctx("eval", "replay", str(trace), "--output", str(output))
        result = json.loads(output.read_text())
        result.update({"source": "fictional trajectories", "labeler": "LLM initial labels",
                       "label_model_reported": answer.get("model"),
                       "designer_agreement_steps": agreement,
                       "human_review_status": "pending", "human_review_count": 2})
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        sample = random.Random(20260927).sample(range(len(moments)), 2)
        REVIEW.write_text(json.dumps([
            {"step": i, "project": moments[i][0], "query": moments[i][1],
             "llm_selected": [name for name, mid in memory.items() if mid in labels[i]],
             "human_selected": None, "reviewer": None, "reason": None}
            for i in sample], ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
