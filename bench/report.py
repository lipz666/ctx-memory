"""Build the comparison tables (Markdown) from results/trackA, trackB1 and trackB2.

  python report.py > ../docs/benchmark-tables.md
"""
import json
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def wilson(successes, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def pct(x):
    return f"{100 * x:.1f}%"


def frac(text):
    a, b = text.split("/")
    return int(a), int(b)


def rows(path):
    return [json.loads(line) for line in open(path)] if path.exists() else []


def track_a():
    """Every system on the same 60-question subset (Mem0 ran only on it); 120-question
    accuracy is shown where available."""
    base = RESULTS / "trackA"
    subset_ids = set(json.loads((base / "subset-60.json").read_text())) if (base / "subset-60.json").exists() else None
    out = ["## Track A: LongMemEval_S", "",
           "Same 60 questions for every system (stratified half of the 120-question sample). 95% Wilson interval in parentheses.", ""]
    names = ["no-memory", "ctx", "ctx-general", "naive-rag", "ctx-general-wide", "mem0", "full-context"]
    types = ["single-session-user", "single-session-assistant", "single-session-preference", "multi-session",
             "knowledge-update", "temporal-reasoning", "abstention"]
    short = ["SS-user", "SS-asst", "SS-pref", "multi", "update", "temporal", "abstain"]
    out.append("| System | Accuracy on 60 (95% CI) | " + " | ".join(short) + " | Accuracy on 120 | Retrieved tokens | Engine LLM calls |")
    out.append("|" + "---|" * (len(short) + 5))
    for name in names:
        items = rows(base / name / "rows.jsonl")
        if not items:
            continue
        summary = json.loads((base / name / "summary.json").read_text()) if (base / name / "summary.json").exists() else {}
        chosen = [r for r in items if subset_ids is None or r["question_id"] in subset_ids]
        k, n = sum(r["correct"] for r in chosen), len(chosen)
        lo, hi = wilson(k, n)
        by_type = {}
        for r in chosen:
            by_type.setdefault("abstention" if r["abstention"] else r["question_type"], []).append(r["correct"])
        cells = [f"{sum(by_type.get(t, []))}/{len(by_type.get(t, []))}" for t in types]
        full = f"{pct(summary['accuracy'])}" if summary.get("questions") == 120 else "–"
        calls = summary.get("usage", {}).get("engine_llm_calls", "–") if isinstance(summary.get("usage"), dict) else "–"
        tokens = "all history" if name == "full-context" else round(sum(r.get("kept_tokens", 0) for r in chosen) / max(n, 1))
        errors = sum("error" in r for r in chosen)
        label = name + (f" ({errors} errors)" if errors else "")
        out.append(f"| {label} | {pct(k / max(n, 1))} ({pct(lo)}–{pct(hi)}) | " + " | ".join(cells) + f" | {full} | {tokens} | {calls} |")
    return out


def track_b1():
    base = RESULTS / "trackB1"
    out = ["## Track B1: retrieval on identical memory text", ""]
    cats = ["paraphrase_zh", "paraphrase_en", "cross_lingual", "conversational", "error", "lexical", "updated_fact"]
    out.append("| System | Recall@1 | Recall@3 | MRR | Test-half R@3 | " + " | ".join(cats) + " | Newer fact ranked first | Off-topic: returned anything | Other project: returned anything |")
    out.append("|" + "---|" * (len(cats) + 8))
    for name in ["ctx", "ctx-wide", "ctx-keyword", "naive-rag", "mem0"]:
        path = base / name / "summary.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        neg = s["negatives_returned_anything"]
        out.append(f"| {name} | {pct(s['recall@1'])} | {pct(s['recall@3'])} | {s['mrr']:.3f} | {pct(s['test_split_recall@3'])} | "
                   + " | ".join(s["by_category_recall@3"].get(c, "–") for c in cats)
                   + f" | {s['updated_fact_newer_first']} | {neg['negative']} | {neg['negative_cross_project']} |")
    return out


def track_b2():
    base = RESULTS / "trackB2"
    out = ["## Track B2: multi-session coding tasks (OpenClaw)", ""]
    out.append("| Configuration | Knowledge applied (95% CI) | convention | fact | update | cross-project apply | Isolation kept | Agent failures | Mean s/session | Agent input tokens |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for name in ["no-memory", "openclaw-native", "mem0-mcp", "ctx-mcp", "ctx"]:
        path = base / name / "summary.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        sessions = rows(base / name / "sessions.jsonl")
        applies = [r for r in sessions if r["role"] == "apply"]
        k, n = sum(r["passed"] for r in applies), len(applies)
        lo, hi = wilson(k, n)
        kinds = s["apply_by_kind"]
        out.append(f"| {name} | {k}/{n} = {pct(k / max(n, 1))} ({pct(lo)}–{pct(hi)}) | {kinds.get('convention', '–')} | {kinds.get('fact', '–')} | "
                   f"{kinds.get('update', '–')} | {kinds.get('cross_project', '–')} | {s['isolation_success']} | {s['agent_failures']} | "
                   f"{s['mean_session_seconds']} | {s['input_tokens']:,} |")
    return out


if __name__ == "__main__":
    for section in (track_a(), track_b1(), track_b2()):
        print("\n".join(section))
        print()
