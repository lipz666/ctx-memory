"""ctx-m read path (docs/ctx-m-design.md, section 5): record-level retrieval over cards,
intent by rules, program-computed figures, and a rendering of at most `budget` tokens.
Mode "render" uses no model; mode "compose" lets one model call write the note from a
wider set of candidates."""
import hashlib
import math
import re
import threading
from collections import Counter, defaultdict
from datetime import date as Date
from pathlib import Path

from common import llm

COMPOSE = (Path(__file__).parent / "prompts/compose.txt").read_text()
CHARS_PER_TOKEN = 4
STOP = set("a an the of to in on at for and or is are was were be been i me my we our you your it its this that "
           "with from by as about what which who whom how many much did do does have has had can could would should "
           "will there their them they any all some throughout across conversations conversation".split())

INTENTS = [
    ("time", r"\bhow (many|much) (days|weeks|months|years|time)\b|\bhow long\b|\bwhen\b|\bwhat date\b|\bwhich date\b|多久|哪天|什么时候|几天"),
    ("order", r"\border\b|\bsequence\b|\bchronolog|\bfirst\b.*\b(then|next|after)\b|先后|顺序"),
    ("summary", r"\bsummar|\boverview\b|\brecap\b|\bprogress(ed)?\b|\bevolved?\b|\bcomprehensive\b|总结|概括|进展"),
    ("count", r"\bhow many\b|\bnumber of\b|\bcount\b|\btotal\b|多少个|几个|几次|一共"),
    ("latest", r"\b(current(ly)?|now|latest|most recent|updated?|still|final)\b|现在|目前|最新"),
    ("yesno", r"^\s*(have|did|do|does|has|was|were|is|am|are)\s+(i|we|my)\b|有没有|是否"),
]


def intent_of(question):
    for name, pattern in INTENTS:
        if re.search(pattern, question, re.I):
            return name
    return "fact"


def words(text):
    return [w for w in re.findall(r"[a-z0-9][a-z0-9.+#-]*|[一-鿿]", text.lower()) if w not in STOP]


def by_label(by):
    return {"assistant": " [assistant's suggestion]", "adopted": " [assistant's suggestion, adopted]"}.get(by, "")


def day(text):
    try:
        return Date.fromisoformat(str(text)[:10])
    except (TypeError, ValueError):
        return None


class Index:
    """Vectors from a ctx instance (`embed(texts, kind)`), cached by text; BM25 per query."""

    def __init__(self, embed):
        self.embed = embed
        self.cache = {}
        self.lock = threading.Lock()

    def vectors(self, texts, kind="document"):
        keys = [hashlib.sha1((kind + "\0" + t).encode()).hexdigest() for t in texts]
        with self.lock:
            missing = sorted({(k, t) for k, t in zip(keys, texts) if k not in self.cache})
        if missing:
            vectors = self.embed([t for _, t in missing], kind)
            with self.lock:
                for (k, _), v in zip(missing, vectors):
                    self.cache[k] = v
        with self.lock:
            return [self.cache[k] for k in keys]

    def scores(self, query, texts):
        """0.75 x cosine + 0.25 x BM25 (normalized by the best) for each text."""
        if not texts:
            return []
        q = self.vectors([query], "query")[0]
        docs = self.vectors(texts)
        sem = [sum(a * b for a, b in zip(q, d)) for d in docs]
        terms = set(words(query))
        tokenized = [words(t) for t in texts]
        df = Counter(w for toks in tokenized for w in set(toks) if w in terms)
        n, avg = len(texts), sum(len(t) for t in tokenized) / len(texts) or 1
        bm = []
        for toks in tokenized:
            tf = Counter(toks)
            bm.append(sum(math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) * tf[w] * 2.2 /
                          (tf[w] + 1.2 * (0.25 + 0.75 * len(toks) / avg)) for w in terms if tf[w]))
        top = max(bm) or 1
        return [0.75 * s + 0.25 * b / top for s, b in zip(sem, bm)]

    def rank_cards(self, store, text):
        cards = list(store.cards.values())
        scored = self.scores(text, [f"{c['title']}: {c['headline']}" for c in cards])
        return [c["id"] for _, c in sorted(zip(scored, cards), key=lambda p: -p[0])]


def records(store):
    """Every retrievable unit with its rendering. kind: value/item/event/note/conflict/
    headline/instruction/preference/session."""
    out = []
    for card in store.cards.values():
        t = card["title"]
        out.append({"card": card["id"], "kind": "headline", "turn": card["updated_turn"],
                    "text": f"{t}: {card['headline']}", "show": card["headline"]})
        for v in card["values"]:
            history = "" if v["status"] == "current" else " (later replaced)"
            out.append({"card": card["id"], "kind": "value", "turn": v["turn"], "ref": v, "text": f"{t} · {v['key']}: {v['value']}",
                        "show": f"#{v['turn']} ({v['date']}) {v['key']}: {v['value']}{history}{by_label(v['by'])}"})
        for i in card["items"]:
            times = f", mentioned {len(i['mentions'])}x" if len(i["mentions"]) > 1 else ""
            out.append({"card": card["id"], "kind": "item", "turn": i["turn"], "ref": i, "text": f"{t} · {i['list']}: {i['name']}",
                        "show": f"#{i['turn']} ({i['date']}) {i['list']}: {i['name']}{by_label(i['by'])}{times}"})
        for e in card["events"]:
            out.append({"card": card["id"], "kind": "event", "turn": e["turn"], "ref": e, "text": f"{t} · {e['what']} ({e['kind']} {e['date']})",
                        "show": f"#{e['turn']} {e['kind']} {e['date'] or 'date unknown'}: {e['what']} (said {e['said']}){by_label(e['by'])}"})
        for n in card["notes"]:
            span = f"#{n['first']}" if n["first"] == n["last"] else f"#{n['first']}-{n['last']}"
            out.append({"card": card["id"], "kind": "note", "turn": n["first"], "text": f"{t} · {n['text']}",
                        "show": f"{span} ({n['date']}) {n['text']}"})
        for c in card["conflicts"]:
            out.append({"card": card["id"], "kind": "conflict", "turn": c["turn"], "ref": c, "text": f"{t} · {c['existing']} / {c['new']}",
                        "show": f"CONFLICT (no change stated): earlier \"{c['existing']}\" vs #{c['turn']} ({c['date']}) \"{c['new']}\""})
    for layer in ("instructions", "preferences"):
        for e in getattr(store, layer):
            out.append({"card": None, "kind": layer[:-1], "turn": e["turn"], "text": e["text"], "show": e["text"]})
    for t in store.turns:
        if t["user"]:
            text = " ".join(t["user"].split())
            out.append({"card": None, "kind": "turn", "turn": t["turn"], "text": text[:600],
                        "show": f"#{t['turn']} ({t['date']}) USER: {text}"})
    for s in store.sessions:
        if s["summary"]:
            out.append({"card": None, "kind": "session", "turn": s["first"], "text": s["summary"],
                        "show": f"#{s['first']}-{s['last']} ({s['date']}) {s['summary']}"})
    return out


class Reader:
    def __init__(self, index, mode="render", stats=None):
        self.index, self.mode, self.stats = index, mode, stats

    def recall(self, store, question, budget=1000, tag="ctxm-compose"):
        """Lines to inject, at most `budget` tokens in total."""
        recs = records(store)
        if not recs:
            return []
        for rec, score in zip(recs, self.index.scores(question, [r["text"] for r in recs])):
            rec["score"] = score
        intent = intent_of(question)
        if self.mode == "compose":
            material = "\n".join(self.render(store, recs, intent, 4000))
            reply = llm.chat([{"role": "system", "content": COMPOSE},
                              {"role": "user", "content": f"Message: {question}\n\nMemory records:\n{material}"}],
                             max_tokens=4000, tag=tag, stats=self.stats)
            return clip([line for line in reply.strip().splitlines() if line.strip()], budget)
        return self.render(store, recs, intent, budget)

    # ---- rendering
    def render(self, store, recs, intent, budget):
        room = budget * CHARS_PER_TOKEN
        lines = []

        def put(line, limit=None):
            nonlocal room
            line = line if limit is None or len(line) <= limit else line[:limit - 1] + "…"
            if len(line) + 3 > room:
                return False
            lines.append(line)
            room -= len(line) + 3
            return True

        by_card = defaultdict(list)
        for r in recs:
            if r["card"]:
                by_card[r["card"]].append(r)
        card_rank = sorted(by_card, key=lambda c: -max(r["score"] for r in by_card[c]))
        best = max(r["score"] for r in recs)

        # Standing layer (about 15%): the most relevant instructions, then preferences.
        # Room is reserved now; the lines go last, next to the question.
        standing, spent = [], 0
        for kind, label, keep in (("instruction", "The user's standing instruction — apply it in your answer", 3),
                                  ("preference", "The user's preference — take it into account", 2)):
            for r in sorted((r for r in recs if r["kind"] == kind), key=lambda r: -r["score"])[:keep]:
                if kind == "preference" and r["score"] < 0.8 * best:
                    continue
                line = f"{label}: {r['show']} (#{r['turn']})"[:400]
                if spent + len(line) + 3 <= room * 0.15:
                    standing.append(line)
                    spent += len(line) + 3
        room -= spent

        # Program-computed figures.
        top_cards = card_rank[:4]
        for line in self.computed(store, recs, by_card, top_cards, intent):
            put(line, 900)

        # Evidence: the user's own words that match best (about a quarter), then cards.
        if intent in ("order", "summary"):
            self.chronological(store, by_card, card_rank[:2], recs, intent, put, lambda: room)
        else:
            quota = room * 0.25
            for r in sorted((r for r in recs if r["kind"] == "turn"), key=lambda r: -r["score"])[:3]:
                line = r["show"] if len(r["show"]) <= 330 else r["show"][:329] + "…"
                if len(line) + 3 > quota:
                    break
                quota -= len(line) + 3
                put(line)
            self.by_relevance(store, by_card, card_rank, put, intent)
        return lines + standing

    def computed(self, store, recs, by_card, top_cards, intent):
        out = []
        if intent == "count":
            lists = defaultdict(list)
            for c in top_cards:
                for r in by_card[c]:
                    if r["kind"] == "item":
                        lists[(c, r["ref"]["list"].lower())].append(r)
            ranked = sorted(lists.items(), key=lambda kv: -max(r["score"] for r in kv[1]))
            for (c, _), members in ranked[:2]:
                card = store.cards[c]
                mine = [r["ref"] for r in members if r["ref"]["by"] != "assistant"]
                theirs = [r["ref"] for r in members if r["ref"]["by"] == "assistant"]
                text = f"Count — {card['title']} · {members[0]['ref']['list']}: {len(mine)} stated by the user"
                if mine:
                    text += ": " + ", ".join(f"{i['name']} (#{i['turn']})" for i in sorted(mine, key=lambda i: i["turn"]))
                if theirs:
                    text += "; only suggested by the assistant: " + ", ".join(i["name"] for i in theirs)
                out.append(text)
        if intent in ("latest", "count", "fact", "time"):
            keys = defaultdict(list)
            for c in top_cards:
                for r in by_card[c]:
                    if r["kind"] == "value":
                        keys[(c, r["ref"]["key"].lower())].append(r)
            best = max(r["score"] for r in recs)
            for (c, _), members in sorted(keys.items(), key=lambda kv: -max(r["score"] for r in kv[1]))[:1 if intent != "latest" else 2]:
                if intent != "latest" and max(r["score"] for r in members) < 0.9 * best:
                    continue
                chain = sorted((r["ref"] for r in members), key=lambda v: v["turn"])
                if len(chain) > 1:
                    steps = " → ".join(f"{v['value']} (#{v['turn']}, {v['date']})" for v in chain[-4:])
                    out.append(f"History of {store.cards[c]['title']} · {chain[-1]['key']}: {steps}; latest: {chain[-1]['value']}")
        if intent == "time":
            events = sorted((r for c in top_cards for r in by_card[c] if r["kind"] == "event" and day(r["ref"]["date"])),
                            key=lambda r: -r["score"])[:3]
            events = sorted((r["ref"] for r in events), key=lambda e: day(e["date"]))
            pairs = [f"\"{a['what']}\" ({a['date']}) → \"{b['what']}\" ({b['date']}): {(day(b['date']) - day(a['date'])).days} days"
                     for i, a in enumerate(events) for b in events[i + 1:]]
            if pairs:
                out.append("Day counts between the closest-matching dated events: " + "; ".join(pairs))
        if intent == "yesno":
            for c in top_cards[:2]:
                for r in by_card[c]:
                    if r["kind"] == "conflict" and r["score"] >= 0.75 * max(x["score"] for x in by_card[c]):
                        out.append(r["show"] + " — the user did not say which is true")
        return out

    def by_relevance(self, store, by_card, card_rank, put, intent):
        """Top cards: headline, then the best-matching records, in turn order. Conflicts
        only for yes/no questions (elsewhere they pull answers off course)."""
        for rank, c in enumerate(card_rank[:4]):
            card = store.cards[c]
            if not put(f"[{card['title']}] {card['headline']}", 500):
                break
            skip = ("headline",) if intent == "yesno" else ("headline", "conflict")
            keep = sorted((r for r in by_card[c] if r["kind"] not in skip), key=lambda r: -r["score"])
            keep = keep[:12 if rank == 0 else 6]
            for r in sorted(keep, key=lambda r: r["turn"]):
                put("  " + r["show"], 320)

    def chronological(self, store, by_card, cards, recs, intent, put, left):
        """Order and summary: headlines, then notes/events of the top cards in turn order,
        the most relevant first when they do not all fit."""
        for c in cards:
            put(f"[{store.cards[c]['title']}] {store.cards[c]['headline']}", 500)
        pool = [r for c in cards for r in by_card[c] if r["kind"] in ("note", "event")]
        if intent == "summary":
            pool += [r for r in recs if r["kind"] == "session"]
        chosen, size = [], 0
        for r in sorted(pool, key=lambda r: -r["score"]):
            line = "  " + (r["show"] if len(r["show"]) <= 260 else r["show"][:259] + "…")
            if size + len(line) + 3 > left():
                continue
            chosen.append((r["turn"], line))
            size += len(line) + 3
        for _, line in sorted(chosen):
            put(line)


def clip(lines, budget):
    out, room = [], budget * CHARS_PER_TOKEN
    for line in lines:
        if len(line) + 3 > room:
            break
        out.append(line)
        room -= len(line) + 3
    return out
