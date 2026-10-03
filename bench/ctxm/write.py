"""ctx-m write path: one model call per session turns the conversation into card changes
(docs/ctx-m-design.md, section 4); the program applies them (history, dedup, merging)."""
import json
import re
from pathlib import Path

from common import llm

from .store import BY, EVENT_KINDS

PROMPT = (Path(__file__).parent / "prompts/extract.txt").read_text()
ASSISTANT_CHARS = 1500   # head of each assistant reply kept for extraction
OUTLINE_CHARS = 700      # plus its headings and bold lines
CARD_CONTEXT_CHARS = 14000


def assistant_view(text):
    """The head of a reply plus its outline (headings, bold phrases) from the rest."""
    if len(text) <= ASSISTANT_CHARS + OUTLINE_CHARS:
        return text
    head, rest = text[:ASSISTANT_CHARS], text[ASSISTANT_CHARS:]
    outline = []
    for line in rest.splitlines():
        line = line.strip()
        if line.startswith("#") or re.match(r"^(\d+\.|[-*])\s+\*\*", line) or line.startswith("**"):
            outline.append(line[:160])
    tail = " / ".join(outline)[:OUTLINE_CHARS]
    return head + " …" + (f" [rest of the reply: {tail}]" if tail else "")


def transcript(turns):
    lines = []
    for t in turns:
        if t["user"]:
            lines.append(f"#{t['turn']} USER: {t['user']}")
        if t["assistant"]:
            lines.append(f"#{t['turn']} ASSISTANT: {assistant_view(t['assistant'])}")
    return "\n\n".join(lines)


def card_context(card, max_items=40):
    """An existing card as the extractor sees it: headline, current values, lists, last notes."""
    lines = [f"[{card['id']}] {card['title']} — {card['headline']}"]
    current = [v for v in card["values"] if v["status"] == "current"]
    if current:
        lines.append("  values: " + "; ".join(f"{v['key']} = {v['value']} (#{v['turn']})" for v in current[-25:]))
    lists = {}
    for item in card["items"]:
        lists.setdefault(item["list"], []).append(item["name"] + ("" if item["by"] != "assistant" else " [assistant]"))
    for name, members in lists.items():
        lines.append(f"  list '{name}': " + ", ".join(members[-max_items:]))
    if card["events"]:
        lines.append("  events: " + "; ".join(f"{e['date']} {e['what']}" for e in card["events"][-8:]))
    if card["notes"]:
        lines.append("  last notes: " + " | ".join(n["text"] for n in card["notes"][-2:]))
    return "\n".join(lines)


def related_cards(store, turns, index):
    """Cards to show the extractor: the most similar to this session, plus the cards touched
    in the previous session, within CARD_CONTEXT_CHARS."""
    if not store.cards:
        return []
    text = " ".join(t["user"] for t in turns)[:3000]
    ranked = index.rank_cards(store, text) if index else list(store.cards)
    last = max((c["updated_turn"] for c in store.cards.values()), default=0)
    recent = [cid for cid, c in store.cards.items() if c["updated_turn"] == last]
    chosen, size = [], 0
    for cid in recent + ranked:
        if cid in chosen:
            continue
        block = card_context(store.cards[cid])
        if size + len(block) > CARD_CONTEXT_CHARS:
            continue
        chosen.append(cid)
        size += len(block)
    return [store.cards[cid] for cid in chosen]


def parse_json(reply):
    text = reply.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def turn_of(value, first, last):
    """'#12', '12', '12-14' -> 12 (clamped to the session); None when absent."""
    match = re.search(r"\d+", str(value or ""))
    if not match:
        return None
    return min(max(int(match.group()), first), last)


def ingest(store, session, date, messages, index=None, stats=None, tag="ctxm-extract"):
    """Add one session: number its turns, extract, apply. Returns the number of records."""
    turns = store.add_turns(session, date, messages)
    if not turns:
        return 0
    first, last = turns[0]["turn"], turns[-1]["turn"]
    related = related_cards(store, turns, index)
    user = (f"Conversation date: {date}\n\nExisting cards:\n" +
            ("\n".join(card_context(c) for c in related) or "(none yet)") +
            f"\n\nConversation (turns #{first}-#{last}):\n\n{transcript(turns)}")
    reply = llm.chat([{"role": "system", "content": PROMPT}, {"role": "user", "content": user}],
                     max_tokens=12000, tag=tag, stats=stats)
    try:
        data = parse_json(reply)
    except (ValueError, json.JSONDecodeError):
        reply = llm.chat([{"role": "system", "content": PROMPT}, {"role": "user", "content": user}],
                         max_tokens=12000, tag=tag + "-retry", stats=stats)
        data = parse_json(reply)
    return apply(store, data, session, date, first, last)


def apply(store, data, session, date, first, last):
    count = 0
    for change in data.get("cards") or []:
        if not isinstance(change, dict):
            continue
        ref = str(change.get("card") or "").strip()
        title = str(change.get("title") or "").strip()
        if (not ref or ref == "new") and not title:
            continue
        card = store.card(ref if ref != "new" else title, title or ref)
        touched = False
        for v in change.get("values") or []:
            turn = turn_of(v.get("turn"), first, last)
            if turn and v.get("key") and v.get("value") not in (None, ""):
                store.set_value(card, str(v["key"]), str(v["value"]), turn, by_of(v))
                touched = True
                count += 1
        for item in change.get("items") or []:
            turn = turn_of(item.get("turn"), first, last)
            if turn and item.get("list") and item.get("name"):
                store.add_item(card, str(item["list"]), str(item["name"]), turn, by_of(item))
                touched = True
                count += 1
        for event in change.get("events") or []:
            turn = turn_of(event.get("turn"), first, last)
            if turn and event.get("what"):
                kind = event.get("kind") if event.get("kind") in EVENT_KINDS else "happened"
                store.add_event(card, str(event["what"]), str(event.get("date") or "") or None, kind, turn, by_of(event))
                touched = True
                count += 1
        for note in change.get("notes") or []:
            turns = re.findall(r"\d+", str(note.get("turns") or ""))
            if note.get("text") and turns:
                a, b = turn_of(turns[0], first, last), turn_of(turns[-1], first, last)
                card["notes"].append({"first": a, "last": b, "date": date, "text": str(note["text"])})
                touched = True
                count += 1
        for conflict in change.get("conflicts") or []:
            turn = turn_of(conflict.get("turn"), first, last)
            if turn and conflict.get("existing") and conflict.get("new"):
                card["conflicts"].append({"existing": str(conflict["existing"]), "new": str(conflict["new"]),
                                          "turn": turn, "date": date})
                touched = True
                count += 1
        if change.get("headline"):
            card["headline"] = str(change["headline"])[:600]
            touched = True
        if touched:
            card["updated_turn"] = last
        elif not card["headline"] and not any(card[k] for k in ("values", "items", "events", "notes")):
            store.cards.pop(card["id"], None)
    for layer in ("instructions", "preferences"):
        for entry in data.get(layer) or []:
            text = entry.get("text") if isinstance(entry, dict) else entry
            turn = turn_of(entry.get("turn") if isinstance(entry, dict) else None, first, last) or first
            if text:
                store.add_profile(layer[:-1], str(text), turn)
                count += 1
    store.sessions.append({"session": session, "date": date, "first": first, "last": last,
                           "summary": str(data.get("session_summary") or "")})
    return count


def by_of(record):
    by = str(record.get("by") or "user").lower()
    return by if by in BY else "user"
