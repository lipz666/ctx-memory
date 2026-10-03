"""ctx-m memory of one namespace: the turn log, cards, the standing layer and session
summaries (docs/ctx-m-design.md, section 3). Plain dicts, saved as one JSON file."""
import json
import re
from pathlib import Path

BY = ("user", "assistant", "adopted")
EVENT_KINDS = ("happened", "planned", "deadline", "started", "finished", "scheduled", "reported")


def norm(text):
    """Key for deduplication: lowercase words only."""
    return " ".join(re.findall(r"[\w.+#-]+", str(text).lower()))


def slug(title):
    return "c_" + ("_".join(re.findall(r"[a-z0-9]+", title.lower()))[:40] or "card")


class Store:
    def __init__(self):
        self.turns = []        # {"turn", "session", "date", "user", "assistant"}
        self.cards = {}        # id -> card
        self.instructions = []  # {"text", "turn", "date"}
        self.preferences = []
        self.sessions = []     # {"session", "date", "first", "last", "summary"}

    # ---- persistence
    def save(self, path):
        Path(path).write_text(json.dumps(self.__dict__, ensure_ascii=False))

    @classmethod
    def load(cls, path):
        store = cls()
        store.__dict__.update(json.loads(Path(path).read_text()))
        return store

    # ---- turns
    def add_turns(self, session, date, messages):
        """Number the session's turns (one user message plus the replies that follow).
        Returns the new turns."""
        added = []
        for message in messages:
            if message["role"] == "user" or not added:
                added.append({"turn": len(self.turns) + len(added) + 1, "session": session, "date": date,
                              "user": message["content"] if message["role"] == "user" else "", "assistant": ""})
                if message["role"] == "user":
                    continue
            added[-1]["assistant"] += ("\n" if added[-1]["assistant"] else "") + message["content"]
        self.turns += added
        return added

    def date_of(self, turn):
        if 1 <= turn <= len(self.turns):
            return self.turns[turn - 1]["date"]
        return None

    # ---- cards
    def card(self, ref, title):
        """The card `ref` names (an id, or a title equal to an existing title), else a new one."""
        if ref in self.cards:
            return self.cards[ref]
        wanted = norm(title or ref)
        for card in self.cards.values():
            if norm(card["title"]) == wanted:
                return card
        base = slug(title or ref)
        card_id, n = base, 2
        while card_id in self.cards:
            card_id, n = f"{base}_{n}", n + 1
        card = {"id": card_id, "title": (title or ref).strip()[:80], "headline": "", "values": [], "items": [],
                "events": [], "notes": [], "conflicts": [], "updated_turn": 0}
        self.cards[card_id] = card
        return card

    def set_value(self, card, key, value, turn, by):
        """Same value again: one more mention. A different value: the current one becomes
        history (status superseded); history is ordered by turn."""
        current = [v for v in card["values"] if norm(v["key"]) == norm(key) and v["status"] == "current"]
        for v in current:
            if norm(v["value"]) == norm(value):
                if turn not in v["mentions"]:
                    v["mentions"].append(turn)
                return
        for v in current:
            if v["turn"] <= turn:
                v["status"] = "superseded"
        status = "current" if all(v["turn"] <= turn for v in current) else "superseded"
        card["values"].append({"key": key, "value": value, "turn": turn, "date": self.date_of(turn), "by": by,
                               "status": status, "mentions": [turn]})

    def add_item(self, card, list_name, name, turn, by):
        for item in card["items"]:
            if norm(item["list"]) == norm(list_name) and norm(item["name"]) == norm(name):
                if turn not in item["mentions"]:
                    item["mentions"].append(turn)
                # A suggestion the user later states or adopts is the user's.
                if item["by"] == "assistant" and by in ("user", "adopted"):
                    item["by"] = "adopted" if by == "adopted" else "user"
                return
        card["items"].append({"list": list_name, "name": name, "turn": turn, "date": self.date_of(turn), "by": by,
                              "mentions": [turn]})

    def add_event(self, card, what, date, kind, turn, by):
        for event in card["events"]:
            if norm(event["what"]) == norm(what) and event["date"] == date:
                return
        card["events"].append({"what": what, "date": date, "kind": kind, "turn": turn, "said": self.date_of(turn),
                               "by": by})

    def add_profile(self, layer, text, turn):
        entries = self.instructions if layer == "instruction" else self.preferences
        if any(norm(e["text"]) == norm(text) for e in entries):
            return
        entries.append({"text": text, "turn": turn, "date": self.date_of(turn)})
