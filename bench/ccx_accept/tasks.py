"""Sealed acceptance tasks for ccx (docs/ccx-accept.md).

The agent sees only a virtual workspace through five tools implemented here: no real
filesystem, shell or network. Expected answers live only in this process, so the agent
cannot read them. Every task is generated from a fixed seed.

Each task tests one ccx capability:
  log_needle         M1 hot tier: the answer is one error line deep inside a huge output
  table_lookup       M1 hot tier + ccx_expand: the answer is an ordinary row deep inside
  early_detail       M1 warm tier: the answer is in an output read many steps earlier
  rules_memory       M2: a rule the user set in turn 1 must hold in every later turn
  topic_switch       M2: facts from turn 1 are asked again after an unrelated task
  sum_many           growth: 20 small outputs whose values must all be kept
"""
import json
import random
import re
from dataclasses import dataclass, field
from typing import Callable

TOOLS = [
    {"type": "function", "function": {
        "name": "list_files", "description": "List files under a directory of the workspace.",
        "parameters": {"type": "object", "properties": {
            "dir": {"type": "string", "description": "Directory, e.g. `config`. Empty for all files."}}}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Return the full text of a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "search_files", "description": "Return lines (path:line: text) containing the text.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                       "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "run", "description": "Run a shell command in the workspace and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "query", "description": "Return rows of a database table as CSV, optionally filtered by `where` (column=value).",
        "parameters": {"type": "object", "properties": {
            "table": {"type": "string"}, "where": {"type": "string"}}, "required": ["table"]}}},
]

SYSTEM = ("You are an agent working in a workspace. Use the tools to do what the user asks. "
          "When you have the answer, reply to the user without calling a tool.")


@dataclass
class Task:
    name: str
    tests: str
    turns: list[str]
    files: dict[str, str] = field(default_factory=dict)
    commands: dict[str, str] = field(default_factory=dict)
    tables: dict[str, list[dict]] = field(default_factory=dict)
    # Checks on the replies: turn index -> (description, function of the reply text).
    checks: dict[int, tuple[str, Callable[[str], bool]]] = field(default_factory=dict)
    # A plausible tool sequence per turn and replies, used only for the offline size estimate.
    oracle: list[tuple[list[tuple[str, dict]], str]] = field(default_factory=list)

    # --- tools -----------------------------------------------------------------------
    def call(self, name: str, args: dict) -> str:
        try:
            if name == "list_files":
                prefix = (args.get("dir") or "").strip("/")
                hits = sorted(p for p in self.files if not prefix or p.startswith(prefix + "/"))
                return "\n".join(hits) if hits else f"no files under {prefix!r}"
            if name == "read_file":
                path = str(args.get("path", "")).strip().lstrip("./")
                return self.files.get(path, f"error: no such file: {path}")
            if name == "search_files":
                text = str(args.get("text", ""))
                if not text:
                    return "error: empty search"
                hits = [f"{p}:{i + 1}: {line}" for p, body in sorted(self.files.items())
                        for i, line in enumerate(body.splitlines()) if text in line]
                return "\n".join(hits[:200]) if hits else "no matches"
            if name == "run":
                command = " ".join(str(args.get("command", "")).split())
                return self.commands.get(command, f"sh: command not found or not allowed: {command}")
            if name == "query":
                rows = self.tables.get(args.get("table", ""))
                if rows is None:
                    return f"error: no table {args.get('table')!r}; tables: {', '.join(self.tables)}"
                where = args.get("where") or ""
                if where:
                    col, _, value = where.partition("=")
                    rows = [r for r in rows if str(r.get(col.strip())) == value.strip().strip("'\"")]
                if not rows:
                    return "0 rows"
                cols = list(rows[0])
                return "\n".join([",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows])
            return f"error: unknown tool {name}"
        except Exception as e:  # a tool must never crash the run
            return f"error: {e}"


def _words(rng: random.Random, n: int) -> str:
    vocab = ("alpha beta gamma delta service cache queue worker region shard replica "
             "index batch window commit deploy metric latency budget owner review").split()
    return " ".join(rng.choice(vocab) for _ in range(n))


def log_needle(seed: int = 1) -> Task:
    rng = random.Random(seed)
    mods = ["auth", "cart", "search", "users", "orders", "billing", "notify", "export"]
    lines, n, at = [], 1500, 830
    for i in range(n):
        if i == at:
            lines.append("FAILED tests/test_billing.py::test_refund_rounding - "
                         "AssertionError: refund total 10.04 != expected 10.05")
        else:
            m = rng.choice(mods)
            lines.append(f"tests/test_{m}.py::test_{m}_{rng.choice(['create', 'update', 'list', 'edge'])}"
                         f"_{i} PASSED [{i * 100 // n:3d}%]")
    lines.append(f"=========== 1 failed, {n - 1} passed in 41.27s ===========")
    return Task(
        name="log_needle", tests="M1 hot: one error line inside a ~22K-token output",
        turns=["Run the test suite with the command `pytest -q`. Which test failed, and what are the "
               "two numbers in its assertion message? Reply as: FAILED <test id> | <number> vs <number>"],
        commands={"pytest -q": "\n".join(lines)},
        checks={0: ("names test_refund_rounding with 10.04 and 10.05",
                    lambda r: "test_refund_rounding" in r and "10.04" in r and "10.05" in r)},
        oracle=[([("run", {"command": "pytest -q"})],
                 "FAILED tests/test_billing.py::test_refund_rounding | 10.04 vs 10.05")],
    )


def table_lookup(seed: int = 2) -> Task:
    rng = random.Random(seed)
    first = "Ana Ben Chen Dara Eli Femi Gus Hana Ivo Jae Kai Lena Milo Nia Omar Pia".split()
    last = "Ito Kim Lopez Moreau Novak Okafor Patel Quinn Rossi Sato Tan Weber".split()
    cities = "Lyon Osaka Porto Leeds Graz Turku Cork Bergen Nantes Utrecht Kobe Basel".split()
    rows = [{"id": f"C-{i:05d}", "name": f"{rng.choice(first)} {rng.choice(last)}",
             "city": rng.choice(cities), "balance": f"{rng.randint(100, 99999) / 100:.2f}"}
            for i in range(1, 1501)]
    target = rows[712]
    target.update(city="Tromso", balance="4821.37")
    return Task(
        name="table_lookup", tests="M1 hot + ccx_expand: an ordinary row deep inside a ~17K-token output",
        turns=["Query the `customers` table (all rows, no filter) and find customer C-00713. "
               "Which city are they in and what is their balance?"],
        tables={"customers": rows},
        checks={0: ("Tromso and 4821.37", lambda r: "Tromso" in r and "4821.37" in r)},
        oracle=[([("query", {"table": "customers"})], "C-00713 is in Tromso with balance 4821.37.")],
    )


def early_detail(seed: int = 3) -> Task:
    rng = random.Random(seed)
    names = ["auth", "billing", "cache", "gateway", "search", "worker"]
    files, values = {}, {}
    for name in names:
        keys = [f"{rng.choice(['pool', 'feature', 'limit', 'region', 'flag'])}_{k}" for k in range(240)]
        body = [f"# {name} service configuration", f"service: {name}"]
        for k, key in enumerate(keys):
            if k == 120:
                values[(name, "timeout_ms")] = rng.randint(1000, 9999)
                body.append(f"timeout_ms: {values[(name, 'timeout_ms')]}")
            elif k == 150:
                values[(name, "retries")] = rng.randint(2, 19)
                body.append(f"retries: {values[(name, 'retries')]}")
            else:
                body.append(f"{key}: {rng.randint(0, 5000)}  # {_words(rng, 3)}")
        files[f"config/{name}.yaml"] = "\n".join(body)
    t, r = values[("billing", "timeout_ms")], values[("search", "retries")]
    return Task(
        name="early_detail", tests="M1 warm: the answer is in an output read four steps earlier",
        turns=["Read every file in config/ one at a time with read_file, in alphabetical order "
               "(do not use search_files). Only after you have read all six, tell me `timeout_ms` "
               "in config/billing.yaml and `retries` in config/search.yaml."],
        files=files,
        checks={0: (f"timeout_ms {t} and retries {r}", lambda x: str(t) in x and re.search(rf"\b{r}\b", x) is not None)},
        oracle=[([("list_files", {"dir": "config"})] + [("read_file", {"path": f"config/{n}.yaml"}) for n in names],
                 f"billing timeout_ms is {t}; search retries is {r}.")],
    )


def _log(rng: random.Random, n: int, errors: int) -> tuple[str, int]:
    picks = set(rng.sample(range(n), errors))
    lines = []
    for i in range(n):
        level = "ERROR" if i in picks else rng.choice(["INFO", "INFO", "INFO", "DEBUG", "WARN"])
        lines.append(f"2026-09-30T{i // 3600 % 24:02d}:{i // 60 % 60:02d}:{i % 60:02d}Z {level} "
                     f"{rng.choice(['api', 'db', 'cache', 'jobs'])}: {_words(rng, 6)}")
    return "\n".join(lines), errors


def rules_memory(seed: int = 4) -> Task:
    rng = random.Random(seed)
    meeting = "\n".join(["# Weekly sync 2026-09-29"] + [f"- {_words(rng, 12)}" for _ in range(180)]
                        + ["Decision: freeze the v2 API until 2026-10-15.", "Owner: Mira."])
    log, errors = _log(rng, 700, 13)
    du = "\n".join(f"{rng.randint(1, 900)}M\t{d}" for d in
                   ["src", "docs", "node_modules", "build", "logs", "assets", "vendor", "tmp"])
    du += "\n38G\tbackups"
    ref = "-- ref 7731"
    turns = [
        f"Ground rules for this whole session: end every reply with the line `{ref}`, and never "
        "recommend deleting files. First task: list the files in the workspace.",
        "Read notes/meeting.md and give me the decision and its owner.",
        "Run `du -sh *` and tell me which directory is the largest.",
        "Read logs/app.log and tell me exactly how many ERROR lines it has.",
        "The disk is almost full. What should I do? Keep it short.",
    ]
    ends = lambda r: r.strip().splitlines()[-1].strip().strip("`") == ref if r.strip() else False
    return Task(
        name="rules_memory", tests="M2: a rule from turn 1 holds in every later turn",
        turns=turns,
        files={"notes/meeting.md": meeting, "logs/app.log": log, "README.md": "Ops workspace."},
        commands={"du -sh *": du},
        checks={i: (f"turn {i + 1} ends with {ref}", ends) for i in range(5)} | {
            3: (f"{errors} ERROR lines and ends with ref", lambda r: re.search(rf"\b{errors}\b", r) is not None and ends(r)),
            4: ("ends with ref; no rm -rf", lambda r: ends(r) and "rm -rf" not in r)},
        oracle=[([("list_files", {})], f"README.md, logs/app.log, notes/meeting.md\n{ref}"),
                ([("read_file", {"path": "notes/meeting.md"})], f"Freeze v2 API until 2026-10-15; owner Mira.\n{ref}"),
                ([("run", {"command": "du -sh *"})], f"backups (38G).\n{ref}"),
                ([("read_file", {"path": "logs/app.log"})], f"{errors} ERROR lines.\n{ref}"),
                ([], f"Move backups to other storage and compress logs.\n{ref}")],
    )


def topic_switch(seed: int = 5) -> Task:
    rng = random.Random(seed)
    lines = []
    for i in range(420):
        t = f"02:{i // 60 % 60:02d}:{i % 60:02d}"
        if i == 254:
            lines.append(f"2026-09-30 02:14:07 CRIT payment-gateway: worker exited (OOM, rss=3.9G)")
        elif i == 301:
            lines.append(f"2026-09-30 02:16:40 CRIT checkout-api: upstream payment-gateway unavailable, exiting")
        else:
            lines.append(f"2026-09-30 {t} {rng.choice(['INFO', 'WARN', 'DEBUG'])} "
                         f"{rng.choice(['checkout-api', 'payment-gateway', 'inventory', 'edge'])}: {_words(rng, 7)}")
    statuses = ["paid", "paid", "paid", "shipped", "shipped", "cancelled"]
    orders = [{"order": f"O-{i:04d}", "status": rng.choice(statuses), "amount": f"{rng.randint(500, 40000) / 100:.2f}"}
              for i in range(600)]
    refunded = [117, 233, 341, 412, 508, 577]
    amounts = ["19.99", "250.00", "74.50", "112.25", "9.75", "38.40"]
    for i, a in zip(refunded, amounts):
        orders[i].update(status="refunded", amount=a)
    total = "504.89"
    roadmap = "\n".join(["# Roadmap"] + [f"## {q}" + "\n" + "\n".join(f"- {_words(rng, 9)}" for _ in range(40))
                                         for q in ["Q1", "Q2", "Q3"]]
                        + ["## Q4", "- Launch EU region", "- SOC 2 audit", "- Mobile checkout v3"])
    return Task(
        name="topic_switch", tests="M2: a turn-1 finding is asked again after unrelated work",
        turns=["Our checkout broke last night. Read logs/incident.log and tell me which service crashed "
               "first and at what time (HH:MM:SS).",
               "Different topic: using the `orders` table, what is the total amount of orders with "
               "status refunded?",
               "Another one: read docs/roadmap.md and list the Q4 milestones.",
               "Back to the outage from my first question: which service crashed first and when? "
               "And remind me of the refunded total."],
        files={"logs/incident.log": "\n".join(lines), "docs/roadmap.md": roadmap},
        tables={"orders": orders},
        checks={0: ("payment-gateway 02:14:07", lambda r: "payment-gateway" in r and "02:14:07" in r),
                1: (f"total {total}", lambda r: total in r),
                3: (f"payment-gateway 02:14:07 and {total}",
                    lambda r: "payment-gateway" in r and "02:14:07" in r and total in r)},
        oracle=[([("read_file", {"path": "logs/incident.log"})], "payment-gateway at 02:14:07 (OOM)."),
                ([("query", {"table": "orders", "where": "status=refunded"})], f"Refunded total: {total}."),
                ([("read_file", {"path": "docs/roadmap.md"})], "Launch EU region; SOC 2 audit; Mobile checkout v3."),
                ([], f"payment-gateway crashed first at 02:14:07; refunded total {total}.")],
    )


def sum_many(seed: int = 6) -> Task:
    rng = random.Random(seed)
    files, total = {}, 0
    for i in range(1, 21):
        value = rng.randint(100, 999)
        total += value
        files[f"data/part_{i:02d}.txt"] = "\n".join(
            [f"part {i:02d}"] + [_words(rng, 8) for _ in range(6)] + [f"subtotal: {value}"]
            + [_words(rng, 8) for _ in range(4)])
    return Task(
        name="sum_many", tests="growth: twenty small outputs whose values must all be kept",
        turns=["List the files in data/, read each one with read_file (one call per file, do not use "
               "search_files), and tell me the sum of all `subtotal` values."],
        files=files,
        checks={0: (f"sum {total}", lambda r: str(total) in r.replace(",", ""))},
        oracle=[([("list_files", {"dir": "data"})] + [("read_file", {"path": p}) for p in sorted(files)],
                 f"The sum is {total}.")],
    )


ALL = [log_needle, table_lookup, early_detail, rules_memory, topic_switch, sum_many]


def by_name() -> dict[str, Callable[[], Task]]:
    return {f.__name__: f for f in ALL}


if __name__ == "__main__":
    for make in ALL:
        t = make()
        sizes = {k: len(v) // 4 for k, v in list(t.files.items())[:3]}
        print(t.name, "|", t.tests, "| turns", len(t.turns), "| sample sizes", sizes,
              {k: len(v) // 4 for k, v in t.commands.items()})
        print("   ", json.dumps({i: d for i, (d, _) in t.checks.items()}))
