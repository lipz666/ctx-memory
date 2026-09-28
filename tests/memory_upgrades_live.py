"""Small live checks of the memory upgrades against an isolated ctx and the real model.

  CTX_GW_KEY=... python3 tests/memory_upgrades_live.py [step ...]   # steps: 1 2 3 4 5

Each step ingests two or three tiny synthetic sessions (a few model calls) and checks
one behaviour: 1 event dates and superseding, 2 topic digests, 3 automatic triggers,
4 deep (planned) search, 5 entity recall.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("CTX_BIN", ROOT / "target/release/ctx"))


class Ctx:
    def __init__(self, general=True):
        self.work = Path(tempfile.mkdtemp(prefix="ctx-upgrade-"))
        self.home = self.work / "ctx"
        self.env = dict(os.environ, CTX_HOME=str(self.home))
        subprocess.run([BIN, "init"], env=self.env, check=True, capture_output=True)
        subprocess.run([BIN, "model", "set", os.environ.get("CTX_TEST_BASE_URL", "https://vps.lpzproxy.xyz/v1"),
                        os.environ.get("CTX_TEST_MODEL", "gemini-3.8-flash-high"), "--credential-ref", "env:CTX_GW_KEY",
                        "--upstream-user-agent", "curl/8.0"], env=self.env, check=True, capture_output=True)
        self.port = 20000 + (os.getpid() + int(time.time())) % 20000
        config = (self.home / "config.yaml").read_text().replace("history: true", "history: false")
        (self.home / "config.yaml").write_text(re.sub(r"^port: \d+", f"port: {self.port}", config, flags=re.M))
        self.process = subprocess.Popen([BIN, "serve"], env=self.env, stdout=subprocess.DEVNULL,
                                        stderr=open(self.work / "ctx.err", "w"))
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/v1/health", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        self.token = (self.home / "token").read_text().strip()
        self.agent = "hermes" if general else "claude-code"

    def api(self, path, data=None, method=None):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                     data=json.dumps(data).encode() if data is not None else None,
                                     headers={"X-Ctx-Token": self.token, "Content-Type": "application/json"},
                                     method=method or ("POST" if data is not None else "GET"))
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.load(response)

    def session(self, key, date, turns, project="home"):
        messages = []
        for user, assistant in turns:
            messages += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        self.api("/api/v1/sessions/ingest", {"session": key, "agent": self.agent, "project": project,
                                             "observed_at": date, "messages": messages})
        return self.api(f"/api/v1/sessions/{urllib.parse.quote(key, safe='')}/extract?wait=true", {})

    def recall(self, query, project="home", **params):
        args = {"q": query, "project": project, "limit": 10, **params}
        return self.api("/api/v1/recall?" + urllib.parse.urlencode(args))

    def memories(self):
        return self.api("/api/v1/memories")

    def close(self):
        self.process.terminate()
        self.process.wait(timeout=30)
        shutil.rmtree(self.work, ignore_errors=True)


def show(memories):
    for m in memories:
        print(f"    [{m['status']}] event={m.get('event_at')} topics={m.get('topics')} entities={m.get('entities')} :: {m['body'][:110]}")


def step1(ctx):
    """Event dates are absolute; a changed fact supersedes the old one, which stays as history."""
    ctx.session("s1", "2023/05/20 (Sat) 10:00", [
        ("I adopted two kittens today, Miso and Tofu! With my old cat Bean that makes 3 cats at home.",
         "Congratulations on Miso and Tofu!")])
    ctx.session("s2", "2023/07/02 (Sun) 18:30", [
        ("Big news: last Saturday I took in a stray, Pickle, so now I have 4 cats. Any tips on introducing them?",
         "Introduce Pickle slowly, with separate rooms first.")])
    memories = ctx.memories()
    show(memories)
    superseded = [m for m in memories if m["status"] == "superseded"]
    current = [m for m in memories if m["status"] == "active" and m.get("supersedes")]
    assert superseded and current, "the cat count was not superseded"
    stray = [m for m in memories if "Pickle" in m["body"] and m.get("event_at")]
    assert any(m["event_at"].startswith("2023-07-01") or m["event_at"].startswith("2023-07") for m in stray), stray
    hits = ctx.recall("How many cats do I have now?")
    top = hits[0]
    print("   top:", top["content"][:100], "| history:", [h["content"][:60] for h in top.get("history", [])])
    assert "4" in top["content"] and all(h["status"] != "superseded" for h in memories if h["id"] == top["id"])
    assert any("3" in h["content"] for hit in hits for h in hit.get("history", [])), "history not returned"


def step2(ctx):
    """Memories of one topic are consolidated into a digest that answers whole-topic questions."""
    ctx.session("w1", "2022/11/20 (Sun) 09:00", [
        ("Just got back from a two-day writing workshop at the literary festival, it was $200 but worth it.",
         "Sounds inspiring! What did you work on?")])
    ctx.session("w2", "2022/12/13 (Tue) 20:00", [
        ("Yesterday I did a half-day mindfulness workshop at the yoga studio near me, only $20. Can you suggest a breathing exercise?",
         "Try box breathing: in 4, hold 4, out 4, hold 4.")])
    ctx.session("w3", "2023/02/26 (Sun) 11:00", [
        ("I'm trying to improve my marketing. I went to a digital marketing workshop last weekend, $500 for two days. Where should I start?",
         "Start with keyword research and on-page SEO.")])
    memories = ctx.memories()
    show(memories)
    digests = [m for m in memories if m["type"] == "digest"]
    assert digests, "no digest was built"
    hits = ctx.recall("How much money did I spend on workshops in total?")
    print("   hits:", [(h["type"], h["content"][:70]) for h in hits[:4]])
    digest_hits = [h for h in hits[:3] if h["type"] == "digest"]
    assert digest_hits, "the digest is not among the top 3 hits"
    assert all(cost in digest_hits[0]["content"] for cost in ("$200", "$20", "$500")), digest_hits[0]["content"]


def step3(ctx):
    """A coding lesson gets grounded triggers and comes back when the agent touches the file or hits the error."""
    ctx.session("c1", "2026/09/20 (Sun) 15:00", [
        ("pytest fails with `psycopg.OperationalError: connection refused (port 5433)` in the ledger repo, why?",
         "The test database is not running. Run `make test-db` first; it starts Postgres on port 5433 and applies "
         "db/migrations/*.sql."),
        ("That worked. Also note: every new migration in db/migrations must be registered in db/manifest.toml, "
         "otherwise `make test-db` silently skips it.",
         "Noted: add each new db/migrations/*.sql file to db/manifest.toml.")], project="ledger")
    memories = [m for m in ctx.memories() if m["status"] == "active"]
    for m in memories:
        print(f"    triggers={[(t['kind'], t['pattern']) for t in m.get('triggers', [])]} :: {m['body'][:100]}")
    with_triggers = [m for m in memories if m.get("triggers")]
    assert with_triggers, "no triggers were extracted"
    trace = ctx.work / "trace.jsonl"
    rows = [
        {"project": "ledger", "features": {"tool": "write", "files": ["/work/ledger/db/migrations/0007_add_fx.sql"]}},
        {"project": "ledger", "features": {"tool": "exec", "error_sig": "psycopg.OperationalError: connection refused (port 5433)"}},
        {"project": "ledger", "features": {"tool": "write", "files": ["/work/ledger/src/report.py"]}},
    ]
    trace.write_text("\n".join(json.dumps(r) for r in rows))
    result = subprocess.run([BIN, "eval", "replay", str(trace)], env=ctx.env, capture_output=True, text=True)
    steps = [row for row in map(json.loads, result.stdout.splitlines()) if "step" in row]
    if not steps:
        print("   replay output:", result.returncode, result.stdout[:500], result.stderr[:500])
    print("   injected per step:", [len(s["memory_ids"]) for s in steps])
    assert steps[0]["memory_ids"], "touching a migration file did not bring the lesson back"
    assert steps[1]["memory_ids"], "the error did not bring the lesson back"
    assert not steps[2]["memory_ids"], "an unrelated file triggered a memory"


def step4(ctx):
    """Deep search plans sub-queries and a date window: time-bounded and two-part questions."""
    ctx.session("m1", "2023/02/12 (Sun) 10:00", [("Last night I saw a jazz quartet at the Blue Room, amazing sax solo.", "Sounds great!")])
    ctx.session("m2", "2023/03/13 (Mon) 09:00", [("Went to a rock concert with Sam on Saturday, my ears are still ringing.", "Glad you had fun!")])
    ctx.session("m3", "2023/04/03 (Mon) 09:00", [("Yesterday's folk concert in the park was so relaxing.", "Lovely!")])
    ctx.session("m4", "2023/03/20 (Mon) 18:00", [("I finally bought the Nikon Z6 camera today! What settings for portraits?", "Try aperture priority at f/2.")])
    ctx.session("m5", "2023/04/10 (Mon) 18:00", [("Just picked up a Canon 50mm lens for my old camera. Is it good for street photos?", "Yes, it's a classic.")])
    ask = lambda q, limit, deep: ctx.recall(q, limit=limit, episodes=0, deep="true" if deep else "false", now="2023-05-01")
    q1 = "Which concert did I go to in March?"
    plain, deep = ask(q1, 1, False), ask(q1, 1, True)
    print("   q1 plain:", [h["content"][:60] for h in plain], "| deep:", [h["content"][:60] for h in deep])
    assert "rock" in deep[0]["content"].lower(), deep
    q2 = "Did I buy the Nikon camera or the Canon lens first?"
    plain, deep = ask(q2, 2, False), ask(q2, 2, True)
    print("   q2 plain:", [h["content"][:60] for h in plain], "| deep:", [h["content"][:60] for h in deep])
    text = " ".join(h["content"] for h in deep)
    assert "Nikon" in text and "Canon" in text, deep


def step5(ctx):
    """Entity index: a question naming someone returns every memory about them."""
    ctx.session("e1", "2023/05/01 (Mon) 10:00", [("My sister Mira is moving to Lisbon next month, I'm helping her find a flat.", "How exciting!")])
    ctx.session("e2", "2023/06/15 (Thu) 10:00", [("Mira started a new job at a design studio. She loves it.", "That's wonderful news!")])
    ctx.session("e3", "2023/07/02 (Sun) 10:00", [("My sister's dog Biscuit has been sick all week, she's really worried. Any advice?", "Take Biscuit to a vet if it lasts.")])
    ctx.session("e4", "2023/07/05 (Wed) 10:00", [("I started learning Spanish with an app, 15 minutes a day.", "Great habit!")])
    memories = [m for m in ctx.memories() if m["status"] == "active" and m["type"] != "digest"]
    show(memories)
    about = {m["id"] for m in memories if any(e.lower() == "mira" for e in m.get("entities") or [])}
    hits = ctx.recall("Tell me everything you know about Mira.", limit=10, episodes=0)
    found = {h["id"] for h in hits}
    print(f"   memories tagged Mira: {len(about)}, found: {len(about & found)}; spanish returned: {any('Spanish' in h['content'] for h in hits)}")
    assert about and about <= found, (about - found)


STEPS = {"1": step1, "2": step2, "3": step3, "4": step4, "5": step5}


def main():
    steps = sys.argv[1:] or list(STEPS)
    for step in steps:
        ctx = Ctx(general=step != "3")
        try:
            print(f"== step {step}: {STEPS[step].__doc__.strip()}")
            STEPS[step](ctx)
            print(f"   step {step} passed")
        finally:
            ctx.close()


if __name__ == "__main__":
    main()
