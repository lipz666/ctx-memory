"""Common interface every memory system implements for the benchmark."""


class MemorySystem:
    name = "base"
    #: True if `search` ignores the token budget on purpose (full-context baseline).
    unbounded = False

    def setup(self):
        """Start services; called once before a track."""

    def teardown(self):
        """Stop services."""

    def ingest_session(self, ns, session_id, messages, timestamp, project=None):
        """Store one past conversation (list of {role, content}) of namespace `ns`, then run
        whatever the system does at the end of a session (extraction, consolidation)."""
        raise NotImplementedError

    def add_memory(self, ns, text, project=None):
        """Store one memory verbatim, without extraction (retrieval-only track)."""
        raise NotImplementedError

    def search(self, ns, query, project=None, limit=20):
        """Return retrieved items as strings, best first (dates included when known)."""
        raise NotImplementedError

    def search_scored(self, ns, query, project=None, limit=20):
        """(text, score) pairs; scores comparable across namespaces of the same system."""
        items = self.search(ns, query, project, limit)
        return [(item, 1.0 - i / max(len(items), 1)) for i, item in enumerate(items)]

    def release(self, ns):
        """Free in-memory state of a namespace that will not be queried again."""

    def usage(self):
        """Engine-side counters (LLM calls, memories stored...)."""
        return {}


def fit_budget(items, budget_tokens):
    """Keep items in order while they fit in the budget (4 characters per token)."""
    kept, used = [], 0
    for item in items:
        cost = len(item) // 4 + 1
        if used + cost > budget_tokens:
            break
        kept.append(item)
        used += cost
    return kept
