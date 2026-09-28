# ctx memory plugin for OpenClaw

Makes ctx OpenClaw's active memory plugin (`plugins.slots.memory`).

- **Before every model turn** (`before_prompt_build`) the few memories that confidently
  match the latest user message are prepended as a `<ctx-memory>` block; rules you
  approved in ctx (`type: rule` or pinned) go into the system prompt.
- **After every run** (`agent_end`) the run's conversation (the last user message and
  the replies after it) is recorded in ctx. When the session goes idle or ends
  (`session_end`) ctx distils memories and keeps the raw conversation as searchable
  excerpts.
- **Tools:** `memory_search` (memories + conversation excerpts, grouped by topic and
  time) and `memory_get` — the same names as the built-in memory, so Active Memory and
  existing prompts keep working — plus `memory_store` and `memory_forget`.
- Memory scope is the workspace directory name unless `project` is set.

Use this plugin **or** `ctx connect openclaw` (the proxy), not both. The workspace
files (`MEMORY.md`, `memory/*.md`) are still loaded by OpenClaw as usual.

## Install

```bash
openclaw plugins install --link /path/to/ctx/integrations/openclaw/ctx-memory
```

`~/.openclaw/openclaw.json`:

```json5
{
  plugins: {
    slots: { memory: "ctx-memory" },
    entries: {
      "ctx-memory": {
        enabled: true,
        // agent_end is a conversation hook; installed plugins must be allowed to read it.
        hooks: { allowConversationAccess: true },
        config: {
          // project: "my-repo",  // default: workspace directory name
          // inject: 3,           // memories injected per turn, 0 disables
          // autoCapture: true,   // record conversations in ctx
        },
      },
    },
  },
}
```

Then `openclaw gateway restart`. The plugin finds the ctx daemon and its token through
`$CTX_HOME` (default `~/.ctx`); `CTX_URL` and `CTX_TOKEN` override them.

Tested with OpenClaw 2026.6.11 (`tests/openclaw_plugin_live.py`).
