# ctx memory provider for Hermes

Gives Hermes long-term memory from a local ctx daemon (`ctx serve`, or the launchd /
systemd service `ctx init` installs).

- **Every turn** is sent to ctx; when the session goes idle or ends, ctx distils atomic
  memories (facts with names, numbers and dates) and keeps the raw conversation as
  searchable excerpts.
- **Before every turn** the few memories that confidently match the user's message are
  injected (`inject: 3`).
- **Tools:** `ctx_recall` (memories + conversation excerpts, grouped by topic and time),
  `ctx_remember`, `ctx_forget`.
- **Rules** you approved in ctx (`type: rule`, or pinned memories) go into the system
  prompt.
- Writes happen only for primary sessions (not cron, flush or subagents); Hermes'
  built-in memory writes are mirrored into ctx.

Use this provider **or** `ctx connect hermes` (the proxy), not both.

## Install

```bash
mkdir -p ~/.hermes/plugins
cp -R integrations/hermes/ctx ~/.hermes/plugins/ctx     # or the profile's HERMES_HOME
```

Then in `~/.hermes/config.yaml` (or the profile's config):

```yaml
memory:
  provider: ctx
  ctx:
    project: hermes-default   # optional; memory scope, default hermes-<profile>
    inject: 3                 # optional; memories injected per turn, 0 disables
```

The provider finds the daemon and its token through `$CTX_HOME` (default `~/.ctx`);
`CTX_URL` and `CTX_TOKEN` override them.
