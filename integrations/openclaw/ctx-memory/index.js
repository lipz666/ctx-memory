// ctx long-term memory as an OpenClaw memory plugin (plugins.slots.memory = "ctx-memory").
//
// - before_prompt_build: recall the confident few memories for the latest user message
//   and prepend them; user-approved ctx rules go into the system prompt.
// - agent_end: send this run's conversation (the last user message and the replies after
//   it) to ctx, which distils memories when the session goes idle or ends and keeps the
//   raw conversation as searchable excerpts.
// - session_end: ask ctx to distil the finished session now.
// - Tools: memory_search / memory_get (same names as the built-in memory, so Active
//   Memory and existing prompts keep working), memory_store, memory_forget.
//
// Use this plugin or `ctx connect openclaw` (the proxy), not both.
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { basename, join } from "node:path";

const TIMEOUT_MS = 5000;

function ctxHome() {
  return process.env.CTX_HOME || join(homedir(), ".ctx");
}

function defaultUrl() {
  try {
    const port = readFileSync(join(ctxHome(), "config.yaml"), "utf8").match(/^port:\s*(\d+)/m);
    if (port) return `http://127.0.0.1:${port[1]}`;
  } catch {}
  return "http://127.0.0.1:7788";
}

function token() {
  if (process.env.CTX_TOKEN) return process.env.CTX_TOKEN;
  try {
    return readFileSync(join(ctxHome(), "token"), "utf8").trim();
  } catch {
    return undefined;
  }
}

// Plain text of a message's content: a string, or text parts of a content array.
function textOf(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((part) => part && typeof part === "object" && ["text", "input_text", "output_text"].includes(part.type))
      .map((part) => part.text || "")
      .join("\n");
  }
  return "";
}

// The messages of the run that just ended: the last user message and what followed.
function lastRun(messages) {
  const list = Array.isArray(messages) ? messages : [];
  let start = -1;
  for (let i = list.length - 1; i >= 0; i--) {
    if (list[i] && list[i].role === "user") {
      start = i;
      break;
    }
  }
  if (start < 0) return [];
  return list
    .slice(start)
    .filter((m) => m && (m.role === "user" || m.role === "assistant"))
    .map((m) => ({ role: m.role, content: textOf(m.content).trim() }))
    .filter((m) => m.content);
}

function latestUserText(event) {
  const run = lastRun(event && event.messages);
  const user = run.find((m) => m.role === "user");
  return (user && user.content) || (event && typeof event.prompt === "string" ? event.prompt : "");
}

function toolResult(value) {
  return { content: [{ type: "text", text: typeof value === "string" ? value : JSON.stringify(value) }] };
}

export default {
  id: "ctx-memory",
  name: "ctx memory",
  description: "Long-term memory from the local ctx daemon",
  kind: "memory",
  register(api) {
    const config = api.pluginConfig || {};
    const log = api.logger || console;
    const url = String(config.url || process.env.CTX_URL || defaultUrl()).replace(/\/$/, "");
    const inject = Number.isInteger(config.inject) ? config.inject : 3;
    const autoCapture = config.autoCapture !== false;

    const projectOf = (ctx) => config.project || (ctx && ctx.workspaceDir ? basename(ctx.workspaceDir) : "openclaw");
    const sessionKey = (ctx) => `openclaw:${(ctx && (ctx.sessionId || ctx.sessionKey)) || "default"}`;

    async function request(method, path, body, timeoutMs = TIMEOUT_MS) {
      const headers = { "Content-Type": "application/json" };
      const auth = token();
      if (auth) headers["X-Ctx-Token"] = auth;
      const response = await fetch(url + path, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (!response.ok) throw new Error(`ctx ${method} ${path}: ${response.status} ${(await response.text()).slice(0, 200)}`);
      return response.json();
    }

    function recall(query, project, limit, mode) {
      const params = new URLSearchParams({ q: query.slice(0, 4000), project, limit: String(limit), mode });
      return request("GET", `/api/v1/recall?${params}`);
    }

    // Rules change rarely; fetch them at most once a minute.
    let rules = { at: 0, project: "", text: "" };
    async function rulesFor(project) {
      if (rules.project === project && Date.now() - rules.at < 60_000) return rules.text;
      let text = "";
      try {
        const memories = await request("GET", "/api/v1/memories");
        const lines = memories
          .filter((m) => ["active", "contested"].includes(m.status) && ["global", project].includes(m.scope))
          .filter((m) => m.type === "rule" || m.pinned)
          .map((m) => `- ${String(m.body || "").trim()}`);
        if (lines.length) text = `Standing rules from the user (ctx memory):\n${lines.join("\n")}`;
      } catch (error) {
        log.debug?.(`ctx rules unavailable: ${error}`);
      }
      rules = { at: Date.now(), project, text };
      return text;
    }

    api.on("before_prompt_build", async (event, ctx) => {
      const project = projectOf(ctx);
      const result = {};
      const standing = await rulesFor(project);
      if (standing) result.prependSystemContext = standing;
      const query = latestUserText(event);
      if (inject > 0 && query.trim().length >= 4) {
        try {
          const hits = await recall(query, project, inject, "inject");
          if (hits.length) {
            const lines = hits.map((h) => {
              const date = String(h.observed_at || h.created_at || "").slice(0, 10);
              return `- ${date ? `[${date}] ` : ""}${String(h.content || "").trim()}`;
            });
            result.prependContext = `<ctx-memory>\nRelevant long-term memory from earlier sessions:\n${lines.join("\n")}\n</ctx-memory>`;
          }
        } catch (error) {
          log.debug?.(`ctx recall failed: ${error}`);
        }
      }
      return Object.keys(result).length ? result : undefined;
    });

    api.on("agent_end", async (event, ctx) => {
      if (!autoCapture) return;
      const messages = lastRun(event && event.messages);
      if (!messages.length) return;
      try {
        await request(
          "POST",
          "/api/v1/sessions/ingest",
          { session: sessionKey(ctx), agent: "openclaw", project: projectOf(ctx), messages },
          20_000,
        );
      } catch (error) {
        log.warn?.(`ctx capture failed: ${error}`);
      }
    });

    api.on("session_end", async (event, ctx) => {
      if (!autoCapture) return;
      const key = `openclaw:${(event && event.sessionId) || (ctx && ctx.sessionId) || "default"}`;
      try {
        await request("POST", `/api/v1/sessions/${encodeURIComponent(key)}/extract`, {}, 10_000);
      } catch (error) {
        log.debug?.(`ctx extract request failed: ${error}`);
      }
    });

    api.registerTool((toolCtx) => [
      {
        name: "memory_search",
        label: "Memory search",
        description:
          "Search long-term memory from earlier sessions: project facts, conventions, past lessons, user preferences, " +
          "and excerpts of the original conversations (exact names, numbers, commands). Results about one topic are " +
          "grouped in chronological order; prefer the latest value when they conflict.",
        parameters: {
          type: "object",
          properties: {
            query: { type: "string", description: "What you need to know, in natural language" },
            maxResults: { type: "integer", minimum: 1, maximum: 30, description: "Memories to return (default 10)" },
          },
          required: ["query"],
        },
        async execute(_id, params) {
          try {
            const hits = await recall(String(params.query || ""), projectOf(toolCtx), params.maxResults || 10, "search");
            return toolResult({
              results: hits.map((h) => ({
                id: h.id,
                type: h.type,
                date: String(h.observed_at || h.created_at || "").slice(0, 19),
                group: h.group,
                content: h.content,
              })),
            });
          } catch (error) {
            return toolResult({ error: `ctx unavailable: ${error}` });
          }
        },
      },
      {
        name: "memory_get",
        label: "Memory get",
        description: "Read one memory in full by id (from memory_search).",
        parameters: { type: "object", properties: { id: { type: "string" } }, required: ["id"] },
        async execute(_id, params) {
          try {
            const memory = await request("GET", `/api/v1/memories/${encodeURIComponent(String(params.id || ""))}`);
            return toolResult({ id: memory.id, type: memory.type, scope: memory.scope, status: memory.status, content: memory.body });
          } catch (error) {
            return toolResult({ error: String(error) });
          }
        },
      },
      {
        name: "memory_store",
        label: "Memory store",
        description:
          "Save one durable fact, convention, lesson or procedure to long-term memory as a self-contained statement " +
          "with specifics. Most memories are extracted automatically after the session; use this when the user asks " +
          "you to remember something.",
        parameters: {
          type: "object",
          properties: {
            content: { type: "string", description: "The statement to remember" },
            type: { type: "string", enum: ["fact", "lesson", "skill"], description: "Default fact" },
            scope: { type: "string", enum: ["project", "global"], description: "Default project" },
          },
          required: ["content"],
        },
        async execute(_id, params) {
          try {
            const scope = params.scope === "global" ? "global" : projectOf(toolCtx);
            const memory = await request("POST", "/api/v1/memories", {
              content: String(params.content || ""),
              type: params.type || "fact",
              scope,
            });
            return toolResult({ id: memory.id, status: memory.status });
          } catch (error) {
            return toolResult({ error: String(error) });
          }
        },
      },
      {
        name: "memory_forget",
        label: "Memory forget",
        description: "Archive a memory that is wrong or no longer true (by id from memory_search).",
        parameters: { type: "object", properties: { id: { type: "string" } }, required: ["id"] },
        async execute(_id, params) {
          try {
            return toolResult(await request("DELETE", `/api/v1/memories/${encodeURIComponent(String(params.id || ""))}`));
          } catch (error) {
            return toolResult({ error: String(error) });
          }
        },
      },
    ]);
  },
};
