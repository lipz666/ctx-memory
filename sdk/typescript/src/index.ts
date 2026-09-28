export type MemoryType = "fact" | "lesson" | "skill" | "rule";
/** A recall hit. `channel` is search, trigger or rule. */
export type RecallHit = {
  id: string; title: string; content: string; type: string; scope: string; score: number;
  /** search, trigger, rule, or episode (a raw conversation excerpt) */
  channel: string; observed_at: string | null; created_at: string;
  /** search mode: hits about one topic share a group, in chronological order */
  group: number | null;
};
export type Session = { key: string; agent: string; project: string | null; steps: number; user_turns: number; status: string; last_seen: string; extracted_at: string | null; error: string | null };

export class Client {
  constructor(
    public readonly agentId: string,
    private readonly token: string,
    public readonly project?: string,
    private readonly baseUrl = "http://127.0.0.1:7788",
  ) {}

  private async request<T>(method: string, path: string, data?: unknown): Promise<T> {
    const response = await fetch(this.baseUrl + path, {
      method,
      headers: { "X-Ctx-Token": this.token, "Content-Type": "application/json" },
      body: data === undefined ? undefined : JSON.stringify(data),
    });
    if (!response.ok) throw new Error(`ctx API ${response.status}: ${await response.text()}`);
    return response.json() as Promise<T>;
  }

  event(type: string, data: Record<string, unknown>, sessionId?: string) {
    return this.request<{ id: string }>("POST", "/api/v1/events", {
      agent_id: this.agentId, project: this.project, session_id: sessionId, type, data,
    });
  }

  taskStart(taskId: string, description: string, sessionId?: string) {
    return this.event("task_start", { task_id: taskId, description }, sessionId);
  }

  /** With a session id, the session's memories are extracted right away. */
  taskEnd(taskId: string, result: "success" | "failure" | "unknown", sessionId?: string) {
    return this.request<{ id: string; extraction_queued: boolean }>("POST", "/api/v1/events", {
      agent_id: this.agentId, project: this.project, session_id: sessionId, type: "task_end",
      data: { task_id: taskId, result },
    });
  }

  /** Saved as agent-sourced; a rule waits for user review before it takes effect. */
  remember(content: string, type: MemoryType = "fact", scope = this.project ?? "global") {
    return this.request<{ id: string; status: string }>("POST", "/api/v1/memories", { content, type, scope });
  }

  /** mode "search": up to `limit` memories by relevance plus conversation excerpts; "inject": the confident few. */
  recall(query: string, limit = 5, mode: "search" | "inject" = "search", episodes?: number) {
    const params = new URLSearchParams({ q: query, limit: String(limit), mode });
    if (episodes !== undefined) params.set("episodes", String(episodes));
    if (this.project) params.set("project", this.project);
    return this.request<RecallHit[]>("GET", "/api/v1/recall?" + params);
  }

  sessions() {
    return this.request<Session[]>("GET", "/api/v1/sessions");
  }

  extractSession(key: string) {
    return this.request<{ queued: boolean }>("POST", `/api/v1/sessions/${encodeURIComponent(key)}/extract`);
  }

  stats() {
    return this.request<Record<string, unknown>>("GET", "/api/v1/stats");
  }

  expand(id: string) {
    const path = id.startsWith("mem_") ? "/api/v1/memories/" : id.startsWith("evt_") ? "/api/v1/events/" : null;
    if (!path) throw new Error("Expected a memory or event id");
    return this.request<unknown>("GET", path + encodeURIComponent(id));
  }

  forget(id: string) {
    return this.request<{ archived: boolean }>("DELETE", "/api/v1/memories/" + encodeURIComponent(id));
  }

  feedback(stepEvent: string, memoryId: string, outcome: { cited?: boolean; actionConsistent?: boolean; taskResult?: "success" | "failure" | "unknown" }) {
    return this.request<{ recorded: boolean }>("POST", "/api/v1/feedback", {
      step_event: stepEvent, memory_id: memoryId,
      cited: outcome.cited, action_consistent: outcome.actionConsistent, task_result: outcome.taskResult,
    });
  }

  missedRecall(eventId: string, memoryId: string, reason: string) {
    return this.request<{ id: string }>("POST", "/api/v1/feedback/miss", {
      event_id: eventId, memory_id: memoryId, reason,
    });
  }

  debugStep(stepEvent: string) {
    return this.request<unknown>("GET", "/api/v1/debug/steps/" + encodeURIComponent(stepEvent));
  }
}
