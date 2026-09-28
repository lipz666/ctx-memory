//! MCP stdio server. The project is `CTX_PROJECT` or the repository of the working
//! directory the Agent started the server in.
use crate::{
    memory::NewMemory,
    recall::{self, Query},
    store::{Store, redact},
};
use anyhow::Result;
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};

pub fn serve(store: &Store) -> Result<()> {
    let project = std::env::var("CTX_PROJECT")
        .ok()
        .filter(|p| !p.trim().is_empty())
        .map(|p| crate::memory::normalize_scope(&p))
        .or_else(|| {
            std::env::current_dir()
                .ok()
                .and_then(|dir| crate::session::project_name(&dir))
        });
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Value>(&line) {
            Ok(request) => handle(store, project.as_deref(), &request),
            Err(_) => Some(
                json!({"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"Parse error"}}),
            ),
        };
        if let Some(value) = response {
            serde_json::to_writer(&mut stdout, &value)?;
            stdout.write_all(b"\n")?;
            stdout.flush()?;
        }
    }
    Ok(())
}

fn handle(store: &Store, project: Option<&str>, request: &Value) -> Option<Value> {
    let id = request.get("id")?.clone();
    let method = request.get("method").and_then(Value::as_str).unwrap_or("");
    let result = match method {
        "initialize" => {
            let proposed = request
                .pointer("/params/protocolVersion")
                .and_then(Value::as_str)
                .unwrap_or("2025-11-25");
            let version = if ["2025-11-25", "2025-06-18"].contains(&proposed) {
                proposed
            } else {
                "2025-11-25"
            };
            Ok(
                json!({"protocolVersion":version,"capabilities":{"tools":{"listChanged":false}},"serverInfo":{"name":"ctx","version":env!("CARGO_PKG_VERSION")},
                "instructions":"ctx is your long-term memory across sessions. Call recall before non-trivial work or when an error looks familiar; call remember when the user states a preference or you learn a project fact or lesson worth keeping."}),
            )
        }
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({"tools":tools(&store.config.mcp_tools)})),
        "tools/call" => Ok(call_tool(
            store,
            project,
            request.get("params").unwrap_or(&Value::Null),
        )),
        _ => Err(format!("Unknown method: {method}")),
    };
    Some(match result {
        Ok(value) => json!({"jsonrpc":"2.0","id":id,"result":value}),
        Err(message) => json!({"jsonrpc":"2.0","id":id,"error":{"code":-32601,"message":message}}),
    })
}

/// Listed tools. `default` covers reading and writing memory; `full` adds flag_memory
/// and set_intent. All tools stay callable by name.
fn tools(mode: &str) -> Vec<Value> {
    let mut tools = vec![
        json!({"name":"recall","description":"Search long-term memory from earlier sessions: project facts, past lessons, user preferences. Use before non-trivial work or when stuck on an error.","inputSchema":{"type":"object","properties":{"query":{"type":"string","description":"What you need to know, in natural language"},"limit":{"type":"integer","minimum":1,"maximum":20}},"required":["query"]}}),
        json!({"name":"remember","description":"Save something future sessions should know: a user preference, a project convention or command, or a lesson from a mistake. One self-contained statement.","inputSchema":{"type":"object","properties":{"content":{"type":"string"},"type":{"type":"string","enum":["fact","lesson","skill"]},"scope":{"type":"string","enum":["project","global"],"description":"project (default): this repository only; global: everywhere, e.g. user preferences"}},"required":["content"]}}),
        json!({"name":"forget","description":"Archive a memory that is wrong or no longer true.","inputSchema":{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}}),
        json!({"name":"expand","description":"Read a memory (mem_...) or archived tool output (evt_...) by id.","inputSchema":{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}}),
    ];
    if mode == "full" {
        tools.push(json!({"name":"flag_memory","description":"Flag a wrong, outdated, or irrelevant memory.","inputSchema":{"type":"object","properties":{"id":{"type":"string"},"reason":{"type":"string","enum":["wrong","outdated","irrelevant"]}},"required":["id","reason"]}}));
        tools.push(json!({"name":"set_intent","description":"Propose a future action for user review.","inputSchema":{"type":"object","properties":{"when":{"type":"string"},"then":{"type":"string"},"expires":{"type":"string"}},"required":["when","then"]}}));
    }
    tools
}

fn arg<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("Missing {key}"))
}

fn call_tool(store: &Store, project: Option<&str>, params: &Value) -> Value {
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let args = params.get("arguments").unwrap_or(&Value::Null);
    let result: Result<Value> = (|| match name {
        "recall" => {
            let query = arg(args, "query")?;
            let limit = args
                .get("limit")
                .and_then(Value::as_u64)
                .unwrap_or(5)
                .min(20) as usize;
            store.ensure_vectors()?;
            let hits = recall::recall(
                store,
                &Query {
                    text: Some(query),
                    project,
                    limit,
                    mode: recall::Mode::Search,
                    episodes: store.config.recall.search_episodes,
                    ..Default::default()
                },
            )?;
            let step = store.event(
                "mcp",
                project,
                None,
                "mcp_call",
                &json!({"text":query}),
                &json!({"tool":"recall","query":query}),
            )?;
            for hit in &hits {
                let _ = store.record_recall(&step, &hit.memory.id, "agent", hit.score, &hit.reason);
            }
            // Hits of one topic share a group and are in chronological order; the date
            // tells which value is current.
            Ok(json!(hits.iter().map(|h| json!({"id":h.memory.id,"type":h.memory.kind,"scope":h.memory.scope,"date":h.memory.observed_at.as_deref().unwrap_or(&h.memory.created_at),"group":h.group,"content":h.memory.body,"score":(h.score*100.0).round()/100.0})).collect::<Vec<_>>()))
        }
        "remember" => {
            let content = arg(args, "content")?;
            let kind = args.get("type").and_then(Value::as_str).unwrap_or("fact");
            let scope = match args.get("scope").and_then(Value::as_str) {
                Some("global") => "global".to_owned(),
                Some("project") | None => project.unwrap_or("global").to_owned(),
                Some(other) => other.to_owned(),
            };
            let memory = store.remember(
                NewMemory {
                    content: content.into(),
                    kind: kind.into(),
                    scope,
                    title: None,
                    triggers: vec![],
                },
                "agent",
            )?;
            Ok(json!({"id":memory.id,"scope":memory.scope,"status":memory.status}))
        }
        "forget" => Ok(json!({"archived":store.archive(arg(args, "id")?)?})),
        "expand" => {
            let id = arg(args, "id")?;
            if id.starts_with("mem_") {
                let memory = store
                    .memory(id)
                    .ok_or_else(|| anyhow::anyhow!("Memory not found"))?;
                let mut value = json!(memory);
                value["body"] = memory.body.into();
                Ok(value)
            } else if id.starts_with("evt_") {
                Ok(
                    json!({"id":id,"payload":store.event_payload(id)?.ok_or_else(|| anyhow::anyhow!("Event not found"))?}),
                )
            } else {
                Err(anyhow::anyhow!("Invalid id"))
            }
        }
        "flag_memory" => {
            Ok(json!({"flagged":store.flag_memory(arg(args, "id")?, arg(args, "reason")?)?}))
        }
        "set_intent" => {
            let memory = store.create_intent(
                arg(args, "when")?,
                arg(args, "then")?,
                args.get("expires").and_then(Value::as_str),
            )?;
            Ok(json!({"id":memory.id,"status":memory.status}))
        }
        _ => Err(anyhow::anyhow!("Unknown tool")),
    })();
    match result {
        Ok(value) => json!({"content":[{"type":"text","text":redact(&value.to_string())}]}),
        Err(error) => {
            json!({"content":[{"type":"text","text":redact(&error.to_string())}],"isError":true})
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_tools_read_and_write_scoped_memory() {
        let dir = tempfile::tempdir().unwrap();
        crate::store::init(dir.path()).unwrap();
        let mut config = crate::store::load_config(dir.path()).unwrap();
        config.embedding.enabled = false;
        Store::save_config(dir.path(), &config).unwrap();
        let store = Store::open(dir.path()).unwrap();
        let call = |project: Option<&str>, id: u64, method: &str, params: Value| {
            handle(
                &store,
                project,
                &json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}),
            )
            .unwrap()
        };
        let init = call(
            None,
            1,
            "initialize",
            json!({"protocolVersion":"2025-11-25"}),
        );
        assert_eq!(init["result"]["protocolVersion"], "2025-11-25");
        let names: Vec<String> = call(None, 2, "tools/list", json!({}))["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap().to_owned())
            .collect();
        assert_eq!(names, ["recall", "remember", "forget", "expand"]);
        let saved = call(
            Some("billing"),
            3,
            "tools/call",
            json!({"name":"remember","arguments":{"content":"Run make migrate before deploying billing"}}),
        );
        let text = saved["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("\"scope\":\"billing\""), "{text}");
        let found = call(
            Some("billing"),
            4,
            "tools/call",
            json!({"name":"recall","arguments":{"query":"make migrate deploy"}}),
        );
        assert!(
            found["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("make migrate")
        );
        let other = call(
            Some("atlas"),
            5,
            "tools/call",
            json!({"name":"recall","arguments":{"query":"make migrate deploy"}}),
        );
        assert_eq!(other["result"]["content"][0]["text"], "[]");
    }
}
