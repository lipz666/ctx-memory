//! Reading and rewriting Agent requests for the three supported protocols.
use crate::recall::Hit;
use anyhow::{Result, bail};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Protocol {
    Chat,
    Responses,
    Anthropic,
}
impl Protocol {
    pub fn from_path(path: &str) -> Option<Self> {
        match path {
            "chat/completions" => Some(Self::Chat),
            "responses" => Some(Self::Responses),
            "messages" => Some(Self::Anthropic),
            _ => None,
        }
    }
}

const PREFIX: &str = "Memories from earlier work (ctx). They may be outdated: prefer what you observe now, and say so if one is wrong.";

/// The conversation items (`messages`, or `input` for Responses).
pub fn messages(protocol: Protocol, body: &Value) -> Option<&Vec<Value>> {
    match protocol {
        Protocol::Responses => body.get("input"),
        _ => body.get("messages"),
    }
    .and_then(Value::as_array)
}

/// Plain text of a message or content value.
pub fn content_text(message: &Value) -> String {
    let content = message
        .get("content")
        .or_else(|| message.get("output"))
        .unwrap_or(message);
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| match p.get("type").and_then(Value::as_str) {
                Some("tool_result") => Some(content_text(p)),
                _ => p.get("text").and_then(Value::as_str).map(str::to_owned),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

pub fn system_text(protocol: Protocol, body: &Value) -> String {
    match protocol {
        Protocol::Anthropic => {
            content_text(&json!({"content": body.get("system").cloned().unwrap_or(Value::Null)}))
        }
        Protocol::Responses => body
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned(),
        Protocol::Chat => messages(protocol, body)
            .map(|items| {
                items
                    .iter()
                    .filter(|m| {
                        matches!(
                            m.get("role").and_then(Value::as_str),
                            Some("system" | "developer")
                        )
                    })
                    .map(content_text)
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default(),
    }
}

fn role_of(protocol: Protocol, message: &Value) -> &'static str {
    if protocol == Protocol::Responses
        && message.get("type").and_then(Value::as_str) == Some("function_call_output")
    {
        return "tool";
    }
    if protocol == Protocol::Anthropic
        && message
            .get("content")
            .and_then(Value::as_array)
            .is_some_and(|parts| {
                parts
                    .iter()
                    .any(|p| p.get("type").and_then(Value::as_str) == Some("tool_result"))
            })
    {
        return "tool";
    }
    match message.get("role").and_then(Value::as_str) {
        Some("user") => "user",
        Some("tool") => "tool",
        Some("assistant") => "assistant",
        Some("system" | "developer") => "system",
        _ => "other",
    }
}

/// Features of the step: role and text of the latest message, the user's text, and tool
/// details (name, args, error signature, files) when the latest message is a tool result.
pub fn features(protocol: Protocol, body: &Value) -> Value {
    let mut features = json!({});
    let latest = match protocol {
        Protocol::Responses if body.get("input").is_some_and(Value::is_string) => {
            Some(("user", body["input"].as_str().unwrap_or("").to_owned()))
        }
        _ => messages(protocol, body).and_then(|items| {
            items
                .iter()
                .rev()
                .find(|m| role_of(protocol, m) != "system" && role_of(protocol, m) != "other")
                .map(|m| (role_of(protocol, m), content_text(m)))
        }),
    };
    if let Some((role, text)) = latest {
        features["role"] = role.into();
        features["text"] = text.chars().take(4000).collect::<String>().into();
        if role == "user" {
            features["user_text"] = features["text"].clone();
        }
    }
    crate::observer::enrich(protocol, body, &mut features);
    features
}

/// Index of the message new memories attach to: the latest user or tool message.
pub fn anchor_index(protocol: Protocol, body: &Value) -> Option<usize> {
    if protocol == Protocol::Responses && body.get("input").is_some_and(Value::is_string) {
        return Some(0);
    }
    let items = messages(protocol, body)?;
    items
        .iter()
        .rposition(|m| matches!(role_of(protocol, m), "user" | "tool"))
}

pub fn render(hit: &Hit) -> String {
    let memory = &hit.memory;
    let note = if memory.status == "contested" {
        " status=\"contested\""
    } else {
        ""
    };
    let date = memory
        .observed_at
        .as_deref()
        .unwrap_or(&memory.created_at)
        .chars()
        .take(10)
        .collect::<String>();
    format!(
        "<ctx-memory id=\"{}\" type=\"{}\" scope=\"{}\" date=\"{date}\"{note}>\n{}\n</ctx-memory>",
        memory.id, memory.kind, memory.scope, memory.body
    )
}

/// Render hits within a token budget (4 characters per token); returns the block and
/// the ids that fit.
pub fn render_block(hits: &[Hit], budget_tokens: usize) -> Option<(String, Vec<String>)> {
    let mut used = 0;
    let mut blocks = vec![];
    let mut ids = vec![];
    for hit in hits {
        let block = render(hit);
        let tokens = block.chars().count().div_ceil(4);
        if used + tokens > budget_tokens {
            continue;
        }
        used += tokens;
        blocks.push(block);
        ids.push(hit.memory.id.clone());
    }
    (!blocks.is_empty()).then(|| (format!("\n\n{PREFIX}\n{}", blocks.join("\n")), ids))
}

/// Add the always-on block to the system prompt and each anchored block to its
/// message. The input is not modified.
pub fn apply(
    protocol: Protocol,
    original: &Value,
    system: Option<&str>,
    anchors: &[(usize, String)],
) -> Result<Value> {
    let mut output = original.clone();
    if let Some(block) = system {
        match protocol {
            Protocol::Chat => {
                let items = output
                    .get_mut("messages")
                    .and_then(Value::as_array_mut)
                    .ok_or_else(|| anyhow::anyhow!("messages missing"))?;
                if let Some(system) = items.iter_mut().find(|m| {
                    matches!(
                        m.get("role").and_then(Value::as_str),
                        Some("system" | "developer")
                    )
                }) {
                    append(system, block, "text")?;
                } else {
                    items.insert(0, json!({"role":"system","content":block.trim_start()}));
                }
            }
            Protocol::Anthropic => match output.get_mut("system") {
                Some(Value::String(s)) => s.push_str(block),
                Some(Value::Array(parts)) => {
                    parts.push(json!({"type":"text","text":block.trim_start()}))
                }
                None | Some(Value::Null) => {
                    output["system"] = Value::String(block.trim_start().into())
                }
                _ => bail!("unsupported system"),
            },
            Protocol::Responses => match output.get_mut("instructions") {
                Some(Value::String(s)) => s.push_str(block),
                None | Some(Value::Null) => {
                    output["instructions"] = Value::String(block.trim_start().into())
                }
                _ => bail!("unsupported instructions"),
            },
        }
    }
    // Inserting a system message at 0 shifts Chat indexes by one.
    let shift = usize::from(
        protocol == Protocol::Chat
            && system.is_some()
            && !messages(protocol, original).is_some_and(|items| {
                items.iter().any(|m| {
                    matches!(
                        m.get("role").and_then(Value::as_str),
                        Some("system" | "developer")
                    )
                })
            }),
    );
    for (index, block) in anchors {
        if protocol == Protocol::Responses && output.get("input").is_some_and(Value::is_string) {
            if let Some(Value::String(s)) = output.get_mut("input") {
                s.push_str(block);
            }
            continue;
        }
        let key = if protocol == Protocol::Responses {
            "input"
        } else {
            "messages"
        };
        let Some(message) = output
            .get_mut(key)
            .and_then(Value::as_array_mut)
            .and_then(|items| items.get_mut(index + shift))
        else {
            bail!("anchor out of range");
        };
        let part = if protocol == Protocol::Responses {
            "input_text"
        } else {
            "text"
        };
        append(message, block, part)?;
    }
    Ok(output)
}

fn append(message: &mut Value, suffix: &str, part_type: &str) -> Result<()> {
    let field = if message.get("content").is_some() {
        "content"
    } else if message.get("output").is_some() {
        "output"
    } else {
        bail!("content missing")
    };
    match message.get_mut(field).unwrap() {
        Value::String(s) => s.push_str(suffix),
        Value::Array(items) => items.push(json!({"type":part_type,"text":suffix.trim_start()})),
        Value::Null => message[field] = Value::String(suffix.trim_start().into()),
        _ => bail!("unsupported content"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn features_distinguish_user_and_tool() {
        let user = json!({"messages":[{"role":"system","content":"s"},{"role":"user","content":"deploy payments"}]});
        let f = features(Protocol::Chat, &user);
        assert_eq!(f["role"], "user");
        assert_eq!(f["user_text"], "deploy payments");
        let tool = json!({"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"t","name":"bash","input":{"cmd":"make"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":"Error: boom"}]}]});
        let f = features(Protocol::Anthropic, &tool);
        assert_eq!(f["role"], "tool");
        assert!(f.get("user_text").is_none());
        assert_eq!(f["tool"], "bash");
        assert_eq!(anchor_index(Protocol::Anthropic, &tool), Some(1));
    }
    #[test]
    fn apply_adds_system_and_anchors_without_touching_input() {
        let body = json!({"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"},{"role":"user","content":[{"type":"text","text":"c"}]}]});
        let out = apply(
            Protocol::Chat,
            &body,
            Some("\n\nRULES"),
            &[(0, "\n\nM0".into()), (2, "\n\nM2".into())],
        )
        .unwrap();
        assert_eq!(out["messages"][0]["content"], "RULES");
        assert_eq!(out["messages"][1]["content"], "a\n\nM0");
        assert_eq!(out["messages"][3]["content"][1]["text"], "M2");
        assert_eq!(body["messages"][0]["content"], "a");
        let responses = json!({"instructions":"i","input":[{"role":"user","content":[{"type":"input_text","text":"q"}]},{"type":"function_call_output","call_id":"x","output":"done"}]});
        let out = apply(
            Protocol::Responses,
            &responses,
            None,
            &[(1, "\n\nM".into())],
        )
        .unwrap();
        assert_eq!(out["input"][1]["output"], "done\n\nM");
        let anthropic = json!({"system":[{"type":"text","text":"s"}],"messages":[{"role":"user","content":"q"}]});
        let out = apply(Protocol::Anthropic, &anthropic, Some("\n\nR"), &[]).unwrap();
        assert_eq!(out["system"][1]["text"], "R");
    }
}
