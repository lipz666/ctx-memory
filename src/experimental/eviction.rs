//! Experimental eviction: replace old, large tool outputs with `expand` placeholders.
//! Off by default: it rewrites history, which also invalidates provider prompt caches.
use crate::{inject::Protocol, store::Store};
use anyhow::Result;
use serde_json::Value;

/// Replace old, large tool outputs in one batch. Persisted content hashes keep later
/// requests byte-stable even when they fall below the eviction threshold.
pub fn compact_history(
    store: &Store,
    protocol: Protocol,
    original: &Value,
    agent: &str,
    project: Option<&str>,
    session: Option<&str>,
) -> Result<(Value, Vec<String>)> {
    let mut output = original.clone();
    let mut candidates = Vec::<(String, String, usize)>::new();
    match protocol {
        Protocol::Chat | Protocol::Anthropic => {
            if let Some(messages) = original.get("messages").and_then(Value::as_array) {
                for (i, message) in messages.iter().enumerate() {
                    if matches!(protocol, Protocol::Chat)
                        && message.get("role").and_then(Value::as_str) == Some("tool")
                        && let Some(s) = message.get("content").and_then(Value::as_str)
                    {
                        candidates.push((format!("/messages/{i}/content"), s.into(), i));
                    }
                    if matches!(protocol, Protocol::Anthropic)
                        && let Some(parts) = message.get("content").and_then(Value::as_array)
                    {
                        for (j, part) in parts.iter().enumerate() {
                            if part.get("type").and_then(Value::as_str) == Some("tool_result")
                                && let Some(s) = part.get("content").and_then(Value::as_str)
                            {
                                candidates.push((
                                    format!("/messages/{i}/content/{j}/content"),
                                    s.into(),
                                    i,
                                ));
                            }
                        }
                    }
                }
            }
        }
        Protocol::Responses => {
            if let Some(input) = original.get("input").and_then(Value::as_array) {
                for (i, item) in input.iter().enumerate() {
                    if item.get("type").and_then(Value::as_str) == Some("function_call_output")
                        && let Some(s) = item.get("output").and_then(Value::as_str)
                    {
                        candidates.push((format!("/input/{i}/output"), s.into(), i));
                    }
                }
            }
        }
    }
    let history_len = match protocol {
        Protocol::Responses => original.get("input"),
        _ => original.get("messages"),
    }
    .and_then(Value::as_array)
    .map(Vec::len)
    .unwrap_or(0);
    let eligible: Vec<_> = candidates
        .iter()
        .filter(|(_, s, i)| {
            *i + 8 < history_len
                && s.chars().count().div_ceil(4) >= store.config.experimental.tool_result_min_tokens
        })
        .collect();
    let pending_tokens: usize = eligible
        .iter()
        .map(|(_, s, _)| s.chars().count().div_ceil(4))
        .sum();
    let batch = pending_tokens >= store.config.experimental.evict_threshold_tokens;
    let mut changed = Vec::new();
    for (path, content, index) in candidates {
        let existing = store.evicted(&content)?;
        let old_enough = index + 8 < history_len;
        let placeholder = if let Some(value) = existing {
            Some(value)
        } else if batch
            && old_enough
            && content.chars().count().div_ceil(4)
                >= store.config.experimental.tool_result_min_tokens
        {
            Some(store.eviction(agent, project, session, &content)?)
        } else {
            None
        };
        if let Some(text) = placeholder
            && let Some(slot) = output.pointer_mut(&path)
        {
            *slot = Value::String(text.clone());
            changed.push(text);
        }
    }
    Ok((output, changed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn batches_and_preserves_eviction() {
        let dir = tempfile::tempdir().unwrap();
        crate::store::init(dir.path()).unwrap();
        let mut store = Store::open(dir.path()).unwrap();
        store.config.experimental.evict_threshold_tokens = 1;
        store.config.experimental.tool_result_min_tokens = 1;
        let large = "tool result ".repeat(100);
        let mut messages = vec![json!({"role":"tool","content":large})];
        for _ in 0..9 {
            messages.push(json!({"role":"user","content":"next"}));
        }
        let request = json!({"messages":messages});
        let (first, ids) =
            compact_history(&store, Protocol::Chat, &request, "test", None, None).unwrap();
        assert_eq!(ids.len(), 1);
        assert!(
            first["messages"][0]["content"]
                .as_str()
                .unwrap()
                .starts_with("[ctx:archived")
        );
        store.config.experimental.evict_threshold_tokens = 999999;
        let (second, _) =
            compact_history(&store, Protocol::Chat, &request, "test", None, None).unwrap();
        assert_eq!(first, second);
    }
}
