//! `ccx_expand`: lets the model read a shortened tool output back. ccx adds the tool to
//! Chat requests and answers its calls itself (an internal round with the upstream), so the
//! Agent never sees it. Results are ephemeral: the next Agent request does not carry them.
use crate::{composition::estimate, digest::excerpt, objects::Objects};
use serde_json::{Value, json};

pub const NAME: &str = "ccx_expand";
const DEFAULT_LINES: usize = 120;

pub fn chat_tool() -> Value {
    json!({"type": "function", "function": {
        "name": NAME,
        "description": "Read a tool output that was shortened to save context. Outputs marked \
            [ccx ...] carry an id. Give `query` to get only matching lines (with line numbers), \
            or `offset` (0-based line) and `lines` to page through. Use it only when the \
            shortened text is not enough.",
        "parameters": {"type": "object", "properties": {
            "id": {"type": "string", "description": "The id shown in the [ccx ...] marker."},
            "query": {"type": "string", "description": "Words to look for; lines matching any word are returned."},
            "offset": {"type": "integer", "description": "First line to return (0-based)."},
            "lines": {"type": "integer", "description": "How many lines to return (default 120)."}
        }, "required": ["id"]}
    }})
}

/// Answer one call. `cap` bounds the answer in tokens.
pub fn run(objects: &Objects, arguments: &str, cap: u64) -> String {
    let args: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);
    let Some(id) = args.get("id").and_then(Value::as_str) else {
        return "ccx_expand: missing id".into();
    };
    let Some(text) = objects.get(id) else {
        return format!("ccx_expand: unknown id {id}");
    };
    let lines: Vec<&str> = text.lines().collect();
    let query = args
        .get("query")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    let body = if !query.is_empty() {
        let words: Vec<String> = query.split_whitespace().map(str::to_lowercase).collect();
        let hits: Vec<String> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| {
                let lower = l.to_lowercase();
                words.iter().any(|w| lower.contains(w.as_str()))
            })
            .map(|(i, l)| format!("{i}: {l}"))
            .collect();
        if hits.is_empty() {
            format!(
                "no lines in {id} match {query:?} ({} lines in total)",
                lines.len()
            )
        } else {
            format!(
                "{} of {} lines in {id} match {query:?}:\n{}",
                hits.len(),
                lines.len(),
                hits.join("\n")
            )
        }
    } else {
        let offset = args.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
        let count = args
            .get("lines")
            .and_then(Value::as_u64)
            .map_or(DEFAULT_LINES, |n| n as usize);
        let end = (offset + count).min(lines.len());
        let start = offset.min(end);
        let window: Vec<String> = (start..end).map(|i| format!("{i}: {}", lines[i])).collect();
        format!(
            "lines {start}-{} of {} in {id}:\n{}",
            end.saturating_sub(1),
            lines.len(),
            window.join("\n")
        )
    };
    if estimate(&body) <= cap {
        body
    } else {
        format!(
            "{}[ccx_expand: answer cut to ~{cap} tokens; narrow the query or use a smaller `lines`]",
            excerpt(&body, cap)
        )
    }
}

/// The `ccx_expand` calls of a Chat response's first choice, if any.
pub fn calls(response: &Value) -> Vec<Value> {
    response
        .pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .filter(|c| c.pointer("/function/name").and_then(Value::as_str) == Some(NAME))
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

/// Append one internal round to a Chat request: the assistant turn reduced to its
/// `ccx_expand` calls (other calls are dropped; the model reissues them after reading),
/// then one tool message per call.
/// Returns (object id, answer) for every call, so the caller can keep what was read.
pub fn append_round(
    body: &mut Value,
    response: &Value,
    objects: &Objects,
    cap: u64,
) -> Vec<(String, String)> {
    let calls = calls(response);
    let content = response
        .pointer("/choices/0/message/content")
        .cloned()
        .unwrap_or(Value::Null);
    let mut turn = json!({"role": "assistant", "content": content, "tool_calls": calls});
    if let Some(reasoning) = response.pointer("/choices/0/message/reasoning_content") {
        turn["reasoning_content"] = reasoning.clone();
    }
    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return vec![];
    };
    messages.push(turn);
    let mut read = vec![];
    for call in &calls {
        let arguments = call
            .pointer("/function/arguments")
            .and_then(Value::as_str)
            .unwrap_or("{}");
        let answer = run(objects, arguments, cap);
        if let Some(id) = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|a| a.get("id").and_then(Value::as_str).map(str::to_owned))
            && objects.get(&id).is_some()
        {
            read.push((id, answer.clone()));
        }
        messages.push(json!({
            "role": "tool",
            "tool_call_id": call.get("id").cloned().unwrap_or(Value::Null),
            "content": answer,
        }));
    }
    read
}

/// Add the tool to a Chat request that already offers tools; true if added.
pub fn offer(body: &mut Value) -> bool {
    match body.get_mut("tools").and_then(Value::as_array_mut) {
        Some(tools) if !tools.is_empty() => {
            tools.push(chat_tool());
            true
        }
        _ => false,
    }
}

/// Remove the tool (last internal round: the model must answer with the Agent's tools).
pub fn withdraw(body: &mut Value) {
    if let Some(tools) = body.get_mut("tools").and_then(Value::as_array_mut) {
        tools.retain(|t| t.pointer("/function/name").and_then(Value::as_str) != Some(NAME));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, Objects, String) {
        let dir = tempfile::tempdir().unwrap();
        let objects = Objects::open(dir.path()).unwrap();
        let text: String = (0..300)
            .map(|i| format!("row {i} user{} ok\n", i % 7))
            .collect();
        let id = objects.put(&text).unwrap();
        (dir, objects, id)
    }

    #[test]
    fn query_and_paging() {
        let (_dir, objects, id) = store();
        let out = run(
            &objects,
            &json!({"id": id, "query": "user3"}).to_string(),
            2000,
        );
        assert!(out.starts_with("43 of 300 lines") && out.contains("3: row 3 user3 ok"));
        let out = run(
            &objects,
            &json!({"id": id, "offset": 10, "lines": 2}).to_string(),
            2000,
        );
        assert_eq!(
            out,
            format!("lines 10-11 of 300 in {id}:\n10: row 10 user3 ok\n11: row 11 user4 ok")
        );
        let out = run(&objects, &json!({"id": id, "lines": 300}).to_string(), 200);
        assert!(out.contains("answer cut"));
        assert!(run(&objects, r#"{"id":"nope"}"#, 100).contains("unknown id"));
    }

    #[test]
    fn internal_round_keeps_only_expand_calls() {
        let (_dir, objects, id) = store();
        let response = json!({"choices": [{"message": {"role": "assistant", "content": null,
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": NAME,
                    "arguments": json!({"id": id, "offset": 0, "lines": 1}).to_string()}},
                {"id": "b", "type": "function", "function": {"name": "python", "arguments": "{}"}}]}}]});
        let mut body = json!({"messages": [{"role": "user", "content": "t"}],
            "tools": [{"type": "function", "function": {"name": "python"}}]});
        assert!(offer(&mut body));
        assert_eq!(calls(&response).len(), 1);
        let read = append_round(&mut body, &response, &objects, 1000);
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].0, id);
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["tool_calls"].as_array().unwrap().len(), 1);
        assert_eq!(messages[2]["tool_call_id"], "a");
        assert!(
            messages[2]["content"]
                .as_str()
                .unwrap()
                .contains("0: row 0")
        );
        withdraw(&mut body);
        assert_eq!(body["tools"].as_array().unwrap().len(), 1);
        assert!(!offer(&mut json!({"messages": []})));
    }
}
