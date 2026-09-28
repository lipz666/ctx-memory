use crate::inject::Protocol;
use serde_json::{Value, json};
use std::sync::LazyLock;

static EXIT: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(
        r"(?i)(?:exit(?:ed)?(?:\s+with)?(?:\s+code|\s+status)?|return code)\s*[:=]?\s*(-?\d+)",
    )
    .unwrap()
});
static TIME: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"\b\d{4}-\d{2}-\d{2}[T ][0-9:.+Z-]+\b").unwrap());
static HASH: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"\b[0-9a-fA-F]{8,}\b").unwrap());
static LINE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"(?i)\bline\s+\d+\b|:\d+(?::\d+)?\b").unwrap());
static NUMBER: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r"\b\d{2,}\b").unwrap());
static TEMP: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"/tmp/\S+|/var/folders/\S+").unwrap());
static FILE: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9]+\b").unwrap()
});

fn text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .filter_map(|part| {
                part.get("text")
                    .or_else(|| part.get("content"))
                    .and_then(Value::as_str)
            })
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}
fn parse_arguments(value: Option<&Value>) -> Value {
    match value {
        Some(Value::String(raw)) => serde_json::from_str(raw).unwrap_or(Value::Null),
        Some(value) => value.clone(),
        None => Value::Null,
    }
}
pub fn tool_result(protocol: Protocol, body: &Value) -> Option<(String, Value, String)> {
    match protocol {
        Protocol::Chat => {
            let messages = body.get("messages")?.as_array()?;
            let last = messages.last()?;
            if last.get("role")?.as_str()? != "tool" {
                return None;
            }
            let call_id = last.get("tool_call_id").and_then(Value::as_str);
            let call = messages
                .iter()
                .rev()
                .filter_map(|m| m.get("tool_calls").and_then(Value::as_array))
                .flat_map(|items| items.iter())
                .find(|call| {
                    call_id.is_none_or(|id| call.get("id").and_then(Value::as_str) == Some(id))
                });
            let name = last
                .get("name")
                .and_then(Value::as_str)
                .or_else(|| {
                    call.and_then(|v| v.pointer("/function/name"))
                        .and_then(Value::as_str)
                })
                .unwrap_or("unknown");
            let args = parse_arguments(call.and_then(|v| v.pointer("/function/arguments")));
            Some((name.into(), args, text(last.get("content")?)))
        }
        Protocol::Responses => {
            let items = body.get("input")?.as_array()?;
            let last = items.last()?;
            if last.get("type")?.as_str()? != "function_call_output" {
                return None;
            }
            let call_id = last.get("call_id").and_then(Value::as_str);
            let call = items.iter().rev().find(|item| {
                item.get("type").and_then(Value::as_str) == Some("function_call")
                    && call_id
                        .is_none_or(|id| item.get("call_id").and_then(Value::as_str) == Some(id))
            });
            let name = call
                .and_then(|v| v.get("name"))
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let args = parse_arguments(call.and_then(|v| v.get("arguments")));
            Some((name.into(), args, text(last.get("output")?)))
        }
        Protocol::Anthropic => {
            let messages = body.get("messages")?.as_array()?;
            let last = messages.last()?;
            let result = last
                .get("content")?
                .as_array()?
                .iter()
                .find(|part| part.get("type").and_then(Value::as_str) == Some("tool_result"))?;
            let call_id = result.get("tool_use_id").and_then(Value::as_str);
            let call = messages
                .iter()
                .rev()
                .filter(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
                .filter_map(|m| m.get("content").and_then(Value::as_array))
                .flat_map(|items| items.iter())
                .find(|part| {
                    part.get("type").and_then(Value::as_str) == Some("tool_use")
                        && call_id
                            .is_none_or(|id| part.get("id").and_then(Value::as_str) == Some(id))
                });
            let name = call
                .and_then(|v| v.get("name"))
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let args = parse_arguments(call.and_then(|v| v.get("input")));
            Some((name.into(), args, text(result.get("content")?)))
        }
    }
}
pub fn error_signature(output: &str, exit: Option<i64>) -> Option<String> {
    if exit.unwrap_or(0) == 0
        && !output.lines().any(|line| {
            let lower = line.to_lowercase();
            ["error", "failed", "exception", "panic", "traceback"]
                .iter()
                .any(|needle| lower.contains(needle))
        })
    {
        return None;
    }
    let line = output
        .lines()
        .find(|line| {
            let lower = line.to_lowercase();
            ["error", "failed", "exception", "panic", "traceback"]
                .iter()
                .any(|needle| lower.contains(needle))
        })
        .or_else(|| output.lines().find(|line| !line.trim().is_empty()))
        .unwrap_or("tool failed");
    let normalized = TEMP.replace_all(line, "<temp>");
    let normalized = TIME.replace_all(&normalized, "<time>");
    let normalized = HASH.replace_all(&normalized, "<hash>");
    let normalized = LINE.replace_all(&normalized, "<line>");
    let normalized = NUMBER.replace_all(&normalized, "<id>");
    Some(
        normalized
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .take(200)
            .collect(),
    )
}
pub fn enrich(protocol: Protocol, body: &Value, features: &mut Value) {
    let Some((tool, args, output)) = tool_result(protocol, body) else {
        return;
    };
    let exit = EXIT
        .captures(&output)
        .and_then(|capture| capture.get(1))
        .and_then(|number| number.as_str().parse::<i64>().ok());
    features["role"] = "tool".into();
    features["tool"] = tool.into();
    features["args"] = args.clone();
    features["text"] = output.chars().take(4000).collect::<String>().into();
    if let Some(code) = exit {
        features["exit_code"] = code.into();
    }
    if let Some(signature) = error_signature(&output, exit) {
        features["error_sig"] = signature.into();
    }
    let files = FILE
        .find_iter(&output)
        .take(8)
        .map(|m| m.as_str().to_owned())
        .collect::<Vec<_>>();
    if !files.is_empty() {
        features["files"] = json!(files);
    }
    if let Some(service) = args.get("service").and_then(Value::as_str) {
        features["entities"] = json!([service]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalizes_error_numbers() {
        let a = error_signature(
            "2026-09-25T10:12:03Z Error at line 12: schema mismatch 12345",
            Some(1),
        )
        .unwrap();
        let b = error_signature(
            "2026-09-26T13:44:03Z Error at line 99: schema mismatch 98765",
            Some(1),
        )
        .unwrap();
        assert_eq!(a, b);
    }
    #[test]
    fn extracts_tool_results_from_three_protocols() {
        let chat = json!({"messages":[{"role":"assistant","tool_calls":[{"id":"call_1","function":{"name":"deploy","arguments":"{\"service\":\"payments\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"Error: schema mismatch exit code 1"}]});
        let responses = json!({"input":[{"type":"function_call","call_id":"call_1","name":"deploy","arguments":"{\"service\":\"payments\"}"},{"type":"function_call_output","call_id":"call_1","output":"Error: schema mismatch exit code 1"}]});
        let anthropic = json!({"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"deploy","input":{"service":"payments"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"Error: schema mismatch exit code 1"}]}]});
        for (protocol, body) in [
            (Protocol::Chat, chat),
            (Protocol::Responses, responses),
            (Protocol::Anthropic, anthropic),
        ] {
            let mut features = json!({});
            enrich(protocol, &body, &mut features);
            assert_eq!(features["tool"], "deploy");
            assert_eq!(features["args"]["service"], "payments");
            assert_eq!(features["exit_code"], 1);
        }
    }
}
