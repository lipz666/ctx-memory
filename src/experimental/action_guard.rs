//! Experimental ActionGuard: before a proposed tool call, surface `before_action`
//! lessons (advisory) or ask the model to confirm the call (recheck).
use crate::{
    inject::{self, Protocol},
    recall::{self, Hit, Query},
    store::Store,
};
use serde_json::{Value, json};

pub fn proposed_call(protocol: Protocol, request: &Value) -> Option<Value> {
    match protocol {
        Protocol::Chat => {
            let messages = request.get("messages")?.as_array()?;
            let assistant = messages
                .iter()
                .rev()
                .find(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))?;
            let call = assistant.get("tool_calls")?.as_array()?.first()?;
            let name = call.pointer("/function/name")?.as_str()?;
            let raw = call
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .unwrap_or("{}");
            Some(
                json!({"tool":name,"args":serde_json::from_str::<Value>(raw).unwrap_or(Value::Null)}),
            )
        }
        Protocol::Responses => {
            let items = request.get("input")?.as_array()?;
            let call = items
                .iter()
                .rev()
                .find(|m| m.get("type").and_then(Value::as_str) == Some("function_call"))?;
            let name = call.get("name")?.as_str()?;
            let raw = call
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or("{}");
            Some(
                json!({"tool":name,"args":serde_json::from_str::<Value>(raw).unwrap_or(Value::Null)}),
            )
        }
        Protocol::Anthropic => {
            let messages = request.get("messages")?.as_array()?;
            let assistant = messages
                .iter()
                .rev()
                .find(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))?;
            let call = assistant
                .get("content")?
                .as_array()?
                .iter()
                .find(|part| part.get("type").and_then(Value::as_str) == Some("tool_use"))?;
            Some(
                json!({"tool":call.get("name")?.as_str()?,"args":call.get("input").cloned().unwrap_or(Value::Null)}),
            )
        }
    }
}
fn last_is_tool_result(protocol: Protocol, request: &Value) -> bool {
    match protocol {
        Protocol::Chat => {
            request
                .pointer("/messages")
                .and_then(Value::as_array)
                .and_then(|m| m.last())
                .and_then(|m| m.get("role"))
                .and_then(Value::as_str)
                == Some("tool")
        }
        Protocol::Responses => {
            request
                .get("input")
                .and_then(Value::as_array)
                .and_then(|m| m.last())
                .and_then(|m| m.get("type"))
                .and_then(Value::as_str)
                == Some("function_call_output")
        }
        Protocol::Anthropic => request
            .get("messages")
            .and_then(Value::as_array)
            .and_then(|m| m.last())
            .and_then(|m| m.get("content"))
            .and_then(Value::as_array)
            .is_some_and(|parts| {
                parts
                    .iter()
                    .any(|p| p.get("type").and_then(Value::as_str) == Some("tool_result"))
            }),
    }
}
pub fn advisory_hits(
    store: &Store,
    protocol: Protocol,
    request: &Value,
    project: Option<&str>,
) -> Vec<Hit> {
    if store.config.experimental.action_guard == "off" || !last_is_tool_result(protocol, request) {
        return vec![];
    }
    let Some(call) = proposed_call(protocol, request) else {
        return vec![];
    };
    before_action_hits(store, &call, project)
}
/// Lessons whose `before_action` trigger matches a proposed call (`{"tool","args"}`).
pub fn before_action_hits(store: &Store, call: &Value, project: Option<&str>) -> Vec<Hit> {
    recall::recall(
        store,
        &Query {
            project,
            features: Some(call),
            limit: store.config.recall.max_injected,
            before_action: true,
            ..Default::default()
        },
    )
    .unwrap_or_default()
}
pub fn response_call(protocol: Protocol, response: &Value) -> Option<Value> {
    match protocol {
        Protocol::Chat => {
            let call = response
                .pointer("/choices/0/message/tool_calls")?
                .as_array()?
                .first()?;
            let name = call.pointer("/function/name")?.as_str()?;
            let raw = call
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .unwrap_or("{}");
            Some(
                json!({"tool":name,"args":serde_json::from_str::<Value>(raw).unwrap_or(Value::Null)}),
            )
        }
        Protocol::Responses => {
            let call =
                response.get("output")?.as_array()?.iter().find(|item| {
                    item.get("type").and_then(Value::as_str) == Some("function_call")
                })?;
            let name = call.get("name")?.as_str()?;
            let raw = call
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or("{}");
            Some(
                json!({"tool":name,"args":serde_json::from_str::<Value>(raw).unwrap_or(Value::Null)}),
            )
        }
        Protocol::Anthropic => {
            let call = response
                .get("content")?
                .as_array()?
                .iter()
                .find(|item| item.get("type").and_then(Value::as_str) == Some("tool_use"))?;
            Some(
                json!({"tool":call.get("name")?.as_str()?,"args":call.get("input").cloned().unwrap_or(Value::Null)}),
            )
        }
    }
}
pub fn recheck_body(
    protocol: Protocol,
    request: &Value,
    proposal: &Value,
    hits: &[Hit],
) -> Option<Value> {
    let mut output = request.clone();
    let reminders = hits
        .iter()
        .map(inject::render)
        .collect::<Vec<_>>()
        .join("\n");
    let text = format!(
        "Before executing the proposed tool call, check this relevant lesson. Confirm the same call or return a corrected tool call. Proposed call: {}\n{}",
        proposal, reminders
    );
    match protocol {
        Protocol::Chat | Protocol::Anthropic => output
            .get_mut("messages")?
            .as_array_mut()?
            .push(json!({"role":"user","content":text})),
        Protocol::Responses => {
            let input = output.get_mut("input")?;
            match input {
                Value::String(value) => value.push_str(&format!("\n{text}")),
                Value::Array(items) => {
                    items.push(json!({"role":"user","content":[{"type":"input_text","text":text}]}))
                }
                _ => return None,
            }
        }
    }
    output["stream"] = Value::Bool(false);
    output.as_object_mut()?.remove("stream_options");
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn extracts_three_protocol_tool_calls() {
        let chat = json!({"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"deploy","arguments":"{\"service\":\"payments\"}"}}]},{"role":"tool","content":"ok"}]});
        assert_eq!(
            proposed_call(Protocol::Chat, &chat).unwrap()["args"]["service"],
            "payments"
        );
        let responses = json!({"input":[{"type":"function_call","name":"deploy","arguments":"{}"},{"type":"function_call_output","output":"ok"}]});
        assert_eq!(
            proposed_call(Protocol::Responses, &responses).unwrap()["tool"],
            "deploy"
        );
        let anthropic = json!({"messages":[{"role":"assistant","content":[{"type":"tool_use","name":"deploy","input":{}}]},{"role":"user","content":[{"type":"tool_result","content":"ok"}]}]});
        assert_eq!(
            proposed_call(Protocol::Anthropic, &anthropic).unwrap()["tool"],
            "deploy"
        );
    }
    #[test]
    fn detects_proposed_chat_response() {
        let response = json!({"choices":[{"message":{"tool_calls":[{"function":{"name":"deploy","arguments":"{\"service\":\"payments\"}"}}]}}]});
        assert_eq!(
            response_call(Protocol::Chat, &response).unwrap()["tool"],
            "deploy"
        );
        let responses =
            json!({"output":[{"type":"function_call","name":"deploy","arguments":"{}"}]});
        assert_eq!(
            response_call(Protocol::Responses, &responses).unwrap()["tool"],
            "deploy"
        );
        let anthropic = json!({"content":[{"type":"tool_use","name":"deploy","input":{}}]});
        assert_eq!(
            response_call(Protocol::Anthropic, &anthropic).unwrap()["tool"],
            "deploy"
        );
    }
}
