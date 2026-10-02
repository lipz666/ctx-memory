//! Where the tokens of one request go: static prefix (system, tools) and the dynamic part
//! (user, assistant, tool calls, tool results, reasoning). Estimates only; `report` scales
//! them to the input tokens the upstream reports.
use ctx::inject::{Protocol, content_text, messages, system_text};
use serde_json::{Value, json};

/// Rough token count without a tokenizer: 4 ASCII characters per token, one per other
/// character (CJK text is close to one token per character).
pub fn estimate(text: &str) -> u64 {
    let (ascii, other) = text.chars().fold((0u64, 0u64), |(a, o), c| {
        if c.is_ascii() { (a + 1, o) } else { (a, o + 1) }
    });
    ascii.div_ceil(4) + other
}

fn json_tokens(value: &Value) -> u64 {
    match value {
        Value::Null => 0,
        Value::String(s) => estimate(s),
        other => estimate(&other.to_string()),
    }
}

#[derive(Default, Debug, PartialEq)]
pub struct Composition {
    pub system: u64,
    pub tools: u64,
    pub user: u64,
    pub assistant: u64,
    pub tool_calls: u64,
    pub tool_results: u64,
    pub reasoning: u64,
    pub other: u64,
    pub messages: usize,
    pub user_messages: usize,
    pub tool_result_count: usize,
    pub largest_tool_result: u64,
}

impl Composition {
    pub fn of(protocol: Protocol, body: &Value) -> Self {
        let mut c = Self {
            system: estimate(&system_text(protocol, body)),
            tools: json_tokens(body.get("tools").unwrap_or(&Value::Null)),
            ..Self::default()
        };
        if protocol == Protocol::Responses
            && let Some(input) = body.get("input").and_then(Value::as_str)
        {
            c.user = estimate(input);
            c.messages = 1;
            c.user_messages = 1;
            return c;
        }
        for item in messages(protocol, body).into_iter().flatten() {
            c.messages += 1;
            match protocol {
                Protocol::Chat => c.chat(item),
                Protocol::Anthropic => c.anthropic(item),
                Protocol::Responses => c.responses(item),
            }
        }
        c
    }

    fn tool_result(&mut self, tokens: u64) {
        self.tool_results += tokens;
        self.tool_result_count += 1;
        self.largest_tool_result = self.largest_tool_result.max(tokens);
    }

    fn chat(&mut self, m: &Value) {
        match m.get("role").and_then(Value::as_str) {
            // Already counted by system_text.
            Some("system" | "developer") => {}
            Some("user") => {
                self.user += estimate(&content_text(m));
                self.user_messages += 1;
            }
            Some("assistant") => {
                self.assistant += estimate(&content_text(m));
                self.tool_calls += json_tokens(m.get("tool_calls").unwrap_or(&Value::Null));
                for key in ["reasoning_content", "reasoning"] {
                    self.reasoning += json_tokens(m.get(key).unwrap_or(&Value::Null));
                }
            }
            Some("tool") => self.tool_result(estimate(&content_text(m))),
            _ => self.other += json_tokens(m),
        }
    }

    fn anthropic(&mut self, m: &Value) {
        let user = m.get("role").and_then(Value::as_str) == Some("user");
        let parts = match m.get("content") {
            Some(Value::String(s)) => vec![json!({"type": "text", "text": s})],
            Some(Value::Array(parts)) => parts.clone(),
            _ => vec![],
        };
        let mut counted_user = false;
        for p in &parts {
            match p.get("type").and_then(Value::as_str) {
                Some("text") => {
                    let t = estimate(p.get("text").and_then(Value::as_str).unwrap_or(""));
                    if user {
                        self.user += t;
                        counted_user = true;
                    } else {
                        self.assistant += t;
                    }
                }
                Some("tool_use") => {
                    self.tool_calls += json_tokens(p.get("input").unwrap_or(&Value::Null))
                }
                Some("tool_result") => self.tool_result(estimate(&content_text(p))),
                Some("thinking") => {
                    self.reasoning += json_tokens(p.get("thinking").unwrap_or(&Value::Null))
                }
                _ => self.other += json_tokens(p),
            }
        }
        if counted_user {
            self.user_messages += 1;
        }
    }

    fn responses(&mut self, item: &Value) {
        match item.get("type").and_then(Value::as_str) {
            Some("function_call") => {
                self.tool_calls += json_tokens(item.get("arguments").unwrap_or(&Value::Null))
            }
            Some("function_call_output") => self.tool_result(estimate(&content_text(item))),
            Some("reasoning") => self.reasoning += json_tokens(item),
            _ => match item.get("role").and_then(Value::as_str) {
                Some("system" | "developer") => self.system += estimate(&content_text(item)),
                Some("user") => {
                    self.user += estimate(&content_text(item));
                    self.user_messages += 1;
                }
                Some("assistant") => self.assistant += estimate(&content_text(item)),
                _ => self.other += json_tokens(item),
            },
        }
    }

    pub fn static_prefix(&self) -> u64 {
        self.system + self.tools
    }

    pub fn dynamic(&self) -> u64 {
        self.user
            + self.assistant
            + self.tool_calls
            + self.tool_results
            + self.reasoning
            + self.other
    }

    pub fn to_json(&self) -> Value {
        json!({
            "system": self.system, "tools": self.tools, "user": self.user,
            "assistant": self.assistant, "tool_calls": self.tool_calls,
            "tool_results": self.tool_results, "reasoning": self.reasoning, "other": self.other,
            "static": self.static_prefix(), "dynamic": self.dynamic(),
            "total": self.static_prefix() + self.dynamic(),
            "messages": self.messages, "user_messages": self.user_messages,
            "tool_result_count": self.tool_result_count,
            "largest_tool_result": self.largest_tool_result,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_ascii_and_cjk() {
        assert_eq!(estimate("abcdefgh"), 2);
        assert_eq!(estimate("中文"), 2);
        assert_eq!(estimate(""), 0);
    }

    #[test]
    fn chat_splits_static_and_dynamic() {
        let body = json!({
            "tools": [{"type": "function", "function": {"name": "read"}}],
            "messages": [
                {"role": "system", "content": "abcd"},
                {"role": "user", "content": "abcdabcd"},
                {"role": "assistant", "content": null, "reasoning_content": "abcd",
                 "tool_calls": [{"id": "1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "1", "content": "x".repeat(400)},
                {"role": "tool", "tool_call_id": "2", "content": "x".repeat(40)},
            ]
        });
        let c = Composition::of(Protocol::Chat, &body);
        assert_eq!((c.system, c.user, c.reasoning), (1, 2, 1));
        assert!(c.tools > 0 && c.tool_calls > 0);
        assert_eq!(
            (c.tool_results, c.tool_result_count, c.largest_tool_result),
            (110, 2, 100)
        );
        assert_eq!((c.messages, c.user_messages), (5, 1));
        assert_eq!(
            c.static_prefix() + c.dynamic(),
            c.to_json()["total"].as_u64().unwrap()
        );
    }

    #[test]
    fn anthropic_tool_result_is_not_user_text() {
        let body = json!({
            "system": [{"type": "text", "text": "abcd"}],
            "messages": [
                {"role": "user", "content": "abcd"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "abcd", "signature": "s"},
                    {"type": "tool_use", "id": "t", "name": "read", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "abcdabcd"}]},
            ]
        });
        let c = Composition::of(Protocol::Anthropic, &body);
        assert_eq!(
            (c.system, c.user, c.user_messages, c.reasoning),
            (1, 1, 1, 1)
        );
        assert_eq!((c.tool_results, c.tool_result_count), (2, 1));
    }

    #[test]
    fn responses_items() {
        let body = json!({"instructions": "abcd", "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "abcd"}]},
            {"type": "function_call", "call_id": "c", "name": "f", "arguments": "abcd"},
            {"type": "function_call_output", "call_id": "c", "output": "abcdabcd"},
        ]});
        let c = Composition::of(Protocol::Responses, &body);
        assert_eq!(
            (c.system, c.user, c.tool_calls, c.tool_results),
            (1, 1, 1, 2)
        );
        let s = Composition::of(Protocol::Responses, &json!({"input": "abcd"}));
        assert_eq!((s.user, s.messages), (1, 1));
    }
}
