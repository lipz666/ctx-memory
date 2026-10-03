//! Tool-output tiers (design §6). Outputs of the latest step are hot: kept whole up to
//! `hot_cap`. Older outputs are warm: over `warm_cap` they become a digest, and an output
//! repeated verbatim later becomes a one-line pointer. Every shortened output is stored
//! whole in `Objects` first.
use crate::{
    composition::estimate,
    digest::{self, Labels},
    objects::Objects,
};
use ctx::inject::Protocol;
use serde_json::{Value, json};
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub struct Caps {
    pub hot: u64,
    pub warm: u64,
}

impl Default for Caps {
    fn default() -> Self {
        Self {
            hot: 2500,
            warm: 400,
        }
    }
}

#[derive(Default, Debug, PartialEq)]
pub struct Stats {
    pub hot_shortened: usize,
    pub warm_digested: usize,
    pub duplicates: usize,
    pub pinned: usize,
    pub tokens_before: u64,
    pub tokens_after: u64,
}

impl Stats {
    pub fn to_json(&self) -> Value {
        json!({
            "hot_shortened": self.hot_shortened, "warm_digested": self.warm_digested,
            "duplicates": self.duplicates, "pinned": self.pinned,
            "tool_tokens_before": self.tokens_before,
            "tool_tokens_after": self.tokens_after,
        })
    }
}

/// One tool output in the request: where its text lives and which tool produced it.
struct Slot {
    pointer: String,
    item: usize,
    tool: String,
    text: String,
}

/// Text of a content value made only of text (a string, or text parts); None otherwise.
fn plain_text(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => parts
            .iter()
            .map(|p| match p.get("type").and_then(Value::as_str) {
                Some("text" | "input_text" | "output_text") => {
                    p.get("text").and_then(Value::as_str).map(str::to_owned)
                }
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .map(|t| t.join("\n")),
        _ => None,
    }
}

fn slots(protocol: Protocol, body: &Value) -> (Vec<Slot>, Option<usize>) {
    let key = if protocol == Protocol::Responses {
        "input"
    } else {
        "messages"
    };
    let Some(items) = body.get(key).and_then(Value::as_array) else {
        return (vec![], None);
    };
    let mut names: HashMap<String, String> = HashMap::new();
    let mut slots = vec![];
    let mut last_call = None;
    for (i, item) in items.iter().enumerate() {
        let role = item.get("role").and_then(Value::as_str);
        let kind = item.get("type").and_then(Value::as_str);
        if role == Some("assistant") || kind == Some("function_call") {
            last_call = Some(i);
        }
        match protocol {
            Protocol::Chat => {
                for call in item
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    if let (Some(id), Some(name)) = (
                        call.get("id").and_then(Value::as_str),
                        call.pointer("/function/name").and_then(Value::as_str),
                    ) {
                        names.insert(id.into(), name.into());
                    }
                }
                if role == Some("tool")
                    && let Some(text) = item.get("content").and_then(plain_text)
                {
                    let id = item
                        .get("tool_call_id")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    slots.push(Slot {
                        pointer: format!("/messages/{i}/content"),
                        item: i,
                        tool: names.get(id).cloned().unwrap_or_else(|| "tool".into()),
                        text,
                    });
                }
            }
            Protocol::Anthropic => {
                for (j, part) in item
                    .get("content")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .enumerate()
                {
                    match part.get("type").and_then(Value::as_str) {
                        Some("tool_use") => {
                            if let (Some(id), Some(name)) = (
                                part.get("id").and_then(Value::as_str),
                                part.get("name").and_then(Value::as_str),
                            ) {
                                names.insert(id.into(), name.into());
                            }
                        }
                        Some("tool_result") => {
                            if let Some(text) = part.get("content").and_then(plain_text) {
                                let id = part.get("tool_use_id").and_then(Value::as_str);
                                slots.push(Slot {
                                    pointer: format!("/messages/{i}/content/{j}/content"),
                                    item: i,
                                    tool: id
                                        .and_then(|id| names.get(id).cloned())
                                        .unwrap_or_else(|| "tool".into()),
                                    text,
                                });
                            }
                        }
                        _ => {}
                    }
                }
            }
            Protocol::Responses => {
                if kind == Some("function_call")
                    && let (Some(id), Some(name)) = (
                        item.get("call_id").and_then(Value::as_str),
                        item.get("name").and_then(Value::as_str),
                    )
                {
                    names.insert(id.into(), name.into());
                }
                if kind == Some("function_call_output")
                    && let Some(text) = item.get("output").and_then(plain_text)
                {
                    let id = item.get("call_id").and_then(Value::as_str).unwrap_or("");
                    slots.push(Slot {
                        pointer: format!("/input/{i}/output"),
                        item: i,
                        tool: names.get(id).cloned().unwrap_or_else(|| "tool".into()),
                        text,
                    });
                }
            }
        }
    }
    (slots, last_call)
}

/// Shorten tool outputs in place. `expand` says whether the request offers `ccx_expand`.
/// `pins` maps an object id to what the model already read from it with `ccx_expand` in
/// this session; it is attached to that output from then on, so it is not read again.
pub fn apply(
    protocol: Protocol,
    body: &mut Value,
    objects: &Objects,
    caps: Caps,
    expand: bool,
    pins: &HashMap<String, String>,
) -> std::io::Result<Stats> {
    let (slots, last_call) = slots(protocol, body);
    let mut stats = Stats::default();
    let mut later: HashMap<String, usize> = HashMap::new();
    for slot in slots.iter().rev() {
        let tokens = estimate(&slot.text);
        stats.tokens_before += tokens;
        let hot = last_call.is_none_or(|last| slot.item > last);
        let cap = if hot { caps.hot } else { caps.warm };
        let repeated = !hot && tokens > caps.warm && later.contains_key(&slot.text);
        *later.entry(slot.text.clone()).or_default() += 1;
        if tokens <= cap && !repeated {
            stats.tokens_after += tokens;
            continue;
        }
        let id = objects.put(&slot.text)?;
        let labels = Labels {
            tool: &slot.tool,
            id: &id,
            expand,
        };
        let mut short = if repeated {
            stats.duplicates += 1;
            digest::duplicate(&labels)
        } else if hot {
            stats.hot_shortened += 1;
            digest::hot(&slot.text, caps.hot, &labels)
        } else {
            stats.warm_digested += 1;
            digest::warm(&slot.text, caps.warm, &labels)
        };
        if !repeated && let Some(read) = pins.get(&id) {
            stats.pinned += 1;
            short.push_str(&digest::pinned(read));
        }
        stats.tokens_after += estimate(&short);
        if let Some(target) = body.pointer_mut(&slot.pointer) {
            *target = Value::String(short);
        }
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn big(tag: &str) -> String {
        (0..600).map(|i| format!("{tag} line {i}\n")).collect()
    }

    fn chat(outputs: &[String]) -> Value {
        let mut messages = vec![json!({"role": "user", "content": "task"})];
        for (i, out) in outputs.iter().enumerate() {
            messages.push(json!({"role": "assistant", "content": null, "tool_calls": [
                {"id": format!("c{i}"), "type": "function",
                 "function": {"name": "python_execute", "arguments": "{}"}}]}));
            messages.push(json!({"role": "tool", "tool_call_id": format!("c{i}"), "content": out}));
        }
        json!({"messages": messages})
    }

    #[test]
    fn warm_digest_hot_excerpt_and_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let objects = Objects::open(dir.path()).unwrap();
        let (a, b) = (big("a"), big("b"));
        let mut body = chat(&[a.clone(), "small".into(), a.clone(), b.clone()]);
        let caps = Caps {
            hot: 1000,
            warm: 100,
        };
        let stats = apply(
            Protocol::Chat,
            &mut body,
            &objects,
            caps,
            true,
            &HashMap::new(),
        )
        .unwrap();
        let content = |i: usize| body["messages"][i]["content"].as_str().unwrap().to_owned();
        assert!(content(2).starts_with("[ccx: same python_execute output"));
        assert_eq!(content(4), "small");
        assert!(content(6).starts_with("[ccx digest: python_execute output"));
        assert!(content(8).contains("[ccx: python_execute output shortened"));
        assert!(content(8).len() > content(6).len());
        assert_eq!(
            (stats.hot_shortened, stats.warm_digested, stats.duplicates),
            (1, 1, 1)
        );
        assert!(stats.tokens_after * 3 < stats.tokens_before);
        let id = crate::objects::id_of(&b);
        assert_eq!(objects.get(&id).as_deref(), Some(b.as_str()));
        // Same input, same bytes.
        let mut again = chat(&[a.clone(), "small".into(), a, b]);
        apply(
            Protocol::Chat,
            &mut again,
            &objects,
            caps,
            true,
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(again, body);
    }

    #[test]
    fn pins_attach_what_was_read() {
        let dir = tempfile::tempdir().unwrap();
        let objects = Objects::open(dir.path()).unwrap();
        let a = big("a");
        let id = crate::objects::id_of(&a);
        let pins = HashMap::from([(id, "1 of 600 lines match: 7: a line 7".to_owned())]);
        let mut body = chat(&[a, "small".into()]);
        let caps = Caps {
            hot: 1000,
            warm: 100,
        };
        let stats = apply(Protocol::Chat, &mut body, &objects, caps, true, &pins).unwrap();
        assert_eq!(stats.pinned, 1);
        let text = body["messages"][2]["content"].as_str().unwrap();
        assert!(text.contains("you read this part with ccx_expand earlier"));
        assert!(text.ends_with("7: a line 7"));
    }

    #[test]
    fn anthropic_and_responses_slots() {
        let dir = tempfile::tempdir().unwrap();
        let objects = Objects::open(dir.path()).unwrap();
        let caps = Caps {
            hot: 1000,
            warm: 100,
        };
        let mut body = json!({"messages": [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                "content": [{"type": "text", "text": big("r")}]}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "Bash", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "ok"}]},
        ]});
        let stats = apply(
            Protocol::Anthropic,
            &mut body,
            &objects,
            caps,
            false,
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(stats.warm_digested, 1);
        let text = body
            .pointer("/messages/2/content/0/content")
            .unwrap()
            .as_str()
            .unwrap();
        assert!(text.starts_with("[ccx digest: Read output") && text.contains("re-run the tool"));
        let mut body = json!({"input": [
            {"type": "function_call", "call_id": "c", "name": "shell", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c", "output": big("s")},
        ]});
        let stats = apply(
            Protocol::Responses,
            &mut body,
            &objects,
            caps,
            false,
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(stats.hot_shortened, 1);
        assert!(
            body["input"][1]["output"]
                .as_str()
                .unwrap()
                .contains("shell output shortened")
        );
    }

    #[test]
    fn non_text_content_is_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        let objects = Objects::open(dir.path()).unwrap();
        let image = json!([{"type": "image_url", "image_url": {"url": "data:"}}]);
        let mut body = json!({"messages": [
            {"role": "assistant", "tool_calls": [{"id": "c", "function": {"name": "shot"}}]},
            {"role": "tool", "tool_call_id": "c", "content": image.clone()},
        ]});
        apply(
            Protocol::Chat,
            &mut body,
            &objects,
            Caps::default(),
            true,
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(body["messages"][1]["content"], image);
    }
}
