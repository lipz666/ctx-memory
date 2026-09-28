//! Conversation sessions seen through the proxy: stable keys, project detection, the
//! new messages of each step (the transcript), and memories already shown per session.
use crate::inject::{self, Protocol};
use crate::store::redact;
use axum::http::HeaderMap;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::{LazyLock, Mutex},
    time::{Duration, Instant},
};

const STATE_TTL: Duration = Duration::from_secs(6 * 3600);

/// A memory block attached to one message of the conversation. It is re-attached on
/// every later step while that message is unchanged, so the model keeps seeing it.
#[derive(Clone)]
pub struct Anchor {
    pub index: usize,
    pub hash: String,
    pub block: String,
    pub ids: Vec<String>,
}

#[derive(Default)]
pub struct State {
    /// Messages already recorded in the transcript.
    pub seen: usize,
    /// Hash of the last recorded message, to notice a rewritten history.
    pub last_hash: Option<String>,
    pub anchors: Vec<Anchor>,
    pub injected: HashSet<String>,
    pub project: Option<String>,
    /// Restored from the database (cursor and project) after a restart.
    pub loaded: bool,
    touched: Option<Instant>,
}

#[derive(Default)]
pub struct Tracker {
    states: Mutex<HashMap<String, State>>,
}
impl Tracker {
    /// Run `f` on the state of `key`, creating it if needed.
    pub fn with<T>(&self, key: &str, f: impl FnOnce(&mut State) -> T) -> T {
        let mut states = self.states.lock().unwrap();
        states.retain(|_, s| s.touched.is_none_or(|t| t.elapsed() < STATE_TTL));
        let state = states.entry(key.to_owned()).or_default();
        state.touched = Some(Instant::now());
        f(state)
    }
}

/// `X-Ctx-Session`, else a session id the client embeds (Claude Code metadata), else a
/// hash of the agent, system prompt and first user message.
pub fn key(agent: &str, headers: &HeaderMap, protocol: Protocol, body: &Value) -> String {
    if let Some(value) = headers
        .get("x-ctx-session")
        .and_then(|v| v.to_str().ok())
        .filter(|v| !v.is_empty() && v.len() <= 200)
    {
        return value.to_owned();
    }
    if let Some(user) = body.pointer("/metadata/user_id").and_then(Value::as_str)
        && let Some((_, session)) = user.rsplit_once("session_")
        && !session.is_empty()
    {
        return format!("{agent}:{}", &session[..session.len().min(64)]);
    }
    let system = inject::system_text(protocol, body);
    let first_user = inject::messages(protocol, body)
        .and_then(|items| {
            items
                .iter()
                .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
                .map(inject::content_text)
        })
        .unwrap_or_default();
    let digest = Sha256::digest(
        format!(
            "{agent}\n{}\n{}",
            system.chars().take(4000).collect::<String>(),
            first_user.chars().take(4000).collect::<String>()
        )
        .as_bytes(),
    );
    format!("{agent}:{}", &hex::encode(digest)[..20])
}

static CWD_PATTERNS: LazyLock<Vec<regex::Regex>> = LazyLock::new(|| {
    [
        r"(?i)(?:current )?working directory(?: is)?\s*[:=]?\s*[`'\x22]?(/[^\s`'\x22<>]+)",
        r"(?i)<cwd>\s*(/[^<\s]+)\s*</cwd>",
        r"(?i)\bcwd\s*[:=]\s*[`'\x22]?(/[^\s`'\x22<>]+)",
        r"(?i)project root(?: is)?\s*[:=]?\s*[`'\x22]?(/[^\s`'\x22<>]+)",
    ]
    .into_iter()
    .map(|p| regex::Regex::new(p).unwrap())
    .collect()
});

/// The project for a request: explicit header or route, else the working directory the
/// Agent states in its prompt.
pub fn detect_project(
    headers: &HeaderMap,
    route: Option<&str>,
    protocol: Protocol,
    body: &Value,
) -> Option<String> {
    if let Some(value) = headers
        .get("x-ctx-project")
        .and_then(|v| v.to_str().ok())
        .or(route)
        .filter(|v| !v.trim().is_empty())
    {
        return Some(crate::memory::normalize_scope(value));
    }
    let mut text = inject::system_text(protocol, body);
    if let Some(items) = inject::messages(protocol, body) {
        for item in items.iter().take(3) {
            text.push('\n');
            text.push_str(&inject::content_text(item));
        }
    }
    let text: String = text.chars().take(60_000).collect();
    CWD_PATTERNS
        .iter()
        .find_map(|p| p.captures(&text).and_then(|c| c.get(1)))
        .and_then(|m| project_name(Path::new(m.as_str().trim_end_matches(['.', ',', ')']))))
}

static GIT_ROOTS: LazyLock<Mutex<HashMap<String, Option<String>>>> =
    LazyLock::new(Default::default);

/// Name of the git repository containing `path`, else the directory name. The home
/// directory and `/` are not projects.
pub fn project_name(path: &Path) -> Option<String> {
    let key = path.display().to_string();
    if let Some(cached) = GIT_ROOTS.lock().unwrap().get(&key) {
        return cached.clone();
    }
    let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
    let root = std::process::Command::new("git")
        .arg("-C")
        .arg(path)
        .args(["rev-parse", "--show-toplevel"])
        .stderr(std::process::Stdio::null())
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| std::path::PathBuf::from(String::from_utf8_lossy(&o.stdout).trim()))
        .unwrap_or_else(|| path.to_path_buf());
    let name = (Some(&root) != home.as_ref() && root.parent().is_some())
        .then(|| {
            root.file_name()
                .and_then(|n| n.to_str())
                .map(str::to_lowercase)
        })
        .flatten();
    GIT_ROOTS.lock().unwrap().insert(key, name.clone());
    name
}

pub fn message_hash(message: &Value) -> String {
    hex::encode(&Sha256::digest(message.to_string().as_bytes())[..12])
}

/// Messages not yet recorded for this session, compacted and redacted for the transcript.
/// Returns (first index, messages). A shorter or rewritten history starts over.
pub fn new_messages(state: &mut State, protocol: Protocol, body: &Value) -> (usize, Vec<Value>) {
    let Some(items) = inject::messages(protocol, body) else {
        return (0, vec![]);
    };
    let consistent = state.seen <= items.len()
        && (state.seen == 0
            || state.last_hash.as_deref() == Some(&message_hash(&items[state.seen - 1])));
    if !consistent {
        state.seen = 0;
        state.anchors.clear();
        state.injected.clear();
    }
    let from = state.seen;
    let delta = items[from..].iter().map(compact_message).collect();
    state.seen = items.len();
    state.last_hash = items.last().map(message_hash);
    (from, delta)
}

fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_owned();
    }
    let head: String = text.chars().take(max * 2 / 3).collect();
    let tail: String = text
        .chars()
        .rev()
        .take(max / 3)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!(
        "{head}\n…[{} chars omitted]…\n{tail}",
        text.chars().count() - max
    )
}

/// One transcript entry: role, text (tool output clipped), and tool calls.
pub fn compact_message(message: &Value) -> Value {
    let role = message
        .get("role")
        .and_then(Value::as_str)
        .or_else(|| match message.get("type").and_then(Value::as_str) {
            Some("function_call_output") => Some("tool"),
            Some("function_call") => Some("assistant"),
            _ => None,
        })
        .unwrap_or("unknown");
    let mut calls = vec![];
    if let Some(items) = message.get("tool_calls").and_then(Value::as_array) {
        for call in items {
            calls.push(json!({"name":call.pointer("/function/name"),"args":clip(call.pointer("/function/arguments").and_then(Value::as_str).unwrap_or(""),400)}));
        }
    }
    if message.get("type").and_then(Value::as_str) == Some("function_call") {
        calls.push(json!({"name":message.get("name"),"args":clip(message.get("arguments").and_then(Value::as_str).unwrap_or(""),400)}));
    }
    let mut tool_result = message.get("type").and_then(Value::as_str)
        == Some("function_call_output")
        || role == "tool";
    if let Some(parts) = message.get("content").and_then(Value::as_array) {
        for part in parts {
            match part.get("type").and_then(Value::as_str) {
                Some("tool_use") => calls.push(json!({"name":part.get("name"),"args":clip(&part.get("input").map(Value::to_string).unwrap_or_default(),400)})),
                Some("tool_result") => tool_result = true,
                _ => {}
            }
        }
    }
    let text = inject::content_text(message);
    let text = if tool_result {
        clip(&text, 1500)
    } else {
        clip(&text, 6000)
    };
    let role = if tool_result && role == "user" {
        "tool"
    } else {
        role
    };
    let mut entry = json!({"role":role,"text":redact(&text)});
    if !calls.is_empty() {
        entry["calls"] = crate::store::redact_value(&json!(calls));
    }
    entry
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn detects_project_from_prompt() {
        let body = json!({"messages":[{"role":"system","content":"You are an agent. Your working directory is: /tmp/some-repo-xyz\nBe careful."},{"role":"user","content":"hi"}]});
        assert_eq!(
            detect_project(&HeaderMap::new(), None, Protocol::Chat, &body).as_deref(),
            Some("some-repo-xyz")
        );
        let codex = json!({"instructions":"x","input":[{"role":"user","content":"<environment_context><cwd>/tmp/other-proj</cwd></environment_context>"}]});
        assert_eq!(
            detect_project(&HeaderMap::new(), None, Protocol::Responses, &codex).as_deref(),
            Some("other-proj")
        );
        let mut headers = HeaderMap::new();
        headers.insert("x-ctx-project", "Payments".parse().unwrap());
        assert_eq!(
            detect_project(&headers, None, Protocol::Chat, &body).as_deref(),
            Some("payments")
        );
    }
    #[test]
    fn keys_are_stable_across_steps() {
        let first = json!({"messages":[{"role":"system","content":"s"},{"role":"user","content":"fix bug"}]});
        let later = json!({"messages":[{"role":"system","content":"s"},{"role":"user","content":"fix bug"},{"role":"assistant","content":"ok"},{"role":"user","content":"more"}]});
        let headers = HeaderMap::new();
        assert_eq!(
            key("a", &headers, Protocol::Chat, &first),
            key("a", &headers, Protocol::Chat, &later)
        );
        let claude =
            json!({"metadata":{"user_id":"user_x_account_y_session_abc-123"},"messages":[]});
        assert_eq!(
            key("cc", &headers, Protocol::Anthropic, &claude),
            "cc:abc-123"
        );
    }
    #[test]
    fn records_only_new_messages() {
        let mut state = State::default();
        let one = json!({"messages":[{"role":"user","content":"a"}]});
        let two = json!({"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"},{"role":"tool","content":"c"}]});
        assert_eq!(new_messages(&mut state, Protocol::Chat, &one).1.len(), 1);
        let (from, delta) = new_messages(&mut state, Protocol::Chat, &two);
        assert_eq!((from, delta.len()), (1, 2));
        assert_eq!(delta[1]["role"], "tool");
        let rewritten = json!({"messages":[{"role":"user","content":"z"}]});
        assert_eq!(new_messages(&mut state, Protocol::Chat, &rewritten).0, 0);
    }
}
