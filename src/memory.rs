//! Memory records: Markdown files with YAML front matter are the source of truth.
//!
//! Schema v2 keeps what recall, review and maintenance use. Files written by v1
//! (`card`, `tier`, `created_by`, trigger match DSL, ...) are read and converted.
use crate::store::{redact, redact_value};
use anyhow::{Result, bail};
use chrono::Utc;
use globset::Glob;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fs, path::Path};

pub const TYPES: [&str; 5] = ["rule", "fact", "lesson", "skill", "intent"];
pub const TRIGGER_KINDS: [&str; 4] = ["keyword", "error", "tool", "file"];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Memory {
    pub id: String,
    /// rule (always injected, user-authored), fact, lesson, skill, intent.
    #[serde(rename = "type")]
    pub kind: String,
    pub title: String,
    /// `global` or a project name.
    #[serde(default = "global")]
    pub scope: String,
    /// active, pending_review, contested, archived, superseded.
    #[serde(default = "active")]
    pub status: String,
    /// user (stated by the user), agent (inferred by an agent or the extractor),
    /// observed (seen in tool output).
    #[serde(default = "user")]
    pub source: String,
    /// Always injected like a rule (set by the user).
    #[serde(default, skip_serializing_if = "is_false")]
    pub pinned: bool,
    #[serde(default = "confidence")]
    pub confidence: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub triggers: Vec<Trigger>,
    /// RFC 3339 time after which the memory is no longer injected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supersedes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub superseded_by: Option<String>,
    /// Session keys or event ids the memory was learned from.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<String>,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub updated_at: String,
    /// When the fact was observed (the session's time), if different from creation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_at: Option<String>,
    /// Markdown body after the front matter.
    #[serde(skip)]
    pub body: String,
}

/// A deterministic recall condition, checked on every step.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Trigger {
    pub id: String,
    /// keyword: substring of the latest user message or tool-call arguments;
    /// error: substring of an error signature; tool: tool name; file: glob over touched paths.
    pub kind: String,
    pub pattern: String,
    #[serde(default, skip_serializing_if = "is_false")]
    pub disabled: bool,
    /// user, agent, repair, migrated.
    #[serde(default = "user")]
    pub origin: String,
    /// Checked before the matching tool call runs (experimental ActionGuard).
    #[serde(default, skip_serializing_if = "is_false")]
    pub before_action: bool,
}

fn global() -> String {
    "global".into()
}
fn active() -> String {
    "active".into()
}
fn user() -> String {
    "user".into()
}
fn confidence() -> f64 {
    0.85
}
fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Deserialize)]
pub struct NewMemory {
    pub content: String,
    #[serde(default = "fact", rename = "type")]
    pub kind: String,
    #[serde(default = "global")]
    pub scope: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub triggers: Vec<NewTrigger>,
}
fn fact() -> String {
    "fact".into()
}
#[derive(Deserialize, Clone)]
pub struct NewTrigger {
    pub kind: String,
    pub pattern: String,
    /// Check before the matching tool call runs (experimental ActionGuard).
    #[serde(default)]
    pub before_action: bool,
}

impl Memory {
    /// Always injected into the system prompt when in scope.
    /// A raw conversation excerpt presented as a search hit (never stored as a file).
    pub fn episode(id: &str, project: Option<&str>, text: String, observed_at: Option<String>) -> Self {
        Self {
            id: id.into(),
            kind: "episode".into(),
            title: "Conversation excerpt".into(),
            scope: project.unwrap_or("global").into(),
            status: "active".into(),
            source: "observed".into(),
            pinned: false,
            confidence: 1.0,
            triggers: vec![],
            expires: None,
            supersedes: vec![],
            superseded_by: None,
            evidence: vec![],
            created_at: String::new(),
            updated_at: String::new(),
            observed_at,
            body: text,
        }
    }
    pub fn always_on(&self) -> bool {
        self.kind == "rule" || self.pinned
    }
    pub fn in_scope(&self, project: Option<&str>) -> bool {
        self.scope == "global" || project.is_some_and(|p| p.eq_ignore_ascii_case(&self.scope))
    }
    pub fn recallable(&self) -> bool {
        matches!(self.status.as_str(), "active" | "contested")
            && self
                .expires
                .as_deref()
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                .is_none_or(|date| date >= Utc::now())
    }
    /// Text used for search: title, body, and keyword or error trigger patterns.
    pub fn search_text(&self) -> String {
        let mut text = if self.body.starts_with(&self.title) {
            self.body.clone()
        } else {
            format!("{}\n{}", self.title, self.body)
        };
        for trigger in self
            .triggers
            .iter()
            .filter(|t| !t.disabled && matches!(t.kind.as_str(), "keyword" | "error"))
        {
            text.push('\n');
            text.push_str(&trigger.pattern);
        }
        text
    }
}

pub fn new_trigger(kind: &str, pattern: &str, origin: &str) -> Result<Trigger> {
    let pattern = redact(pattern.trim());
    if !TRIGGER_KINDS.contains(&kind) {
        bail!("trigger kind must be one of {}", TRIGGER_KINDS.join(", "));
    }
    if pattern.chars().count() < 3 || pattern.len() > 200 {
        bail!("trigger pattern must be 3..200 characters");
    }
    if kind == "file" {
        Glob::new(&pattern)?;
    }
    Ok(Trigger {
        id: format!("trg_{}", ulid::Ulid::new()),
        kind: kind.into(),
        pattern,
        disabled: false,
        origin: origin.into(),
        before_action: false,
    })
}

/// Match a trigger against step features (`text`, `user_text`, `tool`, `args`, `error_sig`, `files`).
pub fn trigger_matches(trigger: &Trigger, features: &Value) -> bool {
    if trigger.disabled {
        return false;
    }
    let pattern = trigger.pattern.to_lowercase();
    let field = |key: &str| {
        features
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_lowercase()
    };
    match trigger.kind.as_str() {
        "keyword" => {
            let args = features
                .get("args")
                .filter(|v| !v.is_null())
                .map(Value::to_string)
                .unwrap_or_default()
                .to_lowercase();
            field("user_text").contains(&pattern) || args.contains(&pattern)
        }
        "error" => field("error_sig").contains(&pattern),
        "tool" => field("tool") == pattern,
        "file" => Glob::new(&trigger.pattern)
            .map(|g| g.compile_matcher())
            .is_ok_and(|matcher| {
                features
                    .get("files")
                    .and_then(Value::as_array)
                    .is_some_and(|files| {
                        files
                            .iter()
                            .filter_map(Value::as_str)
                            .any(|f| matcher.is_match(f))
                    })
            }),
        _ => false,
    }
}

pub fn valid_id(id: &str, prefix: &str) -> bool {
    id.starts_with(prefix)
        && id.len() == prefix.len() + 26
        && id[prefix.len()..]
            .chars()
            .all(|c| c.is_ascii_alphanumeric())
}

/// Build a new memory from user or agent input. `source` is user, agent or observed.
pub fn create(input: NewMemory, source: &str) -> Result<Memory> {
    let content = redact(input.content.trim());
    if content.is_empty() || content.len() > 20_000 {
        bail!("content must be 1..20000 bytes");
    }
    if !TYPES.contains(&input.kind.as_str()) {
        bail!("type must be one of {}", TYPES.join(", "));
    }
    let origin = if source == "user" { "user" } else { "agent" };
    let triggers = input
        .triggers
        .iter()
        .map(|t| {
            new_trigger(&t.kind, &t.pattern, origin).map(|mut trigger| {
                trigger.before_action = t.before_action;
                trigger
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let title = input
        .title
        .map(|t| redact(t.trim()))
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| default_title(&content));
    let now = Utc::now().to_rfc3339();
    // Only a user can make something always-on; agent rules and intents wait for review.
    let blocking = triggers.iter().any(|t| t.before_action);
    let status =
        if source != "user" && (blocking || matches!(input.kind.as_str(), "rule" | "intent")) {
            "pending_review"
        } else {
            "active"
        };
    Ok(Memory {
        id: format!("mem_{}", ulid::Ulid::new()),
        kind: input.kind,
        title,
        scope: normalize_scope(&input.scope),
        status: status.into(),
        source: source.into(),
        pinned: false,
        confidence: if source == "user" { 0.85 } else { 0.6 },
        triggers,
        expires: None,
        supersedes: vec![],
        superseded_by: None,
        evidence: vec![],
        created_at: now.clone(),
        updated_at: now,
        observed_at: None,
        body: content,
    })
}

pub fn normalize_scope(scope: &str) -> String {
    let scope = scope.trim();
    if scope.is_empty() || scope.eq_ignore_ascii_case("global") {
        "global".into()
    } else {
        scope.to_lowercase()
    }
}

/// First line, at most 60 characters.
pub fn default_title(content: &str) -> String {
    let line = content
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();
    let mut title: String = line.chars().take(60).collect();
    if line.chars().count() > 60 {
        title.push('…');
    }
    title
}

/// Read a memory file; returns the memory and whether it was converted from v1.
pub fn read(path: &Path) -> Result<(Memory, bool)> {
    let raw = fs::read_to_string(path)?;
    let Some(rest) = raw.strip_prefix("---\n") else {
        bail!("missing front matter")
    };
    let Some((yaml, body)) = rest.split_once("\n---\n") else {
        bail!("unterminated front matter")
    };
    let head: serde_yaml::Value = serde_yaml::from_str(yaml)?;
    let legacy = ["card", "tier", "created_by", "valid_to", "evidence_events"]
        .iter()
        .any(|key| head.get(*key).is_some())
        || head
            .get("triggers")
            .and_then(serde_yaml::Value::as_sequence)
            .is_some_and(|items| items.iter().any(|t| t.get("on").is_some()));
    let mut memory = if legacy {
        from_v1(&head)?
    } else {
        serde_yaml::from_value(head)?
    };
    memory.body = body.trim().into();
    if memory.body.is_empty() {
        bail!("empty memory body");
    }
    if memory.title.trim().is_empty() {
        memory.title = default_title(&memory.body);
    }
    Ok((memory, legacy))
}

pub fn write(path: &Path, memory: &Memory) -> Result<()> {
    let data = format!(
        "---\n{}---\n{}\n",
        serde_yaml::to_string(memory)?,
        memory.body
    );
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, data)?;
    fs::rename(tmp, path)?;
    Ok(())
}

fn from_v1(head: &serde_yaml::Value) -> Result<Memory> {
    let text = |key: &str| head.get(key).and_then(|v| v.as_str()).map(str::to_owned);
    let list = |key: &str| {
        head.get(key)
            .and_then(|v| v.as_sequence())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    };
    let Some(id) = text("id") else {
        bail!("missing id")
    };
    let kind = match text("type").as_deref() {
        Some("state") | Some("episode") | None => "fact".to_owned(),
        Some(other) => other.to_owned(),
    };
    let source = match text("source").as_deref() {
        Some("user_statement") | None => "user",
        Some("env_observation") | Some("tool_output") => "observed",
        _ => "agent",
    };
    let mut triggers = vec![];
    if let Some(items) = head.get("triggers").and_then(|v| v.as_sequence()) {
        for item in items {
            let json: Value = serde_json::to_value(item)?;
            triggers.extend(trigger_from_v1(&json));
        }
    }
    let title = text("title")
        .filter(|t| !t.trim().is_empty())
        .or_else(|| text("card"))
        .unwrap_or_default();
    Ok(Memory {
        id,
        kind: kind.clone(),
        title,
        scope: normalize_scope(&text("scope").unwrap_or_else(global)),
        status: text("status").unwrap_or_else(active),
        source: source.into(),
        pinned: text("tier").as_deref() == Some("pinned") && kind != "rule",
        confidence: head
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.85),
        triggers,
        expires: text("valid_to"),
        supersedes: list("supersedes"),
        superseded_by: text("superseded_by"),
        evidence: list("evidence_events"),
        created_at: text("created_at").unwrap_or_default(),
        updated_at: text("updated_at").unwrap_or_default(),
        observed_at: None,
        body: String::new(),
    })
}

/// Convert one v1 trigger (`on` + field match DSL) into at most one v2 trigger. v1
/// conditions were AND-ed; v2 keeps the most specific one.
fn trigger_from_v1(old: &Value) -> Option<Trigger> {
    let conditions = old.get("match")?.as_object()?;
    let value_of = |condition: &Value| -> Option<(String, String)> {
        let (op, value) = condition.as_object()?.iter().next()?;
        let value = match value {
            Value::String(s) => s.clone(),
            Value::Array(items) => items.first()?.as_str()?.to_owned(),
            other => other.to_string(),
        };
        Some((op.clone(), value))
    };
    let priority = ["error_sig", "text", "tool", "files", "file"];
    let (path, condition) = priority
        .iter()
        .find_map(|key| conditions.get(*key).map(|c| (key.to_string(), c)))
        .or_else(|| {
            conditions
                .iter()
                .find(|(k, _)| k.starts_with("args."))
                .map(|(k, c)| (k.clone(), c))
        })
        .or_else(|| conditions.iter().next().map(|(k, c)| (k.clone(), c)))?;
    let (op, pattern) = value_of(condition)?;
    let kind = match path.as_str() {
        "error_sig" => "error",
        "tool" => "tool",
        "files" | "file" => "file",
        _ => "keyword",
    };
    let pattern = if kind == "keyword" && op == "glob" {
        pattern.replace('*', "").trim().to_owned()
    } else {
        pattern
    };
    Some(Trigger {
        id: old
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("trg_{}", ulid::Ulid::new())),
        kind: kind.into(),
        disabled: old
            .get("disabled")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            || op == "regex"
            || pattern.chars().count() < 3,
        pattern: redact_value(&Value::String(pattern))
            .as_str()
            .unwrap_or("")
            .to_owned(),
        origin: "migrated".into(),
        before_action: old.get("when").and_then(Value::as_str) == Some("pre_action")
            && old.get("priority").and_then(Value::as_str) == Some("high"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn migrates_v1_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mem_01M3H6EY7F3P3KDHQJPPMJHP51.md");
        fs::write(
            &path,
            "---\nid: mem_01M3H6EY7F3P3KDHQJPPMJHP51\ntype: state\ntitle: ''\ncard: 部署前先迁移\nscope: Payments\ntier: pinned\nstatus: active\nconfidence: 0.85\nsource: env_observation\ntriggers:\n- id: trg_1\n  on: user_msg\n  match:\n    text:\n      contains: deploy\n  when: pre_action\n  priority: high\n  cooldown_steps: 20\n  origin: user\n  suspect: false\n  disabled: false\nbody: ''\ncreated_at: a\nupdated_at: b\npolarity: null\nfuture_use: null\nevidence_events: [evt_1]\nhalf_life_days: null\nverify_how: null\nvalid_to: null\nsupersedes: []\nsuperseded_by: null\nlinks: []\ncreated_by: user\n---\n部署前先执行数据库迁移\n",
        )
        .unwrap();
        let (memory, legacy) = read(&path).unwrap();
        assert!(legacy);
        assert_eq!(memory.kind, "fact");
        assert_eq!(memory.title, "部署前先迁移");
        assert_eq!(memory.scope, "payments");
        assert!(memory.pinned);
        assert_eq!(memory.source, "observed");
        assert_eq!(memory.evidence, vec!["evt_1"]);
        assert_eq!(memory.triggers[0].kind, "keyword");
        assert_eq!(memory.triggers[0].pattern, "deploy");
        assert!(memory.triggers[0].before_action);
        write(&path, &memory).unwrap();
        let (again, legacy) = read(&path).unwrap();
        assert!(!legacy);
        assert_eq!(again, memory);
        let text = fs::read_to_string(&path).unwrap();
        assert!(!text.contains("card:") && !text.contains("tier:"));
    }

    #[test]
    fn triggers_match_features() {
        let keyword = new_trigger("keyword", "deploy", "user").unwrap();
        assert!(trigger_matches(
            &keyword,
            &json!({"user_text":"Deploy payments now"})
        ));
        assert!(trigger_matches(
            &keyword,
            &json!({"args":{"command":"make deploy"}})
        ));
        assert!(!trigger_matches(&keyword, &json!({"text":"deploy logs"})));
        let error = new_trigger("error", "schema mismatch", "user").unwrap();
        assert!(trigger_matches(
            &error,
            &json!({"error_sig":"Error: Schema mismatch at <line>"})
        ));
        let file = new_trigger("file", "migrations/*.sql", "user").unwrap();
        assert!(trigger_matches(
            &file,
            &json!({"files":["migrations/001.sql"]})
        ));
        assert!(new_trigger("regex", "x+", "user").is_err());
    }

    #[test]
    fn agent_rules_need_review() {
        let rule = create(
            NewMemory {
                content: "Never force push".into(),
                kind: "rule".into(),
                scope: "global".into(),
                title: None,
                triggers: vec![],
            },
            "agent",
        )
        .unwrap();
        assert_eq!(rule.status, "pending_review");
        assert_eq!(rule.title, "Never force push");
    }
}
