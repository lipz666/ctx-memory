//! Reflection: one model call consolidates what the memories of a topic mean together —
//! the user's stable preferences and habits in it, the current state of things that
//! changed, and counts or totals of repeated events (an event mentioned twice counted
//! once). It runs after extraction when a topic grows past a size threshold, at most one
//! call per session, and keeps one `reflection` memory per topic, rewritten each time.
use crate::{
    extract::digest_date,
    llm,
    memory::{self, Memory, NewMemory},
    store::Store,
};
use anyhow::Result;
use chrono::Utc;
use serde::Deserialize;
use serde_json::json;
use std::time::Duration;

/// A topic is reflected on when its size first reaches one of these.
const THRESHOLDS: &[usize] = &[5, 10, 20, 40, 80, 160, 320];
const MAX_TOPICS: usize = 5;
const MAX_MEMBERS: usize = 40;
const MEMBER_CHARS: usize = 220;
const MAX_CHARS: usize = 1200;

const PROMPT: &str = "You consolidate a person's long-term memory. For each topic you get its memories (each with a date) and your previous summary of it, if any. Write what the memories mean TOGETHER, for answering future questions about the topic as a whole:
- stable preferences, tastes and habits the memories show (what the user likes, avoids, usually does);
- the current state of anything that changed, with the date it changed (\"as of 2023-06 the user has 4 cats; 3 before\");
- for repeated events or items (workshops attended, things bought, times baked): the distinct occurrences with dates, their count and totals of amounts. Count an event mentioned in several memories once; do not count plans that did not happen.
Only state what the memories support. At most 6 sentences per topic, specific (names, numbers, dates). Return JSON only:
{\"reflections\":[{\"topic\":\"...\",\"content\":\"...\"}]}
Do not wrap the JSON in Markdown.";

#[derive(Deserialize)]
struct Output {
    #[serde(default)]
    reflections: Vec<Reflection>,
}
#[derive(Deserialize)]
struct Reflection {
    topic: String,
    content: String,
}

/// Memories a reflection summarizes: active, not digests or reflections themselves.
fn member(memory: &Memory) -> bool {
    memory.recallable() && !memory.derived()
}

/// The topics among `touched` memories that crossed a size threshold since their last
/// reflection, largest first: (scope, topic, member count).
fn due(store: &Store, touched: &[String]) -> Vec<(String, String, usize)> {
    let all = store.memories();
    let mut keys: Vec<(String, String)> = touched
        .iter()
        .filter_map(|id| store.memory(id))
        .flat_map(|m| m.topics.into_iter().map(move |t| (m.scope.clone(), t)))
        .collect();
    keys.sort();
    keys.dedup();
    let mut due: Vec<(String, String, usize)> = keys
        .into_iter()
        .filter_map(|(scope, topic)| {
            let count = all
                .iter()
                .filter(|m| member(m) && m.scope == scope && m.topics.contains(&topic))
                .count();
            let last = all
                .iter()
                .find(|m| m.kind == "reflection" && m.scope == scope && m.topics.first() == Some(&topic))
                .map_or(0, |m| m.evidence.len());
            THRESHOLDS
                .iter()
                .any(|&t| last < t && count >= t)
                .then_some((scope, topic, count))
        })
        .collect();
    due.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.1.cmp(&b.1)));
    due.truncate(MAX_TOPICS);
    due
}

/// Reflect on the topics these memories belong to, if any crossed a threshold. Returns
/// the reflection memories written.
pub async fn reflect(store: &Store, touched: &[String]) -> Result<Vec<String>> {
    let due = due(store, touched);
    if due.is_empty() {
        return Ok(vec![]);
    }
    let all = store.memories();
    let mut topics = vec![];
    let mut members_of = vec![];
    for (scope, topic, _) in &due {
        let mut members: Vec<&Memory> = all
            .iter()
            .filter(|m| member(m) && &m.scope == scope && m.topics.contains(topic))
            .collect();
        let digits = |m: &Memory| digest_date(m).chars().filter(char::is_ascii_digit).collect::<String>();
        members.sort_by_key(|m| (digits(m), m.created_at.clone()));
        let members = &members[members.len().saturating_sub(MAX_MEMBERS)..];
        let previous = all
            .iter()
            .find(|m| m.kind == "reflection" && &m.scope == scope && m.topics.first() == Some(topic))
            .map(|m| m.body.clone());
        topics.push(json!({
            "topic": topic,
            "previous_summary": previous,
            "memories": members.iter().map(|m| format!("[{}] {}", digest_date(m), m.body.chars().take(MEMBER_CHARS).collect::<String>())).collect::<Vec<_>>(),
        }));
        members_of.push(members.iter().map(|m| m.id.clone()).collect::<Vec<_>>());
    }
    let input = json!({"topics": topics}).to_string();
    let reply = llm::chat(store, "reflect", PROMPT, &input, 2500, Duration::from_secs(120)).await?;
    let output: Output = serde_json::from_str(llm::unfence(&reply)?)?;
    let mut written = vec![];
    for ((scope, topic, _), evidence) in due.iter().zip(members_of) {
        let Some(reflection) = output.reflections.iter().find(|r| r.topic.trim().eq_ignore_ascii_case(topic)) else {
            continue;
        };
        let content: String = crate::store::redact(reflection.content.trim()).chars().take(MAX_CHARS).collect();
        if content.chars().count() < 20 {
            continue;
        }
        let existing = store
            .memories()
            .into_iter()
            .find(|m| m.kind == "reflection" && &m.scope == scope && m.topics.first() == Some(topic));
        let mut memory = match existing {
            Some(existing) if existing.body == content => continue,
            Some(existing) => existing,
            None => memory::create(
                NewMemory {
                    content: content.clone(),
                    kind: "reflection".into(),
                    scope: scope.clone(),
                    title: Some(format!("Summary: {topic}")),
                    triggers: vec![],
                },
                "agent",
            )?,
        };
        memory.body = content;
        memory.source = "observed".into();
        memory.topics = vec![topic.clone()];
        memory.observed_at = evidence
            .iter()
            .filter_map(|id| store.memory(id).and_then(|m| m.observed_at))
            .max();
        memory.evidence = evidence;
        memory.updated_at = Utc::now().to_rfc3339();
        store.save_memory(&memory, &format!("reflect {topic} ({scope})"))?;
        written.push(memory.id);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::NewMemory;

    #[test]
    fn topics_are_due_when_they_cross_a_threshold() {
        let dir = tempfile::tempdir().unwrap();
        crate::store::init(dir.path()).unwrap();
        let mut config = crate::store::load_config(dir.path()).unwrap();
        config.embedding.enabled = false;
        crate::store::Store::save_config(dir.path(), &config).unwrap();
        let store = Store::open(dir.path()).unwrap();
        let add = |i: usize| {
            let mut m = store
                .remember(
                    NewMemory { content: format!("The user attended workshop {i}."), kind: "fact".into(), scope: "home".into(), title: None, triggers: vec![] },
                    "agent",
                )
                .unwrap();
            m.topics = vec!["workshops".into()];
            store.save_memory(&m, "topic").unwrap();
            m.id
        };
        let ids: Vec<String> = (0..4).map(add).collect();
        assert!(due(&store, &ids).is_empty(), "4 members: below the first threshold");
        let fifth = add(4);
        assert_eq!(due(&store, std::slice::from_ref(&fifth)), [("home".to_string(), "workshops".to_string(), 5)]);
        // A reflection covering the 5 members: not due again until 10.
        let mut reflection = memory::create(
            NewMemory { content: "Summary of workshops so far.".into(), kind: "reflection".into(), scope: "home".into(), title: None, triggers: vec![] },
            "agent",
        )
        .unwrap();
        reflection.topics = vec!["workshops".into()];
        reflection.evidence = ids.iter().chain([&fifth]).cloned().collect();
        store.save_memory(&reflection, "reflection").unwrap();
        let sixth = add(5);
        assert!(due(&store, &[sixth]).is_empty());
    }
}
