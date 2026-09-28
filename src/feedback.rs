//! Recall feedback: citation inference from responses, and repairing a missed recall by
//! adding a trigger for the situation where the memory was needed.
use crate::{
    memory::{self, Trigger},
    store::Store,
};
use anyhow::Result;
use serde_json::Value;

/// Mark injected memories that the response cites by id or by their title.
pub fn infer_citations(store: &Store, step: &str, response: &str) -> Result<usize> {
    let mut count = 0;
    for id in store.recall_ids_for_step(step)? {
        let Some(memory) = store.memory(&id) else {
            continue;
        };
        let cited = response.contains(&id)
            || (memory.title.chars().count() >= 12 && response.contains(&memory.title));
        if cited {
            store.record_feedback(step, &id, Some(true), None, None)?;
            count += 1;
        }
    }
    Ok(count)
}

/// Too broad if it matches more than 5% of recent steps.
pub fn trigger_is_broad(store: &Store, trigger: &Trigger) -> Result<bool> {
    let recent = store.recent_features(2000)?;
    let matching = recent
        .iter()
        .filter(|features| memory::trigger_matches(trigger, features))
        .count();
    Ok(recent.len() >= 20 && (matching as f64 / recent.len() as f64) > 0.05)
}

/// Record that `memory_id` should have been recalled at `event`, and add an error or tool
/// trigger for that situation unless it would fire too often.
pub fn record_and_repair(
    store: &Store,
    event: &str,
    memory_id: &str,
    reason: &str,
) -> Result<String> {
    let id = store.record_miss(event, memory_id, reason)?;
    let Some(features) = store.event_features(event)? else {
        return Ok(id);
    };
    let Some(mut memory) = store.memory(memory_id) else {
        return Ok(id);
    };
    if !memory.recallable() {
        return Ok(id);
    }
    let candidate = if let Some(error) = features
        .get("error_sig")
        .and_then(Value::as_str)
        .filter(|s| s.len() >= 6 && s.len() <= 200)
    {
        memory::new_trigger("error", error, "repair")
    } else if let Some(tool) = features
        .get("tool")
        .and_then(Value::as_str)
        .filter(|s| s.len() >= 3)
    {
        memory::new_trigger("tool", tool, "repair")
    } else {
        return Ok(id);
    };
    let Ok(trigger) = candidate else {
        return Ok(id);
    };
    if memory
        .triggers
        .iter()
        .any(|old| old.kind == trigger.kind && old.pattern == trigger.pattern)
        || trigger_is_broad(store, &trigger)?
    {
        return Ok(id);
    }
    memory.triggers.push(trigger);
    memory.updated_at = chrono::Utc::now().to_rfc3339();
    store.save_memory(
        &memory,
        &format!("repair trigger for {} from {event}", memory.id),
    )?;
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::NewMemory;
    use serde_json::json;
    #[test]
    fn misses_repair_from_error_signature() {
        let dir = tempfile::tempdir().unwrap();
        crate::store::init(dir.path()).unwrap();
        let store = Store::open(dir.path()).unwrap();
        let memory = store
            .remember(
                NewMemory {
                    content: "Migrate before deploying payments".into(),
                    kind: "lesson".into(),
                    scope: "global".into(),
                    title: None,
                    triggers: vec![],
                },
                "user",
            )
            .unwrap();
        let event = store
            .event(
                "test",
                None,
                None,
                "request",
                &json!({"error_sig":"schema mismatch"}),
                &json!({}),
            )
            .unwrap();
        let miss = record_and_repair(&store, &event, &memory.id, "missed lesson").unwrap();
        assert!(miss.starts_with("miss_"));
        let repaired = store.memory(&memory.id).unwrap();
        assert_eq!(repaired.triggers[0].origin, "repair");
        assert_eq!(repaired.triggers[0].kind, "error");
    }
}
