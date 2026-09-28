//! Maintenance: expire memories past `expires`, supersede exact duplicates, and disable
//! triggers that keep firing while labeled feedback says they are not used. Memories are
//! never archived just for being unused: long-term memory is the point.
use crate::{memory::Memory, store::Store};
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::{fs, process::Command};

fn mark_run(store: &Store) -> Result<()> {
    fs::write(
        store.root.join("state/maintenance-last-run"),
        Utc::now().to_rfc3339(),
    )?;
    Ok(())
}
pub fn should_run(store: &Store) -> bool {
    if let Ok(raw) = fs::read_to_string(store.root.join("state/maintenance-last-run"))
        && DateTime::parse_from_rfc3339(raw.trim())
            .ok()
            .is_some_and(|last| Utc::now().signed_duration_since(last).num_hours() < 24)
    {
        return false;
    }
    #[cfg(target_os = "macos")]
    {
        let power = Command::new("pmset").args(["-g", "batt"]).output().ok();
        if !power.is_some_and(|out| String::from_utf8_lossy(&out.stdout).contains("AC Power")) {
            return false;
        }
        let idle = Command::new("ioreg")
            .args(["-c", "IOHIDSystem"])
            .output()
            .ok();
        let Some(output) = idle else {
            return false;
        };
        let raw = String::from_utf8_lossy(&output.stdout);
        let nanos = regex::Regex::new(r"HIDIdleTime\s*=\s*(\d+)")
            .ok()
            .and_then(|pattern| pattern.captures(&raw))
            .and_then(|captures| captures.get(1).and_then(|m| m.as_str().parse::<u64>().ok()));
        nanos.is_some_and(|value| value >= 300_000_000_000)
    }
    #[cfg(not(target_os = "macos"))]
    false
}

pub fn run(store: &Store) -> Result<Value> {
    let all = store.memories();
    let now = Utc::now();
    let mut updates = BTreeMap::<String, Memory>::new();
    let mut reasons = Vec::new();
    for memory in &all {
        if !matches!(memory.status.as_str(), "active" | "contested") {
            continue;
        }
        let mut changed = memory.clone();
        if memory
            .expires
            .as_deref()
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .is_some_and(|date| date < now)
        {
            changed.status = "archived".into();
            reasons.push(format!("expire {}", memory.id));
        }
        for trigger in &mut changed.triggers {
            if trigger.disabled {
                continue;
            }
            let (fired, labeled, used) = store.trigger_usage(&trigger.id)?;
            if fired >= 10 && labeled >= 5 && (used as f64 / labeled as f64) < 0.1 {
                trigger.disabled = true;
                reasons.push(format!("disable trigger {}", trigger.id));
            }
        }
        if changed != *memory {
            changed.updated_at = now.to_rfc3339();
            updates.insert(changed.id.clone(), changed);
        }
    }
    let mut seen = BTreeMap::<(String, String, String), String>::new();
    for memory in &all {
        if memory.kind == "rule" || memory.status != "active" {
            continue;
        }
        let key = (
            memory.scope.clone(),
            memory.kind.clone(),
            memory
                .body
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .to_lowercase(),
        );
        if let Some(prior) = seen.get(&key) {
            let mut newer = updates.remove(&memory.id).unwrap_or_else(|| memory.clone());
            newer.status = "superseded".into();
            newer.superseded_by = Some(prior.clone());
            newer.updated_at = now.to_rfc3339();
            updates.insert(newer.id.clone(), newer);
            reasons.push(format!("deduplicate {}", memory.id));
        } else {
            seen.insert(key, memory.id.clone());
        }
    }
    if updates.is_empty() {
        mark_run(store)?;
        return Ok(json!({"changed":0,"batch":null}));
    }
    let changed = updates.len();
    let batch = store.apply_batch("daily", &updates.into_values().collect::<Vec<_>>())?;
    mark_run(store)?;
    Ok(json!({"changed":changed,"batch":batch,"reasons":reasons}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::NewMemory;
    #[test]
    fn expires_and_deduplicates() {
        let dir = tempfile::tempdir().unwrap();
        crate::store::init(dir.path()).unwrap();
        let store = Store::open(dir.path()).unwrap();
        let add = |content: &str| {
            store
                .remember(
                    NewMemory {
                        content: content.into(),
                        kind: "fact".into(),
                        scope: "global".into(),
                        title: None,
                        triggers: vec![],
                    },
                    "user",
                )
                .unwrap()
        };
        let first = add("Staging  is read-only on Fridays");
        let second = add("staging is read-only on fridays");
        let mut old = add("temporary freeze until release");
        old.expires = Some("2020-01-01T00:00:00Z".into());
        store.save_memory(&old, "set expiry").unwrap();
        let result = run(&store).unwrap();
        assert_eq!(result["changed"], 2);
        assert_eq!(store.memory(&second.id).unwrap().status, "superseded");
        assert_eq!(store.memory(&first.id).unwrap().status, "active");
        assert_eq!(store.memory(&old.id).unwrap().status, "archived");
        store
            .rollback_batch(result["batch"].as_str().unwrap())
            .unwrap();
        assert_eq!(store.memory(&second.id).unwrap().status, "active");
    }
}
