//! Contradiction check: "I've never written a Flask route" and "I tested the homepage
//! route" cannot both be true. The extraction model rarely flags such pairs on its own, so
//! after each session the new memories that make a negative claim ("never", "haven't") are
//! compared with similar memories, and new memories with similar negative claims; one model
//! call judges the candidate pairs. Pairs that conflict are linked both ways and marked
//! contested: neither side is dropped, so a reader can say that the user said both.
use crate::{
    llm,
    memory::Memory,
    recall::{self, Query},
    store::Store,
};
use anyhow::Result;
use chrono::Utc;
use serde::Deserialize;
use serde_json::json;
use std::time::Duration;

const NEGATION_CUES: &[&str] = &[
    "never", "haven't", "have not", "hasn't", "has not", "not yet", "didn't", "did not",
    "don't have", "do not have", "no experience", "从未", "从没", "从来没", "未曾", "还没", "没有",
];
/// Similar memories compared with each new memory.
const NEIGHBOURS: usize = 4;
/// Pairs judged per session (one model call).
const MAX_PAIRS: usize = 12;

const PROMPT: &str = "You check a person's memory for contradictions. Each pair holds two statements recorded from the user's conversations, each with the date it was said. A pair CONFLICTS when both cannot be true about the same thing: one says the user never did, has not done or does not have something, and the other says they did or have it (\"I've never written Flask routes in this project\" vs \"the user tested the homepage route\"; \"never completed a coin toss problem\" vs \"completed 5 coin toss problems\").
It does NOT conflict when:
- the negative statement was said BEFORE the other one and the later one reports doing it for the first time (progress: \"never deployed\" in March, \"deployed to Render\" in May);
- the change is described as such (\"no longer\", \"switched to\", \"stopped\");
- they are about different things, projects or people.
Return JSON only: {\"conflicts\":[pair numbers]}. Do not wrap the JSON in Markdown.";

#[derive(Deserialize)]
struct Output {
    #[serde(default)]
    conflicts: Vec<usize>,
}

/// Whether a statement makes a negative claim (whole words for Latin text).
pub fn negative(text: &str) -> bool {
    let text = text.to_lowercase().replace('’', "'");
    NEGATION_CUES.iter().any(|cue| {
        if cue.chars().any(|c| c as u32 >= 0x2E80) {
            return text.contains(cue);
        }
        text.match_indices(cue).any(|(start, _)| {
            let before = text[..start].chars().next_back();
            let after = text[start + cue.len()..].chars().next();
            before.is_none_or(|c| !c.is_alphanumeric()) && after.is_none_or(|c| !c.is_alphanumeric())
        })
    })
}

fn eligible(memory: &Memory) -> bool {
    memory.recallable() && !matches!(memory.kind.as_str(), "rule" | "digest" | "reflection" | "episode")
}

/// When a statement was said (else when it was recorded).
fn said(memory: &Memory) -> String {
    memory
        .observed_at
        .clone()
        .unwrap_or_else(|| memory.created_at.chars().take(10).collect())
}

/// Candidate pairs for the new memories `created`: a new memory and a similar one where
/// at least one side makes a negative claim, most similar first, without repeats.
fn candidates(store: &Store, project: Option<&str>, created: &[String]) -> Result<Vec<(Memory, Memory)>> {
    let mut pairs: Vec<(f64, Memory, Memory)> = vec![];
    for id in created {
        let Some(new) = store.memory(id).filter(eligible) else { continue };
        let hits = recall::recall(
            store,
            &Query { text: Some(&new.body), project, limit: NEIGHBOURS + 1, mode: recall::Mode::Search, ..Default::default() },
        )?;
        for hit in hits.into_iter().filter(|h| h.memory.id != new.id && eligible(&h.memory)).take(NEIGHBOURS) {
            let other = hit.memory;
            if !(negative(&new.body) || negative(&other.body)) || new.conflicts_with.contains(&other.id) {
                continue;
            }
            if pairs.iter().any(|(_, a, b)| (a.id == other.id && b.id == new.id) || (a.id == new.id && b.id == other.id)) {
                continue;
            }
            pairs.push((hit.score, new.clone(), other));
        }
    }
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    pairs.truncate(MAX_PAIRS);
    // Earlier statement first in each pair.
    Ok(pairs
        .into_iter()
        .map(|(_, a, b)| if (said(&b), &b.created_at) < (said(&a), &a.created_at) { (b, a) } else { (a, b) })
        .collect())
}

/// Link the conflicting pairs among `created` and similar memories; returns the ids of
/// the memories marked contested.
pub async fn check(store: &Store, project: Option<&str>, created: &[String]) -> Result<Vec<String>> {
    let pairs = candidates(store, project, created)?;
    if pairs.is_empty() {
        return Ok(vec![]);
    }
    let input = pairs
        .iter()
        .enumerate()
        .map(|(i, (a, b))| {
            json!({"pair": i + 1, "first": {"said": said(a), "statement": a.body.chars().take(400).collect::<String>()},
                   "second": {"said": said(b), "statement": b.body.chars().take(400).collect::<String>()}})
        })
        .collect::<Vec<_>>();
    let reply = llm::chat(store, "contradict", PROMPT, &json!({"pairs": input}).to_string(), 400, Duration::from_secs(120)).await?;
    let output: Output = serde_json::from_str(llm::unfence(&reply)?)?;
    let mut contested = vec![];
    for number in output.conflicts {
        let Some((a, b)) = pairs.get(number.wrapping_sub(1)) else { continue };
        // Re-read: an earlier pair may have changed either side.
        let (Some(mut a), Some(mut b)) = (store.memory(&a.id), store.memory(&b.id)) else { continue };
        let (a_id, b_id) = (a.id.clone(), b.id.clone());
        for (one, other) in [(&mut a, b_id), (&mut b, a_id)] {
            if !one.conflicts_with.contains(&other) {
                one.conflicts_with.push(other);
            }
            if one.source != "user" {
                one.status = "contested".into();
            }
            one.updated_at = Utc::now().to_rfc3339();
        }
        store.save_memory(&a, &format!("{} conflicts with {}", a.id, b.id))?;
        store.save_memory(&b, &format!("{} conflicts with {}", b.id, a.id))?;
        contested.extend([a.id, b.id]);
    }
    contested.sort();
    contested.dedup();
    Ok(contested)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_claims_are_recognised() {
        assert!(negative("The user has never written any Flask routes in this project."));
        assert!(negative("I haven’t integrated Flask-Login yet"));
        assert!(negative("用户从未部署过这个应用"));
        assert!(!negative("The user completed 5 coin toss problems."));
        assert!(!negative("The user wrote a note about nevertheless"), "whole words only");
    }
}
