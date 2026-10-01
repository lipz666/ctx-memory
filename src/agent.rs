//! Memory agent: a brief written after looking further. A first model reads the same
//! material as a one-pass brief (`brief.rs`) and, when that is not enough or a detail has to
//! be checked (every item of a count, the date of each of two events, the latest value,
//! the stages of a summary, what exactly was recommended), looks things up in the memory —
//! search, the user's own messages, whole turns, the timeline, a day calculator — for at
//! most MAX_ROUNDS rounds. The brief is then written by the one-pass brief prompt, with
//! what was found added to the material: a writer that also looks things up lost the
//! brief's format (answers without the dates they rest on) and gave up on details the
//! material had (BEAM 100K: worse on dates and updates). Optional (`agent=true`).
//!
//! Lookups are plain text lines ("LOOKUP read_turns: #12, #45"), not JSON: a prompt that
//! asks for {"tool": ...} objects makes Gemini attempt a native function call, which ends
//! in `malformed_function_call` with an empty reply.
use crate::{
    brief,
    episode::split_turns,
    llm,
    recall::{self, Hit, Mode, Query},
    store::Store,
};
use anyhow::{Result, anyhow, bail};
use chrono::NaiveDate;
use regex::Regex;
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    sync::{Arc, LazyLock},
    time::Duration,
};

/// Rounds of tool calls before the brief must be written, and calls per round.
const MAX_ROUNDS: usize = 4;
const MAX_CALLS: usize = 4;
/// Characters of one tool result.
const RESULT_CHARS: usize = 12_000;
/// `read_turns`: turns per call and characters per turn.
const READ_TURNS: usize = 4;
const TURN_CHARS: usize = 4_000;
/// `user_messages`: messages by relevance, or at most this many from a range of turns.
const USER_MESSAGES: usize = 25;
const RANGE_TURNS: usize = 60;
const MESSAGE_CHARS: usize = 700;

const TOOLS: &str = "Your task now is only to decide what else to look up in the memory before the brief is written (by a separate step, from the material and what you find). Reply with nothing but lookup lines, at most 4, each of the form \"LOOKUP <kind>: <argument>\"; what they find comes back in the next message. When the material and your findings are enough, reply with the single word DONE. You can look up at most 4 times in a row.
Lookups (all within this user's memory):
- LOOKUP search: <words> — the memories and conversation excerpts most relevant to the words, with their dates and turn numbers.
- LOOKUP user_messages: <words> — the user's own messages most relevant to the words, in conversation order. LOOKUP user_messages: #N-#M — all of the user's messages from turn N to turn M.
- LOOKUP read_turns: #N, #M — the whole of up to 4 turns: the user's message and the assistant's full reply.
- LOOKUP timeline: — one line per past conversation session, in order, with its turn numbers.
- LOOKUP days_between: <date> | <date> — the exact number of days (and weeks) between two dates.
Look things up when the material is not enough or a detail has to be checked:
- a count or a total: find every candidate in the user's own messages (look up each kind of item; read the turns in doubt), keep the distinct items of the kind asked, then count;
- a date difference or a duration: find the date of each of the two events in the user's own words (read those turns), then use days_between;
- the latest value of something: find every statement of it and take the one with the highest turn number;
- a summary or how something progressed: go through the timeline and read the turns of each stage, the assistant's replies included;
- what the assistant recommended or explained: read that turn;
- a detail asked about something only mentioned in passing: search for it before concluding that the memory has nothing on it.
Do not look up what the material already settles; reply DONE at once when it settles everything.";

/// The brief for `question`, from `material` (already gathered as for a one-pass brief)
/// and what the writer looks up, in at most `tokens` tokens. Also returns the tool calls
/// made, for inspection.
pub async fn brief(
    store: Arc<Store>,
    question: &str,
    today: Option<&str>,
    project: Option<String>,
    material: &[Hit],
    tokens: usize,
) -> Result<(String, Vec<String>)> {
    let system = format!("{}\n\n{TOOLS}", brief::MATERIAL);
    let rendered: Vec<String> = material.iter().map(|hit| format!("- {}", brief::render(&store, hit))).collect();
    let input = json!({"question": question, "today": today, "material": rendered.join("\n")}).to_string();
    let mut messages = vec![json!({"role":"system","content":system}), json!({"role":"user","content":input})];
    let mut steps = vec![];
    let mut findings = String::new();
    for _ in 0..MAX_ROUNDS {
        let reply = llm::chat_messages(&store, "brief", &messages, 800, Duration::from_secs(180)).await?;
        let Some(wanted) = calls(&reply) else { break };
        let (store_, project_, question_) = (store.clone(), project.clone(), question.to_owned());
        let results = tokio::task::spawn_blocking(move || {
            wanted
                .iter()
                .map(|call| (describe(call), run(&store_, project_.as_deref(), &question_, call)))
                .collect::<Vec<_>>()
        })
        .await?;
        let mut text = String::from("Lookup results:");
        for (what, result) in results {
            let result = result.unwrap_or_else(|error| format!("error: {error:#}"));
            let found = format!("\n\n## {what}\n{}", result.chars().take(RESULT_CHARS).collect::<String>());
            text.push_str(&found);
            findings.push_str(&found);
            steps.push(what);
        }
        messages.push(json!({"role":"assistant","content":reply}));
        messages.push(json!({"role":"user","content":text}));
    }
    let written = brief::brief_with(&store, question, today, material, findings.trim(), tokens).await?;
    Ok((written, steps))
}

static LOOKUP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)^\W*lookup\s+([a-z_]+)\s*:?\s*(.*)$").unwrap());
static RANGE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^#?(\d+)\s*(?:-|–|to)\s*#?(\d+)$").unwrap());
static NUMBER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d+").unwrap());

/// The lookups in a reply, if it starts with one (at most MAX_CALLS), as
/// {"tool": ..., arguments} objects.
fn calls(reply: &str) -> Option<Vec<Value>> {
    let lines: Vec<&str> = reply.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with("```")).collect();
    if !lines.first().is_some_and(|l| LOOKUP.is_match(l)) {
        return None;
    }
    let calls: Vec<Value> = lines
        .iter()
        .filter_map(|line| LOOKUP.captures(line))
        .map(|c| lookup(&c[1].to_lowercase(), c[2].trim()))
        .take(MAX_CALLS)
        .collect();
    (!calls.is_empty()).then_some(calls)
}

/// One lookup line's kind and argument as a call.
fn lookup(kind: &str, argument: &str) -> Value {
    let argument = argument.trim_matches(|c: char| c == '"' || c == '`').trim();
    match kind {
        "user_messages" => match RANGE.captures(argument) {
            Some(c) => json!({"tool": kind, "from": c[1].parse::<u64>().unwrap_or(0), "to": c[2].parse::<u64>().unwrap_or(0)}),
            None => json!({"tool": kind, "query": argument}),
        },
        "read_turns" => {
            let turns: Vec<u64> = NUMBER.find_iter(argument).filter_map(|m| m.as_str().parse().ok()).collect();
            json!({"tool": kind, "turns": turns})
        }
        "days_between" => {
            let (from, to) = argument
                .split_once('|')
                .or_else(|| argument.split_once(" to "))
                .or_else(|| argument.split_once(" and "))
                .unwrap_or((argument, ""));
            json!({"tool": kind, "from": from.trim(), "to": to.trim()})
        }
        "timeline" => json!({"tool": kind}),
        _ => json!({"tool": kind, "query": argument}),
    }
}

/// A tool call in a few words, for the results and the trace.
fn describe(call: &Value) -> String {
    let tool = call.get("tool").and_then(Value::as_str).unwrap_or("?");
    let args: Vec<String> = call
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(key, _)| key.as_str() != "tool")
        .map(|(key, value)| format!("{key}={value}"))
        .collect();
    format!("{tool} {}", args.join(" ")).trim().to_owned()
}

fn text_arg<'a>(call: &'a Value, key: &str) -> Result<&'a str> {
    call.get(key).and_then(Value::as_str).filter(|s| !s.trim().is_empty()).ok_or_else(|| anyhow!("missing \"{key}\""))
}

/// Run one tool call within `project`.
fn run(store: &Store, project: Option<&str>, question: &str, call: &Value) -> Result<String> {
    match call.get("tool").and_then(Value::as_str).unwrap_or_default() {
        "search" => {
            let query = text_arg(call, "query")?;
            let hits = recall::recall(
                store,
                &Query { text: Some(query), project, limit: 12, mode: Mode::Search, episodes: 4, budget: Some(3_000), ..Default::default() },
            )?;
            let lines: Vec<String> = hits.iter().map(|hit| format!("- {}", brief::render(store, hit))).collect();
            Ok(if lines.is_empty() { "Nothing found.".into() } else { lines.join("\n") })
        }
        "user_messages" => user_messages(store, project, call),
        "read_turns" => {
            let wanted: Vec<usize> = call
                .get("turns")
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow!("missing \"turns\""))?
                .iter()
                .filter_map(Value::as_u64)
                .map(|n| n as usize)
                .take(READ_TURNS)
                .collect();
            read_turns(store, project, &wanted)
        }
        "timeline" => Ok(recall::timeline(store, question, project, RESULT_CHARS)?
            .map(|hit| hit.memory.body)
            .unwrap_or_else(|| "No timeline.".into())),
        "days_between" => days_between(text_arg(call, "from")?, text_arg(call, "to")?),
        other => bail!("unknown tool {other:?}"),
    }
}

fn message_line(turn: &crate::episode::Turn) -> String {
    let mut text: String = turn.text.chars().take(MESSAGE_CHARS).collect();
    if text.len() < turn.text.len() {
        text.push('…');
    }
    format!("- #{} [{}] {}", turn.number, turn.observed_at.as_deref().unwrap_or("date unknown"), text.replace('\n', " "))
}

/// The user's messages most relevant to a query, or those of a range of turns, in order.
fn user_messages(store: &Store, project: Option<&str>, call: &Value) -> Result<String> {
    let range = (call.get("from").and_then(Value::as_u64), call.get("to").and_then(Value::as_u64));
    let lines = if let (Some(from), Some(to)) = range {
        store.with_episodes(|e| {
            let mut turns: Vec<&crate::episode::Turn> = e
                .turn_meta
                .iter()
                .filter(|(id, t)| !t.text.is_empty() && e.turn_in_scope(id, project) && (from..=to).contains(&(t.number as u64)))
                .map(|(_, t)| t)
                .collect();
            turns.sort_by_key(|t| t.number);
            turns.into_iter().take(RANGE_TURNS).map(message_line).collect::<Vec<_>>()
        })
    } else {
        let query = text_arg(call, "query")?;
        let vector = match store.embedder.get() {
            Some(embedder) => Some(embedder.embed_query(query)?),
            None => None,
        };
        let ranked = recall::ranked_turns(store, query, vector.as_deref(), project);
        store.with_episodes(|e| {
            let mut turns: Vec<&crate::episode::Turn> =
                ranked.iter().filter_map(|(id, _)| e.turn_meta.get(id)).take(USER_MESSAGES).collect();
            turns.sort_by_key(|t| t.number);
            turns.into_iter().map(message_line).collect::<Vec<_>>()
        })
    };
    Ok(if lines.is_empty() { "No messages.".into() } else { lines.join("\n") })
}

/// Whole turns by number: the user's message and the assistant's full reply.
fn read_turns(store: &Store, project: Option<&str>, numbers: &[usize]) -> Result<String> {
    let wanted: HashSet<usize> = numbers.iter().copied().collect();
    let mut turns: Vec<(usize, Option<String>, Vec<String>)> = store.with_episodes(|e| {
        e.turn_meta
            .iter()
            .filter(|(id, t)| wanted.contains(&t.number) && e.turn_in_scope(id, project))
            .map(|(_, t)| (t.number, t.observed_at.clone(), t.excerpts.clone()))
            .collect()
    });
    turns.sort_by_key(|t| t.0);
    if turns.is_empty() {
        return Ok("No such turns.".into());
    }
    let mut out = vec![];
    for (number, date, ids) in turns {
        let mut excerpts = vec![];
        for id in ids {
            if let Some(text) = store.episode_text(&id)? {
                excerpts.push((id, text));
            }
        }
        let (user, reply) = split_turns(&excerpts)
            .into_iter()
            .next()
            .map(|(_, user, reply)| (user, reply))
            .unwrap_or_default();
        let mut text = format!("### Turn #{number} [{}]\nUser: {user}\nAssistant: {reply}", date.as_deref().unwrap_or("date unknown"));
        if text.chars().count() > TURN_CHARS {
            text = text.chars().take(TURN_CHARS).collect::<String>() + "…";
        }
        out.push(text);
    }
    Ok(out.join("\n\n"))
}

/// Days between two dates in any common form; a date without a year takes the other's.
fn days_between(from: &str, to: &str) -> Result<String> {
    let (Some(&(y1, m1, d1)), Some(&(y2, m2, d2))) = (brief::dates(from).first(), brief::dates(to).first()) else {
        bail!("could not read the dates (use YYYY-MM-DD)");
    };
    let (Some(y1), Some(y2)) = (y1.or(y2), y2.or(y1)) else { bail!("a year is needed") };
    let first = NaiveDate::from_ymd_opt(y1, m1, d1).ok_or_else(|| anyhow!("invalid date {from}"))?;
    let second = NaiveDate::from_ymd_opt(y2, m2, d2).ok_or_else(|| anyhow!("invalid date {to}"))?;
    let days = (second - first).num_days();
    let (weeks, rest) = (days.abs() / 7, days.abs() % 7);
    Ok(format!("{days} days from {first} to {second} ({weeks} weeks and {rest} days)"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn lookups_are_read_from_lines_only_at_the_start() {
        let reply = "LOOKUP search: grocery budget\nLOOKUP read_turns: #12, #45\nlookup user_messages: #3-#9\nLOOKUP days_between: March 12, 2024 | 2024-03-30";
        let found = calls(reply).unwrap();
        assert_eq!(found.len(), 4);
        assert_eq!(found[0], json!({"tool":"search","query":"grocery budget"}));
        assert_eq!(found[1], json!({"tool":"read_turns","turns":[12,45]}));
        assert_eq!(found[2], json!({"tool":"user_messages","from":3,"to":9}));
        assert_eq!(found[3], json!({"tool":"days_between","from":"March 12, 2024","to":"2024-03-30"}));
        assert_eq!(describe(&found[1]), "read_turns turns=[12,45]");
        assert_eq!(calls("LOOKUP timeline:").unwrap(), [json!({"tool":"timeline"})]);
        assert!(calls("Answer: 3 days, from May 1 to May 4, 2024.\nLOOKUP search: x").is_none(), "a brief is not a lookup");
        assert_eq!(calls(&"LOOKUP timeline:\n".repeat(9)).unwrap().len(), MAX_CALLS);
    }

    #[test]
    fn days_between_reads_common_forms() {
        assert_eq!(days_between("2024-07-10", "2024-09-12").unwrap(), "64 days from 2024-07-10 to 2024-09-12 (9 weeks and 1 days)");
        assert_eq!(days_between("April 2", "May 3, 2024").unwrap(), "31 days from 2024-04-02 to 2024-05-03 (4 weeks and 3 days)");
        assert!(days_between("soon", "2024-01-01").is_err());
    }

    #[test]
    fn tools_read_messages_turns_and_the_timeline_of_one_scope() {
        let dir = tempfile::tempdir().unwrap();
        crate::store::init(dir.path()).unwrap();
        let mut config = crate::store::load_config(dir.path()).unwrap();
        config.embedding.enabled = false;
        Store::save_config(dir.path(), &config).unwrap();
        let store = Store::open(dir.path()).unwrap();
        for (key, project, user, reply) in [
            ("q:s1", "q", "My grocery budget is $500 per month.", "Here is a plan:\n### Weekly envelopes\nSplit $500 into four."),
            ("q:s2", "q", "I raised the grocery budget to $550.", "Noted, $550 it is."),
            ("o:s1", "other", "My grocery budget is $900.", "Fine."),
        ] {
            let messages = [json!({"role":"user","content":user}), json!({"role":"assistant","content":reply})];
            store.ingest_session(key, "bench", Some(project), &messages, Some("2024/05/02")).unwrap();
            let session = store.session(key).unwrap().unwrap();
            let entries = crate::extract::transcript(&store, &session).unwrap().entries;
            store.add_episodes(&session, &entries).unwrap();
        }
        let run = |call: Value| run(&store, Some("q"), "grocery budget", &call).unwrap();
        let found = run(json!({"tool":"user_messages","query":"grocery budget"}));
        assert!(found.contains("#1 [2024/05/02] My grocery budget is $500") && found.contains("#2 [2024/05/02] I raised"), "{found}");
        assert!(!found.contains("$900"), "another scope stays out");
        assert_eq!(run(json!({"tool":"user_messages","from":2,"to":5})), "- #2 [2024/05/02] I raised the grocery budget to $550.");
        let turn = run(json!({"tool":"read_turns","turns":[1]}));
        assert!(turn.contains("### Turn #1") && turn.contains("Assistant: Here is a plan:\n### Weekly envelopes\nSplit $500 into four."), "{turn}");
        assert_eq!(run(json!({"tool":"read_turns","turns":[7]})), "No such turns.");
        assert!(run(json!({"tool":"days_between","from":"2024-05-02","to":"2024-05-30"})).starts_with("28 days"));
        assert!(super::run(&store, Some("q"), "x", &json!({"tool":"delete_everything"})).is_err());
    }
}
