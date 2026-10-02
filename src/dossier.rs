//! Topic dossiers: what a person's subjects (a project, a plan, a goal, a relationship, a
//! hobby) are like now and how they went, kept up to date when conversations are stored
//! instead of pieced together from scattered memories for every question. Questions about
//! a subject over time — how many, the current value, how long between two events, how it
//! progressed, in what order — then read one place.
//!
//! After each session's extraction, one separate model call (a dossier output in the
//! extraction call itself cost extracted facts) returns records from the conversation:
//! - values: a quantity, setting, target or status as stated (a budget, a count, a version);
//! - items: distinct things of a kind the user mentioned, used, wanted or completed;
//! - events: what happened or is planned, with its date and the kind of date (planned,
//!   scheduled, deadline, started, done);
//! - stages: what the user worked on or asked, what the assistant recommended or explained,
//!   what was decided or achieved.
//!
//! Records carry the conversation turn they come from and are stored per session (a
//! session extracted again replaces its records). Each topic's records are rendered into
//! one `dossier` memory (search only, like topic digests): current values with the earlier
//! ones (the latest by turn), items per kind, events by date, stages by turn. Each stage
//! is also kept as a `narrative` memory of its own (search only): a few sentences that
//! keep what happened across several turns together (what was asked, recommended and
//! decided), found by search like any memory, where atomic facts each hold one detail
//! (`extraction.narratives`, off by default).
use crate::{
    extract::digest_numbered,
    llm,
    memory::{self, Memory, NewMemory},
    recall::Hit,
    store::{SessionRow, Store},
};
use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::HashMap, time::Duration};

/// Dossiers a brief reads besides its gathered material.
pub const BRIEF_DOSSIERS: usize = 2;
/// Records kept from one session.
const MAX_RECORDS: usize = 120;
/// Dossiers shown to the model as names to reuse, with their attributes and item kinds.
const KNOWN_TOPICS: usize = 30;
const KNOWN_LABELS: usize = 25;
/// Characters per dossier section: values, items, events, stages.
const SECTION_CHARS: [usize; 4] = [3_000, 3_000, 2_500, 6_000];
const LINE_MIN: usize = 120;

const PROMPT: &str = "You maintain topic dossiers in a person's long-term memory: for each subject they talk about with the assistant (a project, a plan, a goal, a relationship, a hobby, a health matter), what it is like now and how it went. You receive one conversation (user messages are numbered \"[user #n]\"), its date (\"session_date\") and the dossiers that already exist (\"known_topics\", with their value names and item kinds).

Return records from this conversation only:
- \"values\": each quantity, setting, target, budget, score, version, count or status of a subject as stated in this conversation, with the value exactly as given (\"$550 per month\", \"85%\", \"12 books by March 1\"). A value that changes gets a new record under the same name; when it takes effect on a date, give that date.
- \"items\": distinct things of one kind that the user mentioned, used, wanted, considered, chose, bought, completed or met (book series, tools, libraries, database columns, people, places, problems solved), under a kind name that says which relation it is (\"book series wanted\", \"libraries used\"). One record per item; keep names exact.
- \"events\": things that happened or are planned, with their absolute date computed from session_date and the kind of date: planned, scheduled, deadline, started, done or reported. A plan made for a later date gets the planned date, not the day it was planned.
- \"stages\": one to three per subject: what the user worked on, asked or reported in these turns, what the assistant recommended or explained (the specific steps, options, numbers and names), and what was decided or achieved. Two or three specific sentences each.
Use the broad subject as the topic (\"weather app\", \"wedding planning\", \"marathon training\"), the same one for all its records; reuse a name from known_topics when it fits, and the value names and item kinds already used there. Give each record the number n of the user message it comes from (for stages, n and n_to).

Return JSON only:
{\"values\":[{\"topic\":\"...\",\"name\":\"...\",\"value\":\"...\",\"effective\":\"YYYY-MM-DD\"|null,\"n\":1}],\"items\":[{\"topic\":\"...\",\"kind\":\"...\",\"item\":\"...\",\"n\":1}],\"events\":[{\"topic\":\"...\",\"event\":\"...\",\"date\":\"YYYY-MM-DD\"|null,\"date_kind\":\"planned\"|\"scheduled\"|\"deadline\"|\"started\"|\"done\"|\"reported\",\"n\":1}],\"stages\":[{\"topic\":\"...\",\"summary\":\"...\",\"n\":1,\"n_to\":2}]}
Do not wrap the JSON in Markdown.";

/// One record of a topic.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// value, item, event or stage.
    pub kind: String,
    /// The value's name, the item's kind; empty for events and stages.
    #[serde(default)]
    pub label: String,
    /// The value, the item, the event, the stage's account.
    pub text: String,
    /// When a value takes effect; an event's date.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub date: Option<String>,
    /// planned, scheduled, deadline, started, done or reported (events).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub date_kind: Option<String>,
    /// The conversation turn it comes from (0 when unknown) and, for a stage, the last one.
    #[serde(default)]
    pub turn: usize,
    #[serde(default)]
    pub last_turn: usize,
    /// When it was said (the session's date).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub said: Option<String>,
}

fn topic_name(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase().chars().take(60).collect()
}

fn scope_of(session: &SessionRow) -> String {
    session.project.clone().unwrap_or_else(|| "global".into())
}

/// The conversation turn number of each of the session's user messages (by position), from
/// the turns stored for the session, matched by the message's words.
fn turn_numbers(store: &Store, session: &SessionRow, user_texts: &[String]) -> Vec<usize> {
    let key = |text: &str| -> String { text.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(120).collect() };
    let numbers: HashMap<String, usize> = store.with_episodes(|e| {
        e.session_turns
            .get(&session.key)
            .into_iter()
            .flatten()
            .filter_map(|id| e.turn_meta.get(id).map(|t| (key(&t.text), t.number)))
            .collect()
    });
    user_texts.iter().map(|text| numbers.get(&key(text)).copied().unwrap_or(0)).collect()
}

/// The records in a model reply, by topic, with turn numbers and the session's date.
fn records(output: &Value, turns: &[usize], said: Option<&str>) -> Vec<(String, Record)> {
    let turn = |v: &Value, key: &str| -> usize {
        v.get(key).and_then(Value::as_u64).and_then(|n| turns.get((n as usize).wrapping_sub(1)).copied()).unwrap_or(0)
    };
    let text = |v: &Value, key: &str| -> String { v.get(key).and_then(Value::as_str).unwrap_or_default().trim().to_owned() };
    let date = |v: &Value, key: &str| -> Option<String> { v.get(key).and_then(Value::as_str).map(str::trim).filter(|d| d.len() >= 4).map(str::to_owned) };
    let mut out = vec![];
    for (section, kind) in [("values", "value"), ("items", "item"), ("events", "event"), ("stages", "stage")] {
        for v in output.get(section).and_then(Value::as_array).into_iter().flatten() {
            let topic = topic_name(&text(v, "topic"));
            let record = match kind {
                "value" => Record { kind: kind.into(), label: text(v, "name"), text: text(v, "value"), date: date(v, "effective"), ..Default::default() },
                "item" => Record { kind: kind.into(), label: text(v, "kind"), text: text(v, "item"), ..Default::default() },
                "event" => Record { kind: kind.into(), text: text(v, "event"), date: date(v, "date"), date_kind: date(v, "date_kind"), ..Default::default() },
                _ => Record { kind: kind.into(), text: text(v, "summary"), last_turn: turn(v, "n_to"), ..Default::default() },
            };
            if topic.is_empty() || record.text.is_empty() || (matches!(kind, "value" | "item") && record.label.is_empty()) {
                continue;
            }
            let record = Record { turn: turn(v, "n"), said: said.map(str::to_owned), ..record };
            out.push((topic, record));
            if out.len() == MAX_RECORDS {
                return out;
            }
        }
    }
    out
}

/// Update the dossiers with the records of `session` (one model call). Returns the ids of
/// the dossiers written.
pub async fn update(store: &Store, session: &SessionRow, entries: &[(String, String)], user_texts: &[String]) -> Result<Vec<String>> {
    let scope = scope_of(session);
    let known: Vec<Value> = store
        .dossier_topics(&scope)?
        .into_iter()
        .take(KNOWN_TOPICS)
        .map(|(topic, records)| {
            let mut names: Vec<String> = vec![];
            let mut kinds: Vec<String> = vec![];
            for r in records {
                let list = if r.kind == "value" { &mut names } else if r.kind == "item" { &mut kinds } else { continue };
                if !list.contains(&r.label) && list.len() < KNOWN_LABELS {
                    list.push(r.label);
                }
            }
            json!({"topic": topic, "value_names": names, "item_kinds": kinds})
        })
        .collect();
    let input = json!({
        "session_date": session.observed_at.as_deref().unwrap_or(&session.last_seen),
        "known_topics": known,
        "transcript": digest_numbered(entries),
    })
    .to_string();
    let reply = llm::chat(store, "dossier", PROMPT, &input, 8000, Duration::from_secs(180)).await?;
    let output: Value = serde_json::from_str(llm::unfence(&reply)?)?;
    let turns = turn_numbers(store, session, user_texts);
    let found = records(&output, &turns, session.observed_at.as_deref());
    store.set_dossier_records(&scope, &session.key, &found)?;
    let mut topics: Vec<String> = found.into_iter().map(|(topic, _)| topic).collect();
    topics.sort();
    topics.dedup();
    refresh(store, &scope, &topics)
}

/// Render the dossiers of `topics` in `scope` from their records and save the ones that
/// changed. Returns their ids.
pub fn refresh(store: &Store, scope: &str, topics: &[String]) -> Result<Vec<String>> {
    let all = store.memories();
    let mut written = vec![];
    for topic in topics {
        let records = store.dossier_records(scope, topic)?;
        if records.is_empty() {
            continue;
        }
        if store.config.extraction.narratives {
            written.extend(sync_narratives(store, &all, scope, topic, &records)?);
        }
        let body = render(topic, &records);
        let existing = all.iter().find(|m| m.kind == "dossier" && m.scope == scope && m.topics.first() == Some(topic));
        let mut dossier = match existing {
            Some(existing) if existing.body == body => continue,
            Some(existing) => existing.clone(),
            None => memory::create(
                NewMemory { content: body.clone(), kind: "dossier".into(), scope: scope.into(), title: Some(format!("Dossier: {topic}")), triggers: vec![] },
                "agent",
            )?,
        };
        dossier.body = body;
        dossier.source = "observed".into();
        dossier.topics = vec![topic.clone()];
        dossier.observed_at = records.iter().filter_map(|r| r.said.clone()).max();
        dossier.updated_at = Utc::now().to_rfc3339();
        store.save_memory(&dossier, &format!("dossier {topic} ({scope})"))?;
        written.push(dossier.id);
    }
    Ok(written)
}

/// A stage as a narrative memory's text.
fn narrative(topic: &str, record: &Record) -> String {
    let turns = match (record.turn, record.last_turn) {
        (0, _) => String::new(),
        (first, last) if last > first => format!(", turns #{first}–#{last}"),
        (first, _) => format!(", turn #{first}"),
    };
    format!("{} (on \"{topic}\"{turns})", record.text)
}

/// Keep one narrative memory per stage of `topic`: new stages are added, stages no longer
/// recorded (their session was extracted again) are archived. Returns the ids changed.
fn sync_narratives(store: &Store, all: &[Memory], scope: &str, topic: &str, records: &[Record]) -> Result<Vec<String>> {
    let wanted: Vec<(String, Option<String>)> =
        records.iter().filter(|r| r.kind == "stage").map(|r| (narrative(topic, r), r.said.clone())).collect();
    let existing: Vec<&Memory> = all
        .iter()
        .filter(|m| m.kind == "narrative" && m.scope == scope && m.recallable() && m.topics.first().map(String::as_str) == Some(topic))
        .collect();
    let mut changed = vec![];
    for (body, said) in &wanted {
        if existing.iter().any(|m| &m.body == body) {
            continue;
        }
        let mut memory = memory::create(
            NewMemory { content: body.clone(), kind: "narrative".into(), scope: scope.into(), title: Some(format!("Narrative: {topic}")), triggers: vec![] },
            "agent",
        )?;
        memory.source = "observed".into();
        memory.topics = vec![topic.to_owned()];
        memory.observed_at = said.clone();
        store.save_memory(&memory, &format!("narrative {topic} ({scope})"))?;
        changed.push(memory.id);
    }
    for memory in existing.into_iter().filter(|m| !wanted.iter().any(|(body, _)| body == &m.body)) {
        store.archive(&memory.id)?;
        changed.push(memory.id.clone());
    }
    Ok(changed)
}

/// Render every dossier again from the stored records (after a change of format).
pub fn rebuild(store: &Store) -> Result<usize> {
    let mut count = 0;
    for scope in store.dossier_scopes()? {
        let topics: Vec<String> = store.dossier_topics(&scope)?.into_iter().map(|(topic, _)| topic).collect();
        count += refresh(store, &scope, &topics)?.len();
    }
    Ok(count)
}

fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

fn at(record: &Record) -> String {
    let turn = if record.turn > 0 { format!("#{}", record.turn) } else { String::new() };
    match (&record.said, turn.is_empty()) {
        (Some(said), false) => format!("{turn}, said {said}"),
        (Some(said), true) => format!("said {said}"),
        (None, _) => turn,
    }
}

/// Lines within `chars`: each cut evenly when they do not all fit whole.
fn fit(lines: Vec<String>, chars: usize) -> Vec<String> {
    let total: usize = lines.iter().map(|l| l.chars().count() + 1).sum();
    if total <= chars || lines.is_empty() {
        return lines;
    }
    let per_line = (chars / lines.len()).max(LINE_MIN);
    let mut out = vec![];
    let mut used = 0;
    for line in lines {
        let mut cut: String = line.chars().take(per_line).collect();
        if cut.chars().count() < line.chars().count() {
            cut.push('…');
        }
        used += cut.chars().count() + 1;
        if used > chars {
            break;
        }
        out.push(cut);
    }
    out
}

/// A topic's dossier from its records.
pub fn render(topic: &str, records: &[Record]) -> String {
    let order = |r: &Record| (r.turn, r.said.clone().unwrap_or_default());
    let mut sorted: Vec<&Record> = records.iter().collect();
    sorted.sort_by_key(|r| order(r));
    // Values: by name, in order of first mention; the latest statement is current.
    let mut names: Vec<(String, Vec<&Record>)> = vec![];
    for r in sorted.iter().filter(|r| r.kind == "value") {
        let key = normalize(&r.label);
        match names.iter_mut().find(|(k, _)| *k == key) {
            Some((_, list)) => list.push(r),
            None => names.push((key, vec![r])),
        }
    }
    let values: Vec<String> = names
        .iter()
        .map(|(_, list)| {
            let current = list.last().unwrap();
            let from = current.date.as_ref().map(|d| format!(", from {d}")).unwrap_or_default();
            let mut line = format!("- {}: {} ({}{from})", current.label, current.text, at(current));
            let mut earlier: Vec<String> = vec![];
            for r in list.iter().rev().skip(1) {
                if normalize(&r.text) != normalize(&current.text) && !earlier.iter().any(|e| e.starts_with(&r.text)) {
                    earlier.push(format!("{} ({})", r.text, at(r)));
                }
            }
            if !earlier.is_empty() {
                line.push_str(&format!(" — earlier: {}", earlier.join("; ")));
            }
            line
        })
        .collect();
    // Items: by kind, each item once (its first mention).
    let mut kinds: Vec<(String, String, Vec<&Record>)> = vec![];
    for r in sorted.iter().filter(|r| r.kind == "item") {
        let key = normalize(&r.label);
        let index = match kinds.iter().position(|(k, _, _)| *k == key) {
            Some(index) => index,
            None => {
                kinds.push((key, r.label.clone(), vec![]));
                kinds.len() - 1
            }
        };
        let list = &mut kinds[index].2;
        if !list.iter().any(|seen| normalize(&seen.text) == normalize(&r.text)) {
            list.push(r);
        }
    }
    let items: Vec<String> = kinds
        .iter()
        .map(|(_, label, list)| {
            let names: Vec<String> = list.iter().map(|r| format!("{} ({})", r.text, at(r))).collect();
            format!("- {label} ({}): {}", list.len(), names.join("; "))
        })
        .collect();
    // Events: by date (undated ones last, by turn).
    let mut events: Vec<&Record> = sorted.iter().copied().filter(|r| r.kind == "event").collect();
    events.sort_by_key(|r| (r.date.is_none(), r.date.clone().unwrap_or_default(), r.turn));
    let events: Vec<String> = events
        .iter()
        .map(|r| {
            let date = r.date.as_deref().unwrap_or("no date");
            let date_kind = r.date_kind.as_deref().map(|k| format!(", {k}")).unwrap_or_default();
            format!("- {date}{date_kind}: {} ({})", r.text, at(r))
        })
        .collect();
    // Stages: by turn.
    let stages: Vec<String> = sorted
        .iter()
        .filter(|r| r.kind == "stage")
        .map(|r| {
            let span = match (r.turn, r.last_turn) {
                (0, _) => String::new(),
                (first, last) if last > first => format!("#{first}–#{last}"),
                (first, _) => format!("#{first}"),
            };
            let said = r.said.as_deref().map(|s| format!(" [{s}]")).unwrap_or_default();
            format!("- {span}{said}: {}", r.text)
        })
        .collect();
    let mut body = format!(
        "Dossier on \"{topic}\", from all conversations (turn numbers # give the order: a higher one is later):"
    );
    for (heading, lines, chars) in [
        ("Current values (with earlier ones)", values, SECTION_CHARS[0]),
        ("Items, by kind", items, SECTION_CHARS[1]),
        ("Dated events", events, SECTION_CHARS[2]),
        ("Stages, in order", stages, SECTION_CHARS[3]),
    ] {
        if !lines.is_empty() {
            body.push_str(&format!("\n{heading}:\n{}", fit(lines, chars).join("\n")));
        }
    }
    body
}

/// The dossiers in scope most relevant to `text`, best first (at most `count`).
pub fn relevant(store: &Store, text: &str, project: Option<&str>, count: usize) -> Result<Vec<Hit>> {
    let vector = match store.embedder.get() {
        Some(embedder) => Some(embedder.embed_query(text)?),
        None => None,
    };
    Ok(store.with_index(|index, all| {
        let allow = |id: &str| all.get(id).is_some_and(|m: &Memory| m.kind == "dossier" && m.recallable() && m.in_scope(project));
        let mut ranked = match vector.as_deref() {
            Some(v) => index.semantic_scores_where(v, allow),
            None => index.keyword_scores_where(text, allow),
        };
        ranked.truncate(count);
        ranked
            .into_iter()
            .filter_map(|(id, score)| all.get(&id).map(|m| (m.clone(), score)))
            .map(|(memory, score)| Hit { memory, channel: "dossier", score: score as f64, reason: "dossier".into(), group: None, turn: None })
            .collect()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(kind: &str, label: &str, text: &str, turn: usize) -> Record {
        Record { kind: kind.into(), label: label.into(), text: text.into(), turn, said: Some("2024/05/02".into()), ..Default::default() }
    }

    #[test]
    fn dossiers_show_the_latest_value_each_item_once_events_by_date_and_stages_in_order() {
        let records = vec![
            Record { date: Some("2024-09-15".into()), ..record("value", "gym membership", "$45 per month", 9) },
            record("value", "gym membership", "$40 per month", 3),
            record("item", "races finished", "Lisbon half marathon", 4),
            record("item", "Races finished", "lisbon  half marathon", 8),
            record("item", "races finished", "Porto 10K", 6),
            Record { date: Some("2024-06-12".into()), date_kind: Some("planned".into()), ..record("event", "", "physio check-up", 5) },
            Record { date: Some("2024-05-01".into()), date_kind: Some("done".into()), ..record("event", "", "first 20 km run", 7) },
            Record { last_turn: 6, ..record("stage", "", "Asked for a plan; the coach-style reply suggested 3 runs a week.", 4) },
            record("stage", "", "Reported the half marathon; decided to train for a full one.", 8),
        ];
        let body = render("marathon training", &records);
        assert!(body.contains("- gym membership: $45 per month (#9, said 2024/05/02, from 2024-09-15) — earlier: $40 per month (#3, said 2024/05/02)"), "{body}");
        assert!(body.contains("- races finished (2): Lisbon half marathon (#4, said 2024/05/02); Porto 10K (#6, said 2024/05/02)"), "{body}");
        let (done, planned) = (body.find("2024-05-01, done: first 20 km run").unwrap(), body.find("2024-06-12, planned: physio").unwrap());
        assert!(done < planned, "events by date");
        assert!(body.find("- #4–#6 [2024/05/02]: Asked").unwrap() < body.find("- #8 [2024/05/02]: Reported").unwrap());
    }

    #[test]
    fn records_take_turn_numbers_and_skip_incomplete_ones() {
        let output = json!({
            "values": [{"topic": " Marathon  Training", "name": "weekly distance", "value": "40 km", "effective": "2024-05-01", "n": 2}],
            "items": [{"topic": "marathon training", "kind": "", "item": "Porto 10K", "n": 1}],
            "events": [{"topic": "marathon training", "event": "race day", "date": "2024-10-06", "date_kind": "scheduled", "n": 9}],
            "stages": [{"topic": "marathon training", "summary": "Asked for a plan.", "n": 1, "n_to": 2}]
        });
        let found = records(&output, &[11, 12], Some("2024/05/02"));
        assert_eq!(found.len(), 3, "an item without a kind is dropped");
        assert_eq!(found[0].0, "marathon training");
        assert_eq!((found[0].1.turn, found[0].1.date.as_deref()), (12, Some("2024-05-01")));
        assert_eq!(found[1].1.turn, 0, "a message number out of range has no turn");
        assert_eq!((found[2].1.turn, found[2].1.last_turn), (11, 12));
    }

    #[test]
    fn records_are_stored_per_session_and_rendered_into_one_dossier_per_topic() {
        let dir = tempfile::tempdir().unwrap();
        crate::store::init(dir.path()).unwrap();
        let mut config = crate::store::load_config(dir.path()).unwrap();
        config.embedding.enabled = false;
        config.extraction.narratives = true;
        crate::store::Store::save_config(dir.path(), &config).unwrap();
        let store = Store::open(dir.path()).unwrap();
        let topic = "marathon training".to_string();
        store.set_dossier_records("q", "q:s1", &[(topic.clone(), record("value", "weekly distance", "30 km", 1))]).unwrap();
        store.set_dossier_records("q", "q:s2", &[(topic.clone(), record("value", "weekly distance", "40 km", 5))]).unwrap();
        let ids = refresh(&store, "q", std::slice::from_ref(&topic)).unwrap();
        let dossier = store.memory(&ids[0]).unwrap();
        // Stages become narrative memories of their own.
        store
            .set_dossier_records("q", "q:s3", &[(topic.clone(), Record { last_turn: 7, ..record("stage", "", "Asked for a plan; the reply suggested 3 runs a week.", 6) })])
            .unwrap();
        refresh(&store, "q", std::slice::from_ref(&topic)).unwrap();
        let narratives: Vec<Memory> = store.memories().into_iter().filter(|m| m.kind == "narrative" && m.recallable()).collect();
        assert_eq!(narratives.len(), 1);
        assert_eq!(narratives[0].body, "Asked for a plan; the reply suggested 3 runs a week. (on \"marathon training\", turns #6–#7)");
        store.set_dossier_records("q", "q:s3", &[]).unwrap();
        refresh(&store, "q", std::slice::from_ref(&topic)).unwrap();
        assert!(store.memories().into_iter().all(|m| m.kind != "narrative" || !m.recallable()), "a dropped stage is archived");
        assert_eq!((dossier.kind.as_str(), dossier.topics.as_slice()), ("dossier", std::slice::from_ref(&topic)));
        assert!(dossier.body.contains("weekly distance: 40 km (#5") && dossier.body.contains("earlier: 30 km (#1"));
        // A session extracted again replaces its records.
        store.set_dossier_records("q", "q:s2", &[(topic.clone(), record("value", "weekly distance", "45 km", 5))]).unwrap();
        assert_eq!(refresh(&store, "q", std::slice::from_ref(&topic)).unwrap(), ids, "the same dossier is rewritten");
        assert!(store.memory(&ids[0]).unwrap().body.contains("45 km (#5") && !store.memory(&ids[0]).unwrap().body.contains("40 km"));
        assert!(refresh(&store, "q", std::slice::from_ref(&topic)).unwrap().is_empty(), "unchanged: not saved again");
        assert_eq!(rebuild(&store).unwrap(), 0);
        let found = relevant(&store, "weekly distance marathon", Some("q"), 3).unwrap();
        assert_eq!(found.iter().map(|h| h.memory.id.clone()).collect::<Vec<_>>(), ids);
    }
}
