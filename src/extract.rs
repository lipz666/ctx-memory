//! Session-end extraction: turn a finished conversation into durable memories.
//!
//! A session is extracted when it has been idle for `extraction.idle_minutes`, or right
//! away after a `task_end`/`session_end` hook. The model sees a compact transcript and
//! the related existing memories, and proposes creates or updates. Near-duplicates are
//! skipped, user-authored memories are never rewritten, and rules are never activated
//! automatically: a verified user preference is saved as a fact instead.
use crate::{
    embed, llm,
    memory::{self, Memory, NewMemory},
    recall::{self, Query},
    store::{SessionRow, Store, redact},
};
use anyhow::Result;
use chrono::Utc;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::HashMap, time::Duration};

const PROMPT: &str = include_str!("../prompts/extract-v1.txt");
/// For personal assistants: atomic facts and events about the user, validated on
/// LongMemEval_S (see docs/benchmark-results.md).
const GENERAL_PROMPT: &str = include_str!("../prompts/extract-general.txt");
const DIGEST_CHARS: usize = 30_000;
/// Cosine similarity above which a new memory is treated as a duplicate.
const DUPLICATE_SIMILARITY: f32 = 0.88;
/// Proposals applied from one session. Memories are atomic (one fact or event each), so a
/// detailed conversation legitimately yields dozens.
const MAX_PROPOSALS: usize = 40;

#[derive(Deserialize)]
struct Output {
    /// Parsed one by one: a malformed proposal is dropped, not the whole reply.
    #[serde(default)]
    memories: Vec<Value>,
    #[serde(default)]
    skip_reason: Option<String>,
    /// What the user brought up in this session and what the assistant gave, in order.
    #[serde(default)]
    session_summary: Option<String>,
    /// For each numbered user message, the gist of the assistant's reply.
    #[serde(default)]
    turns: Vec<Value>,
}
#[derive(Deserialize, Default)]
struct Proposal {
    #[serde(default = "create")]
    action: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default, rename = "type")]
    kind: String,
    #[serde(default)]
    scope: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    evidence: String,
    /// When the event happened (ISO date, possibly partial).
    #[serde(default)]
    event_date: Option<String>,
    #[serde(default)]
    topics: Vec<String>,
    #[serde(default)]
    entities: Vec<String>,
    #[serde(default)]
    triggers: Vec<ProposedTrigger>,
}
#[derive(Deserialize)]
struct ProposedTrigger {
    kind: String,
    pattern: String,
}
fn create() -> String {
    "create".into()
}

pub struct Transcript {
    pub digest: String,
    pub user_texts: Vec<String>,
    /// (role, text) in order, for the episodic excerpts.
    pub entries: Vec<(String, String)>,
}

/// Rebuild the session's conversation from recorded steps: every new message of each
/// request, then the final response.
pub fn transcript(store: &Store, session: &SessionRow) -> Result<Transcript> {
    let events = store.session_events(&session.key, session.extracted_at.as_deref())?;
    let mut entries: Vec<(String, String)> = vec![];
    let mut last_response: Option<(String, String)> = None;
    for (kind, _features, payload) in events {
        match kind.as_str() {
            "request" => {
                for message in payload
                    .get("messages")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    let role = message
                        .get("role")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown");
                    if role == "system" || role == "developer" {
                        continue;
                    }
                    let mut text = message
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_owned();
                    if let Some(calls) = message.get("calls").and_then(Value::as_array) {
                        for call in calls {
                            text.push_str(&format!(
                                "\n→ {}({})",
                                call.get("name").and_then(Value::as_str).unwrap_or("tool"),
                                call.get("args").and_then(Value::as_str).unwrap_or("")
                            ));
                        }
                    }
                    if !text.trim().is_empty() {
                        entries.push((role.to_owned(), text));
                    }
                }
                last_response = None;
            }
            "response" => {
                let text = payload.get("text").and_then(Value::as_str).unwrap_or("");
                if !text.trim().is_empty() {
                    last_response = Some(("assistant".into(), text.to_owned()));
                }
            }
            "hook" => {
                if let Some(result) = payload.pointer("/data/result").and_then(Value::as_str) {
                    entries.push(("event".into(), format!("task finished: {result}")));
                }
            }
            _ => {}
        }
    }
    entries.extend(last_response);
    let user_texts = entries
        .iter()
        .filter(|(role, _)| role == "user")
        .map(|(_, text)| text.clone())
        .collect();
    Ok(Transcript {
        digest: digest(&entries, store.config.extraction.turn_notes),
        user_texts,
        entries,
    })
}

/// Fit entries into the digest budget: user lines are kept, the middle of the rest is
/// dropped first.
fn digest(entries: &[(String, String)], numbered: bool) -> String {
    // With turn notes the user messages are numbered so that the model can refer to each.
    let mut number = 0;
    let lines: Vec<String> = entries
        .iter()
        .map(|(role, text)| {
            if role == "user" && numbered {
                number += 1;
                format!("[user #{number}] {}", text.trim())
            } else {
                format!("[{role}] {}", text.trim())
            }
        })
        .collect();
    let total: usize = lines.iter().map(|l| l.chars().count() + 1).sum();
    if total <= DIGEST_CHARS {
        return lines.join("\n");
    }
    let mut keep = vec![true; lines.len()];
    let mut size = total;
    let middle = lines.len() / 2;
    let mut order: Vec<usize> = (0..lines.len()).collect();
    order.sort_by_key(|&i| (i as isize - middle as isize).abs());
    for i in order {
        if size <= DIGEST_CHARS {
            break;
        }
        if !lines[i].starts_with("[user") && i + 1 != lines.len() {
            keep[i] = false;
            size -= lines[i].chars().count() + 1;
        }
    }
    let mut out = vec![];
    let mut skipped = 0;
    for (line, kept) in lines.iter().zip(keep) {
        if kept {
            if skipped > 0 {
                out.push(format!("…[{skipped} steps omitted]…"));
                skipped = 0;
            }
            out.push(line.chars().take(DIGEST_CHARS / 2).collect());
        } else {
            skipped += 1;
        }
    }
    out.join("\n")
}

/// Extract every session that is due. Errors are recorded per session.
pub async fn run_due(store: &Store) -> Result<Value> {
    if !store.config.extraction.enabled || store.config.model.is_none() {
        return Ok(json!({"sessions":0,"reason":"extraction disabled or no model"}));
    }
    let idle = Duration::from_secs(store.config.extraction.idle_minutes * 60);
    let mut results = vec![];
    for session in store.sessions_due(idle)? {
        results.push(match extract_session(store, &session).await {
            Ok(result) => result,
            Err(error) => {
                store.finish_session(&session.key, "failed", Some(&format!("{error:#}")))?;
                json!({"session":session.key,"error":redact(&format!("{error:#}"))})
            }
        });
    }
    Ok(json!({"sessions":results.len(),"results":results}))
}

/// Build conversation excerpts for every recorded session (sessions from before the
/// episodic tier, or after a model change). Excerpts that already exist are kept once.
/// Rebuild every topic digest (after a change to how digests are built).
pub fn rebuild_digests(store: &Store) -> Result<usize> {
    let ids: Vec<String> = store
        .memories()
        .into_iter()
        .filter(|m| m.recallable() && !m.derived() && !m.topics.is_empty())
        .map(|m| m.id)
        .collect();
    Ok(refresh_digests(store, ids.iter())?.len())
}

pub fn backfill_episodes(store: &Store) -> Result<usize> {
    let sessions = store.sessions(usize::MAX)?;
    let workers = store.config.embedding.workers.max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| -> Result<usize> {
                    let mut added = 0;
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(session) = sessions.get(i) else {
                            return Ok(added);
                        };
                        let whole = SessionRow {
                            extracted_at: None,
                            ..session.clone()
                        };
                        added += store.add_episodes(session, &transcript(store, &whole)?.entries)?;
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("backfill worker panicked"))
            .sum()
    })
}

pub async fn extract_session(store: &Store, session: &SessionRow) -> Result<Value> {
    let transcript = transcript(store, session)?;
    // The raw conversation is kept even when distillation finds nothing or fails.
    let episodes = store.add_episodes(session, &transcript.entries)?;
    if transcript.user_texts.is_empty() || transcript.digest.chars().count() < 40 {
        store.finish_session(&session.key, "skipped", Some("no user messages"))?;
        return Ok(json!({"session":session.key,"skipped":"no user messages"}));
    }
    let project = session.project.as_deref();
    // Existing memories related to each user message (search mode, not the strict
    // injection cutoff): the model must see a fact that changed to supersede it, and an
    // event that is mentioned again to avoid recording it twice.
    let mut related: HashMap<String, recall::Hit> = HashMap::new();
    for text in transcript.user_texts.iter().take(12) {
        let text: String = text.chars().take(1000).collect();
        for hit in recall::recall(
            store,
            &Query {
                text: Some(&text),
                project,
                limit: 6,
                mode: recall::Mode::Search,
                ..Default::default()
            },
        )? {
            if !hit.memory.derived()
                && related.get(&hit.memory.id).is_none_or(|h| h.score < hit.score)
            {
                related.insert(hit.memory.id.clone(), hit);
            }
        }
    }
    let mut related: Vec<recall::Hit> = related.into_values().collect();
    related.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.memory.id.cmp(&b.memory.id)));
    related.truncate(30);
    let existing: Vec<Value> = related
        .iter()
        .map(|h| json!({"id":h.memory.id,"type":h.memory.kind,"scope":h.memory.scope,"title":h.memory.title,"content":h.memory.body.chars().take(400).collect::<String>(),"event_date":h.memory.event_at,"topics":h.memory.topics}))
        .collect();
    let known_topics: Vec<String> = {
        let mut topics: Vec<String> = store
            .memories()
            .into_iter()
            .filter(|m| m.recallable() && m.in_scope(project))
            .flat_map(|m| m.topics)
            .collect();
        topics.sort();
        topics.dedup();
        topics.truncate(80);
        topics
    };
    let input = json!({
        "project": project,
        "session_date": session.observed_at.as_deref().unwrap_or(&session.last_seen),
        "existing_memories": existing,
        "known_topics": known_topics,
        "known_entities": if store.config.extraction.known_entities {
            known_entities(store, project)
        } else {
            vec![]
        },
        "transcript": transcript.digest,
    })
    .to_string();
    let prompt = match &store.config.extraction.prompt_file {
        Some(path) => std::fs::read_to_string(path)?,
        None if store
            .config
            .extraction
            .general_agents
            .iter()
            .any(|a| a.eq_ignore_ascii_case(&session.agent)) =>
        {
            GENERAL_PROMPT.to_owned()
        }
        None => PROMPT.to_owned(),
    };
    let prompt = if store.config.extraction.turn_notes {
        format!("{prompt}\n\n{TURNS_PROMPT}")
    } else {
        prompt
    };
    let mut output = None;
    let mut last_error = String::new();
    for attempt in 0..2 {
        let system = if attempt == 0 {
            prompt.clone()
        } else {
            format!(
                "{prompt}\nYour previous reply was not valid JSON of the required shape. Reply with the JSON object only."
            )
        };
        let reply = llm::chat(
            store,
            "extract",
            &system,
            &input,
            8000,
            Duration::from_secs(180),
        )
        .await?;
        match llm::unfence(&reply).and_then(|j| Ok(serde_json::from_str::<Output>(j)?)) {
            Ok(value) => {
                output = Some(value);
                break;
            }
            Err(error) => last_error = error.to_string(),
        }
    }
    let Some(output) = output else {
        anyhow::bail!("invalid extraction reply: {last_error}");
    };
    let result = apply(
        store,
        session,
        proposals(output.memories),
        &transcript.user_texts,
        &transcript.digest,
        &input,
    )?;
    let digests = refresh_digests(store, result.0.iter().chain(&result.1))?;
    let summary = match output.session_summary.as_deref().map(str::trim) {
        Some(text) if text.chars().count() >= 20 => Some(save_summary(store, session, text)?),
        _ => None,
    };
    let notes = save_turn_notes(store, session, &transcript.user_texts, &output.turns)?;
    // A failed check never fails the extraction.
    let contested = if store.config.extraction.contradictions {
        crate::contradict::check(store, project, &result.0)
            .await
            .unwrap_or_else(|error| vec![format!("failed: {error:#}")])
    } else {
        vec![]
    };
    // A failed reflection never fails the extraction; the next session retries it.
    let touched: Vec<String> = result.0.iter().chain(&result.1).cloned().collect();
    let reflections = if store.config.extraction.reflection {
        crate::reflect::reflect(store, &touched)
            .await
            .unwrap_or_else(|error| vec![format!("failed: {error:#}")])
    } else {
        vec![]
    };
    store.finish_session(&session.key, "done", None)?;
    Ok(
        json!({"session":session.key,"created":result.0,"updated":result.1,"skipped":result.2,"skip_reason":output.skip_reason,"episodes":episodes,"digests":digests,"summary":summary,"turn_notes":notes,"contested":contested,"reflections":reflections}),
    )
}

/// Record that `session` brought the memory `id` up again (once per session); false when
/// the memory does not exist, is not a statement of its own or this session created it.
fn record_mention(store: &Store, id: &str, session: &SessionRow) -> Result<bool> {
    let Some(mut memory) = store.memory(id).filter(|m| m.recallable() && !m.derived()) else {
        return Ok(false);
    };
    if memory.evidence.contains(&session.key) {
        return Ok(false);
    }
    memory.evidence.push(session.key.clone());
    memory
        .mentioned_at
        .push(session.observed_at.clone().unwrap_or_else(|| Utc::now().format("%Y-%m-%d").to_string()));
    store.save_memory(&memory, &format!("{id} mentioned again in {}", session.key))?;
    Ok(true)
}

fn proposals(values: Vec<Value>) -> Vec<Proposal> {
    values.into_iter().filter_map(|value| serde_json::from_value(value).ok()).collect()
}

/// Added to the extraction prompt when `extraction.turn_notes` is on.
const TURNS_PROMPT: &str = "Also return \"turns\": for each user message, numbered \"[user #n]\" in the transcript, {\"n\": n, \"assistant\": \"...\"} with what the assistant answered in one to three sentences and its key specifics (the steps, recommendations, options, numbers, names and code choices it gave), so that a later question about what was recommended or worked out can be answered without the full reply. Leave out user messages the assistant did not answer. Add it to the JSON as \"turns\":[{\"n\":1,\"assistant\":\"...\"}].";

/// Characters kept of the gist of one reply.
const TURN_NOTE_CHARS: usize = 600;

/// Attach the model's gist of each reply ({"n": user message number, "assistant": ...})
/// to the session's turn holding that user message (matched by its text, so a merged or
/// skipped turn never shifts the rest). Returns how many were stored.
pub(crate) fn save_turn_notes(store: &Store, session: &SessionRow, user_texts: &[String], turns: &[Value]) -> Result<usize> {
    let key = |text: &str| -> String { text.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(120).collect() };
    let ids: HashMap<String, String> = store.with_episodes(|e| {
        e.session_turns
            .get(&session.key)
            .into_iter()
            .flatten()
            .filter_map(|id| e.turn_meta.get(id).map(|t| (key(&t.text), id.clone())))
            .collect()
    });
    let mut notes = vec![];
    for turn in turns {
        let (Some(number), Some(note)) = (turn.get("n").and_then(Value::as_u64), turn.get("assistant").and_then(Value::as_str)) else {
            continue;
        };
        let note = redact(note.trim());
        let Some(text) = user_texts.get((number as usize).wrapping_sub(1)) else { continue };
        if note.chars().count() < 10 {
            continue;
        }
        if let Some(id) = ids.get(&key(text)) {
            notes.push((id.clone(), note.chars().take(TURN_NOTE_CHARS).collect()));
        }
    }
    store.set_turn_notes(&session.key, &notes)
}

/// Characters kept of a session summary.
const SUMMARY_CHARS: usize = 700;

/// The session's summary: one `summary` memory per session (replaced when the session is
/// extracted again), in the order sessions were extracted, for the conversation timeline.
fn save_summary(store: &Store, session: &SessionRow, text: &str) -> Result<String> {
    let body: String = redact(text).chars().take(SUMMARY_CHARS).collect();
    let existing = store
        .memories()
        .into_iter()
        .find(|m| m.kind == "summary" && m.evidence.contains(&session.key));
    let mut memory = match existing {
        Some(mut memory) => {
            memory.body = body;
            memory.updated_at = Utc::now().to_rfc3339();
            memory
        }
        None => memory::create(
            NewMemory {
                content: body,
                kind: "summary".into(),
                scope: session.project.clone().unwrap_or_else(|| "global".into()),
                title: Some("Session summary".into()),
                triggers: vec![],
            },
            "agent",
        )?,
    };
    memory.observed_at = session.observed_at.clone();
    if !memory.evidence.contains(&session.key) {
        memory.evidence.push(session.key.clone());
    }
    store.save_memory(&memory, &format!("summary of {}", session.key))?;
    Ok(memory.id)
}

fn normalize(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn apply(
    store: &Store,
    session: &SessionRow,
    proposals: Vec<Proposal>,
    user_texts: &[String],
    transcript: &str,
    grounding: &str,
) -> Result<(Vec<String>, Vec<String>, Vec<String>)> {
    let transcript = transcript.to_lowercase();
    // Names may come from the conversation or from what the model was shown (known
    // entities, related memories), which is how "my sister" becomes "Mira".
    let grounding = grounding.to_lowercase();
    let (mut created, mut updated, mut skipped) = (vec![], vec![], vec![]);
    let users = normalize(&user_texts.join("\n"));
    let embedder = store.embedder.get();
    for proposal in proposals.into_iter().take(MAX_PROPOSALS) {
        // Something already remembered came up again: count the mention.
        if proposal.action == "mention" {
            match proposal.id.as_deref() {
                Some(id) if record_mention(store, id, session)? => updated.push(id.to_owned()),
                _ => skipped.push(format!("mention: {}", proposal.id.unwrap_or_default())),
            }
            continue;
        }
        let content = redact(proposal.content.trim());
        if !["fact", "preference", "instruction", "lesson", "skill", "rule"].contains(&proposal.kind.as_str())
            || content.chars().count() < 10
            || content.len() > 4000
        {
            skipped.push(format!("invalid: {}", proposal.title));
            continue;
        }
        let scope = if proposal.scope == "project" || !store.config.extraction.global_scope {
            session.project.clone().unwrap_or_else(|| "global".into())
        } else {
            "global".into()
        };
        // Rules are never activated automatically; a quoted user preference becomes a fact.
        let mut kind = proposal.kind.clone();
        if kind == "rule" {
            let quote = normalize(&proposal.evidence);
            if quote.chars().count() < 4 || !users.contains(&quote) {
                skipped.push(format!("unverified rule: {}", proposal.title));
                continue;
            }
            kind = "fact".into();
        }
        let title = if proposal.title.trim().is_empty() {
            memory::default_title(&content)
        } else {
            redact(proposal.title.trim()).chars().take(80).collect()
        };
        let event_at = proposal.event_date.as_deref().and_then(normalize_date);
        let topics = normalize_topics(&proposal.topics);
        let entities: Vec<String> = normalize_entities(&proposal.entities)
            .into_iter()
            .filter(|e| grounding.contains(&e.to_lowercase()))
            .collect();
        let triggers = grounded_triggers(&proposal.triggers, &transcript);
        if proposal.action == "update"
            && let Some(id) = proposal.id.as_deref()
            && let Some(mut existing) = store.memory(id)
        {
            if existing.source == "user" || !existing.recallable() || existing.kind == "rule" {
                skipped.push(format!("protected: {id}"));
                continue;
            }
            if normalize(&existing.body) == normalize(&content) {
                skipped.push(format!("unchanged: {id}"));
                continue;
            }
            existing.body = content;
            existing.title = title;
            existing.updated_at = Utc::now().to_rfc3339();
            existing.observed_at = session.observed_at.clone().or(existing.observed_at);
            existing.event_at = event_at.or(existing.event_at);
            for topic in topics {
                if !existing.topics.contains(&topic) {
                    existing.topics.push(topic);
                }
            }
            for entity in entities {
                if !existing.entities.contains(&entity) {
                    existing.entities.push(entity);
                }
            }
            for new in &triggers {
                if !existing.triggers.iter().any(|t| t.kind == new.kind && t.pattern == new.pattern)
                    && let Ok(trigger) = memory::new_trigger(&new.kind, &new.pattern, "agent")
                {
                    existing.triggers.push(trigger);
                }
            }
            if !existing.evidence.contains(&session.key) {
                existing.evidence.push(session.key.clone());
            }
            store.save_memory(
                &existing,
                &format!("extract update {id} from {}", session.key),
            )?;
            updated.push(existing.id);
            continue;
        }
        // A statement that contradicts a memory without describing a change: keep both,
        // contested and linked, so a reader sees that the user said both.
        // So does a "never did X" said after "did X": that denies the earlier statement
        // rather than updating it, whatever the model called it.
        let denies = |old: &Memory| crate::contradict::negative(&content) && !crate::contradict::negative(&old.body);
        let contradicted = match (proposal.action.as_str(), proposal.id.as_deref()) {
            ("contradict", Some(id)) => store.memory(id).filter(|old| {
                old.recallable() && !matches!(old.kind.as_str(), "rule" | "digest" | "reflection" | "summary")
            }),
            ("supersede", Some(id)) => store.memory(id).filter(|old| {
                old.recallable() && !matches!(old.kind.as_str(), "rule" | "digest" | "reflection" | "summary") && denies(old)
            }),
            _ => None,
        };
        // A changed fact: the new memory replaces the old one, which stays as history.
        let replaced = match (proposal.action.as_str(), proposal.id.as_deref()) {
            ("supersede", Some(id)) if contradicted.is_none() => store.memory(id).filter(|old| {
                old.recallable()
                    && old.source != "user"
                    && !matches!(old.kind.as_str(), "rule" | "digest" | "reflection" | "summary")
            }),
            _ => None,
        };
        if replaced.is_none()
            && contradicted.is_none()
            && let Some(duplicate) = near_duplicate(
                store,
                embedder.as_deref(),
                &scope,
                &title,
                &content,
                event_at.as_deref(),
            )?
        {
            record_mention(store, &duplicate, session)?;
            skipped.push(format!("duplicate of {duplicate}"));
            continue;
        }
        let mut memory = memory::create(
            NewMemory {
                content,
                kind,
                scope,
                title: Some(title),
                triggers,
            },
            "agent",
        )?;
        if proposal.kind == "rule" {
            memory.source = "user".into();
            memory.confidence = 0.8;
        }
        memory.evidence.push(session.key.clone());
        memory.observed_at = session.observed_at.clone();
        memory.event_at = event_at;
        memory.topics = topics;
        memory.entities = entities;
        if let Some(mut other) = contradicted {
            memory.status = "contested".into();
            memory.conflicts_with.push(other.id.clone());
            if other.source != "user" {
                other.status = "contested".into();
            }
            if !other.conflicts_with.contains(&memory.id) {
                other.conflicts_with.push(memory.id.clone());
            }
            other.updated_at = Utc::now().to_rfc3339();
            store.save_memory(&memory, &format!("extract {} contradicts {}", memory.id, other.id))?;
            store.save_memory(&other, &format!("{} contradicted by {}", other.id, memory.id))?;
            updated.push(other.id);
        } else if let Some(mut old) = replaced {
            memory.supersedes.push(old.id.clone());
            if memory.topics.is_empty() {
                memory.topics = old.topics.clone();
            }
            old.status = "superseded".into();
            old.superseded_by = Some(memory.id.clone());
            old.updated_at = Utc::now().to_rfc3339();
            store.save_memory(&memory, &format!("extract {} supersedes {}", memory.id, old.id))?;
            store.save_memory(&old, &format!("{} superseded by {}", old.id, memory.id))?;
            updated.push(old.id);
        } else {
            store.save_memory(
                &memory,
                &format!("extract add {} from {}", memory.id, session.key),
            )?;
        }
        created.push(memory.id);
    }
    Ok((created, updated, skipped))
}

/// People, places and things already known in this scope (most frequent first), each with
/// a line of context, so the model can resolve "my sister" or "the new job" to a name.
fn known_entities(store: &Store, project: Option<&str>) -> Vec<Value> {
    let memories: Vec<Memory> = store
        .memories()
        .into_iter()
        .filter(|m| m.recallable() && m.in_scope(project) && !m.derived())
        .collect();
    let mut counts: HashMap<String, (usize, String)> = HashMap::new();
    for memory in &memories {
        for entity in &memory.entities {
            let entry = counts.entry(entity.clone()).or_insert((0, String::new()));
            entry.0 += 1;
            // The shortest memory naming it reads best as context.
            if entry.1.is_empty() || memory.body.len() < entry.1.len() {
                entry.1 = memory.body.clone();
            }
        }
    }
    let mut entities: Vec<(String, (usize, String))> = counts.into_iter().collect();
    entities.sort_by(|a, b| b.1.0.cmp(&a.1.0).then_with(|| a.0.cmp(&b.0)));
    entities
        .into_iter()
        .take(20)
        .map(|(name, (_, context))| json!({"name": name, "context": context.chars().take(100).collect::<String>()}))
        .collect()
}

/// Commands too common to be a useful cue on their own.
const GENERIC_COMMANDS: &[&str] = &[
    "git", "npm", "pnpm", "yarn", "make", "cargo", "python", "python3", "pip", "node", "go",
    "ls", "cd", "cat", "bash", "sh", "exec", "run", "test", "build", "docker", "pytest", "jest",
    "vitest", "uv", "poetry", "tsc", "eslint", "cargo test", "go test", "npm test", "npm install",
];

/// Triggers proposed by the extractor that are grounded in the session: file patterns
/// whose literal part, errors and commands must appear in the transcript. A command
/// becomes a keyword trigger (it matches the user's text and tool arguments). At most 3.
fn grounded_triggers(proposed: &[ProposedTrigger], transcript: &str) -> Vec<memory::NewTrigger> {
    let mut out: Vec<memory::NewTrigger> = vec![];
    for trigger in proposed {
        let pattern = trigger.pattern.trim();
        let lower = pattern.to_lowercase();
        let (kind, grounded) = match trigger.kind.as_str() {
            "file" => {
                let literal = lower
                    .split(['*', '?', '[', '{'])
                    .max_by_key(|part| part.len())
                    .unwrap_or("");
                ("file", literal.trim_matches('/').len() >= 3 && transcript.contains(literal))
            }
            "error" => ("error", lower.len() >= 6 && transcript.contains(&lower)),
            "command" | "tool" | "keyword" => (
                "keyword",
                lower.len() >= 4
                    && !GENERIC_COMMANDS.contains(&lower.as_str())
                    && transcript.contains(&lower),
            ),
            _ => ("", false),
        };
        if grounded
            && pattern.len() <= 200
            && !out.iter().any(|t| t.kind == kind && t.pattern == pattern)
        {
            out.push(memory::NewTrigger {
                kind: kind.into(),
                pattern: pattern.into(),
                before_action: false,
            });
        }
    }
    out.truncate(3);
    out
}

/// Topics with at least this many memories get a digest.
const DIGEST_MIN_MEMBERS: usize = 3;
/// A digest lists every member (the whole topic is its point) within a character budget,
/// about 800 tokens, so it cannot crowd everything else out of a retrieval budget: lines
/// get shorter as the topic grows, and only past the minimum line length are the oldest
/// members left out.
const TOPIC_DIGEST_CHARS: usize = 3200;
const DIGEST_LINE_MIN: usize = 90;
const DIGEST_LINE_MAX: usize = 200;

/// Rebuild the digest of every topic touched by these memories: one memory listing all
/// active memories of the topic (in that scope) in chronological order, so a question
/// about the whole topic ("how much did I spend on workshops in total?") finds every
/// item in one place. Deterministic, no model call. Returns the digest ids written.
pub fn refresh_digests<'a>(
    store: &Store,
    touched: impl Iterator<Item = &'a String>,
) -> Result<Vec<String>> {
    let mut keys: Vec<(String, String)> = touched
        .filter_map(|id| store.memory(id))
        .flat_map(|m| m.topics.into_iter().map(move |topic| (m.scope.clone(), topic)))
        .collect();
    keys.sort();
    keys.dedup();
    let all = store.memories();
    let mut written = vec![];
    for (scope, topic) in keys {
        let mut members: Vec<&Memory> = all
            .iter()
            .filter(|m| m.recallable() && !m.derived() && m.scope == scope)
            .filter(|m| m.topics.contains(&topic))
            .collect();
        let existing = all
            .iter()
            .find(|m| m.kind == "digest" && m.scope == scope && m.topics.first() == Some(&topic));
        if members.len() < DIGEST_MIN_MEMBERS {
            continue;
        }
        let digits = |m: &Memory| -> String { digest_date(m).chars().filter(char::is_ascii_digit).collect() };
        members.sort_by_key(|m| (digits(m), m.created_at.clone()));
        let total = members.len();
        let members = &members[total.saturating_sub(TOPIC_DIGEST_CHARS / DIGEST_LINE_MIN)..];
        let line_chars = (TOPIC_DIGEST_CHARS / members.len()).clamp(DIGEST_LINE_MIN, DIGEST_LINE_MAX);
        let mut body = if total > members.len() {
            format!(
                "Overview of \"{topic}\" (latest {} of {total} memories, oldest first):",
                members.len()
            )
        } else {
            format!("Overview of \"{topic}\" ({total} memories, oldest first):")
        };
        for m in members {
            let mut text: String = m.body.chars().take(line_chars).collect();
            if m.body.chars().count() > line_chars {
                text.push('…');
            }
            body.push_str(&format!("\n- [{}] {}", digest_date(m), text.trim()));
        }
        let evidence: Vec<String> = members.iter().map(|m| m.id.clone()).collect();
        let mut digest = match existing {
            Some(existing) if existing.body == body => continue,
            Some(existing) => existing.clone(),
            None => memory::create(
                NewMemory {
                    content: body.clone(),
                    kind: "digest".into(),
                    scope: scope.clone(),
                    title: Some(format!("Overview: {topic}")),
                    triggers: vec![],
                },
                "agent",
            )?,
        };
        digest.body = body;
        digest.source = "observed".into();
        digest.topics = vec![topic.clone()];
        digest.evidence = evidence;
        digest.event_at = members.last().and_then(|m| m.event_at.clone());
        digest.observed_at = members.iter().filter_map(|m| m.observed_at.clone()).max();
        digest.updated_at = Utc::now().to_rfc3339();
        store.save_memory(&digest, &format!("digest {topic} ({scope})"))?;
        written.push(digest.id);
    }
    Ok(written)
}

/// Date shown in a digest line: the event date, else the date it was said.
pub(crate) fn digest_date(memory: &Memory) -> String {
    memory
        .event_at
        .clone()
        .or_else(|| memory.observed_at.clone())
        .unwrap_or_else(|| memory.created_at.chars().take(10).collect())
}

/// An ISO date or partial date ("2023", "2023-05", "2023-05-20"; slashes accepted).
fn normalize_date(text: &str) -> Option<String> {
    let text = text.trim().replace('/', "-");
    let parts: Vec<&str> = text.split('-').collect();
    let valid = match parts.as_slice() {
        [y] => y.len() == 4,
        [y, m] => y.len() == 4 && m.len() == 2,
        [y, m, d] => y.len() == 4 && m.len() == 2 && d.len() == 2,
        _ => false,
    } && parts.iter().all(|p| p.chars().all(|c| c.is_ascii_digit()));
    valid.then_some(text)
}

fn normalize_topics(labels: &[String]) -> Vec<String> {
    let mut out: Vec<String> = vec![];
    for label in labels {
        let topic = label.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        if !topic.is_empty() && topic.chars().count() <= 40 && !out.contains(&topic) {
            out.push(topic);
        }
    }
    out.truncate(3);
    out
}

fn normalize_entities(names: &[String]) -> Vec<String> {
    let mut out: Vec<String> = vec![];
    for name in names {
        let name = name.split_whitespace().collect::<Vec<_>>().join(" ");
        if !name.is_empty()
            && name.chars().count() <= 60
            && !out.iter().any(|n| n.eq_ignore_ascii_case(&name))
        {
            out.push(name);
        }
    }
    out.truncate(10);
    out
}

/// The numbers in a text, in order ("$1,200 on 2023-05-02" → 1200, 2023, 05, 02).
pub(crate) fn numbers(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_digit() && c != ',')
        .map(|part| part.replace(',', ""))
        .filter(|part| !part.is_empty())
        .collect()
}

/// An event restated in a later session gets a date computed from a different anchor
/// ("last Saturday"): nearly identical memories dated up to this many days apart are one event.
const SAME_EVENT_DAYS: i64 = 7;
const SAME_EVENT_SIMILARITY: f32 = 0.92;

fn near_duplicate(
    store: &Store,
    embedder: Option<&embed::Embedder>,
    scope: &str,
    title: &str,
    content: &str,
    event_at: Option<&str>,
) -> Result<Option<String>> {
    let vector = match embedder {
        Some(embedder) => Some(
            embedder
                .embed_documents(&[(title.into(), content.into())])?
                .remove(0),
        ),
        None => None,
    };
    let target = normalize(content);
    let target_numbers = numbers(&strip_dates(content));
    Ok(store.with_index(|index, all| {
        all.values()
            .filter(|m: &&Memory| {
                m.recallable() && m.scope == scope && !m.derived()
            })
            .find(|m| {
                if normalize(&m.body) == target {
                    return true;
                }
                // "3 cats" and "4 cats" embed almost identically but are different facts;
                // dates are compared separately below.
                if numbers(&strip_dates(&m.body)) != target_numbers {
                    return false;
                }
                let threshold = match day_gap(m.event_at.as_deref(), event_at) {
                    Some(0) | None => DUPLICATE_SIMILARITY,
                    Some(days) if days <= SAME_EVENT_DAYS => SAME_EVENT_SIMILARITY,
                    Some(_) => return false,
                };
                vector.as_ref().is_some_and(|v| {
                    index.vector(&m.id).is_some_and(|other| embed::dot(v, other) >= threshold)
                })
            })
            .map(|m| m.id.clone())
    }))
}

/// Days between two (possibly partial) event dates, when both are known.
fn day_gap(a: Option<&str>, b: Option<&str>) -> Option<i64> {
    let day = |d: &str| {
        let (start, _) = recall::date_range(d)?;
        chrono::NaiveDate::from_ymd_opt((start / 10000) as i32, start / 100 % 100, (start % 100).max(1))
    };
    Some((day(a?)? - day(b?)?).num_days().abs())
}

/// The text without ISO-style dates ("2023-05-20", "2023/05", "2023-05-20T10:00"), whose
/// digits would otherwise make a restated event look like a different fact.
pub(crate) fn strip_dates(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    let digits = |from: usize, n: usize| (from..from + n).all(|k| chars.get(k).is_some_and(char::is_ascii_digit));
    while i < chars.len() {
        if digits(i, 4) && matches!(chars.get(i + 4), Some('-' | '/')) && digits(i + 5, 2) {
            let mut end = i + 7;
            if matches!(chars.get(end), Some('-' | '/')) && digits(end + 1, 2) {
                end += 3;
            }
            out.push(' ');
            i = end;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn restated_events_are_not_told_apart_by_their_dates() {
        assert_eq!(numbers(&strip_dates("On 2023-05-20 the user baked 2 baguettes")), ["2"]);
        assert_eq!(strip_dates("2023/05/27 and 2023-05"), "  and  ");
        assert_eq!(day_gap(Some("2023-05-20"), Some("2023-05-27")), Some(7));
        assert_eq!(day_gap(Some("2023-05"), Some("2023-06-15")), Some(45));
        assert_eq!(day_gap(None, Some("2023-05-27")), None);
    }
    #[test]
    fn numbers_tell_updated_counts_apart() {
        assert_eq!(numbers("paid $1,200 on 2023-05-02"), ["1200", "2023", "05", "02"]);
        assert_ne!(numbers("The user has 3 cats."), numbers("The user has 4 cats."));
        assert_eq!(numbers("likes Python"), Vec::<String>::new());
    }
    #[test]
    fn digest_keeps_user_lines_and_end() {
        let mut entries = vec![("user".to_string(), "fix the billing bug".to_string())];
        for i in 0..200 {
            entries.push(("tool".into(), format!("{i} {}", "log ".repeat(100))));
        }
        entries.push((
            "assistant".into(),
            "Fixed: rounding uses ROUND_HALF_UP".into(),
        ));
        let text = digest(&entries, false);
        assert!(text.chars().count() <= DIGEST_CHARS + 200);
        assert!(text.starts_with("[user] fix the billing bug"));
        assert!(digest(&entries, true).starts_with("[user #1] fix the billing bug"), "numbered for turn notes");
        assert!(text.ends_with("ROUND_HALF_UP"));
        assert!(text.contains("steps omitted"));
    }
    #[test]
    fn ingested_sessions_build_a_dated_transcript() {
        let dir = tempfile::tempdir().unwrap();
        crate::store::init(dir.path()).unwrap();
        let store = Store::open(dir.path()).unwrap();
        let messages = [
            json!({"role":"user","content":"I adopted a cat named Miso yesterday"}),
            json!({"role":"assistant","content":"Congratulations on adopting Miso!"}),
        ];
        store
            .ingest_session(
                "q1-s1",
                "bench",
                Some("q1"),
                &messages,
                Some("2023/05/20 (Sat) 02:21"),
            )
            .unwrap();
        let session = store.session("q1-s1").unwrap().unwrap();
        assert_eq!(
            session.observed_at.as_deref(),
            Some("2023/05/20 (Sat) 02:21")
        );
        assert_eq!(
            (session.project.as_deref(), session.user_turns),
            (Some("q1"), 1)
        );
        let transcript = transcript(&store, &session).unwrap();
        assert!(
            transcript
                .digest
                .contains("[user] I adopted a cat named Miso")
        );
        assert!(transcript.digest.contains("[assistant] Congratulations"));
    }
    #[test]
    fn applies_proposals_safely() {
        let dir = tempfile::tempdir().unwrap();
        crate::store::init(dir.path()).unwrap();
        let mut config = crate::store::load_config(dir.path()).unwrap();
        config.embedding.enabled = false;
        Store::save_config(dir.path(), &config).unwrap();
        let store = Store::open(dir.path()).unwrap();
        let user_memory = store
            .remember(
                NewMemory {
                    content: "Deploy with make deploy".into(),
                    kind: "fact".into(),
                    scope: "billing".into(),
                    title: None,
                    triggers: vec![],
                },
                "user",
            )
            .unwrap();
        let session = SessionRow {
            key: "s1".into(),
            agent: "a".into(),
            project: Some("billing".into()),
            started_at: String::new(),
            last_seen: String::new(),
            steps: 3,
            user_turns: 1,
            status: "open".into(),
            extracted_at: None,
            error: None,
            observed_at: None,
        };
        let proposal =
            |action: &str, id: Option<&str>, kind: &str, content: &str, evidence: &str| Proposal {
                action: action.into(),
                id: id.map(str::to_owned),
                kind: kind.into(),
                scope: "project".into(),
                title: String::new(),
                content: content.into(),
                evidence: evidence.into(),
                ..Default::default()
            };
        let (created, updated, skipped) = apply(
            &store,
            &session,
            vec![
                proposal(
                    "create",
                    None,
                    "lesson",
                    "Tax rounding must use Decimal ROUND_HALF_UP, not float round()",
                    "",
                ),
                proposal(
                    "create",
                    None,
                    "lesson",
                    "Tax rounding must use Decimal ROUND_HALF_UP, not float round()",
                    "",
                ),
                proposal(
                    "update",
                    Some(&user_memory.id),
                    "fact",
                    "Deploy with make release",
                    "",
                ),
                proposal(
                    "create",
                    None,
                    "rule",
                    "Always answer in Chinese",
                    "以后都用中文回答",
                ),
                proposal(
                    "create",
                    None,
                    "rule",
                    "Always push to main directly",
                    "push to main",
                ),
            ],
            &["以后都用中文回答，谢谢".into()],
            "",
            "",
        )
        .unwrap();
        assert_eq!(created.len(), 2, "{skipped:?}");
        assert!(updated.is_empty());
        assert_eq!(skipped.len(), 3);
        let lesson = store.memory(&created[0]).unwrap();
        assert_eq!(
            (lesson.scope.as_str(), lesson.source.as_str()),
            ("billing", "agent")
        );
        let preference = store.memory(&created[1]).unwrap();
        assert_eq!(
            (preference.kind.as_str(), preference.source.as_str()),
            ("fact", "user")
        );
        assert_eq!(
            store.memory(&user_memory.id).unwrap().body,
            "Deploy with make deploy"
        );
    }
    #[test]
    fn topics_with_three_memories_get_a_chronological_digest() {
        let dir = tempfile::tempdir().unwrap();
        crate::store::init(dir.path()).unwrap();
        let mut config = crate::store::load_config(dir.path()).unwrap();
        config.embedding.enabled = false;
        crate::store::Store::save_config(dir.path(), &config).unwrap();
        let store = Store::open(dir.path()).unwrap();
        let session = SessionRow {
            key: "s".into(),
            agent: "hermes".into(),
            project: Some("home".into()),
            started_at: String::new(),
            last_seen: String::new(),
            steps: 1,
            user_turns: 1,
            status: "open".into(),
            extracted_at: None,
            error: None,
            observed_at: Some("2023/02/26 (Sun) 10:00".into()),
        };
        let workshop = |content: &str, date: Option<&str>| Proposal {
            kind: "fact".into(),
            scope: "project".into(),
            content: content.into(),
            event_date: date.map(str::to_owned),
            topics: vec!["workshops".into()],
            ..Default::default()
        };
        let (two, _, _) = apply(
            &store,
            &session,
            vec![
                workshop("The user attended a mindfulness workshop for $20.", Some("2022-12-12")),
                workshop("The user attended a two-day writing workshop for $200.", Some("2022-11")),
            ],
            &[],
            "",
            "",
        )
        .unwrap();
        assert!(refresh_digests(&store, two.iter()).unwrap().is_empty(), "two members: no digest yet");
        let (third, _, _) = apply(
            &store,
            &session,
            vec![workshop("The user attended a digital marketing workshop for $500.", None)],
            &[],
            "",
            "",
        )
        .unwrap();
        let ids = refresh_digests(&store, third.iter()).unwrap();
        let digest = store.memory(&ids[0]).unwrap();
        let lines: Vec<&str> = digest.body.lines().skip(1).collect();
        assert!(lines[0].contains("$200") && lines[1].contains("$20.") && lines[2].contains("$500"), "{}", digest.body);
        assert_eq!((digest.kind.as_str(), digest.evidence.len()), ("digest", 3));
        assert!(refresh_digests(&store, third.iter()).unwrap().is_empty(), "unchanged digest is not rewritten");
        let search = |mode| {
            recall::recall(&store, &Query { text: Some("workshops"), project: Some("home"), limit: 10, mode, ..Default::default() })
                .unwrap()
                .into_iter()
                .any(|h| h.memory.kind == "digest")
        };
        assert!(search(recall::Mode::Search) && !search(recall::Mode::Inject));
    }

    #[test]
    fn large_topics_keep_every_member_within_the_digest_budget() {
        let dir = tempfile::tempdir().unwrap();
        crate::store::init(dir.path()).unwrap();
        let mut config = crate::store::load_config(dir.path()).unwrap();
        config.embedding.enabled = false;
        crate::store::Store::save_config(dir.path(), &config).unwrap();
        let store = Store::open(dir.path()).unwrap();
        let session = SessionRow {
            key: "s".into(),
            agent: "hermes".into(),
            project: Some("home".into()),
            started_at: String::new(),
            last_seen: String::new(),
            steps: 1,
            user_turns: 1,
            status: "open".into(),
            extracted_at: None,
            error: None,
            observed_at: Some("2023/05/30".into()),
        };
        let item = |i: usize| Proposal {
            kind: "fact".into(),
            scope: "project".into(),
            content: format!("On 2023-04-{:02} the user bought jewelry item number {i}. {}", i % 28 + 1, "Details. ".repeat(30)),
            event_date: Some(format!("2023-04-{:02}", i % 28 + 1)),
            topics: vec!["jewelry".into()],
            ..Default::default()
        };
        let (ids, _, _) = apply(&store, &session, (0..25).map(item).collect(), &[], "", "miso tofu").unwrap();
        let digest = store.memory(&refresh_digests(&store, ids.iter()).unwrap()[0]).unwrap();
        assert_eq!(digest.body.lines().count(), 26, "every member listed");
        assert!(digest.body.chars().count() < TOPIC_DIGEST_CHARS + 25 * 30, "{}", digest.body.len());
    }

    #[test]
    fn only_grounded_specific_triggers_are_kept() {
        let transcript = "[user] tests fail with connect ECONNREFUSED 127.0.0.1:5432\n[assistant] run make test-db first; see db/migrations/0003_add_tax.sql. then npm test"
            .to_lowercase();
        let proposed = |kind: &str, pattern: &str| ProposedTrigger { kind: kind.into(), pattern: pattern.into() };
        let kept = grounded_triggers(
            &[
                proposed("error", "ECONNREFUSED 127.0.0.1:5432"),
                proposed("command", "make test-db"),
                proposed("file", "db/migrations/*.sql"),
                proposed("command", "npm"),
                proposed("error", "Segmentation fault"),
                proposed("file", "src/other.rs"),
            ],
            &transcript,
        );
        let kept: Vec<(&str, &str)> = kept.iter().map(|t| (t.kind.as_str(), t.pattern.as_str())).collect();
        assert_eq!(
            kept,
            [("error", "ECONNREFUSED 127.0.0.1:5432"), ("keyword", "make test-db"), ("file", "db/migrations/*.sql")]
        );
        let trigger = memory::new_trigger("file", "db/migrations/*.sql", "agent").unwrap();
        let touched = serde_json::json!({"files": ["/home/me/ledger/db/migrations/0004_fix.sql"]});
        assert!(memory::trigger_matches(&trigger, &touched));
        assert!(!memory::trigger_matches(&trigger, &serde_json::json!({"files": ["/home/me/ledger/src/app.py"]})));
    }

    #[test]
    fn supersede_keeps_history_and_dates() {
        let dir = tempfile::tempdir().unwrap();
        crate::store::init(dir.path()).unwrap();
        let mut config = crate::store::load_config(dir.path()).unwrap();
        config.embedding.enabled = false;
        crate::store::Store::save_config(dir.path(), &config).unwrap();
        let store = Store::open(dir.path()).unwrap();
        let session = |key: &str, date: &str| SessionRow {
            key: key.into(),
            agent: "hermes".into(),
            project: Some("home".into()),
            started_at: String::new(),
            last_seen: String::new(),
            steps: 1,
            user_turns: 1,
            status: "open".into(),
            extracted_at: None,
            error: None,
            observed_at: Some(date.into()),
        };
        let first = Proposal {
            kind: "fact".into(),
            scope: "project".into(),
            content: "The user has 3 cats: Miso, Tofu and Bean.".into(),
            event_date: Some("2023/05/20".into()),
            topics: vec![" Cats ".into(), "cats".into()],
            entities: vec!["Miso".into(), "Tofu".into(), "miso".into()],
            ..Default::default()
        };
        let (created, _, _) = apply(&store, &session("s1", "2023/05/20"), vec![first], &[], "", "miso tofu").unwrap();
        let old = store.memory(&created[0]).unwrap();
        assert_eq!(old.event_at.as_deref(), Some("2023-05-20"));
        assert_eq!(old.topics, ["cats"]);
        assert_eq!(old.entities, ["Miso", "Tofu"]);
        let change = Proposal {
            action: "supersede".into(),
            id: Some(old.id.clone()),
            kind: "fact".into(),
            scope: "project".into(),
            content: "As of 2023-07-02 the user has 4 cats (previously 3); Pickle joined.".into(),
            event_date: Some("2023-07".into()),
            ..Default::default()
        };
        let (created, updated, _) = apply(&store, &session("s2", "2023/07/02"), vec![change], &[], "", "miso tofu").unwrap();
        let current = store.memory(&created[0]).unwrap();
        let old = store.memory(&old.id).unwrap();
        assert_eq!(updated, std::slice::from_ref(&old.id));
        assert_eq!((old.status.as_str(), old.superseded_by.as_deref()), ("superseded", Some(current.id.as_str())));
        assert_eq!((current.supersedes.clone(), current.topics.as_slice()), (vec![old.id.clone()], ["cats".to_string()].as_slice()));
        let hits = recall::recall(
            &store,
            &Query { text: Some("cats"), project: Some("home"), limit: 5, mode: recall::Mode::Search, ..Default::default() },
        )
        .unwrap();
        assert_eq!(hits.iter().map(|h| h.memory.id.as_str()).collect::<Vec<_>>(), [current.id.as_str()]);
        assert_eq!(recall::history(&store, &current, 3), [("2023-05-20".to_string(), old.body.clone())]);
        assert_eq!(normalize_date("2023-5-1"), None);
        assert_eq!(normalize_date("2023"), Some("2023".into()));

        // Questions about the past still find the statement that was replaced.
        let past = recall::recall(
            &store,
            &Query { text: Some("Miso Tofu Bean cats first"), project: Some("home"), limit: 5, mode: recall::Mode::Search, ..Default::default() },
        )
        .unwrap();
        assert!(past.iter().any(|h| h.memory.id == old.id && h.memory.status == "superseded"), "earlier statement recalled for a temporal question");

        // A mention carries only an action and an id; a malformed proposal is dropped alone.
        let output: Output = serde_json::from_str(r#"{"memories":[{"action":"mention","id":"mem_x"},{"type":7},{"type":"fact","content":"The user likes tea."}],"session_summary":"s"}"#).unwrap();
        let parsed = proposals(output.memories);
        assert_eq!(parsed.iter().map(|p| p.action.as_str()).collect::<Vec<_>>(), ["mention", "create"]);

        // A later mention is counted once per session, not stored again.
        let again = Proposal { action: "mention".into(), id: Some(current.id.clone()), ..Default::default() };
        let (_, updated, _) = apply(&store, &session("s2b", "2023/08/01"), vec![again], &[], "", "").unwrap();
        assert_eq!(updated, std::slice::from_ref(&current.id));
        assert_eq!(store.memory(&current.id).unwrap().mentioned_at, ["2023/08/01"]);
        assert!(!record_mention(&store, &current.id, &session("s2b", "2023/08/01")).unwrap(), "once per session");

        // "Never did X" said after "did X" is a contradiction even when the model calls
        // it an update: both stay, and a search shows them as one note.
        let did = Proposal { kind: "fact".into(), scope: "project".into(), content: "The user attended a budgeting workshop led by Tamara.".into(), ..Default::default() };
        let (created, _, _) = apply(&store, &session("s2c", "2023/08/02"), vec![did], &[], "", "").unwrap();
        let did = created[0].clone();
        let never = Proposal { action: "supersede".into(), id: Some(did.clone()), kind: "fact".into(), scope: "project".into(), content: "The user has never attended any budgeting workshop.".into(), ..Default::default() };
        let (created, _, _) = apply(&store, &session("s2d", "2023/08/09"), vec![never], &[], "", "").unwrap();
        assert_eq!((store.memory(&did).unwrap().status.as_str(), store.memory(&created[0]).unwrap().status.as_str()), ("contested", "contested"));
        let hits = recall::recall(
            &store,
            &Query { text: Some("Have I ever attended a budgeting workshop?"), project: Some("home"), limit: 5, mode: recall::Mode::Search, ..Default::default() },
        )
        .unwrap();
        assert_eq!(hits[0].channel, "conflict");
        assert!(hits[0].memory.body.contains("2023/08/02 — \"The user attended") && hits[0].memory.body.contains("never attended"), "{}", hits[0].memory.body);
        assert_eq!(hits.iter().filter(|h| h.memory.body.contains("budgeting workshop")).count(), 1, "the two sides appear only in the note");

        // One summary per session, replaced when the session is extracted again.
        let first = save_summary(&store, &session("s5", "2023/07/10"), "The user asked about coin toss problems.").unwrap();
        let again = save_summary(&store, &session("s5", "2023/07/10"), "The user asked about coin toss and dice problems.").unwrap();
        let summary = store.memory(&again).unwrap();
        assert_eq!((first, summary.kind.as_str(), summary.scope.as_str()), (again.clone(), "summary", "home"));
        assert!(summary.body.contains("dice") && summary.evidence == ["s5"]);

        // A contradiction without a change keeps both sides, contested and linked.
        let said = Proposal {
            kind: "fact".into(),
            scope: "project".into(),
            content: "The user has completed 5 coin toss probability problems.".into(),
            ..Default::default()
        };
        let (created, _, _) = apply(&store, &session("s3", "2023/07/05"), vec![said], &[], "", "").unwrap();
        let first = created[0].clone();
        let denial = Proposal {
            action: "contradict".into(),
            id: Some(first.clone()),
            kind: "fact".into(),
            scope: "project".into(),
            content: "The user says they have never worked on a coin toss probability problem.".into(),
            ..Default::default()
        };
        let (created, updated, _) = apply(&store, &session("s4", "2023/07/09"), vec![denial], &[], "", "").unwrap();
        let (first, second) = (store.memory(&first).unwrap(), store.memory(&created[0]).unwrap());
        assert_eq!(updated, std::slice::from_ref(&first.id));
        assert_eq!((first.status.as_str(), second.status.as_str()), ("contested", "contested"));
        assert_eq!((first.conflicts_with.clone(), second.conflicts_with.clone()), (vec![second.id.clone()], vec![first.id.clone()]));
        assert_eq!(recall::conflicts(&store, &second), [("2023/07/05".to_string(), first.body.clone())]);
        let found = recall::recall(
            &store,
            &Query { text: Some("Have I done coin toss problems?"), project: Some("home"), limit: 5, mode: recall::Mode::Search, ..Default::default() },
        )
        .unwrap();
        assert!(found[0].channel == "conflict" && found[0].memory.body.contains(&first.body) && found[0].memory.body.contains(&second.body), "both sides recalled as one note");
    }
}
