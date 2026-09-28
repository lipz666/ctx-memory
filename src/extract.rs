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
use std::time::Duration;

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
    #[serde(default)]
    memories: Vec<Proposal>,
    #[serde(default)]
    skip_reason: Option<String>,
}
#[derive(Deserialize)]
struct Proposal {
    #[serde(default = "create")]
    action: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    scope: String,
    #[serde(default)]
    title: String,
    content: String,
    #[serde(default)]
    evidence: String,
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
        digest: digest(&entries),
        user_texts,
        entries,
    })
}

/// Fit entries into the digest budget: user lines are kept, the middle of the rest is
/// dropped first.
fn digest(entries: &[(String, String)]) -> String {
    let lines: Vec<String> = entries
        .iter()
        .map(|(role, text)| format!("[{role}] {}", text.trim()))
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
        if !lines[i].starts_with("[user]") && i + 1 != lines.len() {
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
    let related_query: String = transcript
        .user_texts
        .join("\n")
        .chars()
        .take(2000)
        .collect();
    let related = recall::recall(
        store,
        &Query {
            text: Some(&related_query),
            project,
            limit: 8,
            ..Default::default()
        },
    )?;
    let existing: Vec<Value> = related
        .iter()
        .map(|h| json!({"id":h.memory.id,"type":h.memory.kind,"scope":h.memory.scope,"title":h.memory.title,"content":h.memory.body.chars().take(400).collect::<String>()}))
        .collect();
    let input = json!({
        "project": project,
        "session_date": session.observed_at.as_deref().unwrap_or(&session.last_seen),
        "existing_memories": existing,
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
    let result = apply(store, session, output.memories, &transcript.user_texts)?;
    store.finish_session(&session.key, "done", None)?;
    Ok(
        json!({"session":session.key,"created":result.0,"updated":result.1,"skipped":result.2,"skip_reason":output.skip_reason,"episodes":episodes}),
    )
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
) -> Result<(Vec<String>, Vec<String>, Vec<String>)> {
    let (mut created, mut updated, mut skipped) = (vec![], vec![], vec![]);
    let users = normalize(&user_texts.join("\n"));
    let embedder = store.embedder.get();
    for proposal in proposals.into_iter().take(MAX_PROPOSALS) {
        let content = redact(proposal.content.trim());
        if !["fact", "lesson", "skill", "rule"].contains(&proposal.kind.as_str())
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
        if let Some(duplicate) =
            near_duplicate(store, embedder.as_deref(), &scope, &title, &content)?
        {
            skipped.push(format!("duplicate of {duplicate}"));
            continue;
        }
        let mut memory = memory::create(
            NewMemory {
                content,
                kind,
                scope,
                title: Some(title),
                triggers: vec![],
            },
            "agent",
        )?;
        if proposal.kind == "rule" {
            memory.source = "user".into();
            memory.confidence = 0.8;
        }
        memory.evidence.push(session.key.clone());
        memory.observed_at = session.observed_at.clone();
        store.save_memory(
            &memory,
            &format!("extract add {} from {}", memory.id, session.key),
        )?;
        created.push(memory.id);
    }
    Ok((created, updated, skipped))
}

/// The numbers in a text, in order ("$1,200 on 2023-05-02" → 1200, 2023, 05, 02).
fn numbers(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_digit() && c != ',')
        .map(|part| part.replace(',', ""))
        .filter(|part| !part.is_empty())
        .collect()
}

fn near_duplicate(
    store: &Store,
    embedder: Option<&embed::Embedder>,
    scope: &str,
    title: &str,
    content: &str,
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
    let target_numbers = numbers(content);
    Ok(store.with_index(|index, all| {
        all.values()
            .filter(|m: &&Memory| m.recallable() && m.scope == scope)
            .find(|m| {
                normalize(&m.body) == target
                    // "3 cats" and "4 cats" embed almost identically but are different facts.
                    || numbers(&m.body) == target_numbers && vector.as_ref().is_some_and(|v| {
                        index
                            .vector(&m.id)
                            .is_some_and(|other| embed::dot(v, other) >= DUPLICATE_SIMILARITY)
                    })
            })
            .map(|m| m.id.clone())
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let text = digest(&entries);
        assert!(text.chars().count() <= DIGEST_CHARS + 200);
        assert!(text.starts_with("[user] fix the billing bug"));
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
}
