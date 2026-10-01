//! Episodic tier: excerpts of the raw conversations, kept next to the extracted memories.
//!
//! Extraction distills a session into a few memories and inevitably drops details (an
//! amount, a name mentioned in passing, what the assistant recommended). Excerpts keep
//! the original wording; explicit searches (`recall::Mode::Search`) rank them together
//! with the memories. Automatic injection never uses them. Text is stored encrypted like
//! the events it comes from; only the search index lives in memory.
use crate::index::Index;
use regex::Regex;
use std::{collections::HashMap, sync::LazyLock};

/// Target size of one excerpt, in characters.
pub const CHUNK_CHARS: usize = 1200;
/// Excerpts kept per session; very long sessions keep their beginning.
pub const MAX_PER_SESSION: usize = 400;
/// Characters of the user's message repeated on each continuation excerpt.
const CONTEXT_CHARS: usize = 300;

pub struct Meta {
    pub session: String,
    pub project: Option<String>,
    pub observed_at: Option<String>,
    /// The number of the turn it belongs to (see `Turn`); 0 when unknown.
    pub turn: usize,
}

/// One user message and the replies that followed: the unit of conversation order. Turns
/// are numbered over the sessions of a scope in the order the sessions were stored, so a
/// later statement has a higher number even on the same date (a date alone cannot tell
/// which of two values said the same day is the current one).
pub struct Turn {
    pub session: String,
    pub project: Option<String>,
    pub observed_at: Option<String>,
    pub number: usize,
    /// The user's message.
    pub text: String,
    /// What the assistant answered, in brief (written at extraction; `turn_notes`).
    pub note: Option<String>,
    /// The reply's structure taken from its text without a model: its first sentence and
    /// its headings and bold lead-ins (the steps, options and items it gave).
    pub outline: Option<String>,
    /// Its excerpts, in order.
    pub excerpts: Vec<String>,
}

impl Turn {
    /// The gist of the reply: the note written at extraction, else the outline.
    pub fn reply(&self) -> Option<&str> {
        self.note.as_deref().or(self.outline.as_deref())
    }
    /// The indexed text: the user's message. The gist of the reply is left out: in the
    /// index it pulls turns where the assistant discussed a topic ahead of those where
    /// the user stated something about it.
    pub fn document(&self) -> String {
        self.text.clone()
    }
}

#[derive(Default)]
pub struct Episodes {
    pub index: Index,
    pub meta: HashMap<String, Meta>,
    /// The user's messages, one document per turn (ids `tu_…`).
    pub turns: Index,
    pub turn_meta: HashMap<String, Turn>,
    /// Each session's turns, in order.
    pub session_turns: HashMap<String, Vec<String>>,
    /// Highest turn number per scope (lowercase project).
    last_turn: HashMap<Option<String>, usize>,
}

fn same_scope(project: Option<&str>, own: &Option<String>) -> bool {
    match (project, own) {
        (Some(wanted), Some(own)) => wanted.eq_ignore_ascii_case(own),
        (None, None) => true,
        _ => false,
    }
}

impl Episodes {
    pub fn in_scope(&self, id: &str, project: Option<&str>) -> bool {
        self.meta.get(id).is_some_and(|m| same_scope(project, &m.project))
    }
    pub fn turn_in_scope(&self, id: &str, project: Option<&str>) -> bool {
        self.turn_meta.get(id).is_some_and(|t| same_scope(project, &t.project))
    }
    /// Attach the gist of the assistant's reply to a turn. Returns false for an unknown turn.
    pub fn set_note(&mut self, id: &str, note: &str) -> bool {
        let Some(turn) = self.turn_meta.get_mut(id) else { return false };
        turn.note = Some(note.to_owned());
        true
    }
    /// Number a session's excerpts (given in order) as turns after those already known in
    /// its scope, and index the user's messages. Returns the ids of the new turns.
    pub fn add_turns(
        &mut self,
        session: &str,
        project: Option<&str>,
        observed_at: Option<&str>,
        excerpts: &[(String, String)],
    ) -> Vec<String> {
        let scope = project.map(str::to_lowercase);
        let mut added = vec![];
        for (ids, text, reply) in split_turns(excerpts) {
            let number = {
                let last = self.last_turn.entry(scope.clone()).or_insert(0);
                *last += 1;
                *last
            };
            for id in &ids {
                if let Some(meta) = self.meta.get_mut(id) {
                    meta.turn = number;
                }
            }
            let id = turn_id(&ids[0]);
            let outline = outline(&reply);
            self.session_turns.entry(session.to_owned()).or_default().push(id.clone());
            self.turn_meta.insert(
                id.clone(),
                Turn {
                    session: session.to_owned(),
                    project: project.map(str::to_owned),
                    observed_at: observed_at.map(str::to_owned),
                    number,
                    text,
                    note: None,
                    outline,
                    excerpts: ids,
                },
            );
            let turn = &self.turn_meta[&id];
            if !turn.text.is_empty() {
                let document = turn.document();
                self.turns.upsert(&id, &document);
            }
            added.push(id);
        }
        added
    }
}

/// Id of the turn whose first excerpt is `excerpt`.
pub fn turn_id(excerpt: &str) -> String {
    format!("tu_{}", excerpt.trim_start_matches("ep_"))
}

/// Group a session's excerpts (in order) into turns: (excerpt ids, the user's message, the
/// reply). A turn starts with an excerpt holding a new user message; excerpts that
/// continue a long reply repeat the start of the message (same `turn_key`), and those that
/// continue a long user message begin with "[user] …".
pub fn split_turns(excerpts: &[(String, String)]) -> Vec<(Vec<String>, String, String)> {
    let user_part = |text: &str| -> String {
        text.lines().take_while(|l| !l.starts_with("[assistant]")).collect::<Vec<_>>().join("\n")
    };
    // The reply's part of an excerpt; a piece continuing a cut reply starts with "…".
    let reply_part = |text: &str| -> String {
        let lines: Vec<&str> = text.lines().skip_while(|l| !l.starts_with("[assistant]")).collect();
        let joined = lines.join("\n");
        joined.strip_prefix("[assistant] ").unwrap_or(&joined).to_owned()
    };
    let append = |reply: &mut String, piece: String| {
        if let Some(rest) = piece.strip_prefix('…') {
            reply.push(' ');
            reply.push_str(rest);
        } else if !piece.is_empty() {
            if !reply.is_empty() {
                reply.push('\n');
            }
            reply.push_str(&piece);
        }
    };
    let mut out: Vec<(Vec<String>, String, String)> = vec![];
    let mut key = String::new();
    for (id, text) in excerpts {
        let first = text.lines().next().unwrap_or_default();
        if first.starts_with("[user] …")
            && let Some(turn) = out.last_mut()
        {
            turn.0.push(id.clone());
            turn.1.push(' ');
            turn.1.push_str(user_part(text).trim_start_matches("[user] …").trim());
            append(&mut turn.2, reply_part(text));
            continue;
        }
        if first.starts_with("[user] ") && turn_key(text) != key {
            key = turn_key(text);
            let mut reply = String::new();
            append(&mut reply, reply_part(text));
            out.push((vec![id.clone()], user_part(text).trim_start_matches("[user] ").trim().to_owned(), reply));
        } else if let Some(turn) = out.last_mut() {
            turn.0.push(id.clone());
            append(&mut turn.2, reply_part(text));
        } else {
            // A session that starts with a reply: a turn without a user message.
            out.push((vec![id.clone()], String::new(), reply_part(text)));
        }
    }
    out
}

/// Characters of a reply's outline.
const OUTLINE_CHARS: usize = 500;
static HEADING: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^#{1,6}\s+(.+)$").unwrap());
static BOLD_LEAD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(?:[-*•]\s+|\d+[.)]\s+)?\*\*(.+?)\*\*").unwrap());

/// A reply's structure without a model: its first sentence, then its headings and the
/// bold lead-ins of its paragraphs and list items ("Configure Anonymization Settings;
/// Implement Human Oversight; Start With a Pilot Program"). None for an empty reply.
pub fn outline(reply: &str) -> Option<String> {
    let reply = reply.trim();
    if reply.is_empty() {
        return None;
    }
    let first: String = reply
        .split_inclusive(['.', '!', '?', '\n'])
        .next()
        .unwrap_or(reply)
        .trim()
        .chars()
        .take(160)
        .collect();
    let mut items: Vec<String> = vec![];
    for line in reply.lines().map(str::trim) {
        let item = HEADING
            .captures(line)
            .or_else(|| BOLD_LEAD.captures(line))
            .map(|c| c[1].trim().trim_end_matches(':').trim().to_owned());
        if let Some(item) = item.filter(|i| !i.is_empty() && !items.contains(i)) {
            items.push(item);
        }
    }
    let mut out = first;
    if !items.is_empty() {
        out.push_str(" | ");
        out.push_str(&items.join("; "));
    }
    Some(out.chars().take(OUTLINE_CHARS).collect())
}

/// Split a conversation into excerpts. Each user turn starts a new excerpt holding the
/// turn and the replies that follow; long turns continue in further excerpts that repeat
/// the start of the user's message so each one is understandable alone. Tool output and
/// other roles are left out.
pub fn chunks(entries: &[(String, String)]) -> Vec<String> {
    let mut out: Vec<String> = vec![];
    let mut current = String::new();
    let mut question = String::new();
    for (role, text) in entries {
        if role != "user" && role != "assistant" {
            continue;
        }
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        if role == "user" {
            flush(&mut current, &mut out);
            question = text.chars().take(CONTEXT_CHARS).collect();
        }
        for (i, piece) in windows(text, CHUNK_CHARS).into_iter().enumerate() {
            let line = if i == 0 {
                format!("[{role}] {piece}")
            } else {
                format!("[{role}] …{piece}")
            };
            if !current.is_empty()
                && current.chars().count() + line.chars().count() > CHUNK_CHARS
            {
                flush(&mut current, &mut out);
            }
            if current.is_empty() && role == "assistant" && !question.is_empty() {
                current = format!("[user] {question}\n");
            }
            if !current.is_empty() {
                current.push('\n');
            }
            current.push_str(&line);
        }
        if out.len() >= MAX_PER_SESSION {
            break;
        }
    }
    flush(&mut current, &mut out);
    out.truncate(MAX_PER_SESSION);
    out
}

/// Identifies the conversation turn an excerpt belongs to: every excerpt of a turn starts
/// with (at least the first `CONTEXT_CHARS` characters of) the same user message.
pub fn turn_key(excerpt: &str) -> String {
    excerpt
        .lines()
        .next()
        .unwrap_or_default()
        .chars()
        .take("[user] ".len() + CONTEXT_CHARS)
        .collect()
}

fn flush(current: &mut String, out: &mut Vec<String>) {
    if !current.trim().is_empty() {
        out.push(std::mem::take(current).trim().to_owned());
    }
    current.clear();
}

/// Cut text into pieces of at most `size` characters, at whitespace when one is near.
fn windows(text: &str, size: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut pieces = vec![];
    let mut start = 0;
    while start < chars.len() {
        let mut end = (start + size).min(chars.len());
        if end < chars.len()
            && let Some(space) = (start + size * 4 / 5..end).rev().find(|&i| chars[i].is_whitespace())
        {
            end = space;
        }
        pieces.push(chars[start..end].iter().collect::<String>().trim().to_owned());
        start = end;
    }
    pieces.retain(|p| !p.is_empty());
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;
    fn entry(role: &str, text: &str) -> (String, String) {
        (role.into(), text.into())
    }
    #[test]
    fn outlines_keep_the_first_sentence_and_the_structure() {
        let reply = "Using AI hiring software can speed up screening. Here is how:\n\n### Steps to Ensure Fairness\n1. **Start With a Pilot Program**: test it first.\n- **Implement Human Oversight:** keep people in final decisions.\n**Configure Anonymization Settings** to remove names.\nPlain text.";
        assert_eq!(
            outline(reply).unwrap(),
            "Using AI hiring software can speed up screening. | Steps to Ensure Fairness; Start With a Pilot Program; Implement Human Oversight; Configure Anonymization Settings"
        );
        assert_eq!(outline("Sure!").unwrap(), "Sure!");
        assert_eq!(outline("  "), None);
    }
    #[test]
    fn excerpts_group_into_numbered_turns_across_sessions() {
        let long_reply = "word ".repeat(600);
        let long_question = format!("Please review this code {}", "x ".repeat(700));
        let first: Vec<(String, String)> = chunks(&[
            entry("user", "I paid $200 for the writing workshop"),
            entry("assistant", &long_reply),
            entry("user", &long_question),
            entry("assistant", "Looks good."),
        ])
        .into_iter()
        .enumerate()
        .map(|(i, text)| (format!("ep_a{i}"), text))
        .collect();
        let turns = split_turns(&first);
        assert_eq!(turns.len(), 2, "{:?}", turns.iter().map(|t| &t.1).collect::<Vec<_>>());
        assert!(turns[0].2.starts_with("word word") && turns[0].2.len() > 2900, "the reply is joined: {}", turns[0].2.len());
        assert_eq!(turns[1].2, "Looks good.");
        assert_eq!(turns[0].1, "I paid $200 for the writing workshop");
        assert!(turns[0].0.len() >= 2, "a long reply spans several excerpts of one turn");
        assert!(turns[1].1.starts_with("Please review this code") && turns[1].1.len() > 1200, "a long message is joined");
        let mut episodes = Episodes::default();
        for (id, _) in &first {
            episodes.meta.insert(id.clone(), Meta { session: "s1".into(), project: Some("Q".into()), observed_at: None, turn: 0 });
        }
        assert_eq!(episodes.add_turns("s1", Some("Q"), None, &first).len(), 2);
        let second = vec![("ep_b0".to_string(), "[user] What about the budget?\n[assistant] Fine.".to_string())];
        let added = episodes.add_turns("s2", Some("q"), Some("2024/05/02"), &second);
        assert_eq!(episodes.turn_meta[&added[0]].number, 3, "numbering continues in the scope");
        assert_eq!(episodes.meta[&first.last().unwrap().0].turn, 2);
        assert!(episodes.turn_in_scope(&added[0], Some("Q")) && !episodes.turn_in_scope(&added[0], Some("other")));
        assert_eq!(episodes.add_turns("s3", Some("other"), None, &[("ep_c0".into(), "[user] Hi there\n[assistant] Hello".into())]).len(), 1);
        assert_eq!(episodes.turn_meta[&turn_id("ep_c0")].number, 1, "scopes are numbered separately");
    }
    #[test]
    fn one_excerpt_per_turn_and_long_replies_keep_the_question() {
        let long_reply = "word ".repeat(600);
        let entries = [
            entry("user", "I paid $200 for the writing workshop"),
            entry("assistant", "That sounds worthwhile."),
            entry("tool", "ignored tool output"),
            entry("user", "Recommend some sci-fi books"),
            entry("assistant", &long_reply),
        ];
        let chunks = chunks(&entries);
        assert_eq!(
            chunks[0],
            "[user] I paid $200 for the writing workshop\n[assistant] That sounds worthwhile."
        );
        assert!(chunks.iter().all(|c| !c.contains("ignored")));
        assert!(chunks.len() >= 4);
        assert!(chunks[2..].iter().all(|c| c.starts_with("[user] Recommend some sci-fi")));
        let long_question = format!("[user] {}", "why ".repeat(200));
        let excerpts = super::chunks(&[
            entry("user", &long_question),
            entry("assistant", &"answer ".repeat(400)),
        ]);
        assert!(excerpts.len() >= 2);
        assert!(excerpts.iter().all(|e| turn_key(e) == turn_key(&excerpts[0])));
        assert!(chunks.iter().all(|c| c.chars().count() <= CHUNK_CHARS + CONTEXT_CHARS + 20));
    }
    #[test]
    fn windows_cut_cjk_without_spaces() {
        let text = "长".repeat(2500);
        let pieces = windows(&text, 1000);
        assert_eq!(pieces.iter().map(|p| p.chars().count()).collect::<Vec<_>>(), [1000, 1000, 500]);
    }
}
