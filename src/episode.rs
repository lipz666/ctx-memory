//! Episodic tier: excerpts of the raw conversations, kept next to the extracted memories.
//!
//! Extraction distills a session into a few memories and inevitably drops details (an
//! amount, a name mentioned in passing, what the assistant recommended). Excerpts keep
//! the original wording; explicit searches (`recall::Mode::Search`) rank them together
//! with the memories. Automatic injection never uses them. Text is stored encrypted like
//! the events it comes from; only the search index lives in memory.
use crate::index::Index;
use std::collections::HashMap;

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
}

#[derive(Default)]
pub struct Episodes {
    pub index: Index,
    pub meta: HashMap<String, Meta>,
}

impl Episodes {
    pub fn in_scope(&self, id: &str, project: Option<&str>) -> bool {
        self.meta.get(id).is_some_and(|m| match (project, &m.project) {
            (Some(wanted), Some(own)) => wanted.eq_ignore_ascii_case(own),
            (None, None) => true,
            _ => false,
        })
    }
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
