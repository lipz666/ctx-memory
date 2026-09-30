//! Memory brief: instead of handing a reader dozens of loose memories and excerpts, one
//! model call reads a generous retrieval (about BRIEF_GATHER_TOKENS) and writes what the
//! question needs: the relevant facts in date order, worked-out counts and date
//! differences, the current value of things that changed, contradictions in what the user
//! said, the standing instructions that apply, and a plain statement when the memory has
//! nothing on the point. The brief leads the result; the evidence follows in the rest of
//! the budget, since a brief can miss a detail. Optional (`brief=true` on recall): it
//! costs one model call per search.
use crate::{
    llm,
    recall::{self, Hit},
    store::Store,
};
use anyhow::Result;
use serde_json::json;
use std::time::Duration;

/// Tokens of material gathered for a brief.
pub const BRIEF_GATHER_TOKENS: usize = 12_000;
/// Candidates and conversation excerpts gathered for a brief.
pub const BRIEF_LIMIT: usize = 50;
pub const BRIEF_EPISODES: usize = 12;
/// Size of a brief when the caller gives no budget (tokens).
pub const BRIEF_TOKENS: usize = 800;

const PROMPT: &str = "You prepare a memory brief for an assistant that is about to answer the user's question. You get the question, today's date when known, and material retrieved from the user's long-term memory: standing instructions from the user, unresolved contradictions, a timeline of past conversations, memories (each with the date it was said, and where known the event date, earlier values, and how often it came up) and raw conversation excerpts.

Write the brief: everything in the material that the answer needs, and nothing else.
- State the relevant facts with their dates, oldest first, keeping names, numbers, dates and wording exact.
- When the question needs a count, a total, a date difference, a duration or an order, work it out from the material and state the result with the items or the two dates it rests on. Count each separate event or mention once. An order of things the user brought up follows the conversations from the first to the last, not only the beginning.
- When a value changed over time (a moved deadline, a raised budget, a new count), that is an update, not a contradiction: give the current value, what it was before and when it changed.
- Only when the material has an \"Unresolved contradiction\" item about what is asked: say so first, quote both statements with their dates, and note that the user should be asked which one is correct; do not pick a side. Do not call anything else a contradiction.
- List the user's standing instructions that apply to this kind of question under \"Instructions to follow in the answer:\", and the user's preferences that bear on it (tools, formats, styles they like or avoid) under \"Preferences to respect:\".
- For a summary or an account of how something progressed, cover the whole span from the first conversation to the last, including what the assistant recommended.
- Read the conversation excerpts as closely as the memories: a detail asked for may appear only there. Only when nothing in the material states what is asked, write: \"The memory contains no information about <what is missing>.\" Do not fill the gap with related facts or guesses.

Plain text, no preamble, at most WORDS words.";

/// One retrieved item as the brief writer sees it.
fn render(store: &Store, hit: &Hit) -> String {
    let memory = &hit.memory;
    if matches!(hit.channel, "timeline" | "conflict") {
        return memory.body.clone();
    }
    let date = memory
        .observed_at
        .clone()
        .unwrap_or_else(|| memory.created_at.chars().take(10).collect());
    let label = match (hit.channel, memory.kind.as_str(), memory.status.as_str()) {
        ("episode", _, _) => "conversation excerpt: ",
        (_, "instruction", _) => "standing instruction from the user: ",
        (_, "preference", _) => "the user's preference: ",
        (_, _, "superseded") => "earlier statement, changed later: ",
        _ => "",
    };
    let mut text = format!("[{date}] {label}{}", memory.body);
    if let Some(event) = &memory.event_at {
        text.push_str(&format!(" (event date: {event})"));
    }
    if !memory.mentioned_at.is_empty() {
        text.push_str(&format!(
            " (brought up {} times: first as dated, again on {})",
            memory.mentioned_at.len() + 1,
            memory.mentioned_at.join(", ")
        ));
    }
    let history = recall::history(store, memory, 3);
    if !history.is_empty() {
        let earlier: Vec<String> = history.iter().map(|(date, content)| format!("[{date}] {content}")).collect();
        text.push_str(&format!(" (previously: {})", earlier.join("; ")));
    }
    text
}

/// The brief for `question` from `hits` (already packed to the gathering budget), in at
/// most `tokens` tokens.
pub async fn brief(store: &Store, question: &str, today: Option<&str>, hits: &[Hit], tokens: usize) -> Result<String> {
    let material: Vec<String> = hits.iter().map(|hit| format!("- {}", render(store, hit))).collect();
    let words = (tokens as f64 * 0.7) as usize;
    let input = json!({"question": question, "today": today, "material": material.join("\n")}).to_string();
    let reply = llm::chat(
        store,
        "brief",
        &PROMPT.replace("WORDS", &words.to_string()),
        &input,
        (tokens * 2).max(4000) as u32,
        Duration::from_secs(180),
    )
    .await?;
    Ok(reply.trim().to_owned())
}
