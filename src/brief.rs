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
- When the question has a short factual answer (a count, a date or duration, a value, a name, yes or no), begin with one line \"Answer: ...\": the direct answer as a full sentence that carries what it rests on, so that it can be repeated as it stands: a count names the items counted (\"Two columns: 'category' and 'notes'\"), a date difference names both dates (\"21 days, from March 15, 2024 to April 5, 2024\"), a value says since when. Commit to the one best-supported answer; do not offer alternatives (\"or 5 if...\") unless the user's own statements contradict each other. When the question asks for a summary, an explanation, advice, a plan or how to do something, write no answer line: the assistant composes that answer itself from the facts, instructions and preferences below.
- State the relevant facts with their dates, oldest first, keeping names, numbers, dates and wording exact.
- For a count or a total: count exactly what the question asks for. When it asks what the user mentioned, asked or did, count the user's own statements, not the assistant's suggestions or examples; leave out near matches and repeats of the same item. List the counted items with their dates.
- For a date difference or a duration: use the dates of the two events themselves (event dates, or dates stated in the text), not the dates on which they were talked about. Give the difference in the unit asked.
- For an order of things the user brought up: follow the conversations from the first to the last, not only the beginning.
- When a value changed over time (a moved deadline, a raised budget, a new count), that is an update, not a contradiction. The answer is the most recent value (a statement that explicitly changes the value, such as \"raised to\" or \"moved to\", outweighs an older figure repeated in passing), then what it was before and when it changed. Give the one value asked for, not every related figure.
- A contradiction is an \"Unresolved contradiction\" item in the material, or the user denying something they also reported (\"I have never done X\" against an account of doing X) with no change described; different values of one thing over time are never one. When what is asked is contradicted in this way, the answer line says so, quotes both statements with their dates and notes that the user should be asked which one is correct; do not pick a side.
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
    // The date of the event itself leads: a reader asked how long passed between two
    // events otherwise subtracts the dates of the conversations.
    let mut text = match &memory.event_at {
        Some(event) => format!("[event date {event}; said {date}] {label}{}", memory.body),
        None => format!("[said {date}] {label}{}", memory.body),
    };
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
