//! Memory brief: instead of handing a reader dozens of loose memories and excerpts, one
//! model call reads a generous retrieval (about BRIEF_GATHER_TOKENS, plus the user's own
//! messages in conversation order) and writes what the question needs: the relevant facts
//! in order, worked-out counts and date differences (the day count checked against the
//! calendar), the current value of things that changed, contradictions when asked whether
//! something is so, the standing instructions that apply, and a plain statement when the
//! memory has nothing on the point. The brief leads the result; the evidence follows in
//! the rest of the budget, since a brief can miss a detail. Optional (`brief=true` on
//! recall): it costs one model call per search.
use crate::{
    llm,
    recall::{self, Hit},
    store::Store,
};
use anyhow::Result;
use chrono::NaiveDate;
use regex::Regex;
use serde_json::json;
use std::{sync::LazyLock, time::Duration};

/// Tokens of material gathered for a brief.
pub const BRIEF_GATHER_TOKENS: usize = 12_000;
/// Candidates and conversation excerpts gathered for a brief.
pub const BRIEF_LIMIT: usize = 50;
pub const BRIEF_EPISODES: usize = 12;
/// Size of a brief when the caller gives no budget (tokens), and at most for an account
/// of how things went, whose details would not fit in the shorter brief.
pub const BRIEF_TOKENS: usize = 800;
pub const SUMMARY_BRIEF_TOKENS: usize = 2_000;

/// What the brief writer gets (the first paragraph of its prompt).
pub(crate) const MATERIAL: &str = "You prepare a memory brief for an assistant that is about to answer the user's question. You get the question, today's date when known, and material from the user's long-term memory: standing instructions from the user, unresolved contradictions (only when the question asks whether something is so), a timeline of past conversations, topic dossiers (for a subject, from all conversations: current values with earlier ones, items of each kind, dated events with the kind of date, and its stages), the user's own messages in conversation order (each may be followed by \"→ Assistant:\" and the gist of the reply), memories (each with the date it was said, and where known the event date, earlier values and how often it came up) and raw conversation excerpts. Turn numbers (#) give the order of the conversation: a higher number is later, also on the same date.";

/// How the brief is written.
pub(crate) const RULES: &str = "Write the brief: everything in the material that the answer needs, and nothing else.
- When the question has a short factual answer (a count, a date or duration, a value, a name, yes or no), begin with one line \"Answer: ...\": the direct answer as a full sentence that carries what it rests on, so that it can be repeated as it stands: a count names the items counted (\"Two cities: Lisbon and Porto\"), a date difference names both dates (\"12 days, from June 3, 2024 to June 15, 2024\"), a value says since when. Commit to the one best-supported answer; do not offer alternatives (\"or 5 if...\"). When the memory has nothing on what is asked, the answer line is \"The memory contains no information about <what is missing>.\" When the question asks for a summary, an explanation, advice, a plan or how to do something, write no answer line: the assistant composes that answer itself from the facts, instructions and preferences below.
- State the relevant facts with their dates and turn numbers, in conversation order, keeping names, numbers, dates and wording exact.
- For a count or a total: the user's own messages are the record of what they mentioned, asked, did or planned. Count each distinct item of the kind asked once (numbers the user stated count as stated); count the assistant's suggestions only when the question asks about them. List the counted items with their turn numbers.
- For a date difference or a duration: use the dates of the two events themselves (\"booking a dentist appointment for June 12\" gives June 12; an event date; a date stated in the text), not the dates on which they were talked about.
- When a value changed (a moved deadline, a raised budget, a new count), that is an update, not a contradiction: the answer is the value in the latest statement (highest turn number), then what it was before. Give the one value asked for.
- Raise a contradiction only when the question asks whether something happened or is true (yes/no) and the material has an \"Unresolved contradiction\" item, or the user's own messages both deny and report it with no change described: then the answer line says the records contradict each other, quotes both statements with their dates and says the user should be asked which one is correct. For a count, a date, an amount, a summary or advice, use all the statements and do not raise it.
- For the order in which the user brought things up: go through the user's own messages from the first turn to the last and name what they brought up at points spread over the whole record, in order, not only at the beginning.
- List the user's standing instructions that apply to this kind of question under \"Instructions to follow in the answer:\", and the user's preferences that bear on it (tools, formats, styles they like or avoid) under \"Preferences to respect:\".
- For a summary or an account of how something progressed, cover the whole span from the first conversation to the last, including what the assistant recommended.
- Topic dossiers gather a subject across all conversations: take counts from their item lists, current values from their value lines (with what they replaced), event dates from their event lines (keep planned, scheduled, deadline and done dates apart) and the course of a subject from their stages; check details against the user's messages, since a dossier can miss or merge an item.
- Read the user's messages and the conversation excerpts as closely as the memories: a detail asked for may appear only there. Do not fill a gap with related facts or guesses.
- Before answering, check that the material states the very thing asked. What the user did, used, enforced, outlined or decided comes from the user's own words (their messages, or what they reported adopting); what a named person (a colleague, a coach, customers giving feedback) advised or shared comes from that person as the user reported it; the assistant's own suggestions are neither, unless the question asks what the assistant suggested. When the material covers the topic but not the detail asked (the criteria, the rationale, the format, the specific items), the answer line is \"The memory contains no information about <the detail asked>.\" followed by what is known nearby.

Plain text, no preamble, at most WORDS words.";

/// One retrieved item as the brief writer sees it.
pub(crate) fn render(store: &Store, hit: &Hit) -> String {
    let memory = &hit.memory;
    if matches!(hit.channel, "timeline" | "conflict" | "turnlog" | "timechain") || hit.memory.kind == "dossier" {
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
        (_, "narrative", _) => "account of a conversation: ",
        (_, _, "superseded") => "earlier statement, changed later: ",
        _ => "",
    };
    // The date of the event itself leads: a reader asked how long passed between two
    // events otherwise subtracts the dates of the conversations.
    let turn = hit.turn.map(|t| format!("; turn #{t}")).unwrap_or_default();
    let mut text = match &memory.event_at {
        Some(event) => format!("[event date {event}; said {date}{turn}] {label}{}", memory.body),
        None => format!("[said {date}{turn}] {label}{}", memory.body),
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
    brief_with(store, question, today, hits, "", tokens).await
}

/// `brief`, with what was looked up besides (`agent.rs`) at the end of the material.
pub(crate) async fn brief_with(
    store: &Store,
    question: &str,
    today: Option<&str>,
    hits: &[Hit],
    findings: &str,
    tokens: usize,
) -> Result<String> {
    let mut material: Vec<String> = hits.iter().map(|hit| format!("- {}", render(store, hit))).collect();
    if !findings.is_empty() {
        material.push(format!("- Looked up in the memory for this question:\n{findings}"));
    }
    let words = (tokens as f64 * 0.7) as usize;
    let input = json!({"question": question, "today": today, "material": material.join("\n")}).to_string();
    let reply = llm::chat(
        store,
        "brief",
        &format!("{MATERIAL}\n\n{RULES}").replace("WORDS", &words.to_string()),
        &input,
        (tokens * 2).max(4000) as u32,
        Duration::from_secs(180),
    )
    .await?;
    Ok(check_days(reply.trim()))
}

static DAYS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b(\d{1,4})\s+days?\b").unwrap());
static NAMED_DATE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s+(\d{4}))?").unwrap()
});
static NUMERIC_DATE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b").unwrap());

/// A date as named in text: year if given, month, day.
type Named = (Option<i32>, u32, u32);

/// Dates named in `text` in order of appearance.
pub(crate) fn dates(text: &str) -> Vec<Named> {
    const MONTHS: [&str; 12] = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
    let mut found: Vec<(usize, Named)> = vec![];
    for c in NAMED_DATE.captures_iter(text) {
        let month = MONTHS.iter().position(|m| c[1].eq_ignore_ascii_case(m)).unwrap() as u32 + 1;
        let day = c[2].parse().unwrap_or(0);
        let year = c.get(3).and_then(|y| y.as_str().parse().ok());
        found.push((c.get(0).unwrap().start(), (year, month, day)));
    }
    for c in NUMERIC_DATE.captures_iter(text) {
        found.push((c.get(0).unwrap().start(), (c[1].parse().ok(), c[2].parse().unwrap_or(0), c[3].parse().unwrap_or(0))));
    }
    found.sort_by_key(|f| f.0);
    found.into_iter().map(|f| f.1).collect()
}

/// Language models slip in calendar arithmetic ("67 days, from July 10 to September 12"):
/// when the answer line gives one day count and names two dates, the count is set to the
/// days between them.
pub(crate) fn check_days(brief: &str) -> String {
    let Some(start) = brief.find("Answer:") else { return brief.to_owned() };
    let end = brief[start..].find('\n').map_or(brief.len(), |i| start + i);
    let line = &brief[start..end];
    let counts: Vec<regex::Captures> = DAYS.captures_iter(line).collect();
    let named = dates(line);
    if counts.len() != 1 || named.len() < 2 {
        return brief.to_owned();
    }
    let ((y1, m1, d1), (y2, m2, d2)) = (named[0], named[1]);
    let (Some(y1), Some(y2)) = (y1.or(y2), y2.or(y1)) else { return brief.to_owned() };
    let (Some(first), Some(mut second)) = (NaiveDate::from_ymd_opt(y1, m1, d1), NaiveDate::from_ymd_opt(y2, m2, d2)) else {
        return brief.to_owned();
    };
    // "December 20 to January 5" without years crosses into the next year.
    if second < first && named[1].0.is_none() {
        second = NaiveDate::from_ymd_opt(y2 + 1, m2, d2).unwrap_or(second);
    }
    let days = (second - first).num_days().unsigned_abs();
    let stated: u64 = counts[0][1].parse().unwrap_or(days);
    if stated == days {
        return brief.to_owned();
    }
    let span = counts[0].get(0).unwrap();
    let fixed = format!("{days} {}", if days == 1 { "day" } else { "days" });
    let (from, to) = (start + span.start(), start + span.end());
    format!("{}{}{}", &brief[..from], fixed, &brief[to..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn day_counts_follow_the_calendar() {
        let wrong = "Answer: 67 days passed, from July 10, 2024 to September 12, 2024.\nRelevant facts: ...";
        assert_eq!(check_days(wrong), "Answer: 64 days passed, from July 10, 2024 to September 12, 2024.\nRelevant facts: ...");
        assert_eq!(check_days("Answer: 16 days, from April 5 to April 21, 2024."), "Answer: 16 days, from April 5 to April 21, 2024.");
        assert_eq!(check_days("Answer: 2 days (2024-03-10 to 2024-03-12)."), "Answer: 2 days (2024-03-10 to 2024-03-12).");
        assert_eq!(check_days("Answer: 3 days from 2024-03-12 to 2024-03-30"), "Answer: 18 days from 2024-03-12 to 2024-03-30");
        assert_eq!(check_days("Answer: 5 days, from December 30, 2023 to January 4"), "Answer: 5 days, from December 30, 2023 to January 4");
        assert_eq!(check_days("Answer: 5 days or 3 days, from May 10 to May 15, 2024"), "Answer: 5 days or 3 days, from May 10 to May 15, 2024", "two counts: left alone");
        assert_eq!(check_days("Answer: 8 weeks, from January 15 to March 15, 2024"), "Answer: 8 weeks, from January 15 to March 15, 2024", "weeks: left alone");
        assert_eq!(check_days("No answer line: 67 days, from July 10, 2024 to September 12, 2024"), "No answer line: 67 days, from July 10, 2024 to September 12, 2024");
        assert_eq!(check_days("Answer: 2 days from May 1 to April 30, 2024"), "Answer: 1 day from May 1 to April 30, 2024");
    }
}
