//! Search planning for hard questions (`deep` search): one model call turns a question
//! into sub-queries and the date window its answer lies in, then `recall::recall_planned`
//! runs them. Used only when a caller asks for it; ordinary recall makes no model call.
use crate::{llm, recall::Plan, store::Store};
use anyhow::Result;
use serde::Deserialize;
use std::time::Duration;

const PROMPT: &str = "You plan searches over a person's long-term memory (facts and events from earlier conversations, each with a date). Given a question and today's date, return JSON only:
{\"queries\":[\"...\"],\"after\":\"YYYY-MM-DD\"|null,\"before\":\"YYYY-MM-DD\"|null}
- queries: 1 to 4 short search queries that together cover every part of the question. Split comparisons and multi-part questions (one query per thing named: \"considering my form validation, lazy loading and analytics setup, ...\" -> one query for each) (\"did I buy the camera or the lens first?\" -> \"camera purchase\", \"lens purchase\"); name the general category for totals and counts (\"how much did I spend on workshops?\" -> \"workshops attended cost\").
- after/before: the date window the relevant events fall in, computed from today's date, only when the question states or implies one (\"last month\", \"in March\", \"two weeks ago\"); otherwise null.
Do not wrap the JSON in Markdown.";

#[derive(Deserialize)]
struct Reply {
    #[serde(default)]
    queries: Vec<String>,
    #[serde(default)]
    after: Option<String>,
    #[serde(default)]
    before: Option<String>,
}

pub async fn plan(store: &Store, question: &str, today: &str) -> Result<Plan> {
    let input = serde_json::json!({"question": question, "today": today}).to_string();
    let reply = llm::chat(store, "plan", PROMPT, &input, 400, Duration::from_secs(60)).await?;
    parse(&reply)
}

/// Blocking variant for synchronous callers (the MCP server).
pub fn plan_blocking(store: &Store, question: &str, today: &str) -> Result<Plan> {
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()?
                    .block_on(plan(store, question, today))
            })
            .join()
            .map_err(|_| anyhow::anyhow!("planner thread panicked"))?
    })
}

fn parse(reply: &str) -> Result<Plan> {
    let reply: Reply = serde_json::from_str(llm::unfence(reply)?)?;
    let date = |d: Option<String>| d.filter(|d| crate::recall::date_range(d).is_some());
    Ok(Plan {
        queries: reply
            .queries
            .into_iter()
            .map(|q| q.trim().chars().take(300).collect::<String>())
            .filter(|q| !q.is_empty())
            .take(4)
            .collect(),
        after: date(reply.after),
        before: date(reply.before),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_plans_and_drops_bad_dates() {
        let plan = parse("```json\n{\"queries\":[\"camera purchase\",\" \",\"lens purchase\"],\"after\":\"2023-03-01\",\"before\":\"soon\"}\n```").unwrap();
        assert_eq!(plan.queries, ["camera purchase", "lens purchase"]);
        assert_eq!((plan.after.as_deref(), plan.before), (Some("2023-03-01"), None));
        assert_eq!(crate::recall::date_range("2023-03"), Some((20230301, 20230331)));
        assert_eq!(crate::recall::date_range("2023/03/05 (Sun) 10:00"), Some((20230305, 20230305)));
    }
}
