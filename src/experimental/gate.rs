//! Experimental Gate: an LLM decides whether a decision point needs memory search.
//! Synchronous (≤300 ms) for a local model, deferred to the next step for a remote one.
use crate::{llm, store::Store};
use serde::Deserialize;
use serde_json::{Value, json};
use std::time::{Duration, Instant};

const PROMPT: &str = include_str!("../../prompts/gate-v1.txt");

#[derive(Deserialize)]
struct Reply {
    recall: bool,
    #[serde(default)]
    why: String,
    #[serde(default)]
    queries: Vec<String>,
    #[serde(default)]
    card_ids: Vec<String>,
}
pub struct Decision {
    /// Run hybrid search this step.
    pub search: bool,
    pub query: String,
    pub direct_ids: Vec<String>,
    pub trace: Value,
}
impl Decision {
    /// What a failed or pending Gate falls back to: normal search on the step's text.
    pub fn fallback(query: &str, reason: &str, called: bool) -> Self {
        Self {
            search: true,
            query: query.into(),
            direct_ids: vec![],
            trace: json!({"called":called,"decision":"fallback","reason":reason}),
        }
    }
}

pub fn is_decision_point(features: &Value) -> bool {
    features.get("role").and_then(Value::as_str) == Some("user")
        || features.get("error_sig").is_some()
}

/// Remote gateways cannot answer inside 300 ms; their decision applies to the next step.
pub fn deferred(store: &Store) -> bool {
    match store.config.experimental.gate_mode.as_str() {
        "sync" => false,
        "deferred" => true,
        _ => store.config.model.as_ref().is_none_or(|model| {
            reqwest::Url::parse(&model.base_url)
                .ok()
                .and_then(|url| url.host_str().map(str::to_owned))
                .is_none_or(|host| {
                    !matches!(host.as_str(), "127.0.0.1" | "localhost" | "[::1]" | "::1")
                })
        }),
    }
}
pub fn timeout(store: &Store, deferred: bool) -> Duration {
    let config = &store.config.experimental;
    if deferred {
        Duration::from_millis(config.gate_deferred_timeout_ms.clamp(1, 60_000))
    } else {
        Duration::from_millis(config.gate_timeout_ms.clamp(1, 300))
    }
}

pub async fn decide(store: &Store, features: &Value, timeout: Duration) -> Decision {
    let query = features
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let cards: Vec<Value> = store
        .memories()
        .into_iter()
        .filter(|m| m.status == "active")
        .take(20)
        .map(|m| json!({"id":m.id,"title":m.title.chars().take(60).collect::<String>()}))
        .collect();
    if cards.is_empty() {
        return Decision {
            search: false,
            query,
            direct_ids: vec![],
            trace: json!({"called":false,"decision":"no_memories"}),
        };
    }
    let mut compact = features.clone();
    compact["text"] = Value::String(query.chars().take(800).collect());
    let input = crate::store::redact_value(&json!({"features":compact,"cards":cards})).to_string();
    let started = Instant::now();
    let reply = llm::chat(store, "gate", PROMPT, &input, 256, timeout).await;
    let elapsed = started.elapsed().as_millis() as u64;
    let reason = match &reply {
        Err(error) if error.to_string().contains("budget") => "daily_budget",
        Err(error) if format!("{error:#}").contains("timed out") => "timeout",
        Err(_) => "upstream_error",
        Ok(_) => "",
    };
    let Ok(text) = reply else {
        let mut decision = Decision::fallback(&query, reason, true);
        decision.trace["latency_ms"] = elapsed.into();
        return decision;
    };
    let Some(parsed) = llm::unfence(&text)
        .ok()
        .and_then(|json| serde_json::from_str::<Reply>(json).ok())
    else {
        return Decision::fallback(&query, "invalid_json", true);
    };
    Decision {
        search: parsed.recall,
        query: parsed
            .queries
            .into_iter()
            .find(|q| !q.trim().is_empty())
            .unwrap_or(query),
        direct_ids: parsed
            .card_ids
            .into_iter()
            .filter(|id| id.starts_with("mem_"))
            .take(3)
            .collect(),
        trace: json!({"called":true,"decision":parsed.recall,"why":parsed.why,"latency_ms":elapsed}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decision_points_and_modes() {
        assert!(is_decision_point(&json!({"role":"user"})));
        assert!(!is_decision_point(&json!({"role":"tool"})));
        let dir = tempfile::tempdir().unwrap();
        crate::store::init(dir.path()).unwrap();
        let mut store = Store::open(dir.path()).unwrap();
        let model = |url: &str| crate::config::ModelConfig {
            base_url: url.into(),
            model: "m".into(),
            credential_ref: "env:X".into(),
            upstream_user_agent: None,
        };
        store.config.model = Some(model("https://gateway.example/v1"));
        assert!(deferred(&store));
        store.config.experimental.gate_timeout_ms = 5_000;
        assert_eq!(timeout(&store, false), Duration::from_millis(300));
        store.config.model = Some(model("http://127.0.0.1:11434/v1"));
        assert!(!deferred(&store));
    }
}
