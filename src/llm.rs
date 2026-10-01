//! Engine calls to the configured model (extraction, experimental Gate, doctor).
use crate::store::{Store, UsageObservation, credential};
use anyhow::{Result, bail};
use serde_json::{Value, json};
use std::time::Duration;

/// Shared HTTP client: connections are reused instead of a TLS handshake per call.
static CLIENT: std::sync::LazyLock<reqwest::Client> = std::sync::LazyLock::new(|| {
    reqwest::Client::builder()
        .pool_idle_timeout(Duration::from_secs(90))
        .build()
        .expect("http client")
});

/// One chat completion. Counts against the daily engine budget and is logged. Transient
/// failures (connection errors, timeouts, 408/429/5xx, empty replies) are retried with backoff, except
/// for the Gate, whose latency budget does not allow it.
pub async fn chat(
    store: &Store,
    role: &str,
    system: &str,
    user: &str,
    max_tokens: u32,
    timeout: Duration,
) -> Result<String> {
    let messages = [json!({"role":"system","content":system}), json!({"role":"user","content":user})];
    chat_messages(store, role, &messages, max_tokens, timeout).await
}

/// A chat completion over a whole conversation (system, user and assistant messages), with
/// the same budget, logging and retries as `chat`.
pub async fn chat_messages(
    store: &Store,
    role: &str,
    messages: &[Value],
    max_tokens: u32,
    timeout: Duration,
) -> Result<String> {
    let Some(model) = store.config.model.as_ref() else {
        bail!("no model configured (ctx model set)");
    };
    let key = credential(&model.credential_ref)?;
    let url = format!("{}/chat/completions", model.base_url.trim_end_matches('/'));
    let body = json!({"model":model.model,"temperature":0,"max_tokens":max_tokens,"messages":messages});
    let attempts = if role == "gate" { 1 } else { 4 };
    let mut last_error = anyhow::anyhow!("no attempt made");
    for attempt in 0..attempts {
        if attempt > 0 {
            tokio::time::sleep(Duration::from_secs(3 * 3u64.pow(attempt - 1))).await;
        }
        if store.llm_calls_today()? >= store.config.extraction.daily_llm_calls {
            bail!("daily engine LLM call budget reached");
        }
        let request = CLIENT
            .post(&url)
            .timeout(timeout)
            .header(
                "user-agent",
                model.upstream_user_agent.as_deref().unwrap_or("curl/8.0"),
            )
            .bearer_auth(&key)
            .json(&body);
        let response = match request.send().await {
            Ok(response) => response,
            Err(error) => {
                let outcome = if error.is_timeout() {
                    "timeout"
                } else {
                    "network_error"
                };
                let _ = store.record_llm_call(role, &UsageObservation::default(), outcome);
                last_error = error.into();
                continue;
            }
        };
        let status = response.status();
        let raw: Value = response.json().await.unwrap_or(Value::Null);
        let mut observed = UsageObservation::from_response(&raw);
        observed.requested_model = Some(model.model.clone());
        let outcome = if status.is_success() {
            "received"
        } else {
            "upstream_error"
        };
        store.record_llm_call(role, &observed, outcome)?;
        if !status.is_success() {
            last_error = anyhow::anyhow!("model returned {status}");
            if status.as_u16() == 408 || status.as_u16() == 429 || status.is_server_error() {
                continue;
            }
            return Err(last_error);
        }
        // A successful reply without content (gateway hiccup, empty generation) is
        // retried like a transient failure.
        match raw.pointer("/choices/0/message/content").and_then(Value::as_str) {
            Some(content) if !content.trim().is_empty() => return Ok(content.to_owned()),
            _ => {
                let shown: String = raw.to_string().chars().take(600).collect();
                eprintln!("{role}: model reply had no content: {shown}");
                last_error = anyhow::anyhow!("model reply had no content");
            }
        }
    }
    Err(last_error)
}

/// JSON from a model reply, optionally wrapped in one Markdown fence; prose is rejected.
pub fn unfence(text: &str) -> Result<&str> {
    let trimmed = text.trim();
    for prefix in ["```json\n", "```\n"] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            return Ok(rest
                .strip_suffix("```")
                .ok_or_else(|| anyhow::anyhow!("unterminated JSON fence"))?
                .trim());
        }
    }
    Ok(trimmed)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unfences_json_only() {
        assert_eq!(unfence("```json\n{\"a\":1}\n```").unwrap(), "{\"a\":1}");
        assert!(unfence("```json\n{").is_err());
        assert_eq!(unfence(" {} ").unwrap(), "{}");
    }
}
