//! ccx: the context-management branch of ctx (see docs/ccx-design.md).
//!
//! M0 is shadow mode only: requests are forwarded unchanged and each one is recorded as a
//! token composition line (no message content) in `CCX_HOME/steps.jsonl`.
mod composition;
mod report;

use anyhow::Result;
use axum::{
    Router,
    body::{Body, Bytes},
    extract::{OriginalUri, State},
    http::{HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response},
};
use clap::{Parser, Subcommand};
use composition::Composition;
use ctx::{inject::Protocol, store::UsageObservation};
use futures_util::StreamExt;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    io::Write,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Instant,
};

const MAX_BODY: usize = 64 * 1024 * 1024;
const MAX_CAPTURE: usize = 8 * 1024 * 1024;

#[derive(Parser)]
#[command(
    name = "ccx",
    version,
    about = "Lightweight context manager for agents"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Shadow proxy: forward to UPSTREAM unchanged and record each request's composition.
    Serve {
        /// Upstream base URL; the request path is appended (e.g. https://host + /v1/chat/completions).
        #[arg(long)]
        upstream: String,
        #[arg(long, default_value_t = 7789)]
        port: u16,
        /// Label stored on every line (a run name); the `X-Ccx-Tag` header overrides it.
        #[arg(long)]
        tag: Option<String>,
    },
    /// Summarize recorded requests: per-request size and where the tokens go.
    Report {
        #[arg(long)]
        tag: Option<String>,
        #[arg(long)]
        json: bool,
    },
}

fn home() -> PathBuf {
    std::env::var_os("CCX_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".ccx"))
}

struct App {
    upstream: String,
    tag: Option<String>,
    client: reqwest::Client,
    log: Mutex<std::fs::File>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let home = home();
    std::fs::create_dir_all(&home)?;
    let steps = home.join("steps.jsonl");
    match cli.command {
        Command::Serve {
            upstream,
            port,
            tag,
        } => {
            ctx::proxy::upstream_url(&upstream, "v1")
                .ok_or_else(|| anyhow::anyhow!("invalid upstream URL"))?;
            let log = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&steps)?;
            let app = Arc::new(App {
                upstream,
                tag,
                client: reqwest::Client::builder().build()?,
                log: Mutex::new(log),
            });
            let router = Router::new().fallback(handle).with_state(app);
            let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
            eprintln!(
                "ccx shadow proxy on http://127.0.0.1:{port}, recording to {}",
                steps.display()
            );
            axum::serve(listener, router).await?;
        }
        Command::Report { tag, json } => report::run(&steps, tag.as_deref(), json)?,
    }
    Ok(())
}

/// Protocol from a request path such as `/v1/chat/completions` or `/messages`.
fn protocol_of(path: &str) -> Option<Protocol> {
    let path = path.trim_start_matches('/');
    Protocol::from_path(path.strip_prefix("v1/").unwrap_or(path))
}

/// Groups the steps of one conversation: system text plus the first conversation item.
/// Two conversations that open with the same task text share a key.
fn session_key(protocol: Protocol, body: &Value) -> String {
    let mut hasher = Sha256::new();
    hasher.update(ctx::inject::system_text(protocol, body));
    for item in ctx::inject::messages(protocol, body)
        .into_iter()
        .flatten()
        .filter(|m| {
            !matches!(
                m.get("role").and_then(Value::as_str),
                Some("system" | "developer")
            )
        })
        .take(1)
    {
        hasher.update(item.to_string());
    }
    hex::encode(hasher.finalize())[..12].to_owned()
}

async fn handle(
    State(app): State<Arc<App>>,
    OriginalUri(uri): OriginalUri,
    method: Method,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let started = Instant::now();
    let path = uri.path().to_owned();
    let Some(mut url) = ctx::proxy::upstream_url(&app.upstream, &path) else {
        return (StatusCode::BAD_REQUEST, "invalid path").into_response();
    };
    if let Some(query) = uri.query() {
        url = format!("{url}?{query}");
    }
    let body = match axum::body::to_bytes(body, MAX_BODY).await {
        Ok(b) => b,
        Err(e) => return (StatusCode::PAYLOAD_TOO_LARGE, e.to_string()).into_response(),
    };
    let mut line = None;
    if method == Method::POST
        && let Some(protocol) = protocol_of(&path)
        && let Ok(parsed) = serde_json::from_slice::<Value>(&body)
    {
        let tag = headers
            .get("x-ccx-tag")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
            .or_else(|| app.tag.clone());
        line = Some(json!({
            "ts": chrono::Utc::now().to_rfc3339(),
            "tag": tag,
            "path": path,
            "protocol": format!("{protocol:?}").to_lowercase(),
            "model": parsed.get("model"),
            "stream": parsed.get("stream").and_then(Value::as_bool).unwrap_or(false),
            "session": session_key(protocol, &parsed),
            "est": Composition::of(protocol, &parsed).to_json(),
        }));
    }
    let mut request = app.client.request(method, &url).body(body);
    // `accept-encoding` is dropped so the upstream answers uncompressed: ccx reads usage
    // from the body, and it strips `content-encoding` on the way back.
    for (name, value) in &headers {
        if ![
            "host",
            "content-length",
            "connection",
            "transfer-encoding",
            "te",
            "upgrade",
            "accept-encoding",
        ]
        .contains(&name.as_str())
            && !name.as_str().starts_with("x-ccx-")
        {
            request = request.header(name, value);
        }
    }
    let upstream = match request.send().await {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_GATEWAY, format!("upstream error: {e}")).into_response(),
    };
    let status =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut response = Response::builder().status(status);
    for (name, value) in upstream.headers() {
        if ![
            "connection",
            "transfer-encoding",
            "content-length",
            "content-encoding",
        ]
        .contains(&name.as_str())
        {
            response = response.header(name, value);
        }
    }
    let mut stream = upstream.bytes_stream();
    let body = async_stream::stream! {
        let mut captured = Vec::new();
        while let Some(item) = stream.next().await {
            let chunk = match item {
                Ok(chunk) => chunk,
                Err(e) => { yield Err(std::io::Error::other(e)); break; }
            };
            if captured.len() + chunk.len() <= MAX_CAPTURE {
                captured.extend_from_slice(&chunk);
            }
            yield Ok::<Bytes, std::io::Error>(chunk);
        }
        if let Some(mut line) = line {
            let usage = UsageObservation::from_bytes(&captured);
            line["status"] = status.as_u16().into();
            line["latency_ms"] = (started.elapsed().as_millis() as u64).into();
            line["usage"] = json!({
                "input": usage.input_tokens, "output": usage.output_tokens,
                "cached": usage.cached_input_tokens, "model": usage.actual_model,
                "reasoning": usage_field(&captured, "/completion_tokens_details/reasoning_tokens")
                    .or_else(|| usage_field(&captured, "/output_tokens_details/reasoning_tokens")),
                "total": usage_field(&captured, "/total_tokens"),
            });
            if let Ok(mut log) = app.log.lock() {
                let _ = writeln!(log, "{line}");
            }
        }
    };
    response.body(Body::from_stream(body)).unwrap()
}

/// A number under `usage` in a JSON or SSE response body (the last one wins).
fn usage_field(body: &[u8], pointer: &str) -> Option<u64> {
    let read = |v: &Value| {
        v.get("usage")
            .or_else(|| v.pointer("/response/usage"))
            .and_then(|u| u.pointer(pointer))
            .and_then(Value::as_u64)
    };
    if let Ok(value) = serde_json::from_slice::<Value>(body) {
        return read(&value);
    }
    String::from_utf8_lossy(body)
        .lines()
        .rev()
        .filter_map(|l| l.strip_prefix("data:"))
        .filter_map(|d| serde_json::from_str::<Value>(d.trim()).ok())
        .find_map(|v| read(&v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_reasoning_tokens_from_json_and_sse() {
        let body = br#"{"usage":{"total_tokens":115,"completion_tokens_details":{"reasoning_tokens":44}}}"#;
        assert_eq!(usage_field(body, "/total_tokens"), Some(115));
        assert_eq!(
            usage_field(body, "/completion_tokens_details/reasoning_tokens"),
            Some(44)
        );
        let sse =
            b"data: {\"choices\":[]}\n\ndata: {\"usage\":{\"total_tokens\":9}}\n\ndata: [DONE]\n";
        assert_eq!(usage_field(sse, "/total_tokens"), Some(9));
    }

    #[test]
    fn protocol_from_paths() {
        assert_eq!(protocol_of("/v1/chat/completions"), Some(Protocol::Chat));
        assert_eq!(protocol_of("/messages"), Some(Protocol::Anthropic));
        assert_eq!(protocol_of("/v1/models"), None);
    }

    #[test]
    fn session_key_ignores_later_steps() {
        let a = json!({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]});
        let mut b = a.clone();
        b["messages"]
            .as_array_mut()
            .unwrap()
            .push(json!({"role": "assistant", "content": "x"}));
        b["messages"]
            .as_array_mut()
            .unwrap()
            .push(json!({"role": "user", "content": "y"}));
        let c = json!({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "other"}]});
        assert_eq!(
            session_key(Protocol::Chat, &a),
            session_key(Protocol::Chat, &b)
        );
        assert_ne!(
            session_key(Protocol::Chat, &a),
            session_key(Protocol::Chat, &c)
        );
    }
}
