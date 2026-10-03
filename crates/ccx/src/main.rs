//! ccx: the context-management branch of ctx (see docs/ccx-design.md).
//!
//! `serve --mode shadow` forwards requests unchanged; `--mode on` shortens tool outputs by
//! tier (M1) and answers `ccx_expand` calls itself. Either way each request is recorded as
//! a token composition line (no message content) in `CCX_HOME/steps.jsonl`.
mod composition;
mod digest;
mod expand;
mod objects;
mod report;
mod tiers;

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
    collections::HashMap,
    io::Write,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Instant,
};

const MAX_BODY: usize = 64 * 1024 * 1024;
const MAX_CAPTURE: usize = 8 * 1024 * 1024;
const PIN_CAP: u64 = 1500;

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
    /// Proxy to UPSTREAM; records each request's composition.
    Serve {
        /// Upstream base URL; the request path is appended (e.g. https://host + /v1/chat/completions).
        #[arg(long)]
        upstream: String,
        #[arg(long, default_value_t = 7789)]
        port: u16,
        /// Label stored on every line (a run name); the `X-Ccx-Tag` header overrides it.
        #[arg(long)]
        tag: Option<String>,
        /// `shadow` forwards requests unchanged; `on` shortens tool outputs.
        #[arg(long, value_parser = ["shadow", "on"], default_value = "shadow")]
        mode: String,
        /// Token cap for the latest step's tool outputs.
        #[arg(long, default_value_t = 2500)]
        hot_cap: u64,
        /// Token cap for older tool outputs (their digest size).
        #[arg(long, default_value_t = 400)]
        warm_cap: u64,
        /// Do not offer `ccx_expand` to the model.
        #[arg(long)]
        no_expand: bool,
        /// Internal `ccx_expand` rounds per Agent request.
        #[arg(long, default_value_t = 3)]
        max_expand_rounds: usize,
        /// Read the upstream API key from the first line of stdin and send it on every
        /// request instead of the client's credentials (keeps the key out of argv and env).
        #[arg(long)]
        upstream_key_stdin: bool,
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
    on: bool,
    caps: tiers::Caps,
    expand: bool,
    max_rounds: usize,
    upstream_key: Option<String>,
    objects: objects::Objects,
    /// Per session: object id -> what the model read from it with `ccx_expand` (in memory).
    pins: Mutex<HashMap<String, HashMap<String, String>>>,
}

impl App {
    /// Keep what was read with `ccx_expand`, newest last, about `PIN_CAP` tokens per output.
    fn pin(&self, session: &str, read: Vec<(String, String)>) {
        let Ok(mut pins) = self.pins.lock() else {
            return;
        };
        let session = pins.entry(session.to_owned()).or_default();
        for (id, answer) in read {
            let entry = session.entry(id).or_default();
            if !entry.contains(&answer) {
                let joined = if entry.is_empty() {
                    answer
                } else {
                    format!("{entry}\n{answer}")
                };
                *entry = digest::excerpt(&joined, PIN_CAP);
            }
        }
    }

    fn record(&self, line: &Value) {
        if let Ok(mut log) = self.log.lock() {
            let _ = writeln!(log, "{line}");
        }
    }
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
            mode,
            hot_cap,
            warm_cap,
            no_expand,
            max_expand_rounds,
            upstream_key_stdin,
        } => {
            let upstream_key = if upstream_key_stdin {
                let mut key = String::new();
                std::io::stdin().read_line(&mut key)?;
                let key = key.trim().to_owned();
                anyhow::ensure!(!key.is_empty(), "no key on stdin");
                Some(key)
            } else {
                None
            };
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
                on: mode == "on",
                caps: tiers::Caps {
                    hot: hot_cap,
                    warm: warm_cap,
                },
                expand: !no_expand,
                max_rounds: max_expand_rounds,
                upstream_key,
                objects: objects::Objects::open(&home)?,
                pins: Mutex::new(HashMap::new()),
            });
            let router = Router::new().fallback(handle).with_state(app);
            let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
            eprintln!(
                "ccx ({mode}) on http://127.0.0.1:{port}, recording to {}",
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
    let body = match &app.upstream_key {
        Some(key) => redact(body, key),
        None => body,
    };
    let mut line = None;
    let mut sent = body.clone();
    let mut expand_body = None;
    if method == Method::POST
        && let Some(protocol) = protocol_of(&path)
        && let Ok(parsed) = serde_json::from_slice::<Value>(&body)
    {
        let tag = headers
            .get("x-ccx-tag")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
            .or_else(|| app.tag.clone());
        let stream = parsed
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let session = session_key(protocol, &parsed);
        let mut entry = json!({
            "ts": chrono::Utc::now().to_rfc3339(),
            "tag": tag,
            "path": path,
            "protocol": format!("{protocol:?}").to_lowercase(),
            "model": parsed.get("model"),
            "stream": stream,
            "session": session,
            "est": Composition::of(protocol, &parsed).to_json(),
        });
        if app.on {
            let mut rewritten = parsed;
            let pins = app
                .pins
                .lock()
                .ok()
                .and_then(|p| p.get(&session).cloned())
                .unwrap_or_default();
            let expand = app.expand
                && protocol == Protocol::Chat
                && !stream
                && expand::offer(&mut rewritten);
            match tiers::apply(
                protocol,
                &mut rewritten,
                &app.objects,
                app.caps,
                expand,
                &pins,
            ) {
                Ok(stats) => {
                    entry["ccx"] = stats.to_json();
                    entry["est_sent"] = Composition::of(protocol, &rewritten).to_json();
                    if let Ok(bytes) = serde_json::to_vec(&rewritten) {
                        sent = Bytes::from(bytes);
                        if expand {
                            expand_body = Some(rewritten);
                        }
                    }
                }
                // Never break the request: forward it unchanged and say why.
                Err(e) => entry["ccx_fallback"] = e.to_string().into(),
            }
        }
        line = Some(entry);
    }
    if let Some(body) = expand_body {
        return with_expand(
            app,
            method,
            url,
            headers,
            body,
            line.unwrap_or_default(),
            started,
        )
        .await;
    }
    let upstream = match upstream_request(&app, method, &url, &headers, sent)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_GATEWAY, format!("upstream error: {e}")).into_response(),
    };
    let status =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let response = response_head(status, upstream.headers());
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
            line["status"] = status.as_u16().into();
            line["latency_ms"] = (started.elapsed().as_millis() as u64).into();
            line["usage"] = usage_of(&captured);
            app.record(&line);
        }
    };
    response.body(Body::from_stream(body)).unwrap()
}

/// A non-streaming Chat request that offers `ccx_expand`: answer expand calls here, in
/// internal rounds with the upstream, and return only a response without them.
async fn with_expand(
    app: Arc<App>,
    method: Method,
    url: String,
    headers: HeaderMap,
    mut body: Value,
    mut line: Value,
    started: Instant,
) -> Response {
    let session = line["session"].as_str().unwrap_or_default().to_owned();
    let mut rounds = 0;
    let mut expand_calls = 0;
    let mut usages = vec![];
    loop {
        let bytes = Bytes::from(serde_json::to_vec(&body).unwrap_or_default());
        let request = upstream_request(&app, method.clone(), &url, &headers, bytes);
        let upstream = match request.send().await {
            Ok(r) => r,
            Err(e) => {
                return (StatusCode::BAD_GATEWAY, format!("upstream error: {e}")).into_response();
            }
        };
        let status =
            StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        let head = response_head(status, upstream.headers());
        let captured = match upstream.bytes().await {
            Ok(b) => b,
            Err(e) => return (StatusCode::BAD_GATEWAY, e.to_string()).into_response(),
        };
        usages.push(usage_of(&captured));
        let parsed = serde_json::from_slice::<Value>(&captured).ok();
        let calls = parsed.as_ref().map(expand::calls).unwrap_or_default();
        if !status.is_success() || calls.is_empty() || rounds >= app.max_rounds {
            line["status"] = status.as_u16().into();
            line["latency_ms"] = (started.elapsed().as_millis() as u64).into();
            line["usage"] = sum_usage(&usages);
            line["expand_rounds"] = rounds.into();
            line["expand_calls"] = expand_calls.into();
            app.record(&line);
            return head.body(Body::from(captured)).unwrap();
        }
        rounds += 1;
        expand_calls += calls.len();
        if let Some(parsed) = &parsed {
            let read = expand::append_round(&mut body, parsed, &app.objects, app.caps.hot);
            app.pin(&session, read);
        }
        if rounds >= app.max_rounds {
            expand::withdraw(&mut body);
        }
    }
}

fn upstream_request(
    app: &App,
    method: Method,
    url: &str,
    headers: &HeaderMap,
    body: Bytes,
) -> reqwest::RequestBuilder {
    let mut request = app.client.request(method, url).body(body);
    // `accept-encoding` is dropped so the upstream answers uncompressed: ccx reads usage
    // from the body, and it strips `content-encoding` on the way back.
    for (name, value) in headers {
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
            && !(app.upstream_key.is_some() && (name == "authorization" || name == "x-api-key"))
        {
            request = request.header(name, value);
        }
    }
    if let Some(key) = &app.upstream_key {
        request = request.bearer_auth(key);
    }
    request
}

fn response_head(
    status: StatusCode,
    headers: &reqwest::header::HeaderMap,
) -> axum::http::response::Builder {
    let mut response = Response::builder().status(status);
    for (name, value) in headers {
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
    response
}

fn usage_of(captured: &[u8]) -> Value {
    let usage = UsageObservation::from_bytes(captured);
    json!({
        "input": usage.input_tokens, "output": usage.output_tokens,
        "cached": usage.cached_input_tokens, "model": usage.actual_model,
        "reasoning": usage_field(captured, "/completion_tokens_details/reasoning_tokens")
            .or_else(|| usage_field(captured, "/output_tokens_details/reasoning_tokens")),
        "total": usage_field(captured, "/total_tokens"),
    })
}

/// Usage of several upstream calls for one Agent request: counts summed, last model.
fn sum_usage(usages: &[Value]) -> Value {
    let mut total = json!({});
    for key in ["input", "output", "cached", "reasoning", "total"] {
        let values: Vec<u64> = usages.iter().filter_map(|u| u[key].as_u64()).collect();
        total[key] = if values.is_empty() {
            Value::Null
        } else {
            values.iter().sum::<u64>().into()
        };
    }
    total["model"] = usages.last().map_or(Value::Null, |u| u["model"].clone());
    total
}

/// The request with every occurrence of the upstream key replaced, so a key that leaked
/// into the conversation (e.g. a tool printed a process list) never reaches the model.
fn redact(body: Bytes, key: &str) -> Bytes {
    let needle = key.as_bytes();
    if needle.len() < 8 || !body.windows(needle.len()).any(|w| w == needle) {
        return body;
    }
    let text = String::from_utf8_lossy(&body).replace(key, "[REDACTED]");
    Bytes::from(text.into_bytes())
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
    fn redacts_the_upstream_key() {
        let body = Bytes::from_static(br#"{"content":"API_KEY=sk-secret-123 and sk-secret-123"}"#);
        let out = redact(body.clone(), "sk-secret-123");
        assert_eq!(
            &out[..],
            br#"{"content":"API_KEY=[REDACTED] and [REDACTED]"}"#
        );
        assert_eq!(redact(body.clone(), "absent-key"), body);
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
