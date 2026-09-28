//! The Agent-facing proxy: `/a/{agent}/v1/...` and `/a/{agent}/p/{project}/v1/...`.
//!
//! Each JSON request of a known protocol is tied to a session. New messages go to the
//! transcript; on a new user turn or a new error, memories are recalled and attached to
//! that message, and they stay attached on later steps. Always-on rules go into the
//! system prompt. With nothing to add, the original bytes are forwarded unchanged.
use crate::{
    experimental::{action_guard, eviction, gate},
    inject::{self, Protocol},
    recall::{self, Hit, Query},
    server::{App, PendingGate},
    session::{self, Anchor},
    store::{Store, UsageObservation},
};
use axum::{
    body::{Body, Bytes},
    extract::{OriginalUri, Path, State},
    http::{HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response},
};
use futures_util::{StreamExt, TryStreamExt};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::{Duration, Instant},
};

const MAX_BODY: usize = 32 * 1024 * 1024;
const PENDING_GATE_TTL: Duration = Duration::from_secs(600);

struct Meta {
    step_id: Option<String>,
    recall_count: usize,
    agent: String,
    session: Option<String>,
    project: Option<String>,
    upstream_key: Option<String>,
    upstream_user_agent: Option<String>,
    protocol: Option<Protocol>,
    request: Option<Value>,
    hot_path_ms: f64,
}

pub async fn handle(
    State(app): State<App>,
    Path(params): Path<HashMap<String, String>>,
    OriginalUri(uri): OriginalUri,
    method: Method,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let started = Instant::now();
    if let Err(e) = crate::api::require(&headers, &app.store) {
        return e.into_response();
    }
    let agent = params.get("agent").cloned().unwrap_or_default();
    let path = params.get("path").cloned().unwrap_or_default();
    let route_project = params.get("project").cloned();
    let Some(config) = app.store.config.agents.get(&agent) else {
        return (StatusCode::NOT_FOUND, "unknown agent").into_response();
    };
    let Some(mut url) = upstream_url(&config.upstream, &path) else {
        return (StatusCode::BAD_REQUEST, "invalid upstream configuration").into_response();
    };
    if let Some(query) = uri.query() {
        url.push('?');
        url.push_str(query);
    }
    let mut meta = Meta {
        step_id: None,
        recall_count: 0,
        agent: agent.clone(),
        session: None,
        project: None,
        upstream_key: app.upstream_keys.get(&agent).cloned(),
        upstream_user_agent: config.upstream_user_agent.clone(),
        protocol: None,
        request: None,
        hot_path_ms: 0.0,
    };
    let meter_only = config.meter_only;
    let declared = headers
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok());
    if declared.is_some_and(|bytes| bytes > MAX_BODY) {
        let stream = body.into_data_stream().map_err(std::io::Error::other);
        meta.hot_path_ms = started.elapsed().as_secs_f64() * 1000.0;
        return forward(
            &app,
            method,
            &headers,
            url,
            reqwest::Body::wrap_stream(stream),
            meta,
        )
        .await;
    }
    let mut input = body.into_data_stream();
    let mut buffered = Vec::new();
    while let Some(chunk) = input.next().await {
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
        };
        buffered.extend_from_slice(&chunk);
        if buffered.len() > MAX_BODY {
            let replay = futures_util::stream::once(async move {
                Ok::<Bytes, std::io::Error>(Bytes::from(buffered))
            })
            .chain(input.map_err(std::io::Error::other));
            meta.hot_path_ms = started.elapsed().as_secs_f64() * 1000.0;
            return forward(
                &app,
                method,
                &headers,
                url,
                reqwest::Body::wrap_stream(replay),
                meta,
            )
            .await;
        }
    }
    let original = Bytes::from(buffered);
    let mut sent = original.clone();
    let protocol = Protocol::from_path(&path);
    if method == Method::POST
        && headers.get("x-ctx-bypass").is_none()
        && let Some(protocol) = protocol
        && let Ok(raw) = serde_json::from_slice::<Value>(&original)
    {
        let features = inject::features(protocol, &raw);
        let key = session::key(&agent, &headers, protocol, &raw);
        if meter_only {
            meta.step_id = app
                .store
                .event(
                    &agent,
                    None,
                    Some(&key),
                    "request",
                    &features,
                    &json!({"meter_only":true}),
                )
                .ok();
            if let Some(id) = &meta.step_id {
                let _ = app
                    .store
                    .record_step(id, &json!({"meter_only":true,"modified":false}));
            }
        } else {
            let gate = gate_decision(&app, &key, &features).await;
            let task_app = app.clone();
            let task_headers = headers.clone();
            let task_agent = agent.clone();
            let prepared = tokio::task::spawn_blocking(move || {
                prepare(
                    &task_app,
                    &task_agent,
                    route_project.as_deref(),
                    &task_headers,
                    protocol,
                    &raw,
                    features,
                    &key,
                    gate,
                )
            })
            .await;
            match prepared {
                Ok(Ok(result)) => {
                    if let Some(bytes) = result.body {
                        sent = Bytes::from(bytes);
                    }
                    meta.step_id = result.step_id;
                    meta.recall_count = result.recall_count;
                    meta.session = Some(result.session);
                    meta.project = result.project;
                }
                Ok(Err(error)) => {
                    eprintln!("ctx proxy: {}", crate::store::redact(&format!("{error:#}")))
                }
                Err(error) => eprintln!("ctx proxy: {error}"),
            }
        }
        meta.protocol = Some(protocol);
        meta.request = serde_json::from_slice(&sent).ok();
    }
    meta.hot_path_ms = started.elapsed().as_secs_f64() * 1000.0;
    forward(&app, method, &headers, url, reqwest::Body::from(sent), meta).await
}

/// Experimental Gate, when enabled: local classifier, synchronous call, or the deferred
/// decision from this session's previous step.
async fn gate_decision(app: &App, key: &str, features: &Value) -> Option<gate::Decision> {
    if !app.store.config.experimental.gate_enabled {
        return None;
    }
    let query = features.get("text").and_then(Value::as_str).unwrap_or("");
    if let Some(model) = &app.classifier {
        let (recall, probability) = model.predict(features);
        return Some(gate::Decision {
            search: recall,
            query: query.into(),
            direct_ids: vec![],
            trace: json!({"called":false,"decision":recall,"source":"local_classifier","probability":probability}),
        });
    }
    let decision_point = gate::is_decision_point(features);
    if !gate::deferred(&app.store) {
        if !decision_point {
            return None;
        }
        let mut decision =
            gate::decide(&app.store, features, gate::timeout(&app.store, false)).await;
        decision.trace["mode"] = "sync".into();
        return Some(decision);
    }
    let pending = {
        let mut map = app.pending_gate.lock().unwrap();
        map.retain(|_, p| p.at.elapsed() < PENDING_GATE_TTL);
        map.remove(key)
    };
    if decision_point {
        let (store, map, slot, features) = (
            app.store.clone(),
            app.pending_gate.clone(),
            key.to_owned(),
            features.clone(),
        );
        tokio::spawn(async move {
            let mut decision = gate::decide(&store, &features, gate::timeout(&store, true)).await;
            // Labels for this decision must use the features it was made on.
            decision.trace["source_features"] = crate::store::redact_value(&features);
            map.lock().unwrap().insert(
                slot,
                PendingGate {
                    at: Instant::now(),
                    decision,
                },
            );
        });
    }
    match pending {
        Some(p) => {
            let mut decision = p.decision;
            decision.trace["mode"] = "deferred".into();
            Some(decision)
        }
        None if decision_point => {
            let mut decision = gate::Decision::fallback(query, "deferred_pending", false);
            decision.trace["mode"] = "deferred".into();
            Some(decision)
        }
        None => None,
    }
}

struct Prepared {
    body: Option<Vec<u8>>,
    step_id: Option<String>,
    recall_count: usize,
    session: String,
    project: Option<String>,
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    app: &App,
    agent: &str,
    route_project: Option<&str>,
    headers: &HeaderMap,
    protocol: Protocol,
    raw: &Value,
    mut features: Value,
    key: &str,
    gate: Option<gate::Decision>,
) -> anyhow::Result<Prepared> {
    let store = &app.store;
    let items_len = inject::messages(protocol, raw).map(Vec::len).unwrap_or(1);
    // Session bookkeeping: project, transcript delta, current anchors.
    let (project, from, delta, prior_anchors, injected, cursor) = app.tracker.with(key, |state| {
        if !state.loaded {
            state.loaded = true;
            if let Ok(Some((seen, hash, project))) = store.session_cursor(key) {
                state.seen = seen;
                state.last_hash = hash;
                state.project = project;
            }
        }
        if (state.project.is_none()
            || headers.get("x-ctx-project").is_some()
            || route_project.is_some())
            && let Some(found) = session::detect_project(headers, route_project, protocol, raw)
        {
            state.project = Some(found);
        }
        let (from, delta) = session::new_messages(state, protocol, raw);
        (
            state.project.clone(),
            from,
            delta,
            state.anchors.clone(),
            state.injected.clone(),
            (state.seen, state.last_hash.clone()),
        )
    });
    let latest_is_new = items_len > from && !delta.is_empty();
    let user_turn = latest_is_new && features.get("role").and_then(Value::as_str) == Some("user");
    let error_turn = latest_is_new && features.get("error_sig").is_some();
    features["session"] = key.into();
    if let Some(project) = &project {
        features["project"] = project.clone().into();
    }
    // What to search for this step.
    let mut search = if user_turn {
        features
            .get("user_text")
            .and_then(Value::as_str)
            .map(str::to_owned)
    } else if error_turn {
        Some(format!(
            "{} {}\n{}",
            features.get("tool").and_then(Value::as_str).unwrap_or(""),
            features
                .get("error_sig")
                .and_then(Value::as_str)
                .unwrap_or(""),
            features
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .chars()
                .take(600)
                .collect::<String>()
        ))
    } else {
        None
    };
    let mut extra_ids = vec![];
    if let Some(decision) = &gate {
        if !decision.search {
            search = None;
        } else if search.is_some() && !decision.query.is_empty() {
            search = Some(decision.query.clone());
        }
        extra_ids = decision.direct_ids.clone();
    }
    let hits = recall::recall(
        store,
        &Query {
            text: search.as_deref(),
            project: project.as_deref(),
            features: latest_is_new.then_some(&features),
            limit: store.config.recall.max_injected,
            always_on: true,
            exclude: Some(&injected),
            extra_ids: &extra_ids,
            ..Default::default()
        },
    )?;
    let mut hits = hits;
    if latest_is_new {
        hits.extend(
            action_guard::advisory_hits(store, protocol, raw, project.as_deref())
                .into_iter()
                .filter(|h| !injected.contains(&h.memory.id)),
        );
    }
    let (rules, fresh): (Vec<Hit>, Vec<Hit>) = hits.into_iter().partition(|h| h.channel == "rule");
    let mut seen_ids = HashSet::new();
    let fresh: Vec<Hit> = fresh
        .into_iter()
        .filter(|h| seen_ids.insert(h.memory.id.clone()))
        .collect();
    // Keep earlier anchors whose message is unchanged; attach new memories to the latest message.
    let items = inject::messages(protocol, raw);
    let valid = |anchor: &Anchor| match items {
        Some(items) => items
            .get(anchor.index)
            .is_some_and(|m| session::message_hash(m) == anchor.hash),
        None => anchor.index == 0,
    };
    let mut anchors: Vec<Anchor> = prior_anchors.into_iter().filter(valid).collect();
    let mut new_ids = vec![];
    if !fresh.is_empty()
        && let Some(index) = inject::anchor_index(protocol, raw)
        && let Some((block, ids)) = inject::render_block(&fresh, store.config.recall.budget_tokens)
    {
        let hash = items
            .and_then(|items| items.get(index))
            .map(session::message_hash)
            .unwrap_or_default();
        new_ids = ids.clone();
        anchors.push(Anchor {
            index,
            hash,
            block,
            ids,
        });
    }
    app.tracker.with(key, |state| {
        state.anchors = anchors.clone();
        state.injected = anchors.iter().flat_map(|a| a.ids.iter().cloned()).collect();
    });
    let system = inject::render_block(&rules, 2000).map(|(block, _)| block);
    let mut output = raw.clone();
    let mut evictions = vec![];
    if store.config.experimental.eviction_enabled {
        let (compacted, changed) =
            eviction::compact_history(store, protocol, raw, agent, project.as_deref(), Some(key))?;
        output = compacted;
        evictions = changed;
    }
    let anchor_blocks: Vec<(usize, String)> =
        anchors.iter().map(|a| (a.index, a.block.clone())).collect();
    let modified = system.is_some() || !anchor_blocks.is_empty() || !evictions.is_empty();
    if system.is_some() || !anchor_blocks.is_empty() {
        output = inject::apply(protocol, &output, system.as_deref(), &anchor_blocks)?;
    }
    // Record the step.
    let payload = json!({"messages":delta,"from":from,"model":raw.get("model")});
    let step = store.event(
        agent,
        project.as_deref(),
        Some(key),
        "request",
        &features,
        &payload,
    )?;
    store.touch_session(
        key,
        agent,
        project.as_deref(),
        user_turn,
        (cursor.0, cursor.1.as_deref()),
    )?;
    for hit in fresh.iter().filter(|h| new_ids.contains(&h.memory.id)) {
        let _ = store.record_recall(&step, &hit.memory.id, hit.channel, hit.score, &hit.reason);
    }
    let _ = store.record_step(
        &step,
        &json!({
            "session": key, "project": project, "user_turn": user_turn, "error_turn": error_turn,
            "searched": search.is_some(),
            "injected": new_ids,
            "candidates": fresh.iter().map(|h| json!({"memory_id":h.memory.id,"channel":h.channel,"score":h.score,"reason":h.reason})).collect::<Vec<_>>(),
            "anchors": anchors.len(), "rules": rules.len(), "evictions": evictions,
            "modified": modified, "gate": gate.as_ref().map(|g| &g.trace),
        }),
    );
    let body = if modified {
        let bytes = serde_json::to_vec(&output)?;
        if store.config.debug_capture {
            let _ = store.event(
                agent,
                project.as_deref(),
                Some(key),
                "compiled",
                &json!({"step_event":step}),
                &output,
            );
        }
        Some(bytes)
    } else {
        None
    };
    Ok(Prepared {
        body,
        step_id: Some(step),
        recall_count: new_ids.len(),
        session: key.to_owned(),
        project,
    })
}

async fn forward(
    app: &App,
    method: Method,
    headers: &HeaderMap,
    url: String,
    body: reqwest::Body,
    meta: Meta,
) -> Response {
    let requested_model = meta
        .request
        .as_ref()
        .and_then(|b| b.get("model"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let request = outbound_request(
        app,
        method,
        headers,
        &url,
        body,
        meta.upstream_key.as_deref(),
        meta.upstream_user_agent.as_deref(),
    );
    let upstream = match request.send().await {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_GATEWAY, format!("upstream error: {e}")).into_response(),
    };
    let status =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut response = Response::builder().status(status);
    for (name, value) in upstream.headers() {
        if [
            "connection",
            "transfer-encoding",
            "content-length",
            "content-encoding",
            "etag",
            "content-md5",
        ]
        .contains(&name.as_str())
        {
            continue;
        }
        response = response.header(name, value);
    }
    if let Some(id) = &meta.step_id {
        response = response.header("x-ctx-step", id);
    }
    response = response
        .header("x-ctx-recalls", meta.recall_count.to_string())
        .header("x-ctx-hot-path-ms", format!("{:.3}", meta.hot_path_ms));
    let streaming = upstream
        .headers()
        .get("content-type")
        .and_then(|h| h.to_str().ok())
        .is_some_and(|v| v.contains("text/event-stream"));
    let bounded = upstream
        .content_length()
        .is_none_or(|n| n <= 8 * 1024 * 1024);
    if app.store.config.experimental.action_guard == "recheck"
        && status.is_success()
        && !streaming
        && bounded
        && let (Some(protocol), Some(request_body), Some(step)) =
            (meta.protocol, meta.request.clone(), meta.step_id.clone())
    {
        let original = match upstream.bytes().await {
            Ok(value) => value,
            Err(error) => return (StatusCode::BAD_GATEWAY, error.to_string()).into_response(),
        };
        let mut delivered = original.to_vec();
        if let Ok(value) = serde_json::from_slice::<Value>(&original)
            && let Some((rechecked, changed)) = recheck(
                app,
                headers,
                &url,
                &meta,
                protocol,
                &request_body,
                &value,
                &step,
            )
            .await
        {
            delivered = serde_json::to_vec(&rechecked).unwrap_or(delivered);
            response = response.header(
                "x-ctx-recheck",
                if changed { "changed" } else { "confirmed" },
            );
        }
        finish(&app.store, &meta, requested_model, &delivered, false);
        return response.body(Body::from(delivered)).unwrap();
    }
    let mut stream = upstream.bytes_stream();
    let store = app.store.clone();
    let body = async_stream::stream! {
        let mut captured = Vec::new();
        let mut truncated = false;
        while let Some(item) = stream.next().await {
            let chunk = match item { Ok(value) => value, Err(error) => { yield Err(std::io::Error::other(error)); return; } };
            if captured.len() + chunk.len() <= 8 * 1024 * 1024 { captured.extend_from_slice(&chunk); } else { truncated = true; }
            yield Ok::<Bytes, std::io::Error>(chunk);
        }
        finish(&store, &meta, requested_model, &captured, truncated);
    };
    response.body(Body::from_stream(body)).unwrap()
}

/// After the response: usage, the assistant's text for the transcript, and citations.
fn finish(
    store: &Arc<Store>,
    meta: &Meta,
    requested_model: Option<String>,
    body: &[u8],
    truncated: bool,
) {
    let Some(step) = &meta.step_id else {
        return;
    };
    let mut observed = UsageObservation::from_bytes(body);
    observed.requested_model = requested_model;
    let _ = store.record_usage(step, &meta.agent, &observed);
    let text = response_text(body);
    let _ = crate::feedback::infer_citations(store, step, &text);
    let clipped: String = text.chars().take(8000).collect();
    let _ = store.event(
        &meta.agent,
        meta.project.as_deref(),
        meta.session.as_deref(),
        "response",
        &json!({"step_event":step}),
        &json!({"text":crate::store::redact(&clipped),"truncated":truncated}),
    );
}

/// The assistant text of a JSON or SSE response in any of the three protocols.
pub fn response_text(body: &[u8]) -> String {
    fn from_value(value: &Value, out: &mut String) {
        let pieces = [
            value.pointer("/choices/0/message/content"),
            value.pointer("/choices/0/delta/content"),
            value.pointer("/delta/text"),
            value.get("output_text"),
        ];
        for piece in pieces.into_iter().flatten() {
            if let Some(s) = piece.as_str() {
                out.push_str(s);
            }
        }
        if value.get("type").and_then(Value::as_str) == Some("response.output_text.delta")
            && let Some(s) = value.get("delta").and_then(Value::as_str)
        {
            out.push_str(s);
        }
        for key in ["content", "output"] {
            if let Some(items) = value.get(key).and_then(Value::as_array) {
                for item in items {
                    if matches!(
                        item.get("type").and_then(Value::as_str),
                        Some("text" | "output_text")
                    ) && let Some(s) = item.get("text").and_then(Value::as_str)
                    {
                        out.push_str(s);
                    }
                    if let Some(parts) = item.get("content").and_then(Value::as_array) {
                        for part in parts {
                            if part.get("type").and_then(Value::as_str) == Some("output_text")
                                && let Some(s) = part.get("text").and_then(Value::as_str)
                            {
                                out.push_str(s);
                            }
                        }
                    }
                }
            }
        }
    }
    let mut out = String::new();
    if let Ok(value) = serde_json::from_slice::<Value>(body) {
        from_value(&value, &mut out);
        return out;
    }
    for line in String::from_utf8_lossy(body).lines() {
        if let Some(data) = line.strip_prefix("data:")
            && let Ok(value) = serde_json::from_str::<Value>(data.trim())
        {
            from_value(&value, &mut out);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
async fn recheck(
    app: &App,
    headers: &HeaderMap,
    url: &str,
    meta: &Meta,
    protocol: Protocol,
    request: &Value,
    response: &Value,
    step: &str,
) -> Option<(Value, bool)> {
    let proposal = action_guard::response_call(protocol, response)?;
    let hits = action_guard::before_action_hits(&app.store, &proposal, meta.project.as_deref());
    if hits.is_empty()
        || app.store.llm_calls_today().unwrap_or(u32::MAX)
            >= app.store.config.extraction.daily_llm_calls
        || !app
            .store
            .reserve_recheck(meta.session.as_deref().unwrap_or(step), step)
            .unwrap_or(false)
    {
        return None;
    }
    let body = action_guard::recheck_body(protocol, request, &proposal, &hits)?;
    let outbound = outbound_request(
        app,
        Method::POST,
        headers,
        url,
        reqwest::Body::from(serde_json::to_vec(&body).ok()?),
        meta.upstream_key.as_deref(),
        meta.upstream_user_agent.as_deref(),
    )
    .header("accept", "application/json");
    let result = tokio::time::timeout(Duration::from_secs(20), async {
        let response = outbound.send().await.ok()?;
        if !response.status().is_success() {
            return None;
        }
        response.json::<Value>().await.ok()
    })
    .await
    .ok()
    .flatten();
    let outcome = if result.is_some() {
        "received"
    } else {
        "failed"
    };
    let _ = app.store.record_llm_call(
        "recheck",
        &result
            .as_ref()
            .map(UsageObservation::from_response)
            .unwrap_or_default(),
        outcome,
    );
    let rechecked = result?;
    let call = action_guard::response_call(protocol, &rechecked)?;
    let changed = call != proposal;
    let _ = app.store.event(
        &meta.agent,
        meta.project.as_deref(),
        meta.session.as_deref(),
        "recheck",
        &json!({"step_event":step}),
        &json!({"proposal":proposal,"changed":changed}),
    );
    Some((rechecked, changed))
}

fn outbound_request(
    app: &App,
    method: Method,
    headers: &HeaderMap,
    url: &str,
    body: reqwest::Body,
    upstream_key: Option<&str>,
    upstream_user_agent: Option<&str>,
) -> reqwest::RequestBuilder {
    let mut request = app.client.request(method, url).body(body);
    for (name, value) in headers {
        if [
            "host",
            "content-length",
            "connection",
            "transfer-encoding",
            "te",
            "trailer",
            "upgrade",
            "proxy-connection",
        ]
        .contains(&name.as_str())
            || name.as_str().starts_with("x-ctx-")
            || (upstream_key.is_some() && (name == "authorization" || name == "x-api-key"))
            || (upstream_user_agent.is_some() && name == "user-agent")
        {
            continue;
        }
        let token = app.store.token.as_str();
        if (name == "authorization"
            && value.to_str().ok().and_then(|v| v.strip_prefix("Bearer ")) == Some(token))
            || (name == "x-api-key" && value.to_str().ok() == Some(token))
        {
            continue;
        }
        request = request.header(name, value);
    }
    if let Some(key) = upstream_key {
        request = request.bearer_auth(key);
    }
    if let Some(value) = upstream_user_agent {
        request = request.header("user-agent", value);
    }
    request
}

pub fn upstream_url(base: &str, path: &str) -> Option<String> {
    let base = reqwest::Url::parse(base).ok()?;
    if !["http", "https"].contains(&base.scheme())
        || base.username() != ""
        || base.password().is_some()
        || base.query().is_some()
        || path.split('/').any(|p| p == ".." || p == ".")
    {
        return None;
    }
    Some(format!(
        "{}/{}",
        base.as_str().trim_end_matches('/'),
        path.trim_start_matches('/')
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_bad_upstream() {
        assert!(upstream_url("file:///etc/passwd", "messages").is_none());
        assert!(upstream_url("https://api.example.com/v1", "../x").is_none());
        assert!(
            upstream_url("https://api.example.com/v1", "messages")
                .unwrap()
                .ends_with("/v1/messages")
        );
    }
    #[test]
    fn extracts_response_text() {
        assert_eq!(
            response_text(br#"{"choices":[{"message":{"content":"hi"}}]}"#),
            "hi"
        );
        assert_eq!(response_text(b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\ndata: [DONE]\n"), "ab");
        assert_eq!(
            response_text(
                br#"{"content":[{"type":"text","text":"x"},{"type":"tool_use","name":"t"}]}"#
            ),
            "x"
        );
        assert_eq!(response_text(b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"y\"}}\n"), "y");
        assert_eq!(
            response_text(
                br#"{"output":[{"type":"message","content":[{"type":"output_text","text":"z"}]}]}"#
            ),
            "z"
        );
    }
}
