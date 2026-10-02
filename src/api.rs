//! Local REST API and dashboard. Every call needs the token (or a dashboard session).
use crate::{
    memory::NewMemory,
    recall::{self, Query},
    server::App,
    store::Store,
};
use axum::{
    extract::{Path, Query as UrlQuery, State},
    http::{HeaderMap, StatusCode},
    response::{Html, IntoResponse, Response},
};
use serde::Deserialize;
use serde_json::{Value, json};

pub type ApiResult = Result<axum::Json<Value>, (StatusCode, String)>;

fn api_error(err: impl std::fmt::Display) -> (StatusCode, String) {
    (StatusCode::BAD_REQUEST, err.to_string())
}
fn authorized(headers: &HeaderMap, store: &Store) -> bool {
    let explicit = headers.get("x-ctx-token").and_then(|h| h.to_str().ok());
    let bearer = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "));
    let api_key = headers.get("x-api-key").and_then(|h| h.to_str().ok());
    [explicit, bearer, api_key]
        .into_iter()
        .flatten()
        .any(|token| !token.is_empty() && (token == store.token || store.valid_ui_session(token)))
}
pub fn require(headers: &HeaderMap, store: &Store) -> Result<(), (StatusCode, String)> {
    if authorized(headers, store) {
        Ok(())
    } else {
        Err((
            StatusCode::UNAUTHORIZED,
            "missing or invalid ctx token".into(),
        ))
    }
}

pub async fn health(State(app): State<App>, headers: HeaderMap) -> ApiResult {
    require(&headers, &app.store)?;
    Ok(axum::Json(json!({"status":"ok","version":crate::VERSION})))
}

#[derive(Deserialize)]
pub struct EventInput {
    agent_id: String,
    project: Option<String>,
    session_id: Option<String>,
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    data: Value,
}
/// Hooks from agents or scripts. `task_end`/`session_end` with a session id queue that
/// session for extraction now.
pub async fn event(
    State(app): State<App>,
    headers: HeaderMap,
    axum::Json(input): axum::Json<EventInput>,
) -> ApiResult {
    require(&headers, &app.store)?;
    let id = app
        .store
        .event(
            &input.agent_id,
            input.project.as_deref(),
            input.session_id.as_deref(),
            "hook",
            &input.data,
            &json!({"type":input.kind,"data":input.data}),
        )
        .map_err(api_error)?;
    let mut queued = false;
    if matches!(input.kind.as_str(), "task_end" | "session_end")
        && let Some(session) = &input.session_id
    {
        queued = app.store.request_extraction(session).map_err(api_error)?;
        if queued {
            app.extraction_wakeup.notify_one();
        }
    }
    Ok(axum::Json(json!({"id":id,"extraction_queued":queued})))
}
pub async fn expand_event(
    State(app): State<App>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> ApiResult {
    require(&headers, &app.store)?;
    match app.store.event_payload(&id).map_err(api_error)? {
        Some(payload) => Ok(axum::Json(payload)),
        None => Err((StatusCode::NOT_FOUND, "event not found".into())),
    }
}
fn memory_json(memory: &crate::memory::Memory) -> Value {
    let mut value = json!(memory);
    value["body"] = memory.body.clone().into();
    value
}
pub async fn memories(State(app): State<App>, headers: HeaderMap) -> ApiResult {
    require(&headers, &app.store)?;
    Ok(axum::Json(json!(
        app.store
            .memories()
            .iter()
            .map(memory_json)
            .collect::<Vec<_>>()
    )))
}
pub async fn memory(
    State(app): State<App>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> ApiResult {
    require(&headers, &app.store)?;
    match app.store.memory(&id) {
        Some(memory) => Ok(axum::Json(memory_json(&memory))),
        None => Err((StatusCode::NOT_FOUND, "memory not found".into())),
    }
}
/// Memories written through the API count as agent-sourced unless the dashboard (a
/// user session) wrote them.
pub async fn remember(
    State(app): State<App>,
    headers: HeaderMap,
    axum::Json(input): axum::Json<NewMemory>,
) -> ApiResult {
    require(&headers, &app.store)?;
    let from_dashboard = headers
        .get("x-ctx-token")
        .and_then(|h| h.to_str().ok())
        .is_some_and(|t| app.store.valid_ui_session(t));
    let source = if from_dashboard { "user" } else { "agent" };
    let memory = app.store.remember(input, source).map_err(api_error)?;
    Ok(axum::Json(memory_json(&memory)))
}
#[derive(Deserialize)]
pub struct MemoryEdit {
    content: String,
}
pub async fn edit_memory(
    State(app): State<App>,
    Path(id): Path<String>,
    headers: HeaderMap,
    axum::Json(input): axum::Json<MemoryEdit>,
) -> ApiResult {
    require(&headers, &app.store)?;
    match app
        .store
        .update_memory(&id, &input.content)
        .map_err(api_error)?
    {
        Some(memory) => Ok(axum::Json(memory_json(&memory))),
        None => Err((StatusCode::NOT_FOUND, "memory not found".into())),
    }
}
pub async fn forget(
    State(app): State<App>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> ApiResult {
    require(&headers, &app.store)?;
    Ok(axum::Json(
        json!({"archived":app.store.archive(&id).map_err(api_error)?}),
    ))
}
#[derive(Deserialize)]
pub struct ReviewInput {
    decision: String,
}
pub async fn review_memory(
    State(app): State<App>,
    Path(id): Path<String>,
    headers: HeaderMap,
    axum::Json(input): axum::Json<ReviewInput>,
) -> ApiResult {
    require(&headers, &app.store)?;
    let approve = match input.decision.as_str() {
        "approve" => true,
        "reject" => false,
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                "decision must be approve or reject".into(),
            ));
        }
    };
    Ok(axum::Json(
        json!({"reviewed":app.store.review_memory(&id,approve).map_err(api_error)?}),
    ))
}
#[derive(Deserialize)]
pub struct RecallParams {
    q: String,
    project: Option<String>,
    limit: Option<usize>,
    /// "search" (default): best `limit` by relevance; "inject": the proxy's confident few.
    mode: Option<String>,
    /// Conversation excerpts to add in search mode (default: config `search_episodes`).
    episodes: Option<usize>,
    /// Search mode: plan sub-queries and a date window with one model call first.
    deep: Option<bool>,
    /// "Today" for relative dates in a deep search (default: the current date).
    now: Option<String>,
    /// Search mode: pack the result for a reader into this many tokens (memories first,
    /// excerpts cut to their relevant part).
    budget: Option<usize>,
    /// Search mode: put a brief written for the question from a wide retrieval (one model
    /// call) before the hits; `budget` covers the brief (at most 40% of it) and the hits.
    brief: Option<bool>,
    /// Search mode: add the user's own messages in conversation order (all of them for
    /// questions about how things went, in what order or how often; else the most
    /// relevant), within the budget. A brief always reads them.
    turns: Option<bool>,
    /// With `brief`: return the material the brief would be written from instead of
    /// writing it (no model call; for inspecting retrieval).
    dry_run: Option<bool>,
    /// With `brief`: the writer may call tools over the memory (search, the user's
    /// messages, whole turns, the timeline, a day calculator) before writing; more model
    /// calls on questions that need checking (see `agent.rs`).
    agent: Option<bool>,
}
pub async fn recall(
    State(app): State<App>,
    headers: HeaderMap,
    UrlQuery(params): UrlQuery<RecallParams>,
) -> ApiResult {
    require(&headers, &app.store)?;
    let search = params.mode.as_deref() != Some("inject");
    let plan = if search && params.deep == Some(true) {
        let today = params
            .now
            .clone()
            .unwrap_or_else(|| chrono::Utc::now().format("%Y-%m-%d").to_string());
        // Without a plan the search still runs, just as an ordinary search.
        crate::planner::plan(&app.store, &params.q, &today).await.ok()
    } else {
        None
    };
    let store = app.store.clone();
    let brief = search && params.brief == Some(true);
    let (question, today, brief_tokens) = (params.q.clone(), params.now.clone(), params.budget);
    let mut params = params;
    if brief {
        params.limit = Some(crate::brief::BRIEF_LIMIT);
        params.episodes = Some(crate::brief::BRIEF_EPISODES);
        params.budget = Some(crate::brief::BRIEF_GATHER_TOKENS);
    }
    let turns = params.turns == Some(true) && !brief;
    let params_dry_run = params.dry_run == Some(true);
    let use_agent = brief && params.agent == Some(true);
    let project = params.project.clone();
    let (hits, log) = tokio::task::spawn_blocking(move || -> anyhow::Result<(Vec<recall::Hit>, Vec<recall::Hit>)> {
        let query = Query {
                text: Some(&params.q),
                project: params.project.as_deref(),
                limit: params.limit.unwrap_or(5).min(50),
                mode: if params.mode.as_deref() == Some("inject") {
                    recall::Mode::Inject
                } else {
                    recall::Mode::Search
                },
                episodes: params
                    .episodes
                    .unwrap_or(store.config.recall.search_episodes)
                    .min(20),
                budget: params.budget,
                turns,
                ..Default::default()
            };
        let mut hits = match &plan {
            Some(plan) => recall::recall_planned(&store, &query, plan)?,
            None => recall::recall(&store, &query)?,
        };
        // A brief reads the user's own messages and the most relevant topic dossiers
        // besides the gathered hits.
        let mut log: Vec<recall::Hit> = vec![];
        if brief {
            let texts: Vec<&str> =
                std::iter::once(params.q.as_str()).chain(plan.iter().flat_map(|p| p.queries.iter().map(String::as_str))).collect();
            log.extend(recall::turn_log_for(&store, &texts, params.project.as_deref(), recall::TURN_LOG_CHARS)?);
            let wanted = recall::intent(&params.q);
            let history = recall::summary_question(&wanted) || wanted.mention_order;
            for dossier in crate::dossier::for_question(&store, &params.q, params.project.as_deref(), crate::dossier::BRIEF_DOSSIERS, history)? {
                match hits.iter_mut().find(|h| h.memory.id == dossier.memory.id) {
                    Some(hit) => hit.memory.body = dossier.memory.body,
                    None => log.push(dossier),
                }
            }
        }
        Ok((hits, log))
    })
    .await
    .map_err(api_error)?
    .map_err(api_error)?;
    let yes_no = recall::intent(&question).yes_no;
    let dry_run = brief && params_dry_run;
    // The brief comes first, then the evidence it was written from, packed into the rest
    // of the budget (a brief can miss a detail). Without a brief (the model call failed)
    // the hits are returned packed as usual.
    let mut lead = vec![];
    let hits = if dry_run {
        log.into_iter().chain(hits).collect()
    } else if brief {
        let total = brief_tokens.unwrap_or(crate::brief::BRIEF_TOKENS * 5 / 2);
        let cap = if recall::summary_question(&recall::intent(&question)) {
            crate::brief::SUMMARY_BRIEF_TOKENS
        } else {
            crate::brief::BRIEF_TOKENS
        };
        let tokens = (total * 2 / 5).min(cap);
        let material: Vec<recall::Hit> = log.into_iter().chain(hits.iter().cloned()).collect();
        let written = if use_agent {
            crate::agent::brief(app.store.clone(), &question, today.as_deref(), project, &material, tokens).await
        } else {
            crate::brief::brief(&app.store, &question, today.as_deref(), &material, tokens).await.map(|text| (text, vec![]))
        };
        match written {
            Ok((text, steps)) => {
                let used = text.chars().count() / 4 + 12;
                lead.push(json!({"id":"brief","title":"Memory brief","content":text,"type":"brief","channel":"brief","score":1.0,"sources":hits.len(),"steps":steps}));
                recall::pack(hits, &question, total.saturating_sub(used))
            }
            Err(_) => recall::pack(hits, &question, total),
        }
    } else {
        hits
    };
    Ok(axum::Json(json!(lead.into_iter().chain(hits.iter().map(|h| json!({"id":h.memory.id,"title":h.memory.title,"content":h.memory.body,"type":h.memory.kind,"scope":h.memory.scope,"score":h.score,"channel":h.channel,"group":h.group,"observed_at":h.memory.observed_at,"event_at":h.memory.event_at,"topics":h.memory.topics,"history":recall::history(&app.store, &h.memory, 3).into_iter().map(|(date, content)| json!({"date":date,"content":content})).collect::<Vec<_>>(),"conflicts":if yes_no { recall::conflicts(&app.store, &h.memory).into_iter().map(|(date, content)| json!({"date":date,"content":content})).collect::<Vec<_>>() } else { vec![] },"turn":h.turn,"mentioned":if matches!(h.channel, "episode" | "timeline" | "conflict" | "turnlog" | "turn" | "timechain") || h.memory.derived() { serde_json::Value::Null } else { json!(recall::mention_rank(&app.store, &h.memory)) },"status":h.memory.status,"mentioned_at":h.memory.mentioned_at,"created_at":h.memory.created_at}))).collect::<Vec<_>>())))
}
pub async fn step(State(app): State<App>, Path(id): Path<String>, headers: HeaderMap) -> ApiResult {
    require(&headers, &app.store)?;
    match app.store.step(&id).map_err(api_error)? {
        Some(step) => Ok(axum::Json(step)),
        None => Err((StatusCode::NOT_FOUND, "step not found".into())),
    }
}
pub async fn stats(State(app): State<App>, headers: HeaderMap) -> ApiResult {
    require(&headers, &app.store)?;
    Ok(axum::Json(app.store.stats().map_err(api_error)?))
}
pub async fn status(State(app): State<App>, headers: HeaderMap) -> ApiResult {
    require(&headers, &app.store)?;
    Ok(axum::Json(
        crate::status_json(&app.store).map_err(api_error)?,
    ))
}
pub async fn sessions(State(app): State<App>, headers: HeaderMap) -> ApiResult {
    require(&headers, &app.store)?;
    Ok(axum::Json(json!(
        app.store.sessions(100).map_err(api_error)?
    )))
}
#[derive(Deserialize)]
pub struct ExtractParams {
    #[serde(default)]
    wait: bool,
}
/// Queue a session for extraction, or with `?wait=true` extract it now and return the result.
pub async fn extract_session(
    State(app): State<App>,
    Path(key): Path<String>,
    UrlQuery(params): UrlQuery<ExtractParams>,
    headers: HeaderMap,
) -> ApiResult {
    require(&headers, &app.store)?;
    if params.wait {
        let Some(session) = app.store.session(&key).map_err(api_error)? else {
            return Err((StatusCode::NOT_FOUND, "session not found".into()));
        };
        return match crate::extract::extract_session(&app.store, &session).await {
            Ok(result) => Ok(axum::Json(result)),
            Err(error) => {
                let message = format!("{error:#}");
                let _ = app.store.finish_session(&key, "failed", Some(&message));
                Err((StatusCode::BAD_GATEWAY, crate::store::redact(&message)))
            }
        };
    }
    let queued = app.store.request_extraction(&key).map_err(api_error)?;
    if queued {
        app.extraction_wakeup.notify_one();
    }
    Ok(axum::Json(json!({"queued":queued})))
}
#[derive(Deserialize)]
pub struct IngestInput {
    session: String,
    #[serde(default = "ingest_agent")]
    agent: String,
    project: Option<String>,
    /// When the conversation happened (RFC 3339 or any date text).
    observed_at: Option<String>,
    messages: Vec<Value>,
}
fn ingest_agent() -> String {
    "ingest".into()
}
/// Record a conversation directly (imports, evaluations) instead of through the proxy.
pub async fn ingest_session(
    State(app): State<App>,
    headers: HeaderMap,
    axum::Json(input): axum::Json<IngestInput>,
) -> ApiResult {
    require(&headers, &app.store)?;
    if input.session.is_empty() || input.session.len() > 200 || input.messages.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "session and messages are required".into(),
        ));
    }
    let project = input.project.as_deref().map(crate::memory::normalize_scope);
    let event = app
        .store
        .ingest_session(
            &input.session,
            &input.agent,
            project.as_deref(),
            &input.messages,
            input.observed_at.as_deref(),
        )
        .map_err(api_error)?;
    Ok(axum::Json(json!({"event":event,"session":input.session})))
}
#[derive(Deserialize)]
pub struct FeedbackInput {
    step_event: String,
    memory_id: String,
    cited: Option<bool>,
    action_consistent: Option<bool>,
    task_result: Option<String>,
}
pub async fn feedback(
    State(app): State<App>,
    headers: HeaderMap,
    axum::Json(input): axum::Json<FeedbackInput>,
) -> ApiResult {
    require(&headers, &app.store)?;
    app.store
        .record_feedback(
            &input.step_event,
            &input.memory_id,
            input.cited,
            input.action_consistent,
            input.task_result.as_deref(),
        )
        .map_err(api_error)?;
    Ok(axum::Json(json!({"recorded":true})))
}
#[derive(Deserialize)]
pub struct MissInput {
    event_id: String,
    memory_id: String,
    reason: String,
}
pub async fn missed_recall(
    State(app): State<App>,
    headers: HeaderMap,
    axum::Json(input): axum::Json<MissInput>,
) -> ApiResult {
    require(&headers, &app.store)?;
    let id = crate::feedback::record_and_repair(
        &app.store,
        &input.event_id,
        &input.memory_id,
        &input.reason,
    )
    .map_err(api_error)?;
    Ok(axum::Json(json!({"id":id})))
}
pub async fn debug_step(
    State(app): State<App>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> ApiResult {
    require(&headers, &app.store)?;
    Ok(axum::Json(app.store.debug_step(&id).map_err(api_error)?))
}
pub async fn maintenance(State(app): State<App>, headers: HeaderMap) -> ApiResult {
    require(&headers, &app.store)?;
    Ok(axum::Json(
        app.store.maintenance_batches().map_err(api_error)?,
    ))
}
pub async fn ui() -> Html<&'static str> {
    Html(include_str!("../web/index.html"))
}
#[derive(Deserialize)]
pub struct TicketQuery {
    ticket: String,
}
pub async fn ui_bootstrap(
    State(app): State<App>,
    UrlQuery(query): UrlQuery<TicketQuery>,
) -> Response {
    let Some(session) = app.store.redeem_ui_ticket(&query.ticket) else {
        return (StatusCode::UNAUTHORIZED, "ticket expired").into_response();
    };
    Html(format!(
        "<script>sessionStorage.setItem('ctxSession',{});location.replace('/ui')</script>",
        serde_json::to_string(&session).unwrap()
    ))
    .into_response()
}
pub async fn ui_ticket(State(app): State<App>, headers: HeaderMap) -> ApiResult {
    require(&headers, &app.store)?;
    let ticket = app.store.create_ui_ticket();
    Ok(axum::Json(
        json!({"url":format!("http://127.0.0.1:{}/ui/bootstrap?ticket={ticket}",app.store.config.port)}),
    ))
}

#[derive(Deserialize)]
pub struct EmbeddingInput {
    input: Value,
    #[serde(default)]
    model: Option<String>,
}
/// OpenAI-compatible `POST /api/v1/embeddings` backed by the local embedding model, so
/// other tools can share it. Texts are embedded as documents.
pub async fn embeddings(
    State(app): State<App>,
    headers: HeaderMap,
    axum::Json(input): axum::Json<EmbeddingInput>,
) -> ApiResult {
    require(&headers, &app.store)?;
    let texts: Vec<String> = match input.input {
        Value::String(text) => vec![text],
        Value::Array(items) => items
            .into_iter()
            .map(|v| v.as_str().map(str::to_owned).ok_or("input must be strings"))
            .collect::<Result<_, _>>()
            .map_err(api_error)?,
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                "input must be a string or list".into(),
            ));
        }
    };
    if texts.is_empty() || texts.len() > 256 {
        return Err((StatusCode::BAD_REQUEST, "1..256 inputs".into()));
    }
    let store = app.store.clone();
    let (name, vectors) = tokio::task::spawn_blocking(move || {
        let model = store.config.embedding.model.clone();
        let embedder = store
            .embedder
            .load(&model, store.config.embedding.workers)
            .ok_or_else(|| anyhow::anyhow!("embedding model unavailable"))?;
        let docs: Vec<(String, String)> = texts.into_iter().map(|t| ("none".into(), t)).collect();
        anyhow::Ok((embedder.name.clone(), embedder.embed_documents(&docs)?))
    })
    .await
    .map_err(api_error)?
    .map_err(api_error)?;
    let data: Vec<Value> = vectors
        .into_iter()
        .enumerate()
        .map(|(i, v)| json!({"object":"embedding","index":i,"embedding":v}))
        .collect();
    Ok(axum::Json(
        json!({"object":"list","data":data,"model":input.model.unwrap_or(name),"usage":{"prompt_tokens":0,"total_tokens":0}}),
    ))
}
