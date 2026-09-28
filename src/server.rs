//! The daemon: router, shared state and background work (embedding warm-up, session
//! extraction, experimental maintenance, memory file watching).
use crate::{
    api,
    experimental::{classifier::Classifier, gate},
    proxy,
    session::Tracker,
    store::Store,
};
use axum::{
    Router,
    routing::{get, post},
};
use notify::Watcher;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::Notify;

#[derive(Clone)]
pub struct App {
    pub store: Arc<Store>,
    pub client: reqwest::Client,
    pub upstream_keys: Arc<HashMap<String, String>>,
    pub tracker: Arc<Tracker>,
    pub classifier: Option<Classifier>,
    pub pending_gate: Arc<Mutex<HashMap<String, PendingGate>>>,
    pub extraction_wakeup: Arc<Notify>,
}
/// A deferred Gate decision waiting for the next step of the same session.
pub struct PendingGate {
    pub at: Instant,
    pub decision: gate::Decision,
}

pub async fn serve(store: Store) -> anyhow::Result<()> {
    let port = store.config.port;
    let mut upstream_keys = HashMap::new();
    for (agent, config) in &store.config.agents {
        if let Some(reference) = &config.credential_ref {
            upstream_keys.insert(agent.clone(), crate::store::credential(reference)?);
        }
    }
    let app = App {
        classifier: Classifier::load(&store.root)
            .filter(|_| store.config.experimental.gate_enabled),
        store: Arc::new(store),
        client: reqwest::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()?,
        upstream_keys: Arc::new(upstream_keys),
        tracker: Arc::default(),
        pending_gate: Arc::default(),
        extraction_wakeup: Arc::default(),
    };
    // Load the embedding model and embed memories off the request path; until then
    // recall uses keyword search.
    let warm = app.store.clone();
    tokio::task::spawn_blocking(move || {
        if let Err(error) = warm.ensure_vectors() {
            eprintln!("ctx: embedding memories failed: {error:#}");
        }
    });
    let background = app.store.clone();
    let wakeup = app.extraction_wakeup.clone();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_secs(60)) => {}
                _ = wakeup.notified() => {}
            }
            match crate::extract::run_due(&background).await {
                Ok(result) if result["sessions"].as_u64().unwrap_or(0) > 0 => {
                    eprintln!(
                        "ctx extraction: {}",
                        crate::store::redact(&result.to_string())
                    )
                }
                Err(error) => eprintln!(
                    "ctx extraction: {}",
                    crate::store::redact(&error.to_string())
                ),
                _ => {}
            }
        }
    });
    if app.store.config.experimental.maintenance_enabled {
        let background = app.store.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(3600));
            loop {
                tick.tick().await;
                if crate::consolidation::should_run(&background) {
                    let store = background.clone();
                    let _ = tokio::task::spawn_blocking(move || crate::consolidation::run(&store))
                        .await;
                }
            }
        });
    }
    let watcher_store = app.store.clone();
    let mut watcher = notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
        if let Ok(event) = event {
            for path in event
                .paths
                .iter()
                .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("md"))
            {
                if let Err(error) = watcher_store.reload_file(path) {
                    eprintln!("ctx: reload {}: {error:#}", path.display());
                }
            }
        }
    })?;
    watcher.watch(
        &app.store.root.join("memory"),
        notify::RecursiveMode::NonRecursive,
    )?;
    let router = Router::new()
        .route("/ui", get(api::ui))
        .route("/ui/bootstrap", get(api::ui_bootstrap))
        .route("/api/v1/ui-ticket", post(api::ui_ticket))
        .route("/api/v1/health", get(api::health))
        .route(
            "/a/{agent}/v1/{*path}",
            post(proxy::handle).get(proxy::handle),
        )
        .route(
            "/a/{agent}/p/{project}/v1/{*path}",
            post(proxy::handle).get(proxy::handle),
        )
        .route("/api/v1/events", post(api::event))
        .route("/api/v1/events/{id}", get(api::expand_event))
        .route("/api/v1/memories", get(api::memories).post(api::remember))
        .route(
            "/api/v1/memories/{id}",
            get(api::memory).delete(api::forget).patch(api::edit_memory),
        )
        .route("/api/v1/memories/{id}/review", post(api::review_memory))
        .route("/api/v1/recall", get(api::recall))
        .route("/api/v1/sessions", get(api::sessions))
        .route("/api/v1/sessions/{key}/extract", post(api::extract_session))
        .route("/api/v1/sessions/ingest", post(api::ingest_session))
        .route("/api/v1/embeddings", post(api::embeddings))
        .route("/api/v1/steps/{id}", get(api::step))
        .route("/api/v1/stats", get(api::stats))
        .route("/api/v1/status", get(api::status))
        .route("/api/v1/feedback", post(api::feedback))
        .route("/api/v1/feedback/miss", post(api::missed_recall))
        .route("/api/v1/debug/steps/{id}", get(api::debug_step))
        .route("/api/v1/maintenance", get(api::maintenance))
        .with_state(app);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    axum::serve(listener, router).await?;
    drop(watcher);
    Ok(())
}
