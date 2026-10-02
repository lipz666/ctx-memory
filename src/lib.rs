//! ctx library: the memory engine, shared by the `ctx` binary and `ccx`.
pub mod adapters;
pub mod agent;
pub mod api;
pub mod brief;
pub mod config;
pub mod consolidation;
pub mod contradict;
pub mod dossier;
pub mod embed;
pub mod episode;
pub mod experimental;
pub mod extract;
pub mod feedback;
pub mod index;
pub mod inject;
pub mod llm;
pub mod mcp;
pub mod memory;
pub mod observer;
pub mod planner;
pub mod proxy;
pub mod recall;
pub mod reflect;
pub mod server;
pub mod session;
pub mod store;

use anyhow::Result;
use serde_json::{Value, json};
use store::Store;

/// Package version and the git commit the binary was built from.
pub const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (", env!("CTX_GIT_HASH"), ")");

pub fn status_json(store: &Store) -> Result<Value> {
    let config = &store.config;
    Ok(json!({
        "version": VERSION,
        "root": store.root.display().to_string(),
        "port": config.port,
        "agents": config.agents.keys().collect::<Vec<_>>(),
        "model": config.model.as_ref().map(|m| &m.model),
        "embedding": {"enabled": config.embedding.enabled, "model": config.embedding.model, "loaded": store.embedder.get().is_some()},
        "extraction": {"enabled": config.extraction.enabled && config.model.is_some(), "idle_minutes": config.extraction.idle_minutes, "daily_llm_calls": config.extraction.daily_llm_calls, "llm_calls_today": store.llm_calls_today()?},
        "experimental": {"gate": config.experimental.gate_enabled, "action_guard": config.experimental.action_guard, "eviction": config.experimental.eviction_enabled, "maintenance": config.experimental.maintenance_enabled},
        "stats": store.stats()?,
    }))
}
