//! Configuration (`config.yaml`). Unknown or legacy keys are ignored so older files load.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub port: u16,
    pub agents: BTreeMap<String, Agent>,
    pub model: Option<ModelConfig>,
    pub recall: RecallConfig,
    pub embedding: EmbeddingConfig,
    pub extraction: ExtractionConfig,
    /// `default` lists recall, remember, forget and expand over MCP; `full` lists all tools.
    pub mcp_tools: String,
    /// Keep full original and compiled requests for the debug view (large; off by default).
    pub debug_capture: bool,
    /// Record every memory change in the memory directory's git repository.
    pub history: bool,
    pub experimental: ExperimentalConfig,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            port: 7788,
            agents: BTreeMap::new(),
            model: None,
            recall: RecallConfig::default(),
            embedding: EmbeddingConfig::default(),
            extraction: ExtractionConfig::default(),
            mcp_tools: "default".into(),
            debug_capture: false,
            history: true,
            experimental: ExperimentalConfig::default(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Agent {
    pub upstream: String,
    #[serde(default)]
    pub credential_ref: Option<String>,
    #[serde(default)]
    pub upstream_user_agent: Option<String>,
    /// Record usage but never modify requests (used as the direct arm of A/B runs).
    #[serde(default)]
    pub meter_only: bool,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub base_url: String,
    pub model: String,
    pub credential_ref: String,
    #[serde(default)]
    pub upstream_user_agent: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RecallConfig {
    /// Memories injected per new user turn or error, besides always-on rules.
    pub max_injected: usize,
    /// Token budget for one injection block.
    pub budget_tokens: usize,
    /// Minimum semantic similarity for a search hit (embedding model scale).
    pub min_similarity: f32,
    /// Minimum normalized keyword score for a search hit when there is no strong semantic match.
    pub min_keyword: f32,
    /// Keyword threshold when no embedding model is available (keyword-only recall).
    pub min_keyword_fallback: f32,
    /// Weight of semantic similarity in the combined score (rest is keyword coverage).
    pub semantic_weight: f32,
    /// Search hits scoring below this fraction of the best one are dropped.
    pub relative_cutoff: f32,
    /// Semantic floor for explicit searches (API, MCP, CLI), which return the best
    /// `limit` candidates ranked by relevance instead of only the confident few.
    pub search_min_similarity: f32,
    /// Raw conversation excerpts returned by explicit searches besides memories (0: none).
    pub search_episodes: usize,
}
impl Default for RecallConfig {
    fn default() -> Self {
        Self {
            max_injected: 3,
            budget_tokens: 1500,
            min_similarity: 0.34,
            min_keyword: 0.55,
            min_keyword_fallback: 0.30,
            semantic_weight: 0.75,
            relative_cutoff: 0.8,
            search_min_similarity: 0.25,
            search_episodes: 5,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    pub enabled: bool,
    /// fastembed model name; see `embed::model_for`.
    pub model: String,
    /// Model instances embedding in parallel (each uses about 300 MB).
    pub workers: usize,
}
impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model: "embeddinggemma-300m-q".into(),
            workers: 1,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ExtractionConfig {
    /// Extract memories from a session after it goes idle (needs `model`).
    pub enabled: bool,
    pub idle_minutes: u64,
    /// Engine LLM calls per day across extraction and experimental features.
    pub daily_llm_calls: u32,
    /// Replace the built-in extraction prompt with this file.
    pub prompt_file: Option<String>,
    /// Allow extraction to create global memories; when false everything is scoped to the
    /// session's project (isolated namespaces, e.g. benchmarks).
    pub global_scope: bool,
    /// Agents that are personal assistants rather than coding agents: their sessions
    /// use the general extraction prompt (facts and events about the user).
    pub general_agents: Vec<String>,
    /// Consolidate a topic's memories with a model call when it crosses a size threshold
    /// (preferences, current state, counts). Off: no measurable gain on the LongMemEval
    /// dev set for +32% extraction tokens, and a reflection goes stale between thresholds.
    pub reflection: bool,
    /// Show the extractor the most frequent known entities (resolving "my sister" to a
    /// name). Off: no measurable gain on the dev set for +20% extraction input.
    pub known_entities: bool,
    /// After extraction, have the model judge new negative claims ("never did X") against
    /// similar memories and link the pairs that cannot both be true (both kept, contested).
    /// Costs one small call per session that has such a candidate pair.
    pub contradictions: bool,
    /// Have the extractor also write the gist of each reply (stored per turn, shown with
    /// the user's messages for summaries and "what did you recommend"). Off: on BEAM 100K
    /// it cost about 5% of the extracted facts (contradictions and updates suffered more
    /// than summaries gained); outlines taken from the replies without a model are used.
    pub turn_notes: bool,
    /// After extraction, one more model call per session records values, items, dated
    /// events and stages per topic; each topic's records form a dossier (see `dossier.rs`).
    pub dossiers: bool,
}
impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            idle_minutes: 10,
            daily_llm_calls: 100,
            prompt_file: None,
            global_scope: true,
            general_agents: vec!["hermes".into()],
            reflection: false,
            known_entities: false,
            contradictions: true,
            turn_notes: false,
            dossiers: true,
        }
    }
}

/// Frozen features kept for experiments; all off by default.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ExperimentalConfig {
    pub gate_enabled: bool,
    /// `auto`: synchronous for a local model, deferred for a remote one.
    pub gate_mode: String,
    pub gate_timeout_ms: u64,
    pub gate_deferred_timeout_ms: u64,
    pub abstain_threshold: f64,
    /// `off`, `advisory` or `recheck`.
    pub action_guard: String,
    pub eviction_enabled: bool,
    pub evict_threshold_tokens: usize,
    pub tool_result_min_tokens: usize,
    pub maintenance_enabled: bool,
}
impl Default for ExperimentalConfig {
    fn default() -> Self {
        Self {
            gate_enabled: false,
            gate_mode: "auto".into(),
            gate_timeout_ms: 300,
            gate_deferred_timeout_ms: 20_000,
            abstain_threshold: 0.35,
            action_guard: "off".into(),
            eviction_enabled: false,
            evict_threshold_tokens: 20_000,
            tool_result_min_tokens: 500,
            maintenance_enabled: false,
        }
    }
}

/// Carry settings from a pre-v2 config forward. Frozen features (Gate, ActionGuard,
/// eviction, maintenance) are not carried over: they start off after the upgrade.
pub fn migrate_legacy(raw: &serde_yaml::Value, config: &mut Config) {
    if let Some(value) = raw
        .get("automation")
        .and_then(|a| a.get("daily_llm_calls"))
        .and_then(serde_yaml::Value::as_u64)
    {
        config.extraction.daily_llm_calls = value as u32;
    }
    if let Some(value) = raw.get("max_recalls").and_then(serde_yaml::Value::as_u64) {
        config.recall.max_injected = value as usize;
    }
    if let Some(value) = raw
        .get("tail_budget_tokens")
        .and_then(serde_yaml::Value::as_u64)
    {
        config.recall.budget_tokens = value as usize;
    }
}
