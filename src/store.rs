//! Local state: memory files (source of truth) with an in-memory search index, and one
//! SQLite database for events, sessions, recall logs, usage and cached vectors.
use crate::{
    config::{self, Config},
    embed,
    episode::{self, Episodes, Meta as EpisodeMeta},
    index::Index,
    memory::{self, Memory, NewMemory},
};
use aes_gcm::{
    Aes256Gcm, Nonce,
    aead::{Aead, KeyInit},
};
use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use rand::RngCore;
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::{Path, PathBuf},
    sync::{LazyLock, Mutex, RwLock},
    time::{Duration, Instant},
};

#[derive(Clone, Default, Debug)]
pub struct UsageObservation {
    pub requested_model: Option<String>,
    pub actual_model: Option<String>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cached_input_tokens: Option<u64>,
    pub cache_write_tokens: Option<u64>,
}
impl UsageObservation {
    pub fn from_response(value: &Value) -> Self {
        let usage = value
            .get("usage")
            .or_else(|| value.pointer("/message/usage"))
            .or_else(|| value.pointer("/response/usage"));
        let cached_input_tokens = usage
            .and_then(|u| {
                u.pointer("/prompt_tokens_details/cached_tokens")
                    .or_else(|| u.pointer("/input_tokens_details/cached_tokens"))
                    .or_else(|| u.get("cache_read_input_tokens"))
            })
            .and_then(Value::as_u64);
        let cache_write_tokens = usage
            .and_then(|u| u.get("cache_creation_input_tokens"))
            .and_then(Value::as_u64);
        let mut input_tokens = usage
            .and_then(|u| u.get("prompt_tokens").or_else(|| u.get("input_tokens")))
            .and_then(Value::as_u64);
        if usage.is_some_and(|u| {
            u.get("cache_read_input_tokens").is_some()
                || u.get("cache_creation_input_tokens").is_some()
        }) {
            input_tokens = input_tokens.map(|input| {
                input
                    .saturating_add(cached_input_tokens.unwrap_or(0))
                    .saturating_add(cache_write_tokens.unwrap_or(0))
            });
        }
        Self {
            requested_model: None,
            actual_model: value
                .get("model")
                .or_else(|| value.pointer("/message/model"))
                .or_else(|| value.pointer("/response/model"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            input_tokens,
            output_tokens: usage
                .and_then(|u| {
                    u.get("completion_tokens")
                        .or_else(|| u.get("output_tokens"))
                })
                .and_then(Value::as_u64),
            cached_input_tokens,
            cache_write_tokens,
        }
    }
    pub fn from_bytes(body: &[u8]) -> Self {
        if let Ok(value) = serde_json::from_slice::<Value>(body) {
            return Self::from_response(&value);
        }
        let mut merged = Self::default();
        for line in String::from_utf8_lossy(body).lines() {
            if let Some(data) = line.strip_prefix("data:")
                && let Ok(value) = serde_json::from_str::<Value>(data.trim())
            {
                let item = Self::from_response(&value);
                merged.actual_model = item.actual_model.or(merged.actual_model);
                merged.input_tokens = item.input_tokens.or(merged.input_tokens);
                merged.output_tokens = item.output_tokens.or(merged.output_tokens);
                merged.cached_input_tokens =
                    item.cached_input_tokens.or(merged.cached_input_tokens);
                merged.cache_write_tokens = item.cache_write_tokens.or(merged.cache_write_tokens);
            }
        }
        merged
    }
}

#[derive(Clone, Serialize)]
pub struct SessionRow {
    pub key: String,
    pub agent: String,
    pub project: Option<String>,
    pub started_at: String,
    pub last_seen: String,
    pub steps: i64,
    pub user_turns: i64,
    pub status: String,
    pub extracted_at: Option<String>,
    pub error: Option<String>,
    /// When the conversation happened, if supplied by the client (ingested sessions).
    pub observed_at: Option<String>,
}
/// (messages recorded, hash of the last one, project) of a session.
pub type SessionCursor = (usize, Option<String>, Option<String>);
pub struct MemoryUsage {
    pub recalled: u64,
    pub used: u64,
}
pub struct GateSample {
    pub features: Value,
    pub positive: bool,
    pub gate_recalled: bool,
}

#[derive(Default)]
struct Cache {
    all: HashMap<String, Memory>,
    index: Index,
    /// Hash of each memory's search text, to know when a vector is stale.
    text_hash: HashMap<String, String>,
}

pub struct Store {
    pub root: PathBuf,
    pub config: Config,
    pub token: String,
    db: Mutex<Connection>,
    cache: RwLock<Cache>,
    episodes: RwLock<Episodes>,
    pub embedder: embed::Slot,
    ui_tickets: Mutex<HashMap<String, Instant>>,
    ui_sessions: Mutex<HashMap<String, Instant>>,
    cipher: Aes256Gcm,
    /// Serializes git operations on the memory repository.
    git_lock: Mutex<()>,
}

pub fn root_path() -> PathBuf {
    std::env::var_os("CTX_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".ctx"))
}
fn secret_file(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(bytes)?;
    Ok(())
}
fn use_keychain_event_key(root: &Path) -> bool {
    cfg!(target_os = "macos") && std::env::var_os("CTX_HOME").is_none() && root == root_path()
}
fn event_key_reference(root: &Path) -> String {
    let digest = Sha256::digest(root.as_os_str().as_encoded_bytes());
    format!("keychain:ctx-event-key-{}", hex::encode(&digest[..8]))
}
fn init_event_key(root: &Path) -> Result<()> {
    let path = root.join("state/key");
    if use_keychain_event_key(root) {
        #[cfg(target_os = "macos")]
        {
            let reference = event_key_reference(root);
            let service = reference.strip_prefix("keychain:").unwrap();
            let existing = credential(&reference)
                .ok()
                .and_then(|value| hex::decode(value).ok());
            if existing.as_ref().is_none_or(|bytes| bytes.len() != 32) {
                if !path.exists() && root.join("state/events.db").exists() {
                    bail!("existing event database has no accessible encryption key");
                }
                let bytes = if path.exists() {
                    fs::read(&path)?
                } else {
                    let mut bytes = vec![0u8; 32];
                    rand::thread_rng().fill_bytes(&mut bytes);
                    bytes
                };
                if bytes.len() != 32 {
                    bail!("invalid event encryption key");
                }
                let status = std::process::Command::new("security")
                    .args([
                        "add-generic-password",
                        "-U",
                        "-a",
                        "default",
                        "-s",
                        service,
                        "-w",
                    ])
                    .arg(hex::encode(&bytes))
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status()?;
                if !status.success() {
                    bail!("could not store event key in Keychain");
                }
            }
            let stored = hex::decode(credential(&reference)?)?;
            if path.exists() {
                if fs::read(&path)? != stored {
                    bail!("Keychain event key differs from local key; refusing migration");
                }
                fs::remove_file(path)?;
            }
            return Ok(());
        }
    }
    if !path.exists() {
        let mut bytes = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut bytes);
        secret_file(&path, &bytes)?;
    }
    Ok(())
}
fn event_key(root: &Path) -> Result<Vec<u8>> {
    if use_keychain_event_key(root) {
        return Ok(hex::decode(credential(&event_key_reference(root))?)?);
    }
    Ok(fs::read(root.join("state/key"))?)
}
pub fn init(root: &Path) -> Result<()> {
    fs::create_dir_all(root.join("memory"))?;
    fs::create_dir_all(root.join("state"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for dir in [root, &root.join("memory"), &root.join("state")] {
            fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
        }
    }
    if !root.join("config.yaml").exists() {
        fs::write(
            root.join("config.yaml"),
            serde_yaml::to_string(&Config::default())?,
        )?;
    }
    if !root.join("token").exists() {
        let mut bytes = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut bytes);
        secret_file(&root.join("token"), hex::encode(bytes).as_bytes())?;
    }
    init_event_key(root)?;
    // The memory directory is its own git repository, independent of project repositories.
    if !root.join("memory/.git").exists() {
        let _ = std::process::Command::new("git")
            .arg("init")
            .arg("-q")
            .arg(root.join("memory"))
            .status();
    }
    let _ = Store::open(root)?;
    Ok(())
}

pub fn load_config(root: &Path) -> Result<Config> {
    let raw: serde_yaml::Value = serde_yaml::from_slice(&fs::read(root.join("config.yaml"))?)?;
    let mut config: Config = serde_yaml::from_value(raw.clone())?;
    config::migrate_legacy(&raw, &mut config);
    Ok(config)
}

const SCHEMA: &str = "PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, ts TEXT NOT NULL, agent_id TEXT NOT NULL, project_id TEXT, session_id TEXT, kind TEXT NOT NULL, features TEXT NOT NULL, nonce BLOB NOT NULL, payload BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS events_session ON events(session_id, ts);
CREATE TABLE IF NOT EXISTS sessions (key TEXT PRIMARY KEY, agent TEXT NOT NULL, project TEXT, started_at TEXT NOT NULL, last_seen TEXT NOT NULL, steps INTEGER NOT NULL DEFAULT 0, user_turns INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'open', extracted_at TEXT, error TEXT, attempts INTEGER NOT NULL DEFAULT 0, seen INTEGER NOT NULL DEFAULT 0, last_hash TEXT, observed_at TEXT);
CREATE TABLE IF NOT EXISTS recalls (id TEXT PRIMARY KEY, step_event TEXT, memory_id TEXT NOT NULL, channel TEXT NOT NULL, score REAL NOT NULL, reason TEXT NOT NULL, ts TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS recalls_step ON recalls(step_event);
CREATE TABLE IF NOT EXISTS steps (event_id TEXT PRIMARY KEY, decision TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS usage (step_event TEXT PRIMARY KEY, agent_id TEXT NOT NULL, ts TEXT NOT NULL, actual_input_tokens INTEGER, output_tokens INTEGER, requested_model TEXT, actual_model TEXT, cached_input_tokens INTEGER, cache_write_tokens INTEGER);
CREATE TABLE IF NOT EXISTS llm_calls (id TEXT PRIMARY KEY, role TEXT NOT NULL, ts TEXT NOT NULL, input_tokens INTEGER, output_tokens INTEGER, outcome TEXT NOT NULL, requested_model TEXT, actual_model TEXT);
CREATE TABLE IF NOT EXISTS recall_feedback (step_event TEXT NOT NULL, memory_id TEXT NOT NULL, cited INTEGER, action_consistent INTEGER, task_result TEXT, utility REAL, updated_at TEXT NOT NULL, PRIMARY KEY(step_event,memory_id));
CREATE TABLE IF NOT EXISTS missed_recalls (id TEXT PRIMARY KEY, event_id TEXT NOT NULL, memory_id TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS memory_flags (id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS memory_reviews (id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, decision TEXT NOT NULL, wait_seconds INTEGER NOT NULL, reviewed_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS maintenance_batches (id TEXT PRIMARY KEY, task TEXT NOT NULL, status TEXT NOT NULL, before_json TEXT NOT NULL, after_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS vectors (id TEXT PRIMARY KEY, hash TEXT NOT NULL, model TEXT NOT NULL, data BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS evictions (hash TEXT PRIMARY KEY, event_id TEXT NOT NULL, placeholder TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rechecks (step_event TEXT PRIMARY KEY, task_key TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS episodes (id TEXT PRIMARY KEY, session TEXT NOT NULL, project TEXT, observed_at TEXT, seq INTEGER NOT NULL, hash TEXT NOT NULL, nonce BLOB NOT NULL, text BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS turn_notes (id TEXT PRIMARY KEY, session TEXT NOT NULL, nonce BLOB NOT NULL, text BLOB NOT NULL);";

impl Store {
    pub fn open(root: &Path) -> Result<Self> {
        let config = load_config(root)?;
        let key = event_key(root)?;
        if key.len() != 32 {
            bail!("invalid event encryption key");
        }
        let cipher = Aes256Gcm::new_from_slice(&key).map_err(|_| anyhow::anyhow!("invalid key"))?;
        let db = Connection::open(root.join("state/events.db"))?;
        db.execute_batch(SCHEMA)?;
        for (table, column, definition) in [
            ("sessions", "seen", "seen INTEGER NOT NULL DEFAULT 0"),
            ("sessions", "last_hash", "last_hash TEXT"),
            ("sessions", "observed_at", "observed_at TEXT"),
            ("usage", "requested_model", "requested_model TEXT"),
            ("usage", "actual_model", "actual_model TEXT"),
            (
                "usage",
                "cached_input_tokens",
                "cached_input_tokens INTEGER",
            ),
            ("usage", "cache_write_tokens", "cache_write_tokens INTEGER"),
            ("llm_calls", "requested_model", "requested_model TEXT"),
            ("llm_calls", "actual_model", "actual_model TEXT"),
        ] {
            ensure_column(&db, table, column, definition)?;
        }
        // The v1 full-text index lived in a separate database; it is derived and unused now.
        for suffix in ["", "-wal", "-shm"] {
            let _ = fs::remove_file(root.join(format!("state/index.db{suffix}")));
        }
        let store = Self {
            root: root.to_path_buf(),
            token: fs::read_to_string(root.join("token"))?.trim().into(),
            config,
            db: Mutex::new(db),
            cache: RwLock::new(Cache::default()),
            episodes: RwLock::new(Episodes::default()),
            embedder: embed::Slot::default(),
            ui_tickets: Mutex::new(HashMap::new()),
            ui_sessions: Mutex::new(HashMap::new()),
            cipher,
            git_lock: Mutex::new(()),
        };
        if !store.config.embedding.enabled {
            store.embedder.disable();
        }
        store.recover_prepared_batches()?;
        store.reload_memories()?;
        store.load_episodes()?;
        Ok(store)
    }
    pub fn save_config(root: &Path, config: &Config) -> Result<()> {
        fs::write(root.join("config.yaml"), serde_yaml::to_string(config)?)?;
        Ok(())
    }

    // ---- UI sessions ----
    pub fn create_ui_ticket(&self) -> String {
        let token = random_token();
        self.ui_tickets
            .lock()
            .unwrap()
            .insert(token.clone(), Instant::now());
        token
    }
    pub fn redeem_ui_ticket(&self, ticket: &str) -> Option<String> {
        let issued = self.ui_tickets.lock().unwrap().remove(ticket)?;
        if issued.elapsed() > Duration::from_secs(60) {
            return None;
        }
        let session = random_token();
        self.ui_sessions
            .lock()
            .unwrap()
            .insert(session.clone(), Instant::now());
        Some(session)
    }
    pub fn valid_ui_session(&self, token: &str) -> bool {
        self.ui_sessions
            .lock()
            .unwrap()
            .get(token)
            .is_some_and(|at| at.elapsed() < Duration::from_secs(3600))
    }

    // ---- events ----
    pub fn event(
        &self,
        agent: &str,
        project: Option<&str>,
        session: Option<&str>,
        kind: &str,
        features: &Value,
        payload: &Value,
    ) -> Result<String> {
        let id = format!("evt_{}", ulid::Ulid::new());
        let (nonce, encrypted) = self.seal(payload)?;
        self.db.lock().unwrap().execute(
            "INSERT INTO events VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![
                id,
                Utc::now().to_rfc3339(),
                agent,
                project,
                session,
                kind,
                redact_value(features).to_string(),
                nonce.as_slice(),
                encrypted
            ],
        )?;
        Ok(id)
    }
    fn seal(&self, value: &Value) -> Result<([u8; 12], Vec<u8>)> {
        let mut nonce = [0u8; 12];
        rand::thread_rng().fill_bytes(&mut nonce);
        let raw = serde_json::to_vec(value)?;
        let encrypted = self
            .cipher
            .encrypt(Nonce::from_slice(&nonce), raw.as_slice())
            .map_err(|_| anyhow::anyhow!("encryption failed"))?;
        Ok((nonce, encrypted))
    }
    fn decrypt(&self, nonce: &[u8], blob: &[u8]) -> Result<Value> {
        let raw = self
            .cipher
            .decrypt(Nonce::from_slice(nonce), blob)
            .map_err(|_| anyhow::anyhow!("decryption failed"))?;
        Ok(serde_json::from_slice(&raw)?)
    }
    pub fn event_payload(&self, id: &str) -> Result<Option<Value>> {
        let row: Option<(Vec<u8>, Vec<u8>)> = self
            .db
            .lock()
            .unwrap()
            .query_row("SELECT nonce,payload FROM events WHERE id=?1", [id], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .optional()?;
        row.map(|(nonce, blob)| self.decrypt(&nonce, &blob))
            .transpose()
    }
    pub fn event_features(&self, id: &str) -> Result<Option<Value>> {
        let raw: Option<String> = self
            .db
            .lock()
            .unwrap()
            .query_row("SELECT features FROM events WHERE id=?1", [id], |r| {
                r.get(0)
            })
            .optional()?;
        Ok(raw.map(|s| serde_json::from_str(&s)).transpose()?)
    }
    pub fn recent_features(&self, limit: usize) -> Result<Vec<Value>> {
        let db = self.db.lock().unwrap();
        let mut query = db.prepare(
            "SELECT features FROM events WHERE kind='request' ORDER BY ts DESC LIMIT ?1",
        )?;
        let rows = query.query_map([limit as i64], |r| r.get::<_, String>(0))?;
        rows.map(|r| Ok(serde_json::from_str(&r?)?)).collect()
    }
    /// Events of one session after `since` (RFC 3339), oldest first: (kind, features, payload).
    pub fn session_events(
        &self,
        session: &str,
        since: Option<&str>,
    ) -> Result<Vec<(String, Value, Value)>> {
        let rows: Vec<(String, String, Vec<u8>, Vec<u8>)> = {
            let db = self.db.lock().unwrap();
            let mut query = db.prepare(
                "SELECT kind,features,nonce,payload FROM events WHERE session_id=?1 AND ts>?2 ORDER BY ts, rowid",
            )?;
            query
                .query_map(params![session, since.unwrap_or("")], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
                })?
                .collect::<rusqlite::Result<_>>()?
        };
        rows.into_iter()
            .map(|(kind, features, nonce, blob)| {
                Ok((
                    kind,
                    serde_json::from_str(&features)?,
                    self.decrypt(&nonce, &blob)?,
                ))
            })
            .collect()
    }

    // ---- sessions ----
    /// Record a proxied step of a session: `seen` messages recorded so far and the hash of
    /// the last one (restored after a restart so the transcript is not duplicated).
    pub fn touch_session(
        &self,
        key: &str,
        agent: &str,
        project: Option<&str>,
        user_turn: bool,
        cursor: (usize, Option<&str>),
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.db.lock().unwrap().execute(
            "INSERT INTO sessions(key,agent,project,started_at,last_seen,steps,user_turns,status,seen,last_hash) VALUES (?1,?2,?3,?4,?4,1,?5,'open',?6,?7)
             ON CONFLICT(key) DO UPDATE SET last_seen=?4, steps=steps+1, user_turns=user_turns+?5, seen=?6, last_hash=?7,
               project=COALESCE(sessions.project,?3),
               status=CASE WHEN sessions.status IN ('done','skipped','failed') THEN 'open' ELSE sessions.status END",
            params![key, agent, project, now, user_turn as i64, cursor.0 as i64, cursor.1],
        )?;
        Ok(())
    }
    pub fn session_cursor(&self, key: &str) -> Result<Option<SessionCursor>> {
        Ok(self
            .db
            .lock()
            .unwrap()
            .query_row(
                "SELECT seen,last_hash,project FROM sessions WHERE key=?1",
                [key],
                |r| Ok((r.get::<_, i64>(0)? as usize, r.get(1)?, r.get(2)?)),
            )
            .optional()?)
    }
    /// Record a whole conversation (or one more part of it) without the proxy. Messages are
    /// stored like proxied steps; `observed_at` dates the memories extracted from it.
    pub fn ingest_session(
        &self,
        key: &str,
        agent: &str,
        project: Option<&str>,
        messages: &[Value],
        observed_at: Option<&str>,
    ) -> Result<String> {
        let compact: Vec<Value> = messages
            .iter()
            .map(crate::session::compact_message)
            .collect();
        let user_turns = messages
            .iter()
            .filter(|m| m.get("role").and_then(Value::as_str) == Some("user"))
            .count() as i64;
        let event = self.event(
            agent,
            project,
            Some(key),
            "request",
            &json!({"session":key,"project":project,"ingested":true}),
            &json!({"messages":compact,"from":0}),
        )?;
        let now = Utc::now().to_rfc3339();
        self.db.lock().unwrap().execute(
            "INSERT INTO sessions(key,agent,project,started_at,last_seen,steps,user_turns,status,observed_at) VALUES (?1,?2,?3,?4,?4,1,?5,'open',?6)
             ON CONFLICT(key) DO UPDATE SET last_seen=?4, steps=steps+1, user_turns=user_turns+?5,
               project=COALESCE(sessions.project,?3), observed_at=COALESCE(?6,sessions.observed_at),
               status=CASE WHEN sessions.status IN ('done','skipped','failed') THEN 'open' ELSE sessions.status END",
            params![key, agent, project, now, user_turns, observed_at],
        )?;
        Ok(event)
    }
    /// Ask for extraction now (for example on a task_end hook).
    pub fn request_extraction(&self, key: &str) -> Result<bool> {
        Ok(self.db.lock().unwrap().execute(
            "UPDATE sessions SET status='pending' WHERE key=?1 AND status IN ('open','done','failed','skipped')",
            [key],
        )? > 0)
    }
    /// Sessions ready for extraction: pending ones, and open ones idle long enough.
    pub fn sessions_due(&self, idle: Duration) -> Result<Vec<SessionRow>> {
        let cutoff = (Utc::now() - chrono::Duration::from_std(idle)?).to_rfc3339();
        self.sessions_where(
            "(status='pending' OR (status='open' AND last_seen<=?1)) AND attempts<3",
            &cutoff,
        )
    }
    pub fn sessions(&self, limit: usize) -> Result<Vec<SessionRow>> {
        let mut rows = self.sessions_where("1=1 OR ?1=''", "")?;
        rows.truncate(limit);
        Ok(rows)
    }
    pub fn session(&self, key: &str) -> Result<Option<SessionRow>> {
        Ok(self.sessions_where("key=?1", key)?.into_iter().next())
    }
    fn sessions_where(&self, clause: &str, arg: &str) -> Result<Vec<SessionRow>> {
        let db = self.db.lock().unwrap();
        let mut query = db.prepare(&format!(
            "SELECT key,agent,project,started_at,last_seen,steps,user_turns,status,extracted_at,error,observed_at FROM sessions WHERE {clause} ORDER BY last_seen DESC"
        ))?;
        Ok(query
            .query_map([arg], |r| {
                Ok(SessionRow {
                    key: r.get(0)?,
                    agent: r.get(1)?,
                    project: r.get(2)?,
                    started_at: r.get(3)?,
                    last_seen: r.get(4)?,
                    steps: r.get(5)?,
                    user_turns: r.get(6)?,
                    status: r.get(7)?,
                    extracted_at: r.get(8)?,
                    error: r.get(9)?,
                    observed_at: r.get(10)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }
    pub fn finish_session(&self, key: &str, status: &str, error: Option<&str>) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.db.lock().unwrap().execute(
            "UPDATE sessions SET status=?2, error=?3, attempts=CASE WHEN ?2='failed' THEN attempts+1 ELSE 0 END,
               extracted_at=CASE WHEN ?2 IN ('done','skipped') THEN ?4 ELSE extracted_at END WHERE key=?1",
            params![key, status, error.map(|e| redact(e).chars().take(500).collect::<String>()), now],
        )?;
        Ok(())
    }

    // ---- recall logs, steps, usage ----
    pub fn record_recall(
        &self,
        step: &str,
        memory: &str,
        channel: &str,
        score: f64,
        reason: &str,
    ) -> Result<()> {
        self.db.lock().unwrap().execute(
            "INSERT INTO recalls VALUES (?1,?2,?3,?4,?5,?6,?7)",
            params![
                format!("rcl_{}", ulid::Ulid::new()),
                step,
                memory,
                channel,
                score,
                reason,
                Utc::now().to_rfc3339()
            ],
        )?;
        Ok(())
    }
    pub fn record_step(&self, id: &str, decision: &Value) -> Result<()> {
        self.db.lock().unwrap().execute(
            "INSERT OR REPLACE INTO steps VALUES (?1,?2)",
            params![id, decision.to_string()],
        )?;
        Ok(())
    }
    pub fn step(&self, id: &str) -> Result<Option<Value>> {
        let raw: Option<String> = self
            .db
            .lock()
            .unwrap()
            .query_row("SELECT decision FROM steps WHERE event_id=?1", [id], |r| {
                r.get(0)
            })
            .optional()?;
        Ok(raw.map(|s| serde_json::from_str(&s)).transpose()?)
    }
    pub fn record_usage(&self, step: &str, agent: &str, usage: &UsageObservation) -> Result<()> {
        self.db.lock().unwrap().execute(
            "INSERT OR REPLACE INTO usage(step_event,agent_id,ts,actual_input_tokens,output_tokens,requested_model,actual_model,cached_input_tokens,cache_write_tokens) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![step, agent, Utc::now().to_rfc3339(), usage.input_tokens, usage.output_tokens, usage.requested_model, usage.actual_model, usage.cached_input_tokens, usage.cache_write_tokens],
        )?;
        Ok(())
    }
    pub fn record_llm_call(
        &self,
        role: &str,
        usage: &UsageObservation,
        outcome: &str,
    ) -> Result<()> {
        self.db.lock().unwrap().execute(
            "INSERT INTO llm_calls(id,role,ts,input_tokens,output_tokens,outcome,requested_model,actual_model) VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
            params![format!("llm_{}", ulid::Ulid::new()), role, Utc::now().to_rfc3339(), usage.input_tokens, usage.output_tokens, outcome, usage.requested_model, usage.actual_model],
        )?;
        Ok(())
    }
    pub fn llm_calls_today(&self) -> Result<u32> {
        let date = Utc::now().format("%Y-%m-%d").to_string();
        Ok(self.db.lock().unwrap().query_row(
            "SELECT COUNT(*) FROM llm_calls WHERE ts>=?1",
            [date],
            |r| r.get(0),
        )?)
    }
    pub fn stats(&self) -> Result<Value> {
        let db = self.db.lock().unwrap();
        let (steps, input, output, cached): (i64, i64, i64, i64) = db.query_row(
            "SELECT COUNT(*),COALESCE(SUM(actual_input_tokens),0),COALESCE(SUM(output_tokens),0),COALESCE(SUM(cached_input_tokens),0) FROM usage",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )?;
        let (injections, used, misses): (i64, i64, i64) = db.query_row(
            "SELECT (SELECT COUNT(*) FROM recalls),(SELECT COUNT(*) FROM recall_feedback WHERE utility>0),(SELECT COUNT(*) FROM missed_recalls)",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )?;
        let (sessions, extracted, engine_calls): (i64, i64, i64) = db.query_row(
            "SELECT (SELECT COUNT(*) FROM sessions),(SELECT COUNT(*) FROM sessions WHERE extracted_at IS NOT NULL),(SELECT COUNT(*) FROM llm_calls)",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )?;
        let (engine_input, engine_output, engine_failed): (i64, i64, i64) = db.query_row(
            "SELECT COALESCE(SUM(input_tokens),0),COALESCE(SUM(output_tokens),0),COALESCE(SUM(outcome<>'received'),0) FROM llm_calls",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )?;
        let mismatch: i64 = db.query_row(
            "SELECT COUNT(*) FROM usage WHERE requested_model IS NOT NULL AND actual_model IS NOT NULL AND requested_model<>actual_model",
            [],
            |r| r.get(0),
        )?;
        drop(db);
        let memories = self.memories();
        let pending = memories
            .iter()
            .filter(|m| m.status == "pending_review")
            .count();
        let active = memories.iter().filter(|m| m.recallable()).count();
        let episodes = self.with_episodes(|e| e.meta.len());
        Ok(json!({
            "memories": memories.len(), "active_memories": active, "pending_review": pending,
            "episodes": episodes,
            "embedding_model": self.embedder.get().map(|e| e.name.clone()),
            "steps": steps, "injections": injections, "labeled_used": used, "missed_recalls": misses,
            "sessions": sessions, "extracted_sessions": extracted, "engine_llm_calls": engine_calls,
            "engine_input_tokens": engine_input, "engine_output_tokens": engine_output, "engine_failed_calls": engine_failed,
            "input_tokens": input, "output_tokens": output, "cached_input_tokens": cached,
            "model_mismatch_steps": mismatch,
        }))
    }

    // ---- feedback ----
    pub fn record_feedback(
        &self,
        step: &str,
        memory: &str,
        cited: Option<bool>,
        consistent: Option<bool>,
        result: Option<&str>,
    ) -> Result<()> {
        if ![None, Some("success"), Some("failure"), Some("unknown")].contains(&result) {
            bail!("invalid task result");
        }
        let db = self.db.lock().unwrap();
        let exists: i64 = db.query_row(
            "SELECT COUNT(*) FROM recalls WHERE step_event=?1 AND memory_id=?2",
            params![step, memory],
            |r| r.get(0),
        )?;
        if exists == 0 {
            bail!("recall not found for step and memory");
        }
        let utility = match (cited, consistent, result) {
            (Some(true), _, Some("success")) | (_, Some(true), Some("success")) => Some(1.0),
            (Some(true), _, _) | (_, Some(true), _) => Some(0.5),
            (Some(false), Some(false), _) => Some(0.0),
            _ => None,
        };
        db.execute("INSERT INTO recall_feedback(step_event,memory_id,cited,action_consistent,task_result,utility,updated_at) VALUES (?1,?2,?3,?4,?5,?6,?7) ON CONFLICT(step_event,memory_id) DO UPDATE SET cited=excluded.cited,action_consistent=excluded.action_consistent,task_result=excluded.task_result,utility=excluded.utility,updated_at=excluded.updated_at",params![step,memory,cited,consistent,result,utility,Utc::now().to_rfc3339()])?;
        Ok(())
    }
    pub fn recall_ids_for_step(&self, step: &str) -> Result<Vec<String>> {
        let db = self.db.lock().unwrap();
        let mut query = db.prepare("SELECT memory_id FROM recalls WHERE step_event=?1")?;
        Ok(query
            .query_map([step], |r| r.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?)
    }
    pub fn memory_usage(&self, id: &str) -> Result<MemoryUsage> {
        Ok(self.db.lock().unwrap().query_row(
            "SELECT COUNT(*),COALESCE(SUM(CASE WHEN f.utility>0 THEN 1 ELSE 0 END),0) FROM recalls r LEFT JOIN recall_feedback f ON f.step_event=r.step_event AND f.memory_id=r.memory_id WHERE r.memory_id=?1",
            [id],
            |r| Ok(MemoryUsage { recalled: r.get(0)?, used: r.get(1)? }),
        )?)
    }
    /// (times fired, fires with feedback, fires labeled useful).
    pub fn trigger_usage(&self, id: &str) -> Result<(u64, u64, u64)> {
        Ok(self.db.lock().unwrap().query_row(
            "SELECT COUNT(*),COUNT(f.utility),COALESCE(SUM(CASE WHEN f.utility>0 THEN 1 ELSE 0 END),0) FROM recalls r LEFT JOIN recall_feedback f ON f.step_event=r.step_event AND f.memory_id=r.memory_id WHERE r.reason=?1",
            [format!("trigger:{id}")],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )?)
    }
    pub fn record_miss(&self, event: &str, memory: &str, reason: &str) -> Result<String> {
        if self.event_features(event)?.is_none() || self.memory(memory).is_none() {
            bail!("event or memory not found");
        }
        let id = format!("miss_{}", ulid::Ulid::new());
        self.db.lock().unwrap().execute(
            "INSERT INTO missed_recalls VALUES (?1,?2,?3,?4,?5)",
            params![id, event, memory, redact(reason), Utc::now().to_rfc3339()],
        )?;
        Ok(id)
    }
    pub fn debug_step(&self, id: &str) -> Result<Value> {
        let request = self
            .event_payload(id)?
            .ok_or_else(|| anyhow::anyhow!("step event not found"))?;
        let decision = self.step(id)?;
        let (linked, recalls) = {
            let db = self.db.lock().unwrap();
            let mut query = db.prepare(
                "SELECT id,kind FROM events WHERE json_extract(features,'$.step_event')=?1 ORDER BY ts",
            )?;
            let linked = query
                .query_map([id], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            let mut recall_query = db.prepare(
                "SELECT memory_id,channel,score,reason FROM recalls WHERE step_event=?1 ORDER BY score DESC",
            )?;
            let recalls = recall_query
                .query_map([id], |r| {
                    Ok(json!({"memory_id":r.get::<_,String>(0)?,"channel":r.get::<_,String>(1)?,"score":r.get::<_,f64>(2)?,"reason":r.get::<_,String>(3)?}))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            (linked, recalls)
        };
        let mut compiled = None;
        let mut response = None;
        for (event_id, kind) in linked {
            match kind.as_str() {
                "compiled" => compiled = self.event_payload(&event_id)?,
                "response" => response = self.event_payload(&event_id)?,
                _ => {}
            }
        }
        Ok(
            json!({"step_event":id,"request":request,"compiled":compiled,"response":response,"decision":decision,"recalls":recalls}),
        )
    }

    // ---- memories: cache, index, vectors ----
    pub fn memories(&self) -> Vec<Memory> {
        let mut all: Vec<Memory> = self.cache.read().unwrap().all.values().cloned().collect();
        all.sort_by(|a, b| a.id.cmp(&b.id));
        all
    }
    pub fn memory(&self, id: &str) -> Option<Memory> {
        if !memory::valid_id(id, "mem_") {
            return None;
        }
        self.cache.read().unwrap().all.get(id).cloned()
    }
    /// Read every memory file, converting v1 files in place (one git commit), and rebuild
    /// the index. Vectors are reused when the memory text is unchanged.
    pub fn reload_memories(&self) -> Result<()> {
        let mut cache = Cache::default();
        let mut migrated = 0;
        for entry in fs::read_dir(self.root.join("memory"))? {
            let path = entry?.path();
            if path.extension().and_then(|s| s.to_str()) != Some("md") {
                continue;
            }
            let (memory, legacy) = match memory::read(&path) {
                Ok(value) => value,
                Err(error) => {
                    eprintln!("ctx: skipping {}: {error:#}", path.display());
                    continue;
                }
            };
            if legacy {
                memory::write(&path, &memory)?;
                migrated += 1;
            }
            Self::index_into(&mut cache, &memory);
            cache.all.insert(memory.id.clone(), memory);
        }
        if migrated > 0 {
            self.commit_all(&format!("migrate {migrated} memories to schema v2"))?;
        }
        self.load_vectors(&mut cache)?;
        *self.cache.write().unwrap() = cache;
        Ok(())
    }
    fn index_into(cache: &mut Cache, memory: &Memory) {
        let text = memory.search_text();
        cache.index.upsert(&memory.id, &memory.index_text());
        cache.text_hash.insert(
            memory.id.clone(),
            hex::encode(Sha256::digest(text.as_bytes())),
        );
    }
    fn load_vectors(&self, cache: &mut Cache) -> Result<()> {
        let Some(model) = self.vector_model() else {
            return Ok(());
        };
        let db = self.db.lock().unwrap();
        let mut query =
            db.prepare("SELECT id,hash,data FROM vectors WHERE model=?1 AND id NOT LIKE 'ep\\_%' ESCAPE '\\'")?;
        let rows = query.query_map([&model], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Vec<u8>>(2)?,
            ))
        })?;
        for row in rows {
            let (id, hash, data) = row?;
            if cache.text_hash.get(&id) == Some(&hash) {
                cache.index.set_vector(&id, bytes_to_vector(&data));
            }
        }
        Ok(())
    }
    fn vector_model(&self) -> Option<String> {
        self.config
            .embedding
            .enabled
            .then(|| self.config.embedding.model.clone())
    }
    /// Load the embedding model if needed and embed memories that lack a current vector.
    /// Returns how many were embedded. Blocking: seconds on first load.
    pub fn ensure_vectors(&self) -> Result<usize> {
        let Some(model) = self.vector_model() else {
            return Ok(0);
        };
        let Some(embedder) = self.embedder.load(&model, self.config.embedding.workers) else {
            return Ok(0);
        };
        let missing: Vec<(String, String, String, String)> = {
            let cache = self.cache.read().unwrap();
            cache
                .all
                .values()
                .filter(|m| !cache.index.has_vector(&m.id))
                .map(|m| {
                    (
                        m.id.clone(),
                        cache.text_hash.get(&m.id).cloned().unwrap_or_default(),
                        m.title.clone(),
                        m.body.clone(),
                    )
                })
                .collect()
        };
        for chunk in missing.chunks(32) {
            let docs: Vec<(String, String)> = chunk
                .iter()
                .map(|(_, _, title, body)| (title.clone(), body.clone()))
                .collect();
            let vectors = embedder.embed_documents(&docs)?;
            let db = self.db.lock().unwrap();
            let mut cache = self.cache.write().unwrap();
            for ((id, hash, _, _), vector) in chunk.iter().zip(vectors) {
                db.execute(
                    "INSERT OR REPLACE INTO vectors(id,hash,model,data) VALUES (?1,?2,?3,?4)",
                    params![id, hash, model, vector_to_bytes(&vector)],
                )?;
                if cache.text_hash.get(id) == Some(hash) {
                    cache.index.set_vector(id, vector);
                }
            }
        }
        let turns = self.ensure_turn_vectors(&model, &embedder)?;
        Ok(missing.len() + turns)
    }
    // ---- episodes (raw conversation excerpts) ----
    fn load_episodes(&self) -> Result<()> {
        let mut episodes = Episodes::default();
        let db = self.db.lock().unwrap();
        // Sessions in the order they were stored: the order of the conversation.
        let order: HashMap<String, i64> = db
            .prepare("SELECT key, rowid FROM sessions")?
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        let mut query =
            db.prepare("SELECT id,session,project,observed_at,seq,nonce,text FROM episodes")?;
        let rows = query.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<String>>(2)?,
                r.get::<_, Option<String>>(3)?,
                r.get::<_, i64>(4)?,
                r.get::<_, Vec<u8>>(5)?,
                r.get::<_, Vec<u8>>(6)?,
            ))
        })?;
        type SessionExcerpts = (Option<String>, Option<String>, Vec<(i64, String, String)>);
        let mut sessions: HashMap<String, SessionExcerpts> = HashMap::new();
        for row in rows {
            let (id, session, project, observed_at, seq, nonce, blob) = row?;
            let text = self.decrypt(&nonce, &blob)?.as_str().unwrap_or_default().to_owned();
            episodes.index.upsert(&id, &text);
            episodes.meta.insert(
                id.clone(),
                EpisodeMeta {
                    session: session.clone(),
                    project: project.clone(),
                    observed_at: observed_at.clone(),
                    turn: 0,
                },
            );
            sessions.entry(session).or_insert((project, observed_at, vec![])).2.push((seq, id, text));
        }
        let mut sessions: Vec<(String, SessionExcerpts)> = sessions.into_iter().collect();
        sessions.sort_by(|a, b| {
            (order.get(&a.0).copied().unwrap_or(i64::MAX), &a.0).cmp(&(order.get(&b.0).copied().unwrap_or(i64::MAX), &b.0))
        });
        for (session, (project, observed_at, mut excerpts)) in sessions {
            excerpts.sort_by_key(|e| e.0);
            let excerpts: Vec<(String, String)> = excerpts.into_iter().map(|(_, id, text)| (id, text)).collect();
            episodes.add_turns(&session, project.as_deref(), observed_at.as_deref(), &excerpts);
        }
        let mut query = db.prepare("SELECT id,nonce,text FROM turn_notes")?;
        let notes = query.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?, r.get::<_, Vec<u8>>(2)?)))?;
        for row in notes {
            let (id, nonce, blob) = row?;
            let note = self.decrypt(&nonce, &blob)?.as_str().unwrap_or_default().to_owned();
            episodes.set_note(&id, &note);
        }
        if let Some(model) = self.vector_model() {
            let mut query = db.prepare(
                "SELECT id,data FROM vectors WHERE model=?1 AND id LIKE 'ep\\_%' ESCAPE '\\'",
            )?;
            let rows = query.query_map([&model], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
            })?;
            for row in rows {
                let (id, data) = row?;
                if episodes.meta.contains_key(&id) {
                    episodes.index.set_vector(&id, bytes_to_vector(&data));
                }
            }
            // The user's messages; a vector counts only for the text it was made from.
            let mut query = db.prepare(
                "SELECT id,hash,data FROM vectors WHERE model=?1 AND id LIKE 'tu\\_%' ESCAPE '\\'",
            )?;
            let rows = query.query_map([&model], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, Vec<u8>>(2)?))
            })?;
            for row in rows {
                let (id, hash, data) = row?;
                if episodes.turn_meta.get(&id).is_some_and(|t| text_hash(&t.document()) == hash) {
                    episodes.turns.set_vector(&id, bytes_to_vector(&data));
                }
            }
        }
        *self.episodes.write().unwrap() = episodes;
        Ok(())
    }
    /// Store the gist of the assistant's reply for turns of `session` (by turn id) and
    /// re-embed them. Returns how many were stored.
    pub fn set_turn_notes(&self, session: &str, notes: &[(String, String)]) -> Result<usize> {
        let mut stored = 0;
        {
            let db = self.db.lock().unwrap();
            let mut episodes = self.episodes.write().unwrap();
            for (id, note) in notes {
                if !episodes.set_note(id, note) {
                    continue;
                }
                let (nonce, blob) = self.seal(&json!(note))?;
                db.execute(
                    "INSERT OR REPLACE INTO turn_notes(id,session,nonce,text) VALUES (?1,?2,?3,?4)",
                    params![id, session, nonce.as_slice(), blob],
                )?;
                stored += 1;
            }
        }
        if let Some(model) = self.vector_model()
            && let Some(embedder) = self.embedder.load(&model, self.config.embedding.workers)
        {
            self.ensure_turn_vectors(&model, &embedder)?;
        }
        Ok(stored)
    }
    /// Embed the users' messages that lack a vector (turns of earlier versions, or added
    /// while the model was unavailable). Returns how many were embedded.
    fn ensure_turn_vectors(&self, model: &str, embedder: &embed::Embedder) -> Result<usize> {
        let missing: Vec<(String, String)> = self.with_episodes(|e| {
            e.turn_meta
                .iter()
                .filter(|(id, t)| !t.text.is_empty() && !e.turns.has_vector(id))
                .map(|(id, t)| (id.clone(), t.document()))
                .collect()
        });
        for chunk in missing.chunks(32) {
            let docs: Vec<(String, String)> = chunk.iter().map(|(_, text)| ("user message".to_owned(), text.clone())).collect();
            let vectors = embedder.embed_documents(&docs)?;
            let db = self.db.lock().unwrap();
            let mut episodes = self.episodes.write().unwrap();
            for ((id, text), vector) in chunk.iter().zip(vectors) {
                db.execute(
                    "INSERT OR REPLACE INTO vectors(id,hash,model,data) VALUES (?1,?2,?3,?4)",
                    params![id, text_hash(text), model, vector_to_bytes(&vector)],
                )?;
                episodes.turns.set_vector(id, vector);
            }
        }
        Ok(missing.len())
    }
    /// Store a finished session's conversation as searchable excerpts (embedded when the
    /// embedding model is enabled). Returns how many were added.
    pub fn add_episodes(&self, session: &SessionRow, entries: &[(String, String)]) -> Result<usize> {
        // A retried extraction sees the same conversation again; keep excerpts once.
        let known: std::collections::HashSet<String> = {
            let db = self.db.lock().unwrap();
            let mut query = db.prepare("SELECT hash FROM episodes WHERE session=?1")?;
            query
                .query_map([&session.key], |r| r.get::<_, String>(0))?
                .collect::<rusqlite::Result<_>>()?
        };
        let texts: Vec<String> = episode::chunks(entries)
            .into_iter()
            .filter(|t| !known.contains(&hex::encode(Sha256::digest(t.as_bytes()))))
            .collect();
        if texts.is_empty() {
            return Ok(0);
        }
        let model = self.vector_model();
        let embedder = model
            .as_ref()
            .and_then(|m| self.embedder.load(m, self.config.embedding.workers));
        let vectors = match &embedder {
            Some(embedder) => {
                let mut vectors = Vec::with_capacity(texts.len());
                for batch in texts.chunks(16) {
                    let docs: Vec<(String, String)> = batch
                        .iter()
                        .map(|t| ("conversation excerpt".to_owned(), t.clone()))
                        .collect();
                    vectors.extend(embedder.embed_documents(&docs)?);
                }
                Some(vectors)
            }
            None => None,
        };
        let ids: Vec<String> = texts
            .iter()
            .map(|_| format!("ep_{}", ulid::Ulid::new()))
            .collect();
        {
            let mut db = self.db.lock().unwrap();
            let tx = db.transaction()?;
            let first: i64 = tx.query_row(
                "SELECT COUNT(*) FROM episodes WHERE session=?1",
                [&session.key],
                |r| r.get(0),
            )?;
            for (i, (id, text)) in ids.iter().zip(&texts).enumerate() {
                let hash = hex::encode(Sha256::digest(text.as_bytes()));
                let (nonce, blob) = self.seal(&json!(text))?;
                tx.execute(
                    "INSERT INTO episodes VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
                    params![
                        id,
                        session.key,
                        session.project,
                        session.observed_at,
                        first + i as i64,
                        hash,
                        nonce.as_slice(),
                        blob
                    ],
                )?;
                if let (Some(model), Some(vectors)) = (&model, &vectors) {
                    tx.execute(
                        "INSERT OR REPLACE INTO vectors(id,hash,model,data) VALUES (?1,?2,?3,?4)",
                        params![id, hash, model, vector_to_bytes(&vectors[i])],
                    )?;
                }
            }
            tx.commit()?;
        }
        {
            let mut episodes = self.episodes.write().unwrap();
            for (i, (id, text)) in ids.iter().zip(&texts).enumerate() {
                episodes.index.upsert(id, text);
                if let Some(vectors) = &vectors {
                    episodes.index.set_vector(id, vectors[i].clone());
                }
                episodes.meta.insert(
                    id.clone(),
                    EpisodeMeta {
                        session: session.key.clone(),
                        project: session.project.clone(),
                        observed_at: session.observed_at.clone(),
                        turn: 0,
                    },
                );
            }
            let excerpts: Vec<(String, String)> = ids.iter().cloned().zip(texts.iter().cloned()).collect();
            episodes.add_turns(&session.key, session.project.as_deref(), session.observed_at.as_deref(), &excerpts);
        }
        if let (Some(model), Some(embedder)) = (&model, &embedder) {
            self.ensure_turn_vectors(model, embedder)?;
        }
        Ok(texts.len())
    }
    /// Run `f` with the excerpt index (read lock).
    pub fn with_episodes<T>(&self, f: impl FnOnce(&Episodes) -> T) -> T {
        f(&self.episodes.read().unwrap())
    }
    pub fn episode_text(&self, id: &str) -> Result<Option<String>> {
        let row: Option<(Vec<u8>, Vec<u8>)> = self
            .db
            .lock()
            .unwrap()
            .query_row("SELECT nonce,text FROM episodes WHERE id=?1", [id], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .optional()?;
        row.map(|(nonce, blob)| {
            Ok(self
                .decrypt(&nonce, &blob)?
                .as_str()
                .unwrap_or_default()
                .to_owned())
        })
        .transpose()
    }

    /// Run `f` with the search index (read lock).
    pub fn with_index<T>(&self, f: impl FnOnce(&Index, &HashMap<String, Memory>) -> T) -> T {
        let cache = self.cache.read().unwrap();
        f(&cache.index, &cache.all)
    }
    /// A memory file changed on disk (edited by hand or by git); refresh just that entry.
    pub fn reload_file(&self, path: &Path) -> Result<()> {
        let Some(id) = path
            .file_stem()
            .and_then(|s| s.to_str())
            .filter(|s| memory::valid_id(s, "mem_"))
        else {
            return Ok(());
        };
        if !path.exists() {
            let mut cache = self.cache.write().unwrap();
            cache.all.remove(id);
            cache.index.remove(id);
            cache.text_hash.remove(id);
            return Ok(());
        }
        let (memory, legacy) = memory::read(path)?;
        if legacy {
            memory::write(path, &memory)?;
        }
        if self.memory(id).as_ref() == Some(&memory) {
            return Ok(());
        }
        self.apply_to_cache(&memory);
        Ok(())
    }
    fn apply_to_cache(&self, memory: &Memory) {
        let embedder = self.embedder.get();
        let mut cache = self.cache.write().unwrap();
        let text = memory.search_text();
        let hash = hex::encode(Sha256::digest(text.as_bytes()));
        let changed = cache.text_hash.get(&memory.id) != Some(&hash);
        Self::index_into(&mut cache, memory);
        cache.all.insert(memory.id.clone(), memory.clone());
        if changed {
            cache.index.remove_vector_only(&memory.id);
            drop(cache);
            if let Some(embedder) = embedder
                && let Ok(mut vectors) =
                    embedder.embed_documents(&[(memory.title.clone(), memory.body.clone())])
            {
                let vector = vectors.remove(0);
                let _ = self.db.lock().unwrap().execute(
                    "INSERT OR REPLACE INTO vectors(id,hash,model,data) VALUES (?1,?2,?3,?4)",
                    params![memory.id, hash, embedder.name, vector_to_bytes(&vector)],
                );
                let mut cache = self.cache.write().unwrap();
                if cache.text_hash.get(&memory.id) == Some(&hash) {
                    cache.index.set_vector(&memory.id, vector);
                }
            }
        }
    }

    // ---- memory writes ----
    pub fn remember(&self, input: NewMemory, source: &str) -> Result<Memory> {
        let memory = memory::create(input, source)?;
        self.save_memory(&memory, &format!("add {}", memory.id))?;
        Ok(memory)
    }
    /// Write one memory file, commit it, and update the index in place.
    pub fn save_memory(&self, memory: &Memory, reason: &str) -> Result<()> {
        if !memory::valid_id(&memory.id, "mem_") {
            bail!("invalid memory id");
        }
        if memory.kind == "rule" && memory.source != "user" && memory.status == "active" {
            bail!("only a user can activate a rule");
        }
        let path = self.memory_path(&memory.id);
        memory::write(&path, memory)?;
        self.commit_paths(&[&path], reason)?;
        self.apply_to_cache(memory);
        Ok(())
    }
    fn memory_path(&self, id: &str) -> PathBuf {
        self.root.join("memory").join(format!("{id}.md"))
    }
    pub fn apply_batch(&self, task: &str, updates: &[Memory]) -> Result<String> {
        if updates.is_empty() {
            bail!("empty maintenance batch");
        }
        let mut before = BTreeMap::<String, Option<Memory>>::new();
        let mut after = BTreeMap::<String, Option<Memory>>::new();
        for memory in updates {
            if !memory::valid_id(&memory.id, "mem_") || after.contains_key(&memory.id) {
                bail!("invalid or duplicate memory id");
            }
            let prior = self.memory(&memory.id);
            if prior
                .as_ref()
                .is_some_and(|old| old.kind == "rule" && old != memory)
                || (prior.is_none() && memory.kind == "rule")
            {
                bail!("maintenance cannot change rules");
            }
            before.insert(memory.id.clone(), prior);
            after.insert(memory.id.clone(), Some(memory.clone()));
        }
        let id = format!("batch_{}", ulid::Ulid::new());
        self.db.lock().unwrap().execute(
            "INSERT INTO maintenance_batches(id,task,status,before_json,after_json,created_at) VALUES (?1,?2,'prepared',?3,?4,?5)",
            params![id, task, serde_json::to_string(&with_bodies(&before))?, serde_json::to_string(&with_bodies(&after))?, Utc::now().to_rfc3339()],
        )?;
        let result = (|| -> Result<()> {
            let paths = updates
                .iter()
                .map(|m| {
                    let path = self.memory_path(&m.id);
                    memory::write(&path, m).map(|_| path)
                })
                .collect::<Result<Vec<_>>>()?;
            self.commit_paths(
                &paths.iter().map(PathBuf::as_path).collect::<Vec<_>>(),
                &format!("maintenance {task} {id}"),
            )?;
            for memory in updates {
                self.apply_to_cache(memory);
            }
            self.db.lock().unwrap().execute(
                "UPDATE maintenance_batches SET status='applied' WHERE id=?1",
                [&id],
            )?;
            Ok(())
        })();
        if let Err(error) = result {
            let _ = self.rollback_batch(&id);
            return Err(error);
        }
        Ok(id)
    }
    pub fn maintenance_batches(&self) -> Result<Value> {
        let db = self.db.lock().unwrap();
        let mut query = db.prepare(
            "SELECT id,task,status,created_at FROM maintenance_batches ORDER BY created_at DESC LIMIT 50",
        )?;
        let rows = query.query_map([], |r| {
            Ok(json!({"id":r.get::<_,String>(0)?,"task":r.get::<_,String>(1)?,"status":r.get::<_,String>(2)?,"created_at":r.get::<_,String>(3)?}))
        })?;
        Ok(json!(rows.collect::<rusqlite::Result<Vec<_>>>()?))
    }
    pub fn rollback_batch(&self, id: &str) -> Result<()> {
        let (status, before_raw, after_raw): (String, String, String) =
            self.db.lock().unwrap().query_row(
                "SELECT status,before_json,after_json FROM maintenance_batches WHERE id=?1",
                [id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )?;
        if status != "prepared" && status != "applied" {
            bail!("batch is not active");
        }
        let before = restore_bodies(serde_json::from_str(&before_raw)?);
        let after = restore_bodies(serde_json::from_str(&after_raw)?);
        if status == "applied" {
            for (memory_id, expected) in &after {
                let path = self.memory_path(memory_id);
                let current = if path.exists() {
                    Some(memory::read(&path)?.0)
                } else {
                    None
                };
                if &current != expected {
                    bail!("memory {memory_id} changed after batch; refusing rollback");
                }
            }
        }
        let mut paths = vec![];
        for (memory_id, prior) in before {
            let path = self.memory_path(&memory_id);
            if let Some(memory) = prior {
                memory::write(&path, &memory)?;
            } else if path.exists() {
                fs::remove_file(&path)?;
            }
            paths.push(path);
        }
        self.commit_paths(
            &paths.iter().map(PathBuf::as_path).collect::<Vec<_>>(),
            &format!("rollback {id}"),
        )?;
        for path in &paths {
            self.reload_file(path)?;
        }
        self.db.lock().unwrap().execute(
            "UPDATE maintenance_batches SET status='rolled_back' WHERE id=?1",
            [id],
        )?;
        Ok(())
    }
    fn recover_prepared_batches(&self) -> Result<()> {
        let ids = {
            let db = self.db.lock().unwrap();
            let mut query =
                db.prepare("SELECT id FROM maintenance_batches WHERE status='prepared'")?;
            query
                .query_map([], |r| r.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?
        };
        for id in ids {
            self.rollback_batch(&id)?;
        }
        Ok(())
    }
    pub fn review_memory(&self, id: &str, approve: bool) -> Result<bool> {
        let Some(mut memory) = self.memory(id) else {
            return Ok(false);
        };
        if memory.status != "pending_review" {
            bail!("memory is not pending review");
        }
        let pending_since = memory.updated_at.clone();
        memory.status = if approve { "active" } else { "archived" }.into();
        memory.updated_at = Utc::now().to_rfc3339();
        if approve && memory.kind == "rule" {
            // Approval by the user makes the rule the user's own.
            memory.source = "user".into();
        }
        self.save_memory(
            &memory,
            &format!("review {} {id}", if approve { "approve" } else { "reject" }),
        )?;
        let wait_seconds = DateTime::parse_from_rfc3339(&pending_since)
            .map(|since| Utc::now().signed_duration_since(since).num_seconds().max(0))
            .unwrap_or(0);
        self.db.lock().unwrap().execute(
            "INSERT INTO memory_reviews(id,memory_id,decision,wait_seconds,reviewed_at) VALUES (?1,?2,?3,?4,?5)",
            params![format!("rev_{}", ulid::Ulid::new()), id, if approve {"approve"} else {"reject"}, wait_seconds, Utc::now().to_rfc3339()],
        )?;
        Ok(true)
    }
    pub fn flag_memory(&self, id: &str, reason: &str) -> Result<bool> {
        if !["wrong", "outdated", "irrelevant"].contains(&reason) {
            bail!("invalid flag reason");
        }
        let Some(mut memory) = self.memory(id) else {
            return Ok(false);
        };
        self.db.lock().unwrap().execute(
            "INSERT INTO memory_flags VALUES (?1,?2,?3,?4)",
            params![
                format!("flag_{}", ulid::Ulid::new()),
                id,
                reason,
                Utc::now().to_rfc3339()
            ],
        )?;
        if memory.kind != "rule" && reason != "irrelevant" && memory.status == "active" {
            memory.status = "contested".into();
            memory.confidence = (memory.confidence - 0.2).max(0.1);
            memory.updated_at = Utc::now().to_rfc3339();
            self.save_memory(&memory, &format!("flag {reason} {id}"))?;
        }
        Ok(true)
    }
    pub fn create_intent(&self, when: &str, then: &str, expires: Option<&str>) -> Result<Memory> {
        if when.trim().chars().count() < 3 || when.len() > 120 || then.trim().is_empty() {
            bail!("invalid intent");
        }
        if let Some(expires) = expires {
            DateTime::parse_from_rfc3339(expires)?;
        }
        let mut memory = memory::create(
            NewMemory {
                content: format!("When {when}, {then}"),
                kind: "intent".into(),
                scope: "global".into(),
                title: Some(then.chars().take(80).collect()),
                triggers: vec![memory::NewTrigger {
                    kind: "keyword".into(),
                    pattern: when.into(),
                    before_action: false,
                }],
            },
            "agent",
        )?;
        memory.expires = expires.map(str::to_owned);
        self.save_memory(&memory, &format!("propose intent {}", memory.id))?;
        Ok(memory)
    }
    pub fn archive(&self, id: &str) -> Result<bool> {
        let Some(mut memory) = self.memory(id) else {
            return Ok(false);
        };
        memory.status = "archived".into();
        memory.updated_at = Utc::now().to_rfc3339();
        self.save_memory(&memory, &format!("archive {id}"))?;
        Ok(true)
    }
    pub fn update_memory(&self, id: &str, content: &str) -> Result<Option<Memory>> {
        if content.trim().is_empty() || content.len() > 20_000 {
            bail!("content must be 1..20000 bytes");
        }
        let Some(mut memory) = self.memory(id) else {
            return Ok(None);
        };
        if matches!(memory.status.as_str(), "archived" | "superseded") {
            bail!("archived memory cannot be edited");
        }
        let old_default = memory::default_title(&memory.body);
        memory.body = redact(content.trim());
        if memory.title == old_default {
            memory.title = memory::default_title(&memory.body);
        }
        memory.updated_at = Utc::now().to_rfc3339();
        self.save_memory(&memory, &format!("edit {id}"))?;
        Ok(Some(memory))
    }

    // ---- git ----
    fn git(&self, args: &[&str]) -> Result<bool> {
        Ok(std::process::Command::new("git")
            .arg("-C")
            .arg(self.root.join("memory"))
            .args([
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "user.name=ctx",
                "-c",
                "user.email=ctx@localhost",
            ])
            .args(args)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()?
            .success())
    }
    /// Commit the given memory files. History is best-effort when git is unavailable.
    fn commit_paths(&self, paths: &[&Path], message: &str) -> Result<()> {
        let _lock = self.git_lock.lock().unwrap();
        if !self.config.history || !self.root.join("memory/.git").exists() {
            return Ok(());
        }
        let names: Vec<String> = paths
            .iter()
            .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(str::to_owned))
            .collect();
        let mut add = vec!["add", "-A", "--"];
        add.extend(names.iter().map(String::as_str));
        if !self.git(&add)? {
            bail!("git add failed");
        }
        if self.git(&["diff", "--cached", "--quiet"])? {
            return Ok(());
        }
        if !self.git(&["commit", "-qm", message])? {
            bail!("git commit failed");
        }
        Ok(())
    }
    fn commit_all(&self, message: &str) -> Result<()> {
        let _lock = self.git_lock.lock().unwrap();
        if !self.config.history || !self.root.join("memory/.git").exists() {
            return Ok(());
        }
        self.git(&["add", "-A"])?;
        if !self.git(&["diff", "--cached", "--quiet"])? {
            self.git(&["commit", "-qm", message])?;
        }
        Ok(())
    }

    // ---- experimental: eviction, recheck, gate labels ----
    pub fn eviction(
        &self,
        agent: &str,
        project: Option<&str>,
        session: Option<&str>,
        content: &str,
    ) -> Result<String> {
        let hash = hex::encode(Sha256::digest(content.as_bytes()));
        if let Some(existing) = self.evicted(content)? {
            return Ok(existing);
        }
        let event = self.event(
            agent,
            project,
            session,
            "archived_tool_result",
            &json!({"archived":true}),
            &json!({"content":content}),
        )?;
        let first = content
            .lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("tool output");
        let summary: String = redact(first).chars().take(100).collect();
        let placeholder = format!(
            "[ctx:archived {event} | {} lines | {} | expand(\"{event}\") to retrieve]",
            content.lines().count(),
            summary
        );
        self.db.lock().unwrap().execute(
            "INSERT OR IGNORE INTO evictions(hash,event_id,placeholder) VALUES (?1,?2,?3)",
            params![hash, event, placeholder],
        )?;
        Ok(placeholder)
    }
    pub fn evicted(&self, content: &str) -> Result<Option<String>> {
        let hash = hex::encode(Sha256::digest(content.as_bytes()));
        Ok(self
            .db
            .lock()
            .unwrap()
            .query_row(
                "SELECT placeholder FROM evictions WHERE hash=?1",
                [hash],
                |r| r.get(0),
            )
            .optional()?)
    }
    pub fn reserve_recheck(&self, task: &str, step: &str) -> Result<bool> {
        let mut db = self.db.lock().unwrap();
        let tx = db.transaction()?;
        let count: i64 = tx.query_row(
            "SELECT COUNT(*) FROM rechecks WHERE task_key=?1",
            [task],
            |r| r.get(0),
        )?;
        if count >= 5 {
            return Ok(false);
        }
        let inserted = tx.execute(
            "INSERT OR IGNORE INTO rechecks(step_event,task_key,created_at) VALUES (?1,?2,?3)",
            params![step, task, Utc::now().to_rfc3339()],
        )?;
        tx.commit()?;
        Ok(inserted == 1)
    }
    /// Labeled Gate decisions. Fallbacks (timeouts, pending) are excluded; a deferred
    /// decision uses the features of the step it was made on.
    pub fn gate_training_samples(&self) -> Result<Vec<GateSample>> {
        let db = self.db.lock().unwrap();
        let mut query = db.prepare("SELECT e.features,s.decision,MAX(f.utility) FROM recall_feedback f JOIN events e ON e.id=f.step_event JOIN steps s ON s.event_id=e.id WHERE f.utility IS NOT NULL AND json_extract(s.decision,'$.gate.called')=1 AND json_type(s.decision,'$.gate.decision') IN ('true','false') GROUP BY f.step_event ORDER BY e.ts")?;
        let rows = query.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, f64>(2)?,
            ))
        })?;
        let mut samples = vec![];
        for row in rows {
            let (features, decision, utility) = row?;
            let parsed: Value = serde_json::from_str(&decision)?;
            let features = match parsed.pointer("/gate/source_features") {
                Some(source) => source.clone(),
                None => serde_json::from_str(&features)?,
            };
            samples.push(GateSample {
                features,
                positive: utility > 0.0,
                gate_recalled: parsed
                    .pointer("/gate/decision")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            });
        }
        Ok(samples)
    }
}

/// Batch snapshots keep bodies (which `Memory` does not serialize).
fn with_bodies(
    map: &BTreeMap<String, Option<Memory>>,
) -> BTreeMap<String, Option<(Memory, String)>> {
    map.iter()
        .map(|(k, v)| {
            (
                k.clone(),
                v.clone().map(|m| {
                    let body = m.body.clone();
                    (m, body)
                }),
            )
        })
        .collect()
}
fn restore_bodies(
    map: BTreeMap<String, Option<(Memory, String)>>,
) -> BTreeMap<String, Option<Memory>> {
    map.into_iter()
        .map(|(k, v)| {
            (
                k,
                v.map(|(mut m, body)| {
                    m.body = body;
                    m
                }),
            )
        })
        .collect()
}

fn ensure_column(db: &Connection, table: &str, column: &str, definition: &str) -> Result<()> {
    let mut query = db.prepare(&format!("PRAGMA table_info({table})"))?;
    let names = query
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    if !names.iter().any(|name| name == column) {
        db.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {definition}"))?;
    }
    Ok(())
}
fn text_hash(text: &str) -> String {
    hex::encode(Sha256::digest(text.as_bytes()))
}
fn vector_to_bytes(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bytes_to_vector(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn random_token() -> String {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    hex::encode(bytes)
}
pub fn credential(reference: &str) -> Result<String> {
    if let Some(name) = reference.strip_prefix("env:") {
        return std::env::var(name)
            .with_context(|| format!("missing credential env variable {name}"));
    }
    if let Some(service) = reference.strip_prefix("keychain:") {
        #[cfg(target_os = "macos")]
        {
            let output = std::process::Command::new("security")
                .args([
                    "find-generic-password",
                    "-a",
                    "default",
                    "-s",
                    service,
                    "-w",
                ])
                .output()?;
            if !output.status.success() {
                bail!("keychain credential unavailable for service {service}");
            }
            return Ok(String::from_utf8(output.stdout)?.trim().into());
        }
        #[cfg(not(target_os = "macos"))]
        bail!("keychain credentials are only supported on macOS; use env:NAME");
    }
    bail!("credential reference must be env:NAME or keychain:SERVICE")
}
pub fn redact_value(value: &Value) -> Value {
    match value {
        Value::String(text) => Value::String(redact(text)),
        Value::Array(items) => Value::Array(items.iter().map(redact_value).collect()),
        Value::Object(items) => Value::Object(
            items
                .iter()
                .map(|(key, value)| (key.clone(), redact_value(value)))
                .collect(),
        ),
        _ => value.clone(),
    }
}
pub fn redact(text: &str) -> String {
    let mut result = text.to_owned();
    static PATTERNS: LazyLock<Vec<regex::Regex>> = LazyLock::new(|| {
        [
        r"(?s)-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
        r"\beyJ[A-Za-z0-9_-]{12,}\.eyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{8,}\b",
        r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^\s/@:]+:[^\s/@]+@[^\s]+",
        r"(?i)(api[_-]?key|password|secret|token)\s*[:=]\s*[^\s,;]+",
        r#"(?i)"(?:api[_-]?key|password|secret|token)"\s*:\s*"[^"]+""#,
        r"sk-[A-Za-z0-9_-]{20,}",
        r"(?i)bearer\s+[A-Za-z0-9._-]{16,}",
    ].into_iter().map(|pattern|regex::Regex::new(pattern).unwrap()).collect()
    });
    for pattern in PATTERNS.iter() {
        result = pattern.replace_all(&result, "[REDACTED]").into_owned();
    }
    static HIGH_ENTROPY: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"\b[A-Za-z0-9_-]{32,}\b").unwrap());
    result = HIGH_ENTROPY
        .replace_all(&result, |captures: &regex::Captures<'_>| {
            let candidate = captures.get(0).unwrap().as_str();
            let mut counts = HashMap::new();
            for byte in candidate.bytes() {
                *counts.entry(byte).or_insert(0usize) += 1;
            }
            let length = candidate.len() as f64;
            let entropy: f64 = counts
                .values()
                .map(|count| {
                    let p = *count as f64 / length;
                    -p * p.log2()
                })
                .sum();
            if entropy >= 3.5 {
                "[REDACTED:high_entropy]".to_owned()
            } else {
                candidate.to_owned()
            }
        })
        .into_owned();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_store() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        init(dir.path()).unwrap();
        let mut config = load_config(dir.path()).unwrap();
        config.embedding.enabled = false;
        Store::save_config(dir.path(), &config).unwrap();
        let store = Store::open(dir.path()).unwrap();
        (dir, store)
    }
    fn fact(content: &str) -> NewMemory {
        NewMemory {
            content: content.into(),
            kind: "fact".into(),
            scope: "global".into(),
            title: None,
            triggers: vec![],
        }
    }

    #[test]
    fn parses_streamed_actual_model_and_usage() {
        let observed = UsageObservation::from_bytes(b"data: {\"model\":\"actual\"}\n\ndata: {\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":10,\"prompt_tokens_details\":{\"cached_tokens\":25}}}\n\n");
        assert_eq!(observed.actual_model.as_deref(), Some("actual"));
        assert_eq!(observed.input_tokens, Some(100));
        assert_eq!(observed.cached_input_tokens, Some(25));
    }
    #[test]
    fn redacts_common_secret_formats() {
        for value in [
            "password=hunter2",
            "AKIA1234567890ABCDEF",
            "eyJabcdefghijklmnop.eyJabcdefghijklmnop.abcdefghij",
            "postgres://user:password@host/db",
            "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----",
            "{\"api_key\":\"very-secret\"}",
            "test-9f86d081884c7d659a2feaa0c55ad015",
        ] {
            assert!(!redact(value).contains("hunter2"));
            assert!(redact(value).contains("[REDACTED"), "{value}");
        }
    }
    #[test]
    fn encrypted_events_and_incremental_index() {
        let (dir, store) = test_store();
        let id = store
            .event(
                "test",
                None,
                Some("s1"),
                "request",
                &json!({}),
                &json!({"secret":"hello"}),
            )
            .unwrap();
        assert_eq!(
            store.event_payload(&id).unwrap().unwrap()["secret"],
            "hello"
        );
        assert_eq!(store.session_events("s1", None).unwrap().len(), 1);
        let db_bytes = fs::read(dir.path().join("state/events.db")).unwrap();
        assert!(!String::from_utf8_lossy(&db_bytes).contains("hello"));
        let memory = store
            .remember(fact("staging database is read-only"), "user")
            .unwrap();
        let hits = store.with_index(|index, _| index.keyword_scores("staging database"));
        assert_eq!(hits[0].0, memory.id);
        store
            .update_memory(&memory.id, "the cache lives in redis")
            .unwrap();
        assert!(
            store
                .with_index(|index, _| index.keyword_scores("staging database"))
                .is_empty()
        );
        let log = std::process::Command::new("git")
            .arg("-C")
            .arg(dir.path().join("memory"))
            .args(["log", "--oneline"])
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&log.stdout).lines().count(), 2);
    }
    #[test]
    fn migrates_v1_memories_on_open() {
        let (dir, store) = test_store();
        drop(store);
        let id = "mem_01M3H6EY7F3P3KDHQJPPMJHP51";
        fs::write(
            dir.path().join(format!("memory/{id}.md")),
            format!("---\nid: {id}\ntype: lesson\ntitle: t\ncard: c\nscope: global\ntier: recallable\nstatus: active\nconfidence: 0.85\nsource: user_statement\ntriggers: []\nbody: ''\ncreated_at: a\nupdated_at: b\ncreated_by: user\n---\nrun migrations first\n"),
        )
        .unwrap();
        let store = Store::open(dir.path()).unwrap();
        assert_eq!(store.memory(id).unwrap().body, "run migrations first");
        let text = fs::read_to_string(dir.path().join(format!("memory/{id}.md"))).unwrap();
        assert!(!text.contains("card:"));
    }
    #[test]
    fn batch_rollback_restores_memory() {
        let (_dir, store) = test_store();
        let original = store.remember(fact("old fact"), "user").unwrap();
        let mut changed = original.clone();
        changed.body = "new fact".into();
        let batch = store.apply_batch("test", &[changed]).unwrap();
        assert_eq!(store.memory(&original.id).unwrap().body, "new fact");
        store.rollback_batch(&batch).unwrap();
        assert_eq!(store.memory(&original.id).unwrap().body, "old fact");
        let mut again = original.clone();
        again.body = "newer".into();
        let batch = store.apply_batch("test", &[again]).unwrap();
        store
            .update_memory(&original.id, "later user edit")
            .unwrap();
        assert!(store.rollback_batch(&batch).is_err());
    }
    #[test]
    fn gate_labels_exclude_fallbacks_and_use_source_features() {
        let (_dir, store) = test_store();
        let traces = [
            json!({"called":true,"decision":"fallback","reason":"timeout"}),
            json!({"called":false,"decision":"fallback","reason":"deferred_pending"}),
            json!({"called":true,"decision":true,"mode":"sync"}),
            json!({"called":true,"decision":false,"mode":"deferred","source_features":{"text":"source"}}),
        ];
        for trace in traces {
            let step = store
                .event(
                    "a",
                    None,
                    None,
                    "request",
                    &json!({"text":"applied"}),
                    &json!({}),
                )
                .unwrap();
            store.record_step(&step, &json!({"gate":trace})).unwrap();
            store
                .record_recall(&step, "mem_x", "trigger", 0.9, "t")
                .unwrap();
            store
                .record_feedback(&step, "mem_x", Some(true), None, Some("success"))
                .unwrap();
        }
        let samples = store.gate_training_samples().unwrap();
        assert_eq!(samples.len(), 2);
        assert!(samples[0].gate_recalled);
        assert_eq!(samples[1].features["text"], "source");
    }
    #[test]
    fn sessions_become_due_after_idle() {
        let (_dir, store) = test_store();
        store
            .touch_session("s1", "agent", Some("p"), true, (2, Some("h")))
            .unwrap();
        assert_eq!(store.session_cursor("s1").unwrap().unwrap().0, 2);
        assert!(
            store
                .sessions_due(Duration::from_secs(3600))
                .unwrap()
                .is_empty()
        );
        assert_eq!(store.sessions_due(Duration::ZERO).unwrap().len(), 1);
        store.finish_session("s1", "done", None).unwrap();
        assert!(store.sessions_due(Duration::ZERO).unwrap().is_empty());
        store
            .touch_session("s1", "agent", Some("p"), true, (3, None))
            .unwrap();
        assert_eq!(store.session("s1").unwrap().unwrap().status, "open");
    }
}
