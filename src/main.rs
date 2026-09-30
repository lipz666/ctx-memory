mod adapters;
mod api;
mod config;
mod consolidation;
mod contradict;
mod embed;
mod episode;
mod experimental;
mod extract;
mod feedback;
mod index;
mod inject;
mod llm;
mod mcp;
mod planner;
mod memory;
mod observer;
mod proxy;
mod recall;
mod reflect;
mod server;
mod session;
mod store;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use config::{Agent, ModelConfig};
use memory::{NewMemory, NewTrigger};
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufRead, BufReader},
    path::PathBuf,
};
use store::{Store, root_path};

/// Package version and the git commit the binary was built from.
pub const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (", env!("CTX_GIT_HASH"), ")");

#[derive(Parser)]
#[command(
    name = "ctx",
    version = VERSION,
    about = "Local long-term memory and context engine for agents"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Init,
    Serve,
    Mcp,
    Connect {
        agent: String,
        upstream: Option<String>,
        #[arg(long)]
        credential_ref: Option<String>,
        #[arg(long)]
        upstream_user_agent: Option<String>,
        /// Record usage without changing requests (A/B direct arm).
        #[arg(long)]
        meter_only: bool,
    },
    Disconnect {
        agent: String,
    },
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    Doctor {
        #[arg(long)]
        llm: bool,
    },
    Status,
    Open,
    /// Save a memory as the user.
    Remember {
        content: String,
        #[arg(long = "type", visible_alias = "kind", default_value = "fact")]
        kind: String,
        #[arg(long, default_value = "global")]
        scope: String,
        #[arg(long)]
        title: Option<String>,
        /// KIND:PATTERN with KIND keyword, error, tool or file (repeatable).
        #[arg(long)]
        trigger: Vec<String>,
        /// Shorthand for --trigger keyword:TEXT.
        #[arg(long)]
        trigger_text: Option<String>,
        /// Always inject (like a rule) while in scope.
        #[arg(long)]
        pinned: bool,
    },
    Recall {
        query: String,
        /// Project scope; defaults to the repository of the current directory.
        #[arg(long)]
        project: Option<String>,
        /// Global memories only.
        #[arg(long)]
        global: bool,
        #[arg(long, default_value_t = 5)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    Memories {
        #[arg(long)]
        json: bool,
    },
    Edit {
        id: String,
        content: String,
    },
    Forget {
        id: String,
    },
    Expand {
        id: String,
    },
    Review {
        id: String,
        /// approve or reject
        decision: String,
    },
    /// Embed memories that lack a current vector (downloads the model on first use) and
    /// build conversation excerpts for recorded sessions that have none.
    Reindex,
    Sessions,
    /// Extract memories from due sessions, or from one session now.
    Extract {
        #[arg(long)]
        session: Option<String>,
    },
    Replay {
        trace: PathBuf,
    },
    Eval {
        #[command(subcommand)]
        command: EvalCommand,
    },
    Automation {
        #[command(subcommand)]
        command: AutomationCommand,
    },
    Maintenance {
        #[command(subcommand)]
        command: MaintenanceCommand,
    },
}
#[derive(Subcommand)]
enum MaintenanceCommand {
    Run,
    Status,
    Rollback { batch_id: String },
}
#[derive(Subcommand)]
enum EvalCommand {
    /// Labeled replay: {"query","project","features","expected_memory_ids"} per line.
    Replay {
        trace: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Rank memories for {"id","query","project"} lines; prints {"id","ids"} lines.
    Recall { queries: PathBuf },
}
#[derive(Subcommand)]
enum AutomationCommand {
    Status,
    /// extraction, embedding, gate, maintenance or eviction
    Enable {
        component: String,
    },
    Disable {
        component: String,
    },
    /// Extract due sessions now.
    Run,
    Review {
        id: String,
        decision: String,
    },
    TrainGate,
    Guard {
        mode: String,
    },
    Budget {
        daily_llm_calls: u32,
    },
}
#[derive(Subcommand)]
enum ModelCommand {
    Set {
        base_url: String,
        model: String,
        #[arg(long)]
        credential_ref: String,
        #[arg(long)]
        upstream_user_agent: Option<String>,
    },
}

fn valid_base_url(url: &str) -> Result<()> {
    let url = reqwest::Url::parse(url)?;
    if !["http", "https"].contains(&url.scheme())
        || url.username() != ""
        || url.password().is_some()
        || url.query().is_some()
    {
        bail!("base URL must be HTTP(S) without credentials or query");
    }
    Ok(())
}

fn print_json(value: &Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

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

fn project_from_cwd() -> Option<String> {
    std::env::current_dir()
        .ok()
        .and_then(|dir| session::project_name(&dir))
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let root = root_path();
    match cli.command {
        Command::Init => {
            store::init(&root)?;
            adapters::ensure_service(&root)?;
            println!("initialized {}", root.display());
        }
        Command::Serve => server::serve(Store::open(&root)?).await?,
        Command::Mcp => mcp::serve(&Store::open(&root)?)?,
        Command::Connect {
            agent,
            upstream,
            credential_ref,
            upstream_user_agent,
            meter_only,
        } => {
            let mut config = store::load_config(&root)?;
            let upstream = upstream
                .or_else(|| config.model.as_ref().map(|m| m.base_url.clone()))
                .ok_or_else(|| anyhow::anyhow!("provide upstream or configure ctx model set"))?;
            valid_base_url(&upstream)?;
            if !agent
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            {
                bail!("invalid agent id");
            }
            let same_gateway = config
                .model
                .as_ref()
                .filter(|m| m.base_url == upstream)
                .cloned();
            config.agents.insert(
                agent.clone(),
                Agent {
                    upstream,
                    credential_ref: credential_ref
                        .or_else(|| same_gateway.as_ref().map(|m| m.credential_ref.clone())),
                    upstream_user_agent: upstream_user_agent
                        .or_else(|| same_gateway.and_then(|m| m.upstream_user_agent)),
                    meter_only,
                },
            );
            Store::save_config(&root, &config)?;
            let wrapper = adapters::connect(&agent, &root, &config)?;
            adapters::ensure_service(&root)?;
            println!(
                "agent {agent}: http://127.0.0.1:{}/a/{agent}/v1",
                config.port
            );
            println!(
                "Set X-Ctx-Token to the contents of {}/token",
                root.display()
            );
            if let Some(path) = wrapper {
                println!("adapter: {} (open a new shell to use it)", path.display());
            }
        }
        Command::Disconnect { agent } => {
            let mut config = store::load_config(&root)?;
            config.agents.remove(&agent);
            Store::save_config(&root, &config)?;
            adapters::disconnect(&agent, &root)?;
            adapters::ensure_service(&root)?;
            println!("disconnected {agent}");
        }
        Command::Model {
            command:
                ModelCommand::Set {
                    base_url,
                    model,
                    credential_ref,
                    upstream_user_agent,
                },
        } => {
            valid_base_url(&base_url)?;
            let _ = store::credential(&credential_ref)?;
            let mut config = store::load_config(&root)?;
            config.model = Some(ModelConfig {
                base_url,
                model,
                credential_ref,
                upstream_user_agent,
            });
            Store::save_config(&root, &config)?;
            adapters::ensure_service(&root)?;
            println!("model configured; credential remains in external secret store");
        }
        Command::Doctor { llm } => {
            let store = Store::open(&root)?;
            println!("storage: ok; memories: {}", store.memories().len());
            let embedded = store.ensure_vectors()?;
            match store.embedder.get() {
                Some(model) => {
                    println!("embeddings: {} (embedded {embedded} memories)", model.name)
                }
                None => println!("embeddings: unavailable; keyword recall only"),
            }
            if llm {
                let started = std::time::Instant::now();
                let reply = llm::chat(
                    &store,
                    "doctor",
                    "Reply with OK only.",
                    "ping",
                    16,
                    std::time::Duration::from_secs(60),
                )
                .await?;
                println!(
                    "model API: ok in {:.1}s; reply: {}",
                    started.elapsed().as_secs_f64(),
                    reply.trim()
                );
            }
        }
        Command::Status => print_json(&status_json(&Store::open(&root)?)?)?,
        Command::Open => {
            let store = Store::open(&root)?;
            let endpoint = format!("http://127.0.0.1:{}/api/v1/ui-ticket", store.config.port);
            let response = reqwest::Client::new()
                .post(endpoint)
                .header("X-Ctx-Token", &store.token)
                .send()
                .await?;
            if !response.status().is_success() {
                bail!("ctx daemon is not available");
            }
            let payload: Value = response.json().await?;
            let url = payload
                .get("url")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("invalid UI ticket"))?;
            #[cfg(target_os = "macos")]
            let program = "open";
            #[cfg(not(target_os = "macos"))]
            let program = "xdg-open";
            if !std::process::Command::new(program)
                .arg(url)
                .status()?
                .success()
            {
                bail!("could not open browser");
            }
            println!("opened local dashboard");
        }
        Command::Remember {
            content,
            kind,
            scope,
            title,
            trigger,
            trigger_text,
            pinned,
        } => {
            let store = Store::open(&root)?;
            let mut triggers = trigger
                .iter()
                .map(|spec| {
                    let (kind, pattern) = spec
                        .split_once(':')
                        .ok_or_else(|| anyhow::anyhow!("trigger must be KIND:PATTERN"))?;
                    Ok(NewTrigger {
                        kind: kind.into(),
                        pattern: pattern.into(),
                        before_action: false,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if let Some(text) = trigger_text {
                triggers.push(NewTrigger {
                    kind: "keyword".into(),
                    pattern: text,
                    before_action: false,
                });
            }
            let mut memory = memory::create(
                NewMemory {
                    content,
                    kind,
                    scope,
                    title,
                    triggers,
                },
                "user",
            )?;
            memory.pinned = pinned;
            store.save_memory(&memory, &format!("add {}", memory.id))?;
            println!("{}", memory.id);
        }
        Command::Recall {
            query,
            project,
            global,
            limit,
            json: as_json,
        } => {
            let store = Store::open(&root)?;
            store.ensure_vectors()?;
            let project = if global {
                None
            } else {
                project
                    .map(|p| memory::normalize_scope(&p))
                    .or_else(project_from_cwd)
            };
            let hits = recall::recall(
                &store,
                &recall::Query {
                    text: Some(&query),
                    project: project.as_deref(),
                    limit,
                    mode: recall::Mode::Search,
                    episodes: store.config.recall.search_episodes,
                    ..Default::default()
                },
            )?;
            if as_json {
                print_json(&json!(hits.iter().map(|h| json!({"id":h.memory.id,"score":h.score,"channel":h.channel,"reason":h.reason,"title":h.memory.title})).collect::<Vec<_>>()))?;
            } else {
                for hit in hits {
                    println!(
                        "{} {:.2} [{}] {}",
                        hit.memory.id, hit.score, hit.channel, hit.memory.title
                    );
                }
            }
        }
        Command::Memories { json: as_json } => {
            let store = Store::open(&root)?;
            for m in store.memories() {
                if as_json {
                    let mut value = json!(m);
                    value["body"] = m.body.clone().into();
                    println!("{value}");
                } else {
                    println!("{} {} {} {} {}", m.id, m.kind, m.status, m.scope, m.title);
                }
            }
        }
        Command::Edit { id, content } => match Store::open(&root)?.update_memory(&id, &content)? {
            Some(memory) => println!("updated {}", memory.id),
            None => bail!("not found"),
        },
        Command::Forget { id } => println!("archived: {}", Store::open(&root)?.archive(&id)?),
        Command::Expand { id } => {
            let store = Store::open(&root)?;
            if id.starts_with("mem_") {
                let memory = store
                    .memory(&id)
                    .ok_or_else(|| anyhow::anyhow!("not found"))?;
                println!("{}", memory.body);
            } else if id.starts_with("evt_") {
                let payload = store
                    .event_payload(&id)?
                    .ok_or_else(|| anyhow::anyhow!("not found"))?;
                print_json(&payload)?;
            } else {
                bail!("invalid id");
            }
        }
        Command::Review { id, decision }
        | Command::Automation {
            command: AutomationCommand::Review { id, decision },
        } => {
            let approve = match decision.as_str() {
                "approve" => true,
                "reject" => false,
                _ => bail!("decision must be approve or reject"),
            };
            println!(
                "reviewed: {}",
                Store::open(&root)?.review_memory(&id, approve)?
            );
        }
        Command::Reindex => {
            let store = Store::open(&root)?;
            let count = store.ensure_vectors()?;
            let Some(model) = store.embedder.get() else {
                bail!("embedding model unavailable");
            };
            println!("embedded {count} memories with {}", model.name);
            let excerpts = extract::backfill_episodes(&store)?;
            println!("added {excerpts} conversation excerpts");
            let digests = extract::rebuild_digests(&store)?;
            println!("rewrote {digests} topic digests");
        }
        Command::Sessions => print_json(&json!(Store::open(&root)?.sessions(50)?))?,
        Command::Extract { session } => {
            let store = Store::open(&root)?;
            store.ensure_vectors()?;
            match session {
                Some(key) => {
                    let row = store
                        .session(&key)?
                        .ok_or_else(|| anyhow::anyhow!("session not found"))?;
                    print_json(&extract::extract_session(&store, &row).await?)?;
                }
                None => print_json(&extract::run_due(&store).await?)?,
            }
        }
        Command::Replay { trace } => replay(&root, trace, None)?,
        Command::Eval { command } => match command {
            EvalCommand::Replay { trace, output } => replay(&root, trace, output)?,
            EvalCommand::Recall { queries } => eval_recall(&root, queries)?,
        },
        Command::Automation { command } => automation(&root, command).await?,
        Command::Maintenance { command } => {
            let store = Store::open(&root)?;
            match command {
                MaintenanceCommand::Run => println!("{}", consolidation::run(&store)?),
                MaintenanceCommand::Status => println!("{}", store.maintenance_batches()?),
                MaintenanceCommand::Rollback { batch_id } => {
                    store.rollback_batch(&batch_id)?;
                    println!("rolled back {batch_id}");
                }
            }
        }
    }
    Ok(())
}

async fn automation(root: &std::path::Path, command: AutomationCommand) -> Result<()> {
    let mut config = store::load_config(root)?;
    match command {
        AutomationCommand::Status => return print_json(&status_json(&Store::open(root)?)?),
        AutomationCommand::Run => {
            let store = Store::open(root)?;
            store.ensure_vectors()?;
            return print_json(&extract::run_due(&store).await?);
        }
        AutomationCommand::TrainGate => {
            println!("{}", experimental::classifier::train(&Store::open(root)?)?);
            return Ok(());
        }
        AutomationCommand::Review { .. } => unreachable!("handled in main"),
        AutomationCommand::Enable { component } => set_component(&mut config, &component, true)?,
        AutomationCommand::Disable { component } => set_component(&mut config, &component, false)?,
        AutomationCommand::Guard { mode } => {
            if !["off", "advisory", "recheck"].contains(&mode.as_str()) {
                bail!("mode must be off, advisory, or recheck");
            }
            config.experimental.action_guard = mode;
        }
        AutomationCommand::Budget { daily_llm_calls } => {
            if daily_llm_calls > 1000 {
                bail!("daily call budget must be at most 1000")
            }
            config.extraction.daily_llm_calls = daily_llm_calls;
        }
    }
    Store::save_config(root, &config)?;
    adapters::ensure_service(root)?;
    println!("saved");
    Ok(())
}

fn set_component(config: &mut config::Config, component: &str, enabled: bool) -> Result<()> {
    match component {
        "extraction" | "encoder" => config.extraction.enabled = enabled,
        "embedding" => config.embedding.enabled = enabled,
        "gate" => config.experimental.gate_enabled = enabled,
        "maintenance" => config.experimental.maintenance_enabled = enabled,
        "eviction" => config.experimental.eviction_enabled = enabled,
        _ => bail!("component must be extraction, embedding, gate, maintenance or eviction"),
    }
    Ok(())
}

fn eval_recall(root: &std::path::Path, queries: PathBuf) -> Result<()> {
    let store = Store::open(root)?;
    store.ensure_vectors()?;
    for line in BufReader::new(fs::File::open(queries)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)?;
        let hits = recall::recall(
            &store,
            &recall::Query {
                text: row.get("query").and_then(Value::as_str),
                project: row.get("project").and_then(Value::as_str),
                limit: 5,
                ..Default::default()
            },
        )?;
        println!(
            "{}",
            json!({"id":row.get("id"),"ids":hits.iter().map(|h| &h.memory.id).collect::<Vec<_>>(),"reasons":hits.iter().map(|h| &h.reason).collect::<Vec<_>>()})
        );
    }
    Ok(())
}

fn replay(root: &std::path::Path, trace: PathBuf, output: Option<PathBuf>) -> Result<()> {
    let store = Store::open(root)?;
    store.ensure_vectors()?;
    let (
        mut total,
        mut needed,
        mut hit,
        mut injected,
        mut true_positive,
        mut expected_total,
        mut abstained,
    ) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut latencies = vec![];
    for line in BufReader::new(fs::File::open(trace)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)?;
        let query = row.get("query").and_then(Value::as_str);
        let features = row
            .get("features")
            .cloned()
            .unwrap_or_else(|| json!({"user_text":query}));
        let started = std::time::Instant::now();
        let hits = recall::recall(
            &store,
            &recall::Query {
                text: query,
                project: row.get("project").and_then(Value::as_str),
                features: Some(&features),
                limit: store.config.recall.max_injected,
                always_on: true,
                before_action: row.get("moment").and_then(Value::as_str) == Some("pre_action"),
                ..Default::default()
            },
        )?;
        latencies.push(started.elapsed().as_secs_f64() * 1000.0);
        total += 1;
        injected += hits.len();
        abstained += usize::from(hits.is_empty());
        if let Some(expected) = row.get("expected_memory_ids").and_then(Value::as_array) {
            let expected: Vec<&str> = expected.iter().filter_map(Value::as_str).collect();
            expected_total += expected.len();
            needed += usize::from(!expected.is_empty());
            let matched = hits
                .iter()
                .filter(|h| expected.contains(&h.memory.id.as_str()))
                .count();
            true_positive += matched;
            hit += usize::from(matched > 0);
        }
        println!(
            "{}",
            json!({"step":total,"memory_ids":hits.iter().map(|h| &h.memory.id).collect::<Vec<_>>()})
        );
    }
    latencies.sort_by(f64::total_cmp);
    let p95 = latencies
        .get((latencies.len() * 95 / 100).min(latencies.len().saturating_sub(1)))
        .copied();
    let ratio = |a: usize, b: usize| {
        if b == 0 {
            Value::Null
        } else {
            json!(a as f64 / b as f64)
        }
    };
    let report = json!({"steps":total,"needed_steps":needed,"hit_steps":hit,"hit_rate":ratio(hit,needed),
        "memory_recall":ratio(true_positive,expected_total),"injection_precision":ratio(true_positive,injected),
        "abstain_rate":ratio(abstained,total),"retrieval_p95_ms":p95});
    if let Some(path) = output {
        fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    eprintln!("{report}");
    Ok(())
}
