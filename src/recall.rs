//! The one recall path used by the proxy, MCP, CLI, API and evals.
//!
//! Always-on memories (rules, pinned) come first; then trigger matches on the step's
//! features; then hybrid search: embedding similarity plus keyword coverage.
//!
//! Two operating points: `Mode::Inject` (automatic injection into the agent's context)
//! keeps only confident hits; `Mode::Search` (an explicit question from an agent or a
//! person) returns up to `limit` hits ranked by relevance with a low floor.
use crate::{
    config::RecallConfig,
    index::Index,
    memory::{Memory, trigger_matches},
    store::Store,
};
use anyhow::Result;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct Hit {
    pub memory: Memory,
    /// rule, trigger, search, gate.
    pub channel: &'static str,
    pub score: f64,
    pub reason: String,
    /// Search mode: hits about the same topic share a group; groups come in order of
    /// relevance and hits within a group in chronological order.
    pub group: Option<usize>,
}

/// Cosine similarity at which two search hits are treated as the same topic (distinct
/// events of one kind, such as two workshops, measure 0.76 to 0.79 with EmbeddingGemma).
const TOPIC_SIMILARITY: f32 = 0.75;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    #[default]
    Inject,
    Search,
}

#[derive(Clone, Default)]
pub struct Query<'a> {
    pub mode: Mode,
    /// Text to search for; `None` skips search (triggers and always-on only).
    pub text: Option<&'a str>,
    pub project: Option<&'a str>,
    /// Step features for triggers.
    pub features: Option<&'a Value>,
    /// Maximum hits besides always-on memories.
    pub limit: usize,
    pub always_on: bool,
    /// Match only `before_action` triggers (experimental ActionGuard).
    pub before_action: bool,
    pub exclude: Option<&'a HashSet<String>>,
    /// Memory ids proposed by another component (experimental Gate).
    pub extra_ids: &'a [String],
    /// Search mode only: raw conversation excerpts ranked together with the memories.
    pub episodes: usize,
    /// Search mode only: pack the result into this many tokens (see `pack`).
    pub budget: Option<usize>,
}

/// What a question asks for, from its wording (no model call).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Intent {
    /// "how many", "in total", "which ... have I": whole-topic overviews help.
    pub aggregate: bool,
    /// "most recently", "first", "how long ago": dated events in order help.
    pub temporal: bool,
}

const AGGREGATE_CUES: &[&str] = &[
    "how many", "how much", "in total", "total", "number of", "count", "list", "all the", "all of the",
    "which ones", "多少", "几次", "几个", "几种", "总共", "一共", "总计", "哪些", "所有",
];
const TEMPORAL_CUES: &[&str] = &[
    "first", "firstly", "most recent", "most recently", "recently", "latest", "last time", "earliest", "before", "after", "how long", "ago",
    "order", "when did", "what date", "which day", "最近", "第一次", "最早", "之前", "之后", "多久", "哪天", "顺序",
];

pub fn intent(text: &str) -> Intent {
    let text = text.to_lowercase();
    let has = |cues: &[&str]| {
        cues.iter().any(|cue| {
            if cue.chars().any(|c| c as u32 >= 0x2E80) {
                return text.contains(cue);
            }
            text.match_indices(cue).any(|(start, _)| {
                let before = text[..start].chars().next_back();
                let after = text[start + cue.len()..].chars().next();
                before.is_none_or(|c| !c.is_alphanumeric()) && after.is_none_or(|c| !c.is_alphanumeric())
            })
        })
    };
    Intent { aggregate: has(AGGREGATE_CUES), temporal: has(TEMPORAL_CUES) }
}

/// Score added to topic digests for aggregate questions.
const AGGREGATE_DIGEST_BOOST: f64 = 0.15;
/// Share of a packing budget kept for conversation excerpts once memories are placed.
const EXCERPT_SHARE: f64 = 0.35;
/// Characters of an excerpt kept around its best-matching sentence when packing.
const EXCERPT_WINDOW: usize = 400;

/// Hybrid candidates of `index` among the documents `allow` accepts, best first:
/// (id, score, reason). Search mode uses the low floors and no relative cutoff; in
/// injection mode the cutoff is relative to the better of the best candidate and
/// `reference`.
#[allow(clippy::too_many_arguments)]
fn hybrid(
    index: &Index,
    config: &RecallConfig,
    search: bool,
    text: &str,
    vector: Option<&[f32]>,
    pool: usize,
    allow: impl Fn(&str) -> bool + Copy,
    reference: f32,
) -> Vec<(String, f32, String)> {
    let keyword: HashMap<String, f32> = index
        .keyword_scores_where(text, allow)
        .into_iter()
        .take(pool)
        .collect();
    let semantic: HashMap<String, f32> = vector
        .map(|v| {
            index
                .semantic_scores_where(v, allow)
                .into_iter()
                .take(pool)
                .collect()
        })
        .unwrap_or_default();
    let with_vectors = vector.is_some();
    let weight = config.semantic_weight.clamp(0.0, 1.0);
    let (min_similarity, relative_cutoff) = if search {
        (config.search_min_similarity, 0.0)
    } else {
        (config.min_similarity, config.relative_cutoff)
    };
    let min_keyword = if with_vectors && !search {
        config.min_keyword
    } else {
        config.min_keyword_fallback
    };
    let mut scored: Vec<(String, f32, String)> = keyword
        .keys()
        .chain(semantic.keys())
        .collect::<HashSet<_>>()
        .into_iter()
        .filter_map(|id| {
            let k = keyword.get(id).copied().unwrap_or(0.0);
            let s = semantic.get(id).copied();
            let accepted = s.is_some_and(|s| s >= min_similarity) || k >= min_keyword;
            let score = match s {
                Some(s) if with_vectors => weight * s + (1.0 - weight) * k,
                _ if with_vectors => (1.0 - weight) * k,
                _ => k,
            };
            accepted.then(|| {
                (
                    id.clone(),
                    score,
                    format!("semantic:{:.2} keyword:{:.2}", s.unwrap_or(0.0), k),
                )
            })
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let best = scored.first().map(|s| s.1).unwrap_or(0.0).max(reference);
    scored.retain(|s| s.1 >= best * relative_cutoff);
    scored
}

fn episode_hits(
    store: &Store,
    text: &str,
    vector: Option<&[f32]>,
    project: Option<&str>,
    limit: usize,
) -> Result<Vec<Hit>> {
    let scored = store.with_episodes(|episodes| {
        let mut scored = hybrid(
            &episodes.index,
            &store.config.recall,
            true,
            text,
            vector,
            limit * 4,
            |id| episodes.in_scope(id, project),
            0.0,
        );
        // Spare candidates: excerpts of the same turn are dropped below.
        scored.truncate(limit * 3);
        scored
            .into_iter()
            .map(|(id, score, reason)| {
                let meta = episodes.meta.get(&id);
                let observed = meta.and_then(|m| m.observed_at.clone());
                let session = meta.map(|m| m.session.clone()).unwrap_or_default();
                (id, score, reason, observed, session)
            })
            .collect::<Vec<_>>()
    });
    let mut hits = vec![];
    let mut turns = HashSet::new();
    for (id, score, reason, observed_at, session) in scored {
        if hits.len() == limit {
            break;
        }
        if let Some(text) = store.episode_text(&id)? {
            // A long turn is split into several excerpts that all start with the user's
            // message; returning more than one repeats the same statement.
            if !turns.insert((session, crate::episode::turn_key(&text))) {
                continue;
            }
            hits.push(Hit {
                memory: Memory::episode(&id, project, text, observed_at),
                channel: "episode",
                score: score as f64,
                reason,
                group: None,
            });
        }
    }
    Ok(hits)
}

pub fn recall(store: &Store, query: &Query) -> Result<Vec<Hit>> {
    let mut hits = recall_ranked(store, query)?;
    if query.mode == Mode::Search {
        group_by_topic(store, &mut hits);
        if let (Some(budget), Some(text)) = (query.budget, query.text) {
            hits = pack(hits, text, budget);
        }
    }
    Ok(hits)
}

/// Estimated tokens of a hit as a reader sees it (4 characters per token, plus framing).
fn hit_tokens(hit: &Hit) -> usize {
    hit.memory.body.chars().count() / 4 + 12
}

/// Fit search results into `budget` tokens for a reader: memories (facts, preferences,
/// digests) first, conversation excerpts after with a reserved share, each excerpt cut to
/// the user's message and the window most relevant to the question. Items that do not fit
/// are skipped rather than ending the list. For temporal questions the memories are listed
/// in date order.
pub fn pack(hits: Vec<Hit>, question: &str, budget: usize) -> Vec<Hit> {
    let intent = intent(question);
    let (mut memories, excerpts): (Vec<Hit>, Vec<Hit>) = hits.into_iter().partition(|h| h.channel != "episode");
    let excerpts: Vec<Hit> = excerpts
        .into_iter()
        .map(|mut h| {
            h.memory.body = trim_excerpt(&h.memory.body, question);
            h
        })
        .collect();
    let memory_budget = if excerpts.is_empty() { budget } else { (budget as f64 * (1.0 - EXCERPT_SHARE)) as usize };
    let mut used = 0;
    let mut kept_memories = vec![];
    let mut left = vec![];
    for hit in memories.drain(..) {
        let cost = hit_tokens(&hit);
        if hit.channel == "rule" || used + cost <= memory_budget {
            used += cost;
            kept_memories.push(hit);
        } else {
            left.push(hit);
        }
    }
    let mut kept_excerpts = vec![];
    for hit in excerpts {
        let cost = hit_tokens(&hit);
        if used + cost <= budget {
            used += cost;
            kept_excerpts.push(hit);
        }
    }
    // Room left after the excerpts goes back to memories.
    for hit in left {
        let cost = hit_tokens(&hit);
        if used + cost <= budget {
            used += cost;
            kept_memories.push(hit);
        }
    }
    if intent.temporal {
        let rules = kept_memories.iter().take_while(|h| h.channel == "rule").count();
        kept_memories[rules..].sort_by_key(|h| date_key(&h.memory));
    }
    kept_memories.extend(kept_excerpts);
    kept_memories
}

/// The first line of an excerpt (the user's message, shortened) and the window of about
/// `EXCERPT_WINDOW` characters around the sentence sharing most words with the question.
fn trim_excerpt(text: &str, question: &str) -> String {
    if text.chars().count() <= EXCERPT_WINDOW + 200 {
        return text.to_owned();
    }
    let (first, rest) = text.split_once('\n').unwrap_or((text, ""));
    let first: String = first.chars().take(200).collect();
    let terms: HashSet<String> = crate::index::tokens(question).into_iter().collect();
    let sentences: Vec<&str> = rest
        .split_inclusive(['.', '!', '?', '\n', '。', '！', '？'])
        .filter(|s| !s.trim().is_empty())
        .collect();
    let overlap = |s: &str| crate::index::tokens(s).iter().filter(|t| terms.contains(*t)).count();
    let Some(best) = (0..sentences.len()).max_by_key(|&i| (overlap(sentences[i]), std::cmp::Reverse(i))) else {
        return first;
    };
    let (mut start, mut end) = (best, best + 1);
    let len = |a: usize, b: usize| sentences[a..b].iter().map(|s| s.chars().count()).sum::<usize>();
    while len(start, end) < EXCERPT_WINDOW && (start > 0 || end < sentences.len()) {
        if end < sentences.len() {
            end += 1;
        }
        if len(start, end) < EXCERPT_WINDOW && start > 0 {
            start -= 1;
        }
    }
    let window: String = sentences[start..end].concat();
    format!(
        "{first}\n{}{}{}",
        if start > 0 { "…" } else { "" },
        window.trim(),
        if end < sentences.len() { "…" } else { "" }
    )
}

/// A search plan for a hard question (made by `planner`): sub-queries that each cover
/// one part of the question, and the date window its events fall in, if any.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plan {
    pub queries: Vec<String>,
    pub after: Option<String>,
    pub before: Option<String>,
}

/// Score added to a hit dated inside the plan's window.
const WINDOW_BOOST: f64 = 0.15;

/// Search with a plan: the question and every sub-query are searched over a wide pool,
/// hits dated inside the window are boosted, then the best hit of each query is kept
/// first (every part of the question is covered) and the rest filled by score. The
/// result is cut to the query's limit plus excerpts and grouped by topic.
pub fn recall_planned(store: &Store, query: &Query, plan: &Plan) -> Result<Vec<Hit>> {
    let window = (
        plan.after.as_deref().and_then(date_range).map(|r| r.0),
        plan.before.as_deref().and_then(date_range).map(|r| r.1),
    );
    let in_window = |memory: &Memory| {
        (window.0.is_some() || window.1.is_some())
            && date_range(&date_key(memory)).is_some_and(|(start, end)| {
                window.0.is_none_or(|after| end >= after) && window.1.is_none_or(|before| start <= before)
            })
    };
    let texts: Vec<String> = query.text.into_iter().map(str::to_owned).chain(plan.queries.iter().cloned()).collect();
    let mut lists: Vec<Vec<Hit>> = vec![];
    for text in &texts {
        // A wide pool per query: the window boost may promote lower-ranked hits.
        let wide = Query { text: Some(text), limit: (query.limit * 3).max(20), ..query.clone() };
        let mut hits = recall_ranked(store, &wide)?;
        for hit in hits.iter_mut().filter(|h| h.channel != "rule" && in_window(&h.memory)) {
            hit.score += WINDOW_BOOST;
            hit.reason.push_str(" window");
        }
        hits.sort_by(|a, b| {
            (b.channel == "rule")
                .cmp(&(a.channel == "rule"))
                .then(b.score.total_cmp(&a.score))
                .then_with(|| a.memory.id.cmp(&b.memory.id))
        });
        lists.push(hits);
    }
    let rules = lists.first().map_or(0, |l| l.iter().filter(|h| h.channel == "rule").count());
    let cap = query.limit + query.episodes + rules;
    let mut seen = HashSet::new();
    let mut hits: Vec<Hit> = lists
        .first()
        .into_iter()
        .flatten()
        .filter(|h| h.channel == "rule")
        .inspect(|h| {
            seen.insert(h.memory.id.clone());
        })
        .cloned()
        .collect();
    for list in &lists {
        if let Some(best) = list.iter().find(|h| h.channel != "rule" && !seen.contains(&h.memory.id)) {
            seen.insert(best.memory.id.clone());
            hits.push(best.clone());
        }
    }
    let mut rest: Vec<Hit> = lists.into_iter().flatten().filter(|h| !seen.contains(&h.memory.id)).collect();
    rest.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.memory.id.cmp(&b.memory.id)));
    for hit in rest {
        if hits.len() >= cap {
            break;
        }
        if seen.insert(hit.memory.id.clone()) {
            hits.push(hit);
        }
    }
    hits.truncate(cap);
    group_by_topic(store, &mut hits);
    Ok(hits)
}

/// The day range a (possibly partial) date covers, as yyyymmdd numbers:
/// "2023" → 20230101..20231231, "2023-03" → 20230301..20230331, "2023/03/05 (Sun)" → one day.
pub fn date_range(date: &str) -> Option<(u32, u32)> {
    let digits: String = date.chars().filter(char::is_ascii_digit).collect();
    let number = |s: &str| s.parse::<u32>().ok();
    match digits.len() {
        4 => Some((number(&digits)? * 10000 + 101, number(&digits)? * 10000 + 1231)),
        6 => Some((number(&digits)? * 100 + 1, number(&digits)? * 100 + 31)),
        n if n >= 8 => number(&digits[..8]).map(|d| (d, d)),
        _ => None,
    }
}

/// Search without topic grouping (the ranking every mode shares).
fn recall_ranked(store: &Store, query: &Query) -> Result<Vec<Hit>> {
    let text = query
        .text
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| t.chars().take(4000).collect::<String>());
    let query_vector = match (&text, store.embedder.get()) {
        (Some(text), Some(embedder)) => Some(embedder.embed_query(text)?),
        _ => None,
    };
    let excluded = |id: &str| query.exclude.is_some_and(|set| set.contains(id));
    let config = &store.config.recall;
    let mut hits = store.with_index(|index, all| {
        // Topic digests are long overviews for explicit questions, never injected.
        let usable = |m: &Memory| {
            m.recallable()
                && m.in_scope(query.project)
                && !excluded(&m.id)
                && (query.mode == Mode::Search || m.kind != "digest")
        };
        let mut always = vec![];
        let mut found: HashMap<String, Hit> = HashMap::new();
        if query.always_on {
            for memory in all.values().filter(|m| m.always_on() && usable(m)) {
                always.push(Hit {
                    memory: memory.clone(),
                    channel: "rule",
                    score: 1.0,
                    reason: if memory.kind == "rule" {
                        "rule"
                    } else {
                        "pinned"
                    }
                    .into(),
                    group: None,
                });
            }
        }
        if let Some(features) = query.features {
            for memory in all.values().filter(|m| !m.always_on() && usable(m)) {
                if let Some(trigger) = memory.triggers.iter().find(|t| {
                    t.before_action == query.before_action && trigger_matches(t, features)
                }) {
                    found.insert(
                        memory.id.clone(),
                        Hit {
                            memory: memory.clone(),
                            channel: "trigger",
                            score: 0.95,
                            reason: format!("trigger:{}", trigger.id),
                            group: None,
                        },
                    );
                }
            }
        }
        if let Some(text) = text.as_deref() {
            let search = query.mode == Mode::Search;
            let pool = if search { query.limit.max(16) * 3 } else { 50 };
            // Aggregate questions need every item of a topic: look further down the list.
            let pool = if search && intent(text).aggregate { pool * 2 } else { pool };
            // For injection, a much better match in another project means the step is
            // probably not about this project's memories: the cutoff uses the best match
            // anywhere, while candidates come only from memories in scope.
            let reference = if search {
                0.0
            } else {
                hybrid(
                    index,
                    config,
                    false,
                    text,
                    query_vector.as_deref(),
                    1,
                    |id| all.get(id).is_some_and(|m| !m.always_on() && m.recallable()),
                    0.0,
                )
                .first()
                .map_or(0.0, |s| s.1)
            };
            let scored = hybrid(
                index,
                config,
                search,
                text,
                query_vector.as_deref(),
                pool,
                |id| all.get(id).is_some_and(|m| !m.always_on() && usable(m)),
                reference,
            );
            for (id, score, reason) in scored {
                let Some(memory) = all.get(&id).filter(|m| !m.always_on() && usable(m)) else {
                    continue;
                };
                found.entry(id).or_insert(Hit {
                    memory: memory.clone(),
                    channel: "search",
                    score: score as f64,
                    reason,
                    group: None,
                });
            }
            // Entity index (search mode): every memory about a person, place or product
            // the question names, even when its text says "the user's sister".
            if search {
                for memory in all.values().filter(|m| !m.always_on() && usable(m)) {
                    let Some(entity) = memory.entities.iter().find(|e| mentions(text, e)) else {
                        continue;
                    };
                    let hit = found.entry(memory.id.clone()).or_insert(Hit {
                        memory: memory.clone(),
                        channel: "search",
                        score: ENTITY_SCORE,
                        reason: String::new(),
                        group: None,
                    });
                    hit.score = hit.score.max(ENTITY_SCORE) + ENTITY_BOOST;
                    hit.reason.push_str(&format!(" entity:{entity}"));
                }
            }
        }
        for id in query.extra_ids {
            if let Some(memory) = all.get(id).filter(|m| !m.always_on() && usable(m)) {
                found.entry(id.clone()).or_insert(Hit {
                    memory: memory.clone(),
                    channel: "gate",
                    score: 0.9,
                    reason: "gate".into(),
                    group: None,
                });
            }
        }
        let mut hits: Vec<Hit> = found.into_values().collect();
        // Aggregate questions ("how many...") are answered from whole-topic overviews.
        let aggregate = query.mode == Mode::Search && text.as_deref().is_some_and(|t| intent(t).aggregate);
        if aggregate {
            for hit in hits.iter_mut().filter(|h| h.memory.kind == "digest") {
                hit.score += AGGREGATE_DIGEST_BOOST;
                hit.reason.push_str(" aggregate");
            }
        }
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.memory.id.cmp(&b.memory.id))
        });
        hits.truncate(if aggregate { query.limit * 3 / 2 } else { query.limit });
        always.sort_by(|a, b| a.memory.id.cmp(&b.memory.id));
        always.extend(hits);
        always
    });
    if query.mode == Mode::Search
        && query.episodes > 0
        && let Some(text) = text.as_deref()
    {
        let always = hits.iter().take_while(|h| h.channel == "rule").count();
        let mut ranked = hits.split_off(always);
        ranked.extend(episode_hits(
            store,
            text,
            query_vector.as_deref(),
            query.project,
            query.episodes,
        )?);
        ranked.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.memory.id.cmp(&b.memory.id))
        });
        hits.extend(ranked);
    }
    Ok(hits)
}

/// Greedy clustering of ranked vectors: each joins the first group whose best (first)
/// member is similar enough; returns indices per group, groups in order of creation.
fn topics(vectors: &[Option<Vec<f32>>]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = vec![];
    for (i, vector) in vectors.iter().enumerate() {
        let joined = vector.as_ref().and_then(|v| {
            groups.iter_mut().find(|group| {
                vectors[group[0]]
                    .as_ref()
                    .is_some_and(|seed| crate::embed::dot(v, seed) >= TOPIC_SIMILARITY)
            })
        });
        match joined {
            Some(group) => group.push(i),
            None => groups.push(vec![i]),
        }
    }
    groups
}

/// Base score of a memory found only through the entity index, and the boost for any
/// memory whose entities the question names.
const ENTITY_SCORE: f64 = 0.4;
const ENTITY_BOOST: f64 = 0.1;

/// Whether `text` names `entity` as a whole word (case-insensitive; CJK names by
/// substring). Latin names shorter than 3 characters are ignored.
fn mentions(text: &str, entity: &str) -> bool {
    let entity = entity.trim().to_lowercase();
    let cjk = entity.chars().any(|c| c as u32 >= 0x2E80);
    if entity.is_empty() || (!cjk && entity.chars().count() < 3) {
        return false;
    }
    let text = text.to_lowercase();
    let boundary = |c: Option<char>| c.is_none_or(|c| !c.is_alphanumeric());
    text.match_indices(&entity).any(|(start, _)| {
        cjk || (boundary(text[..start].chars().next_back())
            && boundary(text[start + entity.len()..].chars().next()))
    })
}

/// Earlier versions of a memory (the chain it superseded), newest first:
/// (date, content). At most `depth` entries.
pub fn history(store: &Store, memory: &Memory, depth: usize) -> Vec<(String, String)> {
    let mut out = vec![];
    let mut next = memory.supersedes.first().cloned();
    while let Some(id) = next {
        let Some(old) = store.memory(&id) else { break };
        if out.len() == depth {
            break;
        }
        let date = old
            .event_at
            .clone()
            .or_else(|| old.observed_at.clone())
            .unwrap_or_else(|| old.created_at.chars().take(10).collect());
        out.push((date, old.body.clone()));
        next = old.supersedes.first().cloned();
    }
    out
}

/// Sortable date of a hit: when the event happened if known, else when it was said
/// ("2023/05/30 (Tue) 22:16", "2023-05" and RFC 3339 all become digit strings).
fn date_key(memory: &Memory) -> String {
    memory
        .event_at
        .as_deref()
        .or(memory.observed_at.as_deref())
        .unwrap_or(&memory.created_at)
        .chars()
        .filter(char::is_ascii_digit)
        .take(12)
        .collect()
}

/// Cluster ranked search hits into topics by embedding similarity (greedily, against each
/// group's best hit), keep topics in order of relevance and put each topic's hits in
/// chronological order, so a reader sees how a fact developed and which value is the
/// latest. Always-on rules stay first; hits without a vector form their own group.
fn group_by_topic(store: &Store, hits: &mut Vec<Hit>) {
    let always = hits.iter().take_while(|h| h.channel == "rule").count();
    let ranked = hits.split_off(always);
    let vectors: Vec<Option<Vec<f32>>> = ranked
        .iter()
        .map(|h| {
            if h.channel == "episode" {
                store.with_episodes(|e| e.index.vector(&h.memory.id).cloned())
            } else {
                store.with_index(|index, _| index.vector(&h.memory.id).cloned())
            }
        })
        .collect();
    let mut ranked: Vec<Option<Hit>> = ranked.into_iter().map(Some).collect();
    for (number, mut group) in topics(&vectors).into_iter().enumerate() {
        group.sort_by_key(|&i| (date_key(&ranked[i].as_ref().unwrap().memory), i));
        for i in group {
            let mut hit = ranked[i].take().unwrap();
            hit.group = Some(number);
            hits.push(hit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{NewMemory, NewTrigger};
    use serde_json::json;

    fn store() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        crate::store::init(dir.path()).unwrap();
        let mut config = crate::store::load_config(dir.path()).unwrap();
        config.embedding.enabled = false;
        Store::save_config(dir.path(), &config).unwrap();
        let store = Store::open(dir.path()).unwrap();
        (dir, store)
    }
    fn add(
        store: &Store,
        content: &str,
        kind: &str,
        scope: &str,
        trigger: Option<(&str, &str)>,
    ) -> String {
        store
            .remember(
                NewMemory {
                    content: content.into(),
                    kind: kind.into(),
                    scope: scope.into(),
                    title: None,
                    triggers: trigger
                        .map(|(k, p)| {
                            vec![NewTrigger {
                                kind: k.into(),
                                pattern: p.into(),
                                before_action: false,
                            }]
                        })
                        .unwrap_or_default(),
                },
                "user",
            )
            .unwrap()
            .id
    }

    #[test]
    fn one_excerpt_per_conversation_turn() {
        let (_dir, store) = store();
        let reply = "The pottery class covers glazing and wheel throwing. ".repeat(80);
        let messages = [
            json!({"role":"user","content":"Tell me about the pottery class I signed up for"}),
            json!({"role":"assistant","content":reply}),
        ];
        store
            .ingest_session("q:s2", "bench", Some("q"), &messages, None)
            .unwrap();
        let session = store.session("q:s2").unwrap().unwrap();
        let entries = crate::extract::transcript(&store, &session).unwrap().entries;
        assert!(store.add_episodes(&session, &entries).unwrap() >= 3);
        let hits = recall(
            &store,
            &Query {
                text: Some("pottery class glazing wheel throwing"),
                project: Some("q"),
                limit: 5,
                mode: Mode::Search,
                episodes: 5,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(hits.iter().filter(|h| h.channel == "episode").count(), 1);
    }

    #[test]
    fn planned_search_merges_subqueries_and_boosts_the_window() {
        let (_dir, store) = store();
        let dated = |content: &str, date: &str| {
            let id = add(&store, content, "fact", "q", None);
            let mut memory = store.memory(&id).unwrap();
            memory.event_at = Some(date.into());
            store.save_memory(&memory, "date").unwrap();
            id
        };
        let jazz = dated("The user went to a jazz concert downtown", "2023-02-10");
        let rock = dated("The user went to a rock concert with Sam", "2023-03-12");
        dated("The user went to a folk concert in the park", "2023-04-02");
        let lens = add(&store, "The user bought a Canon lens", "fact", "q", None);
        let query = Query { text: Some("concert"), project: Some("q"), limit: 1, mode: Mode::Search, ..Default::default() };
        let ids = |hits: Vec<Hit>| hits.into_iter().map(|h| h.memory.id).collect::<Vec<_>>();
        let march = Plan { after: Some("2023-03-01".into()), before: Some("2023-03-31".into()), ..Default::default() };
        assert_eq!(ids(recall_planned(&store, &query, &march).unwrap()), std::slice::from_ref(&rock));
        let february = Plan { before: Some("2023-02".into()), ..Default::default() };
        assert_eq!(ids(recall_planned(&store, &query, &february).unwrap()), [jazz]);
        let both = Plan { queries: vec!["Canon lens".into()], ..march };
        let found = ids(recall_planned(&store, &Query { limit: 2, ..query }, &both).unwrap());
        assert!(found.contains(&rock) && found.contains(&lens), "{found:?}");
    }

    #[test]
    fn entity_index_finds_memories_that_do_not_repeat_the_name() {
        let (_dir, store) = store();
        let with_entities = |content: &str, entities: &[&str]| {
            let id = add(&store, content, "fact", "q", None);
            let mut memory = store.memory(&id).unwrap();
            memory.entities = entities.iter().map(|e| e.to_string()).collect();
            store.save_memory(&memory, "entities").unwrap();
            id
        };
        let moved = with_entities("The user's sister moved to Lisbon in November", &["Mira", "Lisbon"]);
        let job = with_entities("Her new job is at a design studio", &["Mira"]);
        with_entities("The user adopted a cat", &["Miso"]);
        let search = |text: &str, mode| {
            recall(&store, &Query { text: Some(text), project: Some("q"), limit: 10, mode, ..Default::default() })
                .unwrap()
                .into_iter()
                .map(|h| h.memory.id)
                .collect::<Vec<_>>()
        };
        let found = search("What is new with Mira?", Mode::Search);
        assert!(found.contains(&moved) && found.contains(&job) && found.len() == 2, "{found:?}");
        assert!(!search("What is new with Mira?", Mode::Inject).contains(&moved), "no entity channel in injection");
        assert!(search("Tell me about Miranda", Mode::Search).is_empty(), "whole words only");
        assert!(mentions("妹妹米拉搬家了", "米拉"));
    }

    #[test]
    fn intent_from_wording() {
        assert_eq!(intent("How many times did I bake last month?"), Intent { aggregate: true, temporal: false });
        assert_eq!(intent("Which streaming service did I start using most recently?"), Intent { aggregate: false, temporal: true });
        assert_eq!(intent("我一共去过几次杭州？最近一次是哪天？"), Intent { aggregate: true, temporal: true });
        assert_eq!(intent("What's my sister's name?"), Intent::default());
        assert!(!intent("I'm counting on you").aggregate, "whole words only");
    }

    #[test]
    fn packing_keeps_memories_first_trims_excerpts_and_skips_what_does_not_fit() {
        let hit = |id: &str, channel: &'static str, body: String, date: Option<&str>| {
            let mut memory = Memory::episode(id, Some("q"), body, None);
            memory.kind = if channel == "episode" { "episode".into() } else { "fact".into() };
            memory.event_at = date.map(str::to_owned);
            Hit { memory, channel, score: 0.5, reason: String::new(), group: None }
        };
        let filler = "The weather was nice and we talked about many unrelated things. ".repeat(20);
        let excerpt = format!("[user] Tell me about my baking\n[assistant] {filler} Last week you baked sourdough bread with rye flour. {filler}");
        let hits = vec![
            hit("ep_1", "episode", excerpt, None),
            hit("mem_big", "search", "x".repeat(4000), None),
            hit("mem_b", "search", "The user baked cookies.".into(), Some("2023-05-18")),
            hit("mem_a", "search", "The user baked a cake.".into(), Some("2023-05-02")),
        ];
        let packed = pack(hits, "When did I first bake sourdough bread?", 600);
        let ids: Vec<&str> = packed.iter().map(|h| h.memory.id.as_str()).collect();
        assert_eq!(ids, ["mem_a", "mem_b", "ep_1"], "big memory skipped, dated memories in order, excerpt last");
        let trimmed = &packed[2].memory.body;
        assert!(trimmed.starts_with("[user] Tell me about my baking") && trimmed.contains("sourdough bread with rye"));
        assert!(trimmed.chars().count() < 700, "{}", trimmed.len());
    }

    #[test]
    fn topics_cluster_similar_hits_and_dates_sort() {
        let unit = |a: f32| Some(vec![a.cos(), a.sin()]);
        // cos(0.3) = 0.955 (same topic), cos(1.2) = 0.36 (different).
        let vectors = [unit(0.0), unit(1.2), unit(0.3), None, unit(1.3)];
        assert_eq!(topics(&vectors), vec![vec![0, 2], vec![1, 4], vec![3]]);
        let mut memory = Memory::episode("ep_1", None, "x".into(), Some("2023/05/30 (Tue) 22:16".into()));
        assert_eq!(date_key(&memory), "202305302216");
        memory.observed_at = Some("2023-05-29T08:00:00Z".into());
        assert!(date_key(&memory).as_str() < "202305302216");
    }

    #[test]
    fn search_ranks_conversation_excerpts_with_memories() {
        let (dir, store) = store();
        add(&store, "The user enjoys creative writing", "fact", "q", None);
        let messages = [
            json!({"role":"user","content":"I paid $200 for the Saturday writing workshop downtown"}),
            json!({"role":"assistant","content":"That is a good investment in your craft."}),
        ];
        store
            .ingest_session("q:s1", "bench", Some("q"), &messages, Some("2023/05/20"))
            .unwrap();
        let session = store.session("q:s1").unwrap().unwrap();
        let entries = crate::extract::transcript(&store, &session).unwrap().entries;
        assert_eq!(store.add_episodes(&session, &entries).unwrap(), 1);
        assert_eq!(store.add_episodes(&session, &entries).unwrap(), 0, "retry adds nothing");
        let search = |store: &Store, mode, project| {
            recall(
                store,
                &Query {
                    text: Some("Saturday writing workshop"),
                    project,
                    limit: 5,
                    mode,
                    episodes: 3,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let hits = search(&store, Mode::Search, Some("q"));
        let excerpt = hits.iter().find(|h| h.channel == "episode").unwrap();
        assert!(excerpt.memory.body.contains("$200"));
        assert_eq!(excerpt.memory.observed_at.as_deref(), Some("2023/05/20"));
        assert!(search(&store, Mode::Inject, Some("q")).iter().all(|h| h.channel != "episode"));
        assert!(search(&store, Mode::Search, Some("other")).is_empty());
        drop(store);
        let reopened = Store::open(dir.path()).unwrap();
        assert!(search(&reopened, Mode::Search, Some("q")).iter().any(|h| h.channel == "episode"));
    }

    #[test]
    fn other_projects_do_not_crowd_out_candidates() {
        let (_dir, store) = store();
        for i in 0..200 {
            add(&store, &format!("Unrelated note {i} about tax forms"), "fact", "misc", None);
        }
        for i in 0..80 {
            add(&store, &format!("Hiking boots for trail {i} near Denver"), "fact", "other", None);
        }
        let mine = add(&store, "The user bought hiking boots", "fact", "q", None);
        let hits = recall(
            &store,
            &Query {
                text: Some("hiking boots Denver"),
                project: Some("q"),
                limit: 5,
                mode: Mode::Search,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(hits.iter().map(|h| &h.memory.id).collect::<Vec<_>>(), [&mine]);
    }

    #[test]
    fn search_mode_returns_more_than_injection() {
        let (_dir, store) = store();
        add(&store, "The user bought hiking boots in Denver", "fact", "q", None);
        add(&store, "The user went hiking near Denver with Sam", "fact", "q", None);
        add(&store, "The user went hiking last spring", "fact", "q", None);
        let count = |mode| {
            recall(
                &store,
                &Query {
                    text: Some("hiking boots Denver"),
                    project: Some("q"),
                    limit: 10,
                    mode,
                    ..Default::default()
                },
            )
            .unwrap()
            .len()
        };
        assert!(count(Mode::Search) > count(Mode::Inject));
        assert_eq!((count(Mode::Inject), count(Mode::Search)), (1, 2));
    }

    #[test]
    fn scopes_rules_triggers_and_keywords() {
        let (_dir, store) = store();
        let rule = add(&store, "Answer in Chinese", "rule", "global", None);
        let migrate = add(
            &store,
            "部署 payments-svc 之前必须先执行数据库迁移 (make migrate)",
            "lesson",
            "payments-svc",
            None,
        );
        let pnpm = add(
            &store,
            "atlas-web uses pnpm, never npm",
            "fact",
            "atlas-web",
            Some(("keyword", "npm install")),
        );
        let search = |text: &str, project: Option<&str>| {
            recall(
                &store,
                &Query {
                    text: Some(text),
                    project,
                    limit: 3,
                    always_on: true,
                    ..Default::default()
                },
            )
            .unwrap()
            .into_iter()
            .map(|h| h.memory.id)
            .collect::<Vec<_>>()
        };
        assert_eq!(
            search("make migrate", Some("payments-svc")),
            vec![rule.clone(), migrate.clone()]
        );
        assert_eq!(
            search("make migrate", Some("atlas-web")),
            vec![rule.clone()]
        );
        assert_eq!(search("make migrate", None), vec![rule.clone()]);
        assert_eq!(search("数据库迁移", Some("payments-svc"))[1], migrate);
        let features = json!({"user_text":"please run npm install"});
        let hits = recall(
            &store,
            &Query {
                project: Some("atlas-web"),
                features: Some(&features),
                limit: 3,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(hits[0].memory.id, pnpm);
        assert_eq!(hits[0].channel, "trigger");
        let hits = recall(
            &store,
            &Query {
                text: Some("今天天气怎么样"),
                project: Some("payments-svc"),
                limit: 3,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(hits.is_empty());
    }
}
