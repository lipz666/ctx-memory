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
    /// Position in the conversation (turn number, see `episode::Turn`), when known.
    pub turn: Option<usize>,
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
    /// Search mode: add the user's own messages (`turn_log`) within the budget.
    pub turns: bool,
}

/// What a question asks for, from its wording (no model call).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Intent {
    /// "how many", "in total", "which ... have I": whole-topic overviews help.
    pub aggregate: bool,
    /// "most recently", "first", "how long ago": dated events in order help.
    pub temporal: bool,
    /// "you recommended", "our previous chat", "remind me": the answer is a detail of the
    /// original conversation, so its excerpts matter most.
    pub conversation: bool,
    /// "the order in which I brought up...": memories are listed in the order they were
    /// mentioned (creation order), not by event date.
    pub mention_order: bool,
    /// "summarize", "how has ... progressed", "walk me through": an account of the whole
    /// history, so the conversation timeline helps.
    pub overview: bool,
    /// "Have I ever...", "Did I...": a yes/no question about the user's own history, the
    /// only kind that raises an unresolved contradiction.
    pub yes_no: bool,
}

const AGGREGATE_CUES: &[&str] = &[
    "how many", "how much", "in total", "total", "number of", "count", "list", "all the", "all of the",
    "which ones", "多少", "几次", "几个", "几种", "总共", "一共", "总计", "哪些", "所有",
];
const CONVERSATION_CUES: &[&str] = &[
    "our previous", "previous chat", "previous conversation", "our conversation", "our last chat",
    "last time we", "you told me", "you said", "you mentioned", "you recommended", "you suggested",
    "you provided", "you wrote", "you gave", "you listed", "you shared", "remind me", "we discussed",
    "you recommend", "you suggest", "you advise", "you advised", "you explain", "you explained", "you propose",
    "you proposed", "you outline", "you outlined", "your recommendation", "your advice", "your suggestion",
    "we talked", "我们之前", "之前聊", "上次你", "你说过", "你提到", "你推荐", "你建议", "你给我", "提醒我",
];
const OVERVIEW_CUES: &[&str] = &[
    "summarize", "summarise", "summary", "overview", "recap", "progressed", "progress", "walk me through",
    "over time", "throughout", "across our conversations", "so far", "总结", "概括", "回顾", "进展", "梳理",
];
const YES_NO_STARTS: &[&str] = &[
    "have i ", "has my ", "had i ", "did i ", "do i ", "does my ", "am i ", "was i ", "were i ", "is my ",
    "are my ", "is it true", "have we ", "did we ", "do we ", "我有没有", "我是否", "我有没有", "我曾经", "我做过",
];
const MENTION_ORDER_CUES: &[&str] = &[
    "order in which i", "order i brought", "order i mentioned", "brought up", "i mentioned first",
    "提起的顺序", "提到的顺序", "先后提到", "先提到",
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
    Intent {
        aggregate: has(AGGREGATE_CUES),
        temporal: has(TEMPORAL_CUES),
        conversation: has(CONVERSATION_CUES),
        mention_order: has(MENTION_ORDER_CUES),
        overview: has(OVERVIEW_CUES),
        yes_no: YES_NO_STARTS.iter().any(|start| text.trim_start().starts_with(start)),
    }
}

/// Score added to topic digests for aggregate questions.
const AGGREGATE_DIGEST_BOOST: f64 = 0.15;
/// Share of a packing budget kept for conversation excerpts once memories are placed.
const EXCERPT_SHARE: f64 = 0.35;
/// Share of a packing budget the conversation timeline may take.
const TIMELINE_SHARE: f64 = 0.35;
/// Characters of the timeline without a packing budget.
const TIMELINE_CHARS: usize = 6000;
/// Characters per timeline line: at most, and at least before lines are left out.
const TIMELINE_LINE_MAX: usize = 400;
const TIMELINE_LINE_MIN: usize = 140;
/// Standing instructions attached to every search, most relevant first. A person has few
/// (about four per BEAM conversation) and a missed one costs the answer, so all of them.
const INSTRUCTIONS: usize = 10;
/// The user's own messages given with a search (`turn_log`): the whole record for
/// questions about how things went, in what order or how often, otherwise the most
/// relevant ones; each message cut to TURN_LINE_CHARS.
pub const TURN_LOG_CHARS: usize = 64_000;
const TURN_LINE_CHARS: usize = 700;
/// Characters of the gist of a reply shown after the user's message.
const TURN_NOTE_LINE_CHARS: usize = 500;
pub const RELEVANT_TURNS: usize = 20;
/// Share of a packing budget the user's messages may take (`Query::turns`).
const TURN_SHARE: f64 = 0.3;

/// Hits that stay at the top in this order: rules, the user's standing instructions,
/// unresolved contradictions, the conversation timeline, the user's messages.
fn leading(hit: &Hit) -> bool {
    matches!(hit.channel, "rule" | "instruction" | "conflict" | "timeline" | "turnlog")
}

/// Characters of an excerpt kept around its best-matching sentence when packing. Whole
/// excerpts at large budgets were tried: fewer distinct turns fit, and the share of
/// questions whose source turn was in the context fell (BEAM 100K, 8k budget: 89% to 83%).
const EXCERPT_WINDOW: usize = 400;
/// Aggregate questions ("how many times...", "in total") draw on many turns: more
/// excerpts, each shorter, and a larger share of the budget.
const AGGREGATE_EXCERPT_WINDOW: usize = 250;
const AGGREGATE_EXCERPT_SHARE: f64 = 0.5;

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
        scored.truncate(limit * 6);
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
    // Questions about an earlier conversation may need a later part of a long reply: allow
    // several excerpts of one turn and look at more of them.
    let conversation = intent(text).conversation;
    let (limit, per_turn) = if conversation { (limit * 2, 3) } else { (limit, 1) };
    let mut turns: HashMap<(String, String), usize> = HashMap::new();
    for (id, score, reason, observed_at, session) in scored {
        if hits.len() == limit {
            break;
        }
        if let Some(text) = store.episode_text(&id)? {
            // A long turn is split into several excerpts that all start with the user's
            // message; returning more than one repeats the same statement.
            let seen = turns.entry((session, crate::episode::turn_key(&text))).or_insert(0);
            if *seen >= per_turn {
                continue;
            }
            *seen += 1;
            hits.push(Hit {
                memory: Memory::episode(&id, project, text, observed_at),
                channel: "episode",
                score: score as f64,
                reason,
                group: None, turn: None,
            });
        }
    }
    Ok(hits)
}

pub fn recall(store: &Store, query: &Query) -> Result<Vec<Hit>> {
    let mut hits = recall_ranked(store, query)?;
    if query.mode == Mode::Search {
        for hit in hits.iter_mut() {
            hit.turn = position(store, &hit.memory, hit.channel);
        }
        group_by_topic(store, &mut hits);
        if let Some(text) = query.text {
            let wanted = intent(text);
            // A contradiction is raised only when asked whether something is so; for a
            // count, a date or a summary both statements are just evidence.
            if wanted.yes_no {
                conflict_notes(store, &mut hits);
            }
            if wanted.overview || wanted.mention_order || wanted.aggregate {
                let chars = query.budget.map_or(TIMELINE_CHARS, |b| (b as f64 * 4.0 * TIMELINE_SHARE) as usize);
                if let Some(hit) = timeline(store, text, query.project, chars)? {
                    let at = hits.iter().take_while(|h| leading(h)).count();
                    hits.insert(at, hit);
                }
            }
            if query.turns {
                let chars = query.budget.map_or(TURN_LOG_CHARS / 4, |b| (b as f64 * 4.0 * TURN_SHARE) as usize);
                if let Some(hit) = turn_log(store, text, query.project, chars)? {
                    let at = hits.iter().take_while(|h| leading(h)).count();
                    hits.insert(at, hit);
                }
            }
        }
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
    if intent.conversation {
        return pack_conversation(memories, excerpts, question, budget);
    }
    let (window, share) = if intent.aggregate {
        (AGGREGATE_EXCERPT_WINDOW, AGGREGATE_EXCERPT_SHARE)
    } else {
        (EXCERPT_WINDOW, EXCERPT_SHARE)
    };
    let excerpts: Vec<Hit> = excerpts
        .into_iter()
        .map(|mut h| {
            h.memory.body = trim_excerpt(&h.memory.body, question, window);
            h
        })
        .collect();
    let memory_budget = if excerpts.is_empty() { budget } else { (budget as f64 * (1.0 - share)) as usize };
    let mut used = 0;
    let mut kept_memories = vec![];
    let mut left = vec![];
    for hit in memories.drain(..) {
        let cost = hit_tokens(&hit);
        if leading(&hit) || used + cost <= memory_budget {
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
    let rules = kept_memories.iter().take_while(|h| leading(h)).count();
    if intent.mention_order {
        // The conversation order: turn numbers, else creation order (sessions are
        // extracted in sequence).
        kept_memories[rules..].sort_by(|a, b| {
            (a.turn.unwrap_or(usize::MAX), &a.memory.created_at).cmp(&(b.turn.unwrap_or(usize::MAX), &b.memory.created_at))
        });
    } else if intent.temporal {
        // Same-day memories keep the order they were mentioned in.
        kept_memories[rules..].sort_by(|a, b| {
            date_key(&a.memory)
                .cmp(&date_key(&b.memory))
                .then_with(|| a.turn.cmp(&b.turn))
                .then_with(|| a.memory.created_at.cmp(&b.memory.created_at))
        });
    }
    kept_memories.extend(kept_excerpts);
    kept_memories
}

/// Packing for questions about an earlier conversation ("what did you recommend..."):
/// the answer is a detail of the original text, so excerpts come first and whole (cut to
/// the relevant window only when a whole one no longer fits), memories fill the rest.
fn pack_conversation(memories: Vec<Hit>, excerpts: Vec<Hit>, question: &str, budget: usize) -> Vec<Hit> {
    let mut used = 0;
    let mut kept = vec![];
    for mut hit in excerpts {
        if used + hit_tokens(&hit) > budget {
            hit.memory.body = trim_excerpt(&hit.memory.body, question, EXCERPT_WINDOW);
        }
        let cost = hit_tokens(&hit);
        if used + cost <= budget {
            used += cost;
            kept.push(hit);
        }
    }
    let mut kept_memories = vec![];
    for hit in memories {
        let cost = hit_tokens(&hit);
        if leading(&hit) || used + cost <= budget {
            used += cost;
            kept_memories.push(hit);
        }
    }
    kept_memories.extend(kept);
    kept_memories
}

/// The first line of an excerpt (the user's message, shortened) and the window of about
/// `window` characters around the sentence sharing most words with the question.
fn trim_excerpt(text: &str, question: &str, window: usize) -> String {
    if text.chars().count() <= window + 200 {
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
    while len(start, end) < window && (start > 0 || end < sentences.len()) {
        if end < sentences.len() {
            end += 1;
        }
        if len(start, end) < window && start > 0 {
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
        // Questions about when or in what order things happened also need statements
        // that were later changed ("the plan was April 20", moved since).
        let past = query.mode == Mode::Search && text.as_deref().is_some_and(|t| intent(t).temporal);
        let usable = |m: &Memory| {
            (m.recallable() || (past && m.status == "superseded"))
                && m.in_scope(query.project)
                && !excluded(&m.id)
                && (query.mode == Mode::Search || m.kind != "digest")
                && m.kind != "summary"
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
                    group: None, turn: None,
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
                            group: None, turn: None,
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
                // A statement that was later changed ranks a little below current ones.
                let score = if memory.status == "superseded" { score * SUPERSEDED_DISCOUNT } else { score };
                found.entry(id).or_insert(Hit {
                    memory: memory.clone(),
                    channel: "search",
                    score: score as f64,
                    reason,
                    group: None, turn: None,
                });
            }
            // Entity index (search mode): memories about a person, place or product the
            // question names, even when their text says "the user's sister". They rank
            // below text matches, and only a few of them: a name alone does not make a
            // memory relevant ("the grocery budget Alexis and I agreed on").
            if search {
                let mut named: Vec<&Memory> = all
                    .values()
                    .filter(|m| !m.always_on() && usable(m) && !found.contains_key(&m.id))
                    .filter(|m| m.entities.iter().any(|e| mentions(text, e)))
                    .collect();
                named.sort_by(|a, b| b.created_at.cmp(&a.created_at).then_with(|| a.id.cmp(&b.id)));
                for memory in named.into_iter().take((query.limit / 4).max(3)) {
                    let entity = memory.entities.iter().find(|e| mentions(text, e)).cloned().unwrap_or_default();
                    found.insert(
                        memory.id.clone(),
                        Hit {
                            memory: memory.clone(),
                            channel: "search",
                            score: ENTITY_SCORE,
                            reason: format!("entity:{entity}"),
                            group: None, turn: None,
                        },
                    );
                }
                // The user's standing instructions ("always include a tree diagram when
                // explaining probability"), the most relevant few, whatever their score.
                let instructions = |id: &str| all.get(id).is_some_and(|m| m.kind == "instruction" && !m.always_on() && usable(m));
                let ranked: Vec<(String, f32)> = match query_vector.as_deref() {
                    Some(vector) => index.semantic_scores_where(vector, instructions),
                    None => index.keyword_scores_where(text, instructions),
                };
                for (id, score) in ranked.into_iter().take(INSTRUCTIONS) {
                    found.remove(&id);
                    if let Some(memory) = all.get(&id) {
                        always.push(Hit {
                            memory: memory.clone(),
                            channel: "instruction",
                            score: 1.0,
                            reason: format!("instruction:{score:.2}"),
                            group: None, turn: None,
                        });
                    }
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
                    group: None, turn: None,
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
        // Rules first (by id), then instructions (most relevant first).
        always.sort_by(|a, b| (a.channel != "rule").cmp(&(b.channel != "rule")).then_with(|| {
            if a.channel == "rule" { a.memory.id.cmp(&b.memory.id) } else { std::cmp::Ordering::Equal }
        }));
        always.extend(hits);
        always
    });
    if query.mode == Mode::Search
        && query.episodes > 0
        && let Some(text) = text.as_deref()
    {
        let always = hits.iter().take_while(|h| leading(h)).count();
        let mut ranked = hits.split_off(always);
        let excerpts = episode_hits(
            store,
            text,
            query_vector.as_deref(),
            query.project,
            // Aggregate questions draw on many turns.
            if intent(text).aggregate { query.episodes * 2 } else { query.episodes },
        )?;
        let sources = source_excerpts(store, &ranked, &excerpts, query.project, query.episodes)?;
        ranked.extend(excerpts);
        ranked.extend(sources);
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
const SUPERSEDED_DISCOUNT: f32 = 0.9;
/// A memory's source excerpt must be at least this similar to it, and ranks just below it.
const SOURCE_SIMILARITY: f32 = 0.55;
const SOURCE_DISCOUNT: f64 = 0.95;

/// The conversation excerpts that the best memory hits were distilled from (the most
/// similar excerpt in a memory's session): a question phrased unlike the original turn
/// often matches the distilled memory ("the assistant recommended...") but not the long
/// reply it came from. At most `limit`, none already among `found` or from the same turn.
fn source_excerpts(store: &Store, memories: &[Hit], found: &[Hit], project: Option<&str>, limit: usize) -> Result<Vec<Hit>> {
    let mut seen: HashSet<String> = found.iter().map(|h| h.memory.id.clone()).collect();
    let mut turns: HashSet<String> = found.iter().map(|h| crate::episode::turn_key(&h.memory.body)).collect();
    let mut out = vec![];
    for hit in memories.iter().filter(|h| h.channel == "search" && !h.memory.derived() && !h.memory.evidence.is_empty()) {
        if out.len() == limit {
            break;
        }
        let Some(vector) = store.with_index(|index, _| index.vector(&hit.memory.id).cloned()) else { continue };
        let best = store.with_episodes(|episodes| {
            episodes
                .meta
                .iter()
                .filter(|(id, meta)| hit.memory.evidence.contains(&meta.session) && episodes.in_scope(id, project))
                .filter_map(|(id, meta)| {
                    episodes.index.vector(id).map(|v| (id.clone(), crate::embed::dot(&vector, v), meta.observed_at.clone()))
                })
                .max_by(|a, b| a.1.total_cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
        });
        let Some((id, similarity, observed_at)) = best else { continue };
        if similarity < SOURCE_SIMILARITY || !seen.insert(id.clone()) {
            continue;
        }
        let Some(text) = store.episode_text(&id)? else { continue };
        if !turns.insert(crate::episode::turn_key(&text)) {
            continue;
        }
        out.push(Hit {
            memory: Memory::episode(&id, project, text, observed_at),
            channel: "episode",
            score: hit.score * SOURCE_DISCOUNT,
            reason: format!("source of {} ({similarity:.2})", hit.memory.id),
            group: None, turn: None,
        });
    }
    Ok(out)
}
/// Contradictions shown per search.
const CONFLICT_NOTES: usize = 3;

/// A hit's position in the conversation: an excerpt's or a user message's own turn; a
/// memory's, the turn it was distilled from (its most similar excerpt in the session it
/// was first said in). None for overviews and without vectors.
pub fn position(store: &Store, memory: &Memory, channel: &str) -> Option<usize> {
    match channel {
        "episode" => store.with_episodes(|e| e.meta.get(&memory.id).map(|m| m.turn)).filter(|&t| t > 0),
        "turn" => store.with_episodes(|e| e.turn_meta.get(&memory.id).map(|t| t.number)),
        "timeline" | "conflict" | "turnlog" => None,
        _ if memory.derived() => None,
        _ => {
            let session = memory.evidence.first()?;
            let vector = store.with_index(|index, _| index.vector(&memory.id).cloned())?;
            store.with_episodes(|e| {
                let mut best: Option<(usize, f32)> = None;
                for id in e.session_turns.get(session)? {
                    let Some(turn) = e.turn_meta.get(id) else { continue };
                    for excerpt in &turn.excerpts {
                        if let Some(v) = e.index.vector(excerpt) {
                            let similarity = crate::embed::dot(&vector, v);
                            if best.is_none_or(|(_, b)| similarity > b) {
                                best = Some((turn.number, similarity));
                            }
                        }
                    }
                }
                best.map(|(number, _)| number)
            })
        }
    }
}

/// A question about what the assistant contributed, or an account of how things went
/// (not a count or an order of what the user brought up).
pub fn summary_question(wanted: &Intent) -> bool {
    wanted.conversation || (wanted.overview && !wanted.aggregate && !wanted.mention_order)
}

/// The user's messages in scope ranked by relevance to `text`: (turn id, score).
pub(crate) fn ranked_turns(store: &Store, text: &str, vector: Option<&[f32]>, project: Option<&str>) -> Vec<(String, f32)> {
    store.with_episodes(|e| {
        hybrid(&e.turns, &store.config.recall, true, text, vector, e.turn_meta.len().max(1), |id| e.turn_in_scope(id, project), 0.0)
            .into_iter()
            .map(|(id, score, _)| (id, score))
            .collect()
    })
}

/// The user's own messages as one block in conversation order (turn number, date said,
/// text), within `chars`: every message in scope for questions about how things went, in
/// what order or how often (the most relevant when they do not all fit), otherwise the
/// RELEVANT_TURNS most relevant. The user's own words decide what they said, did or
/// planned, and the turn numbers which of two things came later on the same day.
pub fn turn_log(store: &Store, text: &str, project: Option<&str>, chars: usize) -> Result<Option<Hit>> {
    let vector = match store.embedder.get() {
        Some(embedder) => Some(embedder.embed_query(text)?),
        None => None,
    };
    let wanted = intent(text);
    let whole = wanted.overview || wanted.mention_order || wanted.aggregate;
    // The gist of the assistant's replies only where the question is about them: what
    // was recommended, or an account of how things went. Next to the user's own words it
    // is otherwise taken for what the user did or said (a suggested setting read as the
    // one the user chose, a general figure read as the user's latest value).
    let notes = summary_question(&wanted);
    let ranked = ranked_turns(store, text, vector.as_deref(), project);
    let (lines, count, total) = store.with_episodes(|e| {
        let line = |t: &crate::episode::Turn| {
            let mut body: String = t.text.chars().take(TURN_LINE_CHARS).collect();
            if body.len() < t.text.len() {
                body.push('…');
            }
            let note = t
                .reply()
                .filter(|_| notes)
                .map(|n| format!(" → Assistant: {}", n.chars().take(TURN_NOTE_LINE_CHARS).collect::<String>().replace('\n', " ")))
                .unwrap_or_default();
            format!("- #{} [{}] {}{note}", t.number, t.observed_at.as_deref().unwrap_or("date unknown"), body.replace('\n', " "))
        };
        let in_scope: Vec<(&String, &crate::episode::Turn)> =
            e.turn_meta.iter().filter(|(id, t)| !t.text.is_empty() && e.turn_in_scope(id, project)).collect();
        let total = in_scope.len();
        // Most relevant first; for a whole record, the others after them.
        let mut order: Vec<&String> = ranked.iter().map(|(id, _)| id).filter(|id| e.turn_meta.contains_key(*id)).collect();
        if whole {
            let listed: HashSet<&String> = order.iter().copied().collect();
            let mut rest: Vec<&String> = in_scope.iter().map(|(id, _)| *id).filter(|id| !listed.contains(id)).collect();
            rest.sort_by_key(|id| e.turn_meta[*id].number);
            order.extend(rest);
        } else {
            order.truncate(RELEVANT_TURNS);
        }
        let mut used = 0;
        let mut chosen: Vec<&crate::episode::Turn> = vec![];
        for id in order {
            let turn = &e.turn_meta[id];
            let cost = line(turn).chars().count() + 1;
            if used + cost <= chars {
                used += cost;
                chosen.push(turn);
            }
        }
        chosen.sort_by_key(|t| t.number);
        let count = chosen.len();
        (chosen.into_iter().map(line).collect::<Vec<_>>(), count, total)
    });
    if lines.is_empty() {
        return Ok(None);
    }
    let heading = if count == total {
        format!("The user's own messages, all {total}, in conversation order (turn number: a higher number is later, also on the same date; date said; → the gist of the assistant's reply):")
    } else {
        format!("The user's own messages most relevant to the question ({count} of {total}), in conversation order (turn number: a higher number is later, also on the same date; date said; → the gist of the assistant's reply):")
    };
    let mut memory = Memory::episode("turnlog", project, format!("{heading}\n{}", lines.join("\n")), None);
    memory.kind = "turnlog".into();
    memory.title = "The user's messages".into();
    Ok(Some(Hit { memory, channel: "turnlog", score: 1.0, reason: format!("{count} of {total} turns"), group: None, turn: None }))
}

/// Replace the two sides of an unresolved contradiction among the hits with one note that
/// states both (a reader skims past a remark at the end of one memory).
fn conflict_notes(store: &Store, hits: &mut Vec<Hit>) {
    let mut notes: Vec<Hit> = vec![];
    let mut covered: HashSet<String> = HashSet::new();
    for hit in hits.iter().filter(|h| h.memory.status == "contested" && !leading(h)) {
        if notes.len() == CONFLICT_NOTES || covered.contains(&hit.memory.id) {
            continue;
        }
        let Some(other) = hit.memory.conflicts_with.iter().find_map(|id| store.memory(id)) else { continue };
        // Only a denial against a statement is a contradiction to raise; two different
        // values of one thing read as a change.
        if !(crate::contradict::negative(&hit.memory.body) || crate::contradict::negative(&other.body)) {
            continue;
        }
        let said = |m: &Memory| m.observed_at.clone().unwrap_or_else(|| m.created_at.chars().take(10).collect());
        let (first, second) = if (said(&other), &other.created_at) < (said(&hit.memory), &hit.memory.created_at) {
            (&other, &hit.memory)
        } else {
            (&hit.memory, &other)
        };
        let text = format!(
            "Unresolved contradiction in what the user has said: on {} — \"{}\"; on {} — \"{}\". These cannot both be true, and the user has not said which one is correct.",
            said(first),
            first.body,
            said(second),
            second.body
        );
        let mut memory = Memory::episode(&format!("conflict:{}:{}", first.id, second.id), Some(&hit.memory.scope), text, None);
        memory.kind = "conflict".into();
        memory.title = "Unresolved contradiction".into();
        covered.extend([first.id.clone(), second.id.clone()]);
        notes.push(Hit { memory, channel: "conflict", score: 1.0, reason: "conflict".into(), group: None, turn: None });
    }
    if notes.is_empty() {
        return;
    }
    hits.retain(|h| !covered.contains(&h.memory.id));
    let at = hits.iter().take_while(|h| matches!(h.channel, "rule" | "instruction")).count();
    hits.splice(at..at, notes);
}

/// The conversation timeline: one line per session summary in the scope, oldest first,
/// within `chars`. When every line cannot keep TIMELINE_LINE_MIN characters, the lines
/// most relevant to `text` are kept (still in order).
pub(crate) fn timeline(store: &Store, text: &str, project: Option<&str>, chars: usize) -> Result<Option<Hit>> {
    let vector = match store.embedder.get() {
        Some(embedder) => Some(embedder.embed_query(text)?),
        None => None,
    };
    let lines = store.with_index(|index, all| {
        let mut summaries: Vec<&Memory> = all
            .values()
            .filter(|m| m.kind == "summary" && m.recallable() && m.in_scope(project))
            .collect();
        if summaries.is_empty() {
            return vec![];
        }
        summaries.sort_by(|a, b| a.created_at.cmp(&b.created_at).then_with(|| a.id.cmp(&b.id)));
        let per_line = (chars / summaries.len()).min(TIMELINE_LINE_MAX);
        let (keep, per_line): (HashSet<String>, usize) = if per_line >= TIMELINE_LINE_MIN {
            (summaries.iter().map(|m| m.id.clone()).collect(), per_line)
        } else {
            let allow = |id: &str| all.get(id).is_some_and(|m| m.kind == "summary" && m.recallable() && m.in_scope(project));
            let ranked = match vector.as_deref() {
                Some(v) => index.semantic_scores_where(v, allow),
                None => index.keyword_scores_where(text, allow),
            };
            let count = (chars / TIMELINE_LINE_MIN).max(1);
            (ranked.into_iter().take(count).map(|(id, _)| id).collect(), TIMELINE_LINE_MIN)
        };
        let turns = |m: &Memory| -> String {
            store.with_episodes(|e| {
                let numbers: Vec<usize> = m
                    .evidence
                    .first()
                    .and_then(|session| e.session_turns.get(session))
                    .map(|ids| ids.iter().filter_map(|id| e.turn_meta.get(id).map(|t| t.number)).collect())
                    .unwrap_or_default();
                match (numbers.iter().min(), numbers.iter().max()) {
                    (Some(first), Some(last)) => format!(", turns #{first}–#{last}"),
                    _ => String::new(),
                }
            })
        };
        summaries
            .into_iter()
            .enumerate()
            .filter(|(_, m)| keep.contains(&m.id))
            .map(|(i, m)| {
                let date = m.observed_at.clone().unwrap_or_else(|| m.created_at.chars().take(10).collect());
                let mut body: String = m.body.chars().take(per_line).collect();
                if body.len() < m.body.len() {
                    body.push('…');
                }
                format!("- Session {} [{date}{}] {body}", i + 1, turns(m))
            })
            .collect::<Vec<_>>()
    });
    if lines.is_empty() {
        return Ok(None);
    }
    let mut memory = Memory::episode(
        "timeline",
        project,
        format!("Timeline of our conversations (session, date, turns; what was discussed; oldest first):\n{}", lines.join("\n")),
        None,
    );
    memory.kind = "timeline".into();
    memory.title = "Conversation timeline".into();
    Ok(Some(Hit { memory, channel: "timeline", score: 1.0, reason: "timeline".into(), group: None, turn: None }))
}

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

/// Memories that contradict this one: (date, content).
pub fn conflicts(store: &Store, memory: &Memory) -> Vec<(String, String)> {
    memory
        .conflicts_with
        .iter()
        .filter_map(|id| store.memory(id))
        .filter(|m| crate::contradict::negative(&m.body) || crate::contradict::negative(&memory.body))
        .map(|m| {
            let date = m
                .event_at
                .clone()
                .or_else(|| m.observed_at.clone())
                .unwrap_or_else(|| m.created_at.chars().take(10).collect());
            (date, m.body)
        })
        .collect()
}

/// Position of a memory in the order its scope's memories were mentioned (1-based).
pub fn mention_rank(store: &Store, memory: &Memory) -> usize {
    store.with_index(|_, all| {
        all.values()
            .filter(|m| m.scope == memory.scope && !m.derived())
            .filter(|m| m.created_at < memory.created_at)
            .count()
            + 1
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
    let always = hits.iter().take_while(|h| leading(h)).count();
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
        group.sort_by_key(|&i| {
            let hit = ranked[i].as_ref().unwrap();
            (date_key(&hit.memory), hit.turn, i)
        });
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
    fn timeline_instructions_and_a_few_entity_hits() {
        let (_dir, store) = store();
        for (i, text) in ["Set up the Flask project and the database schema.", "Implemented transaction CRUD with error handling.", "Deployed the app to Render with Gunicorn."].iter().enumerate() {
            let id = add(&store, text, "summary", "q", None);
            let mut memory = store.memory(&id).unwrap();
            memory.observed_at = Some(format!("2024/0{}/01", i + 1));
            memory.created_at = format!("2024-0{}-01T00:00:00Z", i + 1);
            store.save_memory(&memory, "date").unwrap();
        }
        let tip = add(&store, "When explaining probability, always include a tree diagram.", "instruction", "q", None);
        let grocery = add(&store, "The grocery budget was raised to $550 per month.", "fact", "q", None);
        for i in 0..8 {
            let id = add(&store, &format!("Unrelated plan number {i} for the weekend."), "fact", "q", None);
            let mut memory = store.memory(&id).unwrap();
            memory.entities = vec!["Alexis".into()];
            store.save_memory(&memory, "entities").unwrap();
        }
        let search = |text: &str, budget: Option<usize>| {
            recall(&store, &Query { text: Some(text), project: Some("q"), limit: 8, mode: Mode::Search, budget, ..Default::default() }).unwrap()
        };
        let hits = search("Summarize how my Flask app project progressed", Some(2000));
        let timeline = hits.iter().find(|h| h.channel == "timeline").expect("timeline");
        let body = &timeline.memory.body;
        assert!(body.contains("- Session 1 [2024/01/01] Set up the Flask") && body.find("Session 1").unwrap() < body.find("Session 3 [2024/03/01] Deployed").unwrap(), "{body}");
        assert!(search("Flask project", None).iter().all(|h| h.memory.kind != "summary" && h.channel != "timeline"), "summaries only in the timeline");
        let hits = search("What is the probability of drawing a red card?", None);
        assert_eq!((hits[0].channel, hits[0].memory.id.as_str()), ("instruction", tip.as_str()));
        let hits = search("What grocery budget have Alexis and I agreed on?", None);
        assert_eq!(hits[0].memory.id, grocery);
        assert!(hits.iter().filter(|h| h.reason.starts_with("entity:")).count() <= 3, "a few entity-only hits");
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
        assert_eq!(intent("How many times did I bake last month?"), Intent { aggregate: true, ..Default::default() });
        assert_eq!(intent("Which streaming service did I start using most recently?"), Intent { temporal: true, ..Default::default() });
        assert_eq!(intent("我一共去过几次杭州？最近一次是哪天？"), Intent { aggregate: true, temporal: true, ..Default::default() });
        assert!(intent("What was the name of the hostel you recommended last time?").conversation);
        assert!(intent("上次你推荐的那本书叫什么？").conversation);
        assert_eq!(intent("What's my sister's name?"), Intent::default());
        assert!(!intent("I'm counting on you").aggregate, "whole words only");
        let order = intent("Can you list the order in which I brought up the parts of my budget app?");
        assert!(order.mention_order && order.temporal);
        assert!(intent("按我提到的顺序列出我问过的问题").mention_order);
        assert!(!intent("When did I first bake bread?").mention_order);
        assert!(intent("Can you summarize how my weather app project has progressed?").overview);
        assert!(!intent("What is my sister's name?").overview);
    }

    #[test]
    fn mention_order_questions_list_memories_as_they_were_mentioned() {
        let hit = |id: &str, created: &str, date: Option<&str>| {
            let mut memory = Memory::episode(id, Some("q"), format!("memory {id}"), None);
            memory.kind = "fact".into();
            memory.created_at = created.into();
            memory.event_at = date.map(str::to_owned);
            Hit { memory, channel: "search", score: 0.5, reason: String::new(), group: None, turn: None }
        };
        let hits = vec![
            hit("mem_c", "2024-03-01T10:00:02Z", Some("2024-01-01")),
            hit("mem_a", "2024-03-01T10:00:00Z", Some("2024-06-01")),
            hit("mem_b", "2024-03-01T10:00:01Z", Some("2024-06-01")),
        ];
        let ids = |packed: Vec<Hit>| packed.into_iter().map(|h| h.memory.id).collect::<Vec<_>>();
        assert_eq!(ids(pack(hits.clone(), "In what order did I bring up these features? List the order in which I mentioned them.", 1000)), ["mem_a", "mem_b", "mem_c"]);
        assert_eq!(ids(pack(hits, "Which did I do first?", 1000)), ["mem_c", "mem_a", "mem_b"], "event date, then mention order");
    }

    #[test]
    fn packing_keeps_memories_first_trims_excerpts_and_skips_what_does_not_fit() {
        let hit = |id: &str, channel: &'static str, body: String, date: Option<&str>| {
            let mut memory = Memory::episode(id, Some("q"), body, None);
            memory.kind = if channel == "episode" { "episode".into() } else { "fact".into() };
            memory.event_at = date.map(str::to_owned);
            Hit { memory, channel, score: 0.5, reason: String::new(), group: None, turn: None }
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
    fn conversation_questions_keep_whole_excerpts_first() {
        let hit = |id: &str, channel: &'static str, body: String| {
            let mut memory = Memory::episode(id, Some("q"), body, None);
            memory.kind = if channel == "episode" { "episode".into() } else { "fact".into() };
            Hit { memory, channel, score: 0.5, reason: String::new(), group: None, turn: None }
        };
        let reply = format!("[user] Write a script about Andy\n[assistant] {} Andy wore an untidy, stained white shirt. {}", "Scene one. ".repeat(40), "The end. ".repeat(20));
        let hits = vec![hit("mem_a", "search", "The user writes scripts.".into()), hit("ep_1", "episode", reply.clone())];
        let packed = pack(hits.clone(), "What was Andy wearing in the script you wrote for me?", 1000);
        assert_eq!(packed.iter().map(|h| h.memory.id.as_str()).collect::<Vec<_>>(), ["mem_a", "ep_1"]);
        assert_eq!(packed[1].memory.body, reply, "whole excerpt kept");
        let other = pack(hits, "What was Andy wearing?", 1000);
        assert!(other[1].memory.body.len() < reply.len(), "other questions keep the trimmed window");
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
    fn user_messages_are_numbered_in_conversation_order_and_survive_a_reopen() {
        let (dir, store) = store();
        let sessions = [
            ("q:b1-0", "2024/05/02", vec!["My grocery budget is $500 per month.", "I secured 3 interviews for producer roles."]),
            ("q:b1-1", "2024/05/02", vec!["Update: I secured 5 interviews for producer roles.", "The grocery budget is now $550."]),
        ];
        for (key, date, user) in &sessions {
            let messages: Vec<Value> = user
                .iter()
                .flat_map(|text| [json!({"role":"user","content":text}), json!({"role":"assistant","content":"Noted."})])
                .collect();
            store.ingest_session(key, "bench", Some("q"), &messages, Some(date)).unwrap();
            let session = store.session(key).unwrap().unwrap();
            let entries = crate::extract::transcript(&store, &session).unwrap().entries;
            store.add_episodes(&session, &entries).unwrap();
        }
        let log = |store: &Store, text: &str| turn_log(store, text, Some("q"), TURN_LOG_CHARS).unwrap().unwrap().memory.body;
        let whole = log(&store, "How many interviews have I secured in total?");
        assert!(whole.contains("all 4"), "{whole}");
        let first = whole.find("#2 [2024/05/02] I secured 3 interviews").expect("numbered");
        assert!(first < whole.find("#3 [2024/05/02] Update: I secured 5 interviews").unwrap(), "{whole}");
        let few = log(&store, "What is my grocery budget?");
        assert!(few.contains("most relevant") && few.contains("#4 [2024/05/02] The grocery budget is now $550."), "{few}");
        let hits = recall(&store, &Query { text: Some("interviews"), project: Some("q"), limit: 5, mode: Mode::Search, episodes: 4, ..Default::default() }).unwrap();
        assert!(hits.iter().filter(|h| h.channel == "episode").all(|h| h.turn.is_some()), "excerpts carry their turn");
        // The gist of a reply is stored with extraction (`save_turn_notes`), matched by the user's words.
        let session = store.session("q:b1-1").unwrap().unwrap();
        let user_texts = vec!["Update: I secured 5 interviews for producer roles.".to_string(), "The grocery budget is now $550.".to_string()];
        let turns = vec![json!({"n": 2, "assistant": "Suggested splitting the $550 into weekly $137.50 envelopes."}), json!({"n": 9, "assistant": "out of range"})];
        assert_eq!(crate::extract::save_turn_notes(&store, &session, &user_texts, &turns).unwrap(), 1);
        let summary = "Can you summarize how my budget and job search progressed?";
        let noted = log(&store, summary);
        assert!(noted.contains("#4 [2024/05/02] The grocery budget is now $550. → Assistant: Suggested splitting the $550 into weekly $137.50 envelopes."), "{noted}");
        assert!(!log(&store, "How many interviews have I secured in total?").contains("Assistant:"), "a count reads the user's own words only");
        assert!(ranked_turns(&store, "weekly envelopes", None, Some("q")).is_empty(), "the index holds the user's words only");
        drop(store);
        let reopened = Store::open(dir.path()).unwrap();
        assert_eq!(log(&reopened, summary), noted, "the same numbering and notes after a reopen");
    }

    #[test]
    fn contradictions_are_raised_only_for_yes_no_questions() {
        let (_dir, store) = store();
        for filler in ["The user adopted a cat named Miso.", "The user moved to Lisbon in May.", "The user plays tennis on Sundays."] {
            add(&store, filler, "fact", "q", None);
        }
        let did = add(&store, "The user attended a budgeting workshop led by Tamara.", "fact", "q", None);
        let never = add(&store, "The user has never attended any budgeting workshop.", "fact", "q", None);
        for (id, other) in [(&did, &never), (&never, &did)] {
            let mut memory = store.memory(id).unwrap();
            memory.status = "contested".into();
            memory.conflicts_with = vec![other.clone()];
            store.save_memory(&memory, "contest").unwrap();
        }
        let search = |text: &str| recall(&store, &Query { text: Some(text), project: Some("q"), limit: 5, mode: Mode::Search, ..Default::default() }).unwrap();
        assert_eq!(search("Have I ever attended a budgeting workshop?")[0].channel, "conflict");
        assert!(search("How many budgeting workshops did I attend?").iter().all(|h| h.channel != "conflict"));
        assert!(intent("Have I integrated Flask-Login?").yes_no && intent("Did I finish the draft?").yes_no);
        assert!(!intent("How many days did I spend?").yes_no && !intent("What have I done?").yes_no);
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
