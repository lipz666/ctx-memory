//! In-memory search index over memories: BM25 on a CJK-aware tokenizer plus embedding
//! vectors. Derived from the Markdown files; rebuilt on start, updated per write.
use std::collections::{HashMap, HashSet};

const K1: f32 = 1.2;
const B: f32 = 0.75;

static EN_STOP: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of", "in", "on", "for",
    "and", "or", "it", "its", "this", "that", "these", "those", "with", "as", "at", "by", "from",
    "i", "you", "we", "they", "he", "she", "do", "does", "did", "can", "could", "should", "would",
    "will", "how", "what", "why", "when", "where", "which", "who", "my", "me", "your", "our",
    "just", "so", "if", "not", "no", "yes", "please", "have", "has", "had", "about", "any", "some",
    "there", "then", "than", "into", "out", "get", "got", "let", "need", "want", "use", "using",
    "used", "now", "today", "also", "too", "very", "one", "all", "there's", "it's", "i'm", "don't",
    "doesn't", "dont",
];
static CJK_STOP: &[&str] = &[
    "什么",
    "怎么",
    "一下",
    "可以",
    "我们",
    "你们",
    "他们",
    "这个",
    "那个",
    "一个",
    "如何",
    "为什么",
    "怎样",
    "怎么样",
    "哪里",
    "哪些",
    "一些",
    "请问",
    "是否",
    "应该",
    "时候",
    "现在",
    "今天",
    "一直",
    "还是",
    "然后",
    "就是",
    "但是",
    "因为",
    "所以",
    "如果",
    "这样",
    "那样",
    "需要",
    "注意",
    "的",
    "了",
    "吗",
    "呢",
    "么",
    "吧",
    "啊",
    "我",
    "你",
    "他",
    "她",
    "它",
    "是",
    "在",
    "有",
    "和",
    "与",
    "或",
    "就",
    "都",
    "也",
    "还",
    "要",
    "能",
    "会",
    "把",
    "被",
    "给",
    "让",
    "对",
    "从",
    "向",
    "到",
    "为",
    "这",
    "那",
    "个",
    "些",
    "么",
    "一",
    "不",
    "没",
    "得",
    "地",
    "着",
    "过",
    "前",
    "后",
    "里",
    "上",
    "下",
    "中",
    "用",
    "做",
    "说",
    "看",
    "想",
    "好",
    "帮",
    "请",
    "先",
    "再",
    "又",
    "很",
    "最",
    "更",
    "已经",
    "可能",
    "一样",
    "东西",
];
static JIEBA: std::sync::LazyLock<jieba_rs::Jieba> = std::sync::LazyLock::new(jieba_rs::Jieba::new);

fn is_cjk(ch: char) -> bool {
    matches!(ch as u32,
        0x3040..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0xAC00..=0xD7AF)
}

fn stem(word: &str) -> String {
    let n = word.len();
    if !word.is_ascii() {
        return word.into();
    }
    if n > 4 && word.ends_with("ies") {
        format!("{}y", &word[..n - 3])
    } else if n > 5 && word.ends_with("ing") {
        word[..n - 3].into()
    } else if n > 4 && word.ends_with("ed") {
        word[..n - 2].into()
    } else if n > 3 && word.ends_with('s') && !word.ends_with("ss") && !word.ends_with("us") {
        word[..n - 1].into()
    } else {
        word.into()
    }
}

/// Lowercased terms: stemmed words (compound `a_b` also split) and segmented CJK words.
pub fn tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut word = String::new();
    let mut run: Vec<char> = Vec::new();
    let flush_word = |word: &mut String, out: &mut Vec<String>| {
        if word.is_empty() {
            return;
        }
        if word.contains('_') {
            for part in word.split('_').filter(|p| p.chars().count() >= 2) {
                if !EN_STOP.contains(&part) {
                    out.push(stem(part));
                }
            }
        }
        if word.chars().count() >= 2 && !EN_STOP.contains(&word.as_str()) {
            out.push(stem(word));
        }
        word.clear();
    };
    let flush_run = |run: &mut Vec<char>, out: &mut Vec<String>| {
        if run.is_empty() {
            return;
        }
        let segment: String = run.iter().collect();
        for token in JIEBA.cut_for_search(&segment, false) {
            if !CJK_STOP.contains(&token.word) {
                out.push(token.word.to_owned());
            }
        }
        run.clear();
    };
    for ch in text.to_lowercase().chars() {
        if is_cjk(ch) {
            flush_word(&mut word, &mut out);
            run.push(ch);
        } else if ch.is_alphanumeric() || ch == '_' {
            flush_run(&mut run, &mut out);
            word.push(ch);
        } else {
            flush_word(&mut word, &mut out);
            flush_run(&mut run, &mut out);
        }
    }
    flush_word(&mut word, &mut out);
    flush_run(&mut run, &mut out);
    out
}

#[derive(Default)]
struct Doc {
    tf: HashMap<String, u32>,
    len: u32,
}

#[derive(Default)]
pub struct Index {
    docs: HashMap<String, Doc>,
    df: HashMap<String, u32>,
    total_len: u64,
    vectors: HashMap<String, Vec<f32>>,
}

impl Index {
    pub fn upsert(&mut self, id: &str, text: &str) {
        self.remove_terms(id);
        let terms = tokens(text);
        let mut tf = HashMap::new();
        for term in &terms {
            *tf.entry(term.clone()).or_insert(0) += 1;
        }
        for term in tf.keys() {
            *self.df.entry(term.clone()).or_insert(0) += 1;
        }
        self.total_len += terms.len() as u64;
        self.docs.insert(
            id.into(),
            Doc {
                tf,
                len: terms.len() as u32,
            },
        );
    }
    fn remove_terms(&mut self, id: &str) {
        if let Some(doc) = self.docs.remove(id) {
            self.total_len -= doc.len as u64;
            for term in doc.tf.keys() {
                if let Some(count) = self.df.get_mut(term) {
                    *count -= 1;
                    if *count == 0 {
                        self.df.remove(term);
                    }
                }
            }
        }
    }
    pub fn remove(&mut self, id: &str) {
        self.remove_terms(id);
        self.vectors.remove(id);
    }
    pub fn remove_vector_only(&mut self, id: &str) {
        self.vectors.remove(id);
    }
    pub fn set_vector(&mut self, id: &str, vector: Vec<f32>) {
        self.vectors.insert(id.into(), vector);
    }
    pub fn has_vector(&self, id: &str) -> bool {
        self.vectors.contains_key(id)
    }
    pub fn vector(&self, id: &str) -> Option<&Vec<f32>> {
        self.vectors.get(id)
    }
    fn idf(&self, term: &str) -> f32 {
        let n = self.docs.len() as f32;
        let df = *self.df.get(term).unwrap_or(&0) as f32;
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
    }
    /// Keyword relevance in 0..=1: the share of the query's IDF mass found in a document
    /// (terms absent from every memory count against it), scaled by relative BM25.
    pub fn keyword_scores(&self, query: &str) -> Vec<(String, f32)> {
        self.keyword_scores_where(query, |_| true)
    }
    /// `keyword_scores` over the documents `allow` accepts (e.g. one project's memories),
    /// so other projects never crowd a project's candidates out of the pool.
    pub fn keyword_scores_where(
        &self,
        query: &str,
        allow: impl Fn(&str) -> bool,
    ) -> Vec<(String, f32)> {
        let terms: HashSet<String> = tokens(query).into_iter().collect();
        if terms.is_empty() || self.docs.is_empty() {
            return vec![];
        }
        let weighted: Vec<(&String, f32)> = terms.iter().map(|t| (t, self.idf(t))).collect();
        let total: f32 = weighted.iter().map(|(_, idf)| idf).sum();
        let avg = self.total_len as f32 / self.docs.len() as f32;
        let mut raw: Vec<(String, f32, f32)> = self
            .docs
            .iter()
            .filter(|(id, _)| allow(id))
            .filter_map(|(id, doc)| {
                let mut matched = 0.0;
                let mut bm25 = 0.0;
                for (term, idf) in &weighted {
                    if let Some(&tf) = doc.tf.get(*term) {
                        let tf = tf as f32;
                        let norm = K1 * (1.0 - B + B * doc.len as f32 / avg.max(1.0));
                        matched += idf;
                        bm25 += idf * tf * (K1 + 1.0) / (tf + norm);
                    }
                }
                (matched > 0.0).then(|| (id.clone(), matched / total, bm25))
            })
            .collect();
        let best = raw
            .iter()
            .map(|r| r.2)
            .fold(0.0, f32::max)
            .max(f32::EPSILON);
        let mut scores: Vec<(String, f32)> = raw
            .drain(..)
            .map(|(id, coverage, bm25)| (id, coverage * (0.8 + 0.2 * bm25 / best)))
            .collect();
        scores.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scores
    }
    pub fn semantic_scores(&self, query: &[f32]) -> Vec<(String, f32)> {
        self.semantic_scores_where(query, |_| true)
    }
    pub fn semantic_scores_where(
        &self,
        query: &[f32],
        allow: impl Fn(&str) -> bool,
    ) -> Vec<(String, f32)> {
        let mut scores: Vec<(String, f32)> = self
            .vectors
            .iter()
            .filter(|(id, _)| allow(id))
            .map(|(id, v)| (id.clone(), crate::embed::dot(query, v)))
            .collect();
        scores.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scores
    }
    pub fn len(&self) -> usize {
        self.docs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tokenizes_mixed_text() {
        let t = tokens("部署 payments-svc 前要做什么？ROUND_HALF_UP tests");
        assert!(t.contains(&"部署".to_string()));
        assert!(t.contains(&"payment".to_string()));
        assert!(t.contains(&"round_half_up".to_string()));
        assert!(t.contains(&"half".to_string()));
        assert!(t.contains(&"test".to_string()));
        assert!(!t.contains(&"什么".to_string()));
        assert!(tokens("数据库迁移").contains(&"迁移".to_string()));
    }
    #[test]
    fn keyword_scores_are_normalized_and_ranked() {
        let mut index = Index::default();
        index.upsert("a", "部署 payments-svc 之前必须先执行数据库迁移");
        index.upsert("b", "Python 依赖用 uv 管理，不要用 pip install");
        index.upsert("c", "The staging database is read-only on Fridays");
        let scores = index.keyword_scores("部署前要做什么");
        assert_eq!(scores[0].0, "a");
        assert!(scores[0].1 > 0.5 && scores[0].1 <= 1.0);
        assert!(index.keyword_scores("今天天气").is_empty());
        index.remove("a");
        assert!(index.keyword_scores("部署").is_empty());
    }
}
