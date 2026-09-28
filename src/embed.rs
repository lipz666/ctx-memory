//! Local sentence embeddings (fastembed / ONNX Runtime, CPU). No network at query time;
//! the model is downloaded once into a cache shared by all ctx instances.
use anyhow::{Context, Result, bail};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
};

pub struct Embedder {
    /// Independent model instances; a call uses whichever is free.
    models: Vec<Mutex<TextEmbedding>>,
    next: std::sync::atomic::AtomicUsize,
    pub name: String,
    style: Style,
}
#[derive(Clone, Copy)]
enum Style {
    Gemma,
    E5,
    Plain,
}

fn model_for(name: &str) -> Option<(EmbeddingModel, Style)> {
    Some(match name {
        "embeddinggemma-300m-q" => (EmbeddingModel::EmbeddingGemma300MQ, Style::Gemma),
        "embeddinggemma-300m" => (EmbeddingModel::EmbeddingGemma300M, Style::Gemma),
        "multilingual-e5-small" => (EmbeddingModel::MultilingualE5Small, Style::E5),
        "multilingual-e5-base" => (EmbeddingModel::MultilingualE5Base, Style::E5),
        "bge-m3" => (EmbeddingModel::BGEM3, Style::Plain),
        _ => return None,
    })
}

/// `CTX_MODEL_DIR`, else `~/.cache/ctx/models`.
pub fn model_dir() -> PathBuf {
    std::env::var_os("CTX_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/ctx/models")
        })
}

impl Embedder {
    /// Loads (and on first use downloads) the model `workers` times. Takes seconds; call
    /// off the hot path.
    pub fn load(name: &str, workers: usize) -> Result<Self> {
        let Some((model, style)) = model_for(name) else {
            bail!("unknown embedding model {name}");
        };
        let dir = model_dir();
        std::fs::create_dir_all(&dir)?;
        let models = (0..workers.clamp(1, 32))
            .map(|_| {
                TextEmbedding::try_new(
                    TextInitOptions::new(model.clone())
                        .with_cache_dir(dir.clone())
                        .with_show_download_progress(false),
                )
                .map(Mutex::new)
                .with_context(|| format!("loading embedding model {name}"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            models,
            next: Default::default(),
            name: name.into(),
            style,
        })
    }
    pub fn embed_documents(&self, docs: &[(String, String)]) -> Result<Vec<Vec<f32>>> {
        let texts = docs
            .iter()
            .map(|(title, body)| {
                let body: String = body.chars().take(2000).collect();
                match self.style {
                    Style::Gemma => format!("title: {title} | text: {body}"),
                    Style::E5 => format!("passage: {title}\n{body}"),
                    Style::Plain => format!("{title}\n{body}"),
                }
            })
            .collect::<Vec<_>>();
        self.run(&texts)
    }
    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let query: String = query.chars().take(1500).collect();
        let text = match self.style {
            Style::Gemma => format!("task: search result | query: {query}"),
            Style::E5 => format!("query: {query}"),
            Style::Plain => query,
        };
        Ok(self.run(&[text])?.remove(0))
    }
    fn run(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let start = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut model = (0..self.models.len())
            .find_map(|i| self.models[(start + i) % self.models.len()].try_lock().ok())
            .unwrap_or_else(|| self.models[start % self.models.len()].lock().unwrap());
        // One text per call: dynamically quantized models scale each batch differently.
        let mut vectors = Vec::with_capacity(texts.len());
        for text in texts {
            vectors.extend(model.embed([text], None)?);
        }
        for vector in &mut vectors {
            let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                vector.iter_mut().for_each(|x| *x /= norm);
            }
        }
        Ok(vectors)
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// A lazily loaded embedder: `None` until loaded, and stays `None` if loading failed
/// (recall then uses keyword search only).
#[derive(Default)]
pub struct Slot {
    cell: OnceLock<Option<Arc<Embedder>>>,
    loading: Mutex<()>,
}
impl Slot {
    pub fn get(&self) -> Option<Arc<Embedder>> {
        self.cell.get().cloned().flatten()
    }
    pub fn settled(&self) -> bool {
        self.cell.get().is_some()
    }
    /// Load once; later calls return the same result.
    pub fn load(&self, name: &str, workers: usize) -> Option<Arc<Embedder>> {
        let _guard = self.loading.lock().unwrap();
        self.cell
            .get_or_init(|| match Embedder::load(name, workers) {
                Ok(model) => Some(Arc::new(model)),
                Err(error) => {
                    eprintln!("ctx: embeddings unavailable, keyword recall only: {error:#}");
                    None
                }
            })
            .clone()
    }
    pub fn disable(&self) {
        let _ = self.cell.set(None);
    }
}
