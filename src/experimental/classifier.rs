use crate::store::{GateSample, Store};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, time::Instant};

const WIDTH: usize = 256;
#[derive(Clone, Serialize, Deserialize)]
pub struct Classifier {
    version: u32,
    weights: Vec<f64>,
    threshold: f64,
    trained_samples: usize,
    validation_precision: f64,
    gate_precision: f64,
    p95_ms: f64,
}
fn tokens(features: &Value) -> Vec<String> {
    let mut result = vec![];
    for field in ["text", "error_sig", "tool", "role"] {
        if let Some(value) = features.get(field).and_then(Value::as_str) {
            let lower = value.to_lowercase();
            result.extend(
                lower
                    .split(|ch: char| !ch.is_alphanumeric())
                    .filter(|word| word.chars().count() > 1)
                    .take(60)
                    .map(|word| format!("{field}:{word}")),
            );
            let chars = lower.chars().take(80).collect::<Vec<_>>();
            for pair in chars
                .windows(2)
                .filter(|pair| pair[0].is_alphabetic() && pair[1].is_alphabetic())
                .take(60)
            {
                result.push(format!("{field}:{}{}", pair[0], pair[1]));
            }
        }
    }
    result.sort();
    result.dedup();
    result
}
fn vector(features: &Value) -> Vec<usize> {
    let mut indices = tokens(features)
        .into_iter()
        .map(|token| {
            let digest = Sha256::digest(token.as_bytes());
            1 + (u16::from_be_bytes([digest[0], digest[1]]) as usize % (WIDTH - 1))
        })
        .collect::<Vec<_>>();
    indices.sort();
    indices.dedup();
    indices
}
fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value.clamp(-30.0, 30.0)).exp())
}
impl Classifier {
    fn probability(&self, features: &Value) -> f64 {
        let sum = vector(features)
            .into_iter()
            .fold(self.weights[0], |sum, index| sum + self.weights[index]);
        sigmoid(sum)
    }
    pub fn predict(&self, features: &Value) -> (bool, f64) {
        let probability = self.probability(features);
        (probability >= self.threshold, probability)
    }
    pub fn load(root: &Path) -> Option<Self> {
        let raw = fs::read(root.join("state/gate-classifier.json")).ok()?;
        let model: Self = serde_json::from_slice(&raw).ok()?;
        (model.version == 1 && model.weights.len() == WIDTH).then_some(model)
    }
}
fn precision(samples: &[GateSample], predictions: impl Iterator<Item = bool>) -> (f64, usize) {
    let mut selected = 0;
    let mut true_positive = 0;
    for (sample, predicted) in samples.iter().zip(predictions) {
        if predicted {
            selected += 1;
            if sample.positive {
                true_positive += 1;
            }
        }
    }
    (
        if selected > 0 {
            true_positive as f64 / selected as f64
        } else {
            0.0
        },
        selected,
    )
}
pub fn train(store: &Store) -> Result<Value> {
    let samples = store.gate_training_samples()?;
    let positives = samples.iter().filter(|s| s.positive).count();
    if samples.len() < 2000 || positives < 100 || samples.len() - positives < 100 {
        return Ok(
            json!({"trained":false,"reason":"need at least 2000 labeled gate samples with 100 positive and 100 negative","samples":samples.len(),"positives":positives}),
        );
    }
    let split = samples.len() * 4 / 5;
    let (training, validation) = samples.split_at(split);
    let mut model = Classifier {
        version: 1,
        weights: vec![0.0; WIDTH],
        threshold: 0.5,
        trained_samples: samples.len(),
        validation_precision: 0.0,
        gate_precision: 0.0,
        p95_ms: 0.0,
    };
    for epoch in 0..8 {
        let rate = 0.08 / (1.0 + epoch as f64 * 0.3);
        for sample in training {
            let active = vector(&sample.features);
            let sum = active
                .iter()
                .fold(model.weights[0], |sum, index| sum + model.weights[*index]);
            let error = (if sample.positive { 1.0 } else { 0.0 }) - sigmoid(sum);
            model.weights[0] += rate * error;
            for index in active {
                model.weights[index] += rate * (error - 0.001 * model.weights[index]);
            }
        }
    }
    let (gate_precision, gate_count) =
        precision(validation, validation.iter().map(|s| s.gate_recalled));
    let mut chosen = None;
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] {
        model.threshold = threshold;
        let (p, count) = precision(
            validation,
            validation.iter().map(|s| model.predict(&s.features).0),
        );
        if count >= 10
            && p >= gate_precision
            && chosen.as_ref().is_none_or(|(_, old_p, old_count)| {
                p > *old_p || (p == *old_p && count > *old_count)
            })
        {
            chosen = Some((threshold, p, count));
        }
    }
    let Some((threshold, validated, selected)) = chosen else {
        return Ok(
            json!({"trained":false,"reason":"classifier precision below gate on holdout","samples":samples.len(),"gate_precision":gate_precision,"gate_selected":gate_count}),
        );
    };
    model.threshold = threshold;
    model.validation_precision = validated;
    model.gate_precision = gate_precision;
    let mut times = Vec::new();
    for sample in validation.iter().cycle().take(1000) {
        let start = Instant::now();
        let _ = model.predict(&sample.features);
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    model.p95_ms = times[949];
    if model.p95_ms >= 10.0 {
        return Ok(json!({"trained":false,"reason":"classifier too slow","p95_ms":model.p95_ms}));
    }
    let path = store.root.join("state/gate-classifier.json");
    let temp = path.with_extension("tmp");
    fs::write(&temp, serde_json::to_vec_pretty(&model)?)?;
    fs::rename(temp, path)?;
    Ok(
        json!({"trained":true,"samples":samples.len(),"validation_precision":validated,"gate_precision":gate_precision,"selected":selected,"p95_ms":model.p95_ms}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hashed_features_are_stable() {
        let features = json!({"text":"pytest database failed","role":"user"});
        assert_eq!(vector(&features), vector(&features));
        assert!(!vector(&features).is_empty());
    }
    #[test]
    fn refuses_unlabeled_training() {
        let dir = tempfile::tempdir().unwrap();
        crate::store::init(dir.path()).unwrap();
        let store = Store::open(dir.path()).unwrap();
        let result = train(&store).unwrap();
        assert_eq!(result["trained"], false);
        assert!(Classifier::load(dir.path()).is_none());
    }
}
