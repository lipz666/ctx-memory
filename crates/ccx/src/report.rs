//! `ccx report`: per-request size and token composition from `steps.jsonl`.
//! Category estimates are scaled so that, per request, they sum to the input tokens the
//! upstream reported (when it reported them).
use anyhow::Result;
use serde_json::{Value, json};
use std::{collections::BTreeSet, path::Path};

const CATEGORIES: [&str; 8] = [
    "system",
    "tools",
    "user",
    "assistant",
    "tool_calls",
    "tool_results",
    "reasoning",
    "other",
];

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn stats(mut values: Vec<f64>) -> Value {
    values.sort_by(f64::total_cmp);
    let sum: f64 = values.iter().sum();
    json!({
        "mean": if values.is_empty() { 0.0 } else { (sum / values.len() as f64).round() },
        "p50": percentile(&values, 0.5).round(),
        "p95": percentile(&values, 0.95).round(),
        "max": values.last().copied().unwrap_or(0.0).round(),
        "sum": sum.round(),
    })
}

pub fn summarize(lines: &[Value]) -> Value {
    let mut totals = vec![];
    let mut dynamic = vec![];
    let mut statics = vec![];
    let mut reported = 0u64;
    let mut output = 0u64;
    let mut cached = 0u64;
    let mut reasoning = 0u64;
    let mut shares = [0f64; CATEGORIES.len()];
    let mut sessions = BTreeSet::new();
    let mut errors = 0;
    let mut original = vec![];
    let mut shortened = std::collections::BTreeMap::<&str, u64>::new();
    let mut expand_rounds = 0;
    let mut expand_calls = 0;
    for line in lines {
        if line["status"].as_u64().is_some_and(|s| s >= 400) {
            errors += 1;
            continue;
        }
        // In `on` mode the upstream saw the rewritten request.
        let est = if line["est_sent"].is_object() {
            &line["est_sent"]
        } else {
            &line["est"]
        };
        original.push(line["est"]["total"].as_f64().unwrap_or(0.0));
        for key in ["hot_shortened", "warm_digested", "duplicates"] {
            *shortened.entry(key).or_insert(0) += line["ccx"][key].as_u64().unwrap_or(0);
        }
        expand_rounds += line["expand_rounds"].as_u64().unwrap_or(0);
        expand_calls += line["expand_calls"].as_u64().unwrap_or(0);
        let est_total = est["total"].as_f64().unwrap_or(0.0).max(1.0);
        let input = line["usage"]["input"].as_f64();
        let scale = input.map_or(1.0, |i| i / est_total);
        let total = input.unwrap_or(est_total);
        reported += input.map_or(0, |i| i as u64);
        output += line["usage"]["output"].as_u64().unwrap_or(0);
        cached += line["usage"]["cached"].as_u64().unwrap_or(0);
        reasoning += line["usage"]["reasoning"].as_u64().unwrap_or(0);
        for (i, key) in CATEGORIES.iter().enumerate() {
            shares[i] += est[*key].as_f64().unwrap_or(0.0) * scale;
        }
        totals.push(total);
        statics.push(est["static"].as_f64().unwrap_or(0.0) * scale);
        dynamic.push(est["dynamic"].as_f64().unwrap_or(0.0) * scale);
        if let Some(s) = line["session"].as_str() {
            sessions.insert(s.to_owned());
        }
    }
    let all: f64 = shares.iter().sum::<f64>().max(1.0);
    let composition: serde_json::Map<String, Value> = CATEGORIES
        .iter()
        .zip(shares)
        .map(|(k, v)| (k.to_string(), json!((v / all * 1000.0).round() / 10.0)))
        .collect();
    json!({
        "requests": totals.len(),
        "errors": errors,
        "sessions": sessions.len(),
        "input_tokens_reported": reported,
        "output_tokens_reported": output,
        "cached_tokens_reported": cached,
        "reasoning_tokens_reported": reasoning,
        "per_request_input": stats(totals),
        "per_request_static": stats(statics),
        "per_request_dynamic": stats(dynamic),
        "composition_percent": composition,
        "per_request_original_est": stats(original),
        "shortened": shortened,
        "expand_rounds": expand_rounds,
        "expand_calls": expand_calls,
    })
}

pub fn run(path: &Path, tag: Option<&str>, as_json: bool) -> Result<()> {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let lines: Vec<Value> = text
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter(|l| tag.is_none_or(|t| l["tag"].as_str() == Some(t)))
        .collect();
    let summary = summarize(&lines);
    if as_json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
        return Ok(());
    }
    println!(
        "requests {}  errors {}  sessions {}",
        summary["requests"], summary["errors"], summary["sessions"]
    );
    println!(
        "input tokens {}  output {}  reasoning {}  cached {}",
        summary["input_tokens_reported"],
        summary["output_tokens_reported"],
        summary["reasoning_tokens_reported"],
        summary["cached_tokens_reported"]
    );
    for key in [
        "per_request_input",
        "per_request_static",
        "per_request_dynamic",
    ] {
        let s = &summary[key];
        println!(
            "{key:<20} mean {:>8}  p50 {:>8}  p95 {:>8}  max {:>8}",
            s["mean"], s["p50"], s["p95"], s["max"]
        );
    }
    let o = &summary["per_request_original_est"];
    println!(
        "{:<20} mean {:>8}  p50 {:>8}  p95 {:>8}  max {:>8}  (estimate, before ccx)",
        "per_request_original", o["mean"], o["p50"], o["p95"], o["max"]
    );
    println!(
        "shortened {}  expand rounds {}  calls {}",
        summary["shortened"], summary["expand_rounds"], summary["expand_calls"]
    );
    let parts: Vec<String> = CATEGORIES
        .iter()
        .map(|k| format!("{k} {}%", summary["composition_percent"][k]))
        .collect();
    println!("composition  {}", parts.join("  "));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scales_estimates_to_reported_input() {
        let line = |input: Option<u64>, status: u64| {
            json!({
                "status": status, "session": "a",
                "usage": {"input": input, "output": 5},
                "est": {"system": 10, "tools": 10, "tool_results": 80, "static": 20,
                        "dynamic": 80, "total": 100},
            })
        };
        let s = summarize(&[line(Some(200), 200), line(None, 200), line(Some(1), 500)]);
        assert_eq!(s["requests"], 2);
        assert_eq!(s["errors"], 1);
        assert_eq!(s["input_tokens_reported"], 200);
        assert_eq!(s["per_request_input"]["max"], 200.0);
        assert_eq!(s["per_request_dynamic"]["max"], 160.0);
        assert_eq!(s["composition_percent"]["tool_results"], 80.0);
    }
}
