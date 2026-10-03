//! Deterministic shortening of tool output: the same text always gives the same bytes, so
//! a shortened request stays byte-stable from one step to the next (prompt caches).
use crate::composition::estimate;

const MAX_LINE_CHARS: usize = 400;

fn is_signal(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    [
        "error",
        "exception",
        "traceback",
        "fail",
        "warning",
        "denied",
        "not found",
        "crit",
        "fatal",
        "panic",
        "oom",
        "killed",
        "exited",
        "abort",
    ]
    .iter()
    .any(|w| lower.contains(w))
}

fn clip_line(line: &str) -> String {
    if line.chars().count() <= MAX_LINE_CHARS {
        return line.to_owned();
    }
    let head: String = line.chars().take(MAX_LINE_CHARS).collect();
    format!("{head}…")
}

/// Text cut to about `cap` tokens: leading lines (60%), error-looking lines from the middle
/// (15%), trailing lines (25%), with a marker where lines were left out. Text with a few
/// very long lines is cut by characters instead.
pub fn excerpt(text: &str, cap: u64) -> String {
    if estimate(text) <= cap {
        return text.to_owned();
    }
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 8 {
        return char_excerpt(text, cap);
    }
    let take = |budget: u64, indexes: &mut dyn Iterator<Item = usize>| {
        let mut used = 0;
        let mut picked = vec![];
        for i in indexes {
            let cost = estimate(&clip_line(lines[i])) + 1;
            if used + cost > budget {
                break;
            }
            used += cost;
            picked.push(i);
        }
        picked
    };
    let head = take(cap * 60 / 100, &mut (0..lines.len()));
    let start = head.last().map_or(0, |i| i + 1);
    let mut tail = take(cap * 25 / 100, &mut (start..lines.len()).rev());
    tail.reverse();
    let end = tail.first().copied().unwrap_or(lines.len());
    let middle = take(
        cap * 15 / 100,
        &mut (start..end).filter(|&i| is_signal(lines[i])),
    );
    let mut keep: Vec<usize> = head.into_iter().chain(middle).chain(tail).collect();
    keep.sort_unstable();
    keep.dedup();
    let mut out = String::new();
    let mut previous: Option<usize> = None;
    for i in keep {
        let gap = previous.map_or(i, |p| i - p - 1);
        if gap > 0 {
            out.push_str(&format!("[… {gap} lines omitted …]\n"));
        }
        out.push_str(&clip_line(lines[i]));
        out.push('\n');
        previous = Some(i);
    }
    let rest = lines.len() - previous.map_or(0, |p| p + 1);
    if rest > 0 {
        out.push_str(&format!("[… {rest} lines omitted …]\n"));
    }
    out
}

fn char_excerpt(text: &str, cap: u64) -> String {
    // estimate() counts 4 ASCII characters per token; non-ASCII text is cut more tightly.
    let chars: Vec<char> = text.chars().collect();
    let budget = (cap as usize * 4).min(chars.len());
    let head = budget * 70 / 100;
    let tail = budget - head;
    let omitted = chars.len() - head - tail;
    let head: String = chars[..head].iter().collect();
    let tail: String = chars[chars.len() - tail..].iter().collect();
    format!("{head}\n[… {omitted} characters omitted …]\n{tail}")
}

pub struct Labels<'a> {
    pub tool: &'a str,
    pub id: &'a str,
    /// Whether the model can call `ccx_expand` in this request.
    pub expand: bool,
}

fn how_to_read(labels: &Labels) -> String {
    if labels.expand {
        format!(
            "full text: ccx_expand(id=\"{}\", query=… or offset=…)",
            labels.id
        )
    } else {
        format!("stored as {}; re-run the tool for details", labels.id)
    }
}

/// Latest step's output that is over the hot cap: a large excerpt plus how to read the rest.
pub fn hot(text: &str, cap: u64, labels: &Labels) -> String {
    format!(
        "{}[ccx: {} output shortened from ~{} tokens, {} lines; {}]",
        excerpt(text, cap),
        labels.tool,
        estimate(text),
        text.lines().count(),
        how_to_read(labels)
    )
}

/// Older output: a short digest.
pub fn warm(text: &str, cap: u64, labels: &Labels) -> String {
    format!(
        "[ccx digest: {} output, ~{} tokens, {} lines; {}]\n{}",
        labels.tool,
        estimate(text),
        text.lines().count(),
        how_to_read(labels),
        excerpt(text, cap)
    )
}

/// What the model already read back from a shortened output in this session.
pub fn pinned(read: &str) -> String {
    format!("\n[ccx: you read this part with ccx_expand earlier]\n{read}")
}

/// Older output repeated verbatim by a later step.
pub fn duplicate(labels: &Labels) -> String {
    format!(
        "[ccx: same {} output as a later step ({})]",
        labels.tool, labels.id
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_unchanged() {
        assert_eq!(excerpt("a\nb", 100), "a\nb");
    }

    #[test]
    fn long_text_keeps_head_tail_and_errors() {
        let mut lines: Vec<String> = (0..400).map(|i| format!("row {i} value")).collect();
        lines[200] = "Traceback: KeyError 'x'".into();
        let text = lines.join("\n");
        let out = excerpt(&text, 200);
        assert!(out.starts_with("row 0 value\n"));
        assert!(out.contains("Traceback: KeyError"));
        assert!(out.contains("row 399 value"));
        assert!(out.contains("lines omitted"));
        assert!(estimate(&out) <= 260, "{}", estimate(&out));
        assert_eq!(out, excerpt(&text, 200), "deterministic");
    }

    #[test]
    fn one_long_line_is_cut_by_characters() {
        let text = "x".repeat(10_000);
        let out = excerpt(&text, 100);
        assert!(out.contains("characters omitted"));
        assert!(estimate(&out) < 130);
    }

    #[test]
    fn labels_say_how_to_read_more() {
        let l = Labels {
            tool: "python",
            id: "x1",
            expand: true,
        };
        assert!(warm(&"line\n".repeat(500), 50, &l).contains("ccx_expand(id=\"x1\""));
        let l = Labels { expand: false, ..l };
        assert!(hot(&"line\n".repeat(500), 50, &l).contains("re-run the tool"));
    }
}
