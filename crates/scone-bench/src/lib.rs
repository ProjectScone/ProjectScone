//! LongMemEval harness (spec §11, M5): ingest each item's haystack
//! sessions as conversation episodes, distill, answer via recall + the
//! configured LLM, score, and report accuracy / context reduction /
//! latency. LLM-agnostic: mechanics are tested with FakeLlm; real runs
//! use whatever `[llm]` config.toml provides.

use scone_core::{Engine, IngestInput, RecallOpts, auth};

pub struct BenchItem {
    pub question_id: String,
    pub question_type: String,
    pub question: String,
    pub answer: String,
    /// Each session flattened to one "role: content" transcript per turn.
    pub sessions: Vec<Vec<String>>,
}

pub struct ItemOutcome {
    pub question_id: String,
    pub correct: bool,
    pub model_answer: String,
    pub retrieved: String,
    pub stored_bytes: usize,
    pub retrieved_bytes: usize,
    pub recall_ms: f64,
}

#[derive(Default)]
pub struct Report {
    pub total: usize,
    pub correct: usize,
    pub stored_bytes: usize,
    pub retrieved_bytes: usize,
}

impl Report {
    pub fn add(&mut self, correct: bool, stored: usize, retrieved: usize) {
        self.total += 1;
        self.correct += usize::from(correct);
        self.stored_bytes += stored;
        self.retrieved_bytes += retrieved;
    }

    pub fn accuracy(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.correct as f64 / self.total as f64
    }

    /// Fraction of stored context NOT sent to the model (their headline
    /// metric shape: 99.4% context reduction).
    pub fn context_reduction(&self) -> f64 {
        if self.stored_bytes == 0 {
            return 0.0;
        }
        1.0 - (self.retrieved_bytes as f64 / self.stored_bytes as f64)
    }
}

pub fn parse_dataset(raw: &str) -> Result<Vec<BenchItem>, String> {
    let value: serde_json::Value = serde_json::from_str(raw).map_err(|e| e.to_string())?;
    let array = value.as_array().ok_or("dataset must be a JSON array")?;
    array
        .iter()
        .map(|item| {
            let s = |key: &str| {
                item.get(key)
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_owned()
            };
            let sessions = item
                .get("haystack_sessions")
                .and_then(|v| v.as_array())
                .ok_or("item missing haystack_sessions")?
                .iter()
                .map(|session| {
                    session
                        .as_array()
                        .map(|turns| {
                            turns
                                .iter()
                                .map(|turn| {
                                    format!(
                                        "{}: {}",
                                        turn.get("role").and_then(|r| r.as_str()).unwrap_or("user"),
                                        turn.get("content")
                                            .and_then(|c| c.as_str())
                                            .unwrap_or_default()
                                    )
                                })
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default()
                })
                .collect();
            Ok(BenchItem {
                question_id: s("question_id"),
                question_type: s("question_type"),
                question: s("question"),
                answer: s("answer"),
                sessions,
            })
        })
        .collect()
}

/// Substring scoring: deterministic, honest about its crudeness. The
/// official benchmark judges with an LLM; this under-counts paraphrased
/// answers and is labeled as such in reports.
pub fn substring_correct(expected: &str, got: &str) -> bool {
    let expected = expected.trim().to_lowercase();
    let got = got.to_lowercase();
    !expected.is_empty()
        && (got.contains(&expected)
            // Tolerate a leading article mismatch ("a gaggia classic").
            || expected
                .strip_prefix("a ")
                .or_else(|| expected.strip_prefix("the "))
                .is_some_and(|e| got.contains(e)))
}

/// Run one item in its own space (`item-<index>`); the engine accumulates
/// spaces but items never cross-contaminate.
pub fn run_item(
    engine: &mut Engine,
    item: &BenchItem,
    index: usize,
) -> Result<ItemOutcome, String> {
    let space = auth::resolve(engine, &format!("item-{index}"), true).map_err(|e| e.to_string())?;
    let mut stored_bytes = 0usize;
    for session in &item.sessions {
        let transcript = session.join("\n");
        stored_bytes += transcript.len();
        engine
            .ingest(&space, IngestInput::Note { text: transcript })
            .map_err(|e| e.to_string())?;
    }
    if engine.has_llm() {
        // Best-effort: extraction failures are recorded on the queue, and
        // episodic recall still answers (loud degradation, not abort).
        let _ = engine.distill(&space, 1_000);
    }
    let started = std::time::Instant::now();
    let pack = engine
        .recall(
            &space,
            &item.question,
            &RecallOpts {
                limit: 10,
                budget_bytes: None,
                as_of: None,
            },
        )
        .map_err(|e| e.to_string())?;
    let recall_ms = started.elapsed().as_secs_f64() * 1e3;
    let mut retrieved = String::new();
    for f in &pack.facts {
        retrieved.push_str(&format!(
            "- fact: {} {} {}\n",
            f.subject, f.predicate, f.object
        ));
    }
    for i in &pack.items {
        retrieved.push_str(&format!("- {}\n", i.text));
    }
    let model_answer = if engine.has_llm() {
        engine
            .llm_answer(&item.question, &retrieved)
            .unwrap_or_else(|e| format!("<llm error: {e}>"))
    } else {
        retrieved.clone()
    };
    Ok(ItemOutcome {
        question_id: item.question_id.clone(),
        correct: substring_correct(&item.answer, &model_answer),
        model_answer,
        retrieved_bytes: retrieved.len(),
        retrieved,
        stored_bytes,
        recall_ms,
    })
}
