//! LongMemEval harness (spec §11, M5): ingest each item's haystack
//! sessions as conversation episodes, distill, answer via recall + the
//! configured LLM, score, and report accuracy / context reduction /
//! latency. LLM-agnostic: mechanics are tested with FakeLlm; real runs
//! use whatever `[llm]` config.toml provides.

use scone_core::{Engine, RecallOpts, auth};

pub struct BenchItem {
    pub question_id: String,
    pub question_type: String,
    pub question: String,
    pub answer: String,
    /// Each session flattened to one "role: content" transcript per turn.
    pub sessions: Vec<Vec<String>>,
    /// ISO-8601 timestamp per session (from haystack_dates), parallel to
    /// `sessions`; empty string when the dataset lacks one.
    pub session_dates: Vec<String>,
    /// Session ids parallel to `sessions` (ground truth for Recall@k).
    pub session_ids: Vec<String>,
    /// The evidence sessions the answer lives in.
    pub answer_session_ids: Vec<String>,
}

pub struct ItemOutcome {
    pub question_id: String,
    pub correct: bool,
    /// Session ids of retrieved items, best-first (for Recall@k).
    pub retrieved_sessions: Vec<String>,
    /// Ground truth evidence sessions.
    pub answer_sessions: Vec<String>,
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

impl ItemOutcome {
    /// Any evidence session appears in the top-k retrieved items.
    pub fn recall_any_at(&self, k: usize) -> bool {
        let seen: std::collections::HashSet<&str> = self
            .retrieved_sessions
            .iter()
            .take(k)
            .map(String::as_str)
            .collect();
        self.answer_sessions
            .iter()
            .any(|a| seen.contains(a.as_str()))
    }

    /// Every evidence session appears in the top-k retrieved items.
    pub fn recall_all_at(&self, k: usize) -> bool {
        let seen: std::collections::HashSet<&str> = self
            .retrieved_sessions
            .iter()
            .take(k)
            .map(String::as_str)
            .collect();
        !self.answer_sessions.is_empty()
            && self
                .answer_sessions
                .iter()
                .all(|a| seen.contains(a.as_str()))
    }
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

/// LongMemEval dates look like "2023/05/20 (Sat) 02:21"; normalize to
/// ISO-8601 so SQLite's julianday() can subtract them.
fn iso_date(raw: &str) -> String {
    let raw = raw.trim();
    if raw.len() < 10 {
        return String::new();
    }
    let date = raw[..10].replace('/', "-");
    let time = raw
        .rsplit(' ')
        .next()
        .filter(|t| t.contains(':'))
        .unwrap_or("00:00");
    format!("{date}T{time}:00Z")
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
            let session_dates = item
                .get("haystack_dates")
                .and_then(|v| v.as_array())
                .map(|dates| {
                    dates
                        .iter()
                        .map(|d| iso_date(d.as_str().unwrap_or_default()))
                        .collect()
                })
                .unwrap_or_default();
            let string_list = |key: &str| -> Vec<String> {
                item.get(key)
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(str::to_owned))
                            .collect()
                    })
                    .unwrap_or_default()
            };
            Ok(BenchItem {
                question_id: s("question_id"),
                question_type: s("question_type"),
                question: s("question"),
                answer: s("answer"),
                sessions,
                session_dates,
                session_ids: string_list("haystack_session_ids"),
                answer_session_ids: string_list("answer_session_ids"),
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
#[derive(Clone, Copy)]
pub enum Granularity {
    /// One episode per session (default).
    Session,
    /// One episode per conversation turn — finer retrieval targets.
    Turn,
}

pub fn run_item(
    engine: &mut Engine,
    item: &BenchItem,
    index: usize,
) -> Result<ItemOutcome, String> {
    run_item_with(engine, item, index, true, Granularity::Session)
}

/// `distill = false` isolates retrieval + answering from extraction
/// quality (the answer model reads raw episodic recall only).
pub fn run_item_with(
    engine: &mut Engine,
    item: &BenchItem,
    index: usize,
    distill: bool,
    granularity: Granularity,
) -> Result<ItemOutcome, String> {
    let space = auth::resolve(engine, &format!("item-{index}"), true).map_err(|e| e.to_string())?;
    let mut stored_bytes = 0usize;
    for (s_idx, session) in item.sessions.iter().enumerate() {
        match granularity {
            Granularity::Session => {
                let transcript = session.join("\n");
                stored_bytes += transcript.len();
                let date = item
                    .session_dates
                    .get(s_idx)
                    .filter(|d| !d.is_empty())
                    .map(String::as_str);
                // source carries the session id: ground truth for Recall@k.
                let source = item.session_ids.get(s_idx).map(String::as_str);
                match engine.import_episode(&space, "conversation", &transcript, source, date) {
                    Ok(_) => {}
                    // Real datasets contain empty sessions; skip, not abort.
                    Err(scone_core::SconeError::InvalidInput(_)) => {}
                    Err(e) => return Err(e.to_string()),
                }
            }
            Granularity::Turn => {
                let source = item.session_ids.get(s_idx).map(String::as_str);
                for turn in session {
                    stored_bytes += turn.len();
                    match engine.import_episode(&space, "conversation", turn, source, None) {
                        Ok(_) => {}
                        // Repeated short turns ("user: thanks!") dedup or
                        // reject as empty; both are fine at turn scale.
                        Err(scone_core::SconeError::InvalidInput(_)) => {}
                        Err(e) => return Err(e.to_string()),
                    }
                }
            }
        }
    }
    if engine.has_llm() && distill {
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
                limit: 15,
                budget_bytes: None,
                as_of: None,
                expand_neighbors: false,
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
    let retrieved_sessions: Vec<String> =
        pack.items.iter().filter_map(|i| i.source.clone()).collect();
    Ok(ItemOutcome {
        question_id: item.question_id.clone(),
        correct: substring_correct(&item.answer, &model_answer),
        retrieved_sessions,
        answer_sessions: item.answer_session_ids.clone(),
        model_answer,
        retrieved_bytes: retrieved.len(),
        retrieved,
        stored_bytes,
        recall_ms,
    })
}

/// LLM-judge scoring: asks the model whether the produced answer states
/// the expected one. First token YES/NO; anything else counts as NO —
/// conservative, never inflating (memory/benchmarks.md keeps both scores).
pub fn judge_correct(
    llm: &dyn scone_core::llm::LlmProvider,
    question: &str,
    expected: &str,
    got: &str,
) -> Result<bool, String> {
    let verdict = llm
        .answer(
            "You are grading a memory benchmark. Reply with exactly YES or NO: \
             does the candidate answer state the same fact as the reference answer?",
            &format!("Question: {question}\nReference answer: {expected}\nCandidate answer: {got}"),
        )
        .map_err(|e| e.to_string())?;
    Ok(verdict.trim_start().to_uppercase().starts_with("YES"))
}
