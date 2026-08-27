//! LLM providers (spec §9): the semantic lane's only model dependency.
//!
//! The engine works with `None` — lane 2 pauses loudly, episodic search
//! stays at full strength (spec §6, memory/lessons.md L-9).

use crate::error::{Result, SconeError};

/// One fact proposed by extraction, before entity resolution.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f32,
}

pub trait LlmProvider {
    fn id(&self) -> &str;
    fn extract_facts(&self, text: &str) -> Result<Vec<ExtractedFact>>;
    fn answer(&self, question: &str, context: &str) -> Result<String>;
}

/// Deterministic in-process provider for tests: returns programmed facts
/// and records every call.
pub struct FakeLlm {
    facts: Vec<ExtractedFact>,
    fail: Option<String>,
    calls: std::cell::RefCell<Vec<String>>,
}

impl FakeLlm {
    pub fn new(facts: Vec<ExtractedFact>) -> Self {
        Self {
            facts,
            fail: None,
            calls: std::cell::RefCell::new(Vec::new()),
        }
    }

    pub fn failing(message: &str) -> Self {
        Self {
            facts: Vec::new(),
            fail: Some(message.to_owned()),
            calls: std::cell::RefCell::new(Vec::new()),
        }
    }

    pub fn calls(&self) -> Vec<String> {
        self.calls.borrow().clone()
    }
}

impl LlmProvider for FakeLlm {
    fn id(&self) -> &str {
        "fake"
    }

    fn extract_facts(&self, text: &str) -> Result<Vec<ExtractedFact>> {
        self.calls.borrow_mut().push(text.to_owned());
        match &self.fail {
            Some(msg) => Err(SconeError::Llm(msg.clone())),
            None => Ok(self.facts.clone()),
        }
    }

    fn answer(&self, question: &str, context: &str) -> Result<String> {
        match &self.fail {
            Some(msg) => Err(SconeError::Llm(msg.clone())),
            None => Ok(format!(
                "answer to {question} given {} bytes",
                context.len()
            )),
        }
    }
}
