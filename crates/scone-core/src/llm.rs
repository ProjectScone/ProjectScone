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

/// Default system prompt for answering from memory (v1).
pub const ANSWER_SYSTEM_V1: &str = "Answer from the provided memory context. \
Cite nothing you cannot find there; say so when the context lacks the answer.";

/// Extraction-style answering for small readers (v2): short, literal,
/// time-aware. Benchmarked against v1 in memory/benchmarks.md.
pub const ANSWER_SYSTEM_V2: &str = "You answer questions from retrieved \
personal memory. Reply with ONLY the specific fact or detail asked for - a \
short phrase, no preamble, no explanation. Prefer the exact wording found in \
the context. If the question refers to time ('first', 'last', 'in May'), use \
the timestamps and ordering in the context to pick the right instance. If the \
context does not contain the answer, reply exactly: unknown";

/// Evidence-chaining answering (v3): the reader lists the dated context
/// lines it relies on before committing to a final ANSWER: line. Aims at
/// multi-session and temporal synthesis, where small readers lose the
/// thread (E15/E16). Benchmarked as E19.
pub const ANSWER_SYSTEM_V3: &str = "You answer questions from retrieved \
personal memory. Work in two steps, in one reply. Step 1: copy the 2 to 4 \
context lines that bear on the question, each on its own line starting \
EVIDENCE:, keeping their [timestamps]. Step 2: end with one line starting \
ANSWER: followed by the specific fact or detail asked for, short, in the \
context's own wording. When the question involves time ('first', 'last', \
'before', 'in May'), order the evidence timestamps and pick accordingly; \
when facts conflict, the latest timestamp wins. If the evidence does not \
contain the answer, end with exactly: ANSWER: unknown";

/// Pass-1 prompt for the two-pass reader (E20): pull the relevant
/// evidence out of the noisy pack; pass 2 answers from only that.
pub const TWO_PASS_EXTRACT_SYSTEM: &str = "From the retrieved memory \
context, copy every line that could bear on the question, verbatim with \
its [timestamp], one per line. No commentary, no answer. If nothing \
bears on it, reply exactly: NO EVIDENCE";

pub trait LlmProvider: Send {
    fn id(&self) -> &str;
    fn extract_facts(&self, text: &str) -> Result<Vec<ExtractedFact>>;
    /// Answer with an explicit system prompt.
    fn answer_with_system(&self, system: &str, question: &str, context: &str) -> Result<String>;
    /// Answer with the default (v1) system prompt.
    fn answer(&self, question: &str, context: &str) -> Result<String> {
        self.answer_with_system(ANSWER_SYSTEM_V1, question, context)
    }
}

/// Deterministic in-process provider for tests: returns programmed facts
/// and records every call.
pub struct FakeLlm {
    facts: Vec<ExtractedFact>,
    fail: Option<String>,
    answer: Option<String>,
    calls: std::cell::RefCell<Vec<String>>,
}

impl FakeLlm {
    pub fn new(facts: Vec<ExtractedFact>) -> Self {
        Self {
            facts,
            fail: None,
            answer: None,
            calls: std::cell::RefCell::new(Vec::new()),
        }
    }

    /// Program the exact string `answer` returns.
    pub fn with_answer(mut self, answer: &str) -> Self {
        self.answer = Some(answer.to_owned());
        self
    }

    pub fn failing(message: &str) -> Self {
        Self {
            facts: Vec::new(),
            fail: Some(message.to_owned()),
            answer: None,
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

    fn answer_with_system(&self, _system: &str, question: &str, context: &str) -> Result<String> {
        match (&self.fail, &self.answer) {
            (Some(msg), _) => Err(SconeError::Llm(msg.clone())),
            (None, Some(programmed)) => Ok(programmed.clone()),
            (None, None) => Ok(format!(
                "answer to {question} given {} bytes",
                context.len()
            )),
        }
    }
}

const EXTRACTION_PROMPT: &str = "Extract durable factual statements from the text as a STRICT \
JSON array. Each element: {\"subject\": string, \"predicate\": string, \"object\": string, \
\"confidence\": number 0..1}. Subjects are entities (people, projects, tools, places). \
Predicates are short verb phrases. Only facts stated or strongly implied; no speculation. \
Reply with the JSON array ONLY — no prose, no code fences.";

fn parse_extraction(content: &str) -> Result<Vec<ExtractedFact>> {
    let trimmed = content
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    let value: serde_json::Value = serde_json::from_str(trimmed).map_err(|e| {
        SconeError::Llm(format!(
            "model did not return JSON: {e}; got: {}",
            content.chars().take(120).collect::<String>()
        ))
    })?;
    let array = value
        .as_array()
        .ok_or_else(|| SconeError::Llm("model did not return JSON array".into()))?;
    array
        .iter()
        .map(|f| {
            Ok(ExtractedFact {
                subject: f["subject"]
                    .as_str()
                    .ok_or_else(|| SconeError::Llm("fact missing subject".into()))?
                    .to_owned(),
                predicate: f["predicate"]
                    .as_str()
                    .ok_or_else(|| SconeError::Llm("fact missing predicate".into()))?
                    .to_owned(),
                object: f["object"]
                    .as_str()
                    .ok_or_else(|| SconeError::Llm("fact missing object".into()))?
                    .to_owned(),
                confidence: f["confidence"].as_f64().unwrap_or(0.5) as f32,
            })
        })
        .collect()
}

/// Default ceiling for one model call. A hung provider must become a typed
/// error, never a stuck process (the missing-timeout class already cost us
/// a stalled sweep and an unobservable bench leg, 2026-08-27/28).
pub const DEFAULT_LLM_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(180);

fn http_json(
    req: ureq::RequestBuilder<ureq::typestate::WithBody>,
    timeout: std::time::Duration,
    body: serde_json::Value,
) -> Result<serde_json::Value> {
    let mut res = req
        .config()
        .timeout_global(Some(timeout))
        .build()
        .send_json(body)
        .map_err(|e| SconeError::Llm(format!("http: {e}")))?;
    res.body_mut()
        .read_json()
        .map_err(|e| SconeError::Llm(format!("http body: {e}")))
}

/// Any OpenAI-compatible chat endpoint: OpenAI itself, Ollama, vLLM, …
pub struct OpenAiCompatible {
    base_url: String,
    model: String,
    api_key: Option<String>,
    timeout: std::time::Duration,
    think: Option<bool>,
}

impl OpenAiCompatible {
    pub fn new(base_url: &str, model: &str, api_key: Option<String>) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            model: model.to_owned(),
            api_key,
            timeout: DEFAULT_LLM_TIMEOUT,
            think: None,
        }
    }

    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Control thinking on reasoning models (Ollama passes `think`
    /// through; real OpenAI endpoints reject unknown fields, so the
    /// field is only serialized when explicitly set).
    pub fn with_think(mut self, think: bool) -> Self {
        self.think = Some(think);
        self
    }

    fn chat(&self, system: &str, user: &str) -> Result<String> {
        let mut req = ureq::post(format!("{}/chat/completions", self.base_url));
        if let Some(key) = &self.api_key {
            req = req.header("authorization", format!("Bearer {key}"));
        }
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        });
        if let Some(think) = self.think {
            body["think"] = serde_json::Value::Bool(think);
        }
        let value = http_json(req, self.timeout, body)?;
        value["choices"][0]["message"]["content"]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| SconeError::Llm("no content in chat response".into()))
    }
}

impl LlmProvider for OpenAiCompatible {
    fn id(&self) -> &str {
        &self.model
    }

    fn extract_facts(&self, text: &str) -> Result<Vec<ExtractedFact>> {
        parse_extraction(&self.chat(EXTRACTION_PROMPT, text)?)
    }

    fn answer_with_system(&self, system: &str, question: &str, context: &str) -> Result<String> {
        self.chat(
            system,
            &format!("Context:\n{context}\n\nQuestion: {question}"),
        )
    }
}

/// Anthropic's native messages API.
pub struct AnthropicProvider {
    base_url: String,
    model: String,
    api_key: String,
    timeout: std::time::Duration,
}

impl AnthropicProvider {
    pub fn new(base_url: &str, model: &str, api_key: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            model: model.to_owned(),
            api_key: api_key.to_owned(),
            timeout: DEFAULT_LLM_TIMEOUT,
        }
    }

    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = timeout;
        self
    }

    fn message(&self, system: &str, user: &str) -> Result<String> {
        let req = ureq::post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", self.api_key.as_str())
            .header("anthropic-version", "2023-06-01");
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": 1024,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        });
        let value = http_json(req, self.timeout, body)?;
        value["content"][0]["text"]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| SconeError::Llm("no text in messages response".into()))
    }
}

impl LlmProvider for AnthropicProvider {
    fn id(&self) -> &str {
        &self.model
    }

    fn extract_facts(&self, text: &str) -> Result<Vec<ExtractedFact>> {
        parse_extraction(&self.message(EXTRACTION_PROMPT, text)?)
    }

    fn answer_with_system(&self, system: &str, question: &str, context: &str) -> Result<String> {
        self.message(
            system,
            &format!("Context:\n{context}\n\nQuestion: {question}"),
        )
    }
}
