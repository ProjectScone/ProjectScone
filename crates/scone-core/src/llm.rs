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

fn http_json(
    req: ureq::RequestBuilder<ureq::typestate::WithBody>,
    body: serde_json::Value,
) -> Result<serde_json::Value> {
    let mut res = req
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
}

impl OpenAiCompatible {
    pub fn new(base_url: &str, model: &str, api_key: Option<String>) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            model: model.to_owned(),
            api_key,
        }
    }

    fn chat(&self, system: &str, user: &str) -> Result<String> {
        let mut req = ureq::post(format!("{}/chat/completions", self.base_url));
        if let Some(key) = &self.api_key {
            req = req.header("authorization", format!("Bearer {key}"));
        }
        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        });
        let value = http_json(req, body)?;
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

    fn answer(&self, question: &str, context: &str) -> Result<String> {
        self.chat(
            "Answer from the provided memory context. Cite nothing you cannot find there; \
             say so when the context lacks the answer.",
            &format!("Context:\n{context}\n\nQuestion: {question}"),
        )
    }
}

/// Anthropic's native messages API.
pub struct AnthropicProvider {
    base_url: String,
    model: String,
    api_key: String,
}

impl AnthropicProvider {
    pub fn new(base_url: &str, model: &str, api_key: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            model: model.to_owned(),
            api_key: api_key.to_owned(),
        }
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
        let value = http_json(req, body)?;
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

    fn answer(&self, question: &str, context: &str) -> Result<String> {
        self.message(
            "Answer from the provided memory context. Say so when the context lacks the answer.",
            &format!("Context:\n{context}\n\nQuestion: {question}"),
        )
    }
}
