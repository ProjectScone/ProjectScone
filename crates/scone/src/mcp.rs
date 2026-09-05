//! MCP server: persistent memory for any MCP agent (spec §8).
//!
//! Space-scoped and input-bounded from the first commit — the predecessor
//! shipped unscoped document access and unbounded inputs, hardened only
//! fourteen months later (memory/lessons.md L-10, bugs.md P-1/P-4).

use std::sync::Mutex;

use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, ErrorData};
use rmcp::{ServerHandler, tool, tool_handler, tool_router};
use scone_core::{Engine, IngestInput, IngestOutcome, RecallOpts, auth};

const MAX_CONTENT: usize = 100_000;
const MAX_QUERY: usize = 1_000;
const MAX_ENTITY: usize = 200;
const MAX_REASON: usize = 500;
const MAX_LIMIT: usize = 50;

pub struct SconeMcp {
    engine: Mutex<Engine>,
    default_space: String,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct StoreParams {
    /// The content to remember (1..=100000 chars)
    pub content: String,
    /// Space to store into; defaults to the server's space
    pub space: Option<String>,
    /// Tags for focused retrieval later (each 1..=64 chars, max 10)
    pub tags: Option<Vec<String>>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct RecallParams {
    /// Natural-language query (1..=1000 chars)
    pub query: String,
    pub space: Option<String>,
    /// Max items (1..=50)
    pub limit: Option<usize>,
    /// Prepend the space's profile (identity facts + recent activity).
    /// Defaults to true.
    pub include_profile: Option<bool>,
    /// Focus recall to episodes carrying ALL of these tags.
    pub tags: Option<Vec<String>>,
    /// Evaluate fact validity at this ISO-8601 instant (time travel)
    pub as_of: Option<String>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct FactsAboutParams {
    /// Entity to look up (person, project, tool …)
    pub entity: String,
    pub space: Option<String>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct ForgetParams {
    /// Fact id to close (from memory_recall / memory_facts_about output)
    pub fact_id: i64,
    /// Why this fact should be forgotten (recorded, never deleted)
    pub reason: String,
    pub space: Option<String>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct PendingParams {
    /// Max episodes to return (1..=20)
    pub limit: Option<usize>,
    pub space: Option<String>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct SubmittedFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// 0..=1; defaults to 0.8
    pub confidence: Option<f32>,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
pub struct StoreFactsParams {
    /// Episode id from memory_pending
    pub episode_id: i64,
    /// Extracted facts (max 50)
    pub facts: Vec<SubmittedFact>,
    pub space: Option<String>,
}

fn tool_error(msg: impl Into<String>) -> CallToolResult {
    CallToolResult::error(vec![ContentBlock::text(msg.into())])
}

fn ok_text(msg: impl Into<String>) -> CallToolResult {
    CallToolResult::success(vec![ContentBlock::text(msg.into())])
}

impl SconeMcp {
    pub fn new(engine: Engine, default_space: &str) -> Self {
        Self {
            engine: Mutex::new(engine),
            default_space: default_space.to_owned(),
        }
    }

    /// Run one closure against the engine in a named space.
    fn with_space<T>(
        &self,
        space_override: &Option<String>,
        f: impl FnOnce(&mut Engine, &auth::ScopedSpace) -> scone_core::Result<T>,
    ) -> Result<T, String> {
        let name = space_override
            .clone()
            .unwrap_or_else(|| self.default_space.clone());
        let mut engine = self
            .engine
            .lock()
            .map_err(|_| "engine lock poisoned".to_owned())?;
        let space = auth::resolve(&mut engine, &name, true).map_err(|e| e.to_string())?;
        f(&mut engine, &space).map_err(|e| e.to_string())
    }
}

#[tool_router]
impl SconeMcp {
    /// Save content to persistent memory. Returns the episode id; duplicate
    /// content is recognized, not re-stored. When an LLM is configured,
    /// facts are distilled immediately.
    #[tool(name = "memory_store")]
    async fn memory_store(
        &self,
        Parameters(p): Parameters<StoreParams>,
    ) -> Result<CallToolResult, ErrorData> {
        if p.content.is_empty() || p.content.len() > MAX_CONTENT {
            return Ok(tool_error(format!(
                "content must be 1..={MAX_CONTENT} bytes, got {}",
                p.content.len()
            )));
        }
        let tags = p.tags.clone().unwrap_or_default();
        if tags.len() > 10 {
            return Ok(tool_error("at most 10 tags per store"));
        }
        let result = self.with_space(&p.space, |engine, space| {
            let outcome = engine.ingest(
                space,
                IngestInput::Note {
                    text: p.content.clone(),
                },
            )?;
            let episode_id = match &outcome {
                IngestOutcome::Ingested { episode_id, .. }
                | IngestOutcome::Deduplicated { episode_id } => *episode_id,
            };
            if !tags.is_empty() {
                let refs: Vec<&str> = tags.iter().map(String::as_str).collect();
                engine.tag_episode(space, episode_id, &refs)?;
            }
            let lane = if engine.has_llm() {
                let r = engine.distill(space, 10)?;
                format!(
                    "facts: +{} added, {} closed{}",
                    r.facts_added,
                    r.facts_closed,
                    if r.failed > 0 {
                        format!(", {} failed (recorded for retry)", r.failed)
                    } else {
                        String::new()
                    }
                )
            } else {
                "semantic lane paused (no LLM configured); episodic memory stored".to_owned()
            };
            Ok((outcome, lane))
        });
        Ok(match result {
            Ok((IngestOutcome::Ingested { episode_id, chunks }, lane)) => ok_text(format!(
                "stored episode {episode_id} ({chunks} chunks). {lane}"
            )),
            Ok((IngestOutcome::Deduplicated { episode_id }, _)) => ok_text(format!(
                "already stored as episode {episode_id} (deduplicated)"
            )),
            Err(e) => tool_error(e),
        })
    }

    /// Recall relevant memory: temporal facts first, then episodic chunks,
    /// each with provenance. `as_of` answers what was true at a past time.
    #[tool(name = "memory_recall")]
    async fn memory_recall(
        &self,
        Parameters(p): Parameters<RecallParams>,
    ) -> Result<CallToolResult, ErrorData> {
        if p.query.is_empty() || p.query.len() > MAX_QUERY {
            return Ok(tool_error(format!(
                "query must be 1..={MAX_QUERY} chars, got {}",
                p.query.len()
            )));
        }
        // Five, not ten: on a 30-question sweep an 8B reader scored
        // 40.0% at five against 36.7% at ten and fifteen, while five
        // hands over roughly half the bytes. Callers who want more can
        // still ask for it.
        let limit = p.limit.unwrap_or(5).clamp(1, MAX_LIMIT);
        let include_profile = p.include_profile.unwrap_or(true);
        let result = self.with_space(&p.space, |engine, space| {
            let profile = if include_profile {
                Some(engine.profile(space, 5)?)
            } else {
                None
            };
            let pack = engine.recall(
                space,
                &p.query,
                &RecallOpts {
                    limit,
                    budget_bytes: None,
                    as_of: p.as_of.clone(),
                    expand_neighbors: true,
                    decompose: true,
                    tags: p.tags.clone().unwrap_or_default(),
                },
            )?;
            Ok((profile, pack))
        });
        Ok(match result {
            Ok((profile, pack)) => {
                let mut out = String::new();
                if let Some(profile) = profile {
                    if !profile.static_facts.is_empty() {
                        out.push_str(
                            "## Profile
",
                        );
                        for f in &profile.static_facts {
                            out.push_str(&format!(
                                "- {} {} {} (conf {:.2})
",
                                f.subject, f.predicate, f.object, f.confidence
                            ));
                        }
                    }
                    if !profile.dynamic.is_empty() {
                        out.push_str(
                            "## Recent activity
",
                        );
                        for d in &profile.dynamic {
                            out.push_str(&format!(
                                "- {}
",
                                d.replace('\n', " ")
                            ));
                        }
                    }
                }
                for f in &pack.facts {
                    out.push_str(&format!(
                        "fact [{}] {} {} {} (conf {:.2}, {})\n",
                        f.fact_id, f.subject, f.predicate, f.object, f.confidence, f.status
                    ));
                }
                for item in &pack.items {
                    out.push_str(&format!(
                        "memory [{} | episode {}] {}\n",
                        item.day(),
                        item.episode_id,
                        item.text
                    ));
                }
                for d in &pack.degraded {
                    out.push_str(&format!("degraded: {d}\n"));
                }
                if out.is_empty() {
                    out.push_str("no matching memory");
                }
                ok_text(out)
            }
            Err(e) => tool_error(e),
        })
    }

    /// List what is currently known about one entity (active facts only).
    #[tool(name = "memory_facts_about")]
    async fn memory_facts_about(
        &self,
        Parameters(p): Parameters<FactsAboutParams>,
    ) -> Result<CallToolResult, ErrorData> {
        if p.entity.is_empty() || p.entity.len() > MAX_ENTITY {
            return Ok(tool_error(format!(
                "entity must be 1..={MAX_ENTITY} chars, got {}",
                p.entity.len()
            )));
        }
        let result = self.with_space(&p.space, |engine, space| {
            engine.facts_about(space, &p.entity)
        });
        Ok(match result {
            Ok(facts) if facts.is_empty() => ok_text(format!("no facts about {}", p.entity)),
            Ok(facts) => ok_text(
                facts
                    .iter()
                    .map(|f| {
                        format!(
                            "fact [{}] {} {} {} (conf {:.2}, since {})",
                            f.fact_id, f.subject, f.predicate, f.object, f.confidence, f.valid_from
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            Err(e) => tool_error(e),
        })
    }

    /// List episodes awaiting fact extraction. YOU are the extractor:
    /// read each episode, distill durable subject/predicate/object facts
    /// with your own reasoning, then submit them via memory_store_facts.
    #[tool(name = "memory_pending")]
    async fn memory_pending(
        &self,
        Parameters(p): Parameters<PendingParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let limit = p.limit.unwrap_or(5).clamp(1, 20);
        let result = self.with_space(&p.space, |engine, space| {
            engine.pending_episodes(space, limit)
        });
        Ok(match result {
            Ok(rows) if rows.is_empty() => ok_text("nothing pending: memory is fully distilled"),
            Ok(rows) => {
                let mut out = String::from(
                    "Episodes awaiting fact extraction (submit via memory_store_facts):
",
                );
                for (id, content, created_at) in rows {
                    out.push_str(&format!(
                        "--- episode {id} ({created_at})
{content}
"
                    ));
                }
                ok_text(out)
            }
            Err(e) => tool_error(e),
        })
    }

    /// Submit facts you extracted from a pending episode. The engine
    /// applies contradiction closure and provenance; you only propose.
    #[tool(name = "memory_store_facts")]
    async fn memory_store_facts(
        &self,
        Parameters(p): Parameters<StoreFactsParams>,
    ) -> Result<CallToolResult, ErrorData> {
        if p.facts.len() > 50 {
            return Ok(tool_error("at most 50 facts per submission"));
        }
        let facts: Vec<scone_core::llm::ExtractedFact> = p
            .facts
            .iter()
            .map(|f| scone_core::llm::ExtractedFact {
                subject: f.subject.clone(),
                predicate: f.predicate.clone(),
                object: f.object.clone(),
                confidence: f.confidence.unwrap_or(0.8).clamp(0.0, 1.0),
            })
            .collect();
        let result = self.with_space(&p.space, |engine, space| {
            engine.complete_distillation(space, p.episode_id, &facts)
        });
        Ok(match result {
            Ok(report) => ok_text(format!(
                "episode {} distilled: {} fact{} added, {} closed, {} deduplicated",
                p.episode_id,
                report.added,
                if report.added == 1 { "" } else { "s" },
                report.closed,
                report.deduplicated,
            )),
            Err(e) => tool_error(e),
        })
    }

    /// Forget a fact: closes its validity interval with your reason.
    /// History is preserved; nothing is deleted.
    #[tool(name = "memory_forget")]
    async fn memory_forget(
        &self,
        Parameters(p): Parameters<ForgetParams>,
    ) -> Result<CallToolResult, ErrorData> {
        if p.reason.is_empty() || p.reason.len() > MAX_REASON {
            return Ok(tool_error(format!(
                "reason must be 1..={MAX_REASON} chars, got {}",
                p.reason.len()
            )));
        }
        let result = self.with_space(&p.space, |engine, space| {
            engine.facts_close(space, p.fact_id, &p.reason)
        });
        Ok(match result {
            Ok(()) => ok_text(format!("closed fact {}: {}", p.fact_id, p.reason)),
            Err(e) => tool_error(e),
        })
    }
}

#[tool_handler]
impl ServerHandler for SconeMcp {
    fn get_info(&self) -> rmcp::model::ServerInfo {
        let mut info = rmcp::model::ServerInfo::default();
        info.server_info.name = "scone".into();
        info.server_info.title = Some("Scone memory engine".into());
        info.server_info.version = env!("CARGO_PKG_VERSION").into();
        info.instructions = Some(
            "Persistent memory for this agent. Call memory_recall at task start; \
             memory_store for durable observations; memory_facts_about before acting \
             on an entity; memory_forget when the user retracts something. \
             Periodically (session start or idle), call memory_pending and distill \
             the returned episodes into subject/predicate/object facts with your own \
             reasoning, submitting via memory_store_facts; you are the extraction \
             model and no API key is needed."
                .into(),
        );
        info
    }
}
