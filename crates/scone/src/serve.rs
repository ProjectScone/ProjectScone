//! Self-hostable HTTP API (spec §8): the same core calls the CLI and MCP
//! make — no privileged path. Every request authenticates with a Bearer
//! key bound to exactly one space (per-key scoping; memory/bugs.md P-1),
//! and bounds mirror the MCP surface (P-4).

// Handlers return axum's Response in the error position — the idiomatic
// axum shape. The large-Err perf hint is irrelevant on an HTTP edge.
#![allow(clippy::result_large_err)]

use std::sync::{Arc, Mutex};

use axum::extract::{Path as AxPath, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use scone_core::{Engine, IngestOutcome, RecallOpts, auth};

const MAX_CONTENT: usize = 100_000;
const MAX_QUERY: usize = 1_000;

#[derive(Clone)]
pub struct SpaceKey {
    pub key: String,
    pub space: String,
}

#[derive(Clone)]
pub struct ServeConfig {
    pub keys: Vec<SpaceKey>,
}

#[derive(Clone)]
struct AppState {
    engine: Arc<Mutex<Engine>>,
    config: Arc<ServeConfig>,
}

/// The console page, with a placeholder where the session key goes.
const CONSOLE_HTML: &str = include_str!("console.html");
const PLAYGROUND_HTML: &str = include_str!("playground.html");

/// Serve the console at `/` on top of the same API the CLI and agents
/// use. The key is baked into the page rather than the URL so it stays
/// out of browser history and out of anything the user might paste.
pub fn console_router(engine: Engine, config: ServeConfig, key: &str) -> Router {
    let page = CONSOLE_HTML.replace("__SCONE_TOKEN__", key);
    let playground = PLAYGROUND_HTML.replace("__SCONE_TOKEN__", key);
    router_with_playground(engine, config, playground).route(
        "/",
        get(move || {
            let page = page.clone();
            async move { ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], page) }
        }),
    )
}

pub fn router(engine: Engine, config: ServeConfig) -> Router {
    router_with_playground(engine, config, PLAYGROUND_HTML.to_owned())
}

fn router_with_playground(engine: Engine, config: ServeConfig, playground: String) -> Router {
    let state = AppState {
        engine: Arc::new(Mutex::new(engine)),
        config: Arc::new(config),
    };
    Router::new()
        .route(
            "/playground",
            get(move || {
                let page = playground.clone();
                async move { ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], page) }
            }),
        )
        .route("/v1/graph", get(get_graph))
        .route("/v1/capabilities", get(get_capabilities))
        .route("/v1/events", get(get_events).post(post_event))
        .route("/v1/episodes/{id}", get(get_episode))
        .route("/v1/episodes", post(post_episode))
        .route("/v1/sources", get(get_sources))
        .route("/v1/recall", get(get_recall))
        .route("/v1/facts", get(get_facts))
        .route("/v1/facts/{id}/close", post(post_fact_close))
        .route("/v1/profile", get(get_profile))
        .route("/v1/status", get(get_status))
        .route("/v1/tags", get(get_tags))
        .with_state(state)
}

/// Authenticated discovery of implemented operations, without a memory read.
async fn get_capabilities(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
) -> Response {
    if let Err(response) = space_for(&headers, &state.config) {
        return response;
    }
    // Capability discovery must not open a space or emit a memory event.
    Json(serde_json::json!({
        "schema_version": 1,
        "implementation": "rust",
        "features": {
            "recall": true, "facts.read": true, "facts.review": false,
            "facts.close": true, "facts.exclude": false, "facts.include": false,
            "events.read": true, "metrics.read": false, "scopes.read": false,
            "status.read": true, "episodes.list": true
        }
    }))
    .into_response()
}

/// Error body every failure path shares — no silent shapes.
fn err(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(serde_json::json!({ "error": message.into() }))).into_response()
}

/// Resolve the Bearer key to its space, or 401. The space name travels
/// back through auth::resolve (I5) on every request.
fn space_for(headers: &axum::http::HeaderMap, config: &ServeConfig) -> Result<String, Response> {
    let presented = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .ok_or_else(|| err(StatusCode::UNAUTHORIZED, "missing Bearer key"))?;
    config
        .keys
        .iter()
        .find(|k| k.key == presented)
        .map(|k| k.space.clone())
        .ok_or_else(|| err(StatusCode::UNAUTHORIZED, "unknown key"))
}

fn with_engine<T>(
    state: &AppState,
    headers: &axum::http::HeaderMap,
    f: impl FnOnce(&mut Engine, &auth::ScopedSpace) -> scone_core::Result<T>,
) -> Result<T, Response> {
    let space_name = space_for(headers, &state.config)?;
    let mut engine = state
        .engine
        .lock()
        .map_err(|_| err(StatusCode::INTERNAL_SERVER_ERROR, "engine lock poisoned"))?;
    let space = auth::resolve(&mut engine, &space_name, true)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    f(&mut engine, &space).map_err(|e| err(status_for(&e), e.to_string()))
}

/// A caller's mistake is not a server failure. Reporting a missing
/// fact or an empty query as 500 tells every client to retry
/// something that will never succeed.
fn status_for(e: &scone_core::SconeError) -> StatusCode {
    use scone_core::SconeError::*;
    match e {
        NotFound(_) => StatusCode::NOT_FOUND,
        InvalidInput(_) => StatusCode::UNPROCESSABLE_ENTITY,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// Unknown fields are refused rather than ignored: a client that sends
/// a field we silently drop believes it stored something it did not.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct EpisodeBody {
    content: String,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    metadata: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    tags: Vec<String>,
    /// Where this came from, kept as provenance.
    #[serde(default)]
    source: Option<String>,
    /// When it happened. Facts distilled from it inherit this, so
    /// backfilling history over the API dates correctly.
    #[serde(default)]
    created_at: Option<String>,
}

async fn post_episode(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    // Taken as a Result so a malformed body still answers in the JSON
    // error shape every other failure uses, instead of plain text.
    body: Result<Json<EpisodeBody>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(body) = match body {
        Ok(body) => body,
        Err(rejection) => {
            return err(StatusCode::UNPROCESSABLE_ENTITY, rejection.body_text());
        }
    };
    if body.content.is_empty() || body.content.len() > MAX_CONTENT {
        return err(
            StatusCode::UNPROCESSABLE_ENTITY,
            format!("content must be 1..={MAX_CONTENT} bytes"),
        );
    }
    if body.tags.len() > 10 {
        return err(StatusCode::UNPROCESSABLE_ENTITY, "at most 10 tags");
    }
    if body.metadata.len() > 32
        || body
            .metadata
            .iter()
            .any(|(k, v)| k.is_empty() || k.len() > 64 || v.len() > 256)
    {
        return err(
            StatusCode::UNPROCESSABLE_ENTITY,
            "metadata exceeds 32 keys, 64-byte keys or 256-byte values",
        );
    }
    match with_engine(&state, &headers, |engine, space| {
        let outcome = engine.import_episode_outcome(
            space,
            body.kind.as_deref().unwrap_or("note"),
            &body.content,
            body.source.as_deref(),
            body.created_at.as_deref(),
        )?;
        let episode_id = match &outcome {
            IngestOutcome::Ingested { episode_id, .. }
            | IngestOutcome::Deduplicated { episode_id } => *episode_id,
        };
        if !body.metadata.is_empty() {
            engine.set_episode_metadata(space, episode_id, &body.metadata)?;
        }
        if !body.tags.is_empty() {
            let refs: Vec<&str> = body.tags.iter().map(String::as_str).collect();
            engine.tag_episode(space, episode_id, &refs)?;
        }
        Ok(outcome)
    }) {
        Ok(IngestOutcome::Ingested { episode_id, chunks }) => (
            StatusCode::CREATED,
            Json(serde_json::json!({
                "episode_id": episode_id, "chunks": chunks, "deduplicated": false
            })),
        )
            .into_response(),
        Ok(IngestOutcome::Deduplicated { episode_id }) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "episode_id": episode_id, "deduplicated": true
            })),
        )
            .into_response(),
        Err(response) => response,
    }
}

#[derive(serde::Deserialize)]
struct RecallQuery {
    q: String,
    limit: Option<usize>,
    as_of: Option<String>,
    /// Comma-separated tag filter (AND semantics).
    tags: Option<String>,
}

async fn get_recall(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Query(query): Query<RecallQuery>,
) -> Response {
    if query.q.trim().is_empty() || query.q.len() > MAX_QUERY {
        return err(
            StatusCode::UNPROCESSABLE_ENTITY,
            format!("q must be 1..={MAX_QUERY} chars"),
        );
    }
    let opts = RecallOpts {
        limit: query.limit.unwrap_or(10).clamp(1, 50),
        budget_bytes: None,
        as_of: query.as_of.clone(),
        expand_neighbors: false,
        decompose: true,
        tags: query
            .tags
            .as_deref()
            .map(|t| t.split(',').map(|x| x.trim().to_owned()).collect())
            .unwrap_or_default(),
        ..Default::default()
    };
    match with_engine(&state, &headers, |engine, space| {
        let pack = engine.recall(space, &query.q, &opts)?;
        let event_id = engine.record_recall_evidence(space, &query.q, &pack)?;
        Ok((pack, event_id))
    }) {
        Ok((pack, event_id)) => Json(serde_json::json!({
            "event_id":event_id,
            "facts": pack.facts.iter().map(|f| serde_json::json!({
                "fact_id": f.fact_id, "subject": f.subject, "predicate": f.predicate,
                "object": f.object, "confidence": f.confidence,
                "valid_from": f.valid_from, "valid_until": f.valid_until,
                "status": f.status,
            })).collect::<Vec<_>>(),
            "items": pack.items.iter().map(|i| serde_json::json!({
                "episode_id": i.episode_id, "chunk_id":i.chunk_id, "text": i.text, "score": i.score,
                "source": i.source, "created_at": i.created_at,
            })).collect::<Vec<_>>(),
            "degraded": pack.degraded,
            "returned_bytes": pack.returned_bytes,
            "space_bytes": pack.space_bytes,
            "context_reduction": pack.context_reduction(),
        }))
        .into_response(),
        Err(response) => response,
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct EventBody {
    kind: String,
    payload: serde_json::Value,
}
async fn post_event(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    body: Result<Json<EventBody>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(body) = match body {
        Ok(b) => b,
        Err(e) => return err(StatusCode::UNPROCESSABLE_ENTITY, e.body_text()),
    };
    if body.kind != "agent" {
        return err(
            StatusCode::UNPROCESSABLE_ENTITY,
            "only connector-reported agent events may be posted",
        );
    }
    match with_engine(&state, &headers, |engine, space| {
        engine.record_agent_event(space, body.payload)
    }) {
        Ok(v) => Json(v).into_response(),
        Err(e) => e,
    }
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceQuery {
    after_id: Option<i64>,
    limit: Option<usize>,
    kind: Option<String>,
}
async fn get_events(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Query(q): Query<EvidenceQuery>,
) -> Response {
    match with_engine(&state, &headers, |engine, space| {
        engine.evidence_events(space, q.after_id, q.limit.unwrap_or(100), q.kind.as_deref())
    }) {
        Ok(v) => Json(v).into_response(),
        Err(e) => e,
    }
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphQuery {
    limit: Option<usize>,
}
async fn get_graph(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Query(q): Query<GraphQuery>,
) -> Response {
    match with_engine(&state, &headers, |engine, space| {
        engine.evidence_graph(space, q.limit.unwrap_or(200))
    }) {
        Ok(v) => Json(v).into_response(),
        Err(e) => e,
    }
}
async fn get_episode(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    AxPath(id): AxPath<i64>,
) -> Response {
    match with_engine(&state, &headers, |engine, space| {
        engine.evidence_episode(space, id)
    }) {
        Ok(v) => Json(v).into_response(),
        Err(e) => e,
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceQuery {
    before: Option<i64>,
    limit: Option<usize>,
    kind: Option<String>,
}

async fn get_sources(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    query: Result<Query<SourceQuery>, axum::extract::rejection::QueryRejection>,
) -> Response {
    // Authenticate before returning query diagnostics. The key always selects scope.
    if let Err(response) = space_for(&headers, &state.config) {
        return response;
    }
    let Query(query) = match query {
        Ok(query) => query,
        Err(_) => {
            return err(
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid source page query",
            );
        }
    };
    match with_engine(&state, &headers, |engine, space| {
        engine.source_page(
            space,
            query.before,
            query.limit.unwrap_or(25),
            query.kind.as_deref(),
        )
    }) {
        Ok(value) => Json(value).into_response(),
        Err(error) => error,
    }
}

#[derive(serde::Deserialize)]
struct FactsQuery {
    #[serde(default)]
    all: bool,
}

async fn get_facts(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Query(query): Query<FactsQuery>,
) -> Response {
    // The revision is read with the list, not in a second request: a page
    // that freezes what it renders needs to know what it froze, and two
    // calls can straddle a write.
    match with_engine(&state, &headers, |engine, space| {
        let facts = engine.facts_list(space, query.all)?;
        let revision = engine.space_revision(space)?;
        Ok((facts, revision))
    }) {
        Ok((facts, revision)) => Json(serde_json::json!({
            "facts": facts.iter().map(|f| serde_json::json!({
                "fact_id": f.fact_id, "subject": f.subject, "predicate": f.predicate,
                "object": f.object, "confidence": f.confidence,
                "valid_from": f.valid_from, "valid_until": f.valid_until,
                "status": f.status,
            })).collect::<Vec<_>>(),
            "revision": revision,
        }))
        .into_response(),
        Err(response) => response,
    }
}

#[derive(serde::Deserialize)]
struct CloseBody {
    reason: String,
}

async fn post_fact_close(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    AxPath(id): AxPath<i64>,
    Json(body): Json<CloseBody>,
) -> Response {
    if body.reason.is_empty() || body.reason.len() > 500 {
        return err(
            StatusCode::UNPROCESSABLE_ENTITY,
            "reason must be 1..=500 chars",
        );
    }
    match with_engine(&state, &headers, |engine, space| {
        engine.facts_close(space, id, &body.reason)
    }) {
        Ok(()) => Json(serde_json::json!({"closed": id, "reason": body.reason})).into_response(),
        Err(response) => response,
    }
}

async fn get_profile(State(state): State<AppState>, headers: axum::http::HeaderMap) -> Response {
    match with_engine(&state, &headers, |engine, space| engine.profile(space, 8)) {
        Ok(profile) => Json(serde_json::json!({
            "static_facts": profile.static_facts.iter().map(|f| serde_json::json!({
                "fact_id": f.fact_id, "subject": f.subject, "predicate": f.predicate,
                "object": f.object, "confidence": f.confidence,
            })).collect::<Vec<_>>(),
            "dynamic": profile.dynamic,
        }))
        .into_response(),
        Err(response) => response,
    }
}

async fn get_tags(State(state): State<AppState>, headers: axum::http::HeaderMap) -> Response {
    match with_engine(&state, &headers, |engine, space| engine.tags_list(space)) {
        Ok(tags) => Json(serde_json::json!({
            "tags": tags.iter().map(|(name, count)| serde_json::json!({
                "name": name, "count": count,
            })).collect::<Vec<_>>(),
        }))
        .into_response(),
        Err(response) => response,
    }
}

async fn get_status(State(state): State<AppState>, headers: axum::http::HeaderMap) -> Response {
    match with_engine(&state, &headers, |engine, space| {
        let report = engine.status()?;
        let mine = report.spaces.iter().find(|s| s.name == space.name());
        Ok(serde_json::json!({
            "space": space.name(),
            "episodes": mine.map(|s| s.episodes).unwrap_or(0),
            "chunks": mine.map(|s| s.chunks).unwrap_or(0),
            "revision": mine.map(|s| s.revision).unwrap_or(0),
            "semantic_lane": if report.llm_id.is_some() { "active" } else { "paused" },
            "pending_distill": report.pending_distill,
        }))
    }) {
        Ok(value) => Json(value).into_response(),
        Err(response) => response,
    }
}
