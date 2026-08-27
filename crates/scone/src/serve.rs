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
use scone_core::{Engine, IngestInput, IngestOutcome, RecallOpts, auth};

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

pub fn router(engine: Engine, config: ServeConfig) -> Router {
    let state = AppState {
        engine: Arc::new(Mutex::new(engine)),
        config: Arc::new(config),
    };
    Router::new()
        .route("/v1/episodes", post(post_episode))
        .route("/v1/recall", get(get_recall))
        .route("/v1/facts", get(get_facts))
        .route("/v1/facts/{id}/close", post(post_fact_close))
        .route("/v1/profile", get(get_profile))
        .route("/v1/status", get(get_status))
        .with_state(state)
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
    f(&mut engine, &space).map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

#[derive(serde::Deserialize)]
struct EpisodeBody {
    content: String,
}

async fn post_episode(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(body): Json<EpisodeBody>,
) -> Response {
    if body.content.is_empty() || body.content.len() > MAX_CONTENT {
        return err(
            StatusCode::UNPROCESSABLE_ENTITY,
            format!("content must be 1..={MAX_CONTENT} bytes"),
        );
    }
    match with_engine(&state, &headers, |engine, space| {
        engine.ingest(
            space,
            IngestInput::Note {
                text: body.content.clone(),
            },
        )
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
}

async fn get_recall(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Query(query): Query<RecallQuery>,
) -> Response {
    if query.q.is_empty() || query.q.len() > MAX_QUERY {
        return err(
            StatusCode::UNPROCESSABLE_ENTITY,
            format!("q must be 1..={MAX_QUERY} chars"),
        );
    }
    let opts = RecallOpts {
        limit: query.limit.unwrap_or(10).clamp(1, 50),
        budget_bytes: None,
        as_of: query.as_of.clone(),
    };
    match with_engine(&state, &headers, |engine, space| {
        engine.recall(space, &query.q, &opts)
    }) {
        Ok(pack) => Json(serde_json::json!({
            "facts": pack.facts.iter().map(|f| serde_json::json!({
                "fact_id": f.fact_id, "subject": f.subject, "predicate": f.predicate,
                "object": f.object, "confidence": f.confidence,
                "valid_from": f.valid_from, "valid_until": f.valid_until,
                "status": f.status,
            })).collect::<Vec<_>>(),
            "items": pack.items.iter().map(|i| serde_json::json!({
                "episode_id": i.episode_id, "text": i.text, "score": i.score,
                "source": i.source, "created_at": i.created_at,
            })).collect::<Vec<_>>(),
            "degraded": pack.degraded,
        }))
        .into_response(),
        Err(response) => response,
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
    match with_engine(&state, &headers, |engine, space| {
        engine.facts_list(space, query.all)
    }) {
        Ok(facts) => Json(serde_json::json!({
            "facts": facts.iter().map(|f| serde_json::json!({
                "fact_id": f.fact_id, "subject": f.subject, "predicate": f.predicate,
                "object": f.object, "confidence": f.confidence,
                "valid_from": f.valid_from, "valid_until": f.valid_until,
                "status": f.status,
            })).collect::<Vec<_>>(),
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

async fn get_status(State(state): State<AppState>, headers: axum::http::HeaderMap) -> Response {
    let space_name = match space_for(&headers, &state.config) {
        Ok(name) => name,
        Err(response) => return response,
    };
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
        Ok(value) => {
            let _ = space_name;
            Json(value).into_response()
        }
        Err(response) => response,
    }
}
