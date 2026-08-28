#![allow(clippy::unwrap_used)]
//! HTTP API tested hermetically via tower oneshot — no ports, no network.
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use scone::serve::{ServeConfig, SpaceKey, router};
use scone_core::Engine;
use scone_core::embed::HashEmbedder;
use tower::ServiceExt;

fn app(dir: &std::path::Path) -> axum::Router {
    let engine = Engine::open(dir, Box::new(HashEmbedder::new(64))).unwrap();
    router(
        engine,
        ServeConfig {
            keys: vec![
                SpaceKey {
                    key: "sk-alice".into(),
                    space: "alice".into(),
                },
                SpaceKey {
                    key: "sk-bob".into(),
                    space: "bob".into(),
                },
            ],
        },
    )
}

async fn call(
    app: &axum::Router,
    method: &str,
    uri: &str,
    key: Option<&str>,
    body: Option<serde_json::Value>,
) -> (StatusCode, serde_json::Value) {
    let mut req = Request::builder().method(method).uri(uri);
    if let Some(k) = key {
        req = req.header(header::AUTHORIZATION, format!("Bearer {k}"));
    }
    let req = match body {
        Some(b) => req
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(b.to_string()))
            .unwrap(),
        None => req.body(Body::empty()).unwrap(),
    };
    let res = app.clone().oneshot(req).await.unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let value = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap()
    };
    (status, value)
}

#[tokio::test]
async fn unauthorized_without_a_known_key() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(dir.path());
    let (status, _) = call(&app, "GET", "/v1/status", None, None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let (status, _) = call(&app, "GET", "/v1/status", Some("sk-wrong"), None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn store_recall_round_trip_with_key_scoping() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(dir.path());
    let (status, body) = call(
        &app,
        "POST",
        "/v1/episodes",
        Some("sk-alice"),
        Some(serde_json::json!({"content": "alice keeps her notes in scone"})),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert!(body["episode_id"].as_i64().unwrap() >= 1);
    let (status, body) = call(&app, "GET", "/v1/recall?q=notes", Some("sk-alice"), None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        body["items"][0]["text"].as_str().unwrap().contains("notes"),
        "{body}"
    );
    // Bob's key must not see Alice's space (per-key scoping, P-1).
    let (status, body) = call(&app, "GET", "/v1/recall?q=notes", Some("sk-bob"), None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["items"].as_array().unwrap().len(), 0, "{body}");
}

#[tokio::test]
async fn duplicate_store_reports_deduplicated_not_created() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(dir.path());
    let payload = serde_json::json!({"content": "same content"});
    let (s1, _) = call(
        &app,
        "POST",
        "/v1/episodes",
        Some("sk-alice"),
        Some(payload.clone()),
    )
    .await;
    assert_eq!(s1, StatusCode::CREATED);
    let (s2, body) = call(
        &app,
        "POST",
        "/v1/episodes",
        Some("sk-alice"),
        Some(payload),
    )
    .await;
    assert_eq!(s2, StatusCode::OK, "dedup is not a second creation");
    assert_eq!(body["deduplicated"], serde_json::json!(true));
}

#[tokio::test]
async fn oversized_content_is_rejected_with_422() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(dir.path());
    let (status, body) = call(
        &app,
        "POST",
        "/v1/episodes",
        Some("sk-alice"),
        Some(serde_json::json!({"content": "x".repeat(100_001)})),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
}

#[tokio::test]
async fn status_reports_space_and_lane() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(dir.path());
    call(
        &app,
        "POST",
        "/v1/episodes",
        Some("sk-alice"),
        Some(serde_json::json!({"content": "one"})),
    )
    .await;
    let (status, body) = call(&app, "GET", "/v1/status", Some("sk-alice"), None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["space"], serde_json::json!("alice"));
    assert_eq!(body["episodes"], serde_json::json!(1));
    assert_eq!(body["semantic_lane"], serde_json::json!("paused"));
}

#[tokio::test]
async fn profile_endpoint_serves_identity_and_activity() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(dir.path());
    call(
        &app,
        "POST",
        "/v1/episodes",
        Some("sk-alice"),
        Some(serde_json::json!({"content": "alice ships rust code"})),
    )
    .await;
    let (status, body) = call(&app, "GET", "/v1/profile", Some("sk-alice"), None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        body["dynamic"][0].as_str().unwrap().contains("ships rust"),
        "{body}"
    );
    assert!(body["static_facts"].as_array().is_some());
    let (status, _) = call(&app, "GET", "/v1/profile", None, None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn http_tags_flow_from_store_to_focused_recall() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(dir.path());
    call(
        &app,
        "POST",
        "/v1/episodes",
        Some("sk-alice"),
        Some(serde_json::json!({"content": "deploy checklist lives in the wiki", "tags": ["ops"]})),
    )
    .await;
    call(
        &app,
        "POST",
        "/v1/episodes",
        Some("sk-alice"),
        Some(serde_json::json!({"content": "the checklist for baking bread"})),
    )
    .await;
    let (status, body) = call(
        &app,
        "GET",
        "/v1/recall?q=checklist&tags=ops",
        Some("sk-alice"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let items = body["items"].as_array().unwrap();
    assert_eq!(items.len(), 1, "{body}");
    assert!(items[0]["text"].as_str().unwrap().contains("wiki"));
    let (status, body) = call(&app, "GET", "/v1/tags", Some("sk-alice"), None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["tags"][0]["name"].as_str().unwrap() == "ops", "{body}");
}
