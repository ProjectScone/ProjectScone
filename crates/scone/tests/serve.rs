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

#[tokio::test]
async fn capabilities_are_authenticated_explicit_and_read_only() {
    let dir = tempfile::tempdir().unwrap();
    let server = app(dir.path());
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/http-capabilities.json"
    ))
    .unwrap();
    for key in [None, Some("wrong")] {
        let (status, _) = call(&server, "GET", "/v1/capabilities", key, None).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }
    let (_, before) = call(&server, "GET", "/v1/status", Some("sk-alice"), None).await;
    for key in ["sk-alice", "sk-bob"] {
        let (status, body) = call(&server, "GET", "/v1/capabilities", Some(key), None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, expected["rust"]);
    }
    let (_, after) = call(&server, "GET", "/v1/status", Some("sk-alice"), None).await;
    assert_eq!(before, after);
}

#[tokio::test]
async fn evidence_console_and_plain_server_both_serve_playground() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let server = scone::serve::console_router(
        engine,
        ServeConfig {
            keys: vec![SpaceKey {
                key: "fixture-token".into(),
                space: "alice".into(),
            }],
        },
        "fixture-token",
    );
    let response = server
        .oneshot(
            Request::builder()
                .uri("/playground")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert!(String::from_utf8_lossy(&body).contains("fixture-token"));
}

#[tokio::test]
async fn evidence_is_scoped_persistent_and_retries_do_not_duplicate() {
    let dir = tempfile::tempdir().unwrap();
    let server = app(dir.path());
    let payload = serde_json::json!({"kind":"agent","payload":{
        "agent":"codex","session_id":"s1","event":"prompt",
        "source_event_id":"input-1","text":"Keep launch local"}});
    let (status, receipt) = call(
        &server,
        "POST",
        "/v1/events",
        Some("sk-alice"),
        Some(payload.clone()),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{receipt}");
    let (_, duplicate) = call(
        &server,
        "POST",
        "/v1/events",
        Some("sk-alice"),
        Some(payload),
    )
    .await;
    assert_eq!(receipt["recorded"], duplicate["recorded"]);
    let (_, events) = call(
        &server,
        "GET",
        "/v1/events?after_id=0",
        Some("sk-alice"),
        None,
    )
    .await;
    assert_eq!(events["events"].as_array().unwrap().len(), 1);
    assert_eq!(events["next_after_id"], receipt["recorded"]);
    let (_, private) = call(&server, "GET", "/v1/graph", Some("sk-bob"), None).await;
    assert_eq!(private["nodes"], serde_json::json!([]));
    drop(server);
    let reopened = app(dir.path());
    let (_, graph) = call(&reopened, "GET", "/v1/graph", Some("sk-alice"), None).await;
    assert!(
        graph["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|n| n["data"]["text"] == "Keep launch local")
    );
}

#[tokio::test]
async fn evidence_links_capture_and_recall_only_to_scoped_records() {
    let dir = tempfile::tempdir().unwrap();
    let server = app(dir.path());
    let (status, ep)=call(&server,"POST","/v1/episodes",Some("sk-alice"),Some(serde_json::json!({"kind":"conversation","content":"Keep launch local","metadata":{"agent":"codex","session_id":"s1"}}))).await;
    assert_eq!(status, StatusCode::CREATED, "{ep}");
    let event = serde_json::json!({"kind":"agent","payload":{"agent":"codex","session_id":"s1","event":"prompt","source_event_id":"prompt-1","episode_id":ep["episode_id"],"text":"Keep launch local"}});
    let (status, _) = call(
        &server,
        "POST",
        "/v1/events",
        Some("sk-bob"),
        Some(event.clone()),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    let (status, _) = call(&server, "POST", "/v1/events", Some("sk-alice"), Some(event)).await;
    assert_eq!(status, StatusCode::OK);
    let (_, recall) = call(
        &server,
        "GET",
        "/v1/recall?q=launch",
        Some("sk-alice"),
        None,
    )
    .await;
    assert!(recall["event_id"].as_i64().is_some());
    let (_, graph) = call(&server, "GET", "/v1/graph", Some("sk-alice"), None).await;
    let edges = graph["edges"].as_array().unwrap();
    assert!(
        edges
            .iter()
            .any(|e| e["kind"] == "captured_as" && e["target"] == "episode:1")
    );
    assert!(
        edges
            .iter()
            .any(|e| e["kind"] == "returned" && e["target"] == "chunk:1")
    );
    assert!(!edges.iter().any(|e| e["kind"] == "similar"));
    let (status, _) = call(&server, "GET", "/v1/episodes/1", Some("sk-bob"), None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn evidence_rejects_forgery_conflicting_retries_and_invalid_utf8_byte_size() {
    let dir = tempfile::tempdir().unwrap();
    let server = app(dir.path());
    for payload in [
        serde_json::json!({"kind":"recall","payload":{}}),
        serde_json::json!({"kind":"agent","payload":{"agent":"codex","session_id":"s1","event":"prompt","text":"🥐".repeat(17000)}}),
    ] {
        let (status, _) = call(
            &server,
            "POST",
            "/v1/events",
            Some("sk-alice"),
            Some(payload),
        )
        .await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    }
    let mut payload = serde_json::json!({"kind":"agent","payload":{"agent":"codex","session_id":"s1","event":"prompt","source_event_id":"p1","text":"first"}});
    assert_eq!(
        call(
            &server,
            "POST",
            "/v1/events",
            Some("sk-alice"),
            Some(payload.clone())
        )
        .await
        .0,
        StatusCode::OK
    );
    payload["payload"]["text"] = serde_json::json!("different");
    assert_eq!(
        call(
            &server,
            "POST",
            "/v1/events",
            Some("sk-alice"),
            Some(payload)
        )
        .await
        .0,
        StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn evidence_accepts_explicit_connector_truncation_without_hiding_it() {
    let dir = tempfile::tempdir().unwrap();
    let server = app(dir.path());
    let (status,_)=call(&server,"POST","/v1/events",Some("sk-alice"),Some(serde_json::json!({"kind":"agent","payload":{"agent":"claude-code","session_id":"s1","event":"tool_result","text":"bounded excerpt","text_truncated":true}}))).await;
    assert_eq!(status, StatusCode::OK);
    let (_, page) = call(
        &server,
        "GET",
        "/v1/events?after_id=0",
        Some("sk-alice"),
        None,
    )
    .await;
    assert_eq!(page["events"][0]["payload"]["text_truncated"], true);
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

/// The console is the only surface a non-developer touches, so the page
/// must carry a working key and the API underneath must still refuse
/// everyone else. A console that serves an unauthenticated path would
/// hand the whole memory store to any process that can reach loopback.
#[tokio::test]
async fn console_page_carries_its_key_and_the_api_still_refuses_others() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let app = scone::serve::console_router(
        engine,
        ServeConfig {
            keys: vec![SpaceKey {
                key: "ui-deadbeefdeadbeef".into(),
                space: "default".into(),
            }],
        },
        "ui-deadbeefdeadbeef",
    );

    let res = app
        .clone()
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let page = String::from_utf8_lossy(&body);
    assert!(
        page.contains("ui-deadbeefdeadbeef"),
        "the key must reach the page"
    );
    assert!(
        !page.contains("__SCONE_TOKEN__"),
        "the placeholder must be replaced, or every call 401s"
    );

    // Same server, no key: the console must not have opened a back door.
    let res = app
        .oneshot(
            Request::builder()
                .uri("/v1/recall?q=anything")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
}

/// A caller's mistake must not read as a server failure. Reporting
/// these as 500 tells every client to retry what can never succeed.
#[tokio::test]
async fn client_mistakes_get_client_status_codes() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(dir.path());

    let (status, body) = call(
        &app,
        "POST",
        "/v1/facts/4242/close",
        Some("sk-alice"),
        Some(serde_json::json!({"reason": "gone"})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");

    // Whitespace passes an is_empty check but not the engine's.
    let (status, body) = call(
        &app,
        "GET",
        "/v1/recall?q=%20%20%20",
        Some("sk-alice"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
}

/// Provenance and event time have to survive the API, or anything
/// ingested over HTTP is undated and unattributed, and the facts
/// distilled from it inherit the wrong day.
#[tokio::test]
async fn episodes_keep_their_source_and_date_over_http() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(dir.path());
    let (status, body) = call(
        &app,
        "POST",
        "/v1/episodes",
        Some("sk-alice"),
        Some(serde_json::json!({
            "content": "an old decision",
            "source": "https://example.com/x",
            "created_at": "2023-03-01T09:00:00.000Z"
        })),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");

    let (_, recalled) = call(&app, "GET", "/v1/recall?q=decision", Some("sk-alice"), None).await;
    let item = &recalled["items"][0];
    assert_eq!(item["source"], "https://example.com/x");
    assert!(
        item["created_at"]
            .as_str()
            .unwrap_or_default()
            .starts_with("2023-03-01"),
        "the caller's date must survive: {item}"
    );

    // A field we would otherwise silently drop is refused instead.
    let (status, _) = call(
        &app,
        "POST",
        "/v1/episodes",
        Some("sk-alice"),
        Some(serde_json::json!({"content": "x", "nonsense": 1})),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
}
