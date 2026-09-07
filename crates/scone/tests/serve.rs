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
async fn source_inventory_matches_shared_literal_contract() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/source-inventory.json"
    ))
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let server = app(dir.path());
    for record in fixture["records"].as_array().unwrap() {
        let key = if record["owner"] == "alpha" {
            "sk-alice"
        } else {
            "sk-bob"
        };
        assert_eq!(
            call(
                &server,
                "POST",
                "/v1/episodes",
                Some(key),
                Some(record["body"].clone())
            )
            .await
            .0,
            StatusCode::CREATED
        );
    }
    for page in fixture["pages"].as_array().unwrap() {
        let (status, body) = call(
            &server,
            "GET",
            &format!("/v1/sources?{}", page["query"].as_str().unwrap()),
            Some("sk-alice"),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, page["expected"]);
    }
    for query in fixture["invalid_queries"].as_array().unwrap() {
        assert_eq!(
            call(
                &server,
                "GET",
                &format!("/v1/sources?{}", query.as_str().unwrap()),
                Some("sk-alice"),
                None
            )
            .await
            .0,
            StatusCode::UNPROCESSABLE_ENTITY
        );
    }
}

#[tokio::test]
async fn source_inventory_pages_are_scoped_and_not_ranked() {
    let dir = tempfile::tempdir().unwrap();
    let server = app(dir.path());
    let mut ids = Vec::new();
    for i in 0..7 {
        let (status, added) = call(
            &server,
            "POST",
            "/v1/episodes",
            Some("sk-alice"),
            Some(serde_json::json!({
                "content": format!("source {i}"), "kind": if i % 2 == 0 {"file"} else {"note"},
                "created_at": format!("2024-01-{:02}",9-i)
            })),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        ids.push(added["episode_id"].as_i64().unwrap());
        call(
            &server,
            "POST",
            "/v1/episodes",
            Some("sk-bob"),
            Some(serde_json::json!({"content":format!("private {i}")})),
        )
        .await;
    }
    let (status, first) = call(
        &server,
        "GET",
        "/v1/sources?limit=2&kind=file",
        Some("sk-alice"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        first["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["episode_id"].as_i64().unwrap())
            .collect::<Vec<_>>(),
        vec![ids[6], ids[4]]
    );
    assert_eq!(first["has_more"], true);
    assert_eq!(first["next_before"], ids[4]);
    let (_, second) = call(
        &server,
        "GET",
        &format!("/v1/sources?limit=2&kind=file&before={}", ids[4]),
        Some("sk-alice"),
        None,
    )
    .await;
    assert_eq!(
        second["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["episode_id"].as_i64().unwrap())
            .collect::<Vec<_>>(),
        vec![ids[2], ids[0]]
    );
    assert_eq!(second["has_more"], false);
    assert!(second["next_before"].is_null());
    for query in [
        "limit=0",
        "limit=101",
        "before=0",
        "before=-1",
        "kind=",
        "kind=unknown",
    ] {
        assert_eq!(
            call(
                &server,
                "GET",
                &format!("/v1/sources?{query}"),
                Some("sk-alice"),
                None
            )
            .await
            .0,
            StatusCode::UNPROCESSABLE_ENTITY
        );
    }
    assert_eq!(
        call(&server, "GET", "/v1/sources", None, None).await.0,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        call(&server, "GET", "/v1/sources", Some("wrong"), None)
            .await
            .0,
        StatusCode::UNAUTHORIZED
    );
}

#[tokio::test]
async fn source_inventory_has_literal_unicode_previews_and_byte_counts() {
    let dir = tempfile::tempdir().unwrap();
    let server = app(dir.path());
    let text = format!("prefix\0{}", "猫🙂".repeat(300));
    let (_, added) = call(&server,"POST","/v1/episodes",Some("sk-alice"),Some(serde_json::json!({
        "content": text, "kind":"file", "source":"original.txt", "created_at":"2024-01-01T00:00:00.000Z"
    }))).await;
    let (status, page) = call(&server, "GET", "/v1/sources", Some("sk-alice"), None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        page,
        serde_json::json!({"items":[{"episode_id":added["episode_id"],"kind":"file","source":"original.txt",
        "created_at":"2024-01-01T00:00:00.000Z","byte_count":text.len(),"preview":text.chars().take(500).collect::<String>(),"preview_truncated":true,"status":"pending"}],
        "has_more":false,"next_before":null})
    );
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
async fn concept_pages_are_served_by_the_console_host_only_and_never_by_a_catch_all() {
    let dir = tempfile::tempdir().unwrap();
    let config = || ServeConfig {
        keys: vec![SpaceKey {
            key: "fixture-token".into(),
            space: "alice".into(),
        }],
    };
    let engine = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let console = scone::serve::console_router(engine, config(), "fixture-token");
    for path in scone::serve::LEARN_PAGES {
        for method in ["GET", "HEAD"] {
            let response = console
                .clone()
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(path)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{method} {path}");
            assert!(
                response.headers()[header::CONTENT_TYPE]
                    .to_str()
                    .unwrap()
                    .starts_with("text/html")
            );
            if method == "GET" {
                let body = response.into_body().collect().await.unwrap().to_bytes();
                let text = String::from_utf8_lossy(&body);
                assert!(
                    !text.contains("fixture-token"),
                    "a public page carries no configured key: {path}"
                );
                assert!(
                    text.contains("__SCONE_TOKEN__") || !text.contains("SCONE_TOKEN"),
                    "the bundle is served as packaged"
                );
            }
        }
    }
    let stray = console
        .clone()
        .oneshot(
            Request::builder()
                .uri("/learn/anything-else")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(stray.status(), StatusCode::NOT_FOUND, "no catch-all");
    let plain = scone::serve::router(
        Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap(),
        config(),
    );
    for path in scone::serve::LEARN_PAGES {
        let response = plain
            .clone()
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::NOT_FOUND,
            "{path} on the plain server"
        );
    }
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

/// A review page freezes the list it renders and then acts on it, so it
/// needs the revision that list was read at. The Python server carries it
/// beside the facts; this one did not, which left a page built against
/// Rust unable to tell that the space had moved under it.
#[tokio::test]
async fn the_facts_list_carries_the_space_revision() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(dir.path());

    let (_, before) = call(&app, "GET", "/v1/facts", Some("sk-alice"), None).await;
    let first = before["revision"]
        .as_i64()
        .expect("revision beside the facts");

    let (status, _) = call(
        &app,
        "POST",
        "/v1/episodes",
        Some("sk-alice"),
        Some(serde_json::json!({"content": "Ana moved to Lisbon in March."})),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "storing an episode");

    let (_, after) = call(&app, "GET", "/v1/facts", Some("sk-alice"), None).await;
    assert!(
        after["revision"].as_i64().unwrap() > first,
        "a write must move the revision: {first} then {}",
        after["revision"]
    );
}

/// A claim without its source is a claim nobody can check. The engine has
/// kept provenance in fact_provenance since the first schema, and the
/// HTTP surface has never handed it over, so a page built against this
/// server can show what a claim says and never where it came from.
#[tokio::test]
async fn a_fact_names_the_episodes_it_came_from() {
    use scone_core::llm::ExtractedFact;
    use scone_core::{IngestInput, IngestOutcome, auth};

    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    let space = auth::resolve(&mut engine, "alice", true).unwrap();
    let IngestOutcome::Ingested { episode_id, .. } = engine
        .ingest(
            &space,
            IngestInput::Note {
                text: "Ana moved to Lisbon in March.".into(),
            },
        )
        .unwrap()
    else {
        panic!("the seed episode must land")
    };
    engine
        .apply_facts(
            &space,
            episode_id,
            &[ExtractedFact {
                subject: "Ana".into(),
                predicate: "moved_to".into(),
                object: "Lisbon".into(),
                confidence: 0.8,
            }],
        )
        .unwrap();

    let app = router(
        engine,
        ServeConfig {
            keys: vec![SpaceKey {
                key: "sk-alice".into(),
                space: "alice".into(),
            }],
        },
    );

    let (_, listed) = call(&app, "GET", "/v1/facts", Some("sk-alice"), None).await;
    let first = &listed["facts"][0];
    assert_eq!(
        first["sources"].as_array().map(|s| s.len()),
        Some(1),
        "the fact must name its source: {listed}"
    );
    assert_eq!(first["sources"][0].as_i64(), Some(episode_id));
}

/// Review over HTTP. The core has taken proposals and settled them since
/// the fact gate landed, the CLI can do it, and the HTTP surface could
/// not: a page built against this server could show a proposal and had
/// no way to accept or reject it.
#[tokio::test]
async fn a_proposal_can_be_listed_and_settled_over_http() {
    use scone_core::llm::ExtractedFact;
    use scone_core::{IngestInput, IngestOutcome, auth};

    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::open(dir.path(), Box::new(HashEmbedder::new(64))).unwrap();
    engine.set_propose_below(Some(1.0)).unwrap();
    let space = auth::resolve(&mut engine, "alice", true).unwrap();
    let IngestOutcome::Ingested { episode_id, .. } = engine
        .ingest(
            &space,
            IngestInput::Note {
                text: "Ana moved to Lisbon in March.".into(),
            },
        )
        .unwrap()
    else {
        panic!("the seed episode must land")
    };
    engine
        .apply_facts(
            &space,
            episode_id,
            &[
                ExtractedFact {
                    subject: "Ana".into(),
                    predicate: "moved_to".into(),
                    object: "Lisbon".into(),
                    confidence: 0.8,
                },
                ExtractedFact {
                    subject: "Ana".into(),
                    predicate: "works_at".into(),
                    object: "Farfetch".into(),
                    confidence: 0.8,
                },
            ],
        )
        .unwrap();

    let app = router(
        engine,
        ServeConfig {
            keys: vec![SpaceKey {
                key: "sk-alice".into(),
                space: "alice".into(),
            }],
        },
    );

    let (status, pending) = call(
        &app,
        "GET",
        "/v1/facts?status=proposed",
        Some("sk-alice"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{pending}");
    let waiting = pending["facts"].as_array().expect("a list of proposals");
    assert_eq!(
        waiting.len(),
        2,
        "both proposals wait for a person: {pending}"
    );
    let accept = waiting[0]["fact_id"].as_i64().unwrap();
    let reject = waiting[1]["fact_id"].as_i64().unwrap();

    let (status, body) = call(
        &app,
        "POST",
        &format!("/v1/facts/{accept}/approve"),
        Some("sk-alice"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let (status, body) = call(
        &app,
        "POST",
        &format!("/v1/facts/{reject}/decline"),
        Some("sk-alice"),
        Some(serde_json::json!({"reason": "the source does not say this"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    // The accepted one holds; the rejected one is gone from the ledger
    // and nothing is left waiting.
    let (_, ledger) = call(&app, "GET", "/v1/facts", Some("sk-alice"), None).await;
    let held = ledger["facts"].as_array().unwrap();
    assert_eq!(held.len(), 1, "only the approved claim holds: {ledger}");
    assert_eq!(held[0]["fact_id"].as_i64(), Some(accept));

    let (_, left) = call(
        &app,
        "GET",
        "/v1/facts?status=proposed",
        Some("sk-alice"),
        None,
    )
    .await;
    assert_eq!(left["facts"].as_array().map(|f| f.len()), Some(0));

    // A proposal that no longer exists is a client's mistake, not ours.
    let (status, _) = call(
        &app,
        "POST",
        "/v1/facts/4242/approve",
        Some("sk-alice"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

/// The Rust HTTP surface is frozen at the engine essentials, and this
/// test is the freeze.
///
/// Decision of 2026-09-06, recorded in memory/DECISIONS.md: there are two
/// HTTP servers over one engine contract, and every route costs twice,
/// plus a conformance test to keep the two honest. The Rust one exists
/// for self-hosting the engine as a single binary with no Python
/// runtime; the product surface, which is conversations, voice,
/// attachments, the grounding audit and distillation, lives in Python
/// and is not chased here.
///
/// Adding a route to serve.rs fails this test on purpose. If the new
/// route is an engine essential, add it to the list below and say why in
/// DECISIONS.md. If it is product surface, it belongs in the Python
/// server instead. What must not happen is the surface widening by
/// accident, one convenience at a time, until parity is claimed that
/// nobody is keeping.
#[test]
fn the_rust_http_surface_stays_frozen_at_the_engine_essentials() {
    let source = include_str!("../src/serve.rs");
    let mut routes: Vec<&str> = source
        .match_indices(".route(\"")
        .map(|(at, marker)| {
            let rest = &source[at + marker.len()..];
            &rest[..rest.find('"').expect("a closing quote on the route path")]
        })
        .collect();
    routes.sort_unstable();
    routes.dedup();

    let frozen = [
        "/v1/capabilities",
        "/v1/episodes",
        "/v1/episodes/{id}",
        "/v1/events",
        "/v1/facts",
        "/v1/facts/{id}/approve",
        "/v1/facts/{id}/close",
        "/v1/facts/{id}/decline",
        "/v1/graph",
        "/v1/profile",
        "/v1/recall",
        "/v1/sources",
        "/v1/status",
        "/v1/tags",
    ];
    assert_eq!(
        routes, frozen,
        "the Rust server's routes changed; see the decision above before widening it"
    );
}

/// Discovery has to describe this server, not a memory of it.
///
/// `facts.review` said false for a day after approve and decline were
/// mounted, which tells a client the workflow is unavailable while the
/// routes sit there answering. A capability contract that drifts from
/// the router is worse than none: a client trusts it and stops probing.
///
/// This ties each claim to the route that backs it by reading the
/// router's own source, so the two cannot part company again.
#[test]
fn every_capability_claim_matches_a_mounted_route() {
    let source = include_str!("../src/serve.rs");
    let mounted = |path: &str| source.contains(&format!(".route(\"{path}\""));
    let claims = |feature: &str| {
        let at = source
            .find(&format!("\"{feature}\": "))
            .unwrap_or_else(|| panic!("{feature} must appear in the capability block"));
        source[at + feature.len() + 4..].starts_with("true")
    };

    for (feature, path) in [
        ("recall", "/v1/recall"),
        ("facts.read", "/v1/facts"),
        ("facts.review", "/v1/facts/{id}/approve"),
        ("facts.close", "/v1/facts/{id}/close"),
        ("events.read", "/v1/events"),
        ("status.read", "/v1/status"),
        ("episodes.list", "/v1/episodes"),
    ] {
        assert_eq!(
            claims(feature),
            mounted(path),
            "capability {feature} and route {path} disagree"
        );
    }

    // Review needs both halves before it may be claimed: a client told
    // it can review, that can accept and not reject, is worse off than
    // one told it cannot.
    assert_eq!(
        claims("facts.review"),
        mounted("/v1/facts/{id}/approve") && mounted("/v1/facts/{id}/decline"),
        "facts.review must mean both approve and decline"
    );
}
