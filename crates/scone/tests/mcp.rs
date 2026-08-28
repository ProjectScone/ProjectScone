#![allow(clippy::unwrap_used)]
//! MCP server tested in-process over a duplex transport — no stdio, no
//! network, no models.
use rmcp::ServiceExt;
use rmcp::model::CallToolRequestParams;
use scone::mcp::SconeMcp;
use scone_core::embed::HashEmbedder;
use scone_core::llm::ExtractedFact;
use scone_core::{Engine, auth};

fn engine(dir: &std::path::Path) -> Engine {
    Engine::open(dir, Box::new(HashEmbedder::new(64))).unwrap()
}

async fn client_for(
    server: SconeMcp,
) -> rmcp::service::RunningService<rmcp::service::RoleClient, ()> {
    let (client_io, server_io) = tokio::io::duplex(1 << 16);
    tokio::spawn(async move {
        let running = server.serve(server_io).await.unwrap();
        let _ = running.waiting().await;
    });
    ().serve(client_io).await.unwrap()
}

fn text_of(result: &rmcp::model::CallToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|c| c.as_text().map(|t| t.text.clone()))
        .collect::<Vec<_>>()
        .join("\n")
}

#[tokio::test]
async fn lists_the_memory_tools() {
    let dir = tempfile::tempdir().unwrap();
    let client = client_for(SconeMcp::new(engine(dir.path()), "agent")).await;
    let tools = client.list_all_tools().await.unwrap();
    let mut names: Vec<_> = tools.iter().map(|t| t.name.to_string()).collect();
    names.sort();
    assert_eq!(
        names,
        [
            "memory_facts_about",
            "memory_forget",
            "memory_pending",
            "memory_recall",
            "memory_store",
            "memory_store_facts"
        ]
    );
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn store_then_recall_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let client = client_for(SconeMcp::new(engine(dir.path()), "agent")).await;
    let stored = client
        .call_tool({
            let mut p = CallToolRequestParams::new("memory_store");
            p.arguments = serde_json::json!({"content": "the staging db password rotates monthly"})
                .as_object()
                .cloned();
            p
        })
        .await
        .unwrap();
    assert!(text_of(&stored).contains("stored"), "{stored:?}");
    let recalled = client
        .call_tool({
            let mut p = CallToolRequestParams::new("memory_recall");
            p.arguments = serde_json::json!({"query": "staging db password"})
                .as_object()
                .cloned();
            p
        })
        .await
        .unwrap();
    assert!(
        text_of(&recalled).contains("rotates monthly"),
        "{recalled:?}"
    );
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn oversized_input_is_rejected_as_tool_error() {
    let dir = tempfile::tempdir().unwrap();
    let client = client_for(SconeMcp::new(engine(dir.path()), "agent")).await;
    let result = client
        .call_tool({
            let mut p = CallToolRequestParams::new("memory_store");
            p.arguments = serde_json::json!({"content": "x".repeat(100_001)})
                .as_object()
                .cloned();
            p
        })
        .await
        .unwrap();
    assert_eq!(
        result.is_error,
        Some(true),
        "bounded from the first commit (L-10)"
    );
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn facts_about_and_forget_are_space_scoped() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = engine(dir.path());
    let space = auth::resolve(&mut e, "agent", true).unwrap();
    let scone_core::IngestOutcome::Ingested { episode_id, .. } = e
        .ingest(
            &space,
            scone_core::IngestInput::Note {
                text: "seed".into(),
            },
        )
        .unwrap()
    else {
        panic!()
    };
    e.apply_facts(
        &space,
        episode_id,
        &[ExtractedFact {
            subject: "mark".into(),
            predicate: "prefers".into(),
            object: "bun".into(),
            confidence: 0.9,
        }],
    )
    .unwrap();
    let client = client_for(SconeMcp::new(e, "agent")).await;
    let about = client
        .call_tool({
            let mut p = CallToolRequestParams::new("memory_facts_about");
            p.arguments = serde_json::json!({"entity": "mark"}).as_object().cloned();
            p
        })
        .await
        .unwrap();
    assert!(text_of(&about).contains("bun"), "{about:?}");
    let forgotten = client
        .call_tool({
            let mut p = CallToolRequestParams::new("memory_forget");
            p.arguments = serde_json::json!({"fact_id": 1, "reason": "user asked"})
                .as_object()
                .cloned();
            p
        })
        .await
        .unwrap();
    assert!(text_of(&forgotten).contains("closed"), "{forgotten:?}");
    let after = client
        .call_tool({
            let mut p = CallToolRequestParams::new("memory_facts_about");
            p.arguments = serde_json::json!({"entity": "mark"}).as_object().cloned();
            p
        })
        .await
        .unwrap();
    assert!(
        !text_of(&after).contains("bun"),
        "closed facts leave the active view"
    );
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn recall_includes_profile_sections() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = engine(dir.path());
    let space = auth::resolve(&mut e, "agent", true).unwrap();
    let scone_core::IngestOutcome::Ingested { episode_id, .. } = e
        .ingest(
            &space,
            scone_core::IngestInput::Note {
                text: "seed activity".into(),
            },
        )
        .unwrap()
    else {
        panic!()
    };
    e.apply_facts(
        &space,
        episode_id,
        &[ExtractedFact {
            subject: "mark".into(),
            predicate: "lives_in".into(),
            object: "austin".into(),
            confidence: 0.9,
        }],
    )
    .unwrap();
    let client = client_for(SconeMcp::new(e, "agent")).await;
    let result = client
        .call_tool({
            let mut p = CallToolRequestParams::new("memory_recall");
            p.arguments = serde_json::json!({"query": "anything at all"})
                .as_object()
                .cloned();
            p
        })
        .await
        .unwrap();
    let text = text_of(&result);
    assert!(text.contains("## Profile"), "{text}");
    assert!(text.contains("lives_in austin"), "{text}");
    assert!(text.contains("## Recent activity"), "{text}");
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn store_accepts_tags_and_recall_focuses_on_them() {
    let dir = tempfile::tempdir().unwrap();
    let client = client_for(SconeMcp::new(engine(dir.path()), "agent")).await;
    for (content, tags) in [
        ("the api redesign ships tuesday", r#"["work"]"#),
        ("the sourdough recipe needs more salt", r#"["baking"]"#),
    ] {
        let stored = client
            .call_tool({
                let mut p = CallToolRequestParams::new("memory_store");
                p.arguments = serde_json::from_str::<serde_json::Value>(&format!(
                    r#"{{"content": "{content}", "tags": {tags}}}"#
                ))
                .unwrap()
                .as_object()
                .cloned();
                p
            })
            .await
            .unwrap();
        assert_ne!(stored.is_error, Some(true), "{stored:?}");
    }
    let focused = client
        .call_tool({
            let mut p = CallToolRequestParams::new("memory_recall");
            p.arguments = serde_json::json!({
                "query": "what ships", "tags": ["work"], "include_profile": false
            })
            .as_object()
            .cloned();
            p
        })
        .await
        .unwrap();
    let text = text_of(&focused);
    assert!(text.contains("redesign"), "{text}");
    assert!(
        !text.contains("sourdough"),
        "tag focus must exclude: {text}"
    );
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn agent_driven_distillation_loop_works_without_any_llm() {
    let dir = tempfile::tempdir().unwrap();
    let client = client_for(SconeMcp::new(engine(dir.path()), "agent")).await;
    // Store creates a pending episode (no LLM configured on the server).
    client
        .call_tool({
            let mut p = CallToolRequestParams::new("memory_store");
            p.arguments = serde_json::json!({"content": "mark moved the standup to 9am"})
                .as_object()
                .cloned();
            p
        })
        .await
        .unwrap();
    // The agent pulls pending work.
    let pending = client
        .call_tool({
            let mut p = CallToolRequestParams::new("memory_pending");
            p.arguments = serde_json::json!({}).as_object().cloned();
            p
        })
        .await
        .unwrap();
    let text = text_of(&pending);
    assert!(text.contains("standup"), "{text}");
    assert!(text.contains("episode 1"), "{text}");
    // The agent submits facts it extracted with its own model.
    let stored = client
        .call_tool({
            let mut p = CallToolRequestParams::new("memory_store_facts");
            p.arguments = serde_json::json!({
                "episode_id": 1,
                "facts": [
                    {"subject": "standup", "predicate": "moved_to", "object": "9am", "confidence": 0.9}
                ]
            })
            .as_object()
            .cloned();
            p
        })
        .await
        .unwrap();
    assert_ne!(stored.is_error, Some(true), "{stored:?}");
    assert!(text_of(&stored).contains("1 fact"), "{stored:?}");
    // Queue drained; facts queryable.
    let pending = client
        .call_tool({
            let mut p = CallToolRequestParams::new("memory_pending");
            p.arguments = serde_json::json!({}).as_object().cloned();
            p
        })
        .await
        .unwrap();
    assert!(
        text_of(&pending).contains("nothing pending"),
        "{:?}",
        text_of(&pending)
    );
    let about = client
        .call_tool({
            let mut p = CallToolRequestParams::new("memory_facts_about");
            p.arguments = serde_json::json!({"entity": "standup"})
                .as_object()
                .cloned();
            p
        })
        .await
        .unwrap();
    assert!(text_of(&about).contains("9am"), "{:?}", text_of(&about));
    client.cancel().await.unwrap();
}
