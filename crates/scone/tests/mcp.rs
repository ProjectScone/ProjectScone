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
async fn lists_the_four_memory_tools() {
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
            "memory_recall",
            "memory_store"
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
