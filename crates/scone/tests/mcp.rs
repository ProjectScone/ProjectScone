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

/// The host agent is the reader; it cannot order events or answer "when"
/// if recall hands it undated lines. Benchmarks scored zero on temporal
/// questions once dates were lost, so the date travels with every item.
#[tokio::test]
async fn recalled_memory_carries_its_date() {
    let dir = tempfile::tempdir().unwrap();
    let client = client_for(SconeMcp::new(engine(dir.path()), "agent")).await;
    client
        .call_tool({
            let mut p = CallToolRequestParams::new("memory_store");
            p.arguments = serde_json::json!({"content": "switched the deploy target to fly.io"})
                .as_object()
                .cloned();
            p
        })
        .await
        .unwrap();
    let recalled = client
        .call_tool({
            let mut p = CallToolRequestParams::new("memory_recall");
            p.arguments = serde_json::json!({"query": "deploy target"})
                .as_object()
                .cloned();
            p
        })
        .await
        .unwrap();
    let text = text_of(&recalled);
    let dated = text
        .lines()
        .find(|l| l.starts_with("memory ["))
        .unwrap_or_default();
    let day = &dated[8..18];
    assert!(
        day.len() == 10 && day.chars().filter(|c| *c == '-').count() == 2,
        "recalled memory must be dated YYYY-MM-DD: {text}"
    );
    client.cancel().await.unwrap();
}

/// Date arithmetic reaches the agent through memory_recall (E28 adoption,
/// second surface). The planner grounds events by similarity, so this
/// needs the real embedder, as the temporal operator tests do; with the
/// hash embedder every candidate sits below the anchor floor and the
/// planner would decline for the wrong reason.
#[tokio::test]
async fn recall_computes_date_arithmetic_and_shows_its_working() {
    use scone_core::embed::OnnxEmbedder;
    let dir = tempfile::tempdir().unwrap();
    let cache = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".scone");
    let mut e = Engine::open(dir.path(), Box::new(OnnxEmbedder::new(&cache).unwrap())).unwrap();
    let space = auth::resolve(&mut e, "agent", true).unwrap();
    e.import_episode(
        &space,
        "note",
        "attended the Maundy Thursday service at St Mark's this evening",
        None,
        Some("2024-03-28T19:00:00.000Z"),
    )
    .unwrap();
    e.import_episode(
        &space,
        "note",
        "the staging db password rotates monthly",
        None,
        Some("2024-03-30T09:00:00.000Z"),
    )
    .unwrap();
    let client = client_for(SconeMcp::new(e, "agent")).await;
    let recalled = client
        .call_tool({
            let mut p = CallToolRequestParams::new("memory_recall");
            p.arguments = serde_json::json!({
                "query": "How many days ago did I attend the Maundy Thursday service?",
                "as_of": "2024-04-04T12:00:00.000Z",
                "include_profile": false
            })
            .as_object()
            .cloned();
            p
        })
        .await
        .unwrap();
    let text = text_of(&recalled);
    assert!(
        text.starts_with("computed: 7 days\nderived from: "),
        "{text}"
    );
    assert!(
        text.contains("2024-03-28"),
        "the derivation names the anchor date: {text}"
    );
    assert!(text.contains("memory ["), "the pack still follows: {text}");

    // A question the planner cannot read falls through to the plain pack.
    let plain = client
        .call_tool({
            let mut p = CallToolRequestParams::new("memory_recall");
            p.arguments =
                serde_json::json!({"query": "staging db password", "include_profile": false})
                    .as_object()
                    .cloned();
            p
        })
        .await
        .unwrap();
    assert!(!text_of(&plain).contains("computed:"), "{plain:?}");
    client.cancel().await.unwrap();
}

/// Narrowing beyond tags on the agent's surface: kind, source prefix and
/// the created_at bounds reach the engine as given.
#[tokio::test]
async fn recall_narrows_by_kind_source_prefix_and_dates() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = engine(dir.path());
    let space = auth::resolve(&mut e, "agent", true).unwrap();
    let rows: [(&str, &str, Option<&str>, &str); 3] = [
        (
            "note",
            "deploy runbook: rotate the staging keys first",
            None,
            "2024-01-10T09:00:00.000Z",
        ),
        (
            "file",
            "deploy runbook: rotate the staging keys, then restart",
            Some("/ops/runbooks/deploy.md"),
            "2024-02-10T09:00:00.000Z",
        ),
        (
            "conversation",
            "user: where is the deploy runbook for staging keys?",
            Some("session-42"),
            "2024-04-10T09:00:00.000Z",
        ),
    ];
    for (kind, text, source, day) in rows {
        e.import_episode(&space, kind, text, source, Some(day))
            .unwrap();
    }
    let client = client_for(SconeMcp::new(e, "agent")).await;
    let recall = |args: serde_json::Value| {
        let client = &client;
        async move {
            let result = client
                .call_tool({
                    let mut p = CallToolRequestParams::new("memory_recall");
                    p.arguments = args.as_object().cloned();
                    p
                })
                .await
                .unwrap();
            text_of(&result)
        }
    };
    let base =
        serde_json::json!({"query": "deploy runbook staging keys", "include_profile": false});
    let with = |extra: serde_json::Value| {
        let mut v = base.clone();
        v.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        v
    };
    let all = recall(base.clone()).await;
    assert!(
        all.contains("episode 1") && all.contains("episode 2") && all.contains("episode 3"),
        "{all}"
    );
    let files = recall(with(serde_json::json!({"kind": "file"}))).await;
    assert!(
        files.contains("episode 2") && !files.contains("episode 1") && !files.contains("episode 3"),
        "{files}"
    );
    let session = recall(with(serde_json::json!({"source_prefix": "session-"}))).await;
    assert!(
        session.contains("episode 3") && !session.contains("episode 2"),
        "{session}"
    );
    let early = recall(with(
        serde_json::json!({"until": "2024-01-31T00:00:00.000Z"}),
    ))
    .await;
    assert!(
        early.contains("episode 1") && !early.contains("episode 2"),
        "{early}"
    );
    let late = recall(with(
        serde_json::json!({"since": "2024-03-01T00:00:00.000Z"}),
    ))
    .await;
    assert!(
        late.contains("episode 3") && !late.contains("episode 1"),
        "{late}"
    );
    client.cancel().await.unwrap();
}
