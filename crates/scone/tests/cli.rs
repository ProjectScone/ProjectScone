#![allow(clippy::unwrap_used)]
use assert_cmd::Command;

fn scone(dir: &std::path::Path) -> Command {
    let mut c = Command::cargo_bin("scone").unwrap();
    // Tests stay hermetic: the hash embedder needs no model download.
    c.arg("--data-dir").arg(dir).args(["--embedder", "hash"]);
    c
}

#[test]
fn doctor_rebuild_recovers_from_deleted_indexes() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "memory survives rebuilds"])
        .assert()
        .success();
    std::fs::remove_dir_all(dir.path().join("fts")).unwrap();
    scone(dir.path())
        .args(["doctor", "--rebuild"])
        .assert()
        .success()
        .stdout(predicates::str::contains("rebuilt"));
    scone(dir.path())
        .args(["search", "survives"])
        .assert()
        .success()
        .stdout(predicates::str::contains("survives"));
}

#[test]
fn add_then_search_finds_the_note() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "the borrow checker enforces ownership"])
        .assert()
        .success()
        .stdout(predicates::str::contains("ingested"));
    scone(dir.path())
        .args(["search", "borrow checker"])
        .assert()
        .success()
        .stdout(predicates::str::contains("borrow checker"));
}

#[test]
fn duplicate_add_reports_deduplicated() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "same note"])
        .assert()
        .success();
    scone(dir.path())
        .args(["add", "--note", "same note"])
        .assert()
        .success()
        .stdout(predicates::str::contains("deduplicated"));
}

#[test]
fn status_counts_episodes() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "one note"])
        .assert()
        .success();
    scone(dir.path())
        .arg("status")
        .assert()
        .success()
        .stdout(predicates::str::contains("episodes: 1"));
}

#[test]
fn spaces_are_isolated() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "secret in default"])
        .assert()
        .success();
    scone(dir.path())
        .args(["--space", "other", "search", "secret"])
        .assert()
        .success()
        .stdout(predicates::str::contains("no results"));
}

#[test]
fn bad_space_name_is_a_clean_error() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["--space", "BAD NAME!", "add", "--note", "x"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("space name"));
}

#[test]
fn add_reads_files() {
    let dir = tempfile::tempdir().unwrap();
    let f = dir.path().join("doc.md");
    std::fs::write(&f, "# notes\n\nscone is a memory engine").unwrap();
    scone(dir.path()).arg("add").arg(&f).assert().success();
    scone(dir.path())
        .args(["search", "memory engine"])
        .assert()
        .success()
        .stdout(predicates::str::contains("doc.md"));
}

const FAKE_FACTS: &str = r#"[{"subject":"mark","predicate":"prefers","object":"bun"}]"#;
const FAKE_FACTS_2: &str = r#"[{"subject":"mark","predicate":"prefers","object":"pnpm"}]"#;

#[test]
fn status_says_semantic_lane_is_paused_without_llm() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "a note"])
        .assert()
        .success();
    scone(dir.path())
        .arg("status")
        .assert()
        .success()
        .stdout(predicates::str::contains("paused"))
        .stdout(predicates::str::contains("1 pending"));
}

#[test]
fn distill_extracts_facts_and_lists_them() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "mark said he prefers bun"])
        .assert()
        .success();
    scone(dir.path())
        .env("SCONE_FAKE_FACTS", FAKE_FACTS)
        .args(["--llm", "fake", "distill"])
        .assert()
        .success()
        .stdout(predicates::str::contains("1 episode"));
    scone(dir.path())
        .args(["facts", "list"])
        .assert()
        .success()
        .stdout(predicates::str::contains("mark prefers bun"));
}

#[test]
fn contradiction_history_is_visible_and_explained() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "first claim"])
        .assert()
        .success();
    scone(dir.path())
        .env("SCONE_FAKE_FACTS", FAKE_FACTS_2)
        .args(["--llm", "fake", "distill"])
        .assert()
        .success();
    scone(dir.path())
        .args(["add", "--note", "second claim"])
        .assert()
        .success();
    scone(dir.path())
        .env("SCONE_FAKE_FACTS", FAKE_FACTS)
        .args(["--llm", "fake", "distill"])
        .assert()
        .success();
    let out = scone(dir.path()).args(["facts", "list"]).assert().success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    assert!(
        stdout.contains("bun") && !stdout.contains("pnpm"),
        "{stdout}"
    );
    scone(dir.path())
        .args(["facts", "list", "--all"])
        .assert()
        .success()
        .stdout(predicates::str::contains("pnpm"))
        .stdout(predicates::str::contains("superseded"));
}

#[test]
fn facts_link_relates_two_facts_and_links_reads_them_back() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "mark prefers bun today"])
        .assert()
        .success();
    scone(dir.path())
        .env("SCONE_FAKE_FACTS", FAKE_FACTS)
        .args(["--llm", "fake", "distill"])
        .assert()
        .success();
    scone(dir.path())
        .args(["add", "--note", "mark prefers pnpm now"])
        .assert()
        .success();
    scone(dir.path())
        .env("SCONE_FAKE_FACTS", FAKE_FACTS_2)
        .args(["--llm", "fake", "distill"])
        .assert()
        .success();
    scone(dir.path())
        .args(["facts", "link", "2", "1", "supports"])
        .assert()
        .success()
        .stdout(predicates::str::contains("fact 2 supports fact 1"));
    scone(dir.path())
        .args(["facts", "links", "1"])
        .assert()
        .success()
        .stdout(predicates::str::contains("fact 2 supports fact 1"));
    scone(dir.path())
        .args(["facts", "link", "1", "1", "supports"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("itself"));
}

#[test]
fn facts_why_shows_provenance_and_close_takes_reason() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "mark prefers bun today"])
        .assert()
        .success();
    scone(dir.path())
        .env("SCONE_FAKE_FACTS", FAKE_FACTS)
        .args(["--llm", "fake", "distill"])
        .assert()
        .success();
    scone(dir.path())
        .args(["facts", "why", "1"])
        .assert()
        .success()
        .stdout(predicates::str::contains("episode"));
    scone(dir.path())
        .args(["facts", "close", "1", "--reason", "no longer true"])
        .assert()
        .success();
    scone(dir.path())
        .args(["facts", "list", "--all"])
        .assert()
        .success()
        .stdout(predicates::str::contains("no longer true"));
}

#[test]
fn ask_without_llm_prints_context_and_pause_notice() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "the deploy key lives in the vault"])
        .assert()
        .success();
    scone(dir.path())
        .args(["ask", "where is the deploy key?"])
        .assert()
        .success()
        .stdout(predicates::str::contains("deploy key"))
        .stdout(predicates::str::contains("paused"));
}

#[test]
fn ask_with_configured_llm_answers_from_the_stub() {
    use std::io::{Read, Write};
    let dir = tempfile::tempdir().unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || {
        let (mut sock, _) = listener.accept().unwrap();
        let mut buf = [0u8; 65536];
        let mut req = String::new();
        loop {
            let n = sock.read(&mut buf).unwrap();
            req.push_str(&String::from_utf8_lossy(&buf[..n]));
            if req.contains("\r\n\r\n") && req.trim_end().ends_with('}') {
                break;
            }
        }
        let body = r#"{"choices":[{"message":{"content":"in the vault, per your note"}}]}"#;
        let resp = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        sock.write_all(resp.as_bytes()).unwrap();
    });
    std::fs::create_dir_all(dir.path()).unwrap();
    std::fs::write(
        dir.path().join("config.toml"),
        format!("[llm]\nprovider = \"openai\"\nbase_url = \"http://{addr}\"\nmodel = \"stub\"\n"),
    )
    .unwrap();
    scone(dir.path())
        .args(["add", "--note", "the deploy key lives in the vault"])
        .assert()
        .success();
    scone(dir.path())
        .args(["ask", "where is the deploy key?"])
        .assert()
        .success()
        .stdout(predicates::str::contains("in the vault, per your note"));
    handle.join().unwrap();
}

#[test]
fn search_shows_facts_and_supports_as_of_time_travel() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "first claim about tools"])
        .assert()
        .success();
    scone(dir.path())
        .env("SCONE_FAKE_FACTS", FAKE_FACTS_2)
        .args(["--llm", "fake", "distill"])
        .assert()
        .success();
    scone(dir.path())
        .args(["add", "--note", "second claim about tools"])
        .assert()
        .success();
    scone(dir.path())
        .env("SCONE_FAKE_FACTS", FAKE_FACTS)
        .args(["--llm", "fake", "distill"])
        .assert()
        .success();
    // Pin intervals to known dates so as-of is deterministic.
    let db = rusqlite::Connection::open(dir.path().join("scone.db")).unwrap();
    db.execute(
        "UPDATE facts SET valid_from='2026-01-01T00:00:00Z', valid_until='2026-06-01T00:00:00Z'
         WHERE object='pnpm'",
        [],
    )
    .unwrap();
    db.execute(
        "UPDATE facts SET valid_from='2026-06-01T00:00:00Z' WHERE object='bun'",
        [],
    )
    .unwrap();
    drop(db);
    scone(dir.path())
        .args(["search", "mark prefers"])
        .assert()
        .success()
        .stdout(predicates::str::contains("mark prefers bun"));
    scone(dir.path())
        .args(["search", "mark prefers", "--as-of", "2026-03-15T00:00:00Z"])
        .assert()
        .success()
        .stdout(predicates::str::contains("mark prefers pnpm"));
}

#[test]
fn export_import_moves_memory_between_stores() {
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    scone(a.path())
        .args(["add", "--note", "portable memory survives moves"])
        .assert()
        .success();
    let out = scone(a.path()).arg("export").assert().success();
    let jsonl = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    assert!(jsonl.contains("portable memory"));
    let f = a.path().join("dump.jsonl");
    std::fs::write(&f, &jsonl).unwrap();
    scone(b.path())
        .arg("import")
        .arg(&f)
        .assert()
        .success()
        .stdout(predicates::str::contains("1 episode"));
    scone(b.path())
        .args(["search", "portable memory"])
        .assert()
        .success()
        .stdout(predicates::str::contains("survives moves"));
}

#[test]
fn spaces_lists_all_spaces() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "one"])
        .assert()
        .success();
    scone(dir.path())
        .args(["--space", "work", "add", "--note", "two"])
        .assert()
        .success();
    scone(dir.path())
        .arg("spaces")
        .assert()
        .success()
        .stdout(predicates::str::contains("default"))
        .stdout(predicates::str::contains("work"));
}

#[test]
fn watch_once_ingests_a_directory() {
    let dir = tempfile::tempdir().unwrap();
    let notes = tempfile::tempdir().unwrap();
    std::fs::write(notes.path().join("idea.md"), "watch mode found this idea").unwrap();
    scone(dir.path())
        .arg("watch")
        .arg(notes.path())
        .arg("--once")
        .assert()
        .success()
        .stdout(predicates::str::contains("ingested 1"));
    scone(dir.path())
        .args(["search", "idea"])
        .assert()
        .success()
        .stdout(predicates::str::contains("found this idea"));
}

#[test]
fn daemon_once_scans_and_distills() {
    let dir = tempfile::tempdir().unwrap();
    let notes = tempfile::tempdir().unwrap();
    std::fs::write(notes.path().join("fact.md"), "mark uses scone daily").unwrap();
    scone(dir.path())
        .env(
            "SCONE_FAKE_FACTS",
            r#"[{"subject":"mark","predicate":"uses","object":"scone"}]"#,
        )
        .args(["--llm", "fake", "daemon", "--once", "--watch"])
        .arg(notes.path())
        .assert()
        .success()
        .stdout(predicates::str::contains("distilled"));
    scone(dir.path())
        .args(["facts", "list"])
        .assert()
        .success()
        .stdout(predicates::str::contains("mark uses scone"));
}

#[test]
fn profile_shows_identity_and_recent_activity() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "team switched to bun last sprint"])
        .assert()
        .success();
    scone(dir.path())
        .env("SCONE_FAKE_FACTS", FAKE_FACTS)
        .args(["--llm", "fake", "distill"])
        .assert()
        .success();
    scone(dir.path())
        .arg("profile")
        .assert()
        .success()
        .stdout(predicates::str::contains("mark prefers bun"))
        .stdout(predicates::str::contains("last sprint"));
}

#[test]
fn search_reports_context_economy() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..5 {
        scone(dir.path())
            .args([
                "add",
                "--note",
                &format!("filler note number {i} about various topics"),
            ])
            .assert()
            .success();
    }
    scone(dir.path())
        .args(["add", "--note", "the tokens saved line matters"])
        .assert()
        .success();
    scone(dir.path())
        .args(["search", "tokens saved", "--limit", "2"])
        .assert()
        .success()
        .stdout(predicates::str::contains("% saved"));
}

#[test]
fn tagging_focuses_search_and_curates_sources() {
    let dir = tempfile::tempdir().unwrap();
    let notes = tempfile::tempdir().unwrap();
    std::fs::write(
        notes.path().join("paper.md"),
        "the retrieval paper describes fusion",
    )
    .unwrap();
    scone(dir.path())
        .args([
            "add",
            "--note",
            "meeting notes about fusion budget",
            "--tag",
            "meeting",
        ])
        .assert()
        .success();
    scone(dir.path())
        .arg("watch")
        .arg(notes.path())
        .args(["--once", "--tag", "research"])
        .assert()
        .success();
    // Auto extension tag curates the source kind.
    scone(dir.path())
        .arg("tags")
        .assert()
        .success()
        .stdout(predicates::str::contains("meeting"))
        .stdout(predicates::str::contains("research"))
        .stdout(predicates::str::contains("md"));
    scone(dir.path())
        .args(["search", "fusion", "--tag", "research"])
        .assert()
        .success()
        .stdout(predicates::str::contains("retrieval paper"));
    let out = scone(dir.path())
        .args(["search", "fusion", "--tag", "meeting"])
        .assert()
        .success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    assert!(
        stdout.contains("budget") && !stdout.contains("retrieval paper"),
        "{stdout}"
    );
}

#[test]
fn setup_claude_desktop_writes_mcp_config_preserving_existing() {
    let dir = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();
    let cfg_dir = if cfg!(target_os = "macos") {
        home.path().join("Library/Application Support/Claude")
    } else {
        home.path().join(".config/Claude")
    };
    std::fs::create_dir_all(&cfg_dir).unwrap();
    std::fs::write(
        cfg_dir.join("claude_desktop_config.json"),
        r#"{"mcpServers": {"other": {"command": "keepme"}}, "theme": "dark"}"#,
    )
    .unwrap();
    scone(dir.path())
        .env("HOME", home.path())
        .args(["setup", "claude-desktop"])
        .assert()
        .success()
        .stdout(predicates::str::contains("claude_desktop_config.json"));
    let written: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(cfg_dir.join("claude_desktop_config.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        written["mcpServers"]["other"]["command"], "keepme",
        "existing servers kept"
    );
    assert_eq!(written["theme"], "dark", "unrelated settings kept");
    let scone_cfg = &written["mcpServers"]["scone"];
    assert!(scone_cfg["command"].as_str().unwrap().contains("scone"));
    assert!(
        scone_cfg["args"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a == "mcp")
    );
}

#[test]
fn setup_claude_code_invokes_the_claude_cli() {
    let dir = tempfile::tempdir().unwrap();
    let bin = tempfile::tempdir().unwrap();
    let log = bin.path().join("invocations.log");
    let shim = bin.path().join("claude");
    std::fs::write(
        &shim,
        format!("#!/bin/sh\necho \"$@\" >> {}\n", log.display()),
    )
    .unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&shim, std::fs::Permissions::from_mode(0o755)).unwrap();
    let path = format!(
        "{}:{}",
        bin.path().display(),
        std::env::var("PATH").unwrap()
    );
    scone(dir.path())
        .env("PATH", &path)
        .args(["setup", "claude-code"])
        .assert()
        .success()
        .stdout(predicates::str::contains("registered"));
    let logged = std::fs::read_to_string(&log).unwrap();
    assert!(logged.contains("mcp add scone"), "{logged}");
    assert!(logged.contains("mcp"), "{logged}");
}

#[test]
fn hook_session_start_injects_profile_context() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "this project uses tokio and axum"])
        .assert()
        .success();
    scone(dir.path())
        .env(
            "SCONE_FAKE_FACTS",
            r#"[{"subject":"project","predicate":"uses","object":"axum"}]"#,
        )
        .args(["--llm", "fake", "distill"])
        .assert()
        .success();
    scone(dir.path())
        .args(["hook", "session-start"])
        .write_stdin(r#"{"session_id": "s1", "cwd": "/tmp"}"#)
        .assert()
        .success()
        .stdout(predicates::str::contains("project uses axum"));
}

#[test]
fn hook_user_prompt_injects_relevant_memory_and_fails_open() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "the deploy password rotates every monday"])
        .assert()
        .success();
    scone(dir.path())
        .args(["hook", "user-prompt"])
        .write_stdin(r#"{"prompt": "when does the deploy password rotate?"}"#)
        .assert()
        .success()
        .stdout(predicates::str::contains("rotates every monday"));
    // Garbage stdin must not break the session: exit 0, empty stdout.
    let out = scone(dir.path())
        .args(["hook", "user-prompt"])
        .write_stdin("not json at all")
        .assert()
        .success();
    assert!(out.get_output().stdout.is_empty());
}

#[test]
fn hook_session_end_captures_the_transcript() {
    let dir = tempfile::tempdir().unwrap();
    let transcript = dir.path().join("transcript.jsonl");
    std::fs::write(&transcript, concat!(
        r#"{"type":"user","message":{"role":"user","content":[{"type":"text","text":"we decided to use blake3 for hashing"}]}}"#, "\n",
        r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"noted, blake3 it is"}]}}"#, "\n",
        r#"{"type":"other","ignored":true}"#, "\n",
    )).unwrap();
    scone(dir.path())
        .args(["hook", "session-end"])
        .write_stdin(format!(
            r#"{{"transcript_path": "{}"}}"#,
            transcript.display()
        ))
        .assert()
        .success();
    scone(dir.path())
        .args(["search", "blake3 hashing"])
        .assert()
        .success()
        .stdout(predicates::str::contains("blake3"));
    scone(dir.path())
        .arg("tags")
        .assert()
        .success()
        .stdout(predicates::str::contains("claude-code"));
}

#[test]
fn setup_claude_code_hooks_wires_project_settings() {
    let dir = tempfile::tempdir().unwrap();
    let project = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project.path().join(".claude")).unwrap();
    std::fs::write(
        project.path().join(".claude/settings.json"),
        r#"{"permissions": {"allow": ["Bash(ls:*)"]}}"#,
    )
    .unwrap();
    scone(dir.path())
        .current_dir(project.path())
        .args(["setup", "claude-code-hooks"])
        .assert()
        .success()
        .stdout(predicates::str::contains("settings.json"));
    let written: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(project.path().join(".claude/settings.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        written["permissions"]["allow"][0], "Bash(ls:*)",
        "existing settings kept"
    );
    for event in ["SessionStart", "UserPromptSubmit", "SessionEnd"] {
        let cmd = written["hooks"][event][0]["hooks"][0]["command"]
            .as_str()
            .unwrap_or_default();
        assert!(cmd.contains("hook"), "{event}: {cmd}");
        assert!(cmd.contains("scone"), "{event}: {cmd}");
    }
}

#[test]
fn add_url_ingests_page_text_with_domain_tags() {
    use std::io::{Read, Write};
    let dir = tempfile::tempdir().unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || {
        let (mut sock, _) = listener.accept().unwrap();
        let mut buf = [0u8; 8192];
        let _ = sock.read(&mut buf).unwrap();
        let body = "<html><head><title>Q3 Plan</title></head><body>\
            <nav>ignore this chrome</nav>\
            <h1>Quarterly Plan</h1><p>The quarterly plan targets three regions.</p>\
            <script>var junk = 1;</script></body></html>";
        let resp = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/html\r\ncontent-length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        sock.write_all(resp.as_bytes()).unwrap();
    });
    scone(dir.path())
        .args([
            "add",
            "--url",
            &format!("http://{addr}/plan"),
            "--tag",
            "planning",
        ])
        .assert()
        .success()
        .stdout(predicates::str::contains("ingested"));
    handle.join().unwrap();
    scone(dir.path())
        .args(["search", "quarterly plan regions"])
        .assert()
        .success()
        .stdout(predicates::str::contains("three regions"));
    scone(dir.path())
        .arg("tags")
        .assert()
        .success()
        .stdout(predicates::str::contains("url"))
        .stdout(predicates::str::contains("planning"))
        .stdout(predicates::str::contains("127.0.0.1"));
    // Script junk must not be ingested as content.
    let out = scone(dir.path())
        .args(["search", "var junk"])
        .assert()
        .success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    assert!(!stdout.contains("var junk = 1"), "{stdout}");
}

/// Re-injecting the same memory on every prompt is silent waste: the
/// agent already has it, and the user pays for it again each turn.
#[test]
fn hook_user_prompt_does_not_repeat_itself_within_a_session() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "the deploy password rotates every monday"])
        .assert()
        .success();
    let prompt = r#"{"session_id":"s1","prompt":"when does the deploy password rotate?"}"#;

    let first = scone(dir.path())
        .args(["hook", "user-prompt"])
        .write_stdin(prompt)
        .assert()
        .success();
    assert!(
        String::from_utf8_lossy(&first.get_output().stdout).contains("rotates every monday"),
        "the first prompt of a session gets the memory"
    );

    let second = scone(dir.path())
        .args(["hook", "user-prompt"])
        .write_stdin(prompt)
        .assert()
        .success();
    assert!(
        second.get_output().stdout.is_empty(),
        "the same memory must not be paid for twice: {:?}",
        String::from_utf8_lossy(&second.get_output().stdout)
    );

    // A different session has its own context and starts fresh.
    let other = scone(dir.path())
        .args(["hook", "user-prompt"])
        .write_stdin(r#"{"session_id":"s2","prompt":"when does the deploy password rotate?"}"#)
        .assert()
        .success();
    assert!(
        String::from_utf8_lossy(&other.get_output().stdout).contains("rotates every monday"),
        "a new session has not seen it yet"
    );
}

/// A session id arrives from the host agent, so it is untrusted input
/// and must never steer a write outside the data directory.
#[test]
fn hook_session_state_cannot_escape_the_data_directory() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "the deploy password rotates every monday"])
        .assert()
        .success();
    scone(dir.path())
        .args(["hook", "user-prompt"])
        .write_stdin(
            r#"{"session_id":"../../../../tmp/scone-escape","prompt":"when does the deploy password rotate?"}"#,
        )
        .assert()
        .success();
    assert!(
        !std::path::Path::new("/tmp/scone-escape.txt").exists(),
        "a traversing session id must not write outside the data dir"
    );
}

const FAKE_UNSURE: &str =
    r#"[{"subject":"mark","predicate":"lives_in","object":"lisbon","confidence":0.55}]"#;

/// With --propose-below, an unsure extraction waits for a person: listed
/// under `facts pending`, absent from `facts list`, and active only after
/// `facts approve`. Without the flag the same extraction is active at once.
#[test]
fn unsure_facts_wait_for_approval_when_a_gate_is_set() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "mark moved to lisbon in march"])
        .assert()
        .success();
    scone(dir.path())
        .env("SCONE_FAKE_FACTS", FAKE_UNSURE)
        .args(["--llm", "fake", "--propose-below", "0.7", "distill"])
        .assert()
        .success();
    scone(dir.path())
        .args(["facts", "list"])
        .assert()
        .success()
        .stdout(predicates::str::contains("no facts"));
    scone(dir.path())
        .args(["facts", "pending"])
        .assert()
        .success()
        .stdout(predicates::str::contains(
            "[1] mark lives_in lisbon  (conf 0.55, proposed",
        ));
    scone(dir.path())
        .args(["facts", "approve", "1"])
        .assert()
        .success()
        .stdout(predicates::str::contains(
            "approved fact 1 (0 older fact(s) closed)",
        ));
    scone(dir.path())
        .args(["facts", "list"])
        .assert()
        .success()
        .stdout(predicates::str::contains("mark lives_in lisbon"))
        .stdout(predicates::str::contains("active (approved)"));
    scone(dir.path())
        .args(["facts", "pending"])
        .assert()
        .success()
        .stdout(predicates::str::contains("nothing pending"));
    scone(dir.path())
        .args(["facts", "decline", "1", "--reason", "no"])
        .assert()
        .failure();
    scone(dir.path())
        .args(["--propose-below", "1.5", "facts", "pending"])
        .assert()
        .failure()
        .stderr(predicates::str::contains(
            "propose_below must be within 0..=1",
        ));
}

#[test]
fn declining_a_proposal_keeps_it_out_and_says_why() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "mark might like green"])
        .assert()
        .success();
    scone(dir.path())
        .env("SCONE_FAKE_FACTS", FAKE_UNSURE)
        .args(["--llm", "fake", "--propose-below", "0.9", "distill"])
        .assert()
        .success();
    scone(dir.path())
        .args(["facts", "decline", "1", "--reason", "a guess"])
        .assert()
        .success()
        .stdout(predicates::str::contains("declined fact 1: a guess"));
    scone(dir.path())
        .args(["facts", "list", "--all"])
        .assert()
        .success()
        .stdout(predicates::str::contains("declined (a guess)"));
    scone(dir.path())
        .args(["facts", "list"])
        .assert()
        .success()
        .stdout(predicates::str::contains("no facts"));
}

#[test]
fn without_a_gate_an_unsure_fact_is_active_at_once() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["add", "--note", "mark moved to lisbon in march"])
        .assert()
        .success();
    scone(dir.path())
        .env("SCONE_FAKE_FACTS", FAKE_UNSURE)
        .args(["--llm", "fake", "distill"])
        .assert()
        .success();
    scone(dir.path())
        .args(["facts", "list"])
        .assert()
        .success()
        .stdout(predicates::str::contains("mark lives_in lisbon"));
}

/// Contextual code embedding (on by default since E33) changes what the
/// vector sees, never what is stored. Two files with unrelated bodies and
/// a query that is one file's name: the lexical lane finds that file
/// through its path either way, so the observable is the vector lane's
/// margin. With the prefix embedded the other file falls well behind;
/// with --no-contextual-code the two are a near tie.
#[test]
fn contextual_code_is_the_default_widens_the_vector_margin_and_stores_raw_bytes() {
    let margin = |flags: &[&str]| -> (f64, String) {
        let dir = tempfile::tempdir().unwrap();
        let other = dir.path().join("other.rs");
        std::fs::write(&other, "fn twice(y: u32) -> u32 {\n    y * 2\n}\n").unwrap();
        let file = dir.path().join("widget.rs");
        std::fs::write(
            &file,
            "/// Adds one.\nfn add_one(x: u32) -> u32 {\n    x + 1\n}\n",
        )
        .unwrap();
        for path in [&other, &file] {
            scone(dir.path())
                .args(flags)
                .arg("add")
                .arg(path)
                .assert()
                .success();
        }
        let out = scone(dir.path())
            .args(["search", "widget.rs"])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let out = String::from_utf8(out).unwrap();
        let score_of = |needle: &str| -> f64 {
            out.lines()
                .find(|l| l.contains(needle))
                .and_then(|l| l.split_whitespace().next())
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| panic!("no scored line for {needle}: {out}"))
        };
        (score_of("x + 1") - score_of("y * 2"), out)
    };
    let (by_default, out) = margin(&[]);
    assert!(
        by_default > 0.2,
        "the embedded file name should separate the two well beyond a tie: {out}"
    );
    assert!(
        !out.contains("widget.rs | "),
        "the prefix is embedded, not stored: {out}"
    );
    let (switched_off, out) = margin(&["--no-contextual-code"]);
    assert!(
        switched_off < by_default - 0.1,
        "without the prefix the vector lane cannot tell the files apart by name: \
         {switched_off} against {by_default}\n{out}"
    );
}

/// search narrows by kind, source prefix and a date window, the same four
/// filters the core, both MCP servers and the Python CLI already take.
/// Seeded through import so every episode has a real created_at: --since
/// and --until bound that, while --as-of stays about fact validity.
#[test]
fn search_narrows_by_kind_source_and_date_window() {
    let dir = tempfile::tempdir().unwrap();
    let dump = dir.path().join("seed.jsonl");
    std::fs::write(
        &dump,
        concat!(
            r#"{"type":"episode","kind":"note","content":"the harbour crane was repainted","source":"notes://town","created_at":"2024-01-10T00:00:00Z"}"#,
            "\n",
            r#"{"type":"episode","kind":"file","content":"the harbour crane needs paint","source":"file:///docs/harbour.md","created_at":"2024-06-10T00:00:00Z"}"#,
            "\n",
            r#"{"type":"episode","kind":"file","content":"the harbour crane was inspected","source":"file:///archive/harbour.md","created_at":"2023-01-10T00:00:00Z"}"#,
            "\n",
        ),
    )
    .unwrap();
    scone(dir.path())
        .args(["import"])
        .arg(&dump)
        .assert()
        .success();

    let found = |args: &[&str]| -> String {
        let out = scone(dir.path())
            .args(["search", "harbour crane"])
            .args(args)
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        String::from_utf8(out).unwrap()
    };

    let all = found(&[]);
    for needle in ["repainted", "needs paint", "inspected"] {
        assert!(
            all.contains(needle),
            "unnarrowed search misses {needle}: {all}"
        );
    }

    let notes = found(&["--kind", "note"]);
    assert!(notes.contains("repainted"), "{notes}");
    assert!(
        !notes.contains("needs paint") && !notes.contains("inspected"),
        "{notes}"
    );

    let docs = found(&["--source-prefix", "file:///docs/"]);
    assert!(docs.contains("needs paint"), "{docs}");
    assert!(
        !docs.contains("repainted") && !docs.contains("inspected"),
        "{docs}"
    );

    let recent = found(&["--since", "2024-01-01T00:00:00Z"]);
    assert!(
        recent.contains("repainted") && recent.contains("needs paint"),
        "{recent}"
    );
    assert!(!recent.contains("inspected"), "{recent}");

    let window = found(&[
        "--since",
        "2024-01-01T00:00:00Z",
        "--until",
        "2024-03-01T00:00:00Z",
    ]);
    assert!(window.contains("repainted"), "{window}");
    assert!(
        !window.contains("needs paint") && !window.contains("inspected"),
        "{window}"
    );
}

#[test]
fn delete_space_needs_the_name_repeated_and_leaves_the_neighbour() {
    let dir = tempfile::tempdir().unwrap();
    scone(dir.path())
        .args(["--space", "team", "add", "--note", "a note for team"])
        .assert()
        .success();
    scone(dir.path())
        .args(["--space", "other", "add", "--note", "a note for other"])
        .assert()
        .success();
    scone(dir.path())
        .args(["--space", "team", "delete-space", "--dry-run"])
        .assert()
        .success()
        .stdout(predicates::str::contains("would delete space team"))
        .stdout(predicates::str::contains("1 episodes"));
    scone(dir.path())
        .args(["--space", "team", "delete-space", "--confirm", "other"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("--confirm"));
    scone(dir.path())
        .args(["--space", "team", "delete-space", "--confirm", "team"])
        .assert()
        .success()
        .stdout(predicates::str::contains("deleted space team"));
    scone(dir.path())
        .args(["--space", "team", "add", "--note", "back?"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("deleted"));
    scone(dir.path())
        .args(["--space", "other", "profile"])
        .assert()
        .success()
        .stdout(predicates::str::contains("a note for other"));
}
