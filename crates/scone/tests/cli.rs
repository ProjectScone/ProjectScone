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
