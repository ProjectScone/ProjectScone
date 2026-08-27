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
