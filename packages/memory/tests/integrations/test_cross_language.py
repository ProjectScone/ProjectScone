"""Episode-only interoperability, not full-export or retrieval parity.

The literal corpus is the oracle. Losing source/date, rewriting Unicode,
global dedup, or interpreting stored byte offsets as characters must fail.
Set SCONE_TEST_RUST_ROUNDTRIP to the freshly built debug example to exercise
both real runtimes; it is never compiled implicitly during Python-only tests.
"""

from ..paths import REPO_ROOT, TESTS_ROOT

import json
import hashlib
import os
from pathlib import Path
import subprocess

import pytest
from scone_memory import Record

ROOT = REPO_ROOT
CORPUS = json.loads(((TESTS_ROOT / "fixtures") / "episodes-v1.json").read_text(encoding="utf-8"))
FIELDS = ("type", "kind", "content", "source", "created_at")


def identity_outcomes(result):
    # A Milvus/gRPC fork can write diagnostics before the Rust exec. Only
    # the probe's tagged report lines are protocol; never parse all stderr.
    prefix = "SCONE_EPISODE_REPORT "
    return [json.loads(line[len(prefix):]) for line in result.stderr.splitlines()
            if line.startswith(prefix)]


def assert_evidence(records):
    # The header says what the archive is; the episodes follow it.
    records = [record for record in records if record.get("type") != "archive"]
    assert len(records) == len(CORPUS["episodes"])
    for expected in CORPUS["episodes"]:
        matches = [r for r in records if r["content"] == expected["content"]]
        assert len(matches) == 1
        for field in FIELDS:
            assert matches[0][field] == expected[field], field


async def test_shared_episode_evidence_and_scope_local_dedup(engine):
    for space in ("alpha", "beta"):
        first = await engine.import_records(space, CORPUS["episodes"])
        assert first.episodes == 3
        repeat = await engine.import_records(space, CORPUS["episodes"])
        assert (repeat.episodes, repeat.deduplicated) == (0, 3)
        assert_evidence([r async for r in engine.export(space)])


async def test_shared_stored_unicode_span_uses_utf8_bytes(engine):
    case = CORPUS["span"]
    added = await engine.remember("alpha", case["content"])
    recalled = await engine.recall("alpha", case["content"])
    chunks = await engine.documents.get_chunks("alpha", [i.chunk_id for i in recalled.items])
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.episode_id == added.episode_id
    assert (chunk.start, chunk.end) == (case["start"], case["end"])
    assert case["content"].encode("utf-8")[chunk.start:chunk.end].decode("utf-8") == chunk.text == case["content"]


@pytest.fixture
def rust_roundtrip():
    binary = os.environ.get("SCONE_TEST_RUST_ROUNDTRIP")
    if not binary:
        pytest.skip("build Rust episode_roundtrip and set SCONE_TEST_RUST_ROUNDTRIP for cross-runtime tests")
    assert Path(binary).is_file(), "configured Rust probe must exist; build the current source first"

    def run(records, check=True, python_source_space=None):
        args = [binary]
        if python_source_space is not None:
            args += ["--python-source-space", python_source_space]
        return subprocess.run(
            args, input="\n".join(json.dumps(r, ensure_ascii=False) for r in records),
            encoding="utf-8", capture_output=True, check=check, timeout=60,
        )

    return run


@pytest.mark.parametrize("start_in_rust", [False, True], ids=["python-rust-python", "rust-python-rust"])
async def test_real_episode_transfer_both_directions(engine, rust_roundtrip, start_in_rust):
    source = CORPUS["episodes"]
    if start_in_rust:
        source = [json.loads(line) for line in rust_roundtrip(source).stdout.splitlines()]
        assert_evidence(source)
    await engine.import_records("source", source)
    exported = [r async for r in engine.export("source")]
    assert_evidence(exported)
    transfer = rust_roundtrip([r for r in exported if r.get("type") != "archive"],
                              python_source_space="source")
    outcomes = identity_outcomes(transfer)
    assert [r["identity"] for r in outcomes] == ["accepted-verified"] * 3
    from_rust = [json.loads(line) for line in transfer.stdout.splitlines()]
    assert_evidence(from_rust)
    # Prepopulate to ensure the test cannot rely on identical store-local IDs.
    await engine.remember("destination", "unrelated destination record")
    imported = await engine.import_records("destination", from_rust)
    assert imported.episodes == 3
    repeat = await engine.import_records("destination", from_rust)
    assert (repeat.episodes, repeat.deduplicated) == (0, 3)
    destination = [r async for r in engine.export("destination")]
    assert len(destination) == 4
    assert_evidence([r for r in destination if r["content"] != "unrelated destination record"])
    assert_evidence([r async for r in engine.export("source")])


@pytest.mark.parametrize("unsupported", [
    {"type": "fact", "subject": "alice"},
    {**CORPUS["episodes"][0], "tags": ["private"]},
    {**CORPUS["episodes"][0], "metadata": {"user_id": "alice"}},
    {**CORPUS["episodes"][0], "created_at": "2024-01-02"},
    {**CORPUS["episodes"][0], "kind": "chat"},
    {**CORPUS["episodes"][0], "kind": "conversation"},
    {**CORPUS["episodes"][0], "extra_evidence": "must not disappear"},
])
def test_probe_refuses_records_outside_episode_profile(rust_roundtrip, unsupported):
    result = rust_roundtrip([CORPUS["episodes"][1], unsupported], check=False)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "episode-only profile" in result.stderr


@pytest.mark.parametrize("content", ["  Café 🥐\n東京\n", "\x1c\x1dhello\x1e\x1f", "\u00a0hello\u2003"])
def test_probe_verifies_python_identity_without_rewriting_evidence(rust_roundtrip, content):
    # Use the standard digest algorithm, not either engine's identity helper.
    record = {**CORPUS["episodes"][0], "content": content,
              "content_hash": hashlib.sha256(("source\0" + content.strip()).encode()).hexdigest()}
    result = rust_roundtrip([record], python_source_space="source")
    assert json.loads(result.stdout)["content"] == content
    assert [r["identity"] for r in identity_outcomes(result)] == ["accepted-verified"]


@pytest.mark.parametrize("identity", [None, "", "custom-turn", "0" * 64, 123])
def test_probe_refuses_unverifiable_python_identity(rust_roundtrip, identity):
    record = {**CORPUS["episodes"][0], "content_hash": identity}
    result = rust_roundtrip([CORPUS["episodes"][1], record], check=False, python_source_space="source")
    assert result.returncode != 0
    assert result.stdout == ""
    assert "refused-identity" in result.stderr
    assert "record 2" in result.stderr


@pytest.mark.parametrize("source_space", [None, "destination"])
def test_probe_requires_correct_python_source_space(rust_roundtrip, source_space):
    record = {**CORPUS["episodes"][0],
              "content_hash": hashlib.sha256(("source\0" + CORPUS["episodes"][0]["content"].strip()).encode()).hexdigest()}
    result = rust_roundtrip([record], check=False, python_source_space=source_space)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "refused-identity" in result.stderr


def test_probe_refuses_conflicting_records_in_one_dedup_group(rust_roundtrip):
    original = CORPUS["episodes"][2]
    for changed in ({**original, "content": original["content"] + " "},
                    {**original, "source": "notes://different-evidence"}):
        result = rust_roundtrip([original, changed], check=False)
        assert result.returncode != 0
        assert result.stdout == ""
        assert "refused-collision" in result.stderr


def test_probe_rejects_invalid_rust_identity(rust_roundtrip):
    result = rust_roundtrip([{**CORPUS["episodes"][0], "hash": "custom"}], check=False)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "refused-identity" in result.stderr


def test_probe_identical_repeated_evidence_deduplicates(rust_roundtrip):
    record = CORPUS["episodes"][0]
    result = rust_roundtrip([record, record])
    assert len(result.stdout.splitlines()) == 1
    assert json.loads(result.stdout)["content"] == record["content"]
    assert [r["identity"] for r in identity_outcomes(result)] == ["accepted-native"] * 2


@pytest.mark.parametrize("field, changed", [("source", "notes://other"),
                                            ("created_at", "2024-01-02T00:00:00.000Z")])
def test_probe_collision_reports_which_evidence_would_be_lost(rust_roundtrip, field, changed):
    record = CORPUS["episodes"][0]
    result = rust_roundtrip([record, {**record, field: changed}], check=False)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "refused-collision" in result.stderr
    assert field in result.stderr


def test_probe_reads_source_space_from_export(rust_roundtrip):
    record = {**CORPUS["episodes"][2], "space": "source",
              "content_hash": hashlib.sha256(b"source\0deploy checklist: verify evidence").hexdigest()}
    result = rust_roundtrip([record])
    assert json.loads(result.stdout)["content"] == record["content"]
    assert [r["identity"] for r in identity_outcomes(result)] == ["accepted-verified"]


@pytest.mark.parametrize("declared_space", ["different", "", None, 17])
def test_probe_refuses_conflicting_or_invalid_declared_space(rust_roundtrip, declared_space):
    record = {**CORPUS["episodes"][2], "space": declared_space,
              "content_hash": hashlib.sha256(b"source\0deploy checklist: verify evidence").hexdigest()}
    result = rust_roundtrip([record], python_source_space="source", check=False)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "refused-identity" in result.stderr


async def test_real_keyed_turns_are_refused_not_merged(engine, rust_roundtrip):
    await engine.remember_many("source", [
        Record("ok", dedup_key="conversation#1", created_at="2024-01-02T03:04:05.006Z"),
        Record("ok", dedup_key="conversation#2", created_at="2024-01-02T03:04:05.006Z"),
    ])
    exported = [r async for r in engine.export("source")]
    assert len(exported) == 2
    assert len({r["content_hash"] for r in exported}) == 2
    result = rust_roundtrip(exported, check=False)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "refused-identity" in result.stderr
    assert len([r async for r in engine.export("source")]) == 2


def test_probe_refuses_merging_spaces(rust_roundtrip):
    records = []
    for space in ("alpha", "beta"):
        records.append({**CORPUS["episodes"][2], "space": space,
                        "content_hash": hashlib.sha256((space + "\0deploy checklist: verify evidence").encode()).hexdigest()})
    result = rust_roundtrip(records, check=False)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "refused-space" in result.stderr
