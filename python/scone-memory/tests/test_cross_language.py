"""Episode-only interoperability, not full-export or retrieval parity.

The literal corpus is the oracle. Losing source/date, rewriting Unicode,
global dedup, or interpreting stored byte offsets as characters must fail.
Set SCONE_TEST_RUST_ROUNDTRIP to the freshly built debug example to exercise
both real runtimes; it is never compiled implicitly during Python-only tests.
"""

import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
CORPUS = json.loads((ROOT / "crates/scone-core/tests/fixtures/episodes-v1.json").read_text(encoding="utf-8"))
FIELDS = ("type", "kind", "content", "source", "created_at")


def assert_evidence(records):
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

    def run(records, check=True):
        return subprocess.run(
            [binary], input="\n".join(json.dumps(r, ensure_ascii=False) for r in records),
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
    from_rust = [json.loads(line) for line in rust_roundtrip(exported).stdout.splitlines()]
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
