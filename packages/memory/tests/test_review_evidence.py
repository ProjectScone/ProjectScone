"""Prepared answer-review evidence stays tied to delivered retained records."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends.sqlite import SqliteDocumentStore
from scone_memory.realtime.context import MemoryContext
from scone_memory.retrieval.recall_scope import RecallScope


@pytest.fixture(params=["memory", "sqlite"])
async def memory(request, tmp_path):
    store = InMemoryDocumentStore() if request.param == "memory" else SqliteDocumentStore(tmp_path / "review.db")
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder(), clock=lambda: "2026-09-08T00:00:00Z").open()
    yield engine
    await engine.close()


async def prepared(memory, *, paths=True, linked=False):
    scope = RecallScope.validated(where={"team": "blue"}, kind="file", source_prefix="manuals/")
    left = await memory.remember("alpha", "aster depends on beacon.", kind="file", source="manuals/aster", metadata={"team": "blue"})
    right = await memory.remember("alpha", "beacon belongs to cedar.", kind="file", source="manuals/beacon", metadata={"team": "blue"})
    first = await memory.assert_fact("alpha", "aster", "depends on", "beacon", source_episode_id=left.episode_id, quote="aster depends on beacon.")
    second = await memory.assert_fact("alpha", "beacon", "belongs to", "cedar", source_episode_id=right.episode_id, quote="beacon belongs to cedar.")
    if linked:
        link_source = await memory.remember("alpha", "The aster claim supports the beacon claim.", kind="file",
                                            source="manuals/link", metadata={"team": "blue"})
        await memory.link_facts("alpha", first.fact_id, second.fact_id, "supports",
            source_episode_id=link_source.episode_id, quote="The aster claim supports the beacon claim.")
    request, receipt = await MemoryContext(memory, "alpha", "current", where={"team": "blue"}, kind="file",
        source_prefix="manuals/", structured_paths=paths, path_quotes=paths).prepare([{"role": "user", "content": "aster beacon"}])
    assert receipt["status"] == "prepared"
    return scope, request, receipt, (left, right), (first, second)


async def test_capture_delivered_ids_and_validate_without_new_retrieval(memory, monkeypatch):
    from scone_memory.realtime.review_evidence import prepare_review_evidence
    scope, request, receipt, _, facts = await prepared(memory)
    async def forbidden(*args, **kwargs):
        raise AssertionError("review must not retrieve or expand")
    monkeypatch.setattr(memory, "recall", forbidden)
    monkeypatch.setattr(memory.documents, "fact_links_between", forbidden)
    snapshot = await prepare_review_evidence(memory, "alpha", scope, "current", request, receipt)
    assert snapshot.evidence == request[0]["content"]
    assert {f"fact:{fact.fact_id}" for fact in facts} <= set(snapshot.evidence_ids)
    assert any(value.startswith("chunk:") for value in snapshot.evidence_ids)
    assert len(snapshot.evidence_ids) == len(set(snapshot.evidence_ids))
    assert await snapshot.validate() is True
    assert receipt["evidence_revision"] == await memory.documents.revision("alpha")


async def test_ingested_case_normalized_path_survives_packing_and_review(memory):
    from scone_memory.realtime.review_evidence import prepare_review_evidence
    facts = []
    for subject, predicate, obj in [("Morrow", "forwards to", "Nacre"), ("Nacre", "uses", "optical archive")]:
        quote = f"{subject} {predicate} {obj}."
        source = await memory.remember("alpha", quote)
        facts.append(await memory.assert_fact("alpha", subject, predicate, obj,
            source_episode_id=source.episode_id, quote=quote))
    request, receipt = await MemoryContext(memory, "alpha", "case-normalized", path_quotes=True).prepare(
        [{"role": "user", "content": "Morrow Nacre optical archive"}])
    assert receipt["path_count"] == 1
    snapshot = await prepare_review_evidence(memory, "alpha", RecallScope.validated(),
        "case-normalized", request, receipt)
    assert {f"fact:{fact.fact_id}" for fact in facts} <= set(snapshot.evidence_ids)
    assert await snapshot.validate() is True


async def test_initial_revision_drift_rejects_even_unrelated_write(memory):
    from scone_memory.realtime.review_evidence import prepare_review_evidence
    scope, request, receipt, _, _ = await prepared(memory)
    await memory.remember("alpha", "unrelated later native write")
    with pytest.raises(ValueError, match="^prepared evidence unavailable$"):
        await prepare_review_evidence(memory, "alpha", scope, "current", request, receipt)


async def test_validation_rejects_deleted_source_and_does_not_mutate_input(memory):
    from scone_memory.realtime.review_evidence import prepare_review_evidence
    scope, request, receipt, sources, _ = await prepared(memory)
    original_request, original_receipt = copy.deepcopy(request), copy.deepcopy(receipt)
    snapshot = await prepare_review_evidence(memory, "alpha", scope, "current", request, receipt)
    await memory.forget("alpha", sources[0].episode_id)
    assert await snapshot.validate() is False
    assert request == original_request and receipt == original_receipt


@pytest.mark.parametrize("change", ["missing_revision", "missing_graph", "wrong_hash", "missing_packet", "duplicate_packet", "wrong_session", "forged_quote", "forged_fingerprint", "unknown_fact", "broken_path"])
async def test_bad_preparation_proof_is_rejected(memory, change):
    from scone_memory.realtime.review_evidence import prepare_review_evidence
    scope, request, receipt, _, _ = await prepared(memory)
    if change == "missing_revision":
        receipt.pop("evidence_revision", None)
    elif change == "missing_graph":
        receipt.pop("evidence_graph_status", None)
    elif change == "wrong_hash":
        receipt["context_sha256"] = "0" * 64
    elif change == "missing_packet":
        request.pop(0)
    elif change == "duplicate_packet":
        request.insert(0, copy.deepcopy(request[0]))
    elif change == "wrong_session":
        receipt["session_id"] = "other-session"
    else:
        prefix, _, encoded = request[0]["content"].partition("\n")
        packet = json.loads(encoded)
        if change in ("forged_quote", "forged_fingerprint"):
            packet["claims"][0]["quote"] = "invented answer evidence"
            if change == "forged_fingerprint":
                record = packet["claims"][0]
                receipt["claim_fingerprints"][str(record["fact_id"])] = hashlib.sha256(json.dumps(record,
                    ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        elif change == "unknown_fact":
            packet["claims"][0]["fact_id"] = 999999
        else:
            assert packet["paths"]
            packet["paths"][0]["steps"][0]["direction"] = "reverse"
        request[0]["content"] = prefix + "\n" + json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
        receipt["context_sha256"] = hashlib.sha256(request[0]["content"].encode()).hexdigest()
        receipt["context_bytes"] = len(request[0]["content"].encode())
    with pytest.raises(ValueError, match="^prepared evidence unavailable$"):
        await prepare_review_evidence(memory, "alpha", scope, "current", request, receipt)


@pytest.mark.parametrize("change", ["quote", "scope", "session", "source", "kind", "created_at"])
async def test_direct_store_source_mutations_are_rejected_without_revision_change(memory, monkeypatch, change):
    from scone_memory.realtime.review_evidence import prepare_review_evidence
    scope, request, receipt, sources, _ = await prepared(memory)
    snapshot = await prepare_review_evidence(memory, "alpha", scope, "current", request, receipt)
    original = memory.documents.get_episode
    async def altered(space, episode_id):
        episode = await original(space, episode_id)
        if episode is None or episode_id != sources[0].episode_id:
            return episode
        changes = {"quote": {"content": "different source"}, "scope": {"metadata": {"team": "red"}},
                   "session": {"metadata": {"team": "blue", "session_id": "current"}},
                   "source": {"source": "manuals/replaced"}, "kind": {"kind": "note"},
                   "created_at": {"created_at": "2027-01-01T00:00:00Z"}}
        return episode.model_copy(update=changes[change])
    monkeypatch.setattr(memory.documents, "get_episode", altered)
    assert await memory.documents.revision("alpha") == receipt["evidence_revision"]
    assert await snapshot.validate() is False
    with pytest.raises(ValueError, match="^prepared evidence unavailable$"):
        await prepare_review_evidence(memory, "alpha", scope, "current", request, receipt)


async def test_store_error_is_sanitized_and_cancellation_propagates(memory, monkeypatch):
    from scone_memory.realtime.review_evidence import prepare_review_evidence
    scope, request, receipt, _, _ = await prepared(memory)
    snapshot = await prepare_review_evidence(memory, "alpha", scope, "current", request, receipt)
    async def failed(space):
        raise RuntimeError("PRIVATE endpoint, quote and credentials")
    monkeypatch.setattr(memory.documents, "revision", failed)
    with pytest.raises(ValueError, match="^prepared evidence unavailable$"):
        await snapshot.validate()
    async def cancelled(space):
        raise asyncio.CancelledError()
    monkeypatch.setattr(memory.documents, "revision", cancelled)
    with pytest.raises(asyncio.CancelledError):
        await snapshot.validate()


@pytest.mark.parametrize("change", ["quote", "direction", "deleted", "source"])
async def test_link_ids_and_quotes_are_verified_by_point_reads(memory, monkeypatch, change):
    from scone_memory.realtime.review_evidence import prepare_review_evidence
    scope, request, receipt, _, _ = await prepared(memory, linked=True)
    snapshot = await prepare_review_evidence(memory, "alpha", scope, "current", request, receipt)
    assert "link:1" in snapshot.evidence_ids
    assert "The aster claim supports the beacon claim." in snapshot.evidence
    getter = memory.documents.get_fact_link
    async def altered(space, link_id):
        link = await getter(space, link_id)
        if link is None or change == "deleted":
            return None
        updates = {"quote": {"quote": "made up relationship"}, "direction": {"from_fact": link.to_fact, "to_fact": link.from_fact},
                   "source": {"source_episode_id": 999999}}
        return link.model_copy(update=updates[change])
    monkeypatch.setattr(memory.documents, "get_fact_link", altered)
    assert await snapshot.validate() is False
    with pytest.raises(ValueError, match="^prepared evidence unavailable$"):
        await prepare_review_evidence(memory, "alpha", scope, "current", request, receipt)


@pytest.mark.parametrize("change", ["fact_quote", "fact_excluded", "fact_time", "chunk_offset", "chunk_space"])
async def test_direct_record_mutations_invalidate_snapshot_without_revision_change(memory, monkeypatch, change):
    from scone_memory.realtime.review_evidence import prepare_review_evidence
    scope, request, receipt, _, _ = await prepared(memory)
    snapshot = await prepare_review_evidence(memory, "alpha", scope, "current", request, receipt)
    if change.startswith("fact"):
        getter = memory.documents.get_fact
        async def fact_getter(space, fact_id):
            fact = await getter(space, fact_id)
            if fact is None:
                return None
            updates = {"fact_quote": {"quote": "changed claim"}, "fact_excluded": {"excluded_reason": "user removed"},
                       "fact_time": {"valid_until": "2026-01-01T00:00:00Z"}}
            return fact.model_copy(update=updates[change])
        monkeypatch.setattr(memory.documents, "get_fact", fact_getter)
    else:
        getter = memory.documents.get_chunks
        async def chunk_getter(space, ids):
            chunks = await getter(space, ids)
            updates = {"start": 1} if change == "chunk_offset" else {"space": "other"}
            return [chunk.model_copy(update=updates) for chunk in chunks]
        monkeypatch.setattr(memory.documents, "get_chunks", chunk_getter)
    assert await memory.documents.revision("alpha") == receipt["evidence_revision"]
    assert await snapshot.validate() is False


async def test_initial_verification_revision_guard_rejects_concurrent_write(memory, monkeypatch):
    from scone_memory.realtime.review_evidence import prepare_review_evidence
    scope, request, receipt, _, _ = await prepared(memory)
    getter = memory.documents.get_chunks
    changed = False
    async def get_chunks(space, ids):
        nonlocal changed
        if not changed:
            changed = True
            await memory.remember("alpha", "concurrent unrelated native write")
        return await getter(space, ids)
    monkeypatch.setattr(memory.documents, "get_chunks", get_chunks)
    with pytest.raises(ValueError, match="^prepared evidence unavailable$"):
        await prepare_review_evidence(memory, "alpha", scope, "current", request, receipt)


@pytest.mark.parametrize("change", ["large_unicode", "too_many_sources", "duplicate_json_key", "unknown_packet_key"])
async def test_packet_bounds_and_schema_fail_before_source_reads(memory, monkeypatch, change):
    from scone_memory.realtime.review_evidence import prepare_review_evidence
    scope, request, receipt, _, _ = await prepared(memory)
    prefix, _, encoded = request[0]["content"].partition("\n")
    packet = json.loads(encoded)
    if change == "large_unicode":
        packet["coverage"]["note"] = "é" * 64000
    elif change == "too_many_sources":
        packet["sources"] *= 25
    elif change == "unknown_packet_key":
        packet["invented_evidence"] = "private fake answer"
    serialized = json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
    if change == "duplicate_json_key":
        serialized = '{"schema_version":1,' + serialized[1:]
    request[0]["content"] = prefix + "\n" + serialized
    receipt["context_sha256"] = hashlib.sha256(request[0]["content"].encode()).hexdigest()
    receipt["context_bytes"] = len(request[0]["content"].encode())
    reads = []
    async def forbidden(*args):
        reads.append(args)
        raise AssertionError("bounds must precede reads")
    monkeypatch.setattr(memory.documents, "get_episode", forbidden)
    monkeypatch.setattr(memory.documents, "get_chunks", forbidden)
    with pytest.raises(ValueError, match="^prepared evidence unavailable$"):
        await prepare_review_evidence(memory, "alpha", scope, "current", request, receipt)
    assert reads == []


async def test_snapshot_is_detached_from_mutable_request_and_receipt(memory):
    from scone_memory.realtime.review_evidence import prepare_review_evidence
    scope, request, receipt, _, _ = await prepared(memory)
    snapshot = await prepare_review_evidence(memory, "alpha", scope, "current", request, receipt)
    evidence, ids = snapshot.evidence, snapshot.evidence_ids
    request[0]["content"] = "invented later evidence"
    for node in receipt["evidence_graph"]["nodes"]:
        if node["kind"] == "episode":
            node["data"]["preview"] = "later altered receipt"
    receipt["evidence_graph"]["nodes"].clear()
    assert snapshot.evidence == evidence and snapshot.evidence_ids == ids
    assert await snapshot.validate() is True


async def test_source_timeouts_preserve_type_and_redact_provider_message(memory, monkeypatch):
    from scone_memory.realtime.review_evidence import prepare_review_evidence
    scope, request, receipt, _, _ = await prepared(memory)
    snapshot = await prepare_review_evidence(memory, "alpha", scope, "current", request, receipt)
    async def unavailable(*args):
        raise TimeoutError("private provider credentials")
    monkeypatch.setattr(memory.documents, "get_chunks", unavailable)
    with pytest.raises(TimeoutError, match="^prepared evidence unavailable$"):
        await snapshot.validate()
    with pytest.raises(TimeoutError, match="^prepared evidence unavailable$"):
        await prepare_review_evidence(memory, "alpha", scope, "current", request, receipt)
