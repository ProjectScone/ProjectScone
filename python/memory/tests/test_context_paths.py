"""Production context follows only complete, retained, budgeted evidence paths."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends.sqlite import SqliteDocumentStore
from scone_memory.core.models import RecallItem, RecallResult
from scone_memory.realtime.context import MemoryContext, _PREFIX


@pytest.fixture(params=["memory", "sqlite"])
async def memory(request, tmp_path):
    store = InMemoryDocumentStore() if request.param == "memory" else SqliteDocumentStore(tmp_path / "paths.db")
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder()).open()
    yield engine
    await engine.close()


async def chain(memory):
    facts, episodes = [], []
    for subject, predicate, obj in (("aster", "depends on", "beacon"), ("beacon", "operated by", "cedar"),
                                    ("cedar", "located in", "denver")):
        quote = f"{subject} {predicate} {obj}."
        episode = await memory.remember("alpha", quote, source=f"docs/{subject}", kind="file", created_at="2025-01-01",
                                        metadata={"project": "allowed"})
        facts.append(await memory.assert_fact("alpha", subject, predicate, obj, source_episode_id=episode.episode_id, quote=quote))
        episodes.append(episode)
    chunks = await memory.documents.chunks_of("alpha", episodes[0].episode_id)
    chunk = chunks[0]
    item = RecallItem(chunk_id=chunk.chunk_id, episode_id=chunk.episode_id, text=chunk.text, created_at=chunk.created_at,
                      source="docs/aster", score=1, metadata={"project": "allowed"})
    return facts, episodes, item


def packet(messages):
    return next(json.loads(message["content"].split("\n", 1)[1]) for message in messages
                if message.get("content", "").startswith("Scone retrieved source material:"))


async def test_actual_context_expands_seed_and_supplies_maximal_three_source_path(memory, monkeypatch):
    facts, _, item = await chain(memory)
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=[facts[0]], items=[item])))
    memory.documents.list_facts = AsyncMock(side_effect=AssertionError("no unconditional scan"))
    memory.documents.fact_links = AsyncMock(side_effect=AssertionError("no unbounded adjacency"))
    original = [{"role": "user", "content": "Where is the operator of aster's dependency located?"}]
    request, receipt = await MemoryContext(memory, "alpha", "session", where={"project": "allowed"}).prepare(original)
    data = packet(request)
    assert {claim["fact_id"] for claim in data["claims"]} == {fact.fact_id for fact in facts}
    assert len({claim["source_episode_id"] for claim in data["claims"]}) == 3
    assert data["sources"][0]["text"] == item.text
    assert data["paths"] == [{"fact_ids": [fact.fact_id for fact in facts], "steps": [
        {"from_fact": facts[0].fact_id, "to_fact": facts[1].fact_id, "kind": "subject_object", "direction": "forward"},
        {"from_fact": facts[1].fact_id, "to_fact": facts[2].fact_id, "kind": "subject_object", "direction": "forward"}]}]
    assert receipt["path_count"] == 1 and receipt["multihop_status"] == "prepared"
    assert receipt["context_bytes"] <= 8000
    assert request[-1] == original[-1]
    assert all(message["role"] == "user" for message in request)
    assert "paths" not in receipt
    assert set(receipt["claim_fingerprints"]) == {str(fact.fact_id) for fact in facts}


async def test_paths_work_when_all_chain_facts_were_initial_recall_seeds(memory, monkeypatch):
    facts, _, _ = await chain(memory)
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=facts)))
    request, receipt = await MemoryContext(memory, "alpha", "session").prepare([{"role": "user", "content": "aster dependency operator location"}])
    paths = packet(request)["paths"]
    assert paths == [{"fact_ids": [fact.fact_id for fact in facts], "steps": [
        {"from_fact": facts[0].fact_id, "to_fact": facts[1].fact_id, "kind": "subject_object", "direction": "forward"},
        {"from_fact": facts[1].fact_id, "to_fact": facts[2].fact_id, "kind": "subject_object", "direction": "forward"}]}]
    assert receipt["path_count"] == 1


async def test_false_flag_preserves_flat_context_and_skips_expansion(memory, monkeypatch):
    facts, _, _ = await chain(memory)
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=[facts[0]])))
    monkeypatch.setattr("scone_memory.realtime.context.expand_multihop", AsyncMock(side_effect=AssertionError("disabled")))
    request, receipt = await MemoryContext(memory, "alpha", "session", structured_paths=False).prepare([
        {"role": "user", "content": "aster dependency operator location"}])
    data = packet(request)
    assert "paths" not in data
    assert [claim["fact_id"] for claim in data["claims"]] == [facts[0].fact_id]
    assert request[0]["content"].startswith(_PREFIX)
    assert receipt["path_count"] == 0


@pytest.mark.parametrize("change", ["scope", "session", "deleted_after_expand"])
async def test_a_forbidden_or_deleted_bridge_never_enters_a_path(memory, monkeypatch, change):
    from scone_memory.retrieval.multihop import expand_multihop

    facts, episodes, _ = await chain(memory)
    if change == "scope":
        scope = {"source_prefix": "docs/aster"}
    elif change == "session":
        scope = {}
    else:
        scope = {}
    session = "docs/beacon" if change == "session" else "session"
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=[facts[0]])))
    if change == "deleted_after_expand":
        async def deleting(*args, **kwargs):
            result = await expand_multihop(*args, **kwargs)
            await memory.forget("alpha", episodes[1].episode_id)
            return result
        monkeypatch.setattr("scone_memory.realtime.context.expand_multihop", deleting)
    request, receipt = await MemoryContext(memory, "alpha", session, **scope).prepare([
        {"role": "user", "content": "aster dependency operator location"}])
    data = packet(request)
    assert data.get("paths", []) == []
    assert facts[1].fact_id not in {claim["fact_id"] for claim in data["claims"]}
    assert receipt["path_count"] == 0


@pytest.mark.parametrize("budget", [512, 1000, 1800, 3000, 8000])
async def test_path_claims_steps_and_verbatim_source_share_one_byte_budget(memory, monkeypatch, budget):
    facts, _, item = await chain(memory)
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=[facts[0]], items=[item])))
    request, receipt = await MemoryContext(memory, "alpha", "session", max_context_bytes=budget).prepare([
        {"role": "user", "content": "aster dependency operator location"}])
    assert receipt["context_bytes"] <= budget
    if receipt["status"] != "prepared":
        assert receipt["path_count"] == 0
        return
    data = packet(request)
    ids = {claim["fact_id"] for claim in data.get("claims", [])}
    for path in data.get("paths", []):
        assert set(path["fact_ids"]) <= ids
        assert len(path["steps"]) == len(path["fact_ids"]) - 1
    if data.get("paths"):
        assert data["sources"][0]["text"] == item.text
    assert receipt["path_count"] == len(data.get("paths", []))


async def test_reverse_stored_direction_is_retained_and_contradiction_is_not_a_path(memory, monkeypatch):
    facts, episodes, _ = await chain(memory)
    # Distinct objects prevent literal joins, leaving only the stored relation.
    other_source = await memory.remember("alpha", "other disagrees with aster.")
    other = await memory.assert_fact("alpha", "other", "notes", "value", source_episode_id=other_source.episode_id,
                                     quote="other disagrees with aster.")
    relation = await memory.link_facts("alpha", other.fact_id, facts[0].fact_id, "supports",
                                       source_episode_id=other_source.episode_id, quote="other disagrees with aster.")
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=[facts[0]])))
    request, _ = await MemoryContext(memory, "alpha", "session").prepare([{"role": "user", "content": "aster related evidence"}])
    assert any(step == {"from_fact": other.fact_id, "to_fact": facts[0].fact_id, "kind": "supports",
                        "direction": "reverse", "link_id": relation.link_id}
               for path in packet(request)["paths"] for step in path["steps"])
    await memory.link_facts("alpha", facts[0].fact_id, other.fact_id, "contradicts", source_episode_id=other_source.episode_id,
                            quote="other disagrees with aster.")
    request, _ = await MemoryContext(memory, "alpha", "session").prepare([{"role": "user", "content": "aster related evidence"}])
    data = packet(request)
    assert any(relation["kind"] == "contradicts" for relation in data["relations"])
    assert all(step["kind"] != "contradicts" for path in data["paths"] for step in path["steps"])


async def test_optional_timeout_keeps_flat_seed_evidence_and_reports_incomplete_coverage(memory, monkeypatch):
    facts, _, _ = await chain(memory)
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=[facts[0]])))
    async def stalled(*args, **kwargs):
        await asyncio.Event().wait()
    monkeypatch.setattr("scone_memory.realtime.context.expand_multihop", stalled)
    request, receipt = await MemoryContext(memory, "alpha", "session", recall_timeout=.02).prepare([
        {"role": "user", "content": "aster dependency operator location"}])
    assert receipt["status"] == "prepared" and receipt["multihop_status"] == "timeout"
    assert packet(request).get("paths", []) == []
    assert packet(request)["coverage"]["multihop"]["complete"] is False


async def test_cycles_never_repeat_a_fact_in_supplied_paths(memory, monkeypatch):
    facts, _, _ = await chain(memory)
    quote = "denver connects to aster."
    source = await memory.remember("alpha", quote)
    await memory.assert_fact("alpha", "denver", "connects to", "aster", source_episode_id=source.episode_id, quote=quote)
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=[facts[0]])))
    request, _ = await MemoryContext(memory, "alpha", "session").prepare([{"role": "user", "content": "aster chain"}])
    paths = packet(request)["paths"]
    assert paths and all(len(path["fact_ids"]) == len(set(path["fact_ids"])) for path in paths)


def test_structured_paths_requires_a_boolean(memory):
    with pytest.raises(ValueError):
        MemoryContext(memory, "alpha", "session", structured_paths=1)


@pytest.mark.parametrize("change", ["delete", "update"])
async def test_receipt_reconstruction_revalidates_expanded_ids_without_cached_paths(memory, monkeypatch, tmp_path, change):
    from scone_memory.api.conversations import create_conversation_app
    from scone_memory.realtime.text import TextConversation
    from test_conversations_api import client_for, create
    from test_inference_evidence_records import finish_query
    from test_text_conversation import ScriptedModel

    facts, sources, _ = await chain(memory)
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=[facts[0]])))
    app = create_conversation_app(memory, {"alpha-key": "alpha"}, tmp_path / "sessions.db",
        lambda space, sid: TextConversation(memory, space, sid, lambda: ScriptedModel("The retained evidence was considered.")))
    async with client_for(app) as client:
        route, result = await finish_query(client, await create(client))
        context = result["result"]["memory_context"]
        assert context["path_count"] == 1
        assert {node["data"]["fact_id"] for node in context["evidence_graph"]["nodes"] if node["kind"] == "claim"} == {fact.fact_id for fact in facts}
        if change == "delete":
            await memory.forget("alpha", sources[-1].episode_id)
        else:
            await memory.documents.update_fact(facts[-1].model_copy(update={"object": "private-new-value"}))
        refreshed = (await client.get(route)).json()["result"]["memory_context"]
        assert "paths" not in refreshed
        assert "private-new-value" not in json.dumps(refreshed)
        assert facts[-1].fact_id not in {node["data"]["fact_id"] for node in refreshed["evidence_graph"]["nodes"] if node["kind"] == "claim"}


async def test_parallel_link_kinds_do_not_collapse_to_one_path(memory, monkeypatch):
    source = await memory.remember("alpha", "one and two have two recorded relationships.")
    first = await memory.assert_fact("alpha", "one", "is", "a", source_episode_id=source.episode_id, quote="one and two")
    second = await memory.assert_fact("alpha", "two", "is", "b", source_episode_id=source.episode_id, quote="one and two")
    for kind in ("supports", "extends"):
        await memory.link_facts("alpha", first.fact_id, second.fact_id, kind, source_episode_id=source.episode_id, quote="two recorded relationships")
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=[first])))
    request, _ = await MemoryContext(memory, "alpha", "session").prepare([{"role": "user", "content": "one relationships"}])
    assert {step["kind"] for path in packet(request)["paths"] for step in path["steps"]} == {"supports", "extends"}
