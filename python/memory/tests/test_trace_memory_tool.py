"""Agents inspect sourced relationships without scanning or changing memory."""
from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends.sqlite import SqliteDocumentStore
from scone_memory.core.ports import NewFact, NewFactLink
from scone_memory.integrations.tools import ToolBox

STAMP = "2025-01-01T00:00:00Z"
MONGO_URL = os.environ.get("SCONE_TEST_MONGO_URL")


@pytest.fixture(params=["memory", "sqlite"] + (["mongo"] if MONGO_URL else []))
async def box(request, tmp_path):
    if request.param == "mongo":
        from scone_memory.backends.mongo import MongoDocumentStore

        store = MongoDocumentStore(MONGO_URL, "scone_test_trace_" + uuid4().hex)
    else:
        store = InMemoryDocumentStore() if request.param == "memory" else SqliteDocumentStore(tmp_path / "trace.db")
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        yield ToolBox(engine, "alpha")
    finally:
        try:
            if request.param == "mongo":
                await store.drop()
        finally:
            await engine.close()


async def claim(box, subject, predicate, obj, *, space="alpha", tags=(), **changes):
    quote = f"{subject} {predicate} {obj}."
    source = await box.engine.remember(space, quote, tags=tags, created_at=STAMP)
    fields = dict(space=space, subject=subject, predicate=predicate, object=obj,
                  valid_from=STAMP, source_episode_id=source.episode_id, quote=quote)
    return await box.engine.documents.insert_fact(NewFact(**(fields | changes)))


async def relation(box, left, right, kind="supports"):
    return await box.engine.documents.insert_fact_link(NewFactLink(
        space="alpha", from_fact=left.fact_id, to_fact=right.fact_id, kind=kind,
        created_at=STAMP, source_episode_id=left.source_episode_id, quote=left.quote))


async def test_trace_returns_ordered_claim_identity_and_quotes_without_model_or_write(box):
    first = await claim(box, "Juniper", "maintained by", "Mira")
    second = await claim(box, "Mira", "works on", "Platform")
    await claim(box, "Cedar", "maintained by", "Theo")
    await claim(box, "Theo", "works on", "Billing")
    revision = await box.engine.documents.revision("alpha")
    box.engine.recall = AsyncMock(side_effect=AssertionError("trace must not rank or embed"))
    box.engine.documents.list_facts = AsyncMock(side_effect=AssertionError("no ledger scan"))
    result = await box.run("trace_memory", {"seed_fact_id": first.fact_id})
    assert result["ok"] is True
    assert [(f["subject"], f["predicate"], f["object"]) for f in result["claims"]] == [
        ("Juniper", "maintained by", "Mira"), ("Mira", "works on", "Platform")]
    assert result["claims"][0]["quote"] == first.quote
    assert result["claims"][0]["source_episode_id"] == first.source_episode_id
    assert result["claims"][0]["origin"] == first.origin
    assert result["paths"][0]["fact_ids"] == [first.fact_id, second.fact_id]
    assert result["paths"][0]["steps"] == [{"from_fact": first.fact_id, "to_fact": second.fact_id,
                                           "kind": "subject_object", "direction": "forward"}]
    assert result["relations"] == []
    assert result["verified_accuracy"] is False
    assert "Billing" not in json.dumps(result)
    assert await box.engine.documents.revision("alpha") == revision


async def test_reverse_stored_link_preserves_its_direction_and_contradiction_evidence(box):
    first = await claim(box, "Juniper", "is", "green")
    second = await claim(box, "Beacon", "is", "ready")
    competing = await claim(box, "Juniper", "is", "red")
    stored = await relation(box, second, first)
    conflict = await relation(box, first, competing, "contradicts")
    result = await box.run("trace_memory", {"seed_fact_id": first.fact_id})
    assert result["ok"] is True
    assert {f["fact_id"] for f in result["claims"]} == {first.fact_id, second.fact_id, competing.fact_id}
    assert {r["link_id"] for r in result["relations"]} == {stored.link_id, conflict.link_id}
    assert result["paths"][0]["steps"] == [{"from_fact": second.fact_id, "to_fact": first.fact_id,
        "kind": "supports", "direction": "reverse", "link_id": stored.link_id}]
    assert all(competing.fact_id not in path["fact_ids"] for path in result["paths"])


@pytest.mark.parametrize("case", ["foreign", "forgotten", "excluded", "expired", "unquoted", "tags"])
async def test_ineligible_seed_never_returns_source_content(box, case):
    changes = {"excluded_reason": "private"} if case == "excluded" else {}
    if case == "expired":
        changes["valid_until"] = "2025-02-01T00:00:00Z"
    if case == "unquoted":
        changes["quote"] = "not in the source"
    seed = await claim(box, "PrivateMarker", "is", "secret", space="beta" if case == "foreign" else "alpha", **changes)
    if case == "forgotten":
        await box.engine.forget("alpha", seed.source_episode_id)
    args = {"seed_fact_id": seed.fact_id}
    if case == "tags":
        args["tags"] = ["public"]
    result = await box.run("trace_memory", args)
    assert result["ok"] is True
    assert result["status"] == "empty"
    assert result["claims"] == result["relations"] == result["paths"] == []
    assert result["coverage"]["complete"] is False
    assert "PrivateMarker" not in json.dumps(result)


async def test_tags_apply_to_expanded_sources_and_hop_limit_is_visible(box):
    first = await claim(box, "Juniper", "maintained by", "Mira", tags=["public"])
    second = await claim(box, "Mira", "works on", "Platform", tags=["public"])
    await claim(box, "Platform", "owns", "PrivateMarker")
    scoped = await box.run("trace_memory", {"seed_fact_id": first.fact_id, "tags": [" PUBLIC "]})
    assert scoped["ok"] is True
    assert {f["fact_id"] for f in scoped["claims"]} == {first.fact_id, second.fact_id}
    assert "PrivateMarker" not in json.dumps(scoped)
    bounded = await box.run("trace_memory", {"seed_fact_id": first.fact_id, "max_hops": 1})
    assert bounded["coverage"]["complete"] is False
    assert "max_hops" in bounded["coverage"]["reasons"]


@pytest.mark.parametrize("arguments", [{}, {"seed_fact_id": True}, {"seed_fact_id": 0},
    {"seed_fact_id": 2**63}, {"seed_fact_id": "1"}, {"seed_fact_id": 1, "max_hops": 7},
    {"seed_fact_id": 1, "max_hops": 0}, {"seed_fact_id": 1, "space": "beta"}])
async def test_invalid_calls_are_rejected_before_store_access(box, arguments):
    box.engine.documents.get_fact = AsyncMock(side_effect=AssertionError("invalid call reached store"))
    assert (await box.run("trace_memory", arguments))["ok"] is False


async def test_store_failure_is_sanitized_and_cancellation_propagates(box):
    box.engine.documents.get_fact = AsyncMock(side_effect=RuntimeError("private connection secret"))
    result = await box.run("trace_memory", {"seed_fact_id": 1})
    assert result["ok"] is False
    assert "private connection secret" not in json.dumps(result)
    box.engine.documents.get_fact = AsyncMock(side_effect=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        await box.run("trace_memory", {"seed_fact_id": 1})


async def test_revision_change_during_expansion_discards_entire_packet(box):
    first = await claim(box, "Juniper", "maintained by", "Mira")
    read = box.engine.documents.facts_by_subject

    async def write_during_read(space, subject, limit):
        await box.engine.remember("alpha", "a concurrent source update")
        return await read(space, subject, limit)

    box.engine.documents.facts_by_subject = write_during_read
    result = await box.run("trace_memory", {"seed_fact_id": first.fact_id})
    assert result["ok"] is False
    assert result["claims"] == result["relations"] == result["paths"] == []
    assert result["coverage"]["reasons"] == ["stale_evidence"]


async def test_timeout_cancels_store_read_and_returns_no_partial_evidence(box, monkeypatch):
    from scone_memory.integrations import relations

    monkeypatch.setattr(relations, "TRACE_TIMEOUT_SECONDS", 0.01)
    cancelled = asyncio.Event()

    async def stall(*args):
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    box.engine.documents.get_fact = stall
    result = await box.run("trace_memory", {"seed_fact_id": 1})
    assert result["ok"] is False and cancelled.is_set()
    assert result["coverage"]["reasons"] == ["timeout"]
    assert result["claims"] == result["relations"] == result["paths"] == []


async def test_shared_quote_does_not_mix_unrelated_claim_identities_or_truncate_text(box):
    quote = ("Juniper is maintained by Mira. Mira works on Platform. "
             "Cedar is maintained by Theo. Theo works on Billing. " + "Context. " * 60)
    source = await box.engine.remember("alpha", quote, created_at=STAMP)
    facts = [await box.engine.documents.insert_fact(NewFact(space="alpha", subject=subject,
        predicate=predicate, object=obj, valid_from=STAMP, source_episode_id=source.episode_id, quote=quote))
        for subject, predicate, obj in [("Juniper", "maintained by", "Mira"), ("Mira", "works on", "Platform"),
                                      ("Cedar", "maintained by", "Theo"), ("Theo", "works on", "Billing")]]
    result = await box.run("trace_memory", {"seed_fact_id": facts[0].fact_id})
    assert result["ok"] is True
    assert [f["fact_id"] for f in result["claims"]] == [f.fact_id for f in facts[:2]]
    assert all(f["quote"] == quote for f in result["claims"])
    assert result["paths"][0]["fact_ids"] == [f.fact_id for f in facts[:2]]


async def test_output_budget_never_cuts_a_quote_or_drops_only_one_side_of_conflict(box, monkeypatch):
    from scone_memory.integrations import relations

    first = await claim(box, "Juniper", "is", "green")
    other = await claim(box, "Juniper", "is", "red")
    await relation(box, first, other, "contradicts")
    monkeypatch.setattr(relations, "MAX_TRACE_BYTES", 800)
    result = await box.run("trace_memory", {"seed_fact_id": first.fact_id})
    assert result["ok"] is False
    assert result["coverage"]["reasons"] == ["output_bytes"]
    assert result["claims"] == result["relations"] == result["paths"] == []


async def test_search_only_toolbox_cannot_trace(box):
    restricted = ToolBox(box.engine, "alpha", tools=["search_memory"])
    assert (await restricted.run("trace_memory", {"seed_fact_id": 1}))["ok"] is False


async def test_stored_relations_also_reach_query_graph_inspection(box):
    from scone_memory.core.models import RecallResult
    from scone_memory.retrieval.evidence_graph import build_query_evidence_graph
    from scone_memory.retrieval.evidence_records import canonical_evidence

    first = await claim(box, "Juniper", "is", "green")
    second = await claim(box, "Beacon", "is", "ready")
    stored = await relation(box, second, first)
    graph = await build_query_evidence_graph(box.engine.documents, "alpha", "Juniper",
                                             RecallResult(facts=[first, second]))
    records = canonical_evidence(graph)
    assert records.relations == [{"link_id": stored.link_id, "from_fact": second.fact_id,
        "to_fact": first.fact_id, "kind": "supports", "source_episode_id": second.source_episode_id,
        "quote": second.quote}]
