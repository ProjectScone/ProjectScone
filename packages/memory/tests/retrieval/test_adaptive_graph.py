"""Pre-assessment graph expansion, using real retained stores and no model."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from scone_memory import MemoryEngine
from scone_memory.core.models import Episode, Fact, RecallResult
from scone_memory.core.ports import NewFactLink
from scone_memory.retrieval.adaptive import AdaptiveLimits, AdaptiveRetriever, EvidenceCandidate, EvidenceDecision
from scone_memory.retrieval.adaptive_graph import merge_groups
from scone_memory.retrieval.multihop import MultiHopLimits
from scone_memory.retrieval.recall_scope import RecallScope
from ..retrieval.test_adaptive_retrieval import Assessor, STAMP, fact, memory, sufficient


async def chain(engine: MemoryEngine, length: int = 3) -> list[Fact]:
    names = ["Aster", "Beacon", "Cedar", "Denver", "Elysium"]
    return [await fact(engine, names[index], "depends on", names[index + 1]) for index in range(length)]


@pytest.mark.parametrize("delete_bridge", [False, True])
@pytest.mark.parametrize("mutate_candidate", [False, True])
async def test_empty_selection_retains_complete_verified_groups_or_drops_stale_group(memory, monkeypatch, delete_bridge, mutate_candidate):
    records = await chain(memory)
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=records[:1])))
    async def assess(question, candidates):
        if mutate_candidate:
            object.__setattr__(candidates[0], "text", "adapter changed its input")
        if delete_bridge:
            await memory.forget("alpha", records[1].source_episode_id)
        return EvidenceDecision(status="insufficient", selected_ids=())
    result = await AdaptiveRetriever(memory, Assessor(assess), graph_limits=MultiHopLimits()).retrieve(
        "alpha", "Aster destination", scope=RecallScope.validated())
    assert result.recall.facts == ([] if delete_bridge else records)
    assert result.selected_groups == (() if delete_bridge else (tuple(f"fact:{r.fact_id}" for r in records),))
    assert result.evidence_basis == ("none" if delete_bridge else "unselected_candidates")
    assert result.rounds[-1].selected_count == 0
    if not delete_bridge:
        assert "atomic_group_omitted" not in result.reasons
        assert "stale_evidence" not in result.reasons


@pytest.mark.parametrize("fails", [False, True])
async def test_real_recall_expands_missing_chain_before_first_assessment(memory: MemoryEngine, fails: bool) -> None:
    records = await chain(memory)
    baseline = await memory.recall("alpha", "Aster")
    assert baseline.facts == records[:1]

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        assert {candidate.id for candidate in candidates if candidate.id.startswith("fact:")} == {
            f"fact:{record.fact_id}" for record in records}
        if fails:
            raise ValueError("unreliable assessment")
        return await sufficient(question, candidates)

    result = await AdaptiveRetriever(memory, Assessor(assess), graph_limits=MultiHopLimits()).retrieve(
        "alpha", "Aster", scope=RecallScope.validated())
    assert result.recall.facts == records
    assert result.selected_groups == (tuple(f"fact:{record.fact_id}" for record in records),)
    assert result.graph_expansions[0].added_count == 2
    assert result.graph_expansions[0].store_calls <= 256
    assert result.evidence_basis == ("verified_candidates" if fails else "assessed_selection")
    assert result.status == ("uncertain" if fails else "sufficient")
    assert "Aster" not in str(result.graph_expansions)


async def test_graph_none_preserves_search_only_behavior(memory: MemoryEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    records = await chain(memory)
    monkeypatch.setattr(memory.documents, "fact_links_from", AsyncMock(side_effect=AssertionError("no graph")))
    monkeypatch.setattr(memory.documents, "facts_by_subject", AsyncMock(side_effect=AssertionError("no graph")))
    result = await AdaptiveRetriever(memory, Assessor(sufficient)).retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert result.recall.facts == records[:1]
    assert result.graph_expansions == () and result.selected_groups == ()


async def test_components_displace_unrelated_chunks_and_partial_selection_is_omitted(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch) -> None:
    records = await chain(memory)
    for index in range(3):
        await memory.remember("alpha", f"Noise #{index} about Aster.", created_at=STAMP)
    baseline = await memory.recall("alpha", "Aster", limit=3)
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(items=baseline.items, facts=records[:1])))
    monkeypatch.setattr(memory.documents, "list_facts", AsyncMock(side_effect=AssertionError("no full scan")))
    assessor = Assessor(sufficient)
    result = await AdaptiveRetriever(memory, assessor, limits=AdaptiveLimits(candidate_limit=3),
                                    graph_limits=MultiHopLimits()).retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert result.recall.facts == records and not result.recall.items
    assert len(assessor.calls[0]) == 3 and result.graph_expansions[0].omitted_count >= 1

    async def partial(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        return EvidenceDecision(status="sufficient", selected_ids=(f"fact:{records[0].fact_id}",))

    omitted = await AdaptiveRetriever(memory, Assessor(partial), graph_limits=MultiHopLimits()).retrieve(
        "alpha", "Aster", scope=RecallScope.validated())
    assert not omitted.recall.facts and omitted.status == "uncertain"
    assert omitted.selected_groups == () and "atomic_group_omitted" in omitted.reasons


@pytest.mark.parametrize("options", [{"candidate_limit": 2}, {"max_evidence_bytes": 250}])
async def test_over_budget_exact_components_are_omitted_whole(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch, options: dict[str, int]) -> None:
    records = await chain(memory)
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=records[:1])))
    assessor = Assessor(sufficient)
    result = await AdaptiveRetriever(memory, assessor, graph_limits=MultiHopLimits(),
        limits=AdaptiveLimits.model_validate(options)).retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert result.recall.facts == [] and assessor.calls == []
    assert result.graph_expansions[0].truncated
    assert "atomic_group_omitted" in result.graph_expansions[0].reasons
    assert result.graph_expansions[0].omitted_count == 3


@pytest.mark.parametrize("limits", [MultiHopLimits(max_store_calls=1), MultiHopLimits(max_nodes=1),
    MultiHopLimits(max_bytes=512), MultiHopLimits(max_candidates=1)])
async def test_traversal_budgets_are_visible_and_original_seed_remains(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch, limits: MultiHopLimits) -> None:
    records = await chain(memory)
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=records[:1])))
    result = await AdaptiveRetriever(memory, Assessor(sufficient), graph_limits=limits).retrieve(
        "alpha", "Aster", scope=RecallScope.validated())
    assert result.recall.facts == records[:1]
    assert result.graph_expansions[0].truncated and not result.graph_expansions[0].complete
    assert result.graph_expansions[0].store_calls <= limits.max_store_calls


@pytest.mark.parametrize("hidden", ["space", "scope", "session", "source_session", "prefix", "future_fact", "future_source"])
async def test_graph_neighbors_obey_fixed_scope_exclusion_and_time(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch, hidden: str) -> None:
    first = await fact(memory, "Aster", "depends on", "Beacon", metadata={"team": "blue"}, source="public/a")
    hidden_fact = await fact(memory, "Beacon", "depends on", "Cedar", space="beta" if hidden == "space" else "alpha",
        metadata={"team": "red" if hidden == "scope" else "blue", "session_id": "public/current" if hidden == "session" else "other"},
        source="private/a" if hidden == "prefix" else "public/current" if hidden == "source_session" else "public/b",
        at="2099-01-01T00:00:00Z" if hidden in {"future_fact", "future_source"} else STAMP)
    if hidden == "future_source":
        hidden_fact = hidden_fact.model_copy(update={"valid_from": STAMP})
        await memory.documents.update_fact(hidden_fact)
    monkeypatch.setattr(memory, "clock", lambda: "2026-01-01T00:00:00Z")
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=[first])))
    result = await AdaptiveRetriever(memory, Assessor(sufficient), graph_limits=MultiHopLimits()).retrieve(
        "alpha", "Aster", scope=RecallScope.validated(where={"team": "blue"}, source_prefix="public/"),
        exclude_session_id="public/current")
    assert result.recall.facts == [first]
    assert f"fact:{hidden_fact.fact_id}" not in {member for group in result.selected_groups for member in group}


async def test_future_source_seed_uses_same_graph_boundary(memory: MemoryEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    seed = await fact(memory, "Aster", "depends on", "Beacon", at="2099-01-01T00:00:00Z")
    seed = seed.model_copy(update={"valid_from": STAMP})
    await memory.documents.update_fact(seed)
    monkeypatch.setattr(memory, "clock", lambda: "2026-01-01T00:00:00Z")
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=[seed])))
    assessor = Assessor(sufficient)
    result = await AdaptiveRetriever(memory, assessor, graph_limits=MultiHopLimits()).retrieve(
        "alpha", "Aster", scope=RecallScope.validated())
    assert not result.recall.facts and not assessor.calls
    assert result.graph_expansions[0].reasons == ("no_seeds",)
    ordinary = await AdaptiveRetriever(memory, Assessor(sufficient)).retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert ordinary.recall.facts == [seed]


async def test_stored_links_offer_facts_without_inventing_exact_groups(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch) -> None:
    first = await fact(memory, "Aster", "color", "blue")
    second = await fact(memory, "Cedar", "city", "Denver")
    await memory.documents.insert_fact_link(NewFactLink(space="alpha", from_fact=first.fact_id,
        to_fact=second.fact_id, kind="supports", source_episode_id=first.source_episode_id, quote=first.quote, created_at=STAMP))
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=[first])))
    result = await AdaptiveRetriever(memory, Assessor(sufficient), graph_limits=MultiHopLimits()).retrieve(
        "alpha", "Aster", scope=RecallScope.validated())
    assert result.recall.facts == [first, second] and result.selected_groups == ()


@pytest.mark.parametrize("failure", ["revision", "store", "timeout"])
async def test_graph_failures_do_not_trigger_assessor_fallback(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    records = await chain(memory)
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=records[:1])))
    original = memory.documents.facts_by_subject
    calls = 0

    async def changed(space: str, subject: str, limit: int) -> list[Fact]:
        nonlocal calls
        calls += 1
        if failure == "store":
            raise RuntimeError("private storage detail")
        if failure == "timeout":
            await asyncio.Event().wait()
        if calls == 1:
            await memory.remember("alpha", "Unrelated engine write", created_at=STAMP)
        found = await original(space, subject, limit)
        assert isinstance(found, list)
        return found

    monkeypatch.setattr(memory.documents, "facts_by_subject", changed)
    assessor = Assessor(sufficient)
    result = await AdaptiveRetriever(memory, assessor, graph_limits=MultiHopLimits(),
        limits=AdaptiveLimits(timeout_s=1.0)).retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert result.status == "uncertain" and not result.recall.facts and not assessor.calls
    assert result.fallback_status == "not_used"
    assert result.graph_expansions and not result.graph_expansions[0].complete
    assert "private storage detail" not in result.model_dump_json()


async def test_graph_cancellation_propagates(memory: MemoryEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    records = await chain(memory)
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=records[:1])))
    entered = asyncio.Event()

    async def blocked(space: str, subject: str, limit: int) -> list[Fact]:
        entered.set()
        await asyncio.Event().wait()
        return []

    monkeypatch.setattr(memory.documents, "facts_by_subject", blocked)
    assessor = Assessor(sufficient)
    task = asyncio.create_task(AdaptiveRetriever(memory, assessor, graph_limits=MultiHopLimits()).retrieve(
        "alpha", "Aster", scope=RecallScope.validated()))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not assessor.calls


def test_overlapping_components_merge_without_losing_prior_members() -> None:
    assert merge_groups((("fact:1", "chunk:2"), ("fact:3", "fact:4"), ("fact:1", "fact:3"))) == (
        ("fact:3", "fact:4", "fact:1", "chunk:2"),)


async def test_graph_limits_are_copied_strictly(memory: MemoryEngine) -> None:
    limits = MultiHopLimits(max_nodes=10)
    retriever = AdaptiveRetriever(memory, Assessor(sufficient), graph_limits=limits)
    assert retriever.graph_limits == limits and retriever.graph_limits is not limits
    with pytest.raises(ValidationError):
        AdaptiveRetriever(memory, Assessor(sufficient), graph_limits=limits.model_copy(update={"max_nodes": "10"}))


async def test_oversized_discovered_quote_omits_entire_component(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch) -> None:
    first = await fact(memory, "Aster", "depends on", "Beacon")
    oversized = await fact(memory, "Beacon", "notes", "é" * 1000)
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=[first])))
    assessor = Assessor(sufficient)
    result = await AdaptiveRetriever(memory, assessor, graph_limits=MultiHopLimits(),
        limits=AdaptiveLimits(max_evidence_bytes=500)).retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert result.recall.facts == [] and assessor.calls == []
    assert result.graph_expansions[0].truncated and result.graph_expansions[0].omitted_count == 2
    assert {"max_evidence_bytes", "atomic_group_omitted"}.issubset(result.graph_expansions[0].reasons)
    assert oversized.quote is not None
    assert oversized.quote not in result.model_dump_json()


async def test_changed_existing_seed_cannot_be_replaced_by_graph_snapshot(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch) -> None:
    records = await chain(memory)
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=records[:1])))
    original = memory.documents.facts_by_subject
    original_fact = memory.documents.get_fact
    mutated = False

    async def current_fact(space: str, fact_id: int) -> Fact | None:
        found = await original_fact(space, fact_id)
        if not isinstance(found, Fact):
            return None
        return found.model_copy(update={"confidence": 0.5}) if mutated and fact_id == records[0].fact_id else found

    async def neighbors(space: str, subject: str, limit: int) -> list[Fact]:
        nonlocal mutated
        mutated = True
        result = await original(space, subject, limit)
        assert isinstance(result, list)
        return result

    monkeypatch.setattr(memory.documents, "get_fact", current_fact)
    monkeypatch.setattr(memory.documents, "facts_by_subject", neighbors)
    assessor = Assessor(sufficient)
    result = await AdaptiveRetriever(memory, assessor, graph_limits=MultiHopLimits()).retrieve(
        "alpha", "Aster", scope=RecallScope.validated())
    assert result.recall.facts == [] and not assessor.calls
    assert "stale_evidence" in result.reasons


async def test_repeated_expansion_has_per_round_bounds_and_accepts_changed_recall_ranks(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch) -> None:
    records = await chain(memory)
    baseline = await memory.recall("alpha", "Aster")
    assert baseline.items
    calls = 0

    async def recall(space: str, query: str, **kwargs: object) -> RecallResult:
        return RecallResult(items=[item.model_copy(update={"score": 0.5 if calls else 1.0}) for item in baseline.items],
                            facts=records[:1])

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        nonlocal calls
        calls += 1
        if calls == 1:
            return EvidenceDecision(status="insufficient", selected_ids=(), followup_queries=("Beacon details",))
        assert any(candidate.id.startswith("chunk:") for candidate in candidates)
        return await sufficient(question, candidates)

    monkeypatch.setattr(memory, "recall", recall)
    result = await AdaptiveRetriever(memory, Assessor(assess), graph_limits=MultiHopLimits(max_store_calls=80)).retrieve(
        "alpha", "Aster", scope=RecallScope.validated())
    assert result.status == "sufficient" and result.recall.items
    assert len(result.graph_expansions) == 2 and all(receipt.store_calls <= 80 for receipt in result.graph_expansions)
    assert "stale_evidence" not in result.reasons
