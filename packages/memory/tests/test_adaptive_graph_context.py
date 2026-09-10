"""Pre-assessment graph expansion survives native verification and group packing."""

import json
from collections.abc import Callable

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends import SqliteDocumentStore
from scone_memory.realtime.context import MemoryContext
from scone_memory.retrieval.adaptive import AdaptiveLimits, AdaptiveRetriever, EvidenceAssessmentError, EvidenceCandidate, EvidenceDecision
from scone_memory.retrieval.multihop import MultiHopLimits


@pytest.fixture(params=["memory", "sqlite"])
async def memory(request, tmp_path):
    documents = InMemoryDocumentStore() if request.param == "memory" else SqliteDocumentStore(tmp_path / "adaptive-graph.db")
    engine = await MemoryEngine(documents, InMemoryVectorIndex(), HashEmbedder()).open()
    yield engine
    await engine.close()


async def seed_chain(memory, count=4, *, blocked_bridge=False):
    names = ["aster", "beacon", "cedar", "denver", "elysium"]
    facts, episodes = [], []
    for index in range(count):
        quote = f"{names[index]} depends on {names[index + 1]}."
        episode = await memory.remember("alpha", quote, kind="file", source=f"manuals/{names[index]}",
            metadata={"team": "other" if blocked_bridge and index == 1 else "science"})
        fact = await memory.assert_fact("alpha", names[index], "depends on", names[index + 1],
            source_episode_id=episode.episode_id, quote=quote)
        facts.append(fact)
        episodes.append(episode)
    return facts, episodes


class Assessor:
    def __init__(self, decide: Callable[[tuple[EvidenceCandidate, ...]], EvidenceDecision]) -> None:
        self.decide = decide
        self.seen: list[tuple[EvidenceCandidate, ...]] = []

    async def assess(self, question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        self.seen.append(candidates)
        return self.decide(candidates)


def select_facts(candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
    ids = tuple(candidate.id for candidate in candidates if candidate.id.startswith("fact:"))
    return EvidenceDecision(status="sufficient", selected_ids=ids, selected_groups=(ids,) if len(ids) > 1 else ())


def strategy(memory, assessor):
    return AdaptiveRetriever(memory, assessor, limits=AdaptiveLimits(timeout_s=5.0),
        graph_limits=MultiHopLimits(max_hops=4, max_nodes=16, max_bytes=16000, max_store_calls=128))


def packet(request):
    return next(json.loads(message["content"].split("\n", 1)[1]) for message in request
                if message.get("content", "").startswith("Scone retrieved source material:"))


@pytest.mark.parametrize("count", [3, 4])
async def test_actual_seed_expands_complete_path_before_assessment_and_native_delivery(memory, count):
    facts, _ = await seed_chain(memory, count)
    initial = await memory.recall("alpha", "aster")
    assert [fact.fact_id for fact in initial.facts] == [facts[0].fact_id]
    assessor = Assessor(select_facts)
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=strategy(memory, assessor),
                                           recall_timeout=5.0).prepare([{"role": "user", "content": "aster"}])
    expected = {f"fact:{fact.fact_id}" for fact in facts}
    assert {candidate.id for candidate in assessor.seen[0] if candidate.id.startswith("fact:")} == expected
    data = packet(request)
    assert {claim["fact_id"] for claim in data["claims"]} == {fact.fact_id for fact in facts}
    assert any(path["fact_ids"] == [fact.fact_id for fact in facts] for path in data["paths"])
    assert facts[-1].object in json.dumps(data)
    assert receipt["adaptive_atomic_group_omitted_count"] == 0
    assert data["coverage"]["adaptive"]["selection_complete"] is True
    [expansion] = receipt["adaptive_graph_expansions"]
    assert expansion["added_count"] == count - 1 and expansion["store_calls"] > 0
    assert expansion["complete"] is True and expansion["truncated"] is False
    assert set(expansion) == {"candidate_count", "added_count", "omitted_count", "store_calls", "complete", "truncated", "reasons"}
    assert data["coverage"]["adaptive"]["graph_expansions"] == [expansion]
    assert "aster" not in json.dumps(expansion)


async def test_first_assessment_failure_retains_host_group_and_complete_path(memory):
    facts, _ = await seed_chain(memory)

    def fail(candidates):
        raise EvidenceAssessmentError("assessment_provider_failed")

    assessor = Assessor(fail)
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=strategy(memory, assessor),
                                           recall_timeout=5.0).prepare([{"role": "user", "content": "aster"}])
    data = packet(request)
    assert {claim["fact_id"] for claim in data["claims"]} == {fact.fact_id for fact in facts}
    assert any(path["fact_ids"] == [fact.fact_id for fact in facts] for path in data["paths"])
    assert receipt["adaptive_status"] == "uncertain" and receipt["adaptive_fallback_status"] == "retained"
    assert receipt["adaptive_atomic_group_omitted_count"] == 0
    assert data["coverage"]["adaptive"]["model_selected"] is False
    assert receipt["adaptive_graph_expansions"][0]["added_count"] == 3
    assert len(assessor.seen) == 1


async def test_scope_invalid_bridge_never_reappears_before_or_after_assessment(memory):
    facts, _ = await seed_chain(memory, blocked_bridge=True)
    assessor = Assessor(lambda candidates: EvidenceDecision(status="insufficient",
        selected_ids=tuple(candidate.id for candidate in candidates if candidate.id.startswith("fact:"))))
    request, receipt = await MemoryContext(memory, "alpha", "current", where={"team": "science"},
        adaptive_retriever=strategy(memory, assessor), recall_timeout=5.0).prepare([{"role": "user", "content": "aster"}])
    assert {candidate.id for candidate in assessor.seen[0] if candidate.id.startswith("fact:")} == {f"fact:{facts[0].fact_id}"}
    data = packet(request)
    assert {claim["fact_id"] for claim in data["claims"]} == {facts[0].fact_id}
    assert not data.get("paths") and facts[-1].object not in json.dumps(data)
    assert receipt["adaptive_status"] == "insufficient"


async def test_deleted_bridge_removes_host_group_and_preserves_independent_record(memory):
    facts, episodes = await seed_chain(memory)
    independent = await memory.remember("alpha", "aster independent scheduling note.")

    class DeleteBridge:
        async def assess(self, question, candidates):
            ids = tuple(candidate.id for candidate in candidates if candidate.id.startswith("fact:"))
            assert set(ids) == {f"fact:{fact.fact_id}" for fact in facts}
            independent_id = next(candidate.id for candidate in candidates if candidate.episode_id == independent.episode_id)
            await memory.forget("alpha", episodes[1].episode_id)
            return EvidenceDecision(status="sufficient", selected_ids=(*ids, independent_id), selected_groups=(ids,))

    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=strategy(memory, DeleteBridge()),
                                           recall_timeout=5.0).prepare([{"role": "user", "content": "aster"}])
    data = packet(request)
    assert not data.get("claims") and not data.get("paths")
    assert {source["episode_id"] for source in data["sources"]} == {independent.episode_id}
    assert receipt["adaptive_status"] == "uncertain" and "atomic_group_omitted" in receipt["adaptive_reasons"]


async def test_fallback_path_is_atomic_when_native_byte_budget_cannot_fit_group(memory):
    facts, _ = await seed_chain(memory)
    independent = await memory.remember("alpha", "aster independent scheduling note.")

    def fail(candidates):
        raise EvidenceAssessmentError("assessment_provider_failed")

    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=strategy(memory, Assessor(fail)),
        recall_timeout=5.0, max_context_bytes=1800).prepare([{"role": "user", "content": "aster"}])
    data = packet(request)
    assert not data.get("claims") and not data.get("paths")
    assert data["sources"]
    assert receipt["adaptive_atomic_group_omitted_count"] == 1
    assert data["coverage"]["adaptive"]["selected_omitted_count"] >= len(facts)
    assert receipt["context_bytes"] <= 1800
    assert data["coverage"]["adaptive"]["model_selected"] is False


async def test_disabled_preassessment_graph_preserves_flat_selected_evidence(memory):
    facts, _ = await seed_chain(memory)
    adaptive = AdaptiveRetriever(memory, Assessor(select_facts), limits=AdaptiveLimits(timeout_s=5.0))
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive,
                                           recall_timeout=5.0).prepare([{"role": "user", "content": "aster"}])
    data = packet(request)
    assert {claim["fact_id"] for claim in data["claims"]} == {facts[0].fact_id}
    assert not data.get("paths")
    assert "adaptive_graph_expansions" not in receipt and "graph_expansions" not in data["coverage"]["adaptive"]


async def test_preassessment_graph_diagnostics_cannot_leak_raw_reason_text(memory):
    await seed_chain(memory)

    class UnsafeDiagnosticRetriever(AdaptiveRetriever):
        async def retrieve(self, *args, **kwargs):
            result = await super().retrieve(*args, **kwargs)
            expansion = result.graph_expansions[0].model_copy(update={"reasons": ("PRIVATE raw query and model response",)})
            return result.model_copy(update={"graph_expansions": (expansion,)})

    adaptive = UnsafeDiagnosticRetriever(memory, Assessor(select_facts), limits=AdaptiveLimits(timeout_s=5.0),
        graph_limits=MultiHopLimits(max_hops=4, max_nodes=16, max_bytes=16000, max_store_calls=128))
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive,
                                           recall_timeout=5.0).prepare([{"role": "user", "content": "aster"}])
    assert receipt["adaptive_graph_expansions"][0]["reasons"] == ["unknown"]
    assert packet(request)["coverage"]["adaptive"]["graph_expansions"][0]["reasons"] == ["unknown"]
    assert "PRIVATE" not in json.dumps(receipt) and "PRIVATE" not in json.dumps(request)


async def test_preassessment_graph_limit_is_visible_without_late_widening(memory):
    facts, _ = await seed_chain(memory)
    adaptive = AdaptiveRetriever(memory, Assessor(select_facts), limits=AdaptiveLimits(timeout_s=5.0),
        graph_limits=MultiHopLimits(max_hops=0, max_nodes=16, max_bytes=16000, max_store_calls=128))
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive,
                                           recall_timeout=5.0).prepare([{"role": "user", "content": "aster"}])
    data = packet(request)
    assert {claim["fact_id"] for claim in data["claims"]} == {facts[0].fact_id}
    assert not data.get("paths")
    [expansion] = receipt["adaptive_graph_expansions"]
    assert expansion["truncated"] is True and expansion["complete"] is False
    assert expansion["reasons"] == ["max_hops"]
    assert "max_hops" in receipt["adaptive_reasons"]
    assert data["coverage"]["adaptive"]["graph_expansions"] == [expansion]
