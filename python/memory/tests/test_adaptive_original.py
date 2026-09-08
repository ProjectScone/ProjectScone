"""Original query evidence survives an inaccurate later model selection."""
from unittest.mock import AsyncMock

import pytest

from scone_memory.core.models import RecallResult
from scone_memory.retrieval.adaptive import AdaptiveLimits, AdaptiveRetriever, EvidenceDecision
from scone_memory.retrieval.recall_scope import RecallScope
from scone_memory.retrieval.multihop import MultiHopLimits
from test_adaptive_retrieval import Assessor, fact, memory


@pytest.mark.parametrize("policy", ["model_selected", "original_and_selected"])
@pytest.mark.parametrize("limit", [1, 4])
async def test_original_route_is_not_erased_by_wrong_nonempty_selection(memory, monkeypatch, policy, limit):
    original = await fact(memory, "Spruce", "routes to", "Unresolved")
    wrong = await fact(memory, "Other", "uses", "Disk")
    calls = []
    async def recall(space, query, **kwargs):
        calls.append(query)
        return RecallResult(facts=[original] if len(calls) == 1 else [wrong])
    async def assess(question, candidates):
        if len(calls) == 1:
            return EvidenceDecision(status="insufficient", selected_ids=(), followup_queries=("Other storage",))
        return EvidenceDecision(status="sufficient", selected_ids=(f"fact:{wrong.fact_id}",))
    monkeypatch.setattr(memory, "recall", recall)
    result = await AdaptiveRetriever(memory, Assessor(assess), evidence_policy=policy,
        limits=AdaptiveLimits(candidate_limit=limit)).retrieve("alpha", "Where does Spruce finish?", scope=RecallScope.validated())
    assert calls == ["Where does Spruce finish?", "Other storage"]
    if policy == "model_selected":
        assert result.recall.facts == [wrong] and result.original_query_ids == ()
        assert result.model_selected_ids == (f"fact:{wrong.fact_id}",)
        return
    assert result.recall.facts == ([original] if limit == 1 else [original, wrong])
    assert result.original_query_ids == (f"fact:{original.fact_id}",)
    assert result.model_selected_ids == (() if limit == 1 else (f"fact:{wrong.fact_id}",))
    assert result.evidence_basis == "original_and_selected"
    assert result.status == ("uncertain" if limit == 1 else "sufficient")
    assert result.rounds[-1].selected_count == 1


@pytest.mark.parametrize("change", ["delete", "replace_source"])
async def test_original_snapshot_is_never_replaced_by_later_changed_identity(memory, monkeypatch, change):
    original = await fact(memory, "Spruce", "routes to", "Unresolved")
    getter = memory.documents.get_episode
    changed = False
    async def get_episode(space, episode_id):
        source = await getter(space, episode_id)
        if changed and source is not None and episode_id == original.source_episode_id:
            if change == "delete":
                return None
            return source.model_copy(update={"content": source.content + " Later source replacement."})
        return source
    monkeypatch.setattr(memory.documents, "get_episode", get_episode)
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=[original])))
    calls = 0
    async def assess(question, candidates):
        nonlocal calls, changed
        calls += 1
        if calls == 1:
            changed = True
            return EvidenceDecision(status="insufficient", selected_ids=(), followup_queries=("Spruce details",))
        return EvidenceDecision(status="sufficient", selected_ids=tuple(c.id for c in candidates))
    result = await AdaptiveRetriever(memory, Assessor(assess), evidence_policy="original_and_selected").retrieve(
        "alpha", "Spruce destination", scope=RecallScope.validated())
    assert result.recall.facts == [] and result.original_query_ids == () and result.model_selected_ids == ()
    assert result.status == "uncertain" and "stale_evidence" in result.reasons
    assert result.errors == ()


@pytest.mark.parametrize("value", [None, True, "all", 1])
def test_evidence_policy_rejects_unknown_runtime_values(value):
    with pytest.raises(ValueError, match="evidence_policy"):
        AdaptiveRetriever(None, None, evidence_policy=value)


async def test_original_and_selected_graph_groups_compete_as_whole_units(memory, monkeypatch):
    original = [await fact(memory, left, "routes to", right) for left, right in
                (("Spruce", "Birch"), ("Birch", "Cedar"), ("Cedar", "Archive"))]
    later = [await fact(memory, left, "routes to", right) for left, right in
             (("Other", "Wedge"), ("Wedge", "Xray"), ("Xray", "Disk"))]
    calls = 0
    async def recall(*args, **kwargs):
        nonlocal calls
        calls += 1
        return RecallResult(facts=original[:1] if calls == 1 else later[:1])
    async def assess(question, candidates):
        if calls == 1:
            return EvidenceDecision(status="insufficient", selected_ids=(), followup_queries=("Other route",))
        return EvidenceDecision(status="sufficient", selected_ids=tuple(c.id for c in candidates))
    monkeypatch.setattr(memory, "recall", recall)
    result = await AdaptiveRetriever(memory, Assessor(assess), evidence_policy="original_and_selected",
        graph_limits=MultiHopLimits(), limits=AdaptiveLimits(candidate_limit=4)).retrieve(
            "alpha", "Spruce destination", scope=RecallScope.validated())
    assert result.recall.facts == original
    assert result.selected_groups == (tuple(f"fact:{r.fact_id}" for r in original),)
    assert result.model_selected_ids == () and result.status == "uncertain"
    assert {"candidate_limit", "atomic_group_omitted"}.issubset(result.reasons)


@pytest.mark.parametrize("later_round", [False, True])
async def test_learned_group_contract_prevents_restoring_a_surviving_original_fragment(memory, monkeypatch, later_round):
    first = await fact(memory, "Spruce", "routes to", "Birch")
    bridge = await fact(memory, "Birch", "routes to", "Archive")
    calls = 0
    async def recall(*args, **kwargs):
        return RecallResult(facts=[first, bridge])
    async def assess(question, candidates):
        nonlocal calls
        calls += 1
        if later_round and calls == 1:
            return EvidenceDecision(status="insufficient", selected_ids=(f"fact:{first.fact_id}", f"fact:{bridge.fact_id}"),
                selected_groups=((f"fact:{first.fact_id}", f"fact:{bridge.fact_id}"),), followup_queries=("Birch details",))
        await memory.forget("alpha", bridge.source_episode_id)
        return EvidenceDecision(status="insufficient", selected_ids=(f"fact:{first.fact_id}",) if later_round
            else (f"fact:{first.fact_id}", f"fact:{bridge.fact_id}"), selected_groups=() if later_round
            else ((f"fact:{first.fact_id}", f"fact:{bridge.fact_id}"),))
    monkeypatch.setattr(memory, "recall", recall)
    result = await AdaptiveRetriever(memory, Assessor(assess), evidence_policy="original_and_selected").retrieve(
        "alpha", "Spruce destination", scope=RecallScope.validated())
    assert result.recall.facts == [] and result.errors == ()
    assert result.original_query_ids == () and result.selected_groups == ()
    assert "atomic_group_omitted" in result.reasons
