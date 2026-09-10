"""Adaptive conversation retrieval uses retained evidence without storing assessment."""

import asyncio
import copy
import json
from collections.abc import AsyncIterator, Callable

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.realtime.context import MemoryContext
from scone_memory.realtime.events import ReplyCompleted, TextDelta
from scone_memory.realtime.text import TextConversation
from scone_memory.retrieval.adaptive import AdaptiveLimits, AdaptiveRetriever, EvidenceCandidate, EvidenceDecision


@pytest.fixture
async def memory() -> MemoryEngine:
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()


class ScriptedAssessor:
    def __init__(self, decide: Callable[[int, tuple[EvidenceCandidate, ...]], EvidenceDecision]) -> None:
        self.decide = decide
        self.calls: list[tuple[str, tuple[EvidenceCandidate, ...]]] = []

    async def assess(self, question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        self.calls.append((question, candidates))
        return self.decide(len(self.calls), candidates)


def select_all(round_number: int, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
    return EvidenceDecision(status="sufficient", selected_ids=tuple(c.id for c in candidates))


def retriever(memory: MemoryEngine, assessor: ScriptedAssessor) -> AdaptiveRetriever:
    return AdaptiveRetriever(memory, assessor, limits=AdaptiveLimits(timeout_s=1.0))


def payload(request: list[dict[str, object]]) -> dict:
    block = next(message["content"] for message in request
                 if isinstance(message.get("content"), str) and "Scone retrieved source material" in message["content"])
    assert isinstance(block, str)
    return json.loads(block.split("\n", 1)[1])


async def test_two_hop_followup_preserves_fixed_scope_and_only_selected_evidence(memory):
    options = dict(kind="file", source="manuals/approved", created_at="2026-09-02T10:00:00Z", metadata={"team": "science"})
    bridge = await memory.remember("alpha", "Juniper calibration depends on the Meridian reference.", **options)
    endpoint = await memory.remember("alpha", "The Meridian reference is maintained by Celeste.", **options)
    await memory.remember("alpha", "Juniper cafeteria serves scones.", **options)
    for space, changes in [("beta", {}), ("alpha", {"metadata": {"team": "legal"}}),
                           ("alpha", {"source": "private/manual"}),
                           ("alpha", {"created_at": "2026-08-01T10:00:00Z"}),
                           ("alpha", {"metadata": {"team": "science", "session_id": "current"}})]:
        await memory.remember(space, "Juniper Meridian PRIVATE source", **(options | changes))

    def decide(round_number, candidates):
        assert all("PRIVATE" not in c.text for c in candidates)
        useful = tuple(c.id for c in candidates if c.episode_id in {bridge.episode_id, endpoint.episode_id})
        return EvidenceDecision(status="insufficient" if round_number == 1 else "sufficient", selected_ids=useful,
                                followup_queries=("Who maintains the Meridian reference?",) if round_number == 1 else ())

    assessor = ScriptedAssessor(decide)
    where = {"team": "science"}
    context = MemoryContext(memory, "alpha", "current", where=where, kind="file", source_prefix="manuals/",
                            since="2026-09-01T00:00:00Z", until="2026-09-06T23:59:59Z",
                            adaptive_retriever=retriever(memory, assessor))
    where["team"] = "legal"
    messages = [{"role": "user", "content": "Who maintains Juniper's calibration reference?"}]
    request, receipt = await context.prepare(messages)
    assert receipt["status"] == "prepared"
    assert {r["episode_id"] for r in receipt["references"]} == {bridge.episode_id, endpoint.episode_id}
    assert receipt["adaptive_status"] == "sufficient"
    assert receipt["adaptive_round_count"] == 2 and receipt["adaptive_queries_used"] == 2
    assert len(assessor.calls) == 2
    assert [c[0] for c in assessor.calls] == [messages[0]["content"]] * 2
    assert "cafeteria" not in request[0]["content"]
    adaptive = payload(request)["coverage"]["adaptive"]
    assert adaptive["assessment_status"] == "sufficient" and adaptive["assessment_basis"] == "model_judgment"
    assert adaptive["verified_sufficiency"] is False and adaptive["bounded"] is True


@pytest.mark.parametrize("status", ["insufficient", "uncertain"])
async def test_missing_link_retains_useful_evidence_with_incomplete_coverage(memory, status):
    await memory.remember("alpha", "Juniper calibration depends on Meridian; its maintainer is not recorded.")
    assessor = ScriptedAssessor(lambda _, candidates: EvidenceDecision(status=status, selected_ids=(candidates[0].id,)))
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=retriever(memory, assessor)).prepare([
        {"role": "user", "content": "Who maintains Juniper calibration?"}])
    assert receipt["status"] == "prepared" and receipt["adaptive_status"] == status
    assert receipt["references"] and receipt["low_confidence"] is None
    coverage = payload(request)["coverage"]
    assert coverage["adaptive"]["assessment_status"] == status
    assert coverage["adaptive"]["assessment_basis"] == "bounded_retrieval_assessment"
    assert coverage["complete_history"] is False
    assert coverage["context_omitted_count"] == 0
    assert "no_followup_queries" in receipt["adaptive_reasons"]


async def test_changed_source_during_assessment_is_removed(memory):
    episode = await memory.remember("alpha", "Juniper calibration uses Meridian.")

    class DeletingAssessor:
        async def assess(self, question, candidates):
            await memory.forget("alpha", episode.episode_id)
            return EvidenceDecision(status="sufficient", selected_ids=(candidates[0].id,))

    adaptive = AdaptiveRetriever(memory, DeletingAssessor(), limits=AdaptiveLimits(timeout_s=1.0))
    messages = [{"role": "user", "content": "Juniper calibration?"}]
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive).prepare(messages)
    assert request == messages and receipt["status"] == "empty"
    assert receipt["adaptive_status"] == "uncertain" and receipt["references"] == []
    assert "stale_evidence" in receipt["adaptive_reasons"]


async def test_explicit_empty_failure_policy_has_safe_receipt_and_never_falls_back(memory):
    await memory.remember("alpha", "Juniper calibration PRIVATE source.")

    def fail(_, candidates):
        raise RuntimeError("PRIVATE model response and API key")

    assessor = ScriptedAssessor(fail)
    messages = [{"role": "user", "content": "Juniper calibration?"}]
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=AdaptiveRetriever(memory, assessor,
        limits=AdaptiveLimits(timeout_s=1.0), failure_policy="empty")).prepare(messages)
    assert request == messages and receipt["status"] == "empty"
    assert receipt["adaptive_evidence_basis"] == "none" and receipt["adaptive_fallback_status"] == "not_used"
    assert receipt["adaptive_status"] == "uncertain"
    assert receipt["adaptive_errors"] == ["invalid_or_failed_assessment"]
    assert receipt["low_confidence"] is None and receipt["references"] == []
    assert "PRIVATE" not in json.dumps(receipt)


async def test_explicit_optout_preserves_existing_context_payload(memory):
    await memory.remember("alpha", "Juniper calibration uses Meridian.")
    messages = [{"role": "user", "content": "Juniper calibration?"}]
    ordinary, ordinary_receipt = await MemoryContext(memory, "alpha", "current").prepare(messages)
    opted_out, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=None).prepare(messages)
    assert opted_out == ordinary
    assert not any(key.startswith("adaptive_") for key in receipt)
    assert receipt["references"] == ordinary_receipt["references"]
    assert payload(opted_out)["coverage"] == {"mode": "ranked_search"}


@pytest.mark.parametrize("question,expected", [("Hello!", "skipped"), ("Catch me up", "prepared")])
async def test_overview_and_social_routing_skip_assessor(memory, question, expected):
    await memory.remember("alpha", "Juniper calibration completed yesterday. We are preparing the local microphone interface for testing.")
    assessor = ScriptedAssessor(select_all)
    _, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=retriever(memory, assessor)).prepare([
        {"role": "user", "content": question}])
    assert receipt["status"] == expected
    assert not assessor.calls and "adaptive_status" not in receipt


async def test_adaptive_cancellation_propagates(memory):
    await memory.remember("alpha", "Juniper calibration uses Meridian.")
    entered = asyncio.Event()

    class WaitingAssessor:
        async def assess(self, question, candidates):
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    adaptive = AdaptiveRetriever(memory, WaitingAssessor(), limits=AdaptiveLimits(timeout_s=1.0))
    context = MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive)
    task = asyncio.create_task(context.prepare([{"role": "user", "content": "Juniper calibration?"}]))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("entrypoint", ["context", "conversation"])
async def test_adaptive_rejects_other_engine_and_short_timeout(memory, entrypoint):
    other = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    assessor = ScriptedAssessor(select_all)
    make = (lambda **kwargs: MemoryContext(memory, "alpha", "current", **kwargs)) if entrypoint == "context" else (
        lambda **kwargs: TextConversation(memory, "alpha", "current", PublicModel, **kwargs))
    with pytest.raises(ValueError, match="same memory"):
        make(adaptive_retriever=retriever(other, assessor))
    with pytest.raises(ValueError, match="recall_timeout.*timeout_s"):
        make(adaptive_retriever=AdaptiveRetriever(memory, assessor))
    make(adaptive_retriever=AdaptiveRetriever(memory, assessor), recall_timeout=30.0)


class PublicModel:
    def __init__(self) -> None:
        self.requests: list[list[dict[str, str]]] = []

    async def respond(self, messages: list[dict[str, str]]) -> AsyncIterator[TextDelta | ReplyCompleted]:
        self.requests.append(copy.deepcopy(messages))
        yield TextDelta("Celeste maintains the Meridian reference.")
        yield ReplyCompleted()

    async def aclose(self) -> None:
        pass


async def test_text_conversation_uses_adaptive_stage_and_stores_only_public_reply(memory):
    await memory.remember("alpha", "Juniper calibration uses the Meridian reference maintained by Celeste.")
    assessor = ScriptedAssessor(select_all)
    model = PublicModel()
    conversation = TextConversation(memory, "alpha", "current", lambda: model,
                                    adaptive_retriever=retriever(memory, assessor), recall_timeout=2.0)
    question = "Who maintains Juniper calibration?"
    response = await conversation.reply(question)
    assert response["memory_context"]["adaptive_status"] == "sufficient"
    assert "model_judgment" in model.requests[0][1]["content"]
    episodes = await memory.episodes("alpha", {"session_id": "current"})
    assert [episode.content for episode in episodes] == [question, "Celeste maintains the Meridian reference."]
    assert len(assessor.calls) == 1
    assert all(candidate.episode_id != response["user_episode_id"] for candidate in assessor.calls[0][1])
    await conversation.close()


@pytest.mark.parametrize("failure_policy", ["empty", "retain_verified"])
async def test_equal_deadlines_preserve_adaptive_timeout_diagnostics(memory, failure_policy):
    await memory.remember("alpha", "Juniper calibration depends on Meridian.")

    class WaitingAssessor:
        async def assess(self, question, candidates):
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    adaptive = AdaptiveRetriever(memory, WaitingAssessor(), limits=AdaptiveLimits(timeout_s=1.0), failure_policy=failure_policy)
    messages = [{"role": "user", "content": "Juniper calibration?"}]
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive, recall_timeout=1.0).prepare(messages)
    if failure_policy == "empty":
        assert request == messages and receipt["status"] == "empty"
        assert receipt["adaptive_fallback_status"] == "not_used"
    else:
        assert receipt["status"] == "prepared" and len(request) == 2
        assert receipt["adaptive_fallback_status"] == "retained"
        assert payload(request)["coverage"]["adaptive"]["model_selected"] is False
    assert receipt["adaptive_status"] == "uncertain" and receipt["adaptive_truncated"] is True
    assert receipt["adaptive_errors"] == ["timeout"]


async def test_bounded_assessment_and_context_omissions_are_visible_to_model(memory):
    for index in range(3):
        await memory.remember("alpha", f"Juniper calibration note {index}: Meridian reference details remain incomplete.")
    assessor = ScriptedAssessor(lambda _, candidates: EvidenceDecision(status="insufficient",
        selected_ids=tuple(c.id for c in candidates), followup_queries=("Meridian maintainer",)))
    adaptive = AdaptiveRetriever(memory, assessor, limits=AdaptiveLimits(max_rounds=1, timeout_s=1.0))
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive,
                                           limit=1, max_context_bytes=1800).prepare([
        {"role": "user", "content": "Juniper calibration?"}])
    assert receipt["status"] == "prepared" and receipt["context_bytes"] <= 1800
    assert receipt["adaptive_truncated"] is True and receipt["omitted_count"] == 2
    coverage = payload(request)["coverage"]
    assert coverage["adaptive"]["truncated"] is True
    assert coverage["adaptive"]["reasons"] == ["max_rounds"]
    assert coverage["context_omitted_count"] == 2
    assert coverage["adaptive"]["selection_complete"] is False
    assert coverage["adaptive"]["selected_omitted_count"] == 2


async def test_followup_retrieves_endpoint_missing_from_first_candidate_window(memory):
    bridge = await memory.remember("alpha", "Juniper calibration depends on the Meridian reference.")
    endpoint = await memory.remember("alpha", "The Meridian reference is maintained by Celeste.")
    distractor = await memory.remember("alpha", "Juniper cafeteria serves scones.")

    def decide(round_number, candidates):
        if round_number == 1:
            assert {c.episode_id for c in candidates} == {bridge.episode_id, distractor.episode_id}
            return EvidenceDecision(status="insufficient", selected_ids=tuple(c.id for c in candidates if c.episode_id == bridge.episode_id),
                                    followup_queries=("Who maintains the Meridian reference?",))
        assert {c.episode_id for c in candidates} == {bridge.episode_id, endpoint.episode_id}
        return EvidenceDecision(status="sufficient", selected_ids=tuple(c.id for c in candidates))

    assessor = ScriptedAssessor(decide)
    adaptive = AdaptiveRetriever(memory, assessor, limits=AdaptiveLimits(candidate_limit=2, timeout_s=1.0))
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive).prepare([
        {"role": "user", "content": "Juniper calibration"}])
    assert receipt["adaptive_status"] == "sufficient" and receipt["adaptive_round_count"] == 2
    assert {s["episode_id"] for s in payload(request)["sources"]} == {bridge.episode_id, endpoint.episode_id}
    assert receipt["adaptive_queries_used"] == 2


async def test_context_rechecks_sources_after_adaptive_return(memory):
    episode = await memory.remember("alpha", "Juniper calibration depends on Meridian.")

    class DeleteAfterRetrieval(AdaptiveRetriever):
        async def retrieve(self, *args, **kwargs):
            result = await super().retrieve(*args, **kwargs)
            await memory.forget("alpha", episode.episode_id)
            return result

    adaptive = DeleteAfterRetrieval(memory, ScriptedAssessor(select_all), limits=AdaptiveLimits(timeout_s=1.0))
    messages = [{"role": "user", "content": "Juniper calibration?"}]
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive).prepare(messages)
    assert request == messages and receipt["status"] == "empty"
    assert receipt["references"] == [] and receipt["omitted_count"] == 1


async def test_adaptive_paths_connect_selected_facts_without_resurfacing_rejected_neighbor(memory):
    facts, episodes = [], []
    for subject, predicate, obj in [("juniper", "depends on", "meridian"),
                                    ("meridian", "operated by", "celeste"),
                                    ("celeste", "located in", "REJECTED_LOCATION")]:
        quote = f"{subject} {predicate} {obj}."
        episode = await memory.remember("alpha", quote)
        facts.append(await memory.assert_fact("alpha", subject, predicate, obj,
                                              source_episode_id=episode.episode_id, quote=quote))
        episodes.append(episode)
    selected_facts = {facts[0].fact_id, facts[1].fact_id}

    def decide(_, candidates):
        assert f"fact:{facts[2].fact_id}" in {candidate.id for candidate in candidates}
        selected = tuple(candidate.id for candidate in candidates
                         if candidate.id in {f"fact:{fact_id}" for fact_id in selected_facts}
                         or (candidate.id.startswith("chunk:") and candidate.episode_id == episodes[0].episode_id))
        return EvidenceDecision(status="sufficient", selected_ids=selected)

    adaptive = retriever(memory, ScriptedAssessor(decide))
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive).prepare([
        {"role": "user", "content": "Juniper Meridian Celeste"}])
    data = payload(request)
    assert receipt["adaptive_status"] == "sufficient"
    assert {claim["fact_id"] for claim in data["claims"]} == selected_facts
    assert {source["episode_id"] for source in data["sources"]} == {episodes[0].episode_id}
    assert data["paths"] == [{"fact_ids": [facts[0].fact_id, facts[1].fact_id], "steps": [
        {"from_fact": facts[0].fact_id, "to_fact": facts[1].fact_id, "kind": "subject_object", "direction": "forward"}]}]
    assert "REJECTED_LOCATION" not in json.dumps(data)
    assert receipt["multihop_coverage"]["selection_restricted"] is True
    assert receipt["multihop_coverage"]["selection_omitted_facts"] == 1
    assert data["coverage"]["adaptive"]["selection_complete"] is True
    assert data["coverage"]["adaptive"]["selected_omitted_count"] == 0


async def test_byte_budget_reports_partial_delivery_of_sufficient_selection(memory):
    for index in range(2):
        await memory.remember("alpha", f"Juniper calibration note {index}: " + "Meridian reference details. " * 15)
    adaptive = retriever(memory, ScriptedAssessor(select_all))
    messages = [{"role": "user", "content": "Juniper calibration?"}]
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive,
                                           max_context_bytes=1600).prepare(messages)
    data = payload(request)
    assert len(data["sources"]) == 1
    assert receipt["adaptive_status"] == "sufficient"
    assert data["coverage"]["adaptive"]["selection_complete"] is False
    assert data["coverage"]["adaptive"]["selected_omitted_count"] == 1
    assert data["coverage"]["adaptive"]["verified_sufficiency"] is False
    assert len(request[0]["content"].encode()) == receipt["context_bytes"] <= 1600
    complete, _ = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive).prepare(messages)
    assert len(payload(complete)["sources"]) == 2
    assert payload(complete)["coverage"]["adaptive"]["selection_complete"] is True
    assert payload(complete)["coverage"]["adaptive"]["selected_omitted_count"] == 0


async def test_adaptive_graph_failure_never_supplies_deleted_selected_source(memory, monkeypatch):
    episode = await memory.remember("alpha", "Juniper calibration depends on Meridian.")

    class DeleteAfterRetrieval(AdaptiveRetriever):
        async def retrieve(self, *args, **kwargs):
            result = await super().retrieve(*args, **kwargs)
            await memory.forget("alpha", episode.episode_id)
            return result

    async def unavailable(*args, **kwargs):
        raise RuntimeError("PRIVATE graph failure")

    monkeypatch.setattr("scone_memory.realtime.context.build_query_evidence_graph", unavailable)
    adaptive = DeleteAfterRetrieval(memory, ScriptedAssessor(select_all), limits=AdaptiveLimits(timeout_s=1.0))
    messages = [{"role": "user", "content": "Juniper calibration?"}]
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive).prepare(messages)
    assert request == messages and receipt["status"] == "empty"
    assert receipt["references"] == [] and receipt["omitted_count"] == 1
    assert receipt["evidence_graph_status"] == "unavailable"
    assert "PRIVATE" not in json.dumps(receipt)


@pytest.mark.parametrize("reason", ["assessment_timeout", "assessment_provider_failed", "invalid_assessment"])
async def test_typed_assessment_errors_preserve_safe_native_receipt(memory, reason):
    from scone_memory.retrieval.adaptive import EvidenceAssessmentError

    await memory.remember("alpha", "Juniper calibration PRIVATE source.")

    def fail(_, candidates):
        raise EvidenceAssessmentError(reason)

    adaptive = AdaptiveRetriever(memory, ScriptedAssessor(fail), limits=AdaptiveLimits(timeout_s=1.0), failure_policy="empty")
    messages = [{"role": "user", "content": "Juniper calibration?"}]
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive).prepare(messages)
    assert request == messages and receipt["status"] == "empty"
    assert receipt["adaptive_status"] == "uncertain" and receipt["adaptive_errors"] == [reason]
    assert receipt["references"] == [] and "PRIVATE" not in json.dumps(receipt)


async def test_changed_selected_fact_cannot_return_through_expansion_under_same_id(memory):
    first = await memory.remember("alpha", "juniper depends on meridian.")
    second = await memory.remember("alpha", "meridian operated by old. meridian operated by new.")
    stable = await memory.assert_fact("alpha", "juniper", "depends on", "meridian",
                                     source_episode_id=first.episode_id, quote="juniper depends on meridian.")
    changed = await memory.assert_fact("alpha", "meridian", "operated by", "old",
                                      source_episode_id=second.episode_id, quote="meridian operated by old.")
    assessor = ScriptedAssessor(lambda _, candidates: EvidenceDecision(status="sufficient",
        selected_ids=tuple(candidate.id for candidate in candidates if candidate.id.startswith("fact:"))))

    class ReplaceAfterRetrieval(AdaptiveRetriever):
        async def retrieve(self, *args, **kwargs):
            result = await super().retrieve(*args, **kwargs)
            await memory.documents.update_fact(changed.model_copy(update={"object": "new", "quote": "meridian operated by new."}))
            return result

    adaptive = ReplaceAfterRetrieval(memory, assessor, limits=AdaptiveLimits(timeout_s=1.0))
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive).prepare([
        {"role": "user", "content": "juniper meridian"}])
    data = payload(request)
    assert {claim["fact_id"] for claim in data["claims"]} == {stable.fact_id}
    assert "meridian operated by new." not in json.dumps(data)
    assert data["coverage"]["adaptive"]["selection_complete"] is False
    assert data["coverage"]["adaptive"]["selected_omitted_count"] == 1
    assert receipt["multihop_coverage"]["selection_omitted_facts"] == 1


async def test_byte_budget_omits_entire_atomic_group_and_keeps_independent_record(memory):
    independent = await memory.remember("alpha", "Juniper calibration has an independent scheduling note.")
    left = await memory.remember("alpha", "Juniper calibration grouped left: " + "Meridian reference details. " * 13)
    right = await memory.remember("alpha", "Juniper calibration grouped right: " + "Celeste maintains reference. " * 13)

    def decide(_, candidates):
        ids = {candidate.episode_id: candidate.id for candidate in candidates if candidate.id.startswith("chunk:")}
        return EvidenceDecision(status="sufficient",
            selected_ids=(ids[independent.episode_id], ids[left.episode_id], ids[right.episode_id]),
            selected_groups=((ids[left.episode_id], ids[right.episode_id]),))

    adaptive = retriever(memory, ScriptedAssessor(decide))
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive,
                                           max_context_bytes=1800).prepare([
        {"role": "user", "content": "Juniper calibration?"}])
    data = payload(request)
    assert {source["episode_id"] for source in data["sources"]} == {independent.episode_id}
    assert receipt["adaptive_atomic_group_omitted_count"] == 1
    assert data["coverage"]["adaptive"]["atomic_group_omitted_count"] == 1
    assert data["coverage"]["adaptive"]["selected_omitted_count"] == 2
    assert data["coverage"]["adaptive"]["selection_complete"] is False
    assert receipt["context_bytes"] == len(request[0]["content"].encode()) <= 1800
    assert {ref["episode_id"] for ref in receipt["references"]} == {independent.episode_id}


@pytest.mark.parametrize("delete_group_member", [False, True])
async def test_atomic_mixed_group_source_loss_prunes_dependent_paths_and_relations(memory, delete_group_member):
    first = await memory.remember("alpha", "juniper depends on meridian.")
    second = await memory.remember("alpha", "meridian operated by celeste.")
    grouped_source = await memory.remember("alpha", "Juniper calibration group supporting source.")
    stable = await memory.assert_fact("alpha", "juniper", "depends on", "meridian",
                                     source_episode_id=first.episode_id, quote="juniper depends on meridian.")
    grouped_fact = await memory.assert_fact("alpha", "meridian", "operated by", "celeste",
                                           source_episode_id=second.episode_id, quote="meridian operated by celeste.")
    await memory.link_facts("alpha", stable.fact_id, grouped_fact.fact_id, "supports",
                            source_episode_id=first.episode_id, quote="juniper depends on meridian.")

    def decide(_, candidates):
        chunk_id = next(candidate.id for candidate in candidates
                        if candidate.episode_id == grouped_source.episode_id and candidate.id.startswith("chunk:"))
        return EvidenceDecision(status="sufficient",
            selected_ids=(f"fact:{stable.fact_id}", f"fact:{grouped_fact.fact_id}", chunk_id),
            selected_groups=((f"fact:{grouped_fact.fact_id}", chunk_id),))

    class DeleteAfterRetrieval(AdaptiveRetriever):
        async def retrieve(self, *args, **kwargs):
            result = await super().retrieve(*args, **kwargs)
            if delete_group_member:
                await memory.forget("alpha", grouped_source.episode_id)
            return result

    adaptive = DeleteAfterRetrieval(memory, ScriptedAssessor(decide), limits=AdaptiveLimits(timeout_s=1.0))
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive).prepare([
        {"role": "user", "content": "juniper meridian calibration"}])
    data = payload(request)
    if delete_group_member:
        assert {claim["fact_id"] for claim in data["claims"]} == {stable.fact_id}
        assert data["sources"] == [] and not data.get("paths") and not data.get("relations")
        assert receipt["adaptive_atomic_group_omitted_count"] == 1
        assert data["coverage"]["adaptive"]["selected_omitted_count"] == 2
        assert data["coverage"]["adaptive"]["selection_complete"] is False
        assert receipt["path_count"] == 0
        assert f"claim:{grouped_fact.fact_id}" not in {node["id"] for node in receipt["evidence_graph"]["nodes"]}
        assert set(receipt["claim_fingerprints"]) == {str(stable.fact_id)}
    else:
        assert {claim["fact_id"] for claim in data["claims"]} == {stable.fact_id, grouped_fact.fact_id}
        assert len(data["sources"]) == 1 and data["paths"] and data["relations"]
        assert receipt["adaptive_atomic_group_omitted_count"] == 0
        assert data["coverage"]["adaptive"]["selected_omitted_count"] == 0
        assert data["coverage"]["adaptive"]["selection_complete"] is True


@pytest.mark.parametrize("mutation", ["delete_group_source", "unrelated_write"])
async def test_atomic_groups_fail_closed_when_memory_changes_during_graph_read(memory, monkeypatch, mutation):
    episodes, facts = [], []
    for subject, predicate, obj in [("juniper", "depends on", "meridian"), ("meridian", "operated by", "celeste")]:
        quote = f"{subject} {predicate} {obj}."
        episode = await memory.remember("alpha", quote)
        episodes.append(episode)
        facts.append(await memory.assert_fact("alpha", subject, predicate, obj,
                                             source_episode_id=episode.episode_id, quote=quote))

    def decide(_, candidates):
        selected = tuple(candidate.id for candidate in candidates if candidate.id.startswith("fact:"))
        return EvidenceDecision(status="sufficient", selected_ids=selected, selected_groups=(selected,))

    original_links = memory.documents.fact_links_between

    async def mutate_after_sources_checked(space, fact_ids, limit):
        if mutation == "delete_group_source":
            await memory.forget("alpha", episodes[0].episode_id)
        else:
            await memory.remember("alpha", "An unrelated scheduling update arrived during graph verification.")
        return await original_links(space, fact_ids, limit)

    monkeypatch.setattr(memory.documents, "fact_links_between", mutate_after_sources_checked)
    adaptive = retriever(memory, ScriptedAssessor(decide))
    messages = [{"role": "user", "content": "juniper meridian"}]
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive,
                                           structured_paths=False).prepare(messages)
    assert request == messages and receipt["status"] == "empty"
    assert receipt["evidence_graph_status"] == "unavailable"
    assert receipt["claim_count"] == 0 and receipt["references"] == []
    assert receipt["adaptive_atomic_group_omitted_count"] == 1
    assert "stale_evidence" in receipt["adaptive_reasons"]
    assert "evidence_graph" not in receipt and "claim_fingerprints" not in receipt


async def test_ordinary_context_also_omits_confirmed_stale_graph_sources(memory, monkeypatch):
    episode = await memory.remember("alpha", "juniper depends on meridian.")
    await memory.assert_fact("alpha", "juniper", "depends on", "meridian",
                             source_episode_id=episode.episode_id, quote="juniper depends on meridian.")
    original_links = memory.documents.fact_links_between

    async def delete_after_source_checked(space, fact_ids, limit):
        await memory.forget("alpha", episode.episode_id)
        return await original_links(space, fact_ids, limit)

    monkeypatch.setattr(memory.documents, "fact_links_between", delete_after_source_checked)
    messages = [{"role": "user", "content": "juniper meridian"}]
    request, receipt = await MemoryContext(memory, "alpha", "current", structured_paths=False).prepare(messages)
    assert request == messages and receipt["status"] == "empty"
    assert receipt["evidence_graph_status"] == "unavailable" and receipt["evidence_graph_stale"] is True
    assert receipt["references"] == [] and receipt["claim_count"] == 0


@pytest.mark.parametrize("reason", ["assessment_provider_failed", "assessment_timeout", "invalid_assessment"])
async def test_assessor_failure_delivers_verified_candidates_with_unassessed_fallback_labels(memory, reason):
    from scone_memory.retrieval.adaptive import EvidenceAssessmentError

    episode = await memory.remember("alpha", "Juniper calibration uses the Meridian reference.")

    def fail(_, candidates):
        raise EvidenceAssessmentError(reason)

    assessor = ScriptedAssessor(fail)
    adaptive = retriever(memory, assessor)
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive).prepare([
        {"role": "user", "content": "Juniper calibration?"}])
    assert receipt["status"] == "prepared" and {ref["episode_id"] for ref in receipt["references"]} == {episode.episode_id}
    assert receipt["adaptive_status"] == "uncertain" and receipt["adaptive_errors"] == [reason]
    assert receipt["adaptive_evidence_basis"] == "verified_candidates" and receipt["adaptive_fallback_status"] == "retained"
    coverage = payload(request)["coverage"]["adaptive"]
    assert coverage["evidence_basis"] == "verified_candidates" and coverage["fallback_status"] == "retained"
    assert coverage["assessment_basis"] == "unassessed_fallback" and coverage["model_selected"] is False
    assert coverage["verified_sufficiency"] is False and coverage["assessment_status"] == "uncertain"
    assert len(assessor.calls) == 1


async def test_fallback_discards_source_deleted_when_assessment_fails(memory):
    from scone_memory.retrieval.adaptive import EvidenceAssessmentError

    deleted = await memory.remember("alpha", "Juniper calibration obsolete source.")
    retained = await memory.remember("alpha", "Juniper calibration retained independent source.")

    class DeleteThenFail:
        async def assess(self, question, candidates):
            await memory.forget("alpha", deleted.episode_id)
            raise EvidenceAssessmentError("assessment_provider_failed")

    adaptive = AdaptiveRetriever(memory, DeleteThenFail(), limits=AdaptiveLimits(timeout_s=1.0))
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive).prepare([
        {"role": "user", "content": "Juniper calibration?"}])
    assert {source["episode_id"] for source in payload(request)["sources"]} == {retained.episode_id}
    assert receipt["adaptive_fallback_status"] == "retained" and receipt["adaptive_status"] == "uncertain"
    assert "stale_evidence" in receipt["adaptive_reasons"]
    assert "obsolete" not in json.dumps(payload(request))


async def test_fallback_preserves_known_atomic_group_under_context_byte_limit(memory):
    from scone_memory.retrieval.adaptive import EvidenceAssessmentError

    independent = await memory.remember("alpha", "Juniper calibration independent scheduling note.")
    left = await memory.remember("alpha", "Juniper calibration grouped left: " + "Meridian reference details. " * 13)
    right = await memory.remember("alpha", "Juniper calibration grouped right: " + "Celeste maintains reference. " * 13)

    def decide(round_number, candidates):
        if round_number > 1:
            raise EvidenceAssessmentError("assessment_provider_failed")
        ids = {candidate.episode_id: candidate.id for candidate in candidates if candidate.id.startswith("chunk:")}
        return EvidenceDecision(status="insufficient",
            selected_ids=(ids[independent.episode_id], ids[left.episode_id], ids[right.episode_id]),
            selected_groups=((ids[left.episode_id], ids[right.episode_id]),),
            followup_queries=("Juniper calibration scheduling reference",))

    adaptive = retriever(memory, ScriptedAssessor(decide))
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive,
                                           max_context_bytes=1900).prepare([
        {"role": "user", "content": "Juniper calibration?"}])
    data = payload(request)
    assert {source["episode_id"] for source in data["sources"]} == {independent.episode_id}
    assert receipt["adaptive_fallback_status"] == "retained" and receipt["adaptive_status"] == "uncertain"
    assert receipt["adaptive_atomic_group_omitted_count"] == 1
    assert data["coverage"]["adaptive"]["model_selected"] is False
    assert data["coverage"]["adaptive"]["selected_omitted_count"] == 2
    assert receipt["context_bytes"] <= 1900


@pytest.mark.parametrize("assessment_status", ["insufficient", "uncertain"])
async def test_terminal_empty_selection_prepares_scoped_partial_candidates_without_assessment_claim(memory, assessment_status):
    partial = await memory.remember("alpha", "Juniper calibration depends on Meridian. Its maintainer is not recorded.",
                                    metadata={"team": "science"})
    await memory.remember("alpha", "Juniper calibration private unrelated department record.", metadata={"team": "legal"})
    assessor = ScriptedAssessor(lambda _, candidates: EvidenceDecision(status=assessment_status, selected_ids=()))
    adaptive = retriever(memory, assessor)
    messages = [{"role": "user", "content": "Who maintains Juniper calibration?"}]
    request, receipt = await MemoryContext(memory, "alpha", "current", where={"team": "science"},
                                           adaptive_retriever=adaptive).prepare(messages)
    assert receipt["status"] == "prepared" and receipt["adaptive_status"] == assessment_status
    assert {ref["episode_id"] for ref in receipt["references"]} == {partial.episode_id}
    assert receipt["adaptive_evidence_basis"] == "unselected_candidates"
    assert receipt["adaptive_fallback_status"] == "not_used" and receipt["adaptive_errors"] == []
    assert "empty_selection_retained" in receipt["adaptive_reasons"]
    data = payload(request)
    coverage = data["coverage"]["adaptive"]
    assert coverage["assessment_basis"] == "unselected_candidates" and coverage["evidence_basis"] == "unselected_candidates"
    assert coverage["assessment_status"] == assessment_status and coverage["model_selected"] is False
    assert coverage["verified_sufficiency"] is False and coverage["fallback_status"] == "not_used"
    assert coverage["selection_complete"] is True and data["coverage"]["complete_history"] is False
    assert "private unrelated" not in json.dumps(data)
    assert len(assessor.calls) == 1 and request[-1] == messages[-1]


async def test_explicit_empty_selection_policy_preserves_empty_native_context(memory):
    await memory.remember("alpha", "Juniper calibration depends on Meridian.")
    assessor = ScriptedAssessor(lambda _, candidates: EvidenceDecision(status="insufficient", selected_ids=()))
    adaptive = AdaptiveRetriever(memory, assessor, limits=AdaptiveLimits(timeout_s=1.0), empty_selection_policy="empty")
    messages = [{"role": "user", "content": "Juniper calibration?"}]
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive).prepare(messages)
    assert request == messages and receipt["status"] == "empty"
    assert receipt["adaptive_status"] == "insufficient" and receipt["adaptive_evidence_basis"] == "none"
    assert receipt["references"] == [] and receipt["adaptive_errors"] == []
    assert receipt["adaptive_fallback_status"] == "not_used"
    assert "empty_selection_retained" not in receipt["adaptive_reasons"]


@pytest.mark.parametrize("evidence_policy", ["model_selected", "original_and_selected"])
async def test_original_query_blend_preserves_partial_evidence_with_scoped_provenance(memory, evidence_policy):
    partial = await memory.remember("alpha", "Juniper calibration depends on Meridian; its maintainer is unknown.",
                                    metadata={"team": "science"})
    wrong = await memory.remember("alpha", "Juniper calibration cafeteria serves pastries.", metadata={"team": "science"})
    await memory.remember("alpha", "Juniper calibration PRIVATE source.", metadata={"team": "legal"})

    def decide(_, candidates):
        assert all("PRIVATE" not in candidate.text for candidate in candidates)
        return EvidenceDecision(status="sufficient", selected_ids=tuple(
            candidate.id for candidate in candidates if candidate.episode_id == wrong.episode_id))

    assessor = ScriptedAssessor(decide)
    adaptive = AdaptiveRetriever(memory, assessor, limits=AdaptiveLimits(timeout_s=1.0), evidence_policy=evidence_policy)
    request, receipt = await MemoryContext(memory, "alpha", "current", where={"team": "science"},
                                           adaptive_retriever=adaptive).prepare([
        {"role": "user", "content": "Who maintains Juniper calibration?"}])
    assert receipt["status"] == "prepared", receipt
    data = payload(request)
    expected_episodes = {wrong.episode_id, partial.episode_id} if evidence_policy == "original_and_selected" else {wrong.episode_id}
    assert {source["episode_id"] for source in data["sources"]} == expected_episodes
    coverage = data["coverage"]["adaptive"]
    assert coverage["verified_sufficiency"] is False and coverage["fallback_status"] == "not_used"
    assert coverage["assessment_status"] == "sufficient" and receipt["adaptive_errors"] == []
    assert "PRIVATE" not in json.dumps(data) and len(assessor.calls) == 1
    if evidence_policy == "original_and_selected":
        supplied = {f"chunk:{source['chunk_id']}" for source in data["sources"]}
        selected = {candidate.id for candidate in assessor.calls[0][1] if candidate.episode_id == wrong.episode_id}
        assert coverage["assessment_basis"] == "model_judgment_over_selection"
        assert coverage["evidence_basis"] == "original_and_selected" and coverage["model_selected"] is False
        assert set(coverage["original_query_ids"]) == supplied
        assert set(coverage["model_selected_ids"]) == selected
        assert "original_query_blended" in receipt["adaptive_reasons"]
    else:
        assert coverage["model_selected"] is True and coverage["assessment_basis"] == "model_judgment"


async def test_blended_provenance_excludes_records_omitted_by_context_budget(memory):
    partial = await memory.remember("alpha", "Juniper calibration depends on Meridian.")
    wrong = await memory.remember("alpha", "Juniper calibration cafeteria: " + "Pastries served daily. " * 60)
    assessor = ScriptedAssessor(lambda _, candidates: EvidenceDecision(status="sufficient", selected_ids=tuple(
        candidate.id for candidate in candidates if candidate.episode_id == wrong.episode_id)))
    adaptive = AdaptiveRetriever(memory, assessor, limits=AdaptiveLimits(timeout_s=1.0), evidence_policy="original_and_selected")
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive,
                                           max_context_bytes=1700).prepare([
        {"role": "user", "content": "Who maintains Juniper calibration?"}])
    assert receipt["status"] == "prepared", receipt
    data = payload(request)
    assert {source["episode_id"] for source in data["sources"]} == {partial.episode_id}
    coverage = data["coverage"]["adaptive"]
    supplied = {f"chunk:{source['chunk_id']}" for source in data["sources"]}
    assert set(coverage["original_query_ids"]) == supplied and coverage["model_selected_ids"] == []
    assert coverage["selection_complete"] is False and coverage["selected_omitted_count"] > 0
    assert coverage["model_selected"] is False and coverage["verified_sufficiency"] is False
    assert receipt["context_bytes"] <= 1700


async def test_blended_provenance_excludes_source_deleted_after_adaptive_return(memory):
    deleted = await memory.remember("alpha", "Juniper calibration obsolete original source.")
    retained = await memory.remember("alpha", "Juniper calibration cafeteria serves pastries.")
    assessor = ScriptedAssessor(lambda _, candidates: EvidenceDecision(status="sufficient", selected_ids=tuple(
        candidate.id for candidate in candidates if candidate.episode_id == retained.episode_id)))

    class DeleteOriginalAfterRetrieval(AdaptiveRetriever):
        async def retrieve(self, *args, **kwargs):
            result = await super().retrieve(*args, **kwargs)
            await memory.forget("alpha", deleted.episode_id)
            return result

    adaptive = DeleteOriginalAfterRetrieval(memory, assessor, limits=AdaptiveLimits(timeout_s=1.0),
                                            evidence_policy="original_and_selected")
    request, receipt = await MemoryContext(memory, "alpha", "current", adaptive_retriever=adaptive).prepare([
        {"role": "user", "content": "Who maintains Juniper calibration?"}])
    assert receipt["status"] == "prepared", receipt
    data = payload(request)
    assert {source["episode_id"] for source in data["sources"]} == {retained.episode_id}
    supplied = {f"chunk:{source['chunk_id']}" for source in data["sources"]}
    coverage = data["coverage"]["adaptive"]
    assert set(coverage["original_query_ids"]) == set(coverage["model_selected_ids"]) == supplied
    assert coverage["selection_complete"] is False and coverage["selected_omitted_count"] == 1
    assert "obsolete original" not in json.dumps(data)
