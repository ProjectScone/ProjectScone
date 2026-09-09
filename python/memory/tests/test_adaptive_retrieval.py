"""Offline adaptive assessment exercises disclosure, bounds and retention."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends.sqlite import SqliteDocumentStore
from scone_memory.core.models import Episode, Fact, RecallResult
from scone_memory.core.ports import NewFact
from scone_memory.retrieval.adaptive import (AdaptiveLimits, AdaptiveRetriever, EvidenceCandidate,
    EvidenceDecision, EvidenceAssessmentError, evidence_payload_bytes)
from scone_memory.retrieval.recall_scope import RecallScope

STAMP = "2025-01-01T00:00:00Z"
AssessorFn = Callable[[str, tuple[EvidenceCandidate, ...]], Awaitable[EvidenceDecision]]


@pytest.mark.parametrize('constraint', ['candidates', 'bytes'])
async def test_followup_queries_share_free_candidate_capacity(memory, monkeypatch, constraint):
    bridge = await fact(memory, 'root', 'uses', 'branch')
    noise = [await fact(memory, f'noise-{i}', 'uses', f'value-{i}') for i in range(4)]
    if constraint == 'bytes':
        quote = 'Noise uses Value. ' * 40
        episode = await memory.remember('alpha',quote)
        noise = [await memory.documents.insert_fact(NewFact(space='alpha',subject='Noise',predicate='uses',
            object='Value',quote=quote,source_episode_id=episode.episode_id,valid_from=STAMP))]
    answer = await fact(memory, 'branch', 'uses', 'archive')
    calls = []

    async def recall(space, query, **kwargs):
        calls.append(query)
        return RecallResult(facts=[bridge] if query == 'question' else noise if query == 'first branch' else [answer])

    async def assess(question, candidates):
        ids = tuple(row.id for row in candidates)
        if len(calls) == 1:
            return EvidenceDecision(status='insufficient', selected_ids=ids,
                                    followup_queries=('first branch','second branch'))
        assert f'fact:{answer.fact_id}' in ids
        assert len(ids) <= 4
        if constraint == 'bytes':
            assert evidence_payload_bytes(candidates) <= 1024
        return EvidenceDecision(status='sufficient', selected_ids=ids)

    monkeypatch.setattr(memory,'recall',recall)
    limits = AdaptiveLimits(candidate_limit=4) if constraint == 'candidates' else AdaptiveLimits(max_evidence_bytes=1024)
    result = await AdaptiveRetriever(memory,Assessor(assess),limits=limits).retrieve(
        'alpha','question',scope=RecallScope.validated())
    assert result.status == 'sufficient'
    assert calls == ['question','first branch','second branch']
    assert 'query_evidence_share' in result.reasons


@pytest.mark.parametrize("status", ["insufficient", "uncertain"])
@pytest.mark.parametrize("policy", ["retain_verified", "empty"])
async def test_terminal_empty_selection_preserves_partial_evidence_without_claiming_selection(memory, monkeypatch, status, policy):
    route = await fact(memory, "Spruce", "forwards to", "Birch")
    async def recall(*args, **kwargs):
        return RecallResult(facts=[route])
    async def assess(question, candidates):
        return EvidenceDecision(status=status, selected_ids=())
    monkeypatch.setattr(memory, "recall", recall)
    result = await AdaptiveRetriever(memory, Assessor(assess), empty_selection_policy=policy).retrieve(
        "alpha", "Where do Spruce records ultimately finish?", scope=RecallScope.validated())
    assert result.status == status and result.rounds[-1].selected_count == 0
    assert result.fallback_status == "not_used" and result.errors == ()
    assert result.recall.facts == ([route] if policy == "retain_verified" else [])
    assert result.evidence_basis == ("unselected_candidates" if policy == "retain_verified" else "none")
    assert ("empty_selection_retained" in result.reasons) is (policy == "retain_verified")


async def test_empty_selection_keeps_only_final_pool_after_followup(memory, monkeypatch):
    old = await fact(memory, "Old", "routes to", "Distractor")
    final = await fact(memory, "Spruce", "forwards to", "Birch")
    queries = []
    async def recall(space, query, **kwargs):
        queries.append(query)
        return RecallResult(facts=[old] if len(queries) == 1 else [final])
    async def assess(question, candidates):
        return EvidenceDecision(status="insufficient", selected_ids=(),
            followup_queries=("Spruce route",) if len(queries) == 1 else ())
    monkeypatch.setattr(memory, "recall", recall)
    result = await AdaptiveRetriever(memory, Assessor(assess)).retrieve("alpha", "Find destination", scope=RecallScope.validated())
    assert result.recall.facts == [final] and len(result.rounds) == 2
    assert result.evidence_basis == "unselected_candidates"


@pytest.mark.parametrize("selected", [False, True])
async def test_empty_selection_never_restores_deleted_or_pruned_selected_records(memory, monkeypatch, selected):
    route = await fact(memory, "Spruce", "forwards to", "Birch")
    unrelated = await fact(memory, "Other", "uses", "Disk")
    async def recall(*args, **kwargs):
        return RecallResult(facts=[route, unrelated] if selected else [route])
    async def assess(question, candidates):
        await memory.forget("alpha", route.source_episode_id)
        return EvidenceDecision(status="insufficient", selected_ids=(f"fact:{route.fact_id}",) if selected else ())
    monkeypatch.setattr(memory, "recall", recall)
    result = await AdaptiveRetriever(memory, Assessor(assess)).retrieve("alpha", "Where does Spruce finish?", scope=RecallScope.validated())
    assert result.recall.facts == [] and result.evidence_basis == "none"
    assert "empty_selection_retained" not in result.reasons


@pytest.mark.parametrize("policy", [None, True, "anything", 1])
def test_empty_selection_policy_validates_before_retrieval(policy):
    with pytest.raises(ValueError, match="empty_selection_policy"):
        AdaptiveRetriever(None, None, empty_selection_policy=policy)


class Assessor:
    def __init__(self, callback: AssessorFn) -> None:
        self.callback = callback
        self.calls: list[tuple[EvidenceCandidate, ...]] = []

    async def assess(self, question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        self.calls.append(candidates)
        return await self.callback(question, candidates)


async def sufficient(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
    return EvidenceDecision(status="sufficient", selected_ids=tuple(c.id for c in candidates))


@pytest.fixture(params=["memory", "sqlite"])
async def memory(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[MemoryEngine]:
    store = InMemoryDocumentStore() if request.param == "memory" else SqliteDocumentStore(tmp_path / "adaptive.db")
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder()).open()
    yield engine
    await engine.close()


async def fact(memory: MemoryEngine, subject: str, predicate: str, obj: str, *, space: str = "alpha",
               metadata: dict[str, str] | None = None, source: str | None = None,
               at: str = STAMP, kind: str = "note") -> Fact:
    quote = f"{subject} {predicate} {obj}."
    episode = await memory.remember(space, quote, created_at=at, metadata=metadata, source=source, kind=kind)
    result = await memory.documents.insert_fact(NewFact(space=space, subject=subject, predicate=predicate,
        object=obj, valid_from=at, source_episode_id=episode.episode_id, quote=quote))
    assert isinstance(result, Fact)
    return result


async def test_followup_bridges_missing_fact_and_discards_distractors(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = await fact(memory, "Aster", "depends on", "Beacon")
    answer = await fact(memory, "Beacon", "owned by", "Cedar")
    distractor = await fact(memory, "Meteor", "owned by", "Cloud")
    calls: list[str] = []

    async def recall(space: str, query: str, **kwargs: object) -> RecallResult:
        calls.append(query)
        return RecallResult(facts=[bridge, distractor] if len(calls) == 1 else [answer])

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        assert question == "Who owns Aster's dependency?"
        if len(calls) == 1:
            return EvidenceDecision(status="insufficient", selected_ids=(f"fact:{bridge.fact_id}",),
                                    followup_queries=("Who owns Beacon?",))
        assert {c.id for c in candidates} == {f"fact:{bridge.fact_id}", f"fact:{answer.fact_id}"}
        return await sufficient(question, candidates)

    monkeypatch.setattr(memory, "recall", recall)
    monkeypatch.setattr(memory.documents, "list_facts", AsyncMock(side_effect=AssertionError("no scan")))
    result = await AdaptiveRetriever(memory, Assessor(assess)).retrieve("alpha", "Who owns Aster's dependency?",
                                                                      scope=RecallScope.validated())
    assert result.status == "sufficient"
    assert result.recall.facts == [bridge, answer]
    assert len(result.rounds) == 2 and result.queries_used == 2
    assert "Beacon" not in str(result.rounds)


async def test_actual_engine_recall_selected_chunks_only(memory: MemoryEngine) -> None:
    await memory.remember("alpha", "Aster depends on Beacon.", created_at=STAMP)
    await memory.remember("alpha", "Aster has a distracting blue logo.", created_at=STAMP)

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        chosen = next(candidate.id for candidate in candidates if "depends on" in candidate.text)
        return EvidenceDecision(status="sufficient", selected_ids=(chosen,))

    result = await AdaptiveRetriever(memory, Assessor(assess)).retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert result.status == "sufficient"
    assert [item.text for item in result.recall.items] == ["Aster depends on Beacon."]


async def test_explicit_empty_selection_policy_without_followups_abstains(memory: MemoryEngine) -> None:
    await memory.remember("alpha", "Aster depends on Beacon.", created_at=STAMP)

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        return EvidenceDecision(status="insufficient", selected_ids=())

    result = await AdaptiveRetriever(memory, Assessor(assess), empty_selection_policy="empty").retrieve(
        "alpha", "Aster", scope=RecallScope.validated())
    assert result.status == "insufficient" and not result.recall.items and not result.recall.facts
    assert "no_followup_queries" in result.reasons


@pytest.mark.parametrize("invalid", ["unknown", "duplicate", "coercion", "scope", "empty"])
async def test_malicious_or_invalid_decisions_do_not_control_retrieval(memory: MemoryEngine, invalid: str) -> None:
    await memory.remember("alpha", "Aster depends on Beacon.", created_at=STAMP)

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        data: dict[str, object] = {"status": "sufficient", "selected_ids": (candidates[0].id,)}
        if invalid == "unknown":
            data["selected_ids"] = ("fact:999999",)
        elif invalid == "duplicate":
            data["selected_ids"] = (candidates[0].id, candidates[0].id)
        elif invalid == "coercion":
            data["selected_ids"] = [candidates[0].id]
        elif invalid == "scope":
            data["scope"] = {"space": "other"}
        else:
            data["selected_ids"] = ()
        # Bypass model validation as a hostile in-process adapter might.
        decision = EvidenceDecision.model_construct(_fields_set=set(), **data)
        return decision.model_copy(update={"scope": data["scope"]}) if invalid == "scope" else decision

    result = await AdaptiveRetriever(memory, Assessor(assess), failure_policy="empty").retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert result.status == "uncertain" and not result.recall.items and not result.recall.facts
    assert result.errors == ("invalid_or_failed_assessment",)
    assert result.queries_used == 1


async def test_loop_normalization_and_global_query_cap(memory: MemoryEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    item = await fact(memory, "Aster", "depends on", "Beacon")
    calls: list[str] = []

    async def recall(space: str, query: str, **kwargs: object) -> RecallResult:
        calls.append(query)
        return RecallResult(facts=[item])

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        return EvidenceDecision(status="insufficient", selected_ids=(candidates[0].id,),
            followup_queries=("  ASTER  ", "new query", " NEW   QUERY ", "another query", "over budget"))

    monkeypatch.setattr(memory, "recall", recall)
    result = await AdaptiveRetriever(memory, Assessor(assess), limits=AdaptiveLimits(max_queries=3)).retrieve(
        "alpha", "Aster", scope=RecallScope.validated())
    assert calls == ["Aster", "new query", "another query"]
    assert result.queries_used == 3 and result.truncated
    assert {"max_queries", "duplicate_queries", "no_new_queries"}.issubset(result.reasons)
    assert result.status == "insufficient"


async def test_round_cap(memory: MemoryEngine) -> None:
    await memory.remember("alpha", "Aster depends on Beacon.", created_at=STAMP)

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        return EvidenceDecision(status="uncertain", selected_ids=(candidates[0].id,), followup_queries=("Beacon",))

    result = await AdaptiveRetriever(memory, Assessor(assess), limits=AdaptiveLimits(max_rounds=1)).retrieve(
        "alpha", "Aster", scope=RecallScope.validated())
    assert result.queries_used == 1 and result.truncated and "max_rounds" in result.reasons


@pytest.mark.parametrize("hidden", ["space", "metadata", "session", "source_session", "prefix", "since", "until", "kind"])
async def test_scope_and_exclusion_before_model_on_every_round(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch, hidden: str) -> None:
    scope = RecallScope.validated(where={"team": "blue"}, source_prefix="public/", kind="note",
                                since="2024-12-01", until="2025-02-01")
    good = await fact(memory, "Aster", "depends on", "Beacon", metadata={"team": "blue"}, source="public/a")
    metadata = {"team": "red" if hidden == "metadata" else "blue"}
    if hidden == "session":
        metadata["session_id"] = "public/current"
    bad = await fact(memory, "Beacon", "secret", "hidden", space="beta" if hidden == "space" else "alpha",
        metadata=metadata, source="private/a" if hidden == "prefix" else "public/current" if hidden == "source_session" else "public/b",
        at="2024-01-01T00:00:00Z" if hidden == "since" else "2025-03-01T00:00:00Z" if hidden == "until" else STAMP,
        kind="conversation" if hidden == "kind" else "note")
    calls = 0

    async def recall(space: str, query: str, **kwargs: object) -> RecallResult:
        nonlocal calls
        calls += 1
        assert kwargs["where"] == {"team": "blue"}
        where = kwargs["where"]
        assert isinstance(where, dict)
        where["team"] = "red"  # Mutation of one request cannot broaden the next.
        return RecallResult(facts=[good, bad])

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        assert [c.id for c in candidates] == [f"fact:{good.fact_id}"]
        return EvidenceDecision(status="sufficient" if calls == 2 else "insufficient",
            selected_ids=(candidates[0].id,), followup_queries=() if calls == 2 else ("Beacon",))

    monkeypatch.setattr(memory, "recall", recall)
    result = await AdaptiveRetriever(memory, Assessor(assess)).retrieve("alpha", "Aster", scope=scope,
                                                                    exclude_session_id="public/current")
    assert result.status == "sufficient" and result.recall.facts == [good]
    assert scope.where == (("team", "blue"),)
    assert "filtered_evidence" in result.reasons


@pytest.mark.parametrize("change", ["delete", "content", "exclude_fact"])
async def test_mutation_during_assessor_await_invalidates_sources(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    source_fact = await fact(memory, "Aster", "depends on", "Beacon")
    assert source_fact.source_episode_id is not None
    episode_id = source_fact.source_episode_id
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=[source_fact])))

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        await asyncio.sleep(0)
        if change == "delete":
            await memory.forget("alpha", episode_id)
        elif change == "exclude_fact":
            await memory.documents.update_fact(source_fact.model_copy(update={"excluded_reason": "removed"}))
        else:
            original_get = memory.documents.get_episode

            async def changed(space: str, target: int) -> Episode | None:
                episode = await original_get(space, target)
                return episode.model_copy(update={"content": "changed source"}) if episode is not None else None

            monkeypatch.setattr(memory.documents, "get_episode", changed)
        return await sufficient(question, candidates)

    result = await AdaptiveRetriever(memory, Assessor(assess)).retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert result.status == "uncertain" and result.recall.facts == []
    assert "stale_evidence" in result.reasons


async def test_serialized_byte_budget_drops_whole_evidence(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch) -> None:
    big = await fact(memory, "Aster", "described as", "é" * 400)
    small = await fact(memory, "Beacon", "is", "tiny")
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=[big, small])))
    assessor = Assessor(sufficient)
    result = await AdaptiveRetriever(memory, assessor, limits=AdaptiveLimits(max_evidence_bytes=200)).retrieve(
        "alpha", "Aster", scope=RecallScope.validated())
    assert result.recall.facts == [small]
    assert all(evidence_payload_bytes(call) <= 200 for call in assessor.calls)
    assert result.truncated and "max_evidence_bytes" in result.reasons
    assert assessor.calls[0][0].text == small.quote


async def test_timeout_discards_evidence_and_cancellation_propagates(memory: MemoryEngine) -> None:
    await memory.remember("alpha", "Aster depends on Beacon.", created_at=STAMP)
    entered = asyncio.Event()

    async def blocked(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        entered.set()
        await asyncio.Event().wait()
        return await sufficient(question, candidates)

    retriever = AdaptiveRetriever(memory, Assessor(blocked), limits=AdaptiveLimits(timeout_s=1.0), failure_policy="empty")
    result = await retriever.retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert result.status == "uncertain" and result.errors == ("timeout",)
    assert result.recall.items == [] and result.truncated
    entered.clear()
    task = asyncio.create_task(retriever.retrieve("alpha", "Aster", scope=RecallScope.validated()))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("values", [{"max_rounds": 5}, {"max_queries": 13}, {"candidate_limit": 101},
    {"max_evidence_bytes": 128001}, {"timeout_s": float("inf")}, {"max_rounds": "2"}, {"timeout_s": True}])
def test_limits_are_strict_hard_caps(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AdaptiveLimits.model_validate(values)


async def test_full_chunk_window_still_supplies_fact_lane(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch) -> None:
    known = await fact(memory, "Aster", "depends on", "Beacon")
    await memory.remember("alpha", "Aster has a blue logo.", created_at=STAMP)
    baseline = await memory.recall("alpha", "Aster", limit=2)
    assert len(baseline.items) == 2
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(items=baseline.items, facts=[known])))
    assessor = Assessor(sufficient)
    result = await AdaptiveRetriever(memory, assessor, limits=AdaptiveLimits(candidate_limit=2)).retrieve(
        "alpha", "Aster", scope=RecallScope.validated())
    assert len(assessor.calls[0]) == 2
    assert {candidate.id.split(":")[0] for candidate in assessor.calls[0]} == {"chunk", "fact"}
    assert result.recall.facts == [known]
    assert result.truncated


@pytest.mark.parametrize("invalid", ["negative_span", "too_long_span", "unicode_span", "changed_text", "wrong_space"])
async def test_chunks_require_current_exact_utf8_spans(memory: MemoryEngine, monkeypatch: pytest.MonkeyPatch,
        invalid: str) -> None:
    from scone_memory.core.models import Chunk

    await memory.remember("alpha", "Aster café depends on Beacon.", created_at=STAMP)
    baseline = await memory.recall("alpha", "Aster")
    assert baseline.items
    item = baseline.items[0]
    original = (await memory.documents.get_chunks("alpha", [item.chunk_id]))[0]
    update: dict[str, object]
    if invalid == "negative_span":
        update = {"start": -1}
    elif invalid == "too_long_span":
        update = {"end": original.end + 1}
    elif invalid == "unicode_span":
        update = {"end": len(original.text)}
    elif invalid == "wrong_space":
        update = {"space": "beta"}
    else:
        update = {"text": "invented text"}
    corrupt = original.model_copy(update=update)
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=baseline))

    async def chunks(space: str, ids: list[int]) -> list[Chunk]:
        return [corrupt]

    monkeypatch.setattr(memory.documents, "get_chunks", chunks)
    assessor = Assessor(sufficient)
    result = await AdaptiveRetriever(memory, assessor).retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert result.status == "insufficient" and not result.recall.items
    assert assessor.calls == [] and "filtered_evidence" in result.reasons


async def test_chunk_deletion_during_assessment_and_unrelated_writes(memory: MemoryEngine) -> None:
    source = await memory.remember("alpha", "Aster depends on Beacon.", created_at=STAMP)

    async def unrelated(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        await memory.remember("alpha", "Unrelated tool capture", created_at=STAMP)
        return await sufficient(question, candidates)

    okay = await AdaptiveRetriever(memory, Assessor(unrelated)).retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert okay.status == "sufficient" and okay.recall.items

    async def delete(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        await memory.forget("alpha", source.episode_id)
        chosen = next(c.id for c in candidates if "Aster" in c.text)
        return EvidenceDecision(status="sufficient", selected_ids=(chosen,))

    deleted = await AdaptiveRetriever(memory, Assessor(delete)).retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert deleted.status == "uncertain" and deleted.recall.items == []


async def test_revision_change_during_verification_conservatively_discards_snapshot(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch) -> None:
    await memory.remember("alpha", "Aster depends on Beacon.", created_at=STAMP)
    baseline = await memory.recall("alpha", "Aster")
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=baseline))
    revision = memory.documents.revision
    calls = 0

    async def changing_revision(space: str) -> int:
        nonlocal calls
        calls += 1
        value = await revision(space)
        assert isinstance(value, int)
        return value + calls

    monkeypatch.setattr(memory.documents, "revision", changing_revision)
    assessor = Assessor(sufficient)
    result = await AdaptiveRetriever(memory, assessor).retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert not result.recall.items and not assessor.calls
    assert "stale_evidence" in result.reasons


@pytest.mark.parametrize("change", [{"status": "closed"}, {"status": "proposed"},
    {"status": "declined"}, {"quote": "unsupported"}, {"source_episode_id": None},
    {"valid_from": "2099-01-01T00:00:00Z"}])
async def test_only_current_quoted_facts_can_reach_assessor(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch, change: dict[str, object]) -> None:
    stored = await fact(memory, "Aster", "depends on", "Beacon")
    invalid = stored.model_copy(update=change)
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=[invalid])))
    monkeypatch.setattr(memory.documents, "get_fact", AsyncMock(return_value=invalid))
    assessor = Assessor(sufficient)
    result = await AdaptiveRetriever(memory, assessor).retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert result.recall.facts == [] and assessor.calls == []


async def test_exception_text_never_enters_receipts(memory: MemoryEngine) -> None:
    await memory.remember("alpha", "Aster depends on Beacon.", created_at=STAMP)

    async def failing(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        raise RuntimeError("secret token and entire prompt")

    result = await AdaptiveRetriever(memory, Assessor(failing), failure_policy="empty").retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert result.status == "uncertain" and not result.recall.items
    assert "secret" not in result.model_dump_json() and "Aster" not in result.model_dump_json()


async def test_late_return_after_swallowed_cancellation_is_discarded(memory: MemoryEngine) -> None:
    await memory.remember("alpha", "Aster depends on Beacon.", created_at=STAMP)

    async def late(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return await sufficient(question, candidates)
        return await sufficient(question, candidates)

    result = await AdaptiveRetriever(memory, Assessor(late), limits=AdaptiveLimits(timeout_s=1.0), failure_policy="empty").retrieve(
        "alpha", "Aster", scope=RecallScope.validated())
    assert result.status == "uncertain" and not result.recall.items
    assert result.errors == ("timeout",)


def test_sufficient_cannot_request_more_queries_and_decision_fields_are_bounded() -> None:
    with pytest.raises(ValidationError):
        EvidenceDecision(status="sufficient", selected_ids=("fact:1",), followup_queries=("another",))
    with pytest.raises(ValidationError):
        EvidenceDecision(status="uncertain", selected_ids=(), followup_queries=tuple(str(i) for i in range(13)))
    with pytest.raises(ValidationError):
        EvidenceCandidate(id="chunk:1", episode_id=0, text="text")


@pytest.mark.parametrize("reason", ["assessment_timeout", "assessment_provider_failed", "invalid_assessment", "private error"])
async def test_assessment_failure_receipt_preserves_only_safe_codes(memory: MemoryEngine, reason: str) -> None:
    await memory.remember("alpha", "Aster depends on Beacon.", created_at=STAMP)

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        error = EvidenceAssessmentError(reason)
        error.reason = reason  # A caller-owned adapter may bypass the constructor's sanitization.
        raise error

    result = await AdaptiveRetriever(memory, Assessor(assess), failure_policy="empty").retrieve("alpha", "Aster", scope=RecallScope.validated())
    expected = "invalid_or_failed_assessment" if reason == "private error" else reason
    assert result.errors == (expected,)
    assert result.status == "uncertain" and not result.recall.items and not result.recall.facts
    assert "private error" not in result.model_dump_json()


def test_selected_groups_default_to_empty() -> None:
    from scone_memory.retrieval.adaptive import AdaptiveResult

    assert EvidenceDecision(status="insufficient", selected_ids=()).selected_groups == ()
    assert AdaptiveResult().selected_groups == ()


@pytest.mark.parametrize("groups", [
    (("fact:1",),),
    (("fact:1", "fact:1"),),
    (("fact:1", "fact:2"), ("fact:2", "fact:3")),
    (("fact:1", "fact:999"),),
    [["fact:1", "fact:2"]],
    (["fact:1", "fact:2"],),
    tuple((f"fact:{i * 2 + 1}", f"fact:{i * 2 + 2}") for i in range(101)),
])
def test_atomic_group_schema_rejects_invalid_membership_or_coercion(groups: object) -> None:
    with pytest.raises(ValidationError):
        EvidenceDecision.model_validate({"status": "insufficient", "selected_ids": ("fact:1", "fact:2", "fact:3"),
                                         "selected_groups": groups})


@pytest.mark.parametrize("stage", ["assessment", "final_verification"])
@pytest.mark.parametrize("change", ["delete", "content"])
async def test_atomic_groups_omit_every_member_after_retention_change(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch, stage: str, change: str) -> None:
    grouped_first = await fact(memory, "Aster", "depends on", "Beacon")
    grouped_second = await fact(memory, "Beacon", "owned by", "Cedar")
    independent = await fact(memory, "Aster", "color", "blue")
    source_id = grouped_second.source_episode_id
    assert source_id is not None
    initial = RecallResult(facts=[grouped_first, grouped_second, independent])
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=initial))
    original_get = memory.documents.get_episode
    altered = False
    post_assessment_reads = 0

    async def mutate() -> None:
        nonlocal altered
        if change == "delete":
            await memory.forget("alpha", source_id)
        altered = True

    async def current_source(space: str, episode_id: int) -> Episode | None:
        episode = await original_get(space, episode_id)
        if not isinstance(episode, Episode):
            return None
        if altered and change == "content" and episode_id == source_id:
            return episode.model_copy(update={"content": "changed source"})
        return episode

    monkeypatch.setattr(memory.documents, "get_episode", current_source)

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        if stage == "assessment":
            await mutate()
        else:
            async def final_source(space: str, episode_id: int) -> Episode | None:
                nonlocal post_assessment_reads
                if episode_id == source_id:
                    post_assessment_reads += 1
                    # First is the post-model pass, second is the final pass.
                    if post_assessment_reads == 2:
                        await mutate()
                return await current_source(space, episode_id)

            monkeypatch.setattr(memory.documents, "get_episode", final_source)
        return EvidenceDecision(status="sufficient", selected_ids=tuple(c.id for c in candidates),
            selected_groups=((f"fact:{grouped_first.fact_id}", f"fact:{grouped_second.fact_id}"),))

    result = await AdaptiveRetriever(memory, Assessor(assess)).retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert result.status == "uncertain"
    assert result.selected_groups == ()
    assert not {grouped_first.fact_id, grouped_second.fact_id} & {item.fact_id for item in result.recall.facts}
    # Engine deletion during verification invalidates the whole revision snapshot;
    # a content-only point-read replacement leaves independent evidence usable.
    if change == "content" or stage == "assessment":
        assert result.recall.facts == [independent]
    assert "atomic_group_omitted" in result.reasons


@pytest.mark.parametrize("invalid", ["unknown", "outside_selected", "duplicate", "overlap", "list"])
async def test_unvalidated_atomic_groups_fail_closed(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch, invalid: str) -> None:
    records = [await fact(memory, "Aster", "depends on", "Beacon"),
               await fact(memory, "Beacon", "owned by", "Cedar"),
               await fact(memory, "Cedar", "located in", "Denver")]
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=records)))

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        ids = tuple(c.id for c in candidates)
        groups: object = ((ids[0], "fact:99999"),)
        selected = ids
        if invalid == "outside_selected":
            selected, groups = ids[:2], ((ids[0], ids[2]),)
        elif invalid == "duplicate":
            groups = ((ids[0], ids[0]),)
        elif invalid == "overlap":
            groups = ((ids[0], ids[1]), (ids[1], ids[2]))
        elif invalid == "list":
            groups = ([ids[0], ids[1]],)
        return EvidenceDecision.model_construct(status="sufficient", selected_ids=selected,
                                                selected_groups=groups, followup_queries=())

    result = await AdaptiveRetriever(memory, Assessor(assess), failure_policy="empty").retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert result.status == "uncertain" and result.recall.facts == []
    assert result.selected_groups == () and result.errors == ("invalid_or_failed_assessment",)


async def test_complete_atomic_groups_survive_and_last_decision_controls_delivery(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch) -> None:
    records = [await fact(memory, "Aster", "depends on", "Beacon"),
               await fact(memory, "Beacon", "owned by", "Cedar"),
               await fact(memory, "Cedar", "located in", "Denver")]
    calls = 0

    async def recall(space: str, query: str, **kwargs: object) -> RecallResult:
        return RecallResult(facts=records[:2] if calls == 0 else records[2:])

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        nonlocal calls
        calls += 1
        ids = tuple(c.id for c in candidates)
        return EvidenceDecision(status="insufficient" if calls == 1 else "sufficient", selected_ids=ids,
            selected_groups=(ids,), followup_queries=("Cedar location",) if calls == 1 else ())

    monkeypatch.setattr(memory, "recall", recall)
    result = await AdaptiveRetriever(memory, Assessor(assess)).retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert result.status == "sufficient" and result.recall.facts == records
    assert result.selected_groups == (tuple(f"fact:{record.fact_id}" for record in records),)


@pytest.mark.parametrize("recall_first_again", [False, True])
async def test_atomic_carry_prunes_partial_group_before_next_assessment(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch, recall_first_again: bool) -> None:
    first = await fact(memory, "Aster", "depends on", "Beacon")
    removed = await fact(memory, "Beacon", "owned by", "Cedar")
    fresh = await fact(memory, "Aster", "owner", "Delta")
    removed_source = removed.source_episode_id
    assert removed_source is not None
    calls = 0

    async def recall(space: str, query: str, **kwargs: object) -> RecallResult:
        if calls == 0:
            return RecallResult(facts=[first, removed])
        await memory.forget("alpha", removed_source)
        return RecallResult(facts=[first, fresh] if recall_first_again else [fresh])

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        nonlocal calls
        calls += 1
        ids = tuple(c.id for c in candidates)
        if calls == 1:
            return EvidenceDecision(status="insufficient", selected_ids=ids, selected_groups=(ids,),
                                    followup_queries=("Aster owner",))
        expected = [first, fresh] if recall_first_again else [fresh]
        assert ids == tuple(f"fact:{record.fact_id}" for record in expected)
        return EvidenceDecision(status="sufficient", selected_ids=ids)

    monkeypatch.setattr(memory, "recall", recall)
    result = await AdaptiveRetriever(memory, Assessor(assess)).retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert result.status == "sufficient" and result.selected_groups == ()
    assert result.recall.facts == ([first, fresh] if recall_first_again else [fresh])
    assert "atomic_group_omitted" in result.reasons


@pytest.mark.parametrize("failure", ["invalid", "provider", "timeout", "mutated_candidate"])
async def test_default_fallback_returns_only_fresh_verified_unassessed_candidates(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    stored = await fact(memory, "Aster", "depends on", "Beacon")
    recall = AsyncMock(return_value=RecallResult(facts=[stored]))
    monkeypatch.setattr(memory, "recall", recall)

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        if failure == "invalid":
            return EvidenceDecision.model_construct(status="sufficient", selected_ids=("fact:99999",),
                selected_groups=((candidates[0].id, "fact:99999"),), followup_queries=("malicious follow-up",))
        if failure == "timeout":
            await asyncio.Event().wait()
        if failure == "mutated_candidate":
            object.__setattr__(candidates[0], "text", "invented model replacement")
        raise EvidenceAssessmentError("assessment_provider_failed")

    assessor = Assessor(assess)
    retriever = AdaptiveRetriever(memory, assessor, limits=AdaptiveLimits(timeout_s=1.0))
    result = await retriever.retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert retriever.failure_policy == "retain_verified"
    assert result.status == "uncertain" and result.evidence_basis == "verified_candidates"
    assert result.fallback_status == "retained" and result.recall.facts == [stored]
    assert result.selected_groups == ()
    assert recall.await_count == 1 and len(assessor.calls) == 1 and result.queries_used == 1
    assert result.rounds[-1].selected_count == 0
    assert "malicious follow-up" not in str(recall.call_args_list)
    assert "invented model replacement" not in result.model_dump_json()
    assert result.errors == (("invalid_or_failed_assessment",) if failure == "invalid" else
                             ("timeout",) if failure == "timeout" else ("assessment_provider_failed",))


@pytest.mark.parametrize("mutation", ["delete", "content", "scope", "session", "source", "fact"])
async def test_failed_assessment_fallback_rechecks_sources_and_authorization(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    changed = await fact(memory, "Aster", "depends on", "Beacon", metadata={"team": "blue"}, source="public/a")
    stable = await fact(memory, "Beacon", "owned by", "Cedar", metadata={"team": "blue"}, source="public/b")
    changed_id = changed.source_episode_id
    assert changed_id is not None
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=[changed, stable])))
    source = memory.documents.get_episode

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        if mutation == "delete":
            await memory.forget("alpha", changed_id)
        elif mutation == "fact":
            await memory.documents.update_fact(changed.model_copy(update={"excluded_reason": "removed"}))
        else:
            async def altered(space: str, episode_id: int) -> Episode | None:
                episode = await source(space, episode_id)
                if not isinstance(episode, Episode):
                    return None
                if episode_id != changed_id:
                    return episode
                updates: dict[str, object]
                if mutation == "content":
                    updates = {"content": "different content"}
                elif mutation == "scope":
                    updates = {"metadata": {"team": "red"}}
                elif mutation == "session":
                    updates = {"metadata": {"team": "blue", "session_id": "current"}}
                else:
                    updates = {"source": "private/a"}
                return episode.model_copy(update=updates)
            monkeypatch.setattr(memory.documents, "get_episode", altered)
        raise RuntimeError("private provider failure")

    result = await AdaptiveRetriever(memory, Assessor(assess)).retrieve("alpha", "Aster",
        scope=RecallScope.validated(where={"team": "blue"}, source_prefix="public/"), exclude_session_id="current")
    assert result.recall.facts == [stable] and result.status == "uncertain"
    assert result.fallback_status == "retained" and result.evidence_basis == "verified_candidates"
    assert "stale_evidence" in result.reasons and "private provider failure" not in result.model_dump_json()


async def test_fallback_drops_incomplete_prior_atomic_group(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch) -> None:
    first = await fact(memory, "Aster", "depends on", "Beacon")
    removed = await fact(memory, "Beacon", "owned by", "Cedar")
    fresh = await fact(memory, "Cedar", "located in", "Denver")
    removed_id = removed.source_episode_id
    assert removed_id is not None
    calls = 0

    async def recall(space: str, query: str, **kwargs: object) -> RecallResult:
        return RecallResult(facts=[first, removed] if calls == 0 else [fresh])

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        nonlocal calls
        calls += 1
        if calls == 1:
            ids = tuple(candidate.id for candidate in candidates)
            return EvidenceDecision(status="insufficient", selected_ids=ids, selected_groups=(ids,),
                                    followup_queries=("Cedar location",))
        await memory.forget("alpha", removed_id)
        raise EvidenceAssessmentError("invalid_assessment")

    monkeypatch.setattr(memory, "recall", recall)
    result = await AdaptiveRetriever(memory, Assessor(assess)).retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert result.recall.facts == [fresh] and result.selected_groups == ()
    assert result.fallback_status == "retained" and result.evidence_basis == "verified_candidates"
    assert result.status == "uncertain" and "atomic_group_omitted" in result.reasons


@pytest.mark.parametrize("failure", ["exception", "timeout", "no_survivors"])
async def test_fallback_verification_failure_never_emits_unverified_content(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    stored = await fact(memory, "Aster", "depends on", "Beacon")
    source_id = stored.source_episode_id
    assert source_id is not None
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=[stored])))

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        if failure == "no_survivors":
            await memory.forget("alpha", source_id)
        else:
            async def unavailable(space: str, fact_id: int) -> Fact | None:
                if failure == "timeout":
                    await asyncio.Event().wait()
                raise RuntimeError("private store detail")
            monkeypatch.setattr(memory.documents, "get_fact", unavailable)
        raise EvidenceAssessmentError("assessment_provider_failed")

    result = await AdaptiveRetriever(memory, Assessor(assess), limits=AdaptiveLimits(timeout_s=1.0)).retrieve(
        "alpha", "Aster", scope=RecallScope.validated())
    assert result.recall.facts == [] and result.selected_groups == () and result.evidence_basis == "none"
    assert result.status == "uncertain" and result.errors == ("assessment_provider_failed",)
    assert result.fallback_status == {"exception": "verification_failed", "timeout": "verification_timeout",
                                      "no_survivors": "empty"}[failure]
    assert "private store detail" not in result.model_dump_json()


@pytest.mark.parametrize("stage", ["assessment", "fallback"])
async def test_external_cancellation_never_returns_fallback(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    stored = await fact(memory, "Aster", "depends on", "Beacon")
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=[stored])))
    entered = asyncio.Event()
    reads = 0
    source = memory.documents.get_fact

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        nonlocal reads
        if stage == "assessment":
            entered.set()
            await asyncio.Event().wait()
        async def blocked(space: str, fact_id: int) -> Fact | None:
            nonlocal reads
            reads += 1
            entered.set()
            await asyncio.Event().wait()
            result = await source(space, fact_id)
            return result if isinstance(result, Fact) else None
        monkeypatch.setattr(memory.documents, "get_fact", blocked)
        raise EvidenceAssessmentError("assessment_provider_failed")

    task = asyncio.create_task(AdaptiveRetriever(memory, Assessor(assess)).retrieve("alpha", "Aster",
                                                                                 scope=RecallScope.validated()))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert reads == (0 if stage == "assessment" else 1)


@pytest.mark.parametrize("stage", ["retrieval", "verification"])
@pytest.mark.parametrize("timeout", [False, True])
async def test_non_assessment_failures_do_not_enable_fallback(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch, stage: str, timeout: bool) -> None:
    stored = await fact(memory, "Aster", "depends on", "Beacon")
    recalls = 0

    async def fail() -> None:
        if timeout:
            await asyncio.Event().wait()
        raise RuntimeError("private storage failure")

    async def recall(space: str, query: str, **kwargs: object) -> RecallResult:
        nonlocal recalls
        recalls += 1
        if stage == "retrieval" and recalls == 2:
            await fail()
        return RecallResult(facts=[stored])

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        if stage == "verification":
            async def unavailable(space: str) -> int:
                await fail()
                return 0
            monkeypatch.setattr(memory.documents, "revision", unavailable)
        return EvidenceDecision(status="insufficient", selected_ids=(candidates[0].id,), followup_queries=("Beacon",))

    monkeypatch.setattr(memory, "recall", recall)
    result = await AdaptiveRetriever(memory, Assessor(assess), limits=AdaptiveLimits(timeout_s=1.0)).retrieve(
        "alpha", "Aster", scope=RecallScope.validated())
    assert result.status == "uncertain" and not result.recall.facts
    assert result.evidence_basis == "none" and result.fallback_status == "not_used"
    assert result.errors == (("timeout",) if timeout else ("retrieval_failed",))


async def test_fallback_reserve_stays_within_original_deadline(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch) -> None:
    from scone_memory.retrieval import adaptive

    class Clock:
        value = 0.0
        def monotonic(self) -> float:
            return self.value

    clock = Clock()
    monkeypatch.setattr(adaptive, "time", clock)
    stored = await fact(memory, "Aster", "depends on", "Beacon")
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=[stored])))
    current = memory.documents.get_fact
    observed: list[float] = []

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        clock.value = 9.0  # Ten-second budget reserves exactly its last second.
        async def delayed(space: str, fact_id: int) -> Fact | None:
            observed.append(clock.value)
            clock.value = 10.01
            result = await current(space, fact_id)
            return result if isinstance(result, Fact) else None
        monkeypatch.setattr(memory.documents, "get_fact", delayed)
        return await sufficient(question, candidates)

    result = await AdaptiveRetriever(memory, Assessor(assess), limits=AdaptiveLimits(timeout_s=10.0)).retrieve(
        "alpha", "Aster", scope=RecallScope.validated())
    assert observed == [9.0]
    assert result.status == "uncertain" and result.fallback_status == "verification_timeout"
    assert result.evidence_basis == "none" and result.recall.facts == []


async def test_slow_assessor_cancellation_can_exhaust_fallback_reserve(memory: MemoryEngine,
        monkeypatch: pytest.MonkeyPatch) -> None:
    from scone_memory.retrieval import adaptive

    class Clock:
        value = 0.0
        def monotonic(self) -> float:
            return self.value

    clock = Clock()
    monkeypatch.setattr(adaptive, "time", clock)
    stored = await fact(memory, "Aster", "depends on", "Beacon")
    monkeypatch.setattr(memory, "recall", AsyncMock(return_value=RecallResult(facts=[stored])))

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        clock.value = 1.1
        return await sufficient(question, candidates)

    result = await AdaptiveRetriever(memory, Assessor(assess), limits=AdaptiveLimits(timeout_s=1.0)).retrieve(
        "alpha", "Aster", scope=RecallScope.validated())
    assert result.status == "uncertain" and result.fallback_status == "verification_timeout"
    assert not result.recall.facts and result.errors == ("timeout",)


async def test_normal_selection_and_explicit_empty_policy_label_their_basis(memory: MemoryEngine) -> None:
    await memory.remember("alpha", "Aster depends on Beacon.", created_at=STAMP)
    selected = await AdaptiveRetriever(memory, Assessor(sufficient)).retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert selected.evidence_basis == "assessed_selection" and selected.fallback_status == "not_used"

    async def invalid(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        raise ValueError("invalid model response")

    empty = await AdaptiveRetriever(memory, Assessor(invalid), failure_policy="empty").retrieve(
        "alpha", "Aster", scope=RecallScope.validated())
    assert empty.evidence_basis == "none" and empty.fallback_status == "not_used" and empty.recall.items == []


@pytest.mark.parametrize("value", ["retry", "RETAIN_VERIFIED", "", 1, None, True])
async def test_failure_policy_rejects_invalid_runtime_values(memory: MemoryEngine, value: object) -> None:
    from typing import cast
    from scone_memory.retrieval.adaptive import FailurePolicy

    with pytest.raises(ValueError, match="failure_policy"):
        AdaptiveRetriever(memory, Assessor(sufficient), failure_policy=cast(FailurePolicy, value))


@pytest.mark.parametrize("reason", [None, [], 7])
async def test_missing_or_wrong_typed_assessment_reason_still_recovers_verified_candidates(memory: MemoryEngine,
        reason: object) -> None:
    await memory.remember("alpha", "Aster depends on Beacon.", created_at=STAMP)

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        error = EvidenceAssessmentError("assessment_provider_failed")
        if reason is None:
            del error.reason
        else:
            setattr(error, "reason", reason)
        raise error

    result = await AdaptiveRetriever(memory, Assessor(assess)).retrieve("alpha", "Aster", scope=RecallScope.validated())
    assert result.errors == ("invalid_or_failed_assessment",)
    assert result.status == "uncertain" and result.fallback_status == "retained"
    assert result.evidence_basis == "verified_candidates" and result.recall.items
