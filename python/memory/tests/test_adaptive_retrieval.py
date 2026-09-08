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
    EvidenceDecision, evidence_payload_bytes)
from scone_memory.retrieval.recall_scope import RecallScope

STAMP = "2025-01-01T00:00:00Z"
AssessorFn = Callable[[str, tuple[EvidenceCandidate, ...]], Awaitable[EvidenceDecision]]


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


async def test_insufficient_without_followups_abstains(memory: MemoryEngine) -> None:
    await memory.remember("alpha", "Aster depends on Beacon.", created_at=STAMP)

    async def assess(question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        return EvidenceDecision(status="insufficient", selected_ids=())

    result = await AdaptiveRetriever(memory, Assessor(assess)).retrieve("alpha", "Aster", scope=RecallScope.validated())
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

    result = await AdaptiveRetriever(memory, Assessor(assess)).retrieve("alpha", "Aster", scope=RecallScope.validated())
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

    retriever = AdaptiveRetriever(memory, Assessor(blocked), limits=AdaptiveLimits(timeout_s=1.0))
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

    result = await AdaptiveRetriever(memory, Assessor(failing)).retrieve("alpha", "Aster", scope=RecallScope.validated())
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

    result = await AdaptiveRetriever(memory, Assessor(late), limits=AdaptiveLimits(timeout_s=1.0)).retrieve(
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
