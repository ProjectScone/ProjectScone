"""Optional ranking uses only real scoped candidates and keeps a safe baseline."""
from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, asdict
import json
from unittest.mock import AsyncMock

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.retrieval.reranking import RerankScore


STAMP = "2026-09-07T00:00:00.000Z"


class TargetRanker:
    def __init__(self):
        self.calls = []

    async def rerank(self, query, candidates):
        self.calls.append((query, candidates))
        return [RerankScore(candidate.chunk_id, 10.0 if "ANSWER" in candidate.text else 0.0) for candidate in candidates]


@pytest.fixture
async def engine():
    instance = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=lambda: STAMP).open()
    yield instance
    await instance.close()


async def ordered(engine, count=6):
    ids = []
    for number in range(count):
        episode = await engine.remember("alpha", f"{'ANSWER' if number == count - 1 else 'NOISE!'} record {number:02d}", created_at=STAMP)
        ids.append((await engine.documents.chunks_of("alpha", episode.episode_id))[0].chunk_id)
    engine.vectors.search = AsyncMock(side_effect=lambda space, query, limit, *args: [(cid, 0.9 - i / 100) for i, cid in enumerate(ids[:limit])])
    engine.documents.search_text = AsyncMock(return_value=[])
    return ids


async def test_independent_candidate_pool_recovers_answer_without_increasing_return_limit(engine):
    ids = await ordered(engine)
    ranker = TargetRanker()
    engine.reranker = ranker
    coupled = await engine.recall("alpha", "which record answers", limit=1)
    independent = await engine.recall("alpha", "which record answers", limit=1, candidate_limit=8)
    assert [item.chunk_id for item in coupled.items] == [ids[0]]
    assert [item.chunk_id for item in independent.items] == [ids[-1]]
    assert coupled.returned_bytes == independent.returned_bytes
    assert independent.rerank.candidates_sent == 6
    assert independent.rerank.ordering == "rerank"
    assert independent.items[0].rerank_score == 10.0
    assert 0 < independent.items[0].score < 1
    assert independent.items[0].similarity == 0.85
    assert independent.items[0].lanes == {"vector": 6}
    assert independent.top_similarity == coupled.top_similarity == 0.9
    assert independent.low_confidence == coupled.low_confidence
    assert engine.vectors.search.call_args.args[2] == 8


async def test_unconfigured_defaults_and_per_call_disable_preserve_baseline(engine):
    await ordered(engine)
    baseline = await engine.recall("alpha", "query", limit=2)
    assert baseline.rerank is None
    assert all(item.rerank_score is None for item in baseline.items)
    assert engine.vectors.search.call_args.args[2] == 8
    engine.reranker = TargetRanker()
    disabled = await engine.recall("alpha", "query", limit=2, rerank=False)
    assert disabled.model_dump() == baseline.model_dump()
    assert engine.reranker.calls == []


@pytest.mark.parametrize("mode", ["duplicate", "missing", "unknown", "nan", "infinity", "bool", "wrong_type", "raises", "timeout"])
async def test_bad_rankers_fall_back_to_identical_baseline_without_private_diagnostics(engine, mode):
    ids = await ordered(engine)
    baseline = await engine.recall("alpha", "query", limit=2, candidate_limit=8)

    class Bad:
        async def rerank(self, query, candidates):
            if mode == "raises":
                raise RuntimeError("private-provider-key")
            if mode == "timeout":
                await asyncio.Event().wait()
            scores = [RerankScore(c.chunk_id, 1) for c in candidates]
            if mode == "duplicate":
                scores[-1] = scores[0]
            elif mode == "missing":
                scores.pop()
            elif mode == "unknown":
                scores[-1] = RerankScore(999999, 3)
            elif mode in {"nan", "infinity", "bool"}:
                scores[-1] = RerankScore(ids[-1], {"nan": float("nan"), "infinity": float("inf"), "bool": True}[mode])
            elif mode == "wrong_type":
                return [{"chunk_id": c.chunk_id, "score": 1} for c in candidates]
            return scores

    engine.reranker = Bad()
    engine.rerank_timeout = .005
    result = await engine.recall("alpha", "query", limit=2, candidate_limit=8)
    assert result.items == baseline.items
    assert result.facts == baseline.facts and result.history == baseline.history
    assert result.rerank.status == "failed" and result.rerank.ordering == "fusion"
    assert len(result.degraded) == 1 and result.degraded[0].startswith("rerank:")
    assert "private-provider-key" not in result.model_dump_json()


async def test_score_ties_retain_baseline_order_and_do_not_rewrite_scores(engine):
    await ordered(engine)
    baseline = await engine.recall("alpha", "query", limit=3, candidate_limit=8)

    class Equal:
        async def rerank(self, query, candidates):
            return [RerankScore(c.chunk_id, -100) for c in reversed(candidates)]

    engine.reranker = Equal()
    result = await engine.recall("alpha", "query", limit=3, candidate_limit=8)
    assert [item.chunk_id for item in result.items] == [item.chunk_id for item in baseline.items]
    assert [item.score for item in result.items] == [item.score for item in baseline.items]
    assert all(item.rerank_score == -100 for item in result.items)


async def test_callback_sees_no_foreign_wrong_scope_or_deleted_source_text(engine):
    allowed = await engine.remember("alpha", "ANSWER allowed", tags=["approved"], metadata={"team": "a"}, source="docs/a", kind="file", created_at=STAMP)
    private = await engine.remember("alpha", "private-scope-marker", metadata={"team": "b"}, source="private/a", created_at=STAMP)
    foreign = await engine.remember("beta", "foreign-space-marker", created_at=STAMP)
    gone = await engine.remember("alpha", "deleted-source-marker", tags=["approved"], metadata={"team": "a"}, source="docs/gone", kind="file", created_at=STAMP)
    chunks = [(await engine.documents.chunks_of(space, episode.episode_id))[0]
              for space, episode in [("alpha", allowed), ("alpha", private), ("beta", foreign), ("alpha", gone)]]
    await engine.forget("alpha", gone.episode_id)
    # Stale/misbehaving candidates bypass storage filtering on purpose.
    engine.vectors.search = AsyncMock(return_value=[(chunk.chunk_id, .9) for chunk in chunks])
    engine.documents.search_text = AsyncMock(return_value=[])
    engine.documents.get_chunks = AsyncMock(return_value=chunks)
    engine.reranker = TargetRanker()
    result = await engine.recall("alpha", "query", limit=4, candidate_limit=8, tags=["approved"], where={"team": "a"},
                                kind="file", source_prefix="docs/", since="2026-01-01", until="2026-12-31",
                                conditions={"field": "team", "is": "a"})
    assert [candidate.text for candidate in engine.reranker.calls[0][1]] == ["ANSWER allowed"]
    assert [item.episode_id for item in result.items] == [allowed.episode_id]
    assert "private-scope-marker" not in result.model_dump_json()
    assert "foreign-space-marker" not in result.model_dump_json()
    assert "deleted-source-marker" not in result.model_dump_json()


async def test_deleted_during_reranking_cannot_resurface(engine):
    ids = await ordered(engine)
    selected = await engine.documents.get_chunks("alpha", [ids[-1]])

    class ForgetDuring:
        async def rerank(self, query, candidates):
            await engine.forget("alpha", selected[0].episode_id)
            return [RerankScore(c.chunk_id, 10 if c.chunk_id == ids[-1] else 0) for c in candidates]

    engine.reranker = ForgetDuring()
    result = await engine.recall("alpha", "query", limit=1, candidate_limit=8)
    assert result.items == []
    assert "rerank: retained_sources_changed" in result.degraded


async def test_utf8_payload_and_candidate_count_are_bounded_without_clipping(engine):
    await ordered(engine, 10)
    engine.reranker = TargetRanker()
    engine.rerank_limit = 3
    engine.rerank_max_bytes = 512
    result = await engine.recall("alpha", "星" * 20, limit=2, candidate_limit=20)
    query, candidates = engine.reranker.calls[0]
    encoded = json.dumps({"query": query, "candidates": [asdict(c) for c in candidates]}, ensure_ascii=False, separators=(",", ":")).encode()
    assert len(candidates) <= 3 and len(encoded) <= 512
    assert result.rerank.payload_bytes == len(encoded)
    assert result.rerank.candidates_sent == len(candidates)
    assert result.rerank.candidates_omitted == 10 - len(candidates)
    assert all(candidate.text.startswith("NOISE!") and len(candidate.text) == 16 for candidate in candidates)
    with pytest.raises(FrozenInstanceError):
        candidates[0].text = "changed"


async def test_empty_payload_skips_callback_and_preserves_baseline(engine):
    await ordered(engine)
    baseline = await engine.recall("alpha", "星" * 900, limit=2)
    engine.reranker = TargetRanker()
    engine.rerank_max_bytes = 512
    result = await engine.recall("alpha", "星" * 900, limit=2)
    assert result.items == baseline.items
    assert result.rerank.status == "empty" and result.rerank.payload_bytes == 0
    assert not engine.reranker.calls


async def test_cancellation_propagates_and_cancels_ranker(engine):
    await ordered(engine)
    entered, cancelled = asyncio.Event(), asyncio.Event()

    class Pending:
        async def rerank(self, query, candidates):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    engine.reranker = Pending()
    task = asyncio.create_task(engine.recall("alpha", "query"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.parametrize("value", [True, 0, -1, 1001, 1.5, "20"])
async def test_invalid_candidate_budget_stops_before_embedding_or_search(engine, value):
    engine.embedder.embed = AsyncMock(side_effect=AssertionError("must not run"))
    with pytest.raises(InvalidInput):
        await engine.recall("alpha", "query", candidate_limit=value)
    engine.embedder.embed.assert_not_called()


@pytest.mark.parametrize("options", [{"rerank_limit": 0}, {"rerank_limit": 129}, {"rerank_limit": True},
    {"rerank_max_bytes": 511}, {"rerank_max_bytes": 256001}, {"rerank_timeout": 0}, {"rerank_timeout": 11},
    {"rerank_timeout": float("nan")}, {"rerank_timeout": True}])
def test_invalid_reranker_budgets_are_refused_at_construction(options):
    with pytest.raises(InvalidInput):
        MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), **options)


@pytest.mark.parametrize("dimension", ["tags", "where", "kind", "source_prefix", "since", "until", "as_of", "conditions"])
async def test_each_scope_dimension_is_checked_independently_before_ranker(engine, dimension):
    allowed = dict(tags=["approved"], metadata={"team": "a"}, source="docs/a", kind="file", created_at=STAMP)
    rejected = dict(allowed)
    options = {"tags": {"tags": ["approved"]}, "where": {"where": {"team": "a"}}, "kind": {"kind": "file"},
               "source_prefix": {"source_prefix": "docs/"}, "since": {"since": STAMP}, "until": {"until": STAMP},
               "as_of": {"as_of": STAMP}, "conditions": {"conditions": {"field": "team", "is": "a"}}}
    if dimension == "tags":
        rejected["tags"] = []
    elif dimension in {"where", "conditions"}:
        rejected["metadata"] = {"team": "b"}
    elif dimension == "kind":
        rejected["kind"] = "note"
    elif dimension == "source_prefix":
        rejected["source"] = "private/a"
    elif dimension == "since":
        rejected["created_at"] = "2025-01-01"
    else:
        rejected["created_at"] = "2027-01-01"
    retained = await engine.remember("alpha", "ANSWER allowed scope", **allowed)
    hidden = await engine.remember("alpha", "private-single-scope-marker", **rejected)
    chunks = [(await engine.documents.chunks_of("alpha", episode.episode_id))[0] for episode in (retained, hidden)]
    engine.vectors.search = AsyncMock(return_value=[(chunk.chunk_id, .9) for chunk in reversed(chunks)])
    engine.documents.search_text = AsyncMock(return_value=[])
    engine.reranker = TargetRanker()
    result = await engine.recall("alpha", "query", **options[dimension])
    assert [candidate.text for candidate in engine.reranker.calls[0][1]] == ["ANSWER allowed scope"]
    assert [item.episode_id for item in result.items] == [retained.episode_id]


async def test_chunk_bytes_must_match_original_episode_before_callback(engine):
    ids = await ordered(engine, 1)
    chunks = await engine.documents.get_chunks("alpha", ids)
    engine.documents.get_chunks = AsyncMock(return_value=[chunks[0].model_copy(update={"start": 1})])
    engine.reranker = TargetRanker()
    result = await engine.recall("alpha", "query")
    assert result.items == []
    assert result.rerank.status == "empty"
    assert not engine.reranker.calls


async def test_constructor_candidate_default_and_call_override_reach_both_lanes(engine):
    await ordered(engine)
    engine.candidate_limit = 12
    await engine.recall("alpha", "query", limit=1)
    assert engine.vectors.search.call_args.args[2] == 12
    assert engine.documents.search_text.call_args.args[2] == 12
    await engine.recall("alpha", "query", limit=50, candidate_limit=3)
    assert engine.vectors.search.call_args.args[2] == 3
    assert engine.documents.search_text.call_args.args[2] == 3


async def test_reranker_can_promote_third_source_passage_before_diversity_cap(engine):
    engine.chunk_target = 140
    episode = await engine.remember("alpha", "A" * 140 + "B" * 140 + "ANSWER" + "C" * 134, created_at=STAMP)
    chunks = await engine.documents.chunks_of("alpha", episode.episode_id)
    assert len(chunks) == 3 and "ANSWER" in chunks[2].text
    ids = [chunk.chunk_id for chunk in chunks]
    engine.vectors.search = AsyncMock(return_value=[(cid, .9 - i / 100) for i, cid in enumerate(ids)])
    engine.documents.search_text = AsyncMock(return_value=[])
    baseline = await engine.recall("alpha", "query", limit=3, candidate_limit=8)
    assert [item.chunk_id for item in baseline.items] == ids[:2]
    engine.reranker = TargetRanker()
    result = await engine.recall("alpha", "query", limit=3, candidate_limit=8)
    assert [candidate.chunk_id for candidate in engine.reranker.calls[0][1]] == ids
    assert [item.chunk_id for item in result.items] == [ids[2], ids[0]]
    assert result.items[0].score < 1 and result.items[0].rerank_score == 10

    class Fails:
        async def rerank(self, query, candidates):
            raise RuntimeError("failure")

    engine.reranker = Fails()
    fallback = await engine.recall("alpha", "query", limit=3, candidate_limit=8)
    assert fallback.items == baseline.items


async def test_tied_reranker_preserves_restated_baseline_normalization(engine):
    engine.demote_restated = True
    ids = []
    for name in ("Pierre Beaumarchais", "Thomas Kyd"):
        episode = await engine.remember("alpha", f"The author of The Marriage of Figaro is {name}.", created_at=STAMP)
        ids.append((await engine.documents.chunks_of("alpha", episode.episode_id))[0].chunk_id)
    engine.vectors.search = AsyncMock(return_value=[(ids[0], .9), (ids[1], .8)])
    engine.documents.search_text = AsyncMock(return_value=[])
    baseline = await engine.recall("alpha", "author", limit=2)
    assert [item.chunk_id for item in baseline.items] == ids[::-1]
    assert baseline.items[0].score == 1 and baseline.items[1].score > 1

    class Equal:
        async def rerank(self, query, candidates):
            return [RerankScore(candidate.chunk_id, 1) for candidate in candidates]

    engine.reranker = Equal()
    result = await engine.recall("alpha", "author", limit=2)
    assert [(item.chunk_id, item.score) for item in result.items] == [(item.chunk_id, item.score) for item in baseline.items]
