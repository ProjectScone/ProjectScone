"""Malformed backend results must not escape scope or invent source evidence."""
from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.deduplication import DocumentRevision, MemoryCandidateProvider, PassageEmbedding


@pytest.fixture
async def guard_memory():
    engine = MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder())
    await engine.open()
    yield engine
    await engine.close()


def incoming():
    return DocumentRevision("alpha", "incoming", "1", "Juniper calibration uses Polaris for a stable reference clock.")


async def lexical(provider):
    return await provider.candidates(incoming(), query_embeddings=(), embedder_id=None, limit=8)


@pytest.mark.parametrize("mutation", [{"space": "private"}, {"chunk_id": 999999}, {"text": "forged"}, {"start": -1}])
async def test_malformed_candidate_chunk_cannot_be_returned(guard_memory, monkeypatch, mutation):
    await guard_memory.remember("alpha", incoming().text)
    original = guard_memory.documents.get_chunks

    async def malicious(space, ids):
        return [chunk.model_copy(update=mutation) for chunk in await original(space, ids)]

    monkeypatch.setattr(guard_memory.documents, "get_chunks", malicious)
    with pytest.raises(ValueError, match="scope|original"):
        await lexical(MemoryCandidateProvider(guard_memory))


async def test_malformed_candidate_owner_cannot_escape_scope(guard_memory, monkeypatch):
    await guard_memory.remember("alpha", incoming().text)
    original = guard_memory.documents.get_episode

    async def malicious(space, episode_id):
        episode = await original(space, episode_id)
        return episode.model_copy(update={"space": "private"}) if episode else None

    monkeypatch.setattr(guard_memory.documents, "get_episode", malicious)
    with pytest.raises(ValueError, match="scope"):
        await lexical(MemoryCandidateProvider(guard_memory))


async def test_candidate_forgotten_after_chunk_read_is_omitted(guard_memory, monkeypatch):
    saved = await guard_memory.remember("alpha", incoming().text)
    original = guard_memory.documents.get_chunks

    async def forget_between_reads(space, ids):
        chunks = await original(space, ids)
        await guard_memory.forget("alpha", saved.episode_id)
        return chunks

    monkeypatch.setattr(guard_memory.documents, "get_chunks", forget_between_reads)
    result = await lexical(MemoryCandidateProvider(guard_memory))
    assert not result.documents
    assert await guard_memory.documents.get_episode("alpha", saved.episode_id) is None


async def test_cross_scope_vector_hit_is_not_exposed(guard_memory, monkeypatch):
    saved = await guard_memory.remember("private", incoming().text)
    from scone_memory.core.ports import TextFilter
    hits = await guard_memory.documents.search_text("private", incoming().text, 1, TextFilter())
    assert hits

    async def foreign_index(space, vector, limit):
        return [(hits[0][0], .99)]

    monkeypatch.setattr(guard_memory.vectors, "search", foreign_index)
    vector = tuple((await guard_memory.embedder.embed([incoming().text]))[0])
    result = await MemoryCandidateProvider(guard_memory).candidates(
        incoming(), query_embeddings=(PassageEmbedding(0, len(incoming().text.encode()), vector),),
        embedder_id=guard_memory.embedder.id, limit=8,
    )
    assert not result.semantic_matches
    assert await guard_memory.documents.get_episode("private", saved.episode_id) is not None


@pytest.mark.parametrize("embedder_id,vector", [("other-model", (1.0, 0.0)), (None, (1.0, 0.0))])
async def test_invalid_model_or_vector_dimension_is_rejected(guard_memory, embedder_id, vector):
    # None stands for the memory's own embedder, so the second case can only
    # fail on the vector's width.
    with pytest.raises(ValueError, match="embedder|dimension"):
        await MemoryCandidateProvider(guard_memory).candidates(
            incoming(), query_embeddings=(PassageEmbedding(0, 1, vector),),
            embedder_id=guard_memory.embedder.id if embedder_id is None else embedder_id, limit=8,
        )


async def test_lexical_queries_are_bounded_windows(guard_memory, monkeypatch):
    calls = []
    original = guard_memory.documents.search_text

    async def observe(space, query, limit, filter):
        calls.append((space, query, limit))
        return await original(space, query, limit, filter)

    monkeypatch.setattr(guard_memory.documents, "search_text", observe)
    text = " ".join(f"observation-{index}" for index in range(1000))
    await MemoryCandidateProvider(guard_memory).candidates(
        DocumentRevision("alpha", "long", "1", text), query_embeddings=(), embedder_id=None, limit=8,
    )
    assert 1 <= len(calls) <= 8
    assert all(space == "alpha" and len(query) <= 256 and limit == 8 for space, query, limit in calls)


@pytest.mark.parametrize("limit", [0, 257])
async def test_candidate_request_limits_are_validated(guard_memory, limit):
    with pytest.raises(ValueError, match="limit"):
        await MemoryCandidateProvider(guard_memory).candidates(incoming(), query_embeddings=(), embedder_id=None, limit=limit)


async def test_oversized_semantic_batch_is_rejected(guard_memory):
    vector = (1.0,) + (0.0,) * 255
    with pytest.raises(ValueError, match="batch"):
        await MemoryCandidateProvider(guard_memory).candidates(
            incoming(), query_embeddings=(PassageEmbedding(0, 1, vector),) * 257,
            embedder_id=guard_memory.embedder.id, limit=8,
        )
