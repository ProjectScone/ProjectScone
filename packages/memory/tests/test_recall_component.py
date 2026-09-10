"""Recall can operate on storage ports without constructing a memory engine."""

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex
from scone_memory.core.ports import NewChunk, NewEpisode

STAMP = "2026-09-08T00:00:00.000Z"


async def test_component_returns_retained_scoped_text_and_records_its_evidence():
    from scone_memory.retrieval.recall import RecallRuntime, recall

    documents, vectors, embedder = InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()
    await vectors.ensure(embedder.dim)
    for space, content in (("alpha", "needle calibration uses Polaris"), ("beta", "needle private")):
        episode = await documents.insert_episode(NewEpisode(
            space=space, kind="file", content=content, content_hash=space,
            created_at=STAMP, ingested_at=STAMP, source="manuals/calibration",
        ))
        await documents.insert_chunks([NewChunk(
            episode_id=episode.episode_id, space=space, ordinal=0, start=0,
            end=len(content.encode()), text=content, created_at=STAMP,
        )])
    recorded = []

    async def emit(space, kind, payload):
        recorded.append((space, kind, payload))
        return None

    runtime = RecallRuntime(documents, vectors, embedder, lambda: STAMP, emit,
                            lambda query: {"query": "redacted", "query_hashed": True})
    result = await recall(runtime, "alpha", "needle", kind="file", source_prefix="manuals/")

    assert [item.text for item in result.items] == ["needle calibration uses Polaris"]
    assert result.degraded == []
    assert recorded[0][:2] == ("alpha", "recall")
    assert recorded[0][2]["query"] == "redacted"
    assert recorded[0][2]["items"][0]["chunk_id"] == result.items[0].chunk_id
