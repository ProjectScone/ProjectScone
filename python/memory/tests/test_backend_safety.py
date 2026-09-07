"""Backend failures that must not silently corrupt or hide memory."""

from dataclasses import replace

import pytest

from scone_memory.ports import TextFilter, VectorPoint


def point(chunk_id, vector):
    return VectorPoint(
        chunk_id=chunk_id,
        space="default",
        episode_id=chunk_id,
        created_at="2024-01-01T00:00:00.000Z",
        vector=vector,
    )


@pytest.mark.parametrize("malformation", ["short", "long", "nan", "inf"])
async def test_invalid_vector_batch_is_rejected_before_any_write(engine, malformation):
    dim = engine.embedder.dim
    original = point(9001, [1.0] + [0.0] * (dim - 1))
    await engine.vectors.upsert([original])
    bad = list(original.vector)
    if malformation == "short":
        bad.pop()
    elif malformation == "long":
        bad.append(0.0)
    else:
        bad[0] = float(malformation)
    changed = replace(original, vector=[0.0, 1.0] + [0.0] * (dim - 2))
    with pytest.raises(ValueError, match="vector"):
        await engine.vectors.upsert([changed, point(9002, bad)])
    hits = await engine.vectors.search("default", original.vector, 10)
    assert [chunk_id for chunk_id, _ in hits] == [original.chunk_id]
    assert hits[0][1] == pytest.approx(1.0)


@pytest.mark.parametrize("malformation", ["short", "long", "nan", "inf"])
async def test_invalid_vector_query_is_rejected_even_when_empty(engine, malformation):
    vector = [1.0] + [0.0] * (engine.embedder.dim - 1)
    if malformation == "short":
        vector.pop()
    elif malformation == "long":
        vector.append(0.0)
    else:
        vector[0] = float(malformation)
    with pytest.raises(ValueError, match="vector"):
        await engine.vectors.search("default", vector, 1)


@pytest.mark.parametrize("scope", ["tags", "metadata", "both"])
async def test_lexical_filter_finds_matches_beyond_unfiltered_candidate_limit(engine, scope):
    # Higher-ranked distractors fill the former eight-candidate window.
    for i in range(12):
        await engine.remember(
            "default", f"launch launch launch {i}",
            tags=["other"], metadata={"user_id": "bob"},
        )
    wanted = await engine.remember(
        "default", "launch details for the approved deployment schedule",
        tags=["work", "approved"], metadata={"user_id": "alice", "team": "platform"},
    )
    filters = TextFilter(
        tags=("work", "approved") if scope in ("tags", "both") else (),
        where={"user_id": "alice", "team": "platform"} if scope in ("metadata", "both") else {},
    )
    hits = await engine.documents.search_text("default", "launch", 1, filters)
    chunks = await engine.documents.get_chunks("default", [chunk_id for chunk_id, _ in hits])
    assert [chunk.episode_id for chunk in chunks] == [wanted.episode_id]
