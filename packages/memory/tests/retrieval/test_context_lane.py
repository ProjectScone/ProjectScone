"""The context lane: a passage found by what it is under, not only by what it says.

A chunk under "Refunds" in "Billing rules" that never says either word
is invisible to the text lane. With the context lane on, each chunk's
headings, title, source name and document terms -- the ones the chunk
lacks -- are indexed beside its text and searched as a third lane, fused
by rank with a lower weight, so a passage found only through its context
competes on rank and every recall says in ``lanes`` which passages the
context lane placed. Stored text never changes; a store without the lane
says so instead of pretending.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends import SqliteDocumentStore
from scone_memory.core.errors import InvalidInput
from scone_memory.core.ports import TextFilter

DOC = ("# Billing rules\n\nWhat the company does with money.\n\n## Refunds\n\n"
       "A customer may ask for the amount back within thirty days. The request goes to the desk that handled "
       "the sale. Money returns the way it came, and nobody is asked why.\n\n## Late fees\n\n"
       "After the due date two percent is added.\n")
#: Under the heading, past the cut: this chunk says neither "refunds" nor "billing".
DEEP = "nobody is asked why"
NOISE = ["Billing refunds are discussed in the handbook chapter on refunds and billing.",
         "The billing team met about refunds again; billing, refunds, billing.",
         "Refund policy for billing disputes: see billing refunds."]
#: Says "Refunds" but never "billing"; its source name puts it under billing: a text hit on one word, a context hit on the other.
SIDE = "Refunds desk hours are nine to five on weekdays."


@pytest.fixture(params=["memory", "sqlite"])
async def stores(request, tmp_path):
    if request.param == "memory":
        documents = InMemoryDocumentStore()
    else:
        documents = SqliteDocumentStore(str(tmp_path / "s.db"))
    yield documents
    close = getattr(documents, "close", None)
    if close is not None:
        result = close()
        if hasattr(result, "__await__"):
            await result


async def engine_with(documents, *, context_lane: bool, structure_aware: bool = False) -> MemoryEngine:
    engine = await MemoryEngine(documents, InMemoryVectorIndex(), HashEmbedder(), chunk_target=120,
                                context_lane=context_lane, structure_aware=structure_aware).open()
    for text in NOISE:
        await engine.remember("s", text)
    await engine.remember("s", DOC, source="docs/billing-rules.md")
    await engine.remember("s", SIDE, source="docs/billing-desk.md")
    return engine


async def test_the_store_indexes_and_searches_context_beside_text(stores):
    documents = stores
    assert documents.context_lane is True
    from scone_memory.core.ports import NewChunk, NewEpisode
    episode = await documents.insert_episode(NewEpisode(space="s", kind="note", content="The amount is due.",
                                                        content_hash="h1", source=None, tags=(), metadata={},
                                                        created_at="2026-09-13T00:00:00Z",
                                                        ingested_at="2026-09-13T00:00:00Z"))
    [chunk] = await documents.insert_chunks([NewChunk(episode_id=episode.episode_id, space="s", ordinal=0, start=0,
                                                       end=18, text="The amount is due.", created_at=episode.created_at)])
    await documents.index_context("s", chunk.chunk_id, "billing rules refunds")
    assert [cid for cid, _ in await documents.search_context("s", "refunds", 5, TextFilter())] == [chunk.chunk_id]
    assert await documents.search_text("s", "refunds", 5, TextFilter()) == [], "the text lane never sees context words"
    assert await documents.search_context("s", "refunds", 5, TextFilter(as_of="2026-09-12T00:00:00Z")) == []
    await documents.delete_episode("s", episode.episode_id)
    assert await documents.search_context("s", "refunds", 5, TextFilter()) == []
    if isinstance(documents, SqliteDocumentStore):
        assert documents.conn.execute("SELECT count(*) FROM chunk_context").fetchone()[0] == 0, "no row outlives its chunk"
        assert documents.conn.execute("SELECT count(*) FROM chunk_context_fts WHERE chunk_context_fts MATCH 'refunds'").fetchone()[0] == 0


async def test_a_passage_is_found_by_what_it_is_under(stores):
    engine = await engine_with(stores, context_lane=True)
    found = await engine.recall("s", "refunds under the billing rules", limit=6)
    target = next(item for item in found.items if DEEP in item.text)
    assert target.lanes.get("context") is not None and target.lanes.get("text") is None
    assert "refunds" not in target.text.lower() and "billing" not in target.text.lower(), "stored text is unchanged"
    without = await engine_with(type(stores)() if isinstance(stores, InMemoryDocumentStore) else stores.__class__(":memory:"),
                                context_lane=False)
    unfused = await without.recall("s", "refunds under the billing rules", limit=6)
    target_without = next(item for item in unfused.items if DEEP in item.text)
    assert target.score > target_without.score, "the lane's rank is fused in: the passage rises, not only gets labelled"


async def test_without_the_lane_the_same_passage_is_missed(stores):
    engine = await engine_with(stores, context_lane=False)
    found = await engine.recall("s", "refunds under the billing rules", limit=4)
    assert all(DEEP not in item.text for item in found.items)
    assert all("context" not in item.lanes for item in found.items)


async def test_a_store_without_the_lane_says_so():
    class Plain(InMemoryDocumentStore):
        context_lane = False

    engine = await engine_with(Plain(), context_lane=True)
    found = await engine.recall("s", "refunds", limit=3)
    assert any(note.startswith("context lane:") for note in found.degraded)
    assert all("context" not in item.lanes for item in found.items)


async def test_a_passage_that_says_one_word_and_is_under_another_comes_before_one_only_under_them(stores):
    engine = await engine_with(stores, context_lane=True)
    found = await engine.recall("s", "billing refunds", limit=10)
    both = next(item for item in found.items if item.text == SIDE)
    only_under = next(item for item in found.items if DEEP in item.text)
    assert both.lanes.get("text") is not None and both.lanes.get("context") is not None
    assert found.items.index(both) < found.items.index(only_under)
    assert found.items.index(only_under) < len(found.items) - 1, "and one only under the words still beats some chatter"


async def test_the_flag_is_a_boolean_and_reaches_the_ingestion_record():
    with pytest.raises(InvalidInput, match="context_lane"):
        MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), context_lane="yes")  # type: ignore[arg-type]
    engine = await engine_with(InMemoryDocumentStore(), context_lane=True)
    assert engine.context_lane is True
