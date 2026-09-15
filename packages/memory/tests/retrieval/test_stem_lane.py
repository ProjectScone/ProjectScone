"""The text lane finds a word's family when asked to, and says which prefixes it added.

"bills" misses "billing run"; with stem prefixes on, the query also
searches ``bill*`` and finds it through the text lane. The index is
untouched, a store that cannot take prefixes says so on the record, and
nothing changes unless the flag is on.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends import SqliteDocumentStore
from scone_memory.core.errors import InvalidInput
from scone_memory.core.ports import TextFilter

BILLING = "The billing run went out late and the invoices were wrong."
NOISE = ["The garden shed needs a new roof before winter.", "Lisbon in March is mild and green."]


@pytest.fixture(params=["memory", "sqlite"])
async def stores(request, tmp_path):
    documents = InMemoryDocumentStore() if request.param == "memory" else SqliteDocumentStore(str(tmp_path / "s.db"))
    yield documents
    close = getattr(documents, "close", None)
    if close is not None:
        result = close()
        if hasattr(result, "__await__"):
            await result


async def engine_with(documents, *, lexical_stems: bool) -> MemoryEngine:
    engine = await MemoryEngine(documents, InMemoryVectorIndex(), HashEmbedder(), lexical_stems=lexical_stems).open()
    for text in (BILLING, *NOISE):
        await engine.remember("s", text)
    return engine


async def test_the_store_searches_prefixes_beside_whole_terms(stores):
    from scone_memory.core.ports import NewChunk, NewEpisode

    documents = stores
    assert documents.prefix_terms is True
    episode = await documents.insert_episode(NewEpisode(space="s", kind="note", content=BILLING, content_hash="h1", source=None,
                                                        tags=(), metadata={}, created_at="2026-09-13T00:00:00Z",
                                                        ingested_at="2026-09-13T00:00:00Z"))
    [chunk] = await documents.insert_chunks([NewChunk(episode_id=episode.episode_id, space="s", ordinal=0, start=0,
                                                       end=len(BILLING.encode()), text=BILLING, created_at=episode.created_at)])
    assert await documents.search_text("s", "bills", 5, TextFilter()) == []
    found = await documents.search_terms("s", "bills", 5, TextFilter(), prefixes=("bill",))
    assert [cid for cid, _ in found] == [chunk.chunk_id]
    assert await documents.search_terms("s", "bills", 5, TextFilter(as_of="2026-09-12T00:00:00Z"), prefixes=("bill",)) == []


async def test_recall_finds_the_family_and_says_which_prefixes_it_added(stores):
    engine = await engine_with(stores, lexical_stems=True)
    found = await engine.recall("s", "the bills and the invoice", limit=3)
    assert found.items[0].text == BILLING and found.items[0].lanes.get("text") == 1
    assert found.prefixes == {"added": ["bill", "invoic"], "applied": True, "exact_forms": False}
    plain = await engine.recall("s", "Lisbon in March", limit=3)
    assert plain.prefixes == {"added": [], "applied": True, "exact_forms": False}, "a query with no family to add still says so"


async def test_without_the_flag_the_family_is_missed_and_nothing_is_recorded(stores):
    engine = await engine_with(stores, lexical_stems=False)
    found = await engine.recall("s", "the bills and the invoice", limit=3)
    assert all("text" not in item.lanes for item in found.items) and found.prefixes is None


async def test_a_store_without_prefix_terms_answers_and_says_the_prefixes_were_not_applied():
    class Plain(InMemoryDocumentStore):
        prefix_terms = False

    engine = await engine_with(Plain(), lexical_stems=True)
    found = await engine.recall("s", "the bills", limit=3)
    assert found.prefixes == {"added": ["bill"], "applied": False, "exact_forms": False}
    assert all("text" not in item.lanes for item in found.items)


async def test_the_recall_event_carries_the_counts_and_the_flag_is_a_boolean():
    from scone_memory import InMemoryEventLog

    events = InMemoryEventLog()
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), events=events, lexical_stems=True).open()
    await engine.remember("s", BILLING)
    await engine.recall("s", "bills", limit=1)
    [event] = await engine.events.query("s", kind="recall")
    assert event.payload["prefixes"] == {"added": 1, "applied": True, "exact_forms": False}
    with pytest.raises(InvalidInput, match="lexical_stems"):
        MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), lexical_stems="yes")  # type: ignore[arg-type]


async def test_exact_forms_reach_the_store_and_the_record_says_so(stores):
    """With exact forms on, the passage holding the query's own word leads
    the text lane over a shorter one holding only a relative; off, the
    relative leads. Both the result and the recall event say which ran."""
    from scone_memory import InMemoryEventLog

    exact = "The billing statement for the harbour office arrived in March."
    relative = "Bills were paid at the harbour office."
    noise = ["The garden shed needs a new roof before winter.", "Lisbon in March is mild and green.",
             "A kettle whistled in the kitchen upstairs.", "The ferry leaves the pier at seven.",
             "Snow closed the mountain road for a week.", "Her violin lesson moved to Thursday."]
    events = InMemoryEventLog()
    engine = await MemoryEngine(stores, InMemoryVectorIndex(), HashEmbedder(), events=events,
                                lexical_exact_forms=True).open()
    for text in (exact, relative, *noise):
        await engine.remember("s", text)
    found = await engine.recall("s", "billing", limit=2, lanes=("text",))
    assert [item.text for item in found.items] == [exact, relative]
    assert found.prefixes == {"added": ["bill"], "applied": True, "exact_forms": True}
    [event] = await engine.events.query("s", kind="recall")
    assert event.payload["prefixes"] == {"added": 1, "applied": True, "exact_forms": True}
    engine.lexical_exact_forms = False
    plain = await engine.recall("s", "billing", limit=2, lanes=("text",))
    assert [item.text for item in plain.items] == [relative, exact]
    assert plain.prefixes is not None and plain.prefixes["exact_forms"] is False


def test_the_exact_forms_flag_is_a_boolean():
    assert MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).lexical_exact_forms is False
    with pytest.raises(InvalidInput, match="lexical_exact_forms"):
        MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), lexical_exact_forms=1)  # type: ignore[arg-type]
