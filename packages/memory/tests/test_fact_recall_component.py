"""Fact retrieval requires only a document store, without engine construction."""

from scone_memory.backends.memory import InMemoryDocumentStore
from scone_memory.core.ports import NewFact, TextFilter


async def test_manual_fact_retrieval_needs_no_vector_index_or_embedder():
    from scone_memory.retrieval.fact_recall import facts_for_query

    store = InMemoryDocumentStore()
    retained = await store.insert_fact(NewFact(
        space="alpha", subject="Juniper", predicate="calibration", object="Polaris",
        valid_from="2026-01-01T00:00:00Z",
    ))
    await store.insert_fact(NewFact(
        space="beta", subject="Juniper", predicate="calibration", object="Private",
        valid_from="2026-01-01T00:00:00Z",
    ))

    assert await facts_for_query(store, "alpha", "Juniper calibration", "2026-09-01T00:00:00Z") == [retained]
    # A manual fact without a source cannot establish a source-level scope.
    assert await facts_for_query(store, "alpha", "Juniper calibration", "2026-09-01T00:00:00Z",
                                 scope=TextFilter(kind="file")) == []
