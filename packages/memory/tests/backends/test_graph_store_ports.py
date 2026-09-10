"""Bounded graph candidates and point reads agree across document backends."""
from unittest.mock import AsyncMock

import pytest

from scone_memory.retrieval.multihop import BoundedIncidentLinks, BoundedSubjectFacts, PointFactLinks, expand_multihop
from scone_memory.core.models import RecallResult
from scone_memory.core.ports import TextFilter
from ..retrieval.test_multihop_retrieval import chain, fact, link


async def test_graph_store_ports_preserve_exact_subject_order_scope_and_link_identity(engine):
    store = engine.documents
    assert isinstance(store, BoundedIncidentLinks) and isinstance(store, BoundedSubjectFacts)
    assert isinstance(store, PointFactLinks)
    facts = await chain(engine)
    incoming = await link(engine, facts[2], facts[0])
    outgoing = await link(engine, facts[0], facts[2], 'contradicts')
    self_link = await link(engine, facts[0], facts[0])
    assert await link(engine, facts[2], facts[0]) == incoming
    assert await store.fact_links_from('alpha', facts[0].fact_id, 1000) == [incoming, outgoing, self_link]
    assert await store.fact_links_from('alpha', facts[0].fact_id, 1) == [incoming]
    assert await store.fact_links_from('alpha', facts[0].fact_id, -1) == []
    assert await store.fact_links_from('foreign', facts[0].fact_id, 1000) == []
    assert await store.get_fact_link('alpha', incoming.link_id) == incoming
    assert await store.get_fact_link('foreign', incoming.link_id) is None
    assert await store.get_fact_link('alpha', incoming.link_id + 10000) is None
    other = await fact(engine, 'Beacon', 'is', 'newer')
    await fact(engine, 'Beacon x', 'is', 'partial match')
    await fact(engine, 'beacon', 'is', 'case mismatch')
    await fact(engine, 'Beacon', 'is', 'private', space='foreign')
    assert await store.facts_by_subject('alpha', 'Beacon', 1) == [facts[1]]
    assert await store.facts_by_subject('alpha', 'Beacon', 1000) == [facts[1], other]
    assert await store.facts_by_subject('alpha', 'Beacon', 0) == []


async def test_graph_store_expands_quoted_chain_without_unbounded_reads(engine):
    facts = await chain(engine, metadata={'team':'blue'})
    await fact(engine, 'Beacon', 'is', 'PRIVATE', metadata={'team':'red'})
    store = engine.documents
    store.list_facts = AsyncMock(side_effect=AssertionError('no global ledger scan'))
    store.fact_links = AsyncMock(side_effect=AssertionError('no unbounded adjacency scan'))
    result = await expand_multihop(store, 'alpha', seeds=RecallResult(facts=[facts[0]]), scope=TextFilter(where={'team':'blue'}))
    assert result.facts == facts
    assert result.paths[-1].fact_ids == [row.fact_id for row in facts]
    assert result.coverage.complete
    assert 'PRIVATE' not in result.model_dump_json()
