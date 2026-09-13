"""Space-wide link inventory includes each scoped relationship once."""
import pytest
from unittest.mock import AsyncMock, Mock

from scone_memory.core.ports import ArchiveLinkInventory, NewFactLink
from .test_space_deletion_backend_requests import adapter

NOW = '2026-09-12T00:00:00Z'


async def test_space_inventory_is_scoped_ordered_and_survives_neighbour_deletion(engine):
    assert isinstance(engine.documents, ArchiveLinkInventory)
    expected = []
    for space in ['alpha', 'bravo']:
        first = await engine.assert_fact(space, 'alice', 'keeps', 'source')
        second = await engine.assert_fact(space, 'bob', 'keeps', 'other')
        for kind in ['supports', 'contradicts']:
            link = await engine.documents.insert_fact_link(NewFactLink(
                space=space, from_fact=first.fact_id, to_fact=second.fact_id, kind=kind, created_at=NOW,
            ))
            if space == 'alpha':
                expected.append(link)
    assert await engine.documents.space_fact_links('alpha') == expected
    await engine.delete_space('bravo')
    assert await engine.documents.space_fact_links('alpha') == expected
    assert await engine.documents.space_fact_links('bravo') == []


async def test_postgres_uses_one_parameterized_space_query(monkeypatch):
    store = adapter('postgres', monkeypatch)
    assert await store.space_fact_links('alpha') == []
    sql, values = store._rows.call_args.args
    assert values == ('alpha',)
    assert 'WHERE space = %s ORDER BY id' in sql
    assert store._rows.await_count == 1


async def test_mongo_uses_one_scoped_space_cursor(monkeypatch):
    store = adapter('mongo', monkeypatch)
    cursor = AsyncMock()
    cursor.__aiter__.return_value = []
    store._fact_links = Mock()
    store._fact_links.find.return_value.sort.return_value = cursor
    assert await store.space_fact_links('alpha') == []
    assert store._fact_links.find.call_args.args == ({'space': 'alpha'},)
    assert store._fact_links.find.return_value.sort.call_args.args == ('_id', 1)
