"""Complete archive reads across search windows without live Elasticsearch."""
from unittest.mock import AsyncMock

import pytest

pytest.importorskip('elasticsearch')
from scone_memory.backends.elastic import ElasticsearchDocumentStore
from scone_memory.core.affirmations import Affirmation
from scone_memory.core.models import FactLink

NOW = '2026-09-12T00:00:00Z'


def row(kind, identity):
    if kind == 'fact_links':
        return FactLink(link_id=identity, space='alpha', from_fact=1, to_fact=identity+100,
                        kind='supports', created_at=NOW).model_dump()
    return Affirmation(affirmation_id=identity, space='alpha', fact_id=1, valid_from=NOW,
                       recorded_at=NOW).model_dump()


@pytest.mark.parametrize('kind', ['fact_links', 'fact_affirmations'])
async def test_all_record_contract_reads_past_first_window(monkeypatch, kind):
    client = AsyncMock()
    store = ElasticsearchDocumentStore(client=client, refresh=False)
    monkeypatch.setattr(store, 'PAGE', 2)
    field = 'link_id' if kind == 'fact_links' else 'affirmation_id'
    client.search.side_effect = [
        {'hits': {'hits': [{'_source': row(kind, i)} for i in [1, 2]]}},
        {'hits': {'hits': [{'_source': row(kind, 3)}]}},
    ]
    values = await store.fact_links('alpha', 1) if kind == 'fact_links' else await store.space_affirmations('alpha')
    assert [getattr(value, field) for value in values] == [1, 2, 3]
    for call in client.search.call_args_list:
        assert call.kwargs['size'] == 2
        assert call.kwargs['index'] == store._idx(kind)
    assert {'range': {field: {'gt': 2}}} in client.search.call_args.kwargs['query']['bool']['filter']


async def test_archive_visibility_refreshes_all_four_source_indexes():
    client = AsyncMock()
    store = ElasticsearchDocumentStore(client=client, refresh=False)
    await store.prepare_archive_read('alpha')
    client.indices.refresh.assert_awaited_once_with(index=','.join(store._idx(name) for name in (
        'episodes', 'facts', 'fact_links', 'fact_affirmations',
    )))
    assert store.shared.refresh is False


@pytest.mark.parametrize('incomplete', [{'timed_out': True}, {'_shards': {'failed': 1}}, {'terminated_early': True}])
async def test_incomplete_search_cannot_be_exported_as_complete(incomplete):
    client = AsyncMock()
    store = ElasticsearchDocumentStore(client=client)
    client.search.return_value = {'hits': {'hits': []}, **incomplete}
    with pytest.raises(RuntimeError, match='incomplete'):
        await store.space_affirmations('alpha')


async def test_repeated_archive_page_refuses_instead_of_looping(monkeypatch):
    client = AsyncMock()
    store = ElasticsearchDocumentStore(client=client)
    monkeypatch.setattr(store, 'PAGE', 1)
    client.search.return_value = {'hits': {'hits': [{'_source': row('fact_affirmations', 1)}]}}
    with pytest.raises(RuntimeError, match='inventory'):
        await store.space_affirmations('alpha')


async def test_space_links_uses_complete_inventory_without_endpoint_filter(monkeypatch):
    client = AsyncMock()
    store = ElasticsearchDocumentStore(client=client)
    monkeypatch.setattr(store, 'PAGE', 1)
    client.search.side_effect = [
        {'hits': {'hits': [{'_source': row('fact_links', 1)}]}},
        {'hits': {'hits': []}},
    ]
    assert len(await store.space_fact_links('alpha')) == 1
    assert client.search.call_args_list[0].kwargs['query'] == {'term': {'space': 'alpha'}}
    assert client.search.call_args.kwargs['query']['bool']['filter'] == [
        {'term': {'space': 'alpha'}}, {'range': {'link_id': {'gt': 1}}},
    ]
