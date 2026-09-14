"""Table querying over HTTP: the same exact answer, the same quoted cells, the same refusals."""

from __future__ import annotations

import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.ingestion.files import ingest_document

CSV = b'region,revenue\r\nWest,"1,250.50"\r\nEast,35\r\nWest,20\r\n'


@pytest.fixture
async def host():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    app = create_app(memory, {'write': 'alpha', 'other': 'beta'})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://scone.test',
                                headers={'authorization': 'Bearer write'}) as client:
        saved = await ingest_document(memory, 'alpha', CSV, filename='sales.csv')
        yield client, saved.added.episode_id
    await memory.close()


async def test_tables_are_listed_and_queried_with_every_cell_quoted(host):
    client, episode_id = host
    listed = await client.get(f'/v1/episodes/{episode_id}/tables')
    assert listed.status_code == 200, listed.text
    assert listed.json()['tables'] == [{'locator': 'delimited', 'name': 'delimited', 'columns': ['region', 'revenue'],
                                        'rows': 3, 'totals_rows_excluded': 0, 'basis': 'delimited_columns'}]
    asked = await client.post(f'/v1/episodes/{episode_id}/tables/query',
                              json={'operation': 'sum', 'column': 'revenue',
                                    'where': [{'column': 'region', 'op': '==', 'value': 'West'}]})
    assert asked.status_code == 200, asked.text
    record = asked.json()
    assert record['value'] == '1270.5' and record['rows_matched'] == 2 and record['rows_total'] == 3
    assert [c['text'] for c in record['cells']] == ['1,250.50', '20'] and record['verified_accuracy'] is False
    episode = await client.get(f'/v1/episodes/{episode_id}')
    content = episode.json()['content'].encode()
    for cell in record['cells']:
        assert content[cell['start']:cell['end']] == cell['text'].encode(), 'the quote is where the record says'


async def test_refusals_are_422_with_the_reason(host):
    client, episode_id = host
    refused = await client.post(f'/v1/episodes/{episode_id}/tables/query', json={'operation': 'sum', 'column': 'profit'})
    assert refused.status_code == 422 and 'column_not_found' in refused.json()['error']
    malformed = await client.post(f'/v1/episodes/{episode_id}/tables/query', json={'operation': 'total'})
    assert malformed.status_code == 422 and 'invalid' in malformed.json()['error']


async def test_another_space_cannot_see_the_table(host):
    client, episode_id = host
    other = await client.get(f'/v1/episodes/{episode_id}/tables', headers={'authorization': 'Bearer other'})
    assert other.status_code == 404
