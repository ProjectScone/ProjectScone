"""The narrowing disclosure reaches the HTTP recall response."""
import json

import pytest
from httpx import ASGITransport, AsyncClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app


@pytest.fixture
async def prepared():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    for n in range(3):
        await engine.remember('alpha', f'draft plan {n} for the launch', metadata={'status': 'draft'}, source=f'drafts/{n}')
    await engine.remember('alpha', 'the launch plan was published', metadata={'status': 'published'}, source='release/plan')
    yield engine
    await engine.close()


async def test_a_narrowed_recall_says_how_it_narrowed(prepared):
    async with AsyncClient(transport=ASGITransport(app=create_app(prepared, {'k': 'alpha'})), base_url='http://fixture') as client:
        auth = {'authorization': 'Bearer k'}
        plain = (await client.get('/v1/recall', params={'q': 'launch plan'}, headers=auth)).json()
        assert plain['narrowing'] is None
        narrowed = (await client.get('/v1/recall', params={'q': 'launch plan', 'conditions': json.dumps({'field': 'status', 'is': 'published'})}, headers=auth)).json()
        assert [item['source'] for item in narrowed['items']] == ['release/plan']
        report = narrowed['narrowing']
        assert report['conditions'] is True and report['vector_lane'] == 'in_store' and report['text_lane'] == 'in_store'
        assert report['window_exhausted'] is False and report['postfiltered_out'] == 0
        by_source = (await client.get('/v1/recall', params={'q': 'launch plan', 'source_prefix': 'release/'}, headers=auth)).json()
        assert by_source['narrowing']['kind_or_source_or_dates'] is True and by_source['narrowing']['vector_lane'] == 'postfiltered'
