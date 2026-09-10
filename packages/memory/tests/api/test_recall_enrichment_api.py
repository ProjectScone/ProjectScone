"""Opt-in edge retrieval exposes verifiable source context without model calls."""

from ..paths import TESTS_ROOT
import json
from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient
from scone_memory import MemoryEngine, HashEmbedder
from scone_memory.backends.sqlite import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.api import create_app

AUTH={'authorization':'Bearer local-fixture'}

@pytest.fixture
async def prepared(tmp_path):
    path=tmp_path/'fixture.db'
    engine=await MemoryEngine(SqliteDocumentStore(path),SqliteVectorIndex(path),HashEmbedder(),
                              clock=lambda:'2026-09-01T00:00:00Z').open()
    fixture=json.loads((TESTS_ROOT/'fixtures/edge_rag/v1.json').read_text())
    sources={}
    for doc in fixture['documents']:
        sources[doc['id']]=await engine.remember(doc['space'],doc['content'],kind='file',
            source=doc['source'],metadata=doc['where'],tags=doc['tags'],created_at=doc['created_at'])
    for fact in fixture['facts']:
        await engine.assert_fact('fixture',fact['subject'],fact['predicate'],fact['object'],
            source_episode_id=sources[fact['document_id']].episode_id,quote=fact['quote'],valid_from=fact['valid_from'])
    yield engine
    await engine.close()

async def test_structural_context_returns_exact_retained_table_and_source_spans(prepared):
    async with AsyncClient(transport=ASGITransport(app=create_app(prepared,{'local-fixture':'fixture'})), base_url='http://fixture') as client:
        response=await client.get('/v1/recall',params={'q':'Beacon checkpoint protocol','limit':1,
            'where':'project:edge','source_prefix':'docs/beacon.md','structural_context':'true'},headers=AUTH)
    assert response.status_code==200
    context=response.json()['structural_context']
    assert context['sections']
    assert any('| uncertain | require reconciliation |' in section['text'] for section in context['sections'])
    for section in context['sections']:
        source=await prepared.episode('fixture',section['episode_id'])
        assert source.content.encode()[section['start']:section['end']].decode()==section['text']
        assert source.metadata['project']=='edge'

async def test_multihop_follows_recorded_chain_and_respects_source_scope(prepared):
    async with AsyncClient(transport=ASGITransport(app=create_app(prepared,{'local-fixture':'fixture'})), base_url='http://fixture') as client:
        response=await client.get('/v1/recall',params={'q':'beacon routes','limit':1,'multi_hop':'true',
            'where':'project:edge'},headers=AUTH)
    assert response.status_code==200
    graph=response.json()['multi_hop']
    assert {fact['subject'] for fact in graph['facts']}=={'beacon','cedar','ridge'}
    assert graph['paths']
    assert all(fact['quote'] for fact in graph['facts'])

async def test_default_wire_shape_is_unchanged(prepared):
    async with AsyncClient(transport=ASGITransport(app=create_app(prepared,{'local-fixture':'fixture'})), base_url='http://fixture') as client:
        response=await client.get('/v1/recall',params={'q':'beacon'},headers=AUTH)
        features=(await client.get('/v1/capabilities',headers=AUTH)).json()['features']
    assert 'structural_context' not in response.json() and 'multi_hop' not in response.json()
    assert features['recall.structural_context'] and features['recall.multi_hop']

@pytest.mark.parametrize('option,value',[('expansion_max_bytes',0),('expansion_max_bytes',262145),('max_hops',0),('max_hops',7)])
async def test_invalid_expansion_limits_are_rejected(prepared,option,value):
    async with AsyncClient(transport=ASGITransport(app=create_app(prepared,{'local-fixture':'fixture'})), base_url='http://fixture') as client:
        response=await client.get('/v1/recall',params={'q':'beacon',option:value},headers=AUTH)
    assert response.status_code==422
