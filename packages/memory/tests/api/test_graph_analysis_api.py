"""Graph analytics share the authorized recall graph and remain optional."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app


@pytest.fixture
async def prepared():
    engine=await MemoryEngine(InMemoryDocumentStore(),InMemoryVectorIndex(),HashEmbedder()).open()
    for space, owner, text in [('alpha','public','Aster uses Beacon.'),('alpha','private','Aster owns Secret.'),('beta','public','Aster hides OtherTenant.')]:
        source=await engine.remember(space,text,metadata={'owner':owner},source=f'{owner}/note')
        words=text.rstrip('.').split()
        await engine.assert_fact(space,*words,source_episode_id=source.episode_id,quote=text)
    yield engine
    await engine.close()


async def test_analysis_is_opt_in_scoped_and_reuses_one_evidence_graph(prepared,monkeypatch):
    from scone_memory.retrieval import evidence_graph
    reader=AsyncMock(wraps=evidence_graph.build_query_evidence_graph)
    monkeypatch.setattr(evidence_graph,'build_query_evidence_graph',reader)
    async with AsyncClient(transport=ASGITransport(app=create_app(prepared,{'reader':'alpha'})),base_url='http://fixture') as client:
        params={'q':'aster','where':'owner:public'}
        auth={'authorization':'Bearer reader'}
        baseline=await client.get('/v1/recall',params=params,headers=auth)
        assert 'graph_analysis' not in baseline.json() and reader.await_count==0
        enriched=await client.get('/v1/recall',params={**params,'graph_analysis':'true','evidence_graph':'true'},headers=auth)
        assert reader.await_count==1
        body=enriched.json()
        assert body['items']==baseline.json()['items']
        analysis=body['graph_analysis']
        assert analysis['status']=='complete'
        assert analysis['coverage']['recorded_relations']==1
        allowed={node['id'] for node in body['evidence_graph']['nodes'] if node['kind']=='concept'}
        assert {node for group in analysis['communities'] for node in group['node_ids']}==allowed
        assert 'Secret' not in enriched.text and 'OtherTenant' not in enriched.text
        standalone=await client.get('/v1/recall',params={**params,'graph_analysis':'true'},headers=auth)
        assert 'graph_analysis' in standalone.json() and 'evidence_graph' not in standalone.json()
        capabilities=await client.get('/v1/capabilities',headers=auth)
        assert capabilities.json()['features']['recall.graph_analysis'] is True
        assert (await client.get('/v1/recall',params={**params,'graph_analysis':'true'})).status_code==401
        assert (await client.get('/v1/recall',params={**params,'graph_analysis':'invalid'},headers=auth)).status_code==422


@pytest.mark.parametrize('error',[TimeoutError('private timeout'),RuntimeError('private provider detail')])
async def test_graph_failure_preserves_recall_and_returns_sanitized_unavailable_analysis(prepared,monkeypatch,error):
    from scone_memory.retrieval import evidence_graph
    monkeypatch.setattr(evidence_graph,'build_query_evidence_graph',AsyncMock(side_effect=error))
    async with AsyncClient(transport=ASGITransport(app=create_app(prepared,{'reader':'alpha'})),base_url='http://fixture') as client:
        response=await client.get('/v1/recall',params={'q':'aster','where':'owner:public','graph_analysis':'true'},headers={'authorization':'Bearer reader'})
    assert response.status_code==200 and response.json()['items']
    assert response.json()['graph_analysis']['status']=='unavailable'
    assert 'private' not in response.text


async def test_cancellation_of_graph_read_still_propagates(prepared,monkeypatch):
    from scone_memory.retrieval import evidence_graph
    monkeypatch.setattr(evidence_graph,'build_query_evidence_graph',AsyncMock(side_effect=asyncio.CancelledError))
    async with AsyncClient(transport=ASGITransport(app=create_app(prepared,{'reader':'alpha'})),base_url='http://fixture') as client:
        with pytest.raises(asyncio.CancelledError):
            await client.get('/v1/recall',params={'q':'aster','graph_analysis':'true'},headers={'authorization':'Bearer reader'})
