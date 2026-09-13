"""Suggested questions ride inside the optional graph analysis and name their evidence."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app


@pytest.fixture
async def contradicted():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    facts = []
    for owner, text in [('public', 'Aster uses Beacon.'), ('public', 'Aster shuns Beacon.'), ('private', 'Aster owns Secret.')]:
        source = await engine.remember('alpha', text, metadata={'owner': owner}, source=f'{owner}/note')
        facts.append(await engine.assert_fact('alpha', *text.rstrip('.').split(), source_episode_id=source.episode_id, quote=text))
    await engine.link_facts('alpha', facts[0].fact_id, facts[1].fact_id, 'contradicts')
    await engine.link_facts('alpha', facts[0].fact_id, facts[2].fact_id, 'contradicts')
    yield engine, facts
    await engine.close()


async def test_questions_ride_with_the_analysis_and_name_their_evidence(contradicted):
    engine, facts = contradicted
    async with AsyncClient(transport=ASGITransport(app=create_app(engine, {'reader': 'alpha'})), base_url='http://fixture') as client:
        auth = {'authorization': 'Bearer reader'}
        params = {'q': 'aster', 'where': 'owner:public'}
        plain = await client.get('/v1/recall', params=params, headers=auth)
        assert 'graph_analysis' not in plain.json()
        enriched = await client.get('/v1/recall', params={**params, 'graph_analysis': 'true'}, headers=auth)
        questions = enriched.json()['graph_analysis']['questions']
        assert questions['analysis_status'] == 'complete' and questions['truncated'] is False and questions['omitted'] == 0
        asked = [q for q in questions['questions'] if q['type'] == 'contradiction']
        assert len(asked) == 1, questions
        assert 'uses Beacon' in asked[0]['question'] and 'shuns Beacon' in asked[0]['question']
        assert asked[0]['question'].endswith('?') and asked[0]['why']
        evidence = asked[0]['evidence']
        assert evidence['source'] == f'claim:{facts[0].fact_id}' and evidence['target'] == f'claim:{facts[1].fact_id}'
        assert evidence['edge_kind'] == 'contradicts' and isinstance(evidence['provenance_status'], str)
        # The private claim is contradicted too, but it is outside the caller's scope: no question may name it.
        assert 'Secret' not in enriched.text
        capabilities = await client.get('/v1/capabilities', headers=auth)
        assert capabilities.json()['features']['recall.graph_questions'] is True


@pytest.mark.parametrize('error', [TimeoutError('private timeout'), RuntimeError('private provider detail')])
async def test_an_unavailable_graph_yields_no_questions_and_says_why(contradicted, monkeypatch, error):
    engine, _ = contradicted
    from scone_memory.retrieval import evidence_graph
    monkeypatch.setattr(evidence_graph, 'build_query_evidence_graph', AsyncMock(side_effect=error))
    async with AsyncClient(transport=ASGITransport(app=create_app(engine, {'reader': 'alpha'})), base_url='http://fixture') as client:
        response = await client.get('/v1/recall', params={'q': 'aster', 'where': 'owner:public', 'graph_analysis': 'true'}, headers={'authorization': 'Bearer reader'})
    assert response.status_code == 200 and response.json()['items']
    questions = response.json()['graph_analysis']['questions']
    assert questions == {'questions': [], 'omitted': 0, 'truncated': False, 'analysis_status': 'unavailable',
                         'analysis_method': 'unlinked', 'reason': 'evidence_graph_unavailable'}
    assert 'private' not in response.text


async def test_a_failure_inside_question_generation_is_disclosed_not_leaked(contradicted, monkeypatch):
    engine, _ = contradicted
    from scone_memory.retrieval import graph_questions
    monkeypatch.setattr(graph_questions, 'questions_for', lambda graph, analysis: (_ for _ in ()).throw(RuntimeError('private detail')))
    async with AsyncClient(transport=ASGITransport(app=create_app(engine, {'reader': 'alpha'})), base_url='http://fixture') as client:
        response = await client.get('/v1/recall', params={'q': 'aster', 'where': 'owner:public', 'graph_analysis': 'true'}, headers={'authorization': 'Bearer reader'})
    assert response.status_code == 200
    analysis = response.json()['graph_analysis']
    assert analysis['status'] == 'complete', analysis['status']
    assert analysis['questions']['analysis_status'] == 'unavailable' and analysis['questions']['reason'] == 'questions_unavailable'
    assert 'private' not in response.text
