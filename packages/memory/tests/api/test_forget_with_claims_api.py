"""Forgetting with dependent claims over HTTP: opt-in, disclosed, refused when malformed."""
import pytest
from httpx import ASGITransport, AsyncClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app


@pytest.fixture
async def prepared():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    source = await engine.remember('alpha', 'Alice works at Acme.')
    fact = await engine.assert_fact('alpha', 'Alice', 'works_at', 'Acme', source_episode_id=source.episode_id, quote='Alice works at Acme.')
    yield engine, source.episode_id, fact.fact_id
    await engine.close()


async def test_default_delete_leaves_claims_and_says_so(prepared):
    engine, episode_id, fact_id = prepared
    async with AsyncClient(transport=ASGITransport(app=create_app(engine, {'k': 'alpha'})), base_url='http://fixture') as client:
        response = await client.delete(f'/v1/episodes/{episode_id}', headers={'authorization': 'Bearer k'})
    body = response.json()
    assert response.status_code == 200 and body['forgotten'] == episode_id
    assert body['claims_policy'] == 'keep' and body['facts_citing'] == [fact_id] and body['claims_excluded'] == []
    assert not (await engine.documents.get_fact('alpha', fact_id)).excluded


async def test_with_claims_exclude_is_opt_in_and_disclosed(prepared):
    engine, episode_id, fact_id = prepared
    async with AsyncClient(transport=ASGITransport(app=create_app(engine, {'k': 'alpha'})), base_url='http://fixture') as client:
        response = await client.delete(f'/v1/episodes/{episode_id}', params={'with_claims': 'exclude'}, headers={'authorization': 'Bearer k'})
        body = response.json()
        assert response.status_code == 200 and body['claims_policy'] == 'exclude' and body['claims_excluded'] == [fact_id]
        facts = (await client.get('/v1/facts', headers={'authorization': 'Bearer k'})).json()
        rows = facts['facts'] if isinstance(facts, dict) else facts
        assert [row['fact_id'] for row in rows] == []


async def test_a_malformed_policy_is_refused_at_the_boundary_before_the_engine_is_asked(prepared, monkeypatch):
    """The engine refuses a bad policy too; the route must refuse it first,
    so a malformed request never reaches the store at all."""
    engine, episode_id, fact_id = prepared
    asked = []

    async def forget(*args, **kwargs):
        asked.append((args, kwargs))
        raise AssertionError("the engine must not be asked")

    monkeypatch.setattr(engine, 'forget', forget)
    async with AsyncClient(transport=ASGITransport(app=create_app(engine, {'k': 'alpha'})), base_url='http://fixture') as client:
        response = await client.delete(f'/v1/episodes/{episode_id}', params={'with_claims': 'erase'}, headers={'authorization': 'Bearer k'})
    assert response.status_code == 422 and asked == []
    assert await engine.documents.get_episode('alpha', episode_id) is not None
