"""Jev scores retained candidates without changing retrieval authority."""
import asyncio
import json

import httpx
import pytest

from scone_memory.retrieval.reranking import RerankCandidate, rerank_candidates


def candidate(cid, text):
    return RerankCandidate(cid, cid, text, None, '2026-09-19T00:00:00Z', 0.5, None, ())


async def test_jev_decisions_map_answers_to_original_candidates():
    from scone_memory.providers.jev import JevReranker
    requests = []
    def serve(request):
        requests.append(request)
        return httpx.Response(200, json={'model':'typesafe/jev-1.13-20260917',
            'answers':{'passage_0':{'type':'noul','noul':0.02},
                       'passage_1':{'type':'noul','noul':0.97}}})
    ranker = JevReranker(api_key='private-test-key', transport=httpx.MockTransport(serve))
    candidates = (candidate(30,'Cedar runs nightly.'), candidate(42,'Morgan maintains Cedar.'))
    result = await ranker.evaluate('Who maintains Cedar?', candidates)
    assert [(row.chunk_id,row.score) for row in result.scores] == [(30,0.02),(42,0.97)]
    assert result.model == 'typesafe/jev-1.13-20260917'
    assert len(requests) == 1
    assert str(requests[0].url) == 'https://openrouter.ai/api/alpha/decisions'
    assert requests[0].headers['Authorization'] == 'Bearer private-test-key'
    body = json.loads(requests[0].content)
    assert body['model'] == '~typesafe/jev-latest'
    assert 'messages' not in body
    assert body['state']['query'] == 'Who maintains Cedar?'
    assert body['state']['passages'] == [{'id':'passage_0','text':candidates[0].text},
                                       {'id':'passage_1','text':candidates[1].text}]
    # Question-map keys are not supplied to Jev's model; the instructions must identify the passage.
    assert 'passage_0' in body['questions']['passage_0']['instructions']
    assert 'passage_1' in body['questions']['passage_1']['instructions']


@pytest.mark.parametrize('answers', [
    {}, {'passage_0':{'type':'noul','noul':True}},
    {'passage_0':{'type':'noul','noul':1.1}},
    {'passage_0':{'type':'noul','noul':float('nan')}},
    {'passage_0':{'type':'choice','noul':0.9}},
    {'passage_0':{'type':'noul','noul':'0.9'}},
    {'foreign':{'type':'noul','noul':0.9}},
    {'passage_0':{'type':'noul','noul':0.9},'extra':{'type':'noul','noul':0.1}},
])
async def test_invalid_decisions_preserve_baseline(answers):
    from scone_memory.providers.jev import JevReranker
    ranker = JevReranker(api_key='test', transport=httpx.MockTransport(lambda request:
        httpx.Response(200, content=json.dumps({'model':'typesafe/jev-1.13','answers':answers}))))
    result = await rerank_candidates(ranker, 'Cedar?', (candidate(30,'Cedar'),),
                                    limit=4, max_bytes=4000, timeout=1)
    assert result.trace.status == 'failed'
    assert result.ordered_ids == () and result.scores == {}


@pytest.mark.parametrize('status', [302,401,429,500])
async def test_http_failure_never_retries_or_leaks_provider_body(status):
    from scone_memory.providers.jev import JevReranker
    calls = []
    def serve(request):
        calls.append(request)
        return httpx.Response(status, headers={'Location':'https://other.example/'},
                              text='PRIVATE_PROVIDER_BODY')
    ranker = JevReranker(api_key='test', transport=httpx.MockTransport(serve))
    with pytest.raises(RuntimeError, match='Jev decision service unavailable') as error:
        await ranker.rerank('Cedar?', (candidate(30,'Cedar'),))
    assert 'PRIVATE' not in str(error.value)
    assert len(calls) == 1


async def test_empty_candidates_skip_inference_and_cancellation_propagates():
    from scone_memory.providers.jev import JevReranker
    entered = asyncio.Event()
    async def serve(request):
        entered.set()
        await asyncio.Event().wait()
    ranker = JevReranker(api_key='test', transport=httpx.MockTransport(serve))
    assert await ranker.rerank('Cedar?', ()) == ()
    assert not entered.is_set()
    task = asyncio.create_task(ranker.rerank('Cedar?', (candidate(30,'Cedar'),)))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_served_factory_uses_explicit_environment_key_reference(monkeypatch):
    from scone_memory.providers.jev import JevReranker
    from scone_memory.runtime.config import Settings, build_reranker
    monkeypatch.setenv('SCONE_JEV_API_KEY_ENV','TEST_JEV_KEY')
    monkeypatch.setenv('TEST_JEV_KEY','private-test-key')
    settings = Settings.from_env({'SCONE_RERANKER_FACTORY':'scone_memory.providers.jev:create_reranker'})
    assert isinstance(build_reranker(settings),JevReranker)


async def test_recall_api_filters_scope_before_jev_and_exposes_scores():
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.api import create_app
    from scone_memory.providers.jev import JevReranker
    requests = []
    def serve(request):
        body = json.loads(request.content)
        requests.append(body)
        assert 'PRIVATE_' not in json.dumps(body)
        return httpx.Response(200,json={'model':'typesafe/jev-1.13', 'answers':{
            row['id']:{'type':'noul','noul':0.98 if 'Morgan' in row['text'] else 0.03}
            for row in body['state']['passages']}})
    engine = await MemoryEngine(InMemoryDocumentStore(),InMemoryVectorIndex(),HashEmbedder(),
        reranker=JevReranker(api_key='test',transport=httpx.MockTransport(serve))).open()
    try:
        await engine.remember('alpha','Cedar ownership overview.',metadata={'team':'blue'})
        answer = await engine.remember('alpha','Morgan maintains Cedar.',metadata={'team':'blue'})
        await engine.remember('alpha','PRIVATE_TEAM maintains Cedar.',metadata={'team':'red'})
        await engine.remember('foreign','PRIVATE_SPACE maintains Cedar.',metadata={'team':'blue'})
        forgotten = await engine.remember('alpha','PRIVATE_FORGOTTEN maintains Cedar.',metadata={'team':'blue'})
        await engine.forget('alpha',forgotten.episode_id)
        app = create_app(engine,{'test-key':'alpha'})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test',
                                     headers={'Authorization':'Bearer test-key'}) as client:
            response = await client.get('/v1/recall',params={'q':'Who maintains Cedar?',
                'where':'team:blue','limit':1,'candidate_limit':12})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result['rerank']['status'] == 'applied'
        assert result['items'][0]['episode_id'] == answer.episode_id
        assert result['items'][0]['rerank_score'] == 0.98
        assert len(requests) == 1
    finally:
        await engine.close()
