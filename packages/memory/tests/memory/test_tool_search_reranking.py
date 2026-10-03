import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.integrations.scoped_tools import ScopedMemoryTools
from scone_memory.retrieval.recall_scope import RecallScope
from scone_memory.retrieval.reranking import RerankScore
from scone_memory.retrieval.listwise import ListwiseReranker


async def test_tool_reranker_can_reach_evidence_beyond_twenty_candidates():
    class FlatEmbedder:
        id = 'flat-test'
        dim = 2
        async def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), FlatEmbedder(),
                                candidate_limit=64, rerank_limit=32).open()
    seen = []
    try:
        for i in range(40):
            await engine.remember('alpha', f'Cedar maintenance record {i}.', kind='file',
                                  created_at='2026-09-08')
        baseline = await engine.recall('alpha', 'Cedar', limit=40, candidate_limit=64, rerank=False)
        assert len(baseline.items) == 40
        wanted = baseline.items[25].chunk_id

        class Ranker:
            async def rerank(self, query, candidates):
                seen.extend(item.chunk_id for item in candidates)
                return [RerankScore(item.chunk_id, 1.0 if item.chunk_id == wanted else 0.0)
                        for item in candidates]

        engine.reranker = Ranker()
        tools = ScopedMemoryTools(engine, 'alpha', scope=RecallScope.validated(kind='file'))
        result = await tools.run('search_memory', {'query': 'Cedar', 'limit': 1})
        assert wanted in seen, 'Tool pretruncation must not hide an in-budget rerank candidate'
        assert result['items'][0]['chunk_id'] == wanted
        assert len(result['items']) == 1
    finally:
        await engine.close()


async def test_tool_search_preserves_conversation_policy_to_skip_listwise_chat():
    calls = []

    class Chat:
        async def complete(self, system, user):
            calls.append(user)
            raise AssertionError('Tool search must not start long-running listwise chat')

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                reranker=ListwiseReranker(Chat())).open()
    try:
        await engine.remember('alpha', 'Cedar is maintained by Morgan.')
        await engine.remember('alpha', 'Cedar supports document search.')
        tools = ScopedMemoryTools(engine, 'alpha', scope=RecallScope.validated())
        result = await tools.run('search_memory', {'query': 'Cedar'})
        assert result['ok'] is True
        assert result['items']
        assert calls == []
        assert 'rerank' not in result
    finally:
        await engine.close()


@pytest.mark.parametrize('behavior',['rank','fail','delete'])
async def test_tool_search_ranks_only_scoped_retained_evidence_and_rechecks_afterward(behavior):
    engine = await MemoryEngine(InMemoryDocumentStore(),InMemoryVectorIndex(),HashEmbedder()).open()
    calls = []
    try:
        await engine.remember('alpha','Cedar Cedar Cedar maintainer overview.',metadata={'team':'blue'})
        answer = await engine.remember('alpha','Morgan maintains Cedar.',metadata={'team':'blue'})
        await engine.remember('alpha','PRIVATE_TEAM Cedar.',metadata={'team':'red'})
        await engine.remember('alpha','PRIVATE_SESSION Cedar.',metadata={'team':'blue','session_id':'current'})
        await engine.remember('foreign','PRIVATE_SPACE Cedar.',metadata={'team':'blue'})
        forgotten = await engine.remember('alpha','PRIVATE_FORGOTTEN Cedar.',metadata={'team':'blue'})
        await engine.forget('alpha',forgotten.episode_id)
        tools = ScopedMemoryTools(engine,'alpha',scope=RecallScope.validated(where={'team':'blue'}),
                                  exclude_session_id='current')
        baseline = await tools.run('search_memory',{'query':'Cedar','limit':1})

        class Ranker:
            async def rerank(self, query, candidates):
                calls.append(candidates)
                assert all('PRIVATE_' not in item.text for item in candidates)
                if behavior=='fail':
                    raise ValueError('PRIVATE_PROVIDER_ERROR')
                if behavior=='delete':
                    await engine.forget('alpha',answer.episode_id)
                return [RerankScore(item.chunk_id,1.0 if 'Morgan' in item.text else 0.0) for item in candidates]

        engine.reranker = Ranker()
        result = await tools.run('search_memory',{'query':'Cedar','limit':1})
        assert len(calls)==1
        assert 'PRIVATE_' not in json.dumps(result)
        if behavior=='delete':
            assert result['ok'] is False
            assert result['items']==[]
        elif behavior=='fail':
            assert result['items']==baseline['items']
            assert result['rerank']['status']=='failed'
        else:
            assert result['items'][0]['episode_id']==answer.episode_id
            assert result['rerank']['status']=='applied'
    finally:
        await engine.close()
