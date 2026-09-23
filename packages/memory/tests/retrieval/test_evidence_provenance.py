"""Author/source changes must reach judgments and invalidate saved decisions."""
import asyncio
import json

import httpx

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.providers.typesafe_evidence import TypeSafeEvidenceAssessor
from scone_memory.retrieval.adaptive import AdaptiveRetriever, EvidenceDecision
from scone_memory.retrieval.decision_memory import DecisionMemory, RememberedEvidenceAssessor
from scone_memory.retrieval.recall_scope import RecallScope


async def test_current_session_cannot_crowd_sources_out_of_candidate_window():
    from scone_memory.retrieval.adaptive import AdaptiveLimits
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        wanted = await memory.remember('alpha', 'Lyra launch owner is Leo.', source='brief://lyra')
        for index in range(20):
            await memory.remember('alpha', f'Lyra launch owner? Request {index}.', source='current',
                metadata={'session_id': 'current', 'role': 'user'})
        class Assessor:
            async def assess(self, question, candidates):
                assert {c.episode_id for c in candidates} == {wanted.episode_id}
                return EvidenceDecision(status='sufficient', selected_ids=tuple(c.id for c in candidates))
        result = await AdaptiveRetriever(memory, Assessor(), limits=AdaptiveLimits(candidate_limit=1)).retrieve(
            'alpha', 'Lyra launch owner?', scope=RecallScope.validated(), exclude_session_id='current')
        assert result.status == 'sufficient'
        assert result.recall.items[0].episode_id == wanted.episode_id
    finally:
        await memory.close()


async def test_slow_query_embedding_keeps_scoped_local_evidence_available():
    from scone_memory.retrieval.adaptive import AdaptiveLimits
    cancelled = asyncio.Event()
    class SlowQueryEmbedder(HashEmbedder):
        async def embed_queries(self, texts):
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.set()
            return await self.embed(texts)
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), SlowQueryEmbedder()).open()
    try:
        wanted = await memory.remember('alpha', 'Lyra launch owner is Leo.', source='brief://lyra', metadata={'team': 'one'})
        await memory.remember('alpha', 'Lyra launch owner is Maya.', source='brief://private', metadata={'team': 'two'})
        class Assessor:
            async def assess(self, question, candidates):
                assert {c.episode_id for c in candidates} == {wanted.episode_id}
                return EvidenceDecision(status='sufficient', selected_ids=tuple(c.id for c in candidates))
        result = await AdaptiveRetriever(memory, Assessor(), limits=AdaptiveLimits(timeout_s=1.0),
            lexical_fallback=True).retrieve('alpha', 'Lyra launch owner', scope=RecallScope.validated(where={'team': 'one'}))
        assert result.status == 'sufficient'
        assert result.recall.items[0].episode_id == wanted.episode_id
        assert 'lexical_fallback' in result.reasons
        assert cancelled.is_set()
    finally:
        await memory.close()


async def test_retained_source_provenance_reaches_the_assessor():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        episode = await memory.remember('alpha', 'Polaris launch owner is Maya.', source='brief://polaris',
            metadata={'role': 'user', 'project': 'Polaris', 'private_extra': 'do not copy'})
        seen = []
        class Assessor:
            async def assess(self, question, candidates):
                seen.extend(candidates)
                return EvidenceDecision(status='sufficient', selected_ids=tuple(c.id for c in candidates))
        await AdaptiveRetriever(memory, Assessor()).retrieve('alpha', 'Polaris launch owner', scope=RecallScope.validated())
        assert seen
        candidate = next(c.model_dump() for c in seen if c.episode_id == episode.episode_id)
        assert candidate.get('source') == 'brief://polaris'
        assert candidate.get('role') == 'user'
        retained = await memory.documents.get_episode('alpha', episode.episode_id)
        assert candidate.get('created_at') == retained.created_at
        assert candidate.get('kind') == 'note'
        assert 'private_extra' not in json.dumps(candidate)
    finally:
        await memory.close()


async def test_role_only_change_invalidates_a_persisted_judgment(tmp_path):
    from scone_memory.retrieval.adaptive import EvidenceCandidate
    calls = []
    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={'model': 'jev-test', 'usage': {'input_tokens': 2, 'output_tokens': 2},
            'answers': {key: {'type': 'noul', 'noul': .99} for key in body['questions']}})
    direct = TypeSafeEvidenceAssessor('https://api.typesafe.ai', 'jev-latest', api_key='test',
        transport=httpx.MockTransport(respond))
    store = DecisionMemory(tmp_path / 'decisions.db', key=bytes(range(32)))
    assessor = RememberedEvidenceAssessor(direct, store)
    first = EvidenceCandidate(id='chunk:1', episode_id=1, text='Maya owns Polaris.', source='brief://polaris', role='user')
    scope = RecallScope.validated()
    await assessor.assess_scoped('Who owns Polaris?', (first,), space='alpha', scope=scope)
    await assessor.assess_scoped('Who owns Polaris?', (first.model_copy(update={'role': 'assistant'}),), space='alpha', scope=scope)
    assert len(calls) == 2
    assert calls[1]['state']['candidates'][0]['role'] == 'assistant'
    assert store.history('alpha', 'Who owns Polaris?', scope).revisions[-1].reason == 'evidence_changed'
