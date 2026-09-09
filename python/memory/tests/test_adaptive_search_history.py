"""Search history must describe actual scoped work without becoming evidence."""
import asyncio
import json

import httpx
import pytest

from scone_memory.core.models import RecallResult
from scone_memory.core.ports import NewFact
from scone_memory.retrieval.adaptive import AdaptiveLimits, AdaptiveRetriever, EvidenceDecision
from scone_memory.retrieval.recall_scope import RecallScope


async def source_fact(engine, subject, predicate, value):
    quote = f'{subject} {predicate} {value}.'
    source = await engine.remember('alpha', quote, metadata={'project': 'cedar'})
    return await engine.documents.insert_fact(NewFact(space='alpha', subject=subject,
        predicate=predicate, object=value, quote=quote, source_episode_id=source.episode_id,
        valid_from='2020-01-01T00:00:00Z'))


async def test_history_records_empty_followup_and_remaining_budget(engine, monkeypatch):
    bridge = await source_fact(engine, 'Aster', 'routes to', 'Birch')
    answer = await source_fact(engine, 'Birch', 'stored in', 'Cedar')
    seen = []
    queries = []

    async def recall(space, query, **kwargs):
        assert space == 'alpha' and kwargs['where'] == {'project': 'cedar'}
        queries.append(query)
        if len(queries) == 1:
            return RecallResult(facts=[bridge])
        return RecallResult(facts=[answer]) if query == 'Birch archive' else RecallResult(degraded=['vector'])

    class Assessor:
        async def assess(self, question, candidates):
            raise AssertionError('history-aware assessment was bypassed')

        async def assess_with_context(self, question, candidates, context):
            seen.append(context)
            assert question == 'Where do Aster records end up?'
            if len(seen) < 3:
                return EvidenceDecision(status='insufficient', selected_ids=(f'fact:{bridge.fact_id}',),
                    followup_queries=('Birch database',) if len(seen) == 1 else ('Birch archive',))
            return EvidenceDecision(status='sufficient', selected_ids=tuple(c.id for c in candidates))

    monkeypatch.setattr(engine, 'recall', recall)
    result = await AdaptiveRetriever(engine, Assessor(), include_search_history=True,
        limits=AdaptiveLimits(max_queries=4)).retrieve('alpha', 'Where do Aster records end up?',
            scope=RecallScope.validated(where={'project': 'cedar'}))
    assert result.status == 'sufficient'
    assert {f.fact_id for f in result.recall.facts} == {bridge.fact_id, answer.fact_id}
    assert queries == ['Where do Aster records end up?', 'Birch database', 'Birch archive']
    assert [c.queries_remaining for c in seen] == [3, 2, 1]
    assert [c.rounds_remaining for c in seen] == [2, 1, 0]
    assert [len(c.searches) for c in seen] == [1, 2, 3]
    assert [(s.query, s.round_number, s.added_candidates, s.degraded) for s in seen[-1].searches] == [
        ('Where do Aster records end up?', 1, 1, False), ('Birch database', 2, 0, True),
        ('Birch archive', 3, 1, False)]
    # Raw search history is private assessment context, not public diagnostics.
    assert 'Birch database' not in result.model_dump_json()


def test_history_requires_a_compatible_assessor_before_retrieval():
    class Legacy:
        async def assess(self, question, candidates):
            return EvidenceDecision(status='insufficient', selected_ids=())

    with pytest.raises(ValueError, match='assess_with_context'):
        AdaptiveRetriever(None, Legacy(), include_search_history=True)


async def test_provider_receives_history_without_promoting_old_queries_to_evidence():
    from scone_memory.providers.evidence_assessor import SelfHostedEvidenceAssessor
    from scone_memory.retrieval.adaptive import EvidenceCandidate
    from scone_memory.retrieval.search_history import EvidenceAssessmentContext, EvidenceSearchAttempt

    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({
            'status': 'insufficient', 'selected_ids': ['chunk:1'], 'followup_queries': ['Birch archive']})},
            'finish_reason': 'stop'}]})

    assessor = SelfHostedEvidenceAssessor('http://127.0.0.1:11434/v1', 'fixture',
        transport=httpx.MockTransport(respond))
    context = EvidenceAssessmentContext(round_number=2, rounds_remaining=1, queries_remaining=1,
        searches=(EvidenceSearchAttempt(query='Aster storage', round_number=1, added_candidates=1, degraded=False),
                  EvidenceSearchAttempt(query='Birch database', round_number=2, added_candidates=0, degraded=True)))
    candidates = (EvidenceCandidate(id='chunk:1', episode_id=1, text='Aster routes to Birch.'),)
    decision = await assessor.assess_with_context('Where does Aster end up?', candidates, context)
    assert decision.followup_queries == ('Birch archive',)
    data = json.loads(requests[0]['messages'][1]['content'])
    assert data['search_history']['searches'][1] == {
        'query': 'Birch database', 'round_number': 2, 'added_candidates': 0, 'degraded': True}
    assert data['question'] == 'Where does Aster end up?'
    assert data['candidates'] == [candidates[0].model_dump(mode='json')]
    properties = requests[0]['response_format']['json_schema']['schema']['properties']
    assert properties['selected_ids']['items']['enum'] == ['chunk:1']
    assert properties['followup_queries']['maxItems'] == 1

    await assessor.assess('Where does Aster end up?', candidates)
    assert 'search_history' not in json.loads(requests[1]['messages'][1]['content'])
    assert len(requests) == 2


@pytest.mark.parametrize('mutate', ['negative', 'bool', 'duplicate', 'future'])
async def test_invalid_history_is_rejected_before_provider_request(mutate):
    from scone_memory.providers.evidence_assessor import SelfHostedEvidenceAssessor
    from scone_memory.retrieval.adaptive import EvidenceCandidate
    from scone_memory.retrieval.search_history import EvidenceAssessmentContext, EvidenceSearchAttempt

    def unexpected(request):
        pytest.fail('invalid history reached inference')

    assessor = SelfHostedEvidenceAssessor('http://localhost:11434/v1', 'fixture',
        transport=httpx.MockTransport(unexpected))
    attempt = EvidenceSearchAttempt(query='Cedar journal', round_number=1, added_candidates=1, degraded=False)
    context = EvidenceAssessmentContext(round_number=1, rounds_remaining=2, queries_remaining=2, searches=(attempt,))
    updates = {'negative': {'queries_remaining': -1}, 'bool': {'queries_remaining': True},
               'duplicate': {'searches': (attempt, attempt.model_copy(update={'query': 'Ｃｅｄａｒ　journal '}))},
               'future': {'searches': (attempt.model_copy(update={'round_number': 4}),)}}
    invalid = context.model_copy(update=updates[mutate])
    with pytest.raises(ValueError):
        await assessor.assess_with_context('Where?', (EvidenceCandidate(id='chunk:1', episode_id=1, text='Source.'),), invalid)


async def test_adapter_cannot_mutate_history_or_restore_deleted_evidence(engine, monkeypatch):
    bridge = await source_fact(engine, 'Aster', 'routes to', 'Birch')
    answer = await source_fact(engine, 'Birch', 'stored in', 'Cedar')
    contexts = []

    async def recall(space, query, **kwargs):
        return RecallResult(facts=[bridge] if query == 'Aster route' else [answer])

    class Assessor:
        async def assess(self, *args):
            pytest.fail('wrong assessment interface')

        async def assess_with_context(self, question, candidates, context):
            contexts.append(context)
            if len(contexts) == 1:
                context.searches[0].__dict__['query'] = 'poisoned history'
                context.__dict__['queries_remaining'] = 0
                return EvidenceDecision(status='insufficient', selected_ids=(f'fact:{bridge.fact_id}',),
                    followup_queries=('Birch archive',))
            assert [s.query for s in context.searches] == ['Aster route', 'Birch archive']
            assert context.queries_remaining == 1
            await engine.forget('alpha', bridge.source_episode_id)
            return EvidenceDecision(status='sufficient', selected_ids=tuple(c.id for c in candidates))

    monkeypatch.setattr(engine, 'recall', recall)
    result = await AdaptiveRetriever(engine, Assessor(), include_search_history=True,
        limits=AdaptiveLimits(max_queries=3)).retrieve('alpha', 'Aster route', scope=RecallScope.validated())
    assert result.status == 'uncertain'
    assert [f.fact_id for f in result.recall.facts] == [answer.fact_id]
    assert 'stale_evidence' in result.reasons


async def test_history_is_run_local_when_assessor_is_shared(engine, monkeypatch):
    evidence = await source_fact(engine, 'Aster', 'routes to', 'Birch')
    arrived = asyncio.Event()
    count = 0
    histories = []

    async def recall(*args, **kwargs):
        return RecallResult(facts=[evidence])

    class Assessor:
        async def assess(self, *args):
            pytest.fail('wrong assessment interface')

        async def assess_with_context(self, question, candidates, context):
            nonlocal count
            count += 1
            if count == 2:
                arrived.set()
            await asyncio.wait_for(arrived.wait(), timeout=1)
            histories.append((question, tuple(s.query for s in context.searches)))
            return EvidenceDecision(status='sufficient', selected_ids=tuple(c.id for c in candidates))

    monkeypatch.setattr(engine, 'recall', recall)
    strategy = AdaptiveRetriever(engine, Assessor(), include_search_history=True)
    results = await asyncio.gather(*(strategy.retrieve('alpha', q, scope=RecallScope.validated())
                                    for q in ('Aster route', 'Birch archive')))
    assert all(r.status == 'sufficient' and r.queries_used == 1 for r in results)
    assert sorted(histories) == [('Aster route', ('Aster route',)), ('Birch archive', ('Birch archive',))]


async def test_contextual_assessment_cancellation_propagates(engine, monkeypatch):
    evidence = await source_fact(engine, 'Aster', 'routes to', 'Birch')
    entered = asyncio.Event()

    async def recall(*args, **kwargs):
        return RecallResult(facts=[evidence])

    class Assessor:
        async def assess(self, *args):
            pytest.fail('wrong assessment interface')

        async def assess_with_context(self, *args):
            entered.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(engine, 'recall', recall)
    task = asyncio.create_task(AdaptiveRetriever(engine, Assessor(), include_search_history=True).retrieve(
        'alpha', 'Aster route', scope=RecallScope.validated()))
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_zero_remaining_rounds_disallow_model_followups():
    from scone_memory.providers.evidence_assessor import SelfHostedEvidenceAssessor, EvidenceAssessmentError
    from scone_memory.retrieval.adaptive import EvidenceCandidate
    from scone_memory.retrieval.search_history import EvidenceAssessmentContext, EvidenceSearchAttempt

    def respond(request):
        body = json.loads(request.content)
        assert body['response_format']['json_schema']['schema']['properties']['followup_queries']['maxItems'] == 0
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({
            'status': 'insufficient', 'selected_ids': [], 'followup_queries': ['extra search']})},
            'finish_reason': 'stop'}]})

    assessor = SelfHostedEvidenceAssessor('http://localhost:11434/v1', 'fixture', transport=httpx.MockTransport(respond))
    context = EvidenceAssessmentContext(round_number=1, rounds_remaining=0, queries_remaining=3,
        searches=(EvidenceSearchAttempt(query='original', round_number=1, added_candidates=1, degraded=False),))
    with pytest.raises(EvidenceAssessmentError) as error:
        await assessor.assess_with_context('original', (EvidenceCandidate(id='chunk:1', episode_id=1, text='Source.'),), context)
    assert error.value.reason == 'invalid_assessment'


@pytest.mark.parametrize('flag', [None, 1, 'true'])
def test_search_history_requires_an_actual_boolean(flag):
    with pytest.raises(ValueError, match='include_search_history'):
        AdaptiveRetriever(None, None, include_search_history=flag)


async def test_default_uses_legacy_assessment_even_when_context_is_supported(engine, monkeypatch):
    evidence = await source_fact(engine, 'Aster', 'routes to', 'Birch')

    async def recall(*args, **kwargs):
        return RecallResult(facts=[evidence])

    class Assessor:
        async def assess(self, question, candidates):
            return EvidenceDecision(status='sufficient', selected_ids=tuple(c.id for c in candidates))

        async def assess_with_context(self, *args):
            pytest.fail('history must be opt-in')

    monkeypatch.setattr(engine, 'recall', recall)
    result = await AdaptiveRetriever(engine, Assessor()).retrieve('alpha', 'Aster route', scope=RecallScope.validated())
    assert result.status == 'sufficient' and [f.fact_id for f in result.recall.facts] == [evidence.fact_id]


async def test_serialized_history_budget_is_checked_before_inference():
    from scone_memory.providers.evidence_assessor import SelfHostedEvidenceAssessor
    from scone_memory.retrieval.adaptive import EvidenceCandidate
    from scone_memory.retrieval.search_history import EvidenceAssessmentContext, EvidenceSearchAttempt

    def unexpected(request):
        pytest.fail('oversized history reached inference')

    context = EvidenceAssessmentContext(round_number=1, rounds_remaining=0, queries_remaining=0,
        searches=tuple(EvidenceSearchAttempt(query='\x00' * 998 + str(i), round_number=1,
                       added_candidates=0, degraded=False) for i in range(12)))
    assessor = SelfHostedEvidenceAssessor('http://localhost:11434/v1', 'fixture', transport=httpx.MockTransport(unexpected))
    with pytest.raises(ValueError, match='64000 UTF-8 bytes'):
        await assessor.assess_with_context('original', (EvidenceCandidate(id='chunk:1', episode_id=1, text='Source.'),), context)
