import asyncio
import json

import httpx
import pytest

from scone_memory.providers.typesafe_evidence import TypeSafeEvidenceAssessor
from scone_memory.retrieval.adaptive import EvidenceAssessmentError, EvidenceCandidate

CANDIDATES = (
    EvidenceCandidate(id='chunk:1', episode_id=1, text='Maya owns the Polaris launch.'),
    EvidenceCandidate(id='chunk:2', episode_id=2, text='Please find out who owns the Polaris launch.'),
)


def payload():
    return {'model': 'jev-1.13.0', 'usage': {'input_tokens': 100, 'output_tokens': 12},
        'answers': {name: {'type': 'noul', 'noul': value} for name, value in
            [('sufficient', .99), ('candidate_0', .98), ('candidate_1', .02)]}}


def assessor(handler):
    return TypeSafeEvidenceAssessor('https://api.typesafe.ai', 'jev-latest', api_key='test-secret',
        transport=httpx.MockTransport(handler))


async def test_batches_relevance_and_sufficiency_directly_and_logs_usage(caplog):
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(200, json=payload())
    with caplog.at_level('INFO'):
        decision = await assessor(respond).assess('Who owns Polaris?', CANDIDATES)
    assert decision.selected_ids == ('chunk:1',) and decision.status == 'sufficient'
    assert len(calls) == 1 and str(calls[0].url) == 'https://api.typesafe.ai/v1/systemone'
    body = json.loads(calls[0].content)
    assert len(body['questions']) == 3 and body['model'] == 'jev-latest'
    assert 'candidates[1]' in body['questions']['candidate_1']['instructions']
    event = next(r for r in caplog.records if r.msg == 'typesafe_assessment.finished')
    assert event.total_tokens == 112 and event.model_name == 'jev-1.13.0'
    assert 'test-secret' not in caplog.text and 'Maya' not in caplog.text


@pytest.mark.parametrize('bad', ['missing', 'extra', 'nan', 'boolean', 'wrong_type'])
async def test_invalid_decisions_fail_without_exposing_provider_body(bad):
    data = payload()
    if bad == 'missing': del data['answers']['candidate_0']
    if bad == 'extra': data['answers']['invented'] = {'type': 'noul', 'noul': 1.0}
    if bad == 'nan': data['answers']['candidate_0']['noul'] = 'NaN'
    if bad == 'boolean': data['answers']['candidate_0']['noul'] = True
    if bad == 'wrong_type': data['answers']['candidate_0']['type'] = 'choice'
    with pytest.raises(EvidenceAssessmentError) as error:
        await assessor(lambda _: httpx.Response(200, json=data)).assess('Who owns Polaris?', CANDIDATES)
    assert error.value.reason == 'invalid_assessment'


async def test_uncertain_evidence_is_retained_without_sufficiency_claim():
    data = payload()
    for answer in data['answers'].values(): answer['noul'] = .5
    decision = await assessor(lambda _: httpx.Response(200, json=data)).assess('Who owns Polaris?', CANDIDATES)
    assert decision.status == 'uncertain' and decision.selected_ids == ('chunk:1', 'chunk:2')


async def test_empty_evidence_makes_no_request_and_cancellation_propagates():
    entered = asyncio.Event()
    async def wait(request):
        entered.set()
        await asyncio.Event().wait()
    selected = assessor(wait)
    assert (await selected.assess('Who owns Polaris?', ())).status == 'insufficient'
    assert not entered.is_set()
    task = asyncio.create_task(selected.assess('Who owns Polaris?', CANDIDATES))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
