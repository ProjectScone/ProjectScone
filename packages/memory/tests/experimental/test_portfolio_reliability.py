import json

import httpx
import pytest

from research20.jev import JevResearchClient, Probe
from research20.reliability import calibrated_threshold, run, stable_approval
from research20.reliability_cases import calibration_cases, support_cases


def test_stability_gate_and_calibration_do_not_treat_uncertainty_as_approval():
    assert stable_approval((.9, .9, .9))
    assert not stable_approval((.99, .79, .99))
    with pytest.raises(ValueError):
        stable_approval((.9,))
    assert calibrated_threshold(((.9, True), (.7, False))) == .8
    assert calibrated_threshold(((.9, True), (1.0, False))) > 1
    with pytest.raises(ValueError):
        calibrated_threshold(((.9, True),))
    assert {case.name for case in calibration_cases()}.isdisjoint(case.name for case in support_cases())


async def test_all_five_live_experiments_serialize_without_gold_in_provider_payload():
    payloads = []
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        payloads.append(body)
        for question in body['questions'].values():
            sample = question['instructions']['sample']
            assert 'expected' not in sample and 'label' not in sample
        return httpx.Response(200, json={'model': 'test-model',
            'answers': {key: {'type': 'noul', 'noul': .25} for key in body['questions']},
            'usage': {'input_tokens': 5, 'output_tokens': 5}})
    client = JevResearchClient('https://api.typesafe.ai', 'jev-test', 'secret', httpx.MockTransport(handler))
    results = await run(client)
    assert [result.experiment_id for result in results] == list(range(16, 21))
    assert len(payloads) == 5
    assert all(result.cases == 12 and result.evidence_kind == 'live_jev' for result in results)
    assert 'secret' not in json.dumps(client.audits)
    assert 'secret' not in repr(client)


@pytest.mark.parametrize('mode', ['missing', 'extra', 'nan', 'bool', 'error'])
async def test_provider_failure_is_retained_without_fabricated_answers(mode):
    def handler(request: httpx.Request) -> httpx.Response:
        answers = {'p': {'type': 'noul', 'noul': .9}}
        if mode == 'missing':
            answers = {}
        elif mode == 'extra':
            answers['extra'] = {'type': 'noul', 'noul': .1}
        elif mode == 'nan':
            answers['p']['noul'] = 'NaN'
        elif mode == 'bool':
            answers['p']['noul'] = True
        else:
            return httpx.Response(500)
        return httpx.Response(200, json={'model': 'test-model', 'answers': answers,
                                       'usage': {'input_tokens': 5, 'output_tokens': 5}})
    client = JevResearchClient('https://api.typesafe.ai', 'test', 'secret', httpx.MockTransport(handler))
    with pytest.raises(ValueError, match='unavailable'):
        await client.evaluate((Probe('p', 'Does the claim follow?', {'claim': 'a', 'evidence': 'b'}),))
    assert len(client.audits) == 1
    assert 'error_type' in client.audits[0]
    assert 'secret' not in json.dumps(client.audits)
