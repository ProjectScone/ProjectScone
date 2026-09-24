import json

import httpx
import pytest

from scone_memory.experimental.contract_judge import JevContractJudge
from scone_memory.experimental.memory_contracts import ContractRequest, Evidence, evidence_worlds


async def test_worlds_are_isolated_in_question_instructions_and_empty_world_is_local():
    task = ContractRequest('scope', 'Owner?', 'Maya owns deployment.', (
        Evidence('a', 'a', 'Maya owns deployment.'), Evidence('b', 'b', 'An unrelated fact.'),
    ))
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body['state'] == {'question': task.question, 'claim': task.claim}
        assert 'w0_support' not in body['questions']
        assert [e['source_id'] for e in body['questions']['w1_support']['instructions']['evidence']] == ['a']
        assert [e['source_id'] for e in body['questions']['w2_support']['instructions']['evidence']] == ['b']
        return httpx.Response(200, json={'model': 'jev-test', 'usage': {'input_tokens': 100, 'output_tokens': 20},
            'answers': {key: {'type': 'noul', 'noul': .01} for key in body['questions']}})
    judge = JevContractJudge('https://api.typesafe.ai', 'jev-test', api_key='test', transport=httpx.MockTransport(handler))
    result = await judge.assess(task, evidence_worlds(task))
    assert result.judgments[0].status == 'insufficient'
    assert result.questions == 6
    assert result.input_tokens == 100


async def test_out_of_order_worlds_cannot_attach_support_to_the_wrong_source():
    task = ContractRequest('scope', 'Owner?', 'Maya owns deployment.', (
        Evidence('a', 'a', 'Maya owns deployment.'), Evidence('b', 'b', 'Unrelated text.'),
    ))
    worlds = evidence_worlds(task)
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'model': 'test', 'usage': {'input_tokens': 1, 'output_tokens': 1},
            'answers': {key: {'type': 'noul', 'noul': .99 if key == 'w1_support' else .01}
                        for key in json.loads(request.content)['questions']}})
    judge = JevContractJudge('https://api.typesafe.ai', 'test', api_key='test', transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match='order'):
        await judge.assess(task, (worlds[0], worlds[2], worlds[1], worlds[3]))


@pytest.mark.parametrize('mode', ['missing', 'extra', 'nan', 'error'])
async def test_provider_errors_and_malformed_responses_never_become_support(mode):
    task = ContractRequest('scope', 'Owner?', 'Maya owns deployment.', (Evidence('a', 'a', 'Maya owns deployment.'),))
    def handler(request: httpx.Request) -> httpx.Response:
        answers = {key: {'type': 'noul', 'noul': .99} for key in json.loads(request.content)['questions']}
        if mode == 'missing':
            answers.pop('w1_support')
        elif mode == 'extra':
            answers['extra'] = {'type': 'noul', 'noul': .9}
        elif mode == 'nan':
            answers['w1_support']['noul'] = 'NaN'
        elif mode == 'error':
            return httpx.Response(500)
        return httpx.Response(200, json={'model': 'test', 'usage': {'input_tokens': 1, 'output_tokens': 1}, 'answers': answers})
    judge = JevContractJudge('https://api.typesafe.ai', 'test', api_key='test', transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match='unavailable'):
        await judge.assess(task, evidence_worlds(task))
