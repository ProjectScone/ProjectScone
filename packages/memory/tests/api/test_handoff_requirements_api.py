"""Handoff output contracts are advertised honestly and persist exactly."""
import copy
import pytest
from .test_agent_plans_api import setup, auth
from ..agents.test_task_requirements import SCHEMA


def payload():
    return {'expected_revision': 0, 'plan': {'workflow_id': 'research-plan',
        'root_agent': 'research', 'max_handoffs': 3,
        'agents': [{'agent_id': 'research', 'model_id': 'local', 'can_handoff_to': []}],
        'answer_requirements': {'format': 'json_object', 'instructions': 'Return requested fields',
            'max_bytes': 64000, 'max_lines': None, 'output_schema': copy.deepcopy(SCHEMA)}}}


async def test_handoff_contract_round_trip_and_capability(setup):
    client, _, _ = setup
    caps = (await client.get('/v1/capabilities', headers=auth())).json()['features']
    assert caps['agents.handoffs.output_requirements'] and caps['agents.output_schema']
    body = payload()
    saved = await client.put('/v1/agent-plans/research-plan', json=body, headers=auth())
    assert saved.status_code == 200, saved.text
    assert saved.json()['plan'] == body['plan']
    loaded = await client.get('/v1/agent-plans/research-plan', headers=auth('reader'))
    assert loaded.json() == saved.json()


@pytest.mark.parametrize('change', ['unknown', 'remote', 'wrapper-budget', 'wrong-format'])
async def test_invalid_final_contract_never_saves_or_leaks_configuration(setup, change):
    client, _, store = setup
    body = payload()
    contract = body['plan']['answer_requirements']
    if change == 'unknown':
        contract['PRIVATE'] = 'PRIVATE'
    elif change == 'remote':
        contract['output_schema'] = {'$ref': 'https://PRIVATE.invalid/schema'}
    elif change == 'wrapper-budget':
        contract['output_schema'] = {'properties': {f'f{i}': {'type': 'string'} for i in range(254)}}
    else:
        contract['format'] = 'text'
    response = await client.put('/v1/agent-plans/research-plan', json=body, headers=auth())
    assert response.status_code == 422
    assert response.json()['code'] == 'invalid_agent_plan'
    assert 'PRIVATE' not in response.text and store.get('alpha', 'research-plan') is None


async def test_missing_optional_validator_withholds_handoff_contract_capability(setup, monkeypatch):
    from scone_memory.realtime import output_schema
    def unavailable(value):
        raise ImportError('missing optional validator')
    monkeypatch.setattr(output_schema, 'compile_schema', unavailable)
    client, _, store = setup
    caps = (await client.get('/v1/capabilities', headers=auth())).json()['features']
    assert not caps['agents.handoffs.output_requirements']
    body = payload()
    body['plan']['answer_requirements'] = {'max_lines': 1}
    response = await client.put('/v1/agent-plans/research-plan', json=body, headers=auth())
    assert response.status_code == 422 and store.get('alpha', 'research-plan') is None
