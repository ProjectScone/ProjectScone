"""Saved task schemas preserve author intent and obey plan authorization."""
import copy

import pytest

from ..agents.test_task_requirements import SCHEMA
from .test_agent_plans_api import setup, auth, payload


def schema_payload():
    body = payload(model_id='local')
    body['plan']['tasks'][0]['answer_requirements'] = {
        'instructions': 'Use the requested fields', 'max_bytes': 64000, 'max_lines': None,
        'format': 'json_object', 'output_schema': copy.deepcopy(SCHEMA)}
    return body


async def test_saved_acknowledgment_and_reload_preserve_authored_schema(setup):
    client, _, _ = setup
    body = schema_payload()
    caps = (await client.get('/v1/capabilities', headers=auth())).json()['features']
    assert caps['agents.output_requirements'] and caps['agents.output_schema']
    saved = await client.put('/v1/agent-plans/research-plan', json=body, headers=auth())
    assert saved.status_code == 200, saved.text
    assert saved.json()['plan'] == body['plan']
    loaded = await client.get('/v1/agent-plans/research-plan', headers=auth('reader'))
    assert loaded.json() == saved.json()
    assert loaded.headers['cache-control'] == 'no-store'


@pytest.mark.parametrize('change', ['unknown-key', 'remote-ref', 'invalid-schema', 'wrong-format'])
async def test_invalid_contracts_do_not_persist_or_disclose_schema(setup, change):
    client, _, store = setup
    body = schema_payload()
    contract = body['plan']['tasks'][0]['answer_requirements']
    if change == 'unknown-key':
        contract['SECRET'] = 'PRIVATE'
    elif change == 'remote-ref':
        contract['output_schema'] = {'$ref': 'https://PRIVATE.invalid/schema'}
    elif change == 'invalid-schema':
        contract['output_schema'] = {'type': 'PRIVATE'}
    else:
        contract['format'] = 'text'
    response = await client.put('/v1/agent-plans/research-plan', json=body, headers=auth())
    assert response.status_code == 422
    assert response.json()['code'] == 'invalid_agent_plan'
    assert 'PRIVATE' not in response.text
    assert store.get('alpha', 'research-plan') is None


async def test_missing_optional_validator_refuses_schema_but_keeps_simple_contracts(setup, monkeypatch):
    from scone_memory.realtime import output_schema
    def unavailable(value):
        raise ImportError('missing optional dependency')
    monkeypatch.setattr(output_schema, 'compile_schema', unavailable)
    client, _, _ = setup
    caps = (await client.get('/v1/capabilities', headers=auth())).json()['features']
    assert caps['agents.output_requirements'] and not caps['agents.output_schema']
    body = schema_payload()
    rejected = await client.put('/v1/agent-plans/research-plan', json=body, headers=auth())
    assert rejected.status_code == 422 and 'dependency' not in rejected.text
    del body['plan']['tasks'][0]['answer_requirements']['output_schema']
    accepted = await client.put('/v1/agent-plans/research-plan', json=body, headers=auth())
    assert accepted.status_code == 200
