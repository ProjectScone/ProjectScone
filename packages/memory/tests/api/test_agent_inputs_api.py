"""Authenticated human replies remain distinct from explicit execution admission."""
import json

import pytest

from .test_agent_runs_api import auth, setup


async def prepare(client, service, release):
    plan = {'kind': 'interactive', 'workflow_id': 'interactive', 'tasks': [
        {'kind': 'input', 'task_id': 'choose', 'prompt': 'Choose a direction', 'depends_on': [], 'max_response_bytes': 32},
        {'task_id': 'answer', 'agent_id': 'research', 'model_id': 'local', 'prompt': 'Answer', 'depends_on': ['choose']}]}
    saved = await client.put('/v1/agent-plans/interactive', json={'plan': plan, 'expected_revision': 0}, headers=auth())
    assert saved.status_code == 200, saved.text
    release.set()
    started = await client.post('/v1/agent-runs', json={'run_id': 'one', 'workflow_id': 'interactive',
        'plan_revision': 1, 'question': 'Question'}, headers=auth())
    assert started.status_code == 202, started.text
    assert (await service.wait('alpha', 'one')).status == 'awaiting_input'


async def test_reply_is_durable_without_execution_and_continuation_is_explicit(setup):
    client, _, service, _, release, _, calls = setup
    await prepare(client, service, release)
    assert (await client.get('/v1/capabilities', headers=auth())).json()['features']['agents.inputs']
    prompts = await client.get('/v1/agent-runs/one/inputs', headers=auth('reader'))
    assert prompts.status_code == 200 and prompts.headers['cache-control'] == 'no-store'
    row = prompts.json()['items'][0]
    assert row['task_id'] == 'choose' and row['revision'] == 1 and row['context'] == '[]'
    assert 'invocation_digest' not in row and not calls
    url = '/v1/agent-runs/one/inputs/choose/response'
    assert (await client.post(url, json={'response': 'North', 'expected_revision': 1}, headers=auth('reader'))).status_code == 403
    assert (await client.post(url, json={'response': 'North', 'expected_revision': 1}, headers=auth('other'))).status_code == 404
    saved = await client.post(url, json={'response': 'North', 'expected_revision': 1}, headers=auth())
    assert saved.status_code == 200 and saved.json()['revision'] == 2 and not calls
    conflict = await client.post(url, json={'response': 'South', 'expected_revision': 1}, headers=auth())
    assert conflict.status_code == 409
    resumed = await client.post('/v1/agent-runs/one/continue', json={'continuation_id': 'c', 'responses': {'choose': 2}}, headers=auth())
    assert resumed.status_code == 202, resumed.text
    assert (await service.wait('alpha', 'one')).status == 'completed' and len(calls) == 1
    result = await client.get('/v1/agent-runs/one/result', headers=auth('reader'))
    assert result.json()['results']['choose']['kind'] == 'human_input'


@pytest.mark.parametrize('operation', ['response', 'continue'])
async def test_streamed_input_mutation_rechecks_key_scope(setup, operation):
    client, app, service, _, release, _, calls = setup
    await prepare(client, service, release)
    await service.respond('alpha', 'one', 'choose', response='North', expected_revision=1)
    body = {'response': 'North', 'expected_revision': 1} if operation == 'response' else {'continuation_id': 'c', 'responses': {'choose': 2}}
    suffix = 'inputs/choose/response' if operation == 'response' else 'continue'
    encoded = json.dumps(body).encode()
    async def stream():
        yield encoded[:10]
        app.state.keys['writer'] = 'bravo'
        yield encoded[10:]
    response = await client.post('/v1/agent-runs/one/' + suffix, content=stream(), headers=auth())
    assert response.status_code in (401, 403)
    assert (await service.inputs('alpha', 'one'))[0].activation_id is None and not calls


async def test_reply_boundaries_and_prompt_scope_are_enforced(setup):
    client, app, service, _, release, _, calls = setup
    await prepare(client, service, release)
    url = '/v1/agent-runs/one/inputs/choose/response'
    for body in ({'response': 'é' * 17, 'expected_revision': 1}, {'response': 'X', 'expected_revision': True},
                 {'response': 'PRIVATE', 'expected_revision': 1, 'execute': True}):
        response = await client.post(url, json=body, headers=auth())
        assert response.status_code == 422 and 'PRIVATE' not in response.text
    assert (await client.post(url, content=b'x' * 8193, headers=auth())).status_code == 413
    assert (await client.get('/v1/agent-runs/one/inputs', headers=auth('other'))).status_code == 404
    assert (await service.inputs('alpha', 'one'))[0].response is None and not calls
