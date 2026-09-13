"""Exact tool reviews identify the authenticated reviewer and never execute."""

import json

import httpx
import pytest

from scone_memory.api.app import create_app
from ..agents.test_agent_approval_service_recovery import service_host, memory
from .test_agent_runs_api import auth


@pytest.fixture
async def approval_api(service_host):
    host = service_host
    service = host.service
    app = create_app(
        service._memory,
        {'writer': 'alpha', 'reader': 'alpha', 'reviewer': 'alpha', 'reviewer2': 'alpha', 'other': 'bravo'},
        roles={'writer': 'write', 'reader': 'read', 'reviewer': 'review', 'reviewer2': 'review'},
        agent_catalog=service._catalog,
        agent_plan_store=service._plans,
        agent_run_service=service,
    )
    await host.start()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://scone.test'
    ) as client:
        yield client, app, host


async def pending(client):
    response = await client.get('/v1/agent-runs/one/approvals', headers=auth('reader'))
    assert response.status_code == 200, response.text
    assert response.headers['cache-control'] == 'no-store'
    return next(row for row in response.json()['items'] if row['call']['selection_id'] == 'first')


@pytest.mark.parametrize('decision', ['approve', 'deny'])
async def test_review_and_explicit_continuation_use_distinct_roles(approval_api, decision):
    client, app, host = approval_api
    assert (await client.get('/v1/capabilities', headers=auth())).json()['features']['agents.approvals']
    record = await pending(client)
    assert record['call']['arguments_json'] == '{"count":3}' and not host.effects
    assert 'invocation_digest' not in record
    path = '/v1/agent-runs/one/approvals/' + record['request_id'] + '/decision'
    body = {'decision': decision, 'expected_revision': 1}
    for key in ('writer', 'reader'):
        assert (await client.post(path, json=body, headers=auth(key))).status_code == 403
    saved = await client.post(path, json=body, headers=auth('reviewer'))
    assert saved.status_code == 200, saved.text
    assert saved.json()['revision'] == 2 and saved.json()['decided_by'].startswith('key:')
    assert 'reviewer' not in saved.text and not host.effects
    same = await client.post(path, json=body, headers=auth('reviewer'))
    assert same.json() == saved.json()
    other = await client.post(path, json=body, headers=auth('reviewer2'))
    assert other.status_code == 409 and not host.effects
    continuation = {'continuation_id': 'next', 'decisions': {record['request_id']: 2}}
    url = '/v1/agent-runs/one/approval-continuations'
    assert (await client.post(url, json=continuation, headers=auth('reviewer'))).status_code == 403
    admitted = await client.post(url, json=continuation, headers=auth())
    assert admitted.status_code == 202, admitted.text
    receipt = admitted.json()['activation']
    assert receipt['activation_id'] == 'next' and receipt['decisions'] == continuation['decisions']
    assert 'invocation_digest' not in receipt
    assert (await host.service.wait('alpha', 'one')).completed_steps == ('first',)
    assert host.effects == ([3] if decision == 'approve' else [])
    repeated = await client.post(url, json=continuation, headers=auth())
    assert repeated.status_code == 202 and repeated.json()['activation'] == receipt
    assert len(host.models['first'].requests) == 2


async def test_approval_boundaries_and_actor_spoofing_are_rejected(approval_api):
    client, app, host = approval_api
    record = await pending(client)
    path = '/v1/agent-runs/one/approvals/' + record['request_id'] + '/decision'
    for body in (
        {'decision': 'approve', 'expected_revision': True},
        {'decision': 'approve', 'expected_revision': 2},
        {'decision': 'approve', 'expected_revision': 1, 'actor': 'SECRET'},
        {'decision': 'execute', 'expected_revision': 1},
    ):
        result = await client.post(path, json=body, headers=auth('reviewer'))
        assert result.status_code == 422 and 'SECRET' not in result.text
    assert (await client.post(path, content=b'x' * 8193, headers=auth('reviewer'))).status_code == 413
    assert (await client.get('/v1/agent-runs/one/approvals', headers=auth('other'))).status_code == 404
    assert (await pending(client))['revision'] == 1 and not host.effects
    assert (await client.get('/v1/agent-runs/one/approvals')).status_code == 401


@pytest.mark.parametrize('operation', ['decision', 'continuation'])
@pytest.mark.parametrize('change', ['scope', 'role', 'revoke'])
async def test_streamed_approval_mutation_rechecks_auth(approval_api, operation, change):
    client, app, host = approval_api
    record = await pending(client)
    if operation == 'decision':
        path = '/v1/agent-runs/one/approvals/' + record['request_id'] + '/decision'
        body = {'decision': 'approve', 'expected_revision': 1}
        key = 'reviewer'
    else:
        await host.decide()
        path = '/v1/agent-runs/one/approval-continuations'
        body = {'continuation_id': 'next', 'decisions': {record['request_id']: 2}}
        key = 'writer'
    encoded = json.dumps(body).encode()

    async def stream():
        yield encoded[:10]
        if change == 'scope':
            app.state.keys[key] = 'bravo'
        elif change == 'role':
            app.state.roles[key] = 'read'
        else:
            app.state.keys.pop(key)
        yield encoded[10:]

    result = await client.post(path, content=stream(), headers=auth(key))
    assert result.status_code in (401, 403), result.text
    current = host.service._approvals.get('alpha', 'one', record['request_id'])
    assert current.revision == (1 if operation == 'decision' else 2) and not host.effects


@pytest.mark.parametrize('kind', ['decision', 'continuation'])
async def test_duplicate_json_members_refused_before_control_write(approval_api, kind):
    client, app, host = approval_api
    record = await pending(client)
    if kind == 'decision':
        path = '/v1/agent-runs/one/approvals/' + record['request_id'] + '/decision'
        raw = b'{"decision":"deny","decision":"approve","expected_revision":1}'
        key = 'reviewer'
    else:
        await host.decide()
        path = '/v1/agent-runs/one/approval-continuations'
        raw = (
            '{"continuation_id":"not-selected","continuation_id":"next","decisions":{'
            + json.dumps(record['request_id'])
            + ':2}}'
        ).encode()
        key = 'writer'
    result = await client.post(path, content=raw, headers=auth(key))
    assert result.status_code == 422, result.text
    assert host.service._approvals.get('alpha', 'one', record['request_id']).revision == (
        1 if kind == 'decision' else 2
    )


import scone_memory.agents.task_workflow as tasks


@pytest.mark.parametrize(
    'operation,change',
    [
        (op, change)
        for op in ('read', 'decision', 'continuation')
        for change in ('scope', 'revoke', 'role')
        if not (op == 'read' and change == 'role')
    ],
)
async def test_auth_changes_during_source_verifier_refuse_disclosure_and_control(
    approval_api, monkeypatch, operation, change
):
    client, app, host = approval_api
    record = await pending(client)
    key = 'reader' if operation == 'read' else 'reviewer' if operation == 'decision' else 'writer'
    if operation == 'continuation':
        await host.decide()
    original = tasks._verify_agent_evidence

    async def verify(*args, **kwargs):
        value = await original(*args, **kwargs)
        if change == 'scope':
            app.state.keys[key] = 'bravo'
        elif change == 'revoke':
            app.state.keys.pop(key, None)
        else:
            app.state.roles[key] = 'read'
        return value

    monkeypatch.setattr(tasks, '_verify_agent_evidence', verify)
    if operation == 'read':
        response = await client.get('/v1/agent-runs/one/approvals', headers=auth(key))
    elif operation == 'decision':
        response = await client.post(
            '/v1/agent-runs/one/approvals/' + record['request_id'] + '/decision',
            json={'decision': 'approve', 'expected_revision': 1},
            headers=auth(key),
        )
    else:
        response = await client.post(
            '/v1/agent-runs/one/approval-continuations',
            json={'continuation_id': 'next', 'decisions': {record['request_id']: 2}},
            headers=auth(key),
        )
    assert response.status_code in (401, 403), response.text
    assert record['request_id'] not in response.text
    assert host.service._approvals.get('alpha', 'one', record['request_id']).revision == (
        2 if operation == 'continuation' else 1
    )
    assert not host.effects


@pytest.mark.parametrize('shape', ['nested_duplicate', 'deep', 'too_many'])
async def test_control_nested_duplicates_and_depth_bounds_refuse_without_activation(approval_api, shape):
    client, app, host = approval_api
    record = await pending(client)
    await host.decide()
    if shape == 'nested_duplicate':
        ident = json.dumps(record['request_id'])
        raw = ('{"continuation_id":"next","decisions":{' + ident + ':2,' + ident + ':2}}').encode()
    elif shape == 'deep':
        raw = b'{"continuation_id":"next","decisions":' + b'[' * 1200 + b'0' + b']' * 1200 + b'}'
    else:
        raw = json.dumps(
            {'continuation_id': 'next', 'decisions': {format(i, '064x'): 2 for i in range(33)}}
        ).encode()
    result = await client.post(
        '/v1/agent-runs/one/approval-continuations', content=raw, headers=auth('writer')
    )
    assert result.status_code == 422, result.text
    assert host.service._approvals.activation('alpha', 'one', 'next') is None
    assert host.service._approvals.get('alpha', 'one', record['request_id']).revision == 2
    assert not host.effects
