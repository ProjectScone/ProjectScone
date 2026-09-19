"""Human version review over authenticated API never executes a proposal."""
import json
import sqlite3
import threading

import httpx
import pytest

from scone_memory import MemoryEngine, InMemoryDocumentStore, InMemoryVectorIndex, HashEmbedder
from scone_memory.agents.tool_recipe_store import ToolRecipeStore
from scone_memory.agents.workflow import WorkflowError
from scone_memory.api.app import create_app
from tests.agents.test_tool_recipes import capability, recipe


def auth(key='reader'):
    return {'Authorization': 'Bearer ' + key}


@pytest.fixture
async def setup(tmp_path):
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    effects = []
    tool = capability(effects)
    store = ToolRecipeStore(tmp_path / 'recipes', key=b'k' * 32)
    store.propose('alpha', 'one', recipe(tool), proposed_by='author-agent', tools=[tool])
    try:
        app = create_app(memory, {'reader': 'alpha', 'writer': 'alpha', 'reviewer': 'alpha', 'other': 'bravo'},
                     roles={'reader': 'read', 'writer': 'write', 'reviewer': 'review'}, tool_recipe_store=store)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local') as client:
            yield client, app, store, effects, memory
    finally:
        store.close()
        await memory.close()


async def test_scoped_backlog_review_and_revoke(setup):
    client, app, store, effects, _ = setup
    caps = (await client.get('/v1/capabilities', headers=auth())).json()['features']
    assert caps['agents.tool_recipes.review'] is True
    assert (await client.get('/v1/tool-recipes')).status_code == 401
    first = await client.get('/v1/tool-recipes?limit=1', headers=auth())
    assert first.status_code == 200 and first.headers['cache-control'] == 'no-store'
    row = first.json()['items'][0]
    assert row['space'] == 'alpha' and row['status'] == 'pending' and row['revision'] == 1
    assert row['dependencies'][0]['name'] == 'double'
    assert 'dependencies_json' not in row and 'handler' not in first.text
    assert (await client.get('/v1/tool-recipes', headers=auth('other'))).json()['items'] == []
    assert (await client.get('/v1/tool-recipes/one', headers=auth('other'))).status_code == 404
    path = '/v1/tool-recipes/one/decision'
    body = {'decision': 'approve', 'reason': 'Checked the two-step calculation.', 'expected_revision': 1}
    for key in ('reader', 'writer'):
        assert (await client.post(path, json=body, headers=auth(key))).status_code == 403
    response = await client.post(path, json=body, headers=auth('reviewer'))
    assert response.status_code == 200, response.text
    assert response.json()['status'] == 'approved' and response.json()['revision'] == 2
    actor = response.json()['reviews'][0]['actor']
    assert actor.startswith('key:') and 'reviewer' not in actor
    assert not effects
    assert (await client.post(path, json=body, headers=auth('reviewer'))).status_code == 409
    response = await client.post('/v1/tool-recipes/one/revoke', headers=auth('reviewer'),
                                 json={'reason': 'Retired.', 'expected_revision': 2})
    assert response.status_code == 200 and response.json()['status'] == 'revoked'
    assert len(response.json()['reviews']) == 2 and not effects


@pytest.mark.parametrize('change', ['role', 'key', 'space'])
async def test_review_rechecks_authority_after_body_receive(setup, change):
    client, app, store, effects, _ = setup
    raw = json.dumps({'decision': 'approve', 'reason': 'Reviewed.', 'expected_revision': 1}).encode()
    async def body():
        yield raw[:5]
        if change == 'role':
            app.state.roles['reviewer'] = 'read'
        elif change == 'key':
            app.state.keys.pop('reviewer')
        else:
            app.state.keys['reviewer'] = 'bravo'
        yield raw[5:]
    response = await client.post('/v1/tool-recipes/one/decision', content=body(), headers=auth('reviewer'))
    assert response.status_code in (401, 403)
    assert store.get('alpha', 'one').status == 'pending' and not effects


@pytest.mark.parametrize('body', [
    b'{"decision":"approve","reason":"SECRET","expected_revision":true}',
    b'{"decision":"approve","reason":"SECRET","expected_revision":1,"actor":"spoof"}',
    b'{"decision":"deny","decision":"approve","reason":"SECRET","expected_revision":1}',
    b'{"decision":"approve","reason":"  ","expected_revision":1}',
])
async def test_bad_review_input_is_refused_without_leaking_it(setup, body):
    client, app, store, effects, _ = setup
    result = await client.post('/v1/tool-recipes/one/decision', content=body, headers=auth('reviewer'))
    assert result.status_code == 422 and 'SECRET' not in result.text and 'spoof' not in result.text
    assert store.get('alpha', 'one').status == 'pending' and not effects


async def test_review_body_and_query_bounds(setup):
    client, app, store, effects, _ = setup
    assert (await client.post('/v1/tool-recipes/one/decision', content=b'x' * 8193,
                              headers=auth('reviewer'))).status_code == 413
    for query in ('limit=0', 'limit=101', 'limit=1&limit=2', 'unknown=SECRET', 'after=SECRET'):
        result = await client.get('/v1/tool-recipes?' + query, headers=auth())
        assert result.status_code == 422 and 'SECRET' not in result.text
    assert store.get('alpha', 'one').status == 'pending' and not effects


async def test_deleted_space_during_review_body_is_refused(setup):
    client, app, store, effects, memory = setup
    raw = json.dumps({'decision': 'approve', 'reason': 'Reviewed.', 'expected_revision': 1}).encode()
    async def body():
        yield raw[:10]
        await memory.delete_space('alpha')
        yield raw[10:]
    response = await client.post('/v1/tool-recipes/one/decision', content=body(), headers=auth('reviewer'))
    assert response.status_code == 404
    assert store.get('alpha', 'one').status == 'pending' and not effects
    assert (await client.get('/v1/tool-recipes/one', headers=auth())).status_code == 404


async def test_reads_recheck_auth_before_delivery(setup, monkeypatch):
    client, app, store, effects, _ = setup
    original = store.get
    def revoke_reader(*args, **kwargs):
        result = original(*args, **kwargs)
        app.state.keys.pop('reader')
        return result
    monkeypatch.setattr(store, 'get', revoke_reader)
    response = await client.get('/v1/tool-recipes/one', headers=auth())
    assert response.status_code == 401 and 'quadruple' not in response.text


async def test_store_failure_is_not_an_empty_backlog(setup):
    client, app, store, effects, _ = setup
    store.close()
    response = await client.get('/v1/tool-recipes', headers=auth())
    assert response.status_code == 503 and response.json()['code'] == 'tool_recipe_unavailable'


async def test_routes_and_capability_absent_without_host_store():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        app = create_app(memory, {'reader': 'alpha'})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local') as client:
            assert not (await client.get('/v1/capabilities', headers=auth())).json()['features'].get('agents.tool_recipes.review')
            assert (await client.get('/v1/tool-recipes', headers=auth())).status_code == 404
    finally:
        await memory.close()


@pytest.mark.parametrize('change', ['role', 'key', 'space'])
@pytest.mark.parametrize('action', ['decision', 'revoke'])
async def test_authority_change_during_write_lock_prevents_review(setup, monkeypatch, change, action):
    client, app, store, effects, _ = setup
    if action == 'revoke':
        store.decide('alpha', 'one', decision='approve', actor='human', reason='Reviewed', expected_revision=1)
    locked, deciding = threading.Event(), threading.Event()
    method = 'decide' if action == 'decision' else 'revoke'
    original = getattr(store, method)
    def review(*args, **kwargs):
        deciding.set()
        return original(*args, **kwargs)
    monkeypatch.setattr(store, method, review)
    failures = []
    def change_authority():
        connection = None
        try:
            connection = sqlite3.connect(store._path, isolation_level=None)
            connection.execute('BEGIN IMMEDIATE')
            locked.set()
            assert deciding.wait(timeout=3)
            if change == 'role':
                app.state.roles['reviewer'] = 'read'
            elif change == 'key':
                app.state.keys.pop('reviewer')
            else:
                app.state.keys['reviewer'] = 'bravo'
        except BaseException as error:
            failures.append(error)
        finally:
            if connection is not None:
                connection.close()
    thread = threading.Thread(target=change_authority)
    thread.start()
    try:
        assert locked.wait(timeout=1)
        body = {'reason': 'Reviewed', 'expected_revision': 1 if action == 'decision' else 2}
        if action == 'decision':
            body['decision'] = 'approve'
        response = await client.post('/v1/tool-recipes/one/' + action, headers=auth('reviewer'), json=body)
        assert response.status_code == (403 if change == 'role' else 401)
        assert store.get('alpha', 'one').status == ('pending' if action == 'decision' else 'approved')
        assert effects == []
    finally:
        deciding.set()
        thread.join(timeout=5)
        assert not thread.is_alive() and failures == []


async def test_invalid_identifier_is_a_client_error(setup):
    client, _, _, _, _ = setup
    assert (await client.get('/v1/tool-recipes/%21invalid', headers=auth())).status_code == 422
    for action in ('decision', 'revoke'):
        response = await client.post('/v1/tool-recipes/%21invalid/' + action, headers=auth('reviewer'), json={})
        assert response.status_code == 422


@pytest.mark.parametrize('proposal_id', ['.', '..'])
async def test_proposal_id_cannot_be_a_url_navigation_segment(setup, proposal_id):
    _, _, store, _, _ = setup
    tool = capability([])
    with pytest.raises((ValueError, WorkflowError)):
        store.propose('alpha', proposal_id, recipe(tool), proposed_by='agent', tools=[tool])
