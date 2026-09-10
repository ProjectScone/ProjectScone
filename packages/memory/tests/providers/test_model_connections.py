"""Host model settings persist atomically; discovery contacts only a mocked peer."""

import asyncio
import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Request
import httpx
import pytest

from scone_memory.api.model_connections import mount_model_connection_routes, probe_model_connection
from scone_memory.runtime.model_connections import (
    ModelConnection, ModelConnectionStore, ModelConnectionConflict, ModelConnectionError, api_key,
)


def config(**overrides):
    return ModelConnection(base_url='http://localhost:11434/v1', model='local-model', **overrides)


def test_defaults_are_copied_and_only_mutations_create_a_private_atomic_file(tmp_path):
    defaults = {'chat': config()}
    path = tmp_path / 'settings' / 'models.json'
    store = ModelConnectionStore(path, defaults)
    defaults.clear()
    before = store.snapshot()
    assert before['revision'] == 0 and before['schema_version'] == 1
    assert before['connections']['chat']['model'] == 'local-model'
    assert set(before['connections']) == {'chat', 'extraction', 'vision', 'transcription', 'speech'}
    assert not path.exists()
    saved = store.replace('speech', config(voice='local-voice'), expected_revision=0)
    assert saved['revision'] == 1
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text()) == saved
    restarted = ModelConnectionStore(path, {})
    assert restarted.get('speech').voice == 'local-voice'
    assert restarted.get('chat').model == 'local-model'
    assert restarted.replace('chat', None, expected_revision=1)['connections']['chat'] is None
    assert restarted.get('chat') is None


def test_revision_cas_is_checked_against_disk_across_store_instances(tmp_path):
    path = tmp_path / 'models.json'
    first, second = ModelConnectionStore(path, {}), ModelConnectionStore(path, {})
    first.replace('chat', config(), expected_revision=0)
    with pytest.raises(ModelConnectionConflict):
        second.replace('chat', None, expected_revision=0)
    assert second.snapshot()['revision'] == 1
    assert second.get('chat') == config()


def test_failed_atomic_replace_keeps_previous_file_and_revision(tmp_path, monkeypatch):
    path = tmp_path / 'models.json'
    store = ModelConnectionStore(path, {})
    before = store.replace('chat', config(), expected_revision=0)

    def fail(*args):
        raise OSError('private path')

    monkeypatch.setattr(os, 'replace', fail)
    with pytest.raises(ModelConnectionError) as raised:
        store.replace('chat', None, expected_revision=1)
    assert 'private path' not in str(raised.value)
    assert store.snapshot() == before
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('value', [
    {'base_url': 'https://api.openai.com/v1', 'model': 'm'},
    {'base_url': 'http://localhost/v1?key=secret', 'model': 'm'},
    {'base_url': 'http://localhost/v1', 'model': ' '},
    {'base_url': 'http://localhost/v1', 'model': 'm', 'api_key': 'secret'},
    {'base_url': 'http://localhost/v1', 'model': 'm', 'api_key_env': 'bad-name'},
    {'base_url': 'http://localhost/v1', 'model': 'm', 'timeout_s': True},
    {'base_url': 'http://localhost/v1', 'model': 'm', 'timeout_s': float('inf')},
    {'base_url': 'http://localhost/v1', 'model': 'm', 'timeout_s': 601},
    {'base_url': 'http://localhost/v1', 'model': 'm', 'sample_rate': 48001},
    {'base_url': 'http://localhost/v1', 'model': 'm', 'sample_rate': '24000'},
])
def test_connections_validate_local_configuration_without_persisted_secrets(value):
    with pytest.raises(ValueError):
        ModelConnection.model_validate(value)


def test_speech_requires_voice_and_saved_documents_fail_closed(tmp_path):
    path = tmp_path / 'models.json'
    store = ModelConnectionStore(path, {})
    with pytest.raises(ValueError, match='voice'):
        store.replace('speech', config(), expected_revision=0)
    path.write_text('{"schema_version":2,"revision":0,"connections":{}}')
    path.chmod(0o600)
    with pytest.raises(ModelConnectionError):
        ModelConnectionStore(path, {})


def test_store_rejects_symlink_files(tmp_path):
    target = tmp_path / 'target.json'
    target.write_text('{}')
    link = tmp_path / 'models.json'
    link.symlink_to(target)
    with pytest.raises(ModelConnectionError):
        ModelConnectionStore(link, {})
    assert target.read_text() == '{}'


def test_api_key_is_resolved_at_use_and_missing_or_invalid_variables_fail_closed(monkeypatch):
    connection = config(api_key_env='SCONE_TEST_LOCAL_TOKEN')
    monkeypatch.delenv('SCONE_TEST_LOCAL_TOKEN', raising=False)
    assert api_key(config()) is None
    with pytest.raises(ModelConnectionError):
        api_key(connection)
    monkeypatch.setenv('SCONE_TEST_LOCAL_TOKEN', 'local-secret')
    assert api_key(connection) == 'local-secret'
    assert 'local-secret' not in connection.model_dump_json()
    monkeypatch.setenv('SCONE_TEST_LOCAL_TOKEN', 'bad\nheader')
    with pytest.raises(ModelConnectionError) as raised:
        api_key(connection)
    assert 'bad' not in str(raised.value)


async def test_probe_lists_models_with_explicit_token_but_never_starts_inference(monkeypatch):
    requests = []
    monkeypatch.setenv('SCONE_TEST_LOCAL_TOKEN', 'local-secret')

    async def remote(request):
        requests.append(request)
        return httpx.Response(200, json={'data': [{'id': 'local-model'}, {'id': 'other'}, {'id': 'local-model'}]})

    result = await probe_model_connection(config(api_key_env='SCONE_TEST_LOCAL_TOKEN'),
                                          transport=httpx.MockTransport(remote))
    assert result == {'models': ['local-model', 'other'], 'model_available': True}
    assert len(requests) == 1 and requests[0].method == 'GET'
    assert str(requests[0].url) == 'http://localhost:11434/v1/models'
    assert requests[0].headers['authorization'] == 'Bearer local-secret'


@pytest.mark.parametrize('status,payload', [
    (302, {'data': [{'id': 'local-model'}]}), (503, {'error': 'private provider body'}),
    (200, {'models': ['wrong shape']}), (200, {'data': [{'id': 42}]}),
    (200, {'data': [{'id': 'x' * 300000}]}),
])
async def test_probe_errors_are_bounded_sanitized_and_never_redirected(status, payload):
    requests = []

    async def remote(request):
        requests.append(request)
        return httpx.Response(status, json=payload, headers={'location': 'https://cloud.invalid'})

    with pytest.raises(ModelConnectionError) as raised:
        await probe_model_connection(config(), transport=httpx.MockTransport(remote))
    assert 'private' not in str(raised.value) and len(requests) == 1


def app_for(store, on_change=None):
    app = FastAPI()

    async def authorize(request: Request):
        if request.headers.get('authorization') != 'Bearer host-admin':
            raise HTTPException(403, 'Host administration required')

    mount_model_connection_routes(app, store, authorize, on_change=on_change)
    return app


async def test_routes_require_host_authorization_validate_and_report_stale_revisions(tmp_path):
    store = ModelConnectionStore(tmp_path / 'models.json', {})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app_for(store)), base_url='http://test') as client:
        for method, path in [('GET', ''), ('PUT', '/chat'), ('POST', '/probe')]:
            response = await client.request(method, '/v1/model-connections' + path, json={})
            assert response.status_code == 403
        client.headers['authorization'] = 'Bearer host-admin'
        assert (await client.get('/v1/model-connections')).json()['revision'] == 0
        changed = await client.put('/v1/model-connections/chat', json={'expected_revision': 0, 'connection': config().model_dump()})
        assert changed.status_code == 200 and changed.json()['revision'] == 1
        stale = await client.put('/v1/model-connections/chat', json={'expected_revision': 0, 'connection': None})
        assert stale.status_code == 409
        for connection in [dict(base_url='https://cloud.invalid', model='m'), dict(base_url='http://localhost', model='m', api_key='secret')]:
            invalid = await client.put('/v1/model-connections/chat', json={'expected_revision': 1, 'connection': connection})
            assert invalid.status_code == 400 and 'secret' not in invalid.text
        assert (await client.put('/v1/model-connections/speech', json={'expected_revision': 1, 'connection': config().model_dump()})).status_code == 400
        assert (await client.post('/v1/model-connections/probe', json={'connection': dict(base_url='https://cloud.invalid', model='m')})).status_code == 400


def test_racing_writers_cannot_overwrite_the_same_revision(tmp_path):
    path = tmp_path / 'models.json'
    stores = [ModelConnectionStore(path, {}) for _ in range(2)]

    def save(store):
        try:
            return store.replace('chat', config(), expected_revision=0)['revision']
        except ModelConnectionConflict:
            return 'stale'

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(save, stores))
    assert sorted(map(str, outcomes)) == ['1', 'stale']
    assert stores[0].snapshot()['revision'] == 1


async def test_on_change_runs_after_successful_save_only_and_reports_saved_refresh_failure(tmp_path):
    store = ModelConnectionStore(tmp_path / 'models.json', {})
    calls = []

    def changed():
        calls.append(store.snapshot()['revision'])

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app_for(store, changed)),
                                 base_url='http://test', headers={'authorization': 'Bearer host-admin'}) as client:
        assert (await client.get('/v1/model-connections')).status_code == 200
        assert calls == []
        assert (await client.put('/v1/model-connections/chat', json={'expected_revision': 0, 'connection': config().model_dump()})).status_code == 200
        assert calls == [1]
        assert (await client.put('/v1/model-connections/chat', json={'expected_revision': 0, 'connection': None})).status_code == 409
        assert (await client.put('/v1/model-connections/speech', json={'expected_revision': 1, 'connection': config().model_dump()})).status_code == 400
        assert calls == [1]

    def failed_refresh():
        raise RuntimeError('private service token')

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app_for(store, failed_refresh)),
                                 base_url='http://test', headers={'authorization': 'Bearer host-admin'}) as client:
        response = await client.put('/v1/model-connections/chat', json={'expected_revision': 1, 'connection': None})
    assert response.status_code == 503 and response.json()['saved']['revision'] == 2
    assert 'private service token' not in response.text
    assert store.get('chat') is None


async def test_probe_cancellation_closes_response_and_disables_proxies_redirects(monkeypatch):
    entered = asyncio.Event()
    closed = asyncio.Event()
    created = []
    original = httpx.AsyncClient

    class Waiting(httpx.AsyncByteStream):
        async def __aiter__(self):
            entered.set()
            await asyncio.Event().wait()
            yield b'{}'

        async def aclose(self):
            closed.set()

    def create_client(**options):
        created.append(options)
        return original(**options)

    monkeypatch.setattr(httpx, 'AsyncClient', create_client)
    task = asyncio.create_task(probe_model_connection(config(timeout_s=600),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Waiting()))))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    assert created[0]['timeout'] == 10
    assert created[0]['trust_env'] is False and created[0]['follow_redirects'] is False


async def test_probe_http_errors_are_sanitized_and_return_502(tmp_path, monkeypatch):
    import scone_memory.api.model_connections as routes

    async def unavailable(connection):
        raise ModelConnectionError('private provider body')

    monkeypatch.setattr(routes, 'probe_model_connection', unavailable)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app_for(ModelConnectionStore(tmp_path / 'models.json', {}))),
                                 base_url='http://test', headers={'authorization': 'Bearer host-admin'}) as client:
        response = await client.post('/v1/model-connections/probe', json={'connection': config().model_dump()})
    assert response.status_code == 502 and 'error' in response.json()
    assert 'private provider body' not in response.text
