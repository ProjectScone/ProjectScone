"""Saved connections affect future work; explicit host integrations still win."""

import asyncio
from dataclasses import replace

from fastapi.testclient import TestClient
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.__main__ import build_app
from scone_memory.ingestion.worker import PassReport
from scone_memory.realtime.persona import Persona
from scone_memory.retrieval.recall_scope import RecallScope
from scone_memory.runtime.config import Settings
from scone_memory.runtime.model_connections import ModelConnection, ModelConnectionStore
from scone_memory.runtime.model_runtime import DynamicLocalCatalog, LocalModelWorker, local_text_runtime, text_model_factory

AUTH = {'Authorization': 'Bearer solo'}


def settings_for(tmp_path, **overrides):
    return Settings.from_env({'SCONE_API_KEY': 'solo', 'SCONE_MODEL_CONNECTIONS': str(tmp_path / 'models.json'), **overrides})


def connection(model='local:model', **kwargs):
    return ModelConnection(base_url='http://127.0.0.1:1234/v1', model=model, **kwargs)


def engine_for():
    return asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())


@pytest.mark.parametrize('composed', [False, True])
def test_model_routes_mount_in_either_host_without_initial_model_or_network(tmp_path, composed):
    env = {'SCONE_CONVERSATIONS_JOURNAL': str(tmp_path / 'sessions.db')} if composed else {}
    settings = settings_for(tmp_path, **env)
    app = build_app(settings, engine_for())
    with TestClient(app, base_url='http://127.0.0.1', client=('127.0.0.1', 50000)) as client:
        assert client.get('/v1/model-connections', headers=AUTH).json()['revision'] == 0
        assert client.get('/v1/capabilities', headers=AUTH).json()['features']['models.manage'] is True
        assert client.get('/v1/status', headers=AUTH).json()['semantic_lane'] == 'manual'
        assert client.get('/v1/model-connections').status_code == 401
        if composed:
            caps = client.get('/v1/conversations/capabilities', headers=AUTH).json()
            assert caps['text_configured'] is False and caps['voice'] is False
            assert client.post('/v1/conversations', headers=AUTH, json={'request_id': 'empty', 'capture': True}).status_code == 503
        saved = client.put('/v1/model-connections/chat', headers=AUTH,
                           json={'expected_revision': 0, 'connection': connection(timeout_s=77).model_dump()})
        assert saved.status_code == 200
        if composed:
            assert client.get('/v1/conversations/capabilities', headers=AUTH).json()['text_configured'] is True
            created = client.post('/v1/conversations', headers=AUTH, json={'request_id': 'configured', 'capture': True})
            assert created.status_code == 200
            assert client.put('/v1/model-connections/chat', headers=AUTH, json={'expected_revision': 1, 'connection': None}).status_code == 200
            assert client.get('/v1/conversations/capabilities', headers=AUTH).json()['text_configured'] is False
            assert client.post('/v1/conversations', headers=AUTH, json={'request_id': 'removed', 'capture': True}).status_code == 503


@pytest.mark.parametrize('host,client_host,base_url,roles,keys', [
    ('0.0.0.0', '127.0.0.1', 'http://127.0.0.1', {}, {'solo': 'default'}),
    ('127.0.0.1', '192.168.1.10', 'http://127.0.0.1', {}, {'solo': 'default'}),
    ('127.0.0.1', '127.0.0.1', 'http://evil.example', {}, {'solo': 'default'}),
    ('127.0.0.1', '127.0.0.1', 'http://127.0.0.1', {'solo': 'read'}, {'solo': 'default'}),
    ('127.0.0.1', '127.0.0.1', 'http://127.0.0.1', {}, {'solo': 'one', 'other': 'two'}),
])
def test_model_administration_is_local_single_key_full_role_only(tmp_path, host, client_host, base_url, roles, keys):
    settings = replace(settings_for(tmp_path), host=host, roles=roles, keys=keys)
    with TestClient(build_app(settings, engine_for()), base_url=base_url, client=(client_host, 50000)) as client:
        response = client.get('/v1/model-connections', headers={**AUTH, 'X-Forwarded-For': '127.0.0.1'})
        assert response.status_code == 403


async def test_text_session_captures_endpoint_deadline_and_think_setting(tmp_path, monkeypatch):
    import scone_memory.runtime.model_runtime as runtime

    seen = []

    class Model:
        def __init__(self, base_url, model, **kwargs):
            seen.append((base_url, model, kwargs))

        async def aclose(self):
            pass

    monkeypatch.setattr(runtime, 'OpenAICompatibleTextModel', Model)
    store = ModelConnectionStore(tmp_path / 'models.json', {'chat': connection('first:model', timeout_s=73)})
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    captured = []
    original = runtime.TextConversation

    def conversation(*args, **kwargs):
        captured.append((args[3], kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, 'TextConversation', conversation)
    factory = local_text_runtime(engine, store, think=False)
    old = factory('default', 'old', RecallScope.from_mapping({}))
    store.replace('chat', connection('second:model', timeout_s=84), expected_revision=0)
    new = factory('default', 'new', RecallScope.from_mapping({}))
    for model_factory, options in captured:
        await model_factory().aclose()
    assert [entry[1] for entry in seen] == ['first:model', 'second:model']
    assert [options['turn_timeout'] for _, options in captured] == [73, 84]
    assert all(entry[2]['trust_env'] is False and entry[2]['think'] is False for entry in seen)
    await old.close()
    await new.close()
    await engine.close()


async def test_dynamic_catalog_uses_opaque_aliases_and_accepts_real_local_model_ids(tmp_path):
    store = ModelConnectionStore(tmp_path / 'models.json', {
        'chat': connection('reply:latest', timeout_s=65),
        'transcription': connection('whisper:large-v3'),
        'speech': connection('kokoro:latest', voice='voice:custom', sample_rate=32000),
    })
    catalog = DynamicLocalCatalog(store, think=False)
    assert len(catalog.personas) == 1
    persona = catalog.personas[0]
    Persona.model_validate(persona)
    assert ':' not in persona.reply.model
    fingerprint = catalog.fingerprint(persona.id)
    selected = catalog.get(persona.id)
    recognizer, speech = selected.stt_factory(), selected.tts_factory()
    assert recognizer._model == 'whisper:large-v3'
    assert speech._request('Hello')[2]['model'] == 'kokoro:latest'
    assert speech._request('Hello')[2]['voice'] == 'voice:custom'
    assert speech._sample_rate == 32000
    await recognizer.aclose()
    await speech.aclose()
    store.replace('speech', connection('kokoro:next', voice='voice:custom'), expected_revision=0)
    assert catalog.fingerprint(persona.id) != fingerprint
    store.replace('transcription', None, expected_revision=1)
    assert catalog.personas == () and catalog.get(persona.id) is None


async def test_extraction_refresh_keeps_inflight_pass_and_unrelated_edit_state(tmp_path):
    settings = settings_for(tmp_path)
    store = ModelConnectionStore(tmp_path / 'models.json', {'extraction': connection('old:model')})
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    worker = LocalModelWorker(engine, settings, store)
    initial = worker._current
    initial.distiller._done.add(('default', 42))
    store.replace('chat', connection('chat:model'), expected_revision=0)
    worker.refresh()
    assert worker._current is initial and ('default', 42) in worker.distiller._done
    entered, released = asyncio.Event(), asyncio.Event()

    async def old_pass(space):
        entered.set()
        await released.wait()
        assert initial.distiller.chat.connection.model == 'old:model'
        return PassReport(space, proposed=7)

    initial.run_once = old_pass
    running = asyncio.create_task(worker.run_once('default'))
    await asyncio.wait_for(entered.wait(), 1)
    store.replace('extraction', connection('new:model'), expected_revision=1)
    worker.refresh()
    assert worker._current is not initial
    released.set()
    assert (await running).proposed == 7
    assert worker.last['default'].proposed == 7
    assert worker.distiller.chat.connection.model == 'new:model'
    await engine.close()


def test_vision_capability_tracks_saved_configuration_without_requesting_inference(tmp_path):
    app = build_app(settings_for(tmp_path), engine_for())
    with TestClient(app, base_url='http://127.0.0.1', client=('127.0.0.1', 50000)) as client:
        assert 'images.understand' not in client.get('/v1/capabilities', headers=AUTH).json()['features']
        assert client.put('/v1/model-connections/vision', headers=AUTH,
            json={'expected_revision': 0, 'connection': connection().model_dump()}).status_code == 200
        assert client.get('/v1/capabilities', headers=AUTH).json()['features']['images.understand'] is True
        assert client.put('/v1/model-connections/vision', headers=AUTH,
            json={'expected_revision': 1, 'connection': None}).status_code == 200
        assert 'images.understand' not in client.get('/v1/capabilities', headers=AUTH).json()['features']
