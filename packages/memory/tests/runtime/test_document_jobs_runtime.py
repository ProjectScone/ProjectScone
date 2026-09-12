"""Only explicit private local configuration enables durable imports."""
import json

import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.__main__ import build_app
from scone_memory.runtime.config import Settings


def config(tmp_path, **changes):
    path = tmp_path / 'imports.json'
    path.write_text(json.dumps({'schema_version': 1, 'state_dir': 'imports',
        'key_env': 'TEST_IMPORT_KEY', 'parser_revision': 'installed-parser-v1', **changes}))
    path.chmod(0o600)
    return path


@pytest.mark.parametrize('composed', [False, True])
async def test_launcher_owns_passive_local_imports_across_restart(tmp_path, monkeypatch, composed):
    monkeypatch.setenv('TEST_IMPORT_KEY', 'ab' * 32)
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    env = {'SCONE_API_KEY': 'key', 'SCONE_DOCUMENT_JOBS_CONFIG': str(config(tmp_path))}
    if composed:
        env['SCONE_CONVERSATIONS_JOURNAL'] = str(tmp_path / 'conversations.db')
    try:
        original = await engine.attach('default', b'Saturn has rings.', 'text/plain', filename='note.md')
        for first in [True, False]:
            app = build_app(Settings.from_env(env), engine)
            async with app.router.lifespan_context(app), httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url='http://scone.test',
                headers={'authorization': 'Bearer key'}) as client:
                caps = (await client.get('/v1/capabilities')).json()['features']
                assert caps['documents.jobs']
                target = getattr(app.state, 'memory_app', app)
                service = target.state.document_import_service
                if first:
                    response = await client.post('/v1/document-jobs', json={'import_id': 'one',
                        'attachment_id': original.attachment_id, 'filename': 'note.md'})
                    assert response.status_code == 202, response.text
                    await service.wait('default', 'one')
                page = (await client.get('/v1/document-jobs')).json()
                assert page['items'][0]['status'] == 'completed'
                assert page['items'][0]['attempt'] == 1
                assert (await client.get('/v1/document-jobs/one/result')).status_code == 200
            assert service._closed
    finally:
        await engine.close()


@pytest.mark.parametrize('mode', ['public', 'missing_key', 'symlink', 'invalid_limit'])
async def test_bad_configuration_fails_before_server_start(tmp_path, monkeypatch, mode):
    monkeypatch.setenv('TEST_IMPORT_KEY', 'ab' * 32)
    path = config(tmp_path, **({'max_active': True} if mode == 'invalid_limit' else {}))
    if mode == 'public':
        path.chmod(0o644)
    elif mode == 'missing_key':
        monkeypatch.delenv('TEST_IMPORT_KEY')
    elif mode == 'symlink':
        link = tmp_path / 'link.json'; link.symlink_to(path); path = link
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        with pytest.raises(ValueError):
            build_app(Settings.from_env({'SCONE_DOCUMENT_JOBS_CONFIG': str(path)}), engine)
    finally:
        await engine.close()


@pytest.mark.parametrize('composed', [False, True])
def test_launcher_failure_before_lifespan_releases_import_registry(tmp_path, monkeypatch, composed):
    from scone_memory.api import __main__ as host
    monkeypatch.setenv('TEST_IMPORT_KEY', 'ab' * 32)
    env = {'SCONE_API_KEY': 'key', 'SCONE_DOCUMENT_JOBS_CONFIG': str(config(tmp_path))}
    if composed:
        env['SCONE_CONVERSATIONS_JOURNAL'] = str(tmp_path / 'conversation.db')
    owners = []
    original = host.build_app
    async def memory(settings):
        return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    def capture(settings, engine):
        app = original(settings, engine)
        owners.append(getattr(app.state, 'memory_app', app).state.document_import_service)
        return app
    def fail(*args):
        raise RuntimeError('server construction failed')
    monkeypatch.setattr(host, 'build_engine', memory)
    monkeypatch.setattr(host, 'build_app', capture)
    monkeypatch.setattr(host, 'build_server', fail)
    try:
        with pytest.raises(RuntimeError, match='server construction failed'):
            host.main(Settings.from_env(env))
        assert owners[0]._closed
    finally:
        for owner in owners:
            owner.close_idle()
