"""Private host configuration enables local sync without executing it."""
import asyncio
import json
import os

import httpx
import pytest

from scone_memory.api.__main__ import build_app
from scone_memory.runtime.config import Settings
from scone_memory.runtime.directory_sync import load_directory_sync
from tests.ingestion.test_directory_sync import env


def config(env, **changes):
    _, root, parent = env
    path = parent / 'sync.json'
    body = {'schema_version': 1, 'state_dir': 'sync-state', 'key_env': 'TEST_SYNC_KEY',
            'store_id': 'fixture-store', 'collections': [{
                'collection_id': 'notes', 'label': 'Team notes', 'space': 'alpha',
                'root': str(root), 'parser_revision': 'installed-v1', 'allow_delete_missing': True,
            }], **changes}
    path.write_text(json.dumps(body))
    path.chmod(0o600)
    return path


def settings(path, parent, composed=False):
    values = {'SCONE_API_KEYS': 'writer:alpha', 'SCONE_DIRECTORY_SYNC_CONFIG': str(path)}
    if composed:
        values['SCONE_CONVERSATIONS_JOURNAL'] = str(parent / 'conversations.sqlite')
    return Settings.from_env(values)


async def wait_service(service):
    async with asyncio.timeout(10):
        while (status := await service.status('alpha', 'scan')).active_local:
            await asyncio.sleep(.01)
        return status


@pytest.mark.parametrize('composed', [False, True])
async def test_standard_hosts_load_passively_and_reopen_durable_history(env, monkeypatch, composed):
    memory, root, parent = env
    monkeypatch.setenv('TEST_SYNC_KEY', 'ab' * 32)
    path = config(env)
    (root / 'note.txt').write_text('Configured local directory evidence')
    for first in (True, False):
        app = build_app(settings(path, parent, composed), memory)
        target = getattr(app.state, 'memory_app', app)
        service = target.state.directory_sync_service
        assert service is not None and not service._tasks
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://local.test',
            headers={'authorization': 'Bearer writer'}) as client:
            assert (await client.get('/v1/capabilities')).json()['features']['documents.sync']
            if first:
                assert (await memory.documents.counts('alpha')).episodes == 0
                response = await client.post('/v1/sync-runs', json={'run_id': 'scan', 'collection_id': 'notes'})
                assert response.status_code == 202, response.text
                assert (await wait_service(service)).status == 'completed'
            history = await client.get('/v1/sync-runs')
            assert history.json()['items'][0]['record']['attempt'] == 1
            result = await client.get('/v1/sync-runs/scan/result')
            assert result.status_code == 200 and result.json()['items'][0]['source']['status'] == 'added'
        assert service._closed
    assert (await memory.documents.counts('alpha')).episodes == 1


@pytest.mark.parametrize('failure', ['public', 'symlink', 'hardlink', 'missing_key', 'duplicate_key',
                                    'oversize', 'invalid_flag', 'duplicate_root', 'state_under_source'])
async def test_bad_configuration_is_refused_before_opening_registry(env, monkeypatch, failure):
    memory, root, parent = env
    monkeypatch.setenv('TEST_SYNC_KEY', 'ab' * 32)
    path = config(env)
    body = json.loads(path.read_text())
    if failure == 'public':
        path.chmod(0o644)
    elif failure == 'symlink':
        link = parent / 'link.json'
        link.symlink_to(path)
        path = link
    elif failure == 'hardlink':
        os.link(path, parent / 'alias.json')
    elif failure == 'missing_key':
        monkeypatch.delenv('TEST_SYNC_KEY')
    elif failure == 'duplicate_key':
        path.write_text(path.read_text().replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'))
    elif failure == 'oversize':
        path.write_bytes(b' '*131073)
    elif failure == 'invalid_flag':
        body['collections'][0]['allow_delete_missing'] = 1
        path.write_text(json.dumps(body))
    elif failure == 'duplicate_root':
        body['collections'].append({**body['collections'][0], 'collection_id': 'alias'})
        path.write_text(json.dumps(body))
    elif failure == 'state_under_source':
        body['state_dir'] = str(root / 'private-state')
        path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        load_directory_sync(str(path), memory)
    assert not list(parent.rglob('runs.sqlite'))


async def test_relative_roots_and_parser_revision_changes_are_bound(env, monkeypatch):
    memory, root, parent = env
    monkeypatch.setenv('TEST_SYNC_KEY', 'ab' * 32)
    path = config(env)
    body = json.loads(path.read_text())
    body['collections'][0]['root'] = root.name
    path.write_text(json.dumps(body))
    first = load_directory_sync(str(path), memory)
    try:
        before = (await first.catalog('alpha'))[0].configuration
        assert (await first.catalog('bravo')) == ()
    finally:
        first.close_idle()
    body['collections'][0]['parser_revision'] = 'installed-v2'
    path.write_text(json.dumps(body))
    second = load_directory_sync(str(path), memory)
    try:
        assert (await second.catalog('alpha'))[0].configuration != before
    finally:
        second.close_idle()


@pytest.mark.parametrize('composed', [False, True])
async def test_app_construction_failure_closes_new_sync_registry(env, monkeypatch, composed):
    from scone_memory.api import __main__ as host_module
    from scone_memory.runtime import directory_sync
    memory, _, parent = env
    monkeypatch.setenv('TEST_SYNC_KEY', 'ab' * 32)
    path = config(env)
    instances = []
    original = directory_sync.load_directory_sync
    def capture(*args, **kwargs):
        service = original(*args, **kwargs)
        instances.append(service)
        return service
    def broken(*args, **kwargs):
        raise RuntimeError('app construction failed')
    monkeypatch.setattr(directory_sync, 'load_directory_sync', capture)
    monkeypatch.setattr(host_module, '_build_app', broken)
    with pytest.raises(RuntimeError, match='app construction failed'):
        build_app(settings(path, parent, composed), memory)
    assert len(instances) == 1 and instances[0]._closed


async def test_collection_pdf_ocr_preserves_text_and_selected_media_dispatch(env, monkeypatch):
    import shutil
    from scone_memory.ingestion.document_media import DocumentMedia
    from scone_memory.ingestion.document_ocr import DocumentOcr
    from scone_memory.ingestion.files import document_provenance
    from scone_memory.ingestion.formats.media import MediaDocumentParser, TranscriptionSegment
    from tests.api.test_document_ocr import Recognizer
    from tests.api.test_media_documents import audio_bytes
    from tests.ingestion.test_pdf_ingestion import pdf_bytes
    pytest.importorskip('pypdfium2')
    decoder = shutil.which('ffmpeg')
    if decoder is None:
        pytest.skip('configured local decoder unavailable')
    memory, root, _ = env
    monkeypatch.setenv('TEST_SYNC_KEY', 'ab' * 32)
    path = config(env)
    body = json.loads(path.read_text())
    body['collections'][0]['pdf_ocr'] = {'mode': 'all_pages', 'reading_order': 'provider'}
    path.write_text(json.dumps(body))
    (root / 'note.txt').write_text('Text stays on its native parser')
    (root / 'scan.pdf').write_bytes(pdf_bytes(pages=('',)))
    (root / 'clip.wav').write_bytes(audio_bytes())
    calls = []
    async def transcribe(data):
        calls.append(data)
        return (TranscriptionSegment(text='Scripted transcript for structural testing', start_seconds=0.0, end_seconds=.15),)
    recognizer = Recognizer()
    media = DocumentMedia(MediaDocumentParser(transcribe, ffmpeg_executable=decoder), revision='selected-local-v1')
    host = load_directory_sync(str(path), memory, document_ocr=DocumentOcr(recognizer),
                               ocr_identity='fixture-recognizer-v1', document_media=media)
    try:
        assert recognizer.calls == 0 and calls == []
        await host.start('alpha', 'scan', collection_id='notes')
        assert (await wait_service(host)).status == 'completed'
        outcomes = (await host.result('alpha', 'scan')).items
        assert {item.source.path for item in outcomes} == {'note.txt', 'scan.pdf', 'clip.wav'}
        evidence = {item.source.path: await document_provenance(memory, 'alpha', item.source.episode_id) for item in outcomes}
        assert evidence['note.txt'].segments[0].text == 'Text stays on its native parser'
        assert evidence['scan.pdf'].segments[0].text == 'Café Polaris'
        assert json.loads(evidence['scan.pdf'].metadata['pdf_ocr'])['mode'] == 'all_pages'
        assert evidence['clip.wav'].metadata['transcriber_revision'] == 'selected-local-v1'
        assert len(calls) == recognizer.calls == 1
    finally:
        await host.aclose()


@pytest.mark.parametrize('failure', ['space', 'missing_root', 'extensions', 'unavailable_ocr'])
async def test_invalid_source_configuration_is_a_safe_startup_error(env, monkeypatch, failure):
    memory, _, parent = env
    monkeypatch.setenv('TEST_SYNC_KEY', 'ab' * 32)
    path = config(env)
    body = json.loads(path.read_text())
    collection = body['collections'][0]
    if failure == 'space':
        collection['space'] = ''
    elif failure == 'missing_root':
        collection['root'] = str(parent / 'missing-source')
    elif failure == 'extensions':
        collection['extensions'] = ['../txt']
    else:
        collection['pdf_ocr'] = {'mode': 'all_pages', 'reading_order': 'provider'}
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        load_directory_sync(str(path), memory)
    assert not list(parent.rglob('runs.sqlite'))


@pytest.mark.parametrize('composed', [False, True])
def test_server_failure_before_lifespan_still_closes_sync(tmp_path, monkeypatch, composed):
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.api import __main__ as host_module
    root = tmp_path / 'sources'
    root.mkdir()
    path = config((None, root, tmp_path))
    monkeypatch.setenv('TEST_SYNC_KEY', 'ab' * 32)
    instances = []
    original = host_module.build_app
    async def engine(settings):
        return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    def capture(settings, memory):
        app = original(settings, memory)
        target = getattr(app.state, 'memory_app', app)
        instances.append(target.state.directory_sync_service)
        return app
    def fail(*args):
        raise RuntimeError('server construction failed')
    monkeypatch.setattr(host_module, 'build_engine', engine)
    monkeypatch.setattr(host_module, 'build_app', capture)
    monkeypatch.setattr(host_module, 'build_server', fail)
    with pytest.raises(RuntimeError, match='server construction failed'):
        host_module.main(settings(path, tmp_path, composed))
    assert len(instances) == 1 and instances[0]._closed


async def test_one_constructor_cleanup_error_does_not_skip_other_resources(env, monkeypatch):
    from scone_memory.api import __main__ as host_module
    from scone_memory.runtime import directory_sync, document_jobs
    from tests.runtime.test_document_jobs_runtime import config as jobs_config
    memory, _, parent = env
    monkeypatch.setenv('TEST_SYNC_KEY', 'ab' * 32)
    monkeypatch.setenv('TEST_IMPORT_KEY', 'cd' * 32)
    path = config(env)
    values = settings(path, parent)
    from dataclasses import replace
    values = replace(values, document_jobs_config=str(jobs_config(parent)))
    imports = []
    original_sync, original_jobs = directory_sync.load_directory_sync, document_jobs.load_document_imports
    def capture_sync(*args, **kwargs):
        service = original_sync(*args, **kwargs)
        close = service.close_idle
        def broken_close():
            close()
            raise RuntimeError('cleanup failed after closing sync')
        service.close_idle = broken_close
        return service
    def capture_jobs(*args, **kwargs):
        value = original_jobs(*args, **kwargs)
        imports.append(value)
        return value
    def broken_app(*args, **kwargs):
        raise RuntimeError('app construction failed')
    monkeypatch.setattr(directory_sync, 'load_directory_sync', capture_sync)
    monkeypatch.setattr(document_jobs, 'load_document_imports', capture_jobs)
    monkeypatch.setattr(host_module, '_build_app', broken_app)
    try:
        with pytest.raises(RuntimeError, match='cleanup failed'):
            build_app(values, memory)
        assert imports and imports[0]._closed
    finally:
        for value in imports:
            value.close_idle()


async def test_registry_cannot_be_reopened_as_a_different_memory_catalog(env, monkeypatch):
    from scone_memory.agents.workflow import WorkflowError
    memory, _, _ = env
    monkeypatch.setenv('TEST_SYNC_KEY', 'ab' * 32)
    path = config(env)
    original = load_directory_sync(str(path), memory)
    await original.start('alpha', 'scan', collection_id='notes')
    await wait_service(original)
    await original.aclose()
    body = json.loads(path.read_text())
    body['store_id'] = 'replacement-memory-catalog'
    path.write_text(json.dumps(body))
    replacement = None
    try:
        with pytest.raises(WorkflowError, match='key_or_integrity'):
            replacement = load_directory_sync(str(path), memory)
    finally:
        if replacement is not None:
            replacement.close_idle()
