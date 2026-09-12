"""Explicit local media configuration and durable extraction identity."""
import json
import os
from pathlib import Path

import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.__main__ import build_app
from scone_memory.runtime.config import Settings
from scone_memory.runtime.document_media import load_document_media


def configuration(tmp_path, **changes):
    executable=tmp_path/'ffmpeg'
    if not executable.exists():
        executable.write_text('#!/bin/sh\nexit 1\n');executable.chmod(0o700)
    path=tmp_path/'media.json'
    path.write_text(json.dumps({'schema_version':1,'base_url':'http://127.0.0.1:9876/v1',
        'model':'selected-local-model','model_revision':'weights-v1','ffmpeg_executable':str(executable),**changes}))
    path.chmod(0o600)
    return path


def test_load_configures_without_calling_the_endpoint(tmp_path):
    first=load_document_media(str(configuration(tmp_path)))
    assert first.formats()['.wav']['available'] is True
    second=load_document_media(str(tmp_path/'media.json'))
    assert first.revision==second.revision
    assert 'selected-local-model' not in first.revision


def test_decoder_replacement_changes_identity_even_with_same_size_and_mtime(tmp_path):
    path = configuration(tmp_path)
    executable = tmp_path / 'ffmpeg'
    first = load_document_media(str(path))
    original = executable.stat()
    executable.write_text('#!/bin/sh\nexit 2\n')
    os.utime(executable, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert executable.stat().st_size == original.st_size
    assert load_document_media(str(path)).revision != first.revision


def test_decoder_symlink_binds_target_contents_without_executing_it(tmp_path):
    path = configuration(tmp_path)
    executable = tmp_path / 'ffmpeg'
    target = tmp_path / 'decoder-build'
    executable.rename(target)
    executable.symlink_to(target)
    first = load_document_media(str(path))
    target.write_text('#!/bin/sh\nexit 2\n')
    assert load_document_media(str(path)).revision != first.revision


@pytest.mark.parametrize('size', [0, 512 * 1024 * 1024 + 1])
def test_decoder_fingerprint_refuses_empty_or_oversized_executable(tmp_path, size):
    path = configuration(tmp_path)
    with (tmp_path / 'ffmpeg').open('wb') as output:
        output.truncate(size)
    with pytest.raises(ValueError, match='decoder'):
        load_document_media(str(path))


@pytest.mark.parametrize('failure, message', [
    ('permissions', 'configuration file'),
    ('malformed', 'configuration contents'),
    ('missing_secret', 'credential is missing'),
])
def test_configuration_failures_identify_safe_category(tmp_path, failure, message):
    path = configuration(tmp_path)
    if failure == 'permissions':
        path.chmod(0o644)
    elif failure == 'malformed':
        path.write_text('{"secret-marker":')
    else:
        path = configuration(tmp_path, api_key_env='SCONE_MISSING_MEDIA_FIXTURE_KEY')
    with pytest.raises(ValueError, match=message) as error:
        load_document_media(str(path))
    assert 'secret-marker' not in str(error.value)


@pytest.mark.parametrize('changes',[{'model':'other'}, {'model_revision':'weights-v2'},
    {'base_url':'http://127.0.0.1:8765/v1'}, {'max_duration_seconds':30.0}, {'timeout_seconds':60.0},
    {'max_response_bytes':2000000},{'max_segments':5}])
def test_effective_configuration_changes_extraction_identity(tmp_path,changes):
    first=load_document_media(str(configuration(tmp_path)))
    second=load_document_media(str(configuration(tmp_path,**changes)))
    assert first.revision!=second.revision


@pytest.mark.parametrize('mode',['permissions','symlink','hardlink','oversize','duplicate','unknown','boolean','missing_secret','public_endpoint'])
def test_invalid_configuration_refused_without_exposing_contents(tmp_path,mode):
    path=configuration(tmp_path)
    if mode=='permissions': path.chmod(0o644)
    elif mode=='symlink':
        link=tmp_path/'link';link.symlink_to(path);path=link
    elif mode=='hardlink': (tmp_path/'hard').hardlink_to(path)
    elif mode=='oversize':path.write_text(' '*20000)
    elif mode=='duplicate':path.write_text('{"schema_version":1,"schema_version":1}')
    elif mode=='unknown':path=configuration(tmp_path,api_key='secret-marker')
    elif mode=='boolean':path=configuration(tmp_path,schema_version=True)
    elif mode=='missing_secret':path=configuration(tmp_path,api_key_env='SCONE_MISSING_MEDIA_FIXTURE_KEY')
    elif mode=='public_endpoint':path=configuration(tmp_path,base_url='https://public.example/v1')
    with pytest.raises(ValueError) as error:load_document_media(str(path))
    assert 'secret-marker' not in str(error.value)


def test_secret_is_resolved_explicitly_and_is_not_part_of_public_identity(tmp_path,monkeypatch):
    path=configuration(tmp_path,api_key_env='SCONE_MEDIA_FIXTURE_KEY')
    monkeypatch.setenv('SCONE_MEDIA_FIXTURE_KEY','first-secret')
    first=load_document_media(str(path))
    monkeypatch.setenv('SCONE_MEDIA_FIXTURE_KEY','second-secret')
    second=load_document_media(str(path))
    assert first.revision==second.revision
    assert 'secret' not in repr(first)


@pytest.mark.parametrize('composed',[False,True])
async def test_standard_host_loads_media_for_both_compositions(tmp_path,composed):
    path=configuration(tmp_path)
    env={'SCONE_API_KEYS':'reader:alpha:read','SCONE_DOCUMENT_MEDIA_CONFIG':str(path)}
    if composed:env['SCONE_CONVERSATIONS_JOURNAL']=str(tmp_path/'conversations.db')
    settings=Settings.from_env(env)
    assert settings.document_media_config==str(path)
    engine=await MemoryEngine(InMemoryDocumentStore(),InMemoryVectorIndex(),HashEmbedder()).open()
    try:
        app=build_app(settings,engine)
        async with app.router.lifespan_context(app),httpx.AsyncClient(transport=httpx.ASGITransport(app),
            base_url='http://fixture',headers={'authorization':'Bearer reader'}) as client:
            formats=(await client.get('/v1/documents/formats')).json()['formats']
            assert formats['.wav']['available'] is True
            assert formats['.ts']['parser']=='text'
    finally:await engine.close()


def test_chunk_policy_is_opt_in_and_binds_the_host_revision(tmp_path):
    import hashlib
    from scone_memory.runtime.document_media import DocumentMediaConfig, _decoder_digest

    path = configuration(tmp_path)
    whole = load_document_media(str(path))
    settings = DocumentMediaConfig.model_validate_json(path.read_bytes()).model_dump(exclude={'api_key_env', 'chunk_seconds'})
    raw = json.dumps({'implementation': 'local-document-media-v2', 'settings': settings,
        'decoder_sha256': _decoder_digest(str(tmp_path / 'ffmpeg'))}, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False).encode()
    assert whole.revision == 'media-' + hashlib.sha256(raw).hexdigest(), 'retain existing whole-file bindings'
    chunked = load_document_media(str(configuration(tmp_path, chunk_seconds=30)))
    assert chunked.revision != whole.revision
    assert chunked.media_parser._chunk_seconds == 30
    assert chunked.media_parser._transcribe.__self__._allow_empty is True
    changed = load_document_media(str(configuration(tmp_path, chunk_seconds=60)))
    assert changed.revision != chunked.revision
    disabled = load_document_media(str(configuration(tmp_path, chunk_seconds=None)))
    assert disabled.revision == whole.revision


def test_whole_file_revision_matches_the_pre_window_host_fixture(monkeypatch):
    import scone_memory.runtime.document_media as runtime

    executable = '/opt/scone/fixture-ffmpeg'
    config = runtime.DocumentMediaConfig(schema_version=1, base_url='http://127.0.0.1:9876/v1',
        model='selected-local-model', model_revision='weights-v1', ffmpeg_executable=executable)
    is_file, access = Path.is_file, os.access
    monkeypatch.setattr(runtime, '_read', lambda _: config)
    monkeypatch.setattr(runtime, '_decoder_digest', lambda _: 'a' * 64)
    monkeypatch.setattr(Path, 'is_file', lambda path: str(path) == executable or is_file(path))
    monkeypatch.setattr(os, 'access', lambda path, mode: str(path) == executable or access(path, mode))
    # Captured from the loader at 4ea58e8, before audio window configuration existed.
    assert load_document_media('unused').revision == 'media-421b087ae19297b2ca4d743cc8dab3b53d276c46e7b306af1ba824fa05f64422'


@pytest.mark.parametrize('chunk_seconds', [True, 0, 121, '30', 1.5])
def test_invalid_chunk_configuration_is_refused_without_contact(tmp_path, chunk_seconds):
    with pytest.raises(ValueError, match='configuration contents'):
        load_document_media(str(configuration(tmp_path, chunk_seconds=chunk_seconds)))
