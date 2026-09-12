"""Explicit local media configuration and durable extraction identity."""
import json
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
