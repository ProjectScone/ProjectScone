"""Explicit local video OCR settings bind evidence without starting tools."""
import json

import pytest

from scone_memory.runtime.document_video import load_document_video


def configuration(tmp_path, **changes):
    for name in ('ffmpeg', 'ffprobe', 'tesseract'):
        executable = tmp_path / name
        if not executable.exists():
            executable.write_text('#!/bin/sh\nexit 1\n')
            executable.chmod(0o700)
    path = tmp_path / 'video.json'
    path.write_text(json.dumps({'schema_version': 1, 'model_revision': 'trained-data-v1',
        'ffmpeg_executable': str(tmp_path / 'ffmpeg'), 'ffprobe_executable': str(tmp_path / 'ffprobe'),
        'ocr_executable': str(tmp_path / 'tesseract'), **changes}))
    path.chmod(0o600)
    return path


def test_loading_does_not_execute_tools_and_has_stable_identity(tmp_path):
    path = configuration(tmp_path)
    first = load_document_video(str(path))
    assert first.revision == load_document_video(str(path)).revision
    assert first.parser is not None


@pytest.mark.parametrize('changes', [
    {'model_revision': 'trained-data-v2'}, {'language': 'fra'}, {'page_segmentation': 6},
    {'policy': {'interval_seconds': 2}},
])
def test_effective_ocr_configuration_binds_revision(tmp_path, changes):
    first = load_document_video(str(configuration(tmp_path)))
    assert first.revision != load_document_video(str(configuration(tmp_path, **changes))).revision


@pytest.mark.parametrize('executable', ['ffmpeg', 'ffprobe', 'tesseract'])
def test_each_binary_binds_revision(tmp_path, executable):
    path = configuration(tmp_path)
    first = load_document_video(str(path))
    (tmp_path / executable).write_text('#!/bin/sh\nexit 2\n')
    assert first.revision != load_document_video(str(path)).revision


@pytest.mark.parametrize('damage', ['mode', 'symlink', 'hardlink', 'duplicate', 'oversize', 'unknown', 'boolean', 'relative'])
def test_invalid_configuration_is_refused_without_leaking_values(tmp_path, damage):
    path = configuration(tmp_path)
    if damage == 'mode':
        path.chmod(0o644)
    elif damage == 'symlink':
        link = tmp_path / 'link'
        link.symlink_to(path)
        path = link
    elif damage == 'hardlink':
        (tmp_path / 'link').hardlink_to(path)
    elif damage == 'duplicate':
        path.write_text('{"schema_version":1,"schema_version":1}')
    elif damage == 'oversize':
        path.write_bytes(b' ' * 16385)
    elif damage == 'unknown':
        path = configuration(tmp_path, secret='secret-marker')
    elif damage == 'boolean':
        path = configuration(tmp_path, schema_version=True)
    else:
        path = configuration(tmp_path, ocr_executable='tesseract')
    with pytest.raises(ValueError) as error:
        load_document_video(str(path))
    assert 'secret-marker' not in str(error.value)


@pytest.mark.parametrize('composed', [False, True])
async def test_standard_hosts_load_config_without_running_tools(tmp_path, composed):
    import httpx
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.api.__main__ import build_app
    from scone_memory.runtime.config import Settings
    env = {'SCONE_API_KEYS': 'reader:alpha:read', 'SCONE_DOCUMENT_VIDEO_CONFIG': str(configuration(tmp_path))}
    if composed:
        env['SCONE_CONVERSATIONS_JOURNAL'] = str(tmp_path / 'conversation.sqlite')
    settings = Settings.from_env(env)
    assert settings.document_video_config == env['SCONE_DOCUMENT_VIDEO_CONFIG']
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        app = build_app(settings, engine)
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app),
            base_url='http://fixture', headers={'authorization': 'Bearer reader'}) as client:
            formats = (await client.get('/v1/documents/formats')).json()
            assert formats['video_ocr']['available'] is True
            assert formats['pdf_ocr']['available'] is False, 'video OCR must not require PDF extras'
    finally:
        await engine.close()


async def test_replaced_ocr_binary_refuses_before_invocation(tmp_path):
    from scone_memory.core.errors import InvalidInput
    configured = load_document_video(str(configuration(tmp_path)))
    (tmp_path / 'tesseract').write_text('#!/bin/sh\nexit 2\n')
    with pytest.raises(InvalidInput, match='executable changed'):
        await configured.parser._engine.recognize(b'not-an-image')


def test_default_import_choice_preserves_legacy_serialization():
    from scone_memory.ingestion.import_store import DocumentImportSpec
    spec = DocumentImportSpec(attachment_id='a' * 64, filename='notes.txt', parser_revision='v1')
    assert 'video_ocr' not in spec.model_dump()
    assert b'video_ocr' not in spec.model_dump_json().encode()
    assert DocumentImportSpec.model_validate_json(spec.model_dump_json()) == spec
    selected = DocumentImportSpec(attachment_id='a' * 64, filename='slides.mp4', parser_revision='v1', video_ocr=True)
    assert selected.model_dump()['video_ocr'] is True
