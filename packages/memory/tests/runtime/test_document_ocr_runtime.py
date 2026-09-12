"""No OCR is enabled until an operator selects a local executable."""
import shutil

import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.__main__ import build_app
from scone_memory.runtime.config import Settings
from scone_memory.runtime.document_ocr import build_document_ocr


def test_default_launcher_ocr_is_disabled():
    assert build_document_ocr(Settings.from_env({})) is None


@pytest.mark.parametrize('key,value', [
    ('SCONE_DOCUMENT_OCR_EXECUTABLE', 'tesseract'),
    ('SCONE_DOCUMENT_OCR_EXECUTABLE', '/missing/tesseract'),
    ('SCONE_DOCUMENT_OCR_EXECUTABLE', '/tmp'),
    ('SCONE_DOCUMENT_OCR_LANGUAGE', '../language'),
    ('SCONE_DOCUMENT_OCR_PSM', '5'),
    ('SCONE_DOCUMENT_OCR_DPI', '301'),
    ('SCONE_DOCUMENT_OCR_DPI', '71'),
])
def test_invalid_explicit_ocr_configuration_is_refused(key, value):
    env = {'SCONE_DOCUMENT_OCR_EXECUTABLE': '/usr/bin/true', key: value}
    with pytest.raises(ValueError):
        build_document_ocr(Settings.from_env(env))


def test_missing_dependencies_refuse_configured_ocr_before_executing(monkeypatch):
    monkeypatch.setattr('scone_memory.ingestion.document_ocr.find_spec', lambda _: None)
    with pytest.raises(ValueError, match='pdf-ocr'):
        build_document_ocr(Settings.from_env({'SCONE_DOCUMENT_OCR_EXECUTABLE': '/usr/bin/true'}))


@pytest.mark.parametrize('composed', [False, True])
async def test_standard_hosts_mount_ocr_and_retain_real_scan(tmp_path, composed):
    pytest.importorskip('pypdfium2')
    executable = shutil.which('tesseract')
    if executable is None:
        pytest.skip('explicit local Tesseract integration')
    from ..ingestion.test_pdf_ocr import scanned_pdf
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    env = {'SCONE_API_KEY': 'key', 'SCONE_DOCUMENT_OCR_EXECUTABLE': executable,
           'SCONE_DOCUMENT_OCR_PSM': '6', 'SCONE_DOCUMENT_OCR_DPI': '150'}
    if composed:
        env['SCONE_CONVERSATIONS_JOURNAL'] = str(tmp_path/'conversations.sqlite')
    try:
        app = build_app(Settings.from_env(env), memory)
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://scone.test',
            headers={'authorization': 'Bearer key'}) as client:
            assert (await client.get('/v1/documents/formats')).json()['pdf_ocr']['available'] is True
            uploaded = await client.post('/v1/attachments', content=scanned_pdf(),
                headers={'content-type': 'application/pdf', 'x-filename': 'scan.pdf'})
            choice = {'mode': 'missing_text', 'reading_order': 'columns_ltr'}
            response = await client.post('/v1/documents', json={
                'attachment_id': uploaded.json()['attachment_id'], 'pdf_ocr': choice})
            assert response.status_code == 200, response.text
            episode_id = response.json()['added']['episode_id']
            episode = (await client.get(f'/v1/episodes/{episode_id}')).json()
            assert 'Polaris' in episode['content']
            evidence = (await client.get(f'/v1/episodes/{episode_id}/document')).json()
            assert evidence['segments'][0]['metadata']['ocr_engine'] == 'tesseract:eng:psm6'
            assert any(region['text'] == 'Polaris' for region in evidence['segments'][0]['regions'])
    finally:
        await memory.close()
