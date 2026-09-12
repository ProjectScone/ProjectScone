"""Configured OCR follows the same authorized original-backed import route."""
import json

import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.ingestion.document_ocr import DocumentOcr
from scone_memory.ocr import OcrRegion, OcrResult
from scone_memory.ocr.tesseract import png_dimensions
from ..ingestion.test_pdf_ingestion import pdf_bytes


class Recognizer:
    def __init__(self):
        self.calls = 0

    async def recognize(self, image, *, max_pixels, max_regions, timeout_seconds):
        self.calls += 1
        width, height = png_dimensions(image, max_pixels)
        return OcrResult(engine='observed-local', width=width, height=height,
                         regions=(OcrRegion(text='Café Polaris', box=(.1, .1, .9, .2), score=.8),))


@pytest.fixture
async def host():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    recognizer = Recognizer()
    app = create_app(memory, {'write': 'alpha', 'read': 'alpha', 'other': 'beta'},
                     roles={'read': 'read'}, document_ocr=DocumentOcr(recognizer))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://scone.test',
                                headers={'authorization': 'Bearer write'}) as client:
        yield client, memory, recognizer
    await memory.close()


async def upload(client, raw=None, filename='scan.pdf'):
    response = await client.post('/v1/attachments', content=raw or pdf_bytes(pages=('Native text.', '')),
        headers={'content-type': 'application/octet-stream', 'x-filename': filename})
    assert response.status_code == 200, response.text
    return {'attachment_id': response.json()['attachment_id'], 'filename': filename}


@pytest.mark.parametrize('mode,expected_calls', [('missing_text', 1), ('all_pages', 2)])
@pytest.mark.parametrize('order', ['provider', 'columns_ltr', 'columns_rtl'])
async def test_selected_ocr_retains_options_geometry_original_and_dedup(host, mode, expected_calls, order):
    pytest.importorskip('pypdfium2')
    client, memory, recognizer = host
    raw = pdf_bytes(pages=('Native text.', ''))
    body = await upload(client, raw)
    choice = {'mode': mode, 'reading_order': order}
    response = await client.post('/v1/documents', json={**body, 'pdf_ocr': choice})
    assert response.status_code == 200, response.text
    assert recognizer.calls == expected_calls
    receipt = response.json()
    assert receipt['pdf_ocr'] == choice
    episode_id = receipt['added']['episode_id']
    evidence = (await client.get(f'/v1/episodes/{episode_id}/document')).json()
    assert json.loads(evidence['metadata']['pdf_ocr']) == {**choice, 'dpi': 150}
    segments = evidence['segments']
    assert segments[-1]['text'] == 'Café Polaris'
    region = segments[-1]['regions'][0]
    assert region['end'] == len('Café Polaris'.encode())
    assert region['box'] == [.1, .1, .9, .2]
    assert segments[0]['metadata']['extraction'] == ('text_layer' if mode == 'missing_text' else 'ocr')
    assert (await client.get(evidence['download_path'])).content == raw
    repeated = await client.post('/v1/documents', json={**body, 'pdf_ocr': choice})
    assert repeated.json()['added']['episode_id'] == episode_id
    assert repeated.json()['added']['deduplicated'] is True
    assert repeated.json()['added']['chunks'] == 0
    _, manifest = await memory.attachment('alpha', receipt['manifest']['attachment_id'])
    assert json.loads(manifest)['parsed']['metadata'] == evidence['metadata']
    assert (await client.get(f'/v1/episodes/{episode_id}/document',
        headers={'authorization': 'Bearer other'})).status_code == 404


async def test_default_does_not_invoke_ocr_or_change_manifest_identity(host):
    from scone_memory.ingestion.files import ingest_document
    client, memory, recognizer = host
    raw = pdf_bytes(pages=('Native text.',))
    native = await ingest_document(memory, 'alpha', raw, filename='scan.pdf')
    response = await client.post('/v1/documents', json=await upload(client, raw))
    assert response.status_code == 200, response.text
    assert response.json()['manifest']['attachment_id'] == native.manifest.attachment_id
    assert 'pdf_ocr' not in response.json()
    assert recognizer.calls == 0


@pytest.mark.parametrize('choice', [True, {}, {'mode': 'unknown', 'reading_order': 'provider'},
    {'mode': 'missing_text', 'reading_order': 'unknown'},
    {'mode': 'all_pages', 'reading_order': 'provider', 'dpi': 300},
    {'mode': 'all_pages', 'reading_order': 'provider', 'executable': '/bin/anything'}])
async def test_invalid_selection_is_rejected_without_attachment_read_or_ocr(host, choice):
    client, memory, recognizer = host
    response = await client.post('/v1/documents', json={'attachment_id': 'a'*64, 'pdf_ocr': choice})
    assert response.status_code == 400, response.text
    assert (await memory.documents.counts('alpha')).episodes == 0
    assert recognizer.calls == 0


async def test_non_pdf_and_unauthorized_request_never_use_ocr(host):
    client, memory, recognizer = host
    body = await upload(client, b'ordinary text', 'note.txt')
    body['pdf_ocr'] = {'mode': 'all_pages', 'reading_order': 'provider'}
    assert (await client.post('/v1/documents', json=body)).status_code == 422
    assert (await client.post('/v1/documents', json=body,
        headers={'authorization': 'Bearer read'})).status_code == 403
    assert (await client.post('/v1/documents', json=body,
        headers={'authorization': 'Bearer other'})).status_code == 404
    assert recognizer.calls == 0
    assert (await memory.documents.counts('alpha')).episodes == 0


async def test_discovery_and_unconfigured_refusal(host):
    client, memory, _ = host
    catalog = (await client.get('/v1/documents/formats')).json()
    assert catalog['pdf_ocr'] == {'available': True, 'modes': ['missing_text', 'all_pages'],
        'reading_orders': ['provider', 'columns_ltr', 'columns_rtl']}
    app = create_app(memory, {'write': 'alpha'})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://scone.test',
                                headers={'authorization': 'Bearer write'}) as plain:
        assert (await plain.get('/v1/documents/formats')).json()['pdf_ocr']['available'] is False
        response = await plain.post('/v1/documents', json={'attachment_id': 'a'*64,
            'pdf_ocr': {'mode': 'all_pages', 'reading_order': 'provider'}})
        assert response.status_code == 422 and 'configured' in response.text


async def test_different_choices_have_distinct_retained_extraction_identity(host):
    pytest.importorskip('pypdfium2')
    client, _, _ = host
    body = await upload(client, pdf_bytes(pages=('',)))
    receipts = []
    for mode in ('missing_text', 'all_pages'):
        response = await client.post('/v1/documents', json={**body,
            'pdf_ocr': {'mode': mode, 'reading_order': 'provider'}})
        assert response.status_code == 200, response.text
        receipts.append(response.json())
    assert receipts[0]['manifest']['attachment_id'] != receipts[1]['manifest']['attachment_id']
