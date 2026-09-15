"""A request can pick the OCR language for its file, from the languages the host offers.

A host was configured with one Tesseract language for every scanned file,
so a German invoice and a Japanese receipt were read with English data,
and a caller could not say otherwise. The host now lists the languages a
request may choose (``SCONE_DOCUMENT_OCR_LANGUAGES``); a request names one
in ``pdf_ocr.language``, and anything the host did not list is refused
before any file is read. A request that names none is read exactly as
before, down to its retained extraction identity.
"""
import json

import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.document_ocr import DocumentOcr, PdfOcrSelection
from scone_memory.ocr import OcrRegion, OcrResult
from scone_memory.ocr.tesseract import png_dimensions
from ..ingestion.test_pdf_ingestion import pdf_bytes


class Recognizer:
    def __init__(self, language):
        self.language = language
        self.calls = 0

    async def recognize(self, image, *, max_pixels, max_regions, timeout_seconds):
        self.calls += 1
        width, height = png_dimensions(image, max_pixels)
        return OcrResult(engine=f'observed:{self.language}', width=width, height=height,
                         regions=(OcrRegion(text=f'read as {self.language}', box=(.1, .1, .9, .2), score=.8),))


@pytest.fixture
async def host():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    made = {}

    def engine_for(language):
        made[language] = Recognizer(language)
        return made[language]
    ocr = DocumentOcr(Recognizer('eng'), languages=('deu', 'jpn+eng'), engine_for=engine_for)
    app = create_app(memory, {'write': 'alpha'}, document_ocr=ocr)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://scone.test',
                                headers={'authorization': 'Bearer write'}) as client:
        yield client, memory, ocr, made
    await memory.close()


async def upload(client):
    response = await client.post('/v1/attachments', content=pdf_bytes(pages=('',)),
                                 headers={'content-type': 'application/octet-stream', 'x-filename': 'scan.pdf'})
    assert response.status_code == 200, response.text
    return {'attachment_id': response.json()['attachment_id'], 'filename': 'scan.pdf'}


async def test_a_request_reads_its_scan_with_the_language_it_chose(host, monkeypatch):
    pytest.importorskip('pypdfium2')
    client, memory, ocr, made = host
    choice = {'mode': 'all_pages', 'reading_order': 'provider', 'language': 'deu'}
    response = await client.post('/v1/documents', json={**await upload(client), 'pdf_ocr': choice})
    assert response.status_code == 200, response.text
    assert made['deu'].calls == 1 and ocr.engine.calls == 0
    evidence = (await client.get(f"/v1/episodes/{response.json()['added']['episode_id']}/document")).json()
    assert evidence['segments'][-1]['text'] == 'read as deu'
    assert json.loads(evidence['metadata']['pdf_ocr']) == {**choice, 'dpi': 150}


async def test_a_language_the_host_did_not_offer_is_refused_before_anything_is_read(host):
    client, memory, ocr, made = host
    for language in ('fra', 'eng'):
        response = await client.post('/v1/documents', json={'attachment_id': 'a' * 64, 'pdf_ocr': {
            'mode': 'all_pages', 'reading_order': 'provider', 'language': language}})
        assert response.status_code == 422 and 'offered' in response.text, response.text
    malformed = await client.post('/v1/documents', json={'attachment_id': 'a' * 64, 'pdf_ocr': {
        'mode': 'all_pages', 'reading_order': 'provider', 'language': '../eng'}})
    assert malformed.status_code == 400
    assert made == {} and ocr.engine.calls == 0
    assert (await memory.documents.counts('alpha')).episodes == 0


async def test_the_catalog_lists_the_languages_a_request_may_choose(host, monkeypatch):
    monkeypatch.setattr('scone_memory.ingestion.document_ocr.find_spec', lambda _: object())
    client, _, _, _ = host
    catalog = (await client.get('/v1/documents/formats')).json()
    assert catalog['pdf_ocr']['languages'] == ['deu', 'jpn+eng']


def test_a_selection_without_a_language_serializes_as_it_always_did():
    plain = PdfOcrSelection(mode='all_pages', reading_order='provider')
    assert plain.model_dump() == {'mode': 'all_pages', 'reading_order': 'provider'}
    assert json.loads(plain.model_dump_json()) == {'mode': 'all_pages', 'reading_order': 'provider'}
    chosen = PdfOcrSelection(mode='all_pages', reading_order='provider', language='deu')
    assert chosen.model_dump() == {'mode': 'all_pages', 'reading_order': 'provider', 'language': 'deu'}


def test_a_host_that_offers_languages_must_say_how_to_read_them():
    with pytest.raises(ValueError):
        DocumentOcr(Recognizer('eng'), languages=('deu',))
    with pytest.raises(InvalidInput, match='offered'):
        DocumentOcr(Recognizer('eng')).parser(PdfOcrSelection(mode='all_pages', reading_order='provider', language='deu'))
