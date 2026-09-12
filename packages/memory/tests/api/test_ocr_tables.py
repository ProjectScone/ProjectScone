"""Read-only OCR table analysis is bound to retained, authorized PDF evidence."""
import hashlib

import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.ingestion.files import ingest_document
from scone_memory.ingestion.formats.registry import BuiltinDocumentParser
from scone_memory.ingestion.pdf import ParsedPdf, PdfPage, PdfLimits
from scone_memory.ingestion.pdf_ocr import assemble_ocr_pdf
from scone_memory.ocr.types import OcrResult
from ..ingestion.test_ocr_tables import grid


class RetainedParser:
    calls=0
    async def parse(self, data, limits):
        self.calls+=1
        empty=ParsedPdf(text='', parser='fixture', pages=(PdfPage(number=1,start=0,end=0,
            width_points=600.,height_points=800.,rotation=0,empty=True),))
        return assemble_ocr_pdf(empty,{1:OcrResult(engine='fixture',width=600,height=800,regions=tuple(grid()))},PdfLimits())


@pytest.fixture
async def service():
    memory=await MemoryEngine(InMemoryDocumentStore(),InMemoryVectorIndex(),HashEmbedder()).open()
    parser=RetainedParser()
    saved=await ingest_document(memory,'alpha',b'%PDF-test-source',filename='table.pdf',
        parser=BuiltinDocumentParser(pdf_parser=parser))
    app=create_app(memory,{'reader':'alpha','other':'beta'},roles={'reader':'read'})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test',
                                headers={'authorization':'Bearer reader'}) as client:
        yield memory,client,saved,parser
    await memory.close()


async def test_table_read_preserves_source_binding_and_never_runs_ocr(service):
    memory,client,saved,parser=service
    before=await memory.documents.revision('alpha')
    response=await client.get(f'/v1/episodes/{saved.added.episode_id}/document/ocr-tables?page=1')
    assert response.status_code==200,response.text
    body=response.json()
    assert body['space']=='alpha' and body['episode_id']==saved.added.episode_id
    assert body['original_sha256']==saved.original.attachment_id
    assert body['manifest_sha256']==saved.manifest.attachment_id
    evidence=(await client.get(f'/v1/episodes/{saved.added.episode_id}/document')).json()
    assert body['page_text_sha256']==hashlib.sha256(evidence['segments'][0]['text'].encode()).hexdigest()
    assert body['layout']['origin']=='geometry_inferred'
    assert len(body['layout']['tables'])==1
    assert (await memory.documents.revision('alpha'))==before and parser.calls==1
    assert (await client.get('/v1/capabilities')).json()['features']['documents.ocr.tables'] is True


async def test_scope_deleted_source_and_page_bounds(service):
    memory,client,saved,_=service
    path=f'/v1/episodes/{saved.added.episode_id}/document/ocr-tables'
    assert (await client.get(path+'?page=1',headers={'authorization':'Bearer other'})).status_code==404
    for query in ('','?page=0','?page=1001','?page=bad','?page=2'):
        assert (await client.get(path+query)).status_code==422
    await memory.forget('alpha',saved.added.episode_id)
    assert (await client.get(path+'?page=1')).status_code==410


async def test_source_forgotten_during_analysis_cannot_return_stale_layout(service,monkeypatch):
    memory,client,saved,_=service
    original=memory.episode
    calls=0
    async def disappearing(space,identity):
        nonlocal calls
        calls+=1
        if calls==2:
            await memory.forget(space,identity)
        return await original(space,identity)
    monkeypatch.setattr(memory,'episode',disappearing)
    response=await client.get(f'/v1/episodes/{saved.added.episode_id}/document/ocr-tables?page=1')
    assert response.status_code==410,response.text


@pytest.mark.parametrize('change',['rebind','revoke'])
async def test_authority_is_rechecked_after_the_final_source_read(service,monkeypatch,change):
    memory,client,saved,_=service
    original=memory.episode
    calls=0
    async def changing(space,identity):
        nonlocal calls
        held=await original(space,identity)
        calls+=1
        if calls==2:
            keys=client._transport.app.state.keys
            if change=='rebind':
                keys['reader']='beta'
            else:
                keys.pop('reader')
        return held
    monkeypatch.setattr(memory,'episode',changing)
    response=await client.get(f'/v1/episodes/{saved.added.episode_id}/document/ocr-tables?page=1')
    assert response.status_code in (401,403),response.text
    assert 'tables' not in response.json()
