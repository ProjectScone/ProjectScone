"""Generic file citations retain OCR geometry across offset changes and restarts."""
import asyncio
import json

import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.core.errors import InvalidInput
from scone_memory.core.ports import NewChunk
from scone_memory.ingestion import BuiltinDocumentParser, document_provenance, ingest_document
from scone_memory.ingestion.files import DocumentManifest, prepare_document
from scone_memory.ingestion.formats.types import DocumentLimits
from scone_memory.ingestion.pdf import ParsedPdf, PdfPage, PdfTextRegion
from .test_document_workflow import HeldEmbedder, open_memory, workflow


class ObservedPdf:
    async def parse(self, data, limits):
        # Page two is empty: dropping it changes generic-document byte offsets.
        text = 'Native\n\n\n\nCafé Polaris'
        return ParsedPdf(text=text, parser='observed-pdf', pages=(
            PdfPage(number=1, start=0, end=6, width_points=600., height_points=800., rotation=0, empty=False),
            PdfPage(number=2, start=8, end=8, width_points=600., height_points=800., rotation=0, empty=True),
            PdfPage(number=3, start=10, end=23, width_points=600., height_points=800., rotation=90,
                empty=False, extraction='ocr', ocr_engine='observed-v1', regions=(
                    PdfTextRegion(text='Café', start=10, end=15, box=(.1, .2, .3, .4), score=.8, block=2, line=3),
                    PdfTextRegion(text='Polaris', start=16, end=23, box=(.4, .2, .8, .4), score=.9, block=2, line=3))),
        ))


def parser():
    return BuiltinDocumentParser(pdf_parser=ObservedPdf())


@pytest.fixture
async def memory():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        yield engine
    finally:
        await engine.close()


async def test_generic_pdf_citation_preserves_geometry_and_filters_by_chunk(memory):
    result = await ingest_document(memory, 'alpha', b'%PDF-observation', filename='source.pdf', parser=parser())
    episode = await memory.episode('alpha', result.added.episode_id)
    assert episode.content == 'Native\n\nCafé Polaris'
    evidence = await document_provenance(memory, 'alpha', episode.episode_id)
    page = evidence.segments[1]
    assert page.locator == 'page:3' and page.metadata['ocr_engine'] == 'observed-v1'
    assert [(r.text, r.start, r.end) for r in page.regions] == [('Café', 0, 5), ('Polaris', 6, 13)]
    assert page.regions[0].box == (.1, .2, .3, .4)
    assert page.regions[0].score == .8 and page.regions[0].block == 2 and page.regions[0].line == 3
    assert page.regions[0].coordinate_space == 'normalized_displayed_page_top_left'
    chunk, = await memory.documents.insert_chunks([NewChunk(episode_id=episode.episode_id,
        space='alpha', ordinal=99, start=14, end=21, text='Polaris', created_at=episode.created_at)])
    selected = await document_provenance(memory, 'alpha', episode.episode_id, chunk_id=chunk.chunk_id)
    assert len(selected.segments) == 1
    assert [r.text for r in selected.segments[0].regions] == ['Polaris']
    app = create_app(memory, {'reader': 'alpha'}, roles={'reader': 'read'})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.get(f'/v1/episodes/{episode.episode_id}/document',
            params={'chunk_id': chunk.chunk_id}, headers={'authorization': 'Bearer reader'})
    assert response.status_code == 200, response.text
    assert response.json()['segments'][0]['regions'][0]['box'] == [.4, .2, .8, .4]


async def test_plain_manifest_retains_legacy_bytes_and_ocr_uses_version_two(memory):
    result = await ingest_document(memory, 'alpha', b'{"code":"Polaris"}', filename='source.json')
    _, raw = await memory.attachment('alpha', result.manifest.attachment_id)
    expected = b'{"schema_version":1,"offset_unit":"extracted_text_utf8_bytes","original_sha256":"d548bde072fbea8f20e595aff5afa0298102268650876ccde9c9de8721f9239f","filename":"source.json","parsed":{"format":"json","parser":"scone-text-v1","segments":[{"text":"/code: \\"Polaris\\"","locator":"#/code","metadata":{"json_pointer":"/code"}}],"metadata":{}}}'
    assert raw == expected
    assert (await document_provenance(memory, 'alpha', result.added.episode_id)).segments[0].regions == ()
    manifest = await prepare_document(b'%PDF-observation', 'source.pdf', parser=parser(), limits=DocumentLimits())
    assert manifest.schema_version == 2
    payload = json.loads(manifest.model_dump_json())
    payload['schema_version'] = 1
    with pytest.raises(ValueError, match='regions'):
        DocumentManifest.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize('damage', ['utf8', 'overlap', 'coverage', 'box'])
async def test_invalid_extension_region_rejected_before_source_retention(memory, damage):
    parsed = await parser().parse(b'%PDF-observation', 'source.pdf')
    segment = parsed.segments[1]
    assert hasattr(segment, 'regions'), 'generic file parsing discarded OCR geometry'
    regions = segment.regions
    changes = {'utf8': {'start': 1}, 'overlap': {'start': 0}, 'box': {'box': (0., 0., 0., 1.)}}
    if damage == 'coverage':
        regions = regions[:1]
    else:
        index = 1 if damage == 'overlap' else 0
        regions = tuple(r.model_copy(update=changes[damage]) if i == index else r for i, r in enumerate(regions))
    broken = parsed.model_copy(update={'segments': (parsed.segments[0], segment.model_copy(update={'regions': regions}))})
    class Extension:
        async def parse(self, data, filename, limits):
            return broken
    with pytest.raises(InvalidInput, match='region'):
        await ingest_document(memory, 'alpha', b'invalid-observation', filename='source.pdf', parser=Extension())
    assert (await memory.documents.counts('alpha')).episodes == 0
    import hashlib
    from scone_memory.core.errors import NotFound
    with pytest.raises(NotFound):
        await memory.attachment('alpha', hashlib.sha256(b'invalid-observation').hexdigest())


async def test_geometry_survives_extraction_checkpoint_and_storage_restart(tmp_path):
    held = HeldEmbedder()
    memory = await open_memory(tmp_path, held)
    original = await memory.attach('alpha', b'%PDF-observation', 'application/pdf', filename='source.pdf')
    job = workflow(memory, tmp_path, parser())
    task = asyncio.create_task(job.run('ocr', space='alpha', attachment_id=original.attachment_id))
    try:
        await asyncio.wait_for(held.entered.wait(), 5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        job.close()
        await memory.close()
    class NoParse:
        async def parse(self, *args):
            pytest.fail('completed extraction must be reused')
    memory = await open_memory(tmp_path)
    resumed = workflow(memory, tmp_path, NoParse())
    try:
        result = await resumed.run('ocr', space='alpha', attachment_id=original.attachment_id)
        assert result.reused_steps == ('extract',)
        evidence = await document_provenance(memory, 'alpha', result.results['index']['episode_id'])
        assert [r.text for r in evidence.segments[1].regions] == ['Café', 'Polaris']
    finally:
        resumed.close()
        await memory.close()


async def test_rendered_pdf_flows_through_generic_ingestion(memory):
    from scone_memory.ingestion.pdf_ocr import OcrPdfParser
    from .test_pdf_ocr import ObservedOcr, scanned_pdf
    engine = ObservedOcr()
    result = await ingest_document(memory, 'alpha', scanned_pdf(), filename='scan.pdf',
        parser=BuiltinDocumentParser(pdf_parser=OcrPdfParser(engine)))
    evidence = await document_provenance(memory, 'alpha', result.added.episode_id)
    region, = evidence.segments[0].regions
    assert engine.calls == 1 and region.text == 'Café uses Polaris'
    assert region.start == 0 and region.end == len(region.text.encode())
    assert region.box == (.1, .1, .9, .2)


async def test_image_reader_exposes_same_typed_geometry(memory):
    from scone_memory.ingestion.formats.media import ImageDocumentParser
    from .test_media_formats import StructuralOcr, image_bytes
    result = await ingest_document(memory, 'alpha', image_bytes(), filename='scan.png',
        parser=BuiltinDocumentParser(parsers={'.png': ImageDocumentParser(StructuralOcr())}))
    evidence = await document_provenance(memory, 'alpha', result.added.episode_id)
    region, = evidence.segments[0].regions
    assert region.coordinate_space == 'normalized_displayed_frame_top_left'
    assert region.box == (.1, .2, .8, .9) and region.score == .7
    assert region.start == 0 and region.end == len(region.text.encode())
