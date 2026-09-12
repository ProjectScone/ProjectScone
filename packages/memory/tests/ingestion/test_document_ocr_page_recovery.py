"""Generic document jobs retain completed OCR pages across extraction retries."""
import asyncio

import pytest

pytest.importorskip('pypdfium2')

from scone_memory.ingestion.document_ocr import DocumentOcr, PdfOcrSelection
from scone_memory.ingestion.file_workflow import DocumentIngestionWorkflow
from scone_memory.ingestion.files import document_provenance
from .test_document_workflow import KEY, open_memory
from .test_pdf_ingestion import pdf_bytes
from .test_pdf_ocr import ObservedOcr
from .test_pdf_ocr_workflow import PauseSecondPage


def workflow(memory, path, engine):
    parser = DocumentOcr(engine).parser(PdfOcrSelection(mode='all_pages', reading_order='columns_ltr'))
    return DocumentIngestionWorkflow(memory, path / 'document.db', key=KEY,
        parser_revision='recognizer-v1', parser=parser, automatic_retries=False)


async def test_generic_extraction_resumes_completed_pages_after_restart(tmp_path, monkeypatch):
    memory = await open_memory(tmp_path)
    original = await memory.attach('alpha', pdf_bytes(), 'application/pdf', filename='source.pdf')
    paused = PauseSecondPage()
    job = workflow(memory, tmp_path, paused)
    args = dict(space='alpha', attachment_id=original.attachment_id)
    task = asyncio.create_task(job.run('scan', **args))
    try:
        await asyncio.wait_for(paused.entered.wait(), 10)
        progress = job.status('scan', **args)
        assert progress.completed_steps == () and progress.inflight == 'extract'
        assert progress.checkpoint_count == 2
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        job.close()
        await memory.close()

    memory = await open_memory(tmp_path)
    engine = ObservedOcr('Resumed Juniper')
    job = workflow(memory, tmp_path, engine)
    from scone_memory.ingestion.pdf_ocr import OcrPdfParser
    recognize = OcrPdfParser._recognize
    rendered = []
    async def observe(self, data, page, deadline):
        rendered.append(page)
        return await recognize(self, data, page, deadline)
    monkeypatch.setattr(OcrPdfParser, '_recognize', observe)
    try:
        result = await job.run('scan', **args)
        assert rendered == [2] and engine.calls == 1
        evidence = await document_provenance(memory, 'alpha', result.results['index']['episode_id'])
        assert [segment.text for segment in evidence.segments] == ['Checkpoint Polaris', 'Resumed Juniper']
        assert all(segment.regions for segment in evidence.segments)
        assert evidence.metadata['pdf_ocr']
        assert (await memory.documents.counts('alpha')).episodes == 1
        assert await job.read_result('scan', **args) is not None
        assert rendered == [2]
    finally:
        job.close()
        await memory.close()
    assert b'Checkpoint Polaris' not in (tmp_path / 'document.db').read_bytes()


class Receipts:
    def __init__(self):
        self.values = {}
    def get(self, key):
        return self.values.get(key)
    def put(self, key, value):
        self.values[key] = value


async def test_checkpointed_and_direct_extraction_have_identical_manifests():
    from scone_memory.ingestion.files import prepare_document, encode_manifest
    from scone_memory.ingestion.formats.types import DocumentLimits
    raw = pdf_bytes()
    engine = ObservedOcr('Café Polaris')
    parser = DocumentOcr(engine).parser(PdfOcrSelection(mode='all_pages', reading_order='columns_ltr'))
    direct = await prepare_document(raw, 'scan.pdf', parser=parser, limits=DocumentLimits())
    checkpoints = Receipts()
    recovered = await prepare_document(raw, 'scan.pdf', parser=parser, limits=DocumentLimits(),
                                       extraction_checkpoint=checkpoints)
    replay = await prepare_document(raw, 'scan.pdf', parser=parser, limits=DocumentLimits(),
                                    extraction_checkpoint=checkpoints)
    assert encode_manifest(direct) == encode_manifest(recovered) == encode_manifest(replay)
    assert engine.calls == 4  # two direct, two checkpointed, no replay recognition


@pytest.mark.parametrize('change', ['source', 'mode', 'order', 'dpi', 'limits', 'dependency'])
async def test_page_receipts_refuse_changed_binding_even_when_no_page_needs_ocr(change, monkeypatch):
    from scone_memory.core.errors import InvalidInput
    from scone_memory.ingestion.pdf import PdfLimits
    from scone_memory.ingestion.pdf_ocr import OcrPdfOptions, OcrPdfParser
    raw = pdf_bytes()
    engine = ObservedOcr()
    options = OcrPdfOptions(mode='all_pages')
    receipts = Receipts()
    await OcrPdfParser(engine, options=options).parse_checkpointed(raw, PdfLimits(), receipts)
    if change == 'source':
        raw = pdf_bytes(pages=('Different.',))
    if change in ('mode', 'order', 'dpi'):
        options = options.model_copy(update={'mode':'missing_text'} if change == 'mode'
            else {'reading_order':'columns_ltr'} if change == 'order' else {'dpi':200})
    if change == 'dependency':
        monkeypatch.setattr('scone_memory.ingestion.pdf_ocr.version', lambda name: 'different')
    limits = PdfLimits(max_pages=10) if change == 'limits' else PdfLimits()
    with pytest.raises(InvalidInput, match='checkpoints do not match'):
        await OcrPdfParser(engine, options=options).parse_checkpointed(raw, limits, receipts)
    assert engine.calls == 2


@pytest.mark.parametrize('damage', ['json', 'page', 'binding', 'geometry', 'dimensions', 'count', 'bytes'])
async def test_malformed_page_receipts_fail_without_recognition(damage):
    import json
    from scone_memory.core.errors import InvalidInput
    from scone_memory.ingestion.pdf import PdfLimits
    from scone_memory.ingestion.pdf_ocr import OcrPdfOptions, OcrPdfParser
    parser = OcrPdfParser(ObservedOcr(), options=OcrPdfOptions(mode='all_pages', max_regions=1))
    raw, receipts = pdf_bytes(), Receipts()
    await parser.parse_checkpointed(raw, PdfLimits(), receipts)
    payload = json.loads(receipts.values['ocr-page:1'])
    if damage == 'page':
        payload['page'] = 2
    if damage == 'binding':
        payload['binding'] = '0' * 64
    if damage == 'geometry':
        payload['result']['regions'][0]['box'] = [0, 0, 2, 2]
    if damage == 'dimensions':
        payload['result']['width'] = payload['result']['height'] = 100000
    if damage == 'count':
        payload['result']['regions'] *= 2
    receipts.values['ocr-page:1'] = (b'{' if damage == 'json' else b' ' * (16*1024*1024+1)
                                    if damage == 'bytes' else json.dumps(payload).encode())
    with pytest.raises(InvalidInput, match='page checkpoint is invalid'):
        await parser.parse_checkpointed(raw, PdfLimits(), receipts)
    assert parser.engine.calls == 2


async def test_native_text_is_preserved_and_not_given_ocr_receipts():
    from scone_memory.ingestion.pdf import PdfLimits
    from scone_memory.ingestion.pdf_ocr import OcrPdfParser
    receipts, engine = Receipts(), ObservedOcr()
    output = await OcrPdfParser(engine).parse_checkpointed(pdf_bytes(), PdfLimits(), receipts)
    assert 'Polaris' in output.text and engine.calls == 0
    assert set(receipts.values) == {'ocr-binding'}


@pytest.mark.parametrize('kind', ['document', 'pdf', 'selected', 'wrapped', 'instance'])
async def test_inherited_checkpoint_dispatch_preserves_custom_parse_overrides(kind):
    from scone_memory.ingestion.document_ocr import SelectedDocumentOcr
    from scone_memory.ingestion.files import prepare_document
    from scone_memory.ingestion.formats.registry import BuiltinDocumentParser
    from scone_memory.ingestion.formats.types import DocumentLimits, DocumentSegment, ParsedDocument
    from scone_memory.ingestion.pdf import ParsedPdf, PdfPage
    from scone_memory.ingestion.pdf_ocr import OcrPdfParser
    safe = ParsedDocument(format='pdf', parser='redactor',
        segments=(DocumentSegment(text='Redacted', locator='page:1'),))
    calls = []
    async def redact(data, filename, limits):
        calls.append('override')
        return safe
    class CustomDocument(BuiltinDocumentParser):
        async def parse(self, data, filename, limits):
            return await redact(data, filename, limits)
    class CustomPdf(OcrPdfParser):
        async def parse(self, data, limits):
            calls.append('override')
            return ParsedPdf(text='Redacted', parser='redactor', pages=(PdfPage(number=1,
                start=0, end=8, width_points=600., height_points=800., rotation=0, empty=False),))
    class CustomSelected(SelectedDocumentOcr):
        async def parse(self, data, filename, limits):
            return await redact(data, filename, limits)
    engine = ObservedOcr('ORIGINAL PRIVATE TEXT')
    if kind == 'document':
        parser = CustomDocument()
    elif kind == 'pdf':
        parser = BuiltinDocumentParser(pdf_parser=CustomPdf(engine))
    elif kind == 'wrapped':
        parser = SelectedDocumentOcr(CustomDocument(), PdfOcrSelection(mode='all_pages', reading_order='provider'), 150)
    elif kind == 'selected':
        parser = CustomSelected(BuiltinDocumentParser(), PdfOcrSelection(mode='all_pages', reading_order='provider'), 150)
    else:
        parser = BuiltinDocumentParser()
        parser.parse = redact
    result = await prepare_document(pdf_bytes(), 'scan.pdf', parser=parser,
        limits=DocumentLimits(), extraction_checkpoint=Receipts())
    assert result.parsed.segments[0].text == 'Redacted'
    assert calls == ['override'] and engine.calls == 0


async def test_http_job_process_kill_reuses_completed_ocr_pages(tmp_path):
    import os
    from pathlib import Path
    import sys
    import httpx
    import scone_memory
    memory = await open_memory(tmp_path)
    original = await memory.attach('alpha', pdf_bytes(), 'application/pdf', filename='scan.pdf')
    await memory.close()
    worker = Path(__file__).parent / 'fixtures/document_ocr_http_worker.py'
    child = None
    async def start(mode):
        nonlocal child
        (tmp_path/'port').unlink(missing_ok=True)
        child = await asyncio.create_subprocess_exec(sys.executable, str(worker), str(tmp_path), mode,
            env={**{key:value for key,value in os.environ.items() if key in {'PATH','TMPDIR','LANG'}},
                 'PYTHONPATH':str(Path(scone_memory.__file__).resolve().parent.parent)},
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        async def ready():
            while not (tmp_path/'port').exists():
                if child.returncode is not None:
                    raise AssertionError((await child.stderr.read()).decode())
                await asyncio.sleep(.02)
        await asyncio.wait_for(ready(), 15)
        return httpx.AsyncClient(base_url='http://127.0.0.1:'+ (tmp_path/'port').read_text(),
                                 headers={'authorization':'Bearer writer'}, trust_env=False)
    async def stop(kill=False):
        if child is not None and child.returncode is None:
            if kill:
                child.kill()
            else:
                child.stdin.write(b'stop\n')
                await child.stdin.drain()
            await asyncio.wait_for(child.wait(), 10)
        if not kill:
            assert child.returncode == 0, (await child.stderr.read()).decode()
    try:
        async with await start('hold') as client:
            started = await client.post('/v1/document-jobs', json={'import_id':'scan',
                'attachment_id':original.attachment_id, 'filename':'scan.pdf',
                'pdf_ocr':{'mode':'all_pages','reading_order':'provider'}})
            assert started.status_code == 202, started.text
            async def entered():
                while not (tmp_path/'page-two-entered').exists():
                    await asyncio.sleep(.02)
            await asyncio.wait_for(entered(), 15)
            assert (tmp_path/'recognized').read_text() == 'hold:1\n'
            await stop(kill=True)
        async with await start('resume') as client:
            status = (await client.get('/v1/document-jobs/scan')).json()
            assert status['status'] == 'interrupted' and status['completed_steps'] == []
            assert (tmp_path/'recognized').read_text() == 'hold:1\n'
            control = {'expected_revision':status['revision']}
            denied = await client.post('/v1/document-jobs/scan/resume', json=control,
                                       headers={'authorization':'Bearer reader'})
            assert denied.status_code == 403
            assert (await client.get('/v1/document-jobs/scan', headers={'authorization':'Bearer other'})).status_code == 404
            resumed = await client.post('/v1/document-jobs/scan/resume', json=control)
            assert resumed.status_code == 202, resumed.text
            async def completed():
                while True:
                    response = await client.get('/v1/document-jobs/scan/result')
                    if response.status_code == 200:
                        return response.json()
                    assert response.status_code == 409, response.text
                    await asyncio.sleep(.03)
            result = await asyncio.wait_for(completed(), 15)
            episode_id = result['added']['episode_id']
            evidence = (await client.get(f'/v1/episodes/{episode_id}/document')).json()
            assert [s['text'] for s in evidence['segments']] == ['Saved Polaris','Resumed Juniper']
            assert (tmp_path/'recognized').read_text() == 'hold:1\nresume:1\n'
        await stop()
        async with await start('readonly') as client:
            assert (await client.get('/v1/document-jobs/scan/result')).json()['added']['episode_id'] == episode_id
            assert (tmp_path/'recognized').read_text() == 'hold:1\nresume:1\n'
            assert (await client.delete(f'/v1/episodes/{episode_id}')).is_success
            assert (await client.get('/v1/document-jobs/scan/result')).status_code == 409
        await stop()
    finally:
        if child is not None and child.returncode is None:
            child.kill()
            await child.wait()


async def test_oversized_page_text_is_not_committed_as_reusable_work():
    from scone_memory.core.errors import InvalidInput
    from scone_memory.ingestion.pdf import PdfLimits
    from scone_memory.ingestion.pdf_ocr import OcrPdfOptions, OcrPdfParser
    receipts = Receipts()
    parser = OcrPdfParser(ObservedOcr('Long text beyond the chosen budget'), options=OcrPdfOptions(mode='all_pages'))
    with pytest.raises(InvalidInput, match='text exceeds'):
        await parser.parse_checkpointed(pdf_bytes(), PdfLimits(max_text_bytes=4), receipts)
    assert set(receipts.values) == {'ocr-binding'}
