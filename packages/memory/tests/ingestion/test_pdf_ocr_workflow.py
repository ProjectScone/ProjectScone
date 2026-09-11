"""Page receipts survive interruption and remain bound to retained evidence."""
import asyncio
import os
from pathlib import Path
import sys
import time

import pytest

pytest.importorskip('pypdfium2')

from scone_memory.agents import WorkflowError
from scone_memory.ingestion import pdf_provenance
from scone_memory.ingestion.pdf import PdfLimits
from scone_memory.ingestion.pdf_ocr import OcrPdfOptions
from .test_document_workflow import KEY, HeldEmbedder, open_memory
from .test_pdf_ingestion import pdf_bytes
from .test_pdf_ocr import ObservedOcr


class PauseSecondPage(ObservedOcr):
    def __init__(self):
        super().__init__('Checkpoint Polaris')
        self.entered = asyncio.Event()

    async def recognize(self, image, **kwargs):
        if self.calls == 1:
            self.entered.set()
            await asyncio.Event().wait()
        return await super().recognize(image, **kwargs)


def workflow(memory, path, engine, **kwargs):
    import scone_memory.ingestion as ingestion
    assert hasattr(ingestion, 'PdfOcrWorkflow'), 'per-page OCR recovery is missing'
    return ingestion.PdfOcrWorkflow(memory, path / 'ocr.db', key=KEY, engine=engine,
        recognizer_revision=kwargs.pop('recognizer_revision', 'fixture-v1'),
        options=kwargs.pop('options', OcrPdfOptions(mode='all_pages')), **kwargs)


async def retain(memory, raw=None, space='alpha'):
    return await memory.attach(space, raw or pdf_bytes(), 'application/pdf', filename='source.pdf')


@pytest.mark.parametrize('fail_read', ['first', 'after_binding', 'last'])
async def test_completed_pdf_survives_temporary_original_storage_outage(tmp_path, monkeypatch, fail_read):
    memory = await open_memory(tmp_path)
    engine = ObservedOcr()
    original = await retain(memory)
    job = workflow(memory, tmp_path, engine)
    args = dict(space='alpha', attachment_id=original.attachment_id)
    try:
        first = await job.run('scan', **args)
        calls = engine.calls
        attachment = memory.attachment
        reads = 0

        async def observed(*args, **kwargs):
            nonlocal reads
            reads += 1
            return await attachment(*args, **kwargs)

        monkeypatch.setattr(memory, 'attachment', observed)
        await job.run('scan', **args)
        fail_at = {'first': 1, 'after_binding': 3, 'last': reads}[fail_read]
        reads = 0

        async def unavailable(*args, **kwargs):
            nonlocal reads
            reads += 1
            if reads == fail_at:
                raise ConnectionError('private backend details')
            return await attachment(*args, **kwargs)

        monkeypatch.setattr(memory, 'attachment', unavailable)
        with pytest.raises(WorkflowError, match='^verification_unavailable$'):
            await job.run('scan', **args)
        job.close()
        monkeypatch.setattr(memory, 'attachment', attachment)
        job = workflow(memory, tmp_path, engine)
        result = await job.run('scan', **args)
        assert result.reused_pages == (1, 2) and result.reused_index
        assert result.added.episode_id == first.added.episode_id
        assert engine.calls == calls
    finally:
        job.close()
        await memory.close()


async def test_cancelled_second_page_resumes_without_rendering_or_recognizing_first(tmp_path, monkeypatch):
    memory = await open_memory(tmp_path)
    original = await retain(memory)
    paused = PauseSecondPage()
    job = workflow(memory, tmp_path, paused)
    task = asyncio.create_task(job.run('scan', space='alpha', attachment_id=original.attachment_id))
    try:
        await asyncio.wait_for(paused.entered.wait(), 10)
        assert job.page_status('scan', space='alpha', attachment_id=original.attachment_id, page=1).status == 'completed'
        assert job.page_status('scan', space='alpha', attachment_id=original.attachment_id, page=2).status == 'running'
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        job.close()
        await memory.close()

    memory = await open_memory(tmp_path)
    engine = ObservedOcr('Resumed Juniper')
    resumed = workflow(memory, tmp_path, engine)
    from scone_memory.ingestion.pdf_ocr import OcrPdfParser
    recognize = OcrPdfParser.recognize_page
    rendered = []

    async def record(self, data, page, **kwargs):
        rendered.append(page)
        return await recognize(self, data, page, **kwargs)

    monkeypatch.setattr(OcrPdfParser, 'recognize_page', record)
    try:
        result = await resumed.run('scan', space='alpha', attachment_id=original.attachment_id)
        assert result.reused_pages == (1,) and not result.reused_index
        assert rendered == [2] and engine.calls == 1
        evidence = await pdf_provenance(memory, 'alpha', result.added.episode_id)
        assert [p.regions[0].text for p in evidence.pages] == ['Checkpoint Polaris', 'Resumed Juniper']
        assert any(i.episode_id == result.added.episode_id for i in (await memory.recall('alpha', 'Polaris')).items)
        replay = await resumed.run('scan', space='alpha', attachment_id=original.attachment_id)
        assert replay.reused_pages == (1, 2) and replay.reused_index
        assert replay.added.episode_id == result.added.episode_id and engine.calls == 1
        await memory.forget('alpha', result.added.episode_id)
        with pytest.raises(WorkflowError, match='sources_invalid'):
            await resumed.run('scan', space='alpha', attachment_id=original.attachment_id)
        assert (await memory.documents.counts('alpha')).episodes == 0
    finally:
        resumed.close()
        await memory.close()
    assert b'Checkpoint Polaris' not in (tmp_path / 'ocr.db').read_bytes()
    assert b'Resumed Juniper' not in (tmp_path / 'ocr.db').read_bytes()


async def test_index_cancellation_reuses_all_recognition_after_restart(tmp_path):
    memory = await open_memory(tmp_path, HeldEmbedder())
    original = await retain(memory)
    engine = ObservedOcr()
    job = workflow(memory, tmp_path, engine)
    task = asyncio.create_task(job.run('scan', space='alpha', attachment_id=original.attachment_id))
    try:
        await asyncio.wait_for(memory.embedder.entered.wait(), 10)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        job.close()
        await memory.close()
    memory = await open_memory(tmp_path)
    resumed = workflow(memory, tmp_path, engine)
    try:
        result = await resumed.run('scan', space='alpha', attachment_id=original.attachment_id)
        assert result.reused_pages == (1, 2) and engine.calls == 2
        assert (await memory.documents.counts('alpha')).episodes == 1
    finally:
        resumed.close()
        await memory.close()


@pytest.mark.parametrize('change', ['source', 'space', 'revision', 'options', 'limits'])
async def test_checkpoint_refuses_changed_binding(tmp_path, change):
    memory = await open_memory(tmp_path)
    original = await retain(memory)
    engine = ObservedOcr()
    job = workflow(memory, tmp_path, engine)
    try:
        await job.run('scan', space='alpha', attachment_id=original.attachment_id)
    finally:
        job.close()
    kwargs = {}
    space = 'other' if change == 'space' else 'alpha'
    if change in {'space', 'source'}:
        original = await retain(memory, pdf_bytes(title='changed') if change == 'source' else None, space)
    if change == 'revision': kwargs['recognizer_revision'] = 'fixture-v2'
    if change == 'options': kwargs['options'] = OcrPdfOptions(mode='all_pages', dpi=200)
    if change == 'limits': kwargs['limits'] = PdfLimits(max_pages=3)
    resumed = workflow(memory, tmp_path, engine, **kwargs)
    try:
        with pytest.raises(WorkflowError, match='binding_mismatch'):
            await resumed.run('scan', space=space, attachment_id=original.attachment_id)
        assert engine.calls == 2
    finally:
        resumed.close()
        await memory.close()


async def test_native_pages_stay_native_and_have_no_ocr_checkpoint(tmp_path):
    memory = await open_memory(tmp_path)
    original = await retain(memory, pdf_bytes(pages=('Native Polaris', '')))
    engine = ObservedOcr()
    job = workflow(memory, tmp_path, engine, options=OcrPdfOptions())
    try:
        result = await job.run('mixed', space='alpha', attachment_id=original.attachment_id)
        assert engine.calls == 1
        assert job.page_status('mixed', space='alpha', attachment_id=original.attachment_id, page=1) is None
        evidence = await pdf_provenance(memory, 'alpha', result.added.episode_id)
        assert [p.extraction for p in evidence.pages] == ['text_layer', 'ocr']
    finally:
        job.close()
        await memory.close()


async def test_total_document_time_may_exceed_the_individual_page_deadline(tmp_path):
    class SlowPages(ObservedOcr):
        async def recognize(self, image, **kwargs):
            await asyncio.sleep(.4)
            return await super().recognize(image, **kwargs)

    memory = await open_memory(tmp_path)
    original = await retain(memory, pdf_bytes(pages=('a', 'b', 'c', 'd')))
    engine = SlowPages()
    job = workflow(memory, tmp_path, engine, limits=PdfLimits(timeout_seconds=1.5))
    try:
        start = time.monotonic()
        result = await job.run('long', space='alpha', attachment_id=original.attachment_id)
        assert time.monotonic() - start > 1.5
        assert engine.calls == 4 and result.added.episode_id > 0
    finally:
        job.close()
        await memory.close()


async def test_page_deadline_retains_prior_pages_and_a_retry_can_finish(tmp_path):
    memory = await open_memory(tmp_path)
    original = await retain(memory)
    paused = PauseSecondPage()
    job = workflow(memory, tmp_path, paused, limits=PdfLimits(timeout_seconds=1.5))
    try:
        with pytest.raises(WorkflowError, match='deadline'):
            await job.run('timeout', space='alpha', attachment_id=original.attachment_id)
        assert job.page_status('timeout', space='alpha', attachment_id=original.attachment_id, page=1).status == 'completed'
        assert job.page_status('timeout', space='alpha', attachment_id=original.attachment_id, page=2).status == 'deadline'
    finally:
        job.close()
    engine = ObservedOcr()
    resumed = workflow(memory, tmp_path, engine, limits=PdfLimits(timeout_seconds=1.5))
    try:
        result = await resumed.run('timeout', space='alpha', attachment_id=original.attachment_id)
        assert result.reused_pages == (1,) and engine.calls == 1
    finally:
        resumed.close()
        await memory.close()


async def test_checkpoint_payload_limit_does_not_publish_partial_document(tmp_path):
    memory = await open_memory(tmp_path)
    original = await retain(memory)
    job = workflow(memory, tmp_path, ObservedOcr('x' * 10000), max_checkpoint_bytes=2000)
    try:
        with pytest.raises(WorkflowError, match='payload_limit'):
            await job.run('oversized', space='alpha', attachment_id=original.attachment_id)
        assert (await memory.documents.counts('alpha')).episodes == 0
        assert job.page_status('oversized', space='alpha', attachment_id=original.attachment_id, page=1).status == 'failed'
    finally:
        job.close()
        await memory.close()


async def test_process_kill_reuses_the_last_committed_page(tmp_path):
    memory = await open_memory(tmp_path)
    original = await retain(memory)
    await memory.close()
    worker = tmp_path / 'ocr_child.py'
    worker.write_text('''
import asyncio
from pathlib import Path
import sys
from scone_memory import HashEmbedder, MemoryEngine
from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.backends.blobs import FileBlobStore
from scone_memory.ingestion import PdfOcrWorkflow, OcrPdfOptions
from scone_memory.ocr import OcrResult, OcrRegion
from scone_memory.ocr.tesseract import png_dimensions

class Engine:
    calls = 0
    async def recognize(self, image, *, max_pixels, **kwargs):
        self.calls += 1
        if self.calls == 2:
            Path(sys.argv[1], 'page-two-entered').write_text('ready')
            await asyncio.Event().wait()
        width, height = png_dimensions(image, max_pixels)
        return OcrResult(engine='fixture', width=width, height=height,
            regions=(OcrRegion(text='Child Polaris', box=(.1,.1,.9,.2)),))

async def main():
    root = Path(sys.argv[1])
    memory = await MemoryEngine(SqliteDocumentStore(root/'memory.db'),
        SqliteVectorIndex(root/'memory.db'), HashEmbedder(), blobs=FileBlobStore(root/'blobs')).open()
    job = PdfOcrWorkflow(memory, root/'ocr.db', key=b'w'*32, engine=Engine(),
        recognizer_revision='fixture-v1', options=OcrPdfOptions(mode='all_pages'))
    await job.run('crash', space='alpha', attachment_id=sys.argv[2])

asyncio.run(main())
''')
    import scone_memory
    package_root = str(Path(scone_memory.__file__).resolve().parent.parent)
    child = await asyncio.create_subprocess_exec(sys.executable, str(worker), str(tmp_path), original.attachment_id,
        env={**{k: v for k, v in os.environ.items() if k in {'PATH', 'TMPDIR', 'LANG'}}, 'PYTHONPATH': package_root},
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    try:
        async def entered():
            while not (tmp_path / 'page-two-entered').exists():
                if child.returncode is not None:
                    raise AssertionError((await child.stderr.read()).decode())
                await asyncio.sleep(.01)
        await asyncio.wait_for(entered(), 15)
    finally:
        if child.returncode is None:
            child.kill()
        await child.wait()
    memory = await open_memory(tmp_path)
    engine = ObservedOcr('Parent Juniper')
    job = workflow(memory, tmp_path, engine)
    try:
        result = await job.run('crash', space='alpha', attachment_id=original.attachment_id)
        assert result.reused_pages == (1,) and engine.calls == 1
        evidence = await pdf_provenance(memory, 'alpha', result.added.episode_id)
        assert [p.regions[0].text for p in evidence.pages] == ['Child Polaris', 'Parent Juniper']
    finally:
        job.close()
        await memory.close()


@pytest.mark.parametrize('replacement', ['native_source', 'deleted_then_reattached'])
async def test_document_binding_survives_changing_which_pages_need_ocr(tmp_path, replacement):
    memory = await open_memory(tmp_path)
    raw = pdf_bytes(pages=('', ''))
    original = await retain(memory, raw)
    holder = await memory.remember('alpha', 'Original holder', attachment_ids=(original.attachment_id,))
    paused = PauseSecondPage()
    job = workflow(memory, tmp_path, paused, options=OcrPdfOptions())
    task = asyncio.create_task(job.run('bound', space='alpha', attachment_id=original.attachment_id))
    try:
        await asyncio.wait_for(paused.entered.wait(), 10)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        job.close()
    resumed = workflow(memory, tmp_path, ObservedOcr(), options=OcrPdfOptions())
    try:
        if replacement == 'deleted_then_reattached':
            await memory.forget('alpha', holder.episode_id)
            with pytest.raises(WorkflowError, match='sources_invalid'):
                await resumed.run('bound', space='alpha', attachment_id=original.attachment_id)
            new_original = await retain(memory, raw)
            reason = 'sources_invalid'
        else:
            new_original = await retain(memory, pdf_bytes(pages=('Entirely native replacement',)))
            reason = 'binding_mismatch'
        with pytest.raises(WorkflowError, match=reason):
            await resumed.run('bound', space='alpha', attachment_id=new_original.attachment_id)
    finally:
        resumed.close()
        await memory.close()
