"""Extraction checkpoints survive cancellation and refuse changed/deleted sources."""
import asyncio

import pytest

from scone_memory import HashEmbedder, MemoryEngine
from scone_memory.agents import WorkflowError
from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.backends.blobs import FileBlobStore
from scone_memory.ingestion.pdf import PypdfParser
from .test_pdf_ingestion import pdf_bytes

KEY = b'w' * 32


class CountingParser(PypdfParser):
    def __init__(self):
        self.calls = 0

    async def parse(self, data, filename, limits):
        from scone_memory.ingestion.formats.registry import BuiltinDocumentParser
        self.calls += 1
        return await BuiltinDocumentParser().parse(data, filename, limits)


class HeldEmbedder(HashEmbedder):
    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()

    async def embed(self, texts):
        self.entered.set()
        await asyncio.Event().wait()


async def open_memory(path, embedder=None):
    return await MemoryEngine(SqliteDocumentStore(path / 'memory.db'), SqliteVectorIndex(path / 'memory.db'),
                              embedder or HashEmbedder(), blobs=FileBlobStore(path / 'blobs')).open()


def workflow(memory, path, parser, **kwargs):
    import scone_memory.ingestion as ingestion
    assert hasattr(ingestion, 'DocumentIngestionWorkflow'), 'document extraction has no durable workflow'
    return ingestion.DocumentIngestionWorkflow(memory, path / 'jobs.db', key=KEY, parser=parser,
                                         parser_revision='pypdf-test-v1', **kwargs)


async def test_cancelled_indexing_resumes_after_storage_and_journal_restart(tmp_path):
    blocked = HeldEmbedder()
    memory = await open_memory(tmp_path, blocked)
    parser = CountingParser()
    raw = pdf_bytes()
    original = await memory.attach('alpha', raw, 'application/pdf', filename='fixture.pdf')
    job = workflow(memory, tmp_path, parser)
    task = asyncio.create_task(job.run('document-1', space='alpha', attachment_id=original.attachment_id))
    try:
        await asyncio.wait_for(blocked.entered.wait(), 5)
        progress = job.status('document-1', space='alpha', attachment_id=original.attachment_id)
        assert progress.completed_steps == ('extract',)
        assert progress.inflight == 'index'
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        job.close()
        await memory.close()

    memory = await open_memory(tmp_path)
    resumed = workflow(memory, tmp_path, parser)
    try:
        progress = resumed.status('document-1', space='alpha', attachment_id=original.attachment_id)
        assert progress.status == 'cancelled' and progress.error_class == 'CancelledError'
        result = await resumed.run('document-1', space='alpha', attachment_id=original.attachment_id)
        assert result.reused_steps == ('extract',)
        assert parser.calls == 1
        assert (await memory.documents.counts('alpha')).episodes == 1
        episode_id = result.results['index']['episode_id']
        from scone_memory.ingestion import document_provenance
        evidence = await document_provenance(memory, 'alpha', episode_id)
        assert evidence.original.attachment_id == original.attachment_id
        assert evidence.segments[0].locator == 'page:1'
        assert any(item.episode_id == episode_id for item in (await memory.recall('alpha', 'Polaris')).items)
        replay = await resumed.run('document-1', space='alpha', attachment_id=original.attachment_id)
        assert replay.reused_steps == ('extract', 'index') and parser.calls == 1
        await memory.forget('alpha', episode_id)
        with pytest.raises(WorkflowError, match='sources_invalid'):
            await resumed.run('document-1', space='alpha', attachment_id=original.attachment_id)
        assert (await memory.documents.counts('alpha')).episodes == 0
    finally:
        resumed.close()
        await memory.close()
    assert b'Polaris' not in (tmp_path / 'jobs.db').read_bytes()


@pytest.mark.parametrize('change', ['space', 'source', 'revision', 'limits'])
async def test_pdf_checkpoint_binding_cannot_be_repurposed(tmp_path, change):
    from scone_memory.ingestion.formats.types import DocumentLimits
    memory = await open_memory(tmp_path)
    parser = CountingParser()
    original = await memory.attach('alpha', pdf_bytes(), 'application/pdf', filename='fixture.pdf')
    other = await memory.attach('alpha', pdf_bytes(pages=('Changed source.',)), 'application/pdf', filename='other.pdf')
    first = workflow(memory, tmp_path, parser)
    await first.run('document-1', space='alpha', attachment_id=original.attachment_id)
    first.close()
    import scone_memory.ingestion as ingestion
    second = ingestion.DocumentIngestionWorkflow(memory, tmp_path / 'jobs.db', key=KEY, parser=parser,
        parser_revision='v2' if change == 'revision' else 'pypdf-test-v1',
        limits=DocumentLimits(max_segments=2) if change == 'limits' else DocumentLimits())
    args = {'space': 'beta' if change == 'space' else 'alpha',
            'attachment_id': other.attachment_id if change == 'source' else original.attachment_id}
    try:
        with pytest.raises(WorkflowError, match='binding_mismatch'):
            await second.run('document-1', **args)
        with pytest.raises(WorkflowError, match='binding_mismatch'):
            second.status('document-1', **args)
        assert parser.calls == 1
    finally:
        second.close()
        await memory.close()


async def test_invalid_pdf_has_durable_stage_failure_and_no_searchable_record(tmp_path):
    memory = await open_memory(tmp_path)
    original = await memory.attach('alpha', b'%PDF-1.7 invalid', 'application/pdf', filename='bad.pdf')
    job = workflow(memory, tmp_path, CountingParser(), max_retries=0)
    try:
        assert job.status('new', space='alpha', attachment_id=original.attachment_id) is None
        with pytest.raises(WorkflowError, match='step_failed'):
            await job.run('bad', space='alpha', attachment_id=original.attachment_id)
        state = job.status('bad', space='alpha', attachment_id=original.attachment_id)
        assert state.status == 'failed' and state.inflight == 'extract'
        assert state.attempts == {'extract': 1} and state.error_class == 'InvalidInput'
        assert state.completed_steps == ()
        assert (await memory.documents.counts('alpha')).episodes == 0
    finally:
        job.close()
        await memory.close()
