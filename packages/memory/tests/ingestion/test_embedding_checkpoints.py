"""Durable indexing reuses only validated batches for exactly matching inputs."""
import asyncio
import os
import subprocess
import sys

import pytest

from scone_memory import HashEmbedder
from scone_memory.agents import WorkflowError
from scone_memory.ingestion.batch import EMBED_BATCH, _embed_chunks
from .test_document_workflow import CountingParser, open_memory, workflow


class CountingEmbedder(HashEmbedder):
    def __init__(self, *, interrupt=False):
        super().__init__()
        self.calls = []
        self.interrupt = interrupt

    async def embed(self, texts):
        self.calls.append(list(texts))
        if self.interrupt and len(self.calls) == 2:
            raise asyncio.CancelledError()
        return await super().embed(texts)


class Receipts:
    def __init__(self):
        self.values = {}
    def get(self, key):
        return self.values.get(key)
    def put(self, key, value):
        self.values[key] = value


async def test_document_restart_reuses_completed_embedding_batches(tmp_path):
    embedder = CountingEmbedder(interrupt=True)
    memory = await open_memory(tmp_path, embedder)
    memory.chunk_target = 24
    parser = CountingParser()
    raw = ('Distinct source sentence with details.\n' * 150).encode()
    original = await memory.attach('alpha', raw, 'text/plain', filename='source.txt')
    args = dict(space='alpha', attachment_id=original.attachment_id)
    job = workflow(memory, tmp_path, parser)
    with pytest.raises(asyncio.CancelledError):
        await job.run('source', **args)
    assert len(embedder.calls) == 2
    assert job.status('source', **args).checkpoint_count == 1
    assert (await memory.documents.counts('alpha')).episodes == 0
    job.close()
    await memory.close()

    resumed_embedder = CountingEmbedder()
    memory = await open_memory(tmp_path, resumed_embedder)
    memory.chunk_target = 24
    job = workflow(memory, tmp_path, parser)
    try:
        result = await job.run('source', **args)
        assert parser.calls == 1 and result.reused_steps == ('extract',)
        count = result.results['index']['chunks']
        assert count > EMBED_BATCH
        assert sum(map(len, resumed_embedder.calls)) == count - EMBED_BATCH
        episode_id = result.results['index']['episode_id']
        assert len(await memory.vectors.ids('alpha')) == count
        assert len(await memory.documents.chunks_of('alpha', episode_id)) == count
        assert job.status('source', **args).checkpoint_count == 0
    finally:
        job.close()
        await memory.close()


@pytest.mark.parametrize('changed', ['text', 'order', 'model', 'dimension'])
async def test_embedding_receipts_do_not_cross_changed_inputs_or_models(changed):
    receipts = Receipts()
    first = CountingEmbedder()
    texts = ['first', 'second']
    await _embed_chunks(first, texts, checkpoint=receipts)
    second = CountingEmbedder()
    if changed == 'text':
        texts[-1] = 'changed'
    elif changed == 'order':
        texts.reverse()
    elif changed == 'model':
        second.id = 'different-model-revision'
    elif changed == 'dimension':
        second.dim += 1
    await _embed_chunks(second, texts, checkpoint=receipts)
    assert len(second.calls) == 1


async def test_malformed_provider_batch_is_not_checkpointed():
    from .test_embedding_receipts import MalformedEmbedder
    receipts = Receipts()
    with pytest.raises(ValueError, match='embedding'):
        await _embed_chunks(MalformedEmbedder('nan', fail_call=2),
                            [str(i) for i in range(EMBED_BATCH + 1)], checkpoint=receipts)
    assert len(receipts.values) == 1
    resumed = CountingEmbedder()
    await _embed_chunks(resumed, [str(i) for i in range(EMBED_BATCH + 1)], checkpoint=receipts)
    assert resumed.calls == [[str(EMBED_BATCH)]]


async def test_corrupt_receipt_cannot_be_used_as_a_searchable_vector():
    receipts = Receipts()
    await _embed_chunks(CountingEmbedder(), ['first'], checkpoint=receipts)
    key, = receipts.values
    receipts.values[key] = b'[[true]]'
    with pytest.raises(ValueError, match='embedding'):
        await _embed_chunks(CountingEmbedder(), ['first'], checkpoint=receipts)


@pytest.mark.parametrize('change', ['model', 'chunking', 'context'])
async def test_changed_index_configuration_recomputes_pending_embeddings(tmp_path, change):
    first = CountingEmbedder(interrupt=True)
    memory = await open_memory(tmp_path, first)
    memory.chunk_target = 24
    original = await memory.attach('alpha', b'Original source words.\n' * 150, 'text/plain', filename='source.txt')
    args = dict(space='alpha', attachment_id=original.attachment_id)
    job = workflow(memory, tmp_path, CountingParser())
    try:
        with pytest.raises(asyncio.CancelledError):
            await job.run('source', **args)
        second = CountingEmbedder()
        if change == 'model':
            second.id = 'new-model-revision'
        elif change == 'chunking':
            memory.chunk_target = 48
        else:
            memory.contextual_embeddings = True
        memory.embedder = second
        result = await job.run('source', **args)
        assert sum(map(len, second.calls)) == result.results['index']['chunks']
    finally:
        job.close()
        await memory.close()


async def test_deleted_source_invalidates_pending_embedding_receipts(tmp_path):
    first = CountingEmbedder(interrupt=True)
    memory = await open_memory(tmp_path, first)
    memory.chunk_target = 24
    original = await memory.attach('alpha', b'Original source words.\n' * 150, 'text/plain', filename='source.txt')
    args = dict(space='alpha', attachment_id=original.attachment_id)
    job = workflow(memory, tmp_path, CountingParser())
    try:
        with pytest.raises(asyncio.CancelledError):
            await job.run('source', **args)
        await memory.delete_space('alpha')
        second = CountingEmbedder()
        memory.embedder = second
        with pytest.raises(WorkflowError, match='sources_invalid'):
            await job.run('source', **args)
        assert second.calls == []
        assert job.status('source', **args).checkpoint_count == 0
    finally:
        job.close()
        await memory.close()


async def test_pdf_ocr_restart_reuses_pages_and_completed_embedding_batches(tmp_path):
    pytest.importorskip('pypdfium2')
    from .test_pdf_ocr import ObservedOcr
    from .test_pdf_ocr_workflow import retain, workflow as ocr_workflow
    embedder = CountingEmbedder(interrupt=True)
    memory = await open_memory(tmp_path, embedder)
    memory.chunk_target = 24
    original = await retain(memory)
    ocr = ObservedOcr('Recognized source words. ' * 150)
    job = ocr_workflow(memory, tmp_path, ocr)
    args = dict(space='alpha', attachment_id=original.attachment_id)
    with pytest.raises(asyncio.CancelledError):
        await job.run('scan', **args)
    assert ocr.calls == 2 and len(embedder.calls) == 2
    job.close()
    await memory.close()
    resumed_embedder = CountingEmbedder()
    memory = await open_memory(tmp_path, resumed_embedder)
    memory.chunk_target = 24
    job = ocr_workflow(memory, tmp_path, ocr)
    try:
        result = await job.run('scan', **args)
        assert ocr.calls == 2 and result.reused_pages == (1, 2)
        assert not result.reused_index
        assert sum(map(len, resumed_embedder.calls)) == result.added.chunks - EMBED_BATCH
    finally:
        job.close()
        await memory.close()


_CHILD = '''
import asyncio,sys
from pathlib import Path
from scone_memory import HashEmbedder,MemoryEngine
from scone_memory.backends import SqliteDocumentStore,SqliteVectorIndex
from scone_memory.backends.blobs import FileBlobStore
from scone_memory.ingestion import DocumentIngestionWorkflow
path=Path(sys.argv[1])
class Interrupted(HashEmbedder):
    calls=0
    async def embed(self,texts):
        self.calls+=1
        if self.calls==2:
            (path/'ready').write_text('first embedding batch committed')
            await asyncio.Event().wait()
        return await super().embed(texts)
async def main():
    memory=await MemoryEngine(SqliteDocumentStore(path/'memory.db'),SqliteVectorIndex(path/'memory.db'),
        Interrupted(),chunk_target=24,blobs=FileBlobStore(path/'blobs')).open()
    job=DocumentIngestionWorkflow(memory,path/'jobs.db',key=b'w'*32,parser_revision='pypdf-test-v1')
    await job.run('source',space='alpha',attachment_id=sys.argv[2])
asyncio.run(main())
'''


@pytest.mark.skipif(os.name != 'posix', reason='workflow journal requires POSIX locks')
async def test_sigkill_during_embedding_resumes_only_unfinished_batches(tmp_path):
    memory = await open_memory(tmp_path)
    original = await memory.attach('alpha', b'Original source words.\n' * 150, 'text/plain', filename='source.txt')
    await memory.close()
    with (tmp_path / 'child.log').open('wb') as log:
        child = subprocess.Popen([sys.executable, '-c', _CHILD, str(tmp_path), original.attachment_id],
                                 stdout=log, stderr=subprocess.STDOUT)
        async def ready():
            while not (tmp_path / 'ready').exists():
                assert child.poll() is None, (tmp_path / 'child.log').read_text()
                await asyncio.sleep(.01)
        try:
            await asyncio.wait_for(ready(), 10)
        finally:
            child.kill()
            child.wait(timeout=5)
    embedder = CountingEmbedder()
    memory = await open_memory(tmp_path, embedder)
    memory.chunk_target = 24
    job = workflow(memory, tmp_path, CountingParser())
    args = dict(space='alpha', attachment_id=original.attachment_id)
    try:
        state = job.status('source', **args)
        assert state.status == 'running' and state.checkpoint_count == 1
        result = await job.run('source', **args)
        assert sum(map(len, embedder.calls)) == result.results['index']['chunks'] - EMBED_BATCH
        assert (await memory.documents.counts('alpha')).episodes == 1
    finally:
        job.close()
        await memory.close()
