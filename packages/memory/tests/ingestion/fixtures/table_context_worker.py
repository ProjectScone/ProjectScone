"""Owned process proving contextual embedding checkpoints survive process death."""
import asyncio
import json
from pathlib import Path
import sys

from scone_memory import HashEmbedder, MemoryEngine
from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.backends.blobs import FileBlobStore
from scone_memory.ingestion.file_workflow import DocumentIngestionWorkflow


async def run():
    state, phase = Path(sys.argv[1]), sys.argv[2]

    class Model(HashEmbedder):
        calls = 0

        async def embed(self, texts):
            self.calls += 1
            with (state / 'calls.jsonl').open('a') as log:
                log.write(json.dumps({'phase': phase, 'count': len(texts),
                                      'context': any('Table context:' in text for text in texts)}) + '\n')
            if phase == 'hold' and self.calls == 2:
                (state / 'entered').write_text('ready')
                await asyncio.Event().wait()
            return await super().embed(texts)

    memory = await MemoryEngine(SqliteDocumentStore(state / 'memory.db'),
        SqliteVectorIndex(state / 'memory.db'), Model(), blobs=FileBlobStore(state / 'blobs'),
        chunk_target=120, table_context_embeddings=True).open()
    job = DocumentIngestionWorkflow(memory, state / 'jobs.db', key=b't' * 32,
        parser_revision='declared-html-v1', automatic_retries=False)
    try:
        original = await memory.attach('alpha', (state / 'input.html').read_bytes(), 'text/html', filename='input.html')
        if phase == 'inspect':
            status = job.status('table', space='alpha', attachment_id=original.attachment_id)
            assert status is not None and status.checkpoint_count >= 1
            return
        result = await job.run('table', space='alpha', attachment_id=original.attachment_id)
        (state / 'result.json').write_text(json.dumps(result.results['index']))
    finally:
        job.close()
        await memory.close()


asyncio.run(run())
