"""Owned local process used to interrupt extraction after its model work."""
import asyncio
import json
from pathlib import Path
import sys

from scone_memory import HashEmbedder, MemoryEngine
from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.backends.blobs import FileBlobStore
from scone_memory.ingestion.document_media import DocumentMedia
from scone_memory.ingestion.file_workflow import DocumentIngestionWorkflow
from scone_memory.ingestion.files import document_provenance
from scone_memory.ingestion.formats.media import MediaDocumentParser, TranscriptionSegment


async def run():
    state, executable, phase = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    memory = await MemoryEngine(SqliteDocumentStore(state / 'memory.db'),
        SqliteVectorIndex(state / 'memory.db'), HashEmbedder(),
        blobs=FileBlobStore(state / 'blobs')).open()

    async def transcribe(audio):
        with (state / 'model-calls').open('a') as calls:
            calls.write(phase + '\n')
        return (TranscriptionSegment(text='Checkpointed Café survives process death.',
                                     start_seconds=0.0, end_seconds=0.05),)

    config = DocumentMedia(MediaDocumentParser(transcribe, ffmpeg_executable=executable),
                           revision='scripted-local-v1')
    job = DocumentIngestionWorkflow(memory, state / 'journal.db', key=b'r' * 32,
        parser=config.parser(), parser_revision='media-worker-v1', automatic_retries=False)
    try:
        original = await memory.attach('alpha', (state / 'input.wav').read_bytes(),
                                       'audio/wav', filename='input.wav')
        if phase == 'hold':
            attach = memory.attach

            async def pause(space, raw, media_type, **kwargs):
                if kwargs.get('filename') == 'document-provenance.json':
                    (state / 'manifest-entered').write_text('ready')
                    await asyncio.Event().wait()
                return await attach(space, raw, media_type, **kwargs)

            memory.attach = pause
        result = await job.run('audio', space='alpha', attachment_id=original.attachment_id)
        evidence = await document_provenance(memory, 'alpha', result.results['index']['episode_id'])
        (state / 'result.json').write_text(json.dumps({
            'episode_id': result.results['index']['episode_id'],
            'text': evidence.segments[0].text,
            'audio_sha256': evidence.metadata['audio_wav_sha256'],
        }))
    finally:
        job.close()
        await memory.close()


asyncio.run(run())
