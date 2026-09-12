"""Owned synthetic-video worker for testing encrypted OCR recovery after SIGKILL."""
import asyncio
import hashlib
import json
from pathlib import Path
import sys

from scone_memory import HashEmbedder, MemoryEngine
from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.backends.blobs import FileBlobStore
from scone_memory.ingestion.file_workflow import DocumentIngestionWorkflow
from scone_memory.ingestion.files import document_provenance
from scone_memory.ingestion.video_frames import VideoFrameDecoder, VideoFramePolicy
from scone_memory.ingestion.video_ocr import VideoDocumentParser
from scone_memory.ocr.types import OcrRegion, OcrResult


async def main():
    state, ffmpeg, ffprobe, phase = Path(sys.argv[1]), *sys.argv[2:5]
    memory = await MemoryEngine(SqliteDocumentStore(state / 'memory.db'), SqliteVectorIndex(state / 'memory.db'),
                               HashEmbedder(), blobs=FileBlobStore(state / 'blobs')).open()
    calls = 0

    class Ocr:
        async def recognize(self, image, **options):
            nonlocal calls
            calls += 1
            with (state / 'ocr-calls').open('a') as output:
                output.write(phase + ':' + hashlib.sha256(image).hexdigest() + '\n')
            if phase == 'hold' and calls == 3:
                (state / 'ocr-entered').write_text('ready')
                await asyncio.Event().wait()
            regions = () if phase == 'hold' and calls == 2 else (
                OcrRegion(text='Retained Café survives termination.', box=(0.1, 0.1, 0.9, 0.8)),)
            return OcrResult(engine='fixture-ocr', width=64, height=32, regions=regions)

    parser = VideoDocumentParser(VideoFrameDecoder(ffmpeg_path=ffmpeg, ffprobe_path=ffprobe), Ocr(),
                                 model_revision='fixture-v1', policy=VideoFramePolicy(interval_seconds=1))
    workflow = DocumentIngestionWorkflow(memory, state / 'journal.db', key=b'v' * 32,
        parser=parser, parser_revision='video-ocr-fixture-v1', automatic_retries=False)
    try:
        source = await memory.attach('alpha', (state / 'input.mp4').read_bytes(), 'video/mp4', filename='input.mp4')
        operation = workflow.read_result if phase == 'read' else workflow.run
        result = await operation('video', space='alpha', attachment_id=source.attachment_id)
        episode_id = result.results['index']['episode_id']
        evidence = await document_provenance(memory, 'alpha', episode_id)
        (state / 'result.json').write_text(json.dumps({'episode_id': episode_id,
            'empty': [frame.empty for frame in evidence.video.frames],
            'frame_hashes': [frame.png_sha256 for frame in evidence.video.frames],
            'text': [segment.text for segment in evidence.segments]}))
    finally:
        workflow.close()
        await memory.close()


asyncio.run(main())
