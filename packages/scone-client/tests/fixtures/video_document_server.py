"""Persistent native host with explicit frame OCR and observable recognition."""
import asyncio
import json
import os
from pathlib import Path
import shutil
import sys

import uvicorn

from scone_memory import HashEmbedder, MemoryEngine, FileBlobStore
from scone_memory.api.__main__ import build_app
from scone_memory.backends.sqlite import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.ingestion.document_video import DocumentVideo
from scone_memory.ingestion.video_frames import VideoFrameDecoder, VideoFramePolicy
from scone_memory.ingestion.video_ocr import VideoDocumentParser
from scone_memory.ocr.types import OcrRegion, OcrResult
from scone_memory.runtime.config import Settings


async def run():
    state, port = Path(sys.argv[1]), int(sys.argv[2])
    memory = await MemoryEngine(SqliteDocumentStore(state / 'catalog.db'), SqliteVectorIndex(state / 'catalog.db'),
                                HashEmbedder(), blobs=FileBlobStore(state / 'blobs')).open()

    class Recognizer:
        async def recognize(self, image, **options):
            with (state / 'recognitions').open('a') as output:
                output.write('frame\n')
            return OcrResult(engine='native-client-fixture', width=64, height=32,
                regions=(OcrRegion(text='Café launch Friday', box=(0.1, 0.1, 0.9, 0.8), score=0.8),))

    video = DocumentVideo(VideoDocumentParser(VideoFrameDecoder(ffmpeg_path=shutil.which('ffmpeg'),
        ffprobe_path=shutil.which('ffprobe')), Recognizer(), model_revision='native-fixture-v1',
        policy=VideoFramePolicy(interval_seconds=1)), revision='native-host-v1')
    config = state / 'jobs.json'
    config.write_text(json.dumps({'schema_version': 1, 'state_dir': 'imports',
        'key_env': 'NATIVE_VIDEO_JOB_KEY', 'parser_revision': 'native-v1'}))
    config.chmod(0o600)
    os.environ['NATIVE_VIDEO_JOB_KEY'] = 'ab' * 32
    app = build_app(Settings.from_env({'SCONE_API_KEYS': 'agent-fixture:alpha:write',
        'SCONE_DOCUMENT_JOBS_CONFIG': str(config)}), memory, document_video=video)
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning'))

    async def commands():
        await asyncio.to_thread(sys.stdin.readline)
        server.should_exit = True

    task = asyncio.create_task(commands())
    try:
        await server.serve()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await memory.close()


asyncio.run(run())
