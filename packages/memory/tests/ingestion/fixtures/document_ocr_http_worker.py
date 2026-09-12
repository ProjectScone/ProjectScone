"""Isolated process fixture: actual HTTP/SQLite/PDF renderer, scripted OCR text."""
import asyncio
from pathlib import Path
import socket
import sys

import uvicorn

from scone_memory import HashEmbedder, MemoryEngine
from scone_memory.api import create_app
from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.backends.blobs import FileBlobStore
from scone_memory.ingestion.document_ocr import DocumentOcr
from scone_memory.ingestion.import_service import DocumentImportService, ImportParserBinding
from scone_memory.ocr import OcrRegion, OcrResult
from scone_memory.ocr.tesseract import png_dimensions


async def main():
    root, mode = Path(sys.argv[1]), sys.argv[2]
    class Recognizer:
        calls = 0
        async def recognize(self, image, *, max_pixels, **kwargs):
            self.calls += 1
            if mode == 'hold' and self.calls == 2:
                (root / 'page-two-entered').write_text('ready')
                await asyncio.Event().wait()
            if mode == 'readonly':
                raise AssertionError('read-only restart attempted OCR')
            width, height = png_dimensions(image, max_pixels)
            with (root / 'recognized').open('a') as output:
                output.write(mode + ':' + str(self.calls) + '\n')
            return OcrResult(engine='scripted-fixture', width=width, height=height,
                regions=(OcrRegion(text='Saved Polaris' if mode == 'hold' else 'Resumed Juniper', box=(.1,.1,.9,.2)),))
    memory = await MemoryEngine(SqliteDocumentStore(root/'memory.db'),
        SqliteVectorIndex(root/'memory.db'), HashEmbedder(), blobs=FileBlobStore(root/'blobs')).open()
    config = DocumentOcr(Recognizer())
    service = DocumentImportService(root/'imports', key=b'w'*32, memory=memory,
        parser_for=lambda selection: ImportParserBinding('fixture-v1', config.parser(selection)), max_active=1)
    app = create_app(memory, {'writer':'alpha', 'reader':'alpha', 'other':'beta'},
        roles={'writer':'write', 'reader':'read'}, document_ocr=config, document_import_service=service)
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level='error', lifespan='off'))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    async def requested_stop():
        await asyncio.to_thread(sys.stdin.readline)
        server.should_exit = True
    control = asyncio.create_task(requested_stop())
    try:
        while not server.started:
            if task.done():
                await task
            await asyncio.sleep(.01)
        (root/'port').write_text(str(listener.getsockname()[1]))
        await task
    finally:
        control.cancel()
        await service.aclose()
        await memory.close()
        listener.close()


if __name__ == '__main__':
    asyncio.run(main())
