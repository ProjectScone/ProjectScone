"""Persistent local document API with an observable, explicitly released parser."""
import asyncio
import json
from pathlib import Path
import sys

import uvicorn

from scone_memory import HashEmbedder, MemoryEngine, FileBlobStore
from scone_memory.backends.sqlite import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.api.app import create_app
from scone_memory.ingestion.formats.registry import BuiltinDocumentParser
from scone_memory.ingestion.import_service import DocumentImportService, ImportParserBinding


async def run():
    state, port = Path(sys.argv[1]), int(sys.argv[2])
    memory = await MemoryEngine(SqliteDocumentStore(state / 'catalog.db'), SqliteVectorIndex(state / 'catalog.db'),
                                HashEmbedder(), blobs=FileBlobStore(state / 'blobs')).open()

    class Parser:
        async def parse(self, data, filename, limits):
            with (state / 'parses.jsonl').open('a') as output:
                output.write(json.dumps({'filename': filename, 'bytes': len(data)}) + '\n')
            while not (state / 'release').exists():
                await asyncio.sleep(0.01)
            return await BuiltinDocumentParser().parse(data, filename, limits)

    service = DocumentImportService(state / 'imports', key=b'k' * 32, memory=memory,
        parser_for=lambda _: ImportParserBinding('native-client-test-v1', Parser()))
    app = create_app(memory, {'agent-fixture': 'alpha'}, document_import_service=service)
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
        await service.aclose()
        await memory.close()


asyncio.run(run())
