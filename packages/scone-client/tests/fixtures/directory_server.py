"""Standard configured local host for the independently installed client."""
import asyncio
import json
import os
from pathlib import Path
import sys

import uvicorn
from scone_memory import HashEmbedder, MemoryEngine, FileBlobStore
from scone_memory.backends.sqlite import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.api.__main__ import build_app
from scone_memory.runtime.config import Settings


async def run():
    state, port = Path(sys.argv[1]), int(sys.argv[2])
    (state / 'notes').mkdir(exist_ok=True)
    config = state / 'directory.json'
    if not config.exists():
        config.write_text(json.dumps({'schema_version': 1, 'state_dir': 'sync-state', 'key_env': 'SCONE_CLIENT_SYNC_KEY',
            'store_id': 'client-directory-fixture', 'collections': [{'collection_id': 'notes', 'label': 'Local notes',
                'space': 'alpha', 'root': 'notes', 'parser_revision': 'client-fixture-v1', 'allow_delete_missing': True}]}))
        config.chmod(0o600)
    os.environ['SCONE_CLIENT_SYNC_KEY'] = 'ab'*32
    memory = await MemoryEngine(SqliteDocumentStore(state/'catalog.db'), SqliteVectorIndex(state/'catalog.db'),
                                HashEmbedder(), blobs=FileBlobStore(state/'blobs')).open()
    app = build_app(Settings.from_env({'SCONE_API_KEYS': 'agent-fixture:alpha', 'SCONE_DIRECTORY_SYNC_CONFIG': str(config)}), memory)
    service = app.state.directory_sync_service
    parser = service._collections[('alpha', 'notes')].sync.parser
    original = parser.parse
    async def counted(*args, **kwargs):
        with (state/'parses').open('a') as output:
            output.write('parse\n')
        while (state/'pause').exists():
            await asyncio.sleep(.01)
        return await original(*args, **kwargs)
    parser.parse = counted
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
