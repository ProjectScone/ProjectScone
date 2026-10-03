"""Native execution must work even when add-on frameworks cannot be imported."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

from ..paths import PACKAGE_ROOT


def test_native_retrieval_and_subsystems_do_not_load_framework_addons(tmp_path):
    # A fresh interpreter catches transitive imports even when another test has
    # already loaded an adapter. Raise AssertionError so optional-import fallbacks
    # cannot conceal a native path trying to load an external framework.
    program = textwrap.dedent('''\
        import asyncio
        import importlib
        import importlib.abc
        from pathlib import Path
        import secrets
        import socket
        import sys

        forbidden = (
            'langchain', 'langchain_core', 'langchain_community', 'langsmith',
            'llama_index', 'agents', 'graphify', 'paddleocr', 'paddle',
            'pipecat', 'ragflow', 'supermemory', 'SupermemoryReference', 'reference',
            'scone_memory.integrations.composition',
            'scone_memory.integrations.langchain',
            'scone_memory.integrations.llamaindex',
            'scone_memory.integrations.openai_agents',
            'scone_memory.backends.langchain', 'scone_memory.bench.comparative',
        )

        class NativeImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + '.') for name in forbidden):
                    raise AssertionError('Native execution loaded an add-on: ' + fullname)

        def no_network(*args, **kwargs):
            raise AssertionError('Native execution attempted a network connection')

        sys.meta_path.insert(0, NativeImports())
        socket.socket.connect = no_network
        socket.socket.connect_ex = no_network
        socket.create_connection = no_network

        for name in (
            'agents.run_service', 'agents.evidence_loop', 'agents.handoff_workflow',
            'ingestion.documents', 'ingestion.directory_sync', 'entities.service',
            'retrieval.adaptive', 'realtime.audio', 'ocr.layout_json', 'telephony.transport',
        ):
            importlib.import_module('scone_memory.' + name)

        from scone_memory import HashEmbedder, MemoryEngine
        from scone_memory.agents.retrieval import build_edge_retrieval_runner
        from scone_memory.backends.sqlite import SqliteDocumentStore, SqliteVectorIndex

        async def main():
            database = Path('native.db')
            memory = await MemoryEngine(SqliteDocumentStore(database), SqliteVectorIndex(database), HashEmbedder()).open()
            try:
                source = await memory.remember('team', 'Native calibration evidence.',
                    source='docs/native', metadata={'project': 'allowed'})
                await memory.remember('team', 'Private calibration evidence.', metadata={'project': 'other'})
                runner = build_edge_retrieval_runner(memory, Path('workflow.db'), key=secrets.token_bytes(32),
                    space='team', where={'project': 'allowed'})
                try:
                    result = await runner.run('lookup', space='team', scope={'where': {'project': 'allowed'}},
                        inputs={'query': 'calibration'})
                    assert len(result.results['evidence']) == 1
                    record = result.results['evidence'][0]
                    assert record['episode_id'] == source.episode_id
                    assert record['text'] == 'Native calibration evidence.'
                    assert record['source'] == 'docs/native'
                finally:
                    runner.close()
            finally:
                await memory.close()

        asyncio.run(main())
        assert not any(name == root or name.startswith(root + '.') for name in sys.modules for root in forbidden)
    ''')
    result = subprocess.run(
        [sys.executable, '-c', program], cwd=tmp_path, capture_output=True, text=True, timeout=30,
        env={**os.environ, 'PYTHONPATH': str(PACKAGE_ROOT / 'src'), 'LANGSMITH_TRACING': 'true',
             'LANGCHAIN_TRACING_V2': 'true'},
    )
    assert result.returncode == 0, result.stdout + result.stderr
