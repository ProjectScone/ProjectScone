"""A relative import is followed to a file that was actually read.

`map` resolved relative imports with a closure over the files it walked;
`sync` recorded claims through the engine with no resolver at all, so a
synced tree's graph had no edge for `from .store import Shelf` while a
mapped one did. The resolver is now one function of the paths a walk saw,
shared by both, and sync hands it to the engine for every file it stores.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.ingestion.code_resolution import file_resolver
from scone_memory.ingestion.sync import sync_directory

pytestmark = pytest.mark.asyncio

SEEN = {'pkg/__init__.py', 'pkg/store.py', 'pkg/api.py', 'pkg/sub/__init__.py', 'web/x.ts', 'web/dir/index.ts',
        'web/util.js', 'svc/main.go', 'svc/lib.rs'}


@pytest.mark.parametrize('path, level, module, expected', [
    ('pkg/api.py', 1, 'store', 'pkg/store.py'),                 # from .store import Shelf
    ('pkg/api.py', 1, '', 'pkg/__init__.py'),                   # from . import store
    ('pkg/sub/__init__.py', 2, 'store', 'pkg/store.py'),        # from ..store import Shelf
    ('pkg/api.py', 1, 'sub', 'pkg/sub/__init__.py'),            # from .sub import thing
    ('web/x.ts', 1, './dir', 'web/dir/index.ts'),               # import x from './dir'
    ('web/x.ts', 1, './util', 'web/util.js'),                   # extension left off
    ('web/dir/index.ts', 1, '../x', 'web/x.ts'),                # a parent directory
    ('pkg/api.py', 1, 'missing', None),                         # not a file the walk read
    ('pkg/api.py', 3, 'store', None),                           # above the root
])
async def test_a_relative_import_resolves_only_to_a_file_the_walk_read(path, level, module, expected):
    assert file_resolver(SEEN)(path, level, module) == expected


async def test_the_resolver_is_bound_to_the_paths_it_was_given():
    assert file_resolver(set())('pkg/api.py', 1, 'store') is None
    assert file_resolver({'pkg/store.py'})('pkg/api.py', 1, 'store') == 'pkg/store.py'


async def test_sync_records_the_edge_a_relative_import_makes(tmp_path):
    """The whole point: a synced tree has the same import edges a mapped one
    does. Before, the engine recorded the file's claims with no resolver
    and the relative import was left out rather than guessed at."""
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'pkg' / 'store.py').write_text('class Shelf:\n    pass\n', encoding='utf-8')
    (tmp_path / 'pkg' / 'api.py').write_text('from .store import Shelf\n\ndef put():\n    return Shelf()\n', encoding='utf-8')
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), code_graph=True).open()
    try:
        await sync_directory(memory, 's', str(tmp_path), marker='tree', apply=True)
        held = {(f.subject, f.predicate, f.object) for f in await memory.facts('s')}
        assert ('pkg/api.py', 'imports', 'pkg/store.py') in held, sorted(t for t in held if t[1] == 'imports')
    finally:
        await memory.close()


async def test_a_changed_file_keeps_its_resolved_edge_and_loses_the_one_it_dropped(tmp_path):
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'pkg' / 'store.py').write_text('class Shelf:\n    pass\n', encoding='utf-8')
    (tmp_path / 'pkg' / 'util.py').write_text('def helper():\n    pass\n', encoding='utf-8')
    (tmp_path / 'pkg' / 'api.py').write_text('from .store import Shelf\nfrom .util import helper\n', encoding='utf-8')
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), code_graph=True).open()
    try:
        await sync_directory(memory, 's', str(tmp_path), marker='tree', apply=True)
        (tmp_path / 'pkg' / 'api.py').write_text('from .store import Shelf\n', encoding='utf-8')
        receipt = await sync_directory(memory, 's', str(tmp_path), marker='tree', apply=True)
        held = {(f.subject, f.predicate, f.object) for f in await memory.facts('s')}
        assert ('pkg/api.py', 'imports', 'pkg/store.py') in held
        assert ('pkg/api.py', 'imports', 'pkg/util.py') not in held
        assert receipt.claims_closed == 1, receipt.text()
    finally:
        await memory.close()


async def test_files_remembered_together_follow_imports_among_themselves():
    """A batch is a walk of its own: the engine resolves relative imports
    among the files it was handed, without a caller's resolver."""
    from scone_memory.ingestion.records import Record

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), code_graph=True).open()
    try:
        await memory.remember_many('s', [
            Record(content='class Shelf:\n    pass\n', kind='file', source='pkg/store.py'),
            Record(content='from .store import Shelf\n', kind='file', source='pkg/api.py'),
        ])
        held = {(f.subject, f.predicate, f.object) for f in await memory.facts('s')}
        assert ('pkg/api.py', 'imports', 'pkg/store.py') in held, sorted(t for t in held if t[1] == 'imports')
    finally:
        await memory.close()
