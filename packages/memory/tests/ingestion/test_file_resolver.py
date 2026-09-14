"""A relative import is followed only to a file that was actually read.

Without a resolver the reader names a candidate for every relative import
by path arithmetic, whether or not the file exists. `map` confirmed its
candidates against the files it walked, through a closure of its own;
`sync` recorded claims through the engine with no resolver, so a synced
tree's graph carried an edge to `./missing` that a mapped one left out.
The resolver is now one function of the paths a walk saw, shared by map,
sync and a batch of files remembered together.
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


async def test_sync_keeps_the_import_the_tree_holds_and_leaves_out_the_one_it_does_not(tmp_path):
    """The whole point. Without a resolver the reader names a candidate by
    path arithmetic for every relative import, existing or not; with one
    over the walked files, an import of a file the tree does not hold is
    left out rather than guessed at. Sync now has the same precision as
    map."""
    (tmp_path / 'web').mkdir()
    (tmp_path / 'web' / 'store.ts').write_text('export class Shelf {}\n', encoding='utf-8')
    (tmp_path / 'web' / 'api.ts').write_text("import {Shelf} from './store';\nimport {gone} from './missing';\n", encoding='utf-8')
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), code_graph=True).open()
    try:
        await sync_directory(memory, 's', str(tmp_path), marker='tree', apply=True)
        imports = sorted(f.object for f in await memory.facts('s') if f.subject == 'web/api.ts' and f.predicate == 'imports')
        assert imports == ['web/store.ts'], imports
    finally:
        await memory.close()


async def test_a_lone_file_still_names_a_python_relative_import_by_its_own_path(tmp_path):
    """One file is no tree. A resolver over it alone would decline every
    relative import; the reader's own path arithmetic still names the
    candidate, and a file remembered on its own keeps that."""
    from scone_memory.ingestion.records import Record

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), code_graph=True).open()
    try:
        await memory.replace('s', Record(content='from .store import Shelf\n', kind='file', source='pkg/api.py', dedup_key='api'))
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


async def test_files_remembered_together_resolve_imports_among_themselves():
    """A batch is a walk of its own: the engine confirms relative imports
    against the files it was handed, without a caller's resolver, and
    leaves out an import of a file the batch does not hold."""
    from scone_memory.ingestion.records import Record

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), code_graph=True).open()
    try:
        await memory.remember_many('s', [
            Record(content='export class Shelf {}\n', kind='file', source='web/store.ts'),
            Record(content="import {Shelf} from './store';\nimport {gone} from './missing';\n", kind='file', source='web/api.ts'),
        ])
        imports = sorted(f.object for f in await memory.facts('s') if f.subject == 'web/api.ts' and f.predicate == 'imports')
        assert imports == ['web/store.ts'], imports
    finally:
        await memory.close()
