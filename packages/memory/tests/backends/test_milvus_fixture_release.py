"""An embedded database's server is stopped by the fixture that started it.

`milvus_lite` keeps one background gRPC server per distinct `.db` path in
a module-level registry until `atexit`, and `MilvusClient.close` releases
only the client connection -- so closing the index cannot stop the
server. A fresh database per backend parametrisation therefore
accumulated one live server per test for the life of the process: a
full-suite run held 47 `grpc/_server.py` serving threads and 53 idle pool
workers at once, and logged
`GOAWAY ... ENHANCE_YOUR_CALM: too_many_pings`.

Whether that leak caused the stall that hung two gate runs was never
established -- the stall had a different cause entirely -- so this is
worth fixing on its own terms and not as a cure for something else.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, MemoryEngine

pytestmark = pytest.mark.asyncio
needs = pytest.mark.skipif(importlib.util.find_spec("milvus_lite") is None,
                           reason="milvus-lite not installed")


@needs
async def test_releasing_stops_the_given_database_and_no_other():
    """Only the paths handed in. Another test's client may still be using
    its own database, so `release_all()` would be a fixture reaching into
    a store it did not create."""
    import tempfile

    from milvus_lite.server_manager import server_manager_instance

    from scone_memory.backends import MilvusVectorIndex
    from tests.conftest import _release_embedded

    with tempfile.TemporaryDirectory() as directory:
        mine = os.path.join(directory, "mine.db")
        other = os.path.join(directory, "other.db")
        engines = []
        for path in (mine, other):
            engine = await MemoryEngine(InMemoryDocumentStore(), MilvusVectorIndex(path),
                                        HashEmbedder()).open()
            await engine.remember("default", "A note, so the index is actually used.")
            engines.append(engine)
        running = server_manager_instance._servers
        assert os.path.abspath(mine) in running and os.path.abspath(other) in running, \
            "both databases should have started a server for this test to be about anything"
        try:
            await engines[0].close()
            _release_embedded([mine])
            assert os.path.abspath(mine) not in running, "the given database was not released"
            assert os.path.abspath(other) in running, \
                "releasing one database must not stop another's server"
        finally:
            await engines[1].close()
            _release_embedded([mine, other])
    assert os.path.abspath(other) not in server_manager_instance._servers


@needs
async def test_releasing_nothing_and_an_unknown_path_are_both_quiet():
    """Teardown runs after failures too, so it must not raise: a
    `milvus_lite` without this call is a reason to leak a server, not to
    fail a suite."""
    from tests.conftest import _release_embedded

    _release_embedded([])
    _release_embedded(["/nonexistent/never-opened.db"])
