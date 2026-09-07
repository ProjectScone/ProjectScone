"""An engine that can be closed.

Every backend that holds a client or a file has a close(), and nothing
above them called it: the MCP server closed two of the four stores by
hand, and a fixture that tried `await memory.close()` found no such
method. Closing is the engine's job, because only the engine knows which
stores it holds.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.blobs import InMemoryBlobStore
from scone_memory.events import InMemoryEventLog


class Closable:
    """A store that records whether it was closed, and how many times."""

    def __init__(self, inner):
        self._inner, self.closed = inner, 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def close(self) -> None:
        self.closed += 1


async def test_closing_the_engine_closes_every_store_it_holds():
    documents = Closable(InMemoryDocumentStore())
    vectors = Closable(InMemoryVectorIndex())
    events = Closable(InMemoryEventLog())
    blobs = Closable(InMemoryBlobStore())
    engine = await MemoryEngine(documents, vectors, HashEmbedder(), events=events, blobs=blobs).open()

    await engine.close()

    assert (documents.closed, vectors.closed, events.closed, blobs.closed) == (1, 1, 1, 1)


async def test_closing_twice_is_harmless_and_closes_each_store_once():
    documents = Closable(InMemoryDocumentStore())
    engine = await MemoryEngine(documents, InMemoryVectorIndex(), HashEmbedder()).open()

    await engine.close()
    await engine.close()

    assert documents.closed == 1


async def test_a_store_without_close_is_simply_skipped():
    """The in-memory stores have nothing to release. Requiring every
    backend to grow a no-op close() would be the wrong side of the
    contract to widen."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()

    await engine.close()  # must not raise


async def test_one_store_failing_to_close_does_not_stop_the_others():
    """A client that fails to disconnect must not leave the file store
    open beside it. Every store is asked; the first failure is raised
    after all of them have been."""

    class Broken(Closable):
        async def close(self) -> None:
            self.closed += 1
            raise RuntimeError("connection reset during close")

    documents = Broken(InMemoryDocumentStore())
    vectors = Closable(InMemoryVectorIndex())
    engine = await MemoryEngine(documents, vectors, HashEmbedder()).open()

    with pytest.raises(RuntimeError, match="connection reset"):
        await engine.close()
    assert vectors.closed == 1
