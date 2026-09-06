"""The sync wrapper runs an engine on its own loop in a thread. A store
whose client binds to the loop it was opened on (pymongo, psycopg's pool)
must therefore be opened on that loop, not on a throwaway one."""

from __future__ import annotations

import asyncio

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine, SyncMemoryEngine
from scone_memory.backends.memory import InMemoryDocumentStore as Reference


class LoopBound(Reference):
    """Behaves like a client that refuses to work on any loop but the one
    it was opened on."""

    name = "loop-bound"

    async def open(self):
        self.loop = asyncio.get_running_loop()
        return self

    async def counts(self, space):
        if asyncio.get_running_loop() is not self.loop:
            raise RuntimeError("cannot use this client in a different event loop")
        return await super().counts(space)


def test_from_env_builds_on_the_loop_that_serves(monkeypatch):
    import scone_memory.config as config

    async def fake_build(settings):
        documents = await LoopBound().open()
        return await MemoryEngine(documents, InMemoryVectorIndex(), HashEmbedder()).open()

    monkeypatch.setattr(config, "build_engine", fake_build)
    with SyncMemoryEngine.from_env({}) as memory:
        assert memory.status("default").episodes == 0, "the store answers because it lives on the wrapper's loop"


def test_a_prebuilt_engine_still_works_and_a_closed_wrapper_refuses():
    memory = SyncMemoryEngine(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()))
    memory.remember("default", "kept")
    assert memory.recall("default", "kept").items[0].episode_id == 1
    memory.close()
    with pytest.raises(RuntimeError, match="closed"):
        memory.status("default")
