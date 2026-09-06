"""Tools for anyone writing a store adapter.

Import the contract into a test module and provide an ``engine`` fixture
built on your store; the behavioural suite that every in-tree backend
passes then runs against yours::

    # tests/test_my_store.py  (pytest-asyncio in auto mode)
    import pytest
    from scone_memory import HashEmbedder, InMemoryVectorIndex, MemoryEngine
    from scone_memory.testing import Clock
    from scone_memory.testing.contract import *  # noqa: F401,F403

    @pytest.fixture
    async def engine():
        clock = Clock()
        e = await MemoryEngine(MyDocumentStore(...), InMemoryVectorIndex(), HashEmbedder(),
                               chunk_target=200, clock=clock).open()
        e.test_clock = clock
        yield e

Pair your DocumentStore with the in-process vector index (or your
VectorIndex with the in-process document store) so a failure points at
one component.
"""


class Clock:
    """A clock the test moves by hand, so "now" is a value, not a race."""

    def __init__(self, start: str = "2025-01-01T00:00:00.000Z") -> None:
        self.now = start

    def __call__(self) -> str:
        return self.now


__all__ = ["Clock"]
