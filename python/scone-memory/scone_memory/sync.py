"""A blocking facade for code that is not async: scripts, notebooks,
Airflow tasks, pandas pipelines.

The engine runs on its own event loop in a daemon thread, so this works
from plain Python and from inside a running loop (Jupyter) alike, and
the caller never sees a coroutine.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Awaitable, Callable, Iterable, Mapping, Optional, Sequence

from .engine import ImportSummary, MemoryEngine, Profile, Record
from .models import Added, Episode, Fact, RecallResult, Status


class SyncMemoryEngine:
    """``engine`` may be a MemoryEngine or a coroutine function that builds
    one. The builder form runs on the wrapper's own loop, which matters
    for stores whose async clients bind to the loop they were opened on
    (pymongo, psycopg's pool): built anywhere else, every later call
    would fail with "cannot use ... in a different event loop"."""

    def __init__(self, engine: MemoryEngine | Callable[[], Awaitable[MemoryEngine]]) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="scone-memory", daemon=True)
        self._pending: set[concurrent.futures.Future] = set()
        self._lock = threading.Lock()
        self._thread.start()
        if callable(engine):
            self._engine = self._run(engine())
        else:
            self._engine = engine
        self._run(self._engine.open())

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "SyncMemoryEngine":
        import os

        from .config import Settings, build_engine

        settings = Settings.from_env(env if env is not None else os.environ)
        return cls(lambda: build_engine(settings))

    @property
    def engine(self) -> MemoryEngine:
        return self._engine

    def _run(self, coro):
        if not self._thread.is_alive():
            coro.close()  # otherwise "coroutine was never awaited" at collection
            raise RuntimeError("SyncMemoryEngine is closed")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        with self._lock:
            self._pending.add(future)
        try:
            return future.result()
        except concurrent.futures.CancelledError:
            raise RuntimeError("SyncMemoryEngine was closed while this call was pending") from None
        finally:
            with self._lock:
                self._pending.discard(future)

    def close(self) -> None:
        # A call still running when the loop stops would never resolve and
        # its caller would wait forever; cancel it so it raises instead.
        with self._lock:
            pending = list(self._pending)
        for future in pending:
            future.cancel()
        if self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
        self._loop.close()

    def __enter__(self) -> "SyncMemoryEngine":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def remember(self, space: str, content: str, **kwargs) -> Added:
        return self._run(self._engine.remember(space, content, **kwargs))

    def remember_many(self, space: str, records: Iterable[Record]) -> list[Added]:
        return self._run(self._engine.remember_many(space, list(records)))

    def recall(self, space: str, query: str, **kwargs) -> RecallResult:
        return self._run(self._engine.recall(space, query, **kwargs))

    def episodes(self, space: str, where: Mapping[str, str], limit: Optional[int] = None) -> list[Episode]:
        return self._run(self._engine.episodes(space, where, limit))

    def forget(self, space: str, episode_id: int) -> None:
        return self._run(self._engine.forget(space, episode_id))

    def assert_fact(self, space: str, subject: str, predicate: str, object: str, **kwargs) -> Fact:
        return self._run(self._engine.assert_fact(space, subject, predicate, object, **kwargs))

    def close_fact(self, space: str, fact_id: int, reason: str) -> Fact:
        return self._run(self._engine.close_fact(space, fact_id, reason))

    def facts(self, space: str, **kwargs) -> list[Fact]:
        return self._run(self._engine.facts(space, **kwargs))

    def profile(self, space: str, limit: int = 10) -> Profile:
        return self._run(self._engine.profile(space, limit))

    def tags(self, space: str) -> dict[str, int]:
        return self._run(self._engine.tags(space))

    def status(self, space: str) -> Status:
        return self._run(self._engine.status(space))

    def export(self, space: str) -> list[dict]:
        async def collect():
            return [r async for r in self._engine.export(space)]

        return self._run(collect())

    def import_records(self, space: str, records: Iterable[Mapping]) -> ImportSummary:
        return self._run(self._engine.import_records(space, list(records)))
