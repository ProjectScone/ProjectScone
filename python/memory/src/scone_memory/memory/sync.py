"""A blocking facade for code that is not async: scripts, notebooks,
Airflow tasks, pandas pipelines.

The engine runs on its own event loop in a daemon thread, so this works
from plain Python and from inside a running loop (Jupyter) alike, and
the caller never sees a coroutine.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
import threading
from typing import Awaitable, Callable, Iterable, Mapping, Optional, Sequence

from .engine import ImportSummary, MemoryEngine, Profile, Record
from ..core.models import Added, Episode, Fact, RecallResult, Status
from ..core.ports import SourcePage


class SyncMemoryEngine:
    """``engine`` may be a MemoryEngine or a coroutine function that builds
    one. The builder form runs on the wrapper's own loop, which matters
    for stores whose async clients bind to the loop they were opened on
    (pymongo, psycopg's pool): built anywhere else, every later call
    would fail with "cannot use ... in a different event loop"."""

    def __init__(self, engine: MemoryEngine | Callable[[], Awaitable[MemoryEngine]]) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="scone-memory", daemon=True)
        self._lock = threading.Lock()
        self._pending: set[concurrent.futures.Future] = set()
        self._closing = False
        self._shutdown_error: Optional[BaseException] = None
        self._thread.start()
        try:
            if callable(engine):
                self._engine = self._run(engine())
            else:
                self._engine = engine
            self._run(self._engine.open())
        except BaseException:
            self.close()
            raise

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()
            asyncio.set_event_loop(None)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "SyncMemoryEngine":
        import os

        from ..runtime.config import Settings, build_engine

        settings = Settings.from_env(env if env is not None else os.environ)
        return cls(lambda: build_engine(settings))

    @property
    def engine(self) -> MemoryEngine:
        return self._engine

    def _run(self, coro):
        if threading.current_thread() is self._thread:
            coro.close()
            raise RuntimeError("Blocking SyncMemoryEngine calls cannot run on its own event loop")
        # Admission and shutdown use the same lock, so every accepted coroutine
        # is enqueued before the drain task. Rejected coroutines are never leaked.
        with self._lock:
            if self._closing or not self._thread.is_alive():
                coro.close()
                raise RuntimeError("SyncMemoryEngine is closed or closing")
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            self._pending.add(future)
        try:
            return future.result()
        except concurrent.futures.CancelledError:
            raise RuntimeError("SyncMemoryEngine was closed while this call was pending") from None
        finally:
            with self._lock:
                self._pending.discard(future)

    async def _settle_calls(self) -> None:
        # Completed asyncio Tasks disappear from all_tasks before their result
        # is delivered to a thread-safe Future. Await that bridge explicitly.
        with self._lock:
            pending = list(self._pending)
        if pending:
            await asyncio.gather(*(asyncio.wrap_future(future) for future in pending), return_exceptions=True)

    async def _drain_tasks(self) -> None:
        current = asyncio.current_task()
        while pending := {task for task in asyncio.all_tasks() if task is not current}:
            for task in pending:
                task.cancel()
            results = await asyncio.gather(*pending, return_exceptions=True)
            for task, result in zip(pending, results):
                if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                    self._loop.call_exception_handler({"message": "Task failed during SyncMemoryEngine shutdown",
                                                       "exception": result, "task": task})

    async def _shutdown(self) -> None:
        try:
            await self._drain_tasks()
            await self._settle_calls()
            await self._loop.shutdown_asyncgens()
            await self._drain_tasks()
            await self._loop.shutdown_default_executor()
            # A finishing executor worker can enqueue loop work while we join
            # it. Finalize that tail before destroying the loop as well.
            await self._drain_tasks()
            await self._loop.shutdown_asyncgens()
            await self._drain_tasks()
        except BaseException as error:
            self._shutdown_error = error
        finally:
            self._loop.stop()

    def close(self, *, timeout: float = 5.0) -> None:
        """Cancel and drain the owned loop. A timeout leaves it draining, not
        destroyed; subsequent calls wait for the same shutdown. Underlying
        backend clients remain caller-managed, as for the async engine.
        """
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("close timeout must be a finite positive number")
        if threading.current_thread() is self._thread:
            raise RuntimeError("SyncMemoryEngine.close cannot block its own event loop")
        with self._lock:
            if not self._closing:
                self._closing = True
                self._loop.call_soon_threadsafe(self._loop.create_task, self._shutdown())
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError("SyncMemoryEngine shutdown is still draining; call close again to wait")
        if self._shutdown_error is not None:
            raise RuntimeError("SyncMemoryEngine shutdown failed") from self._shutdown_error

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

    def source_page(self, space: str, *, before: Optional[int] = None,
                    limit: int = 25, kind: Optional[str] = None) -> SourcePage:
        return self._run(self._engine.source_page(space, before=before, limit=limit, kind=kind))

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
