"""Owned, bounded vector indexing after durable native text capture."""
from __future__ import annotations

import asyncio
import logging
import math

from ..ingestion.deferred_text import CapturedText
from ..ingestion.records import Record
from ..memory.engine import MemoryEngine
from .lifecycle import cancel_once, settle


class DeferredTextCapture:
    """One worker, with the document store's ingestion journal as recovery intent.

    The queue bounds active and waiting sources together. Overflow and exhausted
    retries leave durable intents for explicit engine.recover() or next open().
    Capture acknowledges retained text, not vector availability. The owner must
    close this service before closing its engine; closing never waits for all
    embeddings and never discards a pending intent.
    """

    def __init__(self, memory: MemoryEngine, *, max_pending: int = 128,
                 max_attempts: int = 3, retry_delay: float = 0.1):
        if type(max_pending) is not int or not 1 <= max_pending <= 10000:
            raise ValueError('max_pending must be an integer in 1..10000')
        if type(max_attempts) is not int or not 1 <= max_attempts <= 10:
            raise ValueError('max_attempts must be an integer in 1..10')
        if type(retry_delay) not in (int, float) or not math.isfinite(retry_delay) or not 0 <= retry_delay <= 60:
            raise ValueError('retry_delay must be finite seconds in 0..60')
        self.memory = memory
        self._limit = max_pending
        self._attempts = max_attempts
        self._delay = retry_delay
        self._queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue(maxsize=max_pending)
        self._pending: set[tuple[str, int]] = set()
        self._worker: asyncio.Task[None] | None = None
        self._shutdown_task: asyncio.Task[None] | None = None
        self._closed = False
        self._capture_lock = asyncio.Lock()

    def start(self) -> None:
        if self._closed:
            raise RuntimeError('deferred capture service is closed')
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name='scone-text-indexing')

    async def capture(self, space: str, record: Record) -> CapturedText:
        async with self._capture_lock:
            if self._closed:
                raise RuntimeError('deferred capture service is closed')
            if self._worker is None or self._worker.done():
                raise RuntimeError('deferred capture service is not running')
            result = await self.memory.capture_text(space, record)
            key = (space, result.episode_id)
            if result.indexing == 'pending' and key not in self._pending:
                if len(self._pending) < self._limit:
                    self._pending.add(key)
                    self._queue.put_nowait(key)
                else:
                    logging.getLogger(__name__).warning('capture.indexing_deferred',
                        extra={'event': 'capture.indexing_deferred', 'reason': 'queue_full'})
            return result

    async def wait_idle(self) -> None:
        """Wait for admitted jobs; durable overflow/failed intents may remain."""
        await self._queue.join()

    async def aclose(self) -> None:
        self._closed = True
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self._shutdown(), name='scone-text-indexing-shutdown')
        outcome, cancelled = await settle(self._shutdown_task)
        if isinstance(outcome, BaseException):
            raise outcome
        if cancelled:
            raise asyncio.CancelledError()

    async def _shutdown(self) -> None:
        # A capture already writing owns its intent until the write settles.
        async with self._capture_lock:
            if self._worker is not None:
                cancel_once(self._worker)
                outcome, cancelled = await settle(self._worker)
                if isinstance(outcome, Exception):
                    raise outcome
                while not self._queue.empty():
                    self._queue.get_nowait()
                    self._queue.task_done()
                self._pending.clear()
                if cancelled:
                    raise asyncio.CancelledError()

    async def _run(self) -> None:
        while True:
            key = await self._queue.get()
            try:
                await self._index(*key)
            finally:
                self._pending.discard(key)
                self._queue.task_done()

    async def _index(self, space: str, episode_id: int) -> None:
        for attempt in range(self._attempts):
            try:
                await self.memory.index_captured_text(space, episode_id)
                return
            except Exception as error:
                logging.getLogger(__name__).warning('capture.indexing_failed', extra={
                    'event': 'capture.indexing_failed', 'exception_type': type(error).__name__,
                    'attempt': attempt + 1})
                if attempt + 1 < self._attempts:
                    await asyncio.sleep(self._delay)
