"""Process-local public text window, not a transcript or durable event log."""

from __future__ import annotations

import asyncio
from collections import deque
import json

MAX_BYTES = 65536
MAX_CHUNKS = 256


class TextWindow:
    def __init__(self):
        self._chunks: deque[tuple[int, str, int]] = deque()
        self._bytes = 0
        self.last_sequence = 0
        self.closed = False
        self.failed = False
        self.changed = asyncio.Event()

    def append(self, text: str) -> None:
        if self.closed:
            raise RuntimeError("text observation is closed")
        try:
            if not isinstance(text, str):
                raise ValueError("public text must be a string")
            size = len(text.encode("utf-8"))
            if size > MAX_BYTES or self.last_sequence == 2**63 - 1:
                raise ValueError("public text exceeds the stream limit")
        except ValueError:
            # A custom runtime may catch callback exceptions. Latch failure so
            # neither a later valid chunk nor its final return can hide it.
            self.failed = True
            self.finish()
            raise
        if not size:
            return
        self.last_sequence += 1
        self._chunks.append((self.last_sequence, text, size))
        self._bytes += size
        while self._bytes > MAX_BYTES or len(self._chunks) > MAX_CHUNKS:
            self._bytes -= self._chunks.popleft()[2]
        self.changed.set()

    def finish(self) -> None:
        self.closed = True
        self._chunks.clear()
        self._bytes = 0
        self.changed.set()

    def next_after(self, cursor: int) -> tuple[int | None, tuple[int, str] | None]:
        """A missing prefix, next whole chunk, or no available text."""
        if type(cursor) is not int or not 0 <= cursor <= self.last_sequence:
            raise ValueError("invalid stream cursor")
        if self.closed or not self._chunks:
            return None, None
        first = self._chunks[0][0]
        if cursor < first - 1:
            return first, None
        for sequence, text, _ in self._chunks:
            if sequence > cursor:
                return None, (sequence, text)
        return None, None

    async def wait_after(self, cursor: int) -> None:
        # Recheck durable-in-process sequence when scheduled, not just an Event
        # flag another reader can clear before this coroutine starts waiting.
        while not self.closed and self.last_sequence <= cursor:
            self.changed.clear()
            await self.changed.wait()


def sse(event: str, data: dict, *, sequence: int | None = None) -> str:
    # Event names and sequence are server-owned, never model/caller strings.
    identifier = "" if sequence is None else f"id: {sequence}\n"
    return f"event: {event}\n{identifier}data: {json.dumps(data, ensure_ascii=False, allow_nan=False)}\n\n"
