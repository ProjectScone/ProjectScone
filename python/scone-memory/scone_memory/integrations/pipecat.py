"""Optional Pipecat public-transcript capture into Scone episodes.

This adapter observes context-message and assistant-turn events. It does not
subscribe to thoughts, raw frames, audio, tools or complete context snapshots.
An aggregated transcript is not proof of response completion or audio playback.
"""

from __future__ import annotations

import asyncio
import math
from uuid import uuid4

try:
    from pipecat.processors.aggregators.llm_response_universal import (
        AssistantTurnStoppedMessage,
        LLMContextAggregatorPair,
        UserTurnMessageAddedMessage,
    )
except ImportError as exc:  # optional; base Scone never imports this module
    raise ImportError(
        "Pipecat capture requires Python 3.11+ and pip install 'scone-memory[pipecat]'"
    ) from exc

from ..engine import MemoryEngine, Record, check_space


class CaptureError(RuntimeError):
    """Capture became incomplete; a pipeline finishing is not a successful save."""


class SconePipecatMemory:
    """Capture public transcript events from one Pipecat aggregator pair.

    Attach before running the pipeline. Pipecat owns its event-handler tasks;
    after the runner has cleaned up, call ``raise_if_failed()`` and ``detach()``.
    Poll ``error`` during a live session to surface a capture failure promptly.
    A failure latches: later events are not silently saved past an unknown gap.

    Each instance has a fresh capture identity, so reconnecting or repeating
    text cannot overwrite earlier turns. This is not a durable replay queue or
    an exactly-once delivery contract. Sequence numbers are callback-observation
    order within that capture, not global session order or token timing.
    """

    def __init__(
        self, memory: MemoryEngine, space: str, session_id: str, *,
        write_timeout: float = 5.0, max_pending: int = 128,
    ) -> None:
        check_space(space)
        if not isinstance(session_id, str) or not 1 <= len(session_id) <= 128:
            raise ValueError("session_id must be a string of 1..128 characters")
        if isinstance(write_timeout, bool) or not isinstance(write_timeout, (int, float)) or not math.isfinite(write_timeout) or write_timeout <= 0:
            raise ValueError("write_timeout must be finite and positive")
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending <= 0:
            raise ValueError("max_pending must be a positive integer")
        self.memory = memory
        self.space = space
        self.session_id = session_id
        self.capture_id = uuid4().hex
        self.stored_count = 0
        self.empty_count = 0
        self.unrecorded_count = 0
        self.write_timeout = write_timeout
        self.max_pending = max_pending
        self._pending = 0
        self._seq = 0
        self._lock = asyncio.Lock()
        self._error: BaseException | None = None
        self._pair: LLMContextAggregatorPair | None = None

    @property
    def error(self) -> BaseException | None:
        """First capture failure, retained even when Pipecat catches exceptions."""
        return self._error

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise CaptureError(
                "Pipecat transcript capture is incomplete. Inspect stored records "
                "before retrying; the last write may have taken effect."
            ) from self._error

    def attach(self, pair: LLMContextAggregatorPair) -> None:
        if self._pair is not None:
            raise ValueError("capture is already attached; detach it before reusing")
        self.raise_if_failed()
        pair.user().add_event_handler("on_user_turn_message_added", self.on_user_message)
        pair.assistant().add_event_handler("on_assistant_turn_stopped", self.on_assistant_turn)
        self._pair = pair

    def detach(self) -> None:
        """Remove future subscriptions; does not cancel already dispatched events."""
        if self._pair is not None:
            self._pair.user().remove_event_handler("on_user_turn_message_added", self.on_user_message)
            self._pair.assistant().remove_event_handler("on_assistant_turn_stopped", self.on_assistant_turn)
            self._pair = None

    async def on_user_message(self, aggregator, message: UserTurnMessageAddedMessage) -> None:
        await self._capture("user", message.content, message.timestamp, "context_message", message.user_id)

    async def on_assistant_turn(self, aggregator, message: AssistantTurnStoppedMessage) -> None:
        await self._capture(
            "assistant", message.content, message.timestamp,
            "interrupted" if message.interrupted else "aggregated",
        )

    async def _capture(
        self, role: str, content: str, timestamp: str, status: str, user_id: str | None = None,
    ) -> None:
        if not content or not content.strip():
            self.empty_count += 1
            return
        if self._error is not None:
            self.unrecorded_count += 1
            return
        if self._pending >= self.max_pending:
            self._error = CaptureError("Pipecat capture backlog exceeded max_pending")
            self.unrecorded_count += 1
            return
        seq = self._seq
        self._seq += 1
        metadata = {
            "integration": "pipecat", "session_id": self.session_id,
            "capture_id": self.capture_id, "seq": str(seq), "role": role,
            "capture_status": status, "representation": "aggregated_text",
        }
        if user_id:
            metadata["user_id"] = user_id
        if timestamp:
            metadata["source_timestamp"] = timestamp
        record = Record(
            content, kind="conversation", source=self.session_id,
            created_at=timestamp or None, metadata=metadata,
            dedup_key=f"pipecat:{self.capture_id}:{seq}",
        )
        self._pending += 1
        try:
            # Keep lock release and failure handling in this task. On Python
            # 3.11, wait_for wraps _store in another task: a queued write could
            # acquire its released lock before this handler latched the error.
            async with asyncio.timeout(self.write_timeout):
                stored = await self._store(record)
            if stored:
                self.stored_count += 1
            else:
                self.unrecorded_count += 1
        except asyncio.CancelledError as exc:
            self._error = self._error or exc
            self.unrecorded_count += 1
            raise
        except Exception as exc:
            self._error = self._error or exc
            self.unrecorded_count += 1
        finally:
            self._pending -= 1

    async def _store(self, record: Record) -> bool:
        async with self._lock:
            if self._error is not None:
                return False
            await self.memory.remember_many(self.space, [record])
            return True


__all__ = ["CaptureError", "SconePipecatMemory"]
