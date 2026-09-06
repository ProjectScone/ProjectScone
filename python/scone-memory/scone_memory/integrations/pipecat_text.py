"""Owned, bounded text turns through a supplied Pipecat model processor.

No default provider, authentication, HTTP routes or restart recovery. Results
are aggregated public text with a response-end frame, not a browser token stream
or proof of provider completion or model use of retrieved sources. The caller
authorizes the fixed memory space and capture.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import re
from collections.abc import Callable, Mapping
from uuid import uuid4

try:
    from pipecat.frames.frames import (
        LLMContextFrame, LLMFullResponseEndFrame,
        LLMFullResponseStartFrame, LLMTextFrame,
        FunctionCallsStartedFrame, FunctionCallInProgressFrame, FunctionCallResultFrame, InterruptionFrame,
    )
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.workers.runner import WorkerRunner
except ImportError as exc:
    raise ImportError("Text conversations require Python 3.11+ and scone-memory[pipecat]") from exc

from ..engine import MemoryEngine, Record, check_space
from .pipecat_context import SconeMemoryContextProcessor
from ._pipecat_scope import RecallScope


def _bytes(messages) -> int:
    return len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))


class _PublicReply(FrameProcessor):
    def __init__(self, done: asyncio.Future, max_bytes: int):
        super().__init__()
        self.done, self.max_bytes = done, max_bytes
        self.started = False
        self.parts: list[str] = []
        self.size = 0
        self.unsupported = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            self.unsupported = True
            self.parts.clear()
            if not self.done.done():
                self.done.set_exception(RuntimeError("model turn was interrupted"))
        if isinstance(frame, (FunctionCallsStartedFrame, FunctionCallInProgressFrame, FunctionCallResultFrame)):
            self.unsupported = True
            if not self.done.done():
                self.done.set_exception(RuntimeError("tool execution is not supported by this text boundary"))
        if direction == FrameDirection.DOWNSTREAM and not self.done.done():
            if isinstance(frame, LLMFullResponseStartFrame):
                self.started = True
            elif isinstance(frame, LLMTextFrame) and self.started:
                self.size += len(frame.text.encode("utf-8"))
                if self.size > self.max_bytes:
                    self.done.set_exception(RuntimeError("model reply exceeded the byte limit"))
                else:
                    self.parts.append(frame.text)
            elif isinstance(frame, LLMFullResponseEndFrame):
                text = "".join(self.parts)
                if not self.started or not text.strip():
                    self.done.set_exception(RuntimeError("model ended without a public reply"))
                else:
                    self.done.set_result(text)
        await self.push_frame(frame, direction)


class _OwnedPipeline(Pipeline):
    """Observe cleanup directly; the runner can absorb worker failures."""

    cleanup_complete = False

    async def _cleanup_processors(self, processors):
        outcomes = await asyncio.gather(*(p.cleanup() for p in processors), return_exceptions=True)
        if any(isinstance(outcome, BaseException) for outcome in outcomes):
            raise RuntimeError("owned processor cleanup failed")

    async def cleanup(self):
        self.cleanup_complete = False
        await super().cleanup()
        self.cleanup_complete = True


class PipecatTextConversation:
    """One active turn, fixed scope, fresh processor per turn, in-memory history.

    Factory must return a fresh, compatible text model FrameProcessor whose
    cleanup releases its owned clients. Configure credentials and endpoints
    server-side. Failures and interrupted capture close the instance. Cancellation
    during model execution can continue only after successful owned cleanup;
    callers must not silently retry uncertain provider or store writes.
    This boundary supports neither tools nor media nor live output streaming.
    """

    def __init__(
        self, memory: MemoryEngine, space: str, session_id: str,
        model_factory: Callable[[], FrameProcessor], *,
        system_prompt: str = "You are a helpful assistant.",
        where: Mapping[str, str] | None = None,
        kind: str | None = None, source_prefix: str | None = None,
        since: str | None = None, until: str | None = None,
        turn_timeout: float = 30.0, max_reply_bytes: int = 64000,
        max_history_bytes: int = 128000,
    ):
        check_space(space)
        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", session_id):
            raise ValueError("session_id must be an opaque identifier of 1..128 characters")
        if not callable(model_factory):
            raise ValueError("model_factory must supply a fresh model processor")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be nonempty text")
        if isinstance(turn_timeout, bool) or not isinstance(turn_timeout, (float, int)) or not math.isfinite(turn_timeout) or turn_timeout <= 0:
            raise ValueError("turn_timeout must be finite and positive")
        for value in (max_reply_bytes, max_history_bytes):
            if isinstance(value, bool) or not isinstance(value, int) or not 512 <= value <= 1_000_000:
                raise ValueError("byte limits must be integers in 512..1000000")
        self._history = [{"role": "system", "content": system_prompt}]
        if _bytes(self._history) > max_history_bytes:
            raise ValueError("system prompt exceeds history byte limit")
        self._memory, self._space, self._session_id = memory, space, session_id
        self._factory = model_factory
        self._scope = RecallScope.validated(where=where, kind=kind, source_prefix=source_prefix, since=since, until=until)
        self._timeout, self._max_reply, self._max_history = turn_timeout, max_reply_bytes, max_history_bytes
        self._active: asyncio.Task | None = None
        self._closed = False
        self._cancel_reusable = False

    @property
    def closed(self) -> bool:
        """Whether lifecycle owners must refuse further turns."""
        return self._closed

    async def reply(self, text: str) -> dict:
        if self._closed:
            raise RuntimeError("conversation is closed")
        if self._active is not None:
            raise RuntimeError("a conversation turn is already active")
        if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 32000:
            raise ValueError("message must contain 1..32000 UTF-8 bytes of nonempty text")
        messages = [*self._history, {"role": "user", "content": text}]
        if _bytes(messages) > self._max_history:
            raise ValueError("conversation history byte limit reached; start a new conversation")
        self._active = asyncio.create_task(self._reply(messages))
        try:
            return await self._active
        except asyncio.CancelledError:
            self._closed = self._closed or not self._cancel_reusable
            raise
        except BaseException:
            self._closed = True
            raise
        finally:
            self._active = None

    async def close(self) -> None:
        self._closed = True
        active = self._active
        if active is not None:
            active.cancel()
            await asyncio.gather(active, return_exceptions=True)

    async def _record(self, turn_id: str, role: str, text: str) -> int:
        # An interrupted write acknowledgment cannot prove what was persisted.
        self._cancel_reusable = False
        metadata = {"integration": "pipecat-text", "session_id": self._session_id,
                    "turn_id": turn_id, "role": role, "representation": "aggregated_text",
                    "capture_status": "submitted" if role == "user" else "aggregated"}
        if role == "assistant":
            metadata["completion_evidence"] = "response_end_frame"
            metadata["provider_completion"] = "unverified"
        [added] = await self._memory.remember_many(self._space, [Record(
            text, kind="conversation", source=self._session_id,
            metadata=metadata,
            dedup_key=f"pipecat-text:{self._session_id}:{turn_id}:{role}",
        )])
        return added.episode_id

    async def _reply(self, messages: list[dict]) -> dict:
        turn_id = uuid4().hex
        async with asyncio.timeout(self._timeout):
            user_id = await self._record(turn_id, "user", messages[-1]["content"])
            text, receipt = await self._execute(messages)
            complete = [*messages, {"role": "assistant", "content": text}]
            if _bytes(complete) > self._max_history:
                raise RuntimeError("model reply exceeded the conversation history byte limit")
            if self._closed:
                raise asyncio.CancelledError()
            assistant_id = await self._record(turn_id, "assistant", text)
            self._history = complete
            return {"turn_id": turn_id, "text": text, "provider_completion": "unverified", "user_episode_id": user_id,
                    "assistant_episode_id": assistant_id, "memory_context": receipt}

    async def _execute(self, messages: list[dict]) -> tuple[str, dict | None]:
        done = asyncio.get_running_loop().create_future()
        started = asyncio.Event()
        pipeline_failed = False
        recall = SconeMemoryContextProcessor(self._memory, self._space, self._session_id, **self._scope.kwargs())
        model = self._factory()
        if not isinstance(model, FrameProcessor):
            raise TypeError("model_factory must return a Pipecat FrameProcessor")
        collector = _PublicReply(done, self._max_reply)
        pipeline = _OwnedPipeline([recall, model, collector])
        worker = PipelineWorker(pipeline,
                                enable_rtvi=False, cancel_on_idle_timeout=False,
                                cancel_timeout_secs=1)
        runner = WorkerRunner(handle_sigint=False, check_dangling_tasks=False)

        @worker.event_handler("on_pipeline_started")
        async def on_started(_worker, _frame):
            started.set()

        @worker.event_handler("on_pipeline_error")
        async def on_error(_worker, _frame):
            nonlocal pipeline_failed
            pipeline_failed = True
            if not done.done():
                done.set_exception(RuntimeError("model pipeline failed"))

        @worker.event_handler("on_pipeline_timeout")
        async def on_timeout(_worker, _frame):
            await on_error(_worker, _frame)

        @worker.event_handler("on_pipeline_finished")
        async def on_finished(_worker, _frame):
            if not done.done():
                done.set_exception(RuntimeError("model pipeline ended before a complete reply"))

        running = None
        try:
            await runner.add_workers(worker)
            running = asyncio.create_task(runner.run())
            await asyncio.wait_for(started.wait(), 5)
            await worker.queue_frame(LLMContextFrame(LLMContext(copy.deepcopy(messages))))
            text = await done
            # No audio needs draining in a text-only turn. EndFrame draining
            # can wait forever and leave a queued cancellation behind it.
            # The reply boundary was already observed; cancel remaining work.
            await runner.cancel(reason="public text response complete")
            # WorkerRunner handles its own CancelledError during teardown. Do
            # not let that consume the enclosing turn deadline/cancellation.
            await asyncio.shield(running)
            if pipeline_failed or collector.unsupported or not pipeline.cleanup_complete:
                raise RuntimeError("model pipeline failed during finalization")
            return text, recall.last_receipt.as_metadata() if recall.last_receipt else None
        finally:
            if running is not None and not running.done():
                await runner.cancel()
                await asyncio.wait_for(running, 5)
            if done.done() and not done.cancelled():
                done.exception()  # observe startup/error receipt even when cancellation won
            self._cancel_reusable = (
                running is not None and running.done() and not running.cancelled()
                and running.exception() is None and pipeline.cleanup_complete
                and not pipeline_failed and not collector.unsupported
            )


__all__ = ["PipecatTextConversation"]
