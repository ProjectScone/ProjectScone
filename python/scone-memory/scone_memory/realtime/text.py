"""Scone-owned text conversations: public streams, scoped context and capture."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Awaitable, Callable, Mapping
from contextlib import aclosing
from contextvars import ContextVar
from uuid import uuid4

from ..engine import MemoryEngine, Record, check_space
from ..recall_scope import RecallScope
from .context import MemoryContext
from .events import TextDelta, ReplyCompleted, TextModel
from .lifecycle import cancel_once, settle

_OWNER = ContextVar("scone_text_owner", default=None)


def _bytes(messages):
    return len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))


class TextConversation:
    """One active turn, immutable scope, fresh provider per turn.

    Models implement respond(messages), an async generator of public TextDelta
    and one ReplyCompleted, plus aclose(). No tool/media/reasoning events are
    accepted. The caller owns authentication, capture consent and provider keys.
    Cleanup is cooperative and may exceed the cancellation deadline.
    """

    def __init__(self, memory: MemoryEngine, space: str, session_id: str,
                 model_factory: Callable[[], TextModel], *,
                 system_prompt="You are a helpful assistant.",
                 where: Mapping[str, str] | None = None, kind=None,
                 source_prefix=None, since=None, until=None, turn_timeout=30.0,
                 max_reply_bytes=64000, max_history_bytes=128000):
        check_space(space)
        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", session_id):
            raise ValueError("session_id must be an opaque identifier of 1..128 characters")
        if not callable(model_factory):
            raise ValueError("model_factory must supply a fresh model")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be nonempty text")
        if isinstance(turn_timeout, bool) or not isinstance(turn_timeout, (float, int)) or not math.isfinite(turn_timeout) or turn_timeout <= 0:
            raise ValueError("turn_timeout must be finite and positive")
        for value in (max_reply_bytes, max_history_bytes):
            if type(value) is not int or not 512 <= value <= 1_000_000:
                raise ValueError("byte limits must be integers in 512..1000000")
        self._history = [{"role": "system", "content": system_prompt}]
        if _bytes(self._history) > max_history_bytes:
            raise ValueError("system prompt exceeds history byte limit")
        self._memory, self._space, self._session_id = memory, space, session_id
        self._factory = model_factory
        scope = RecallScope.validated(where=where, kind=kind, source_prefix=source_prefix, since=since, until=until)
        self._context = MemoryContext(memory, space, session_id, **scope.kwargs())
        self._timeout, self._max_reply, self._max_history = turn_timeout, max_reply_bytes, max_history_bytes
        self._active = None
        self._closed = False
        self._cancel_reusable = False

    @property
    def closed(self):
        return self._closed

    async def reply(self, text: str, *, on_text: Callable[[str], Awaitable[None]] | None = None) -> dict:
        """Observe provisional public chunks; returned result confirms capture."""
        if self._closed:
            raise RuntimeError("conversation is closed")
        if self._active is not None:
            raise RuntimeError("a conversation turn is already active")
        if on_text is not None and not callable(on_text):
            raise ValueError("on_text must be an async callable")
        if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 32000:
            raise ValueError("message must contain 1..32000 UTF-8 bytes of nonempty text")
        messages = [*self._history, {"role": "user", "content": text}]
        if _bytes(messages) > self._max_history:
            raise ValueError("conversation history byte limit reached; start a new conversation")
        self._cancel_reusable = False
        active = self._active = asyncio.create_task(self._reply(messages, on_text))
        try:
            return await asyncio.shield(active)
        except asyncio.CancelledError:
            cancel_once(active)
            outcome, _ = await settle(active)
            self._closed = self._closed or not self._cancel_reusable
            if isinstance(outcome, Exception):
                self._closed = True
                raise outcome
            raise
        except BaseException:
            self._closed = True
            raise
        finally:
            self._active = None

    async def close(self):
        if _OWNER.get() is self:
            raise RuntimeError("close must be called outside the provider/public text observer")
        self._closed = True
        if self._active is not None:
            cancel_once(self._active)
            outcome, cancelled = await settle(self._active)
            if isinstance(outcome, Exception):
                raise outcome
            if cancelled:
                raise asyncio.CancelledError()

    async def _record(self, turn_id, role, text):
        self._cancel_reusable = False
        metadata = dict(integration="scone-text", session_id=self._session_id,
                        turn_id=turn_id, role=role, representation="aggregated_text",
                        capture_status="submitted" if role == "user" else "aggregated")
        if role == "assistant":
            metadata.update(completion_evidence="adapter_end_and_stream_closed", provider_completion="unverified")
        [added] = await self._memory.remember_many(self._space, [Record(
            text, kind="conversation", source=self._session_id, metadata=metadata,
            dedup_key=f"scone-text:{self._session_id}:{turn_id}:{role}")])
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError()
        return added.episode_id

    async def _reply(self, messages, on_text):
        token = _OWNER.set(self)
        try:
            async with asyncio.timeout(self._timeout):
                turn_id = uuid4().hex
                user_id = await self._record(turn_id, "user", messages[-1]["content"])
                text, receipt = await self._execute(messages, on_text)
                complete = [*messages, {"role": "assistant", "content": text}]
                if _bytes(complete) > self._max_history:
                    raise RuntimeError("model reply exceeded the conversation history byte limit")
                if self._closed or asyncio.current_task().cancelling():
                    raise asyncio.CancelledError()
                assistant_id = await self._record(turn_id, "assistant", text)
                self._history = complete
                return dict(turn_id=turn_id, text=text, provider_completion="unverified",
                            user_episode_id=user_id, assistant_episode_id=assistant_id,
                            memory_context=receipt)
        finally:
            _OWNER.reset(token)

    async def _execute(self, messages, on_text):
        request, receipt = await self._context.prepare(messages)
        if self._closed or asyncio.current_task().cancelling():
            raise asyncio.CancelledError()
        model = self._factory()
        if not callable(getattr(model, "aclose", None)):
            raise TypeError("model must implement respond and aclose")
        safe_cancel = True
        try:
            if not callable(getattr(model, "respond", None)):
                raise TypeError("model must implement respond and aclose")
            parts, size, completed = [], 0, False
            async with aclosing(model.respond(request)) as events:
                async for event in events:
                    if self._closed or asyncio.current_task().cancelling():
                        raise asyncio.CancelledError()
                    if completed:
                        raise RuntimeError("provider emitted events after completion")
                    if isinstance(event, ReplyCompleted):
                        completed = True
                    elif isinstance(event, TextDelta) and isinstance(event.text, str):
                        size += len(event.text.encode("utf-8"))
                        if size > self._max_reply:
                            raise RuntimeError("model reply exceeded the byte limit")
                        parts.append(event.text)
                        if on_text is not None and event.text:
                            safe_cancel = False  # consumer effects are uncertain until it returns
                            try:
                                await on_text(event.text)
                            except asyncio.CancelledError as exc:
                                if asyncio.current_task().cancelling():
                                    raise
                                raise RuntimeError("public text observer failed") from exc
                            except Exception as exc:
                                raise RuntimeError("public text observer failed") from exc
                            safe_cancel = True
                    else:
                        raise RuntimeError("provider emitted an unsupported event")
            text = "".join(parts)
            if not completed or not text.strip():
                raise RuntimeError("provider ended without a complete public reply")
            return text, receipt
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            safe_cancel = False
            # Keep diagnostics out of public exceptions; preserve cause for hosts.
            if str(exc).startswith(("model reply exceeded", "public text observer")):
                raise
            raise RuntimeError("model provider failed") from exc
        finally:
            async def cleanup():
                await model.aclose()
            closing = asyncio.create_task(cleanup())
            outcome, cancelled = await settle(closing)
            if isinstance(outcome, BaseException):
                self._cancel_reusable = False
                raise RuntimeError("model provider cleanup failed") from outcome
            self._cancel_reusable = safe_cancel and not self._closed
            if cancelled:
                raise asyncio.CancelledError()
