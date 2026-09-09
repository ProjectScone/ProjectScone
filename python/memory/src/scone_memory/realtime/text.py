"""Scone-owned text conversations: public streams, scoped context and capture."""

from __future__ import annotations

import asyncio
import json
import math
import logging
import time
import re
from collections.abc import Awaitable, Callable, Mapping
from contextlib import aclosing
from contextvars import ContextVar
from typing import Literal
from uuid import uuid4

from ..agents.evidence_loop import EvidenceToolLoop, ToolLoopLimits, ToolLoopResult, ToolModel
from ..integrations.scoped_tools import ScopedMemoryTools
from ..memory.engine import MemoryEngine, Record, check_space
from ..retrieval.recall_scope import RecallScope
from ..retrieval.adaptive import AdaptiveRetriever
from .answer_requirements import AnswerRequirements, validated_requirements
from .answer_review import AnswerReviewer, AnswerReviewLimits, ReviewedAnswer, review_answer, require_contextual_reviewer
from .context import ContextReceipt, MemoryContext
from .evidence_answer import EvidenceAnswerError, EvidenceSelector, construct_evidence_answer, _ABSTENTION
from .events import TextDelta, ReplyCompleted, TextModel
from .lifecycle import cancel_once, settle
from .review_evidence import prepare_review_evidence
from .tool_answer import tool_context_receipt, review_tool_answer

_OWNER = ContextVar("scone_text_owner", default=None)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful conversational assistant. Answer the latest user message naturally. "
    "Use the actual conversation history when discussing what we have said. Retrieved memory "
    "is optional background, not a new user request or a replacement for this conversation. "
    "Ignore irrelevant matches. Cite sources only when they support an answer; do not report "
    "record IDs or capture metadata unless asked."
)


def _bytes(messages):
    return len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))


class _AnswerReviewFailure(RuntimeError):
    def __init__(self, message: str, receipt: dict[str, object]) -> None:
        super().__init__(message)
        self.answer_review = receipt


class _EvidenceAnswerFailure(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__("evidence answer unavailable")
        self.evidence_answer: dict[str, object] = {
            "status": "unavailable", "errors": [reason], "verified_accuracy": False}


class TextConversation:
    """One active turn, immutable scope, fresh provider per turn.

    Models implement respond(messages), an async generator of public TextDelta
    and one ReplyCompleted, plus aclose(). No tool/media/reasoning events are
    accepted. The caller owns authentication, capture consent and provider keys.
    Cleanup is cooperative and may exceed the cancellation deadline.
    Optional answer review buffers the draft until source checks and review
    finish; only the accepted final text reaches the observer and capture.
    Optional answer_requirements are shared with generation and review. They
    buffer all generation and gate final publication on format, byte and line
    checks; natural-language instructions remain model guidance, not proof.
    Optional evidence selection instead constructs answers from checked source
    cards without a generation provider. Turns without prepared memory retain
    normal generation and streaming, with a skipped evidence-answer receipt.
    Native tool mode instead runs a bounded ToolModel over ordinary history,
    checks retained sources before publication and again before capture, and
    returns transient evidence packets plus cacheable identifiers/fingerprints.
    ToolModel adapters own and close resources within each complete() call.
    """

    _memory: MemoryEngine
    _space: str
    _session_id: str
    _scope: RecallScope
    _tool_factory: Callable[[], ToolModel] | None
    _tool_limits: ToolLoopLimits
    _tool_initial_search: bool
    _tool_compute: bool
    _evidence_answer_policy: Literal["when_available", "required"]
    _answer_reviewer: AnswerReviewer | None
    _answer_requirements: AnswerRequirements | None
    _review_limits: AnswerReviewLimits
    _review_policy: Literal["report", "require_supported"]
    _closed: bool
    _history: list[dict[str, str]]
    _max_history: int
    _max_reply: int
    _cancel_reusable: bool
    _evidence_selector: EvidenceSelector | None
    _evidence_answer_timeout: float

    def __init__(self, memory: MemoryEngine, space: str, session_id: str,
                 model_factory: Callable[[], TextModel] | None = None, *,
                 system_prompt=DEFAULT_SYSTEM_PROMPT,
                 where: Mapping[str, str] | None = None, kind=None,
                 source_prefix=None, since=None, until=None, turn_timeout=30.0,
                 max_reply_bytes=64000, max_history_bytes=128000,
                 adaptive_retriever: AdaptiveRetriever | None = None, recall_timeout: float = 2.0,
                 neighbor_chunks: int = 0,
                 answer_reviewer: AnswerReviewer | None = None, review_limits: AnswerReviewLimits | None = None,
                 review_policy: Literal["report", "require_supported"] = "report",
                 answer_requirements: AnswerRequirements | None = None,
                 evidence_selector: EvidenceSelector | None = None, evidence_answer_timeout: float = 20.0,
                 evidence_answer_policy: Literal["when_available", "required"] = "when_available",
                 tool_model_factory: Callable[[], ToolModel] | None = None,
                 tool_limits: ToolLoopLimits | None = None, tool_initial_search: bool = False,
                 tool_compute: bool = False):
        check_space(space)
        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", session_id):
            raise ValueError("session_id must be an opaque identifier of 1..128 characters")
        if type(evidence_answer_policy) is not str or evidence_answer_policy not in ("when_available", "required"):
            raise ValueError("evidence_answer_policy must be when_available or required")
        if evidence_answer_policy == "required" and evidence_selector is None:
            raise ValueError("required evidence answers need evidence_selector")
        if not callable(model_factory) and not (model_factory is None and (tool_model_factory is not None
                or (evidence_selector is not None and evidence_answer_policy == "required"))):
            raise ValueError("model_factory must supply a fresh model")
        if tool_model_factory is not None and not callable(tool_model_factory):
            raise ValueError("tool_model_factory must supply a fresh native tool model")
        if tool_limits is not None and (tool_model_factory is None or not isinstance(tool_limits, ToolLoopLimits)):
            raise ValueError("tool limits require a tool model and ToolLoopLimits")
        if type(tool_initial_search) is not bool or (tool_initial_search and tool_model_factory is None):
            raise ValueError("tool_initial_search requires a boolean and a tool model")
        if tool_model_factory is not None and any(value is not None for value in
                (evidence_selector, adaptive_retriever)):
            raise ValueError("tool mode cannot combine independent retrieval or extractive evidence")
        if tool_model_factory is not None and neighbor_chunks != 0:
            raise ValueError("neighbor_chunks is for ordinary search; tool mode uses read_memory")
        if type(tool_compute) is not bool or (tool_compute and tool_model_factory is None):
            raise ValueError("tool_compute requires a boolean and a tool model")
        self._tool_compute = tool_compute
        self._tool_factory = tool_model_factory
        self._tool_initial_search = tool_initial_search
        self._tool_limits = ToolLoopLimits.model_validate((tool_limits or ToolLoopLimits()).model_dump())
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be nonempty text")
        if isinstance(turn_timeout, bool) or not isinstance(turn_timeout, (float, int)) or not math.isfinite(turn_timeout) or turn_timeout <= 0:
            raise ValueError("turn_timeout must be finite and positive")
        for value in (max_reply_bytes, max_history_bytes):
            if type(value) is not int or not 512 <= value <= 1_000_000:
                raise ValueError("byte limits must be integers in 512..1000000")
        if type(review_policy) is not str or review_policy not in ("report", "require_supported"):
            raise ValueError("review_policy must be report or require_supported")
        if answer_reviewer is not None and not callable(getattr(answer_reviewer, "review", None)):
            raise ValueError("answer_reviewer must implement review")
        self._answer_requirements = validated_requirements(answer_requirements)
        if answer_reviewer is not None:
            require_contextual_reviewer(answer_reviewer, self._answer_requirements)
        if answer_reviewer is None and (review_limits is not None or review_policy != "report"):
            raise ValueError("review settings require answer_reviewer")
        if review_limits is not None and not isinstance(review_limits, AnswerReviewLimits):
            raise ValueError("review_limits must be AnswerReviewLimits")
        if evidence_selector is not None and not callable(getattr(evidence_selector, "select", None)):
            raise ValueError("evidence_selector must implement select")
        if evidence_selector is not None and answer_reviewer is not None:
            raise ValueError("evidence_selector and answer_reviewer cannot be combined")
        if (isinstance(evidence_answer_timeout, bool) or not isinstance(evidence_answer_timeout, (int, float))
                or not math.isfinite(evidence_answer_timeout) or not 1 <= evidence_answer_timeout <= 180):
            raise ValueError("evidence_answer_timeout must be finite in 1..180")
        if evidence_selector is None and evidence_answer_timeout != 20.0:
            raise ValueError("evidence answer settings require evidence_selector")
        self._evidence_selector, self._evidence_answer_timeout = evidence_selector, float(evidence_answer_timeout)
        self._evidence_answer_policy = evidence_answer_policy
        self._answer_reviewer, self._review_policy = answer_reviewer, review_policy
        self._review_limits = (AnswerReviewLimits.model_validate(dict(vars(review_limits)), strict=True)
                               if review_limits is not None else AnswerReviewLimits())
        if self._answer_requirements is not None:
            system_prompt += '\n\n' + self._answer_requirements.prompt()
        self._history = [{"role": "system", "content": system_prompt}]
        if _bytes(self._history) > max_history_bytes:
            raise ValueError("system prompt exceeds history byte limit")
        self._memory, self._space, self._session_id = memory, space, session_id
        self._factory = model_factory
        scope = RecallScope.validated(where=where, kind=kind, source_prefix=source_prefix, since=since, until=until)
        self._scope = scope
        self._context = MemoryContext(memory, space, session_id, **scope.kwargs(),
                                      adaptive_retriever=adaptive_retriever, recall_timeout=recall_timeout,
                                      neighbor_chunks=neighbor_chunks)
        self._timeout, self._max_reply, self._max_history = turn_timeout, max_reply_bytes, max_history_bytes
        self._active: asyncio.Task[dict] | None = None
        self._closed = False
        self._cancel_reusable = False

    @property
    def closed(self):
        return self._closed

    async def reply(self, text: str, *, on_text: Callable[[str], Awaitable[None]] | None = None) -> dict:
        """Observe provisional chunks, or one checked final text; return confirms capture."""
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

    async def _record(self, turn_id, role, text, *, extractive=False, tool_source_status=None, evidence_abstention=False):
        started = time.perf_counter()
        self._cancel_reusable = False
        metadata = dict(integration="scone-text", session_id=self._session_id,
                        turn_id=turn_id, role=role, representation="aggregated_text",
                        capture_status="submitted" if role == "user" else "aggregated")
        if role == "assistant":
            metadata.update(completion_evidence="source_checked_extractive_answer" if extractive else "adapter_end_and_stream_closed",
                            provider_completion="unverified")
            if evidence_abstention:
                metadata['completion_evidence'] = 'evidence_abstention'
            if tool_source_status is not None:
                metadata['completion_evidence'] = ('source_checked_tool_answer' if tool_source_status == 'retained'
                                                   else 'native_tool_answer')
        try:
            [added] = await self._memory.remember_many(self._space, [Record(
                text, kind="conversation", source=self._session_id, metadata=metadata,
                dedup_key=f"scone-text:{self._session_id}:{turn_id}:{role}")])
        except BaseException as error:
            logging.getLogger(__name__).warning("capture.failed", extra={"event": "capture.failed",
                "session_id": self._session_id, "role": role, "exception_type": type(error).__name__,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)})
            raise
        logging.getLogger(__name__).info("capture.finished", extra={"event": "capture.finished",
            "session_id": self._session_id, "role": role, "episode_id": added.episode_id,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)})
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError()
        return added.episode_id

    async def _reply(self, messages, on_text):
        token = _OWNER.set(self)
        try:
            async with asyncio.timeout(self._timeout):
                turn_id = uuid4().hex
                user_id = await self._record(turn_id, "user", messages[-1]["content"])
                text, receipt, answer_review, evidence_answer = await self._execute(messages, on_text)
                complete = [*messages, {"role": "assistant", "content": text}]
                if _bytes(complete) > self._max_history:
                    raise RuntimeError("model reply exceeded the conversation history byte limit")
                if self._closed or asyncio.current_task().cancelling():
                    raise asyncio.CancelledError()
                assistant_id = await self._record(turn_id, "assistant", text,
                    extractive=evidence_answer is not None and evidence_answer.get("status") != "skipped",
                    evidence_abstention=evidence_answer is not None and evidence_answer.get("source_status") == "none",
                    tool_source_status=receipt.get('tool_retrieval', {}).get('source_status'))
                self._history = complete
                result = dict(turn_id=turn_id, text=text, provider_completion="unverified",
                            user_episode_id=user_id, assistant_episode_id=assistant_id,
                            memory_context=receipt)
                if answer_review is not None:
                    result["answer_review"] = answer_review
                if evidence_answer is not None:
                    result["evidence_answer"] = evidence_answer
                return result
        finally:
            _OWNER.reset(token)

    async def _execute(self, messages, on_text):
        if self._tool_factory is not None:
            return await self._execute_tools(messages, on_text)
        request, receipt = await self._context.prepare(messages)
        if self._closed or asyncio.current_task().cancelling():
            raise asyncio.CancelledError()
        if self._evidence_selector is not None and receipt["status"] == "prepared":
            text, evidence_answer = await self._construct_answer(messages[-1]["content"], request, receipt)
            await self._emit_final(messages, text, on_text)
            return text, receipt, None, evidence_answer
        if self._evidence_answer_policy == "required":
            if receipt["status"] not in ("empty", "skipped"):
                raise _EvidenceAnswerFailure("source_validation_failed")
            await self._emit_final(messages, _ABSTENTION, on_text)
            return _ABSTENTION, receipt, None, {
                "status": "no_selection", "reason": "no_memory_evidence", "evidence_ids": [],
                "source_status": "none", "mode": "extractive", "verified_accuracy": False}
        buffered = self._answer_reviewer is not None or self._answer_requirements is not None
        draft = await self._generate(request, None if buffered else on_text)
        if self._answer_reviewer is None:
            if buffered:
                await self._emit_final(messages, draft, on_text)
            skipped = ({"status": "skipped", "reason": "no_memory_evidence", "verified_accuracy": False}
                       if self._evidence_selector is not None else None)
            return draft, receipt, None, skipped
        text, answer_review = await self._review_draft(messages[-1]["content"], draft, request, receipt)
        await self._emit_final(messages, text, on_text)
        return text, receipt, answer_review, None

    async def _execute_tools(self, messages, on_text):
        factory = self._tool_factory
        if factory is None:
            raise RuntimeError('tool model unavailable')
        try:
            model = factory()
            if not callable(getattr(model, 'complete', None)):
                raise ValueError()
        except Exception:
            raise RuntimeError('tool model unavailable') from None
        tools = ScopedMemoryTools(self._memory, self._space, scope=self._scope,
                                  exclude_session_id=self._session_id, enable_computation=self._tool_compute)
        result = await EvidenceToolLoop(model, tools, limits=self._tool_limits,
                                       initial_search=self._tool_initial_search).run(messages)
        receipt = tool_context_receipt(result, self._session_id)
        text, answer_review = result.text, None
        if self._answer_reviewer is not None:
            text, answer_review = await self._review_tool_draft(messages[-1]['content'], result)
        if len(text.encode('utf-8')) > self._tool_limits.max_reply_bytes:
            raise RuntimeError('tool reply exceeded the byte limit')
        await self._emit_final(messages, text, on_text)
        if not await result.validate():
            raise RuntimeError('tool evidence changed before capture')
        return text, receipt, answer_review, None

    async def _review_tool_draft(self, question: str, result: ToolLoopResult) -> tuple[str, dict[str, object]]:
        reviewer = self._answer_reviewer
        if reviewer is None:
            raise RuntimeError('answer review unavailable')
        try:
            reviewed = await review_tool_answer(reviewer, question, result, self._review_limits,
                requirements=self._answer_requirements)
            task = asyncio.current_task()
            if self._closed or (task is not None and task.cancelling()):
                raise asyncio.CancelledError()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            code = 'source_validation_timeout' if isinstance(error, TimeoutError) else 'source_validation_failed'
            failure: dict[str, object] = {'status':'unavailable', 'rounds':0, 'revised':False,
                'issue_codes':[], 'errors':[code], 'source_status':'unavailable', 'verified_accuracy':False}
            raise _AnswerReviewFailure('answer review evidence unavailable', failure) from error
        return self._accept_review(reviewed)

    async def _emit_final(self, messages, text, on_text):
        if self._answer_requirements is not None and not self._answer_requirements.accepts(text):
            raise RuntimeError('answer format violates configured requirements')
        if len(text.encode("utf-8")) > self._max_reply:
            raise RuntimeError("model reply exceeded the byte limit")
        if _bytes([*messages, {"role": "assistant", "content": text}]) > self._max_history:
            raise RuntimeError("model reply exceeded the conversation history byte limit")
        if self._closed or asyncio.current_task().cancelling():
            raise asyncio.CancelledError()
        if on_text is not None:
            self._cancel_reusable = False
            try:
                await on_text(text)
            except asyncio.CancelledError as exc:
                if asyncio.current_task().cancelling():
                    raise
                raise RuntimeError("public text observer failed") from exc
            except Exception as exc:
                raise RuntimeError("public text observer failed") from exc
            self._cancel_reusable = not self._closed

    async def _construct_answer(self, question: str, request: list[dict[str, object]],
                                context_receipt: ContextReceipt) -> tuple[str, dict[str, object]]:
        selector = self._evidence_selector
        if selector is None:
            raise _EvidenceAnswerFailure("invalid_selection")
        deadline = time.monotonic() + self._evidence_answer_timeout
        self._cancel_reusable = not self._closed
        try:
            async with asyncio.timeout_at(deadline):
                material = await prepare_review_evidence(self._memory, self._space, self._scope,
                    self._session_id, request, context_receipt)
            task = asyncio.current_task()
            if self._closed or (task is not None and task.cancelling()):
                raise asyncio.CancelledError()
            if time.monotonic() >= deadline:
                raise TimeoutError()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            reason = "source_validation_timeout" if isinstance(error, TimeoutError) else "source_validation_failed"
            raise _EvidenceAnswerFailure(reason) from error
        try:
            constructed = await construct_evidence_answer(selector, question, material,
                timeout_s=self._evidence_answer_timeout, max_answer_bytes=min(self._max_reply, 16000), deadline=deadline)
            if self._closed or (task is not None and task.cancelling()):
                raise asyncio.CancelledError()
            if time.monotonic() >= deadline:
                raise _EvidenceAnswerFailure("selection_timeout")
        except asyncio.CancelledError:
            raise
        except EvidenceAnswerError as error:
            raise _EvidenceAnswerFailure(error.reason) from error
        return constructed.answer, constructed.receipt

    async def _review_draft(self, question: str, draft: str, request: list[dict[str, object]],
                            context_receipt: ContextReceipt) -> tuple[str, dict[str, object]]:
        if context_receipt["status"] != "prepared":
            return draft, {"status": "skipped", "reason": "no_memory_evidence", "verified_accuracy": False}
        reviewer = self._answer_reviewer
        if reviewer is None:
            raise RuntimeError("answer review unavailable")
        deadline = time.monotonic() + self._review_limits.timeout_s
        try:
            # Proof preparation and review share one deadline. The controller
            # owns its remaining timer so its final source check can complete.
            async with asyncio.timeout_at(deadline):
                prepared = await prepare_review_evidence(self._memory, self._space, self._scope,
                    self._session_id, request, context_receipt)
            task = asyncio.current_task()
            if self._closed or (task is not None and task.cancelling()):
                raise asyncio.CancelledError()
            if time.monotonic() >= deadline:
                raise TimeoutError()
            reviewed = await review_answer(reviewer, question, draft, prepared.evidence, prepared.evidence_ids,
                limits=self._review_limits, validate_evidence=prepared.validate, deadline=deadline,
                requirements=self._answer_requirements)
            if self._closed or (task is not None and task.cancelling()):
                raise asyncio.CancelledError()
            if time.monotonic() >= deadline:
                raise TimeoutError()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            code = "source_validation_timeout" if isinstance(error, TimeoutError) else "source_validation_failed"
            failure: dict[str, object] = {"status": "unavailable", "rounds": 0, "revised": False,
                "issue_codes": [], "errors": [code], "source_status": "unavailable", "verified_accuracy": False}
            raise _AnswerReviewFailure("answer review evidence unavailable", failure) from error
        return self._accept_review(reviewed)

    def _accept_review(self, reviewed: ReviewedAnswer) -> tuple[str, dict[str, object]]:
        receipt: dict[str, object] = reviewed.receipt.model_dump(mode="json")
        if reviewed.receipt.source_status != "retained":
            raise _AnswerReviewFailure("answer review evidence is stale or unavailable", receipt)
        if reviewed.receipt.format_status == 'rejected':
            raise _AnswerReviewFailure('answer format violates configured requirements', receipt)
        if self._review_policy == "require_supported" and reviewed.receipt.status != "supported":
            raise _AnswerReviewFailure("answer review did not support the reply", receipt)
        return reviewed.answer, receipt

    async def _generate(self, request, on_text):
        factory = self._factory
        if factory is None:
            raise RuntimeError('text model unavailable')
        model = factory()
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
            return text
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
