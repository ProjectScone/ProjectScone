"""Chat models behind one protocol, so distillation never names a vendor.

The engine works without a model: episodic recall is untouched, only
fact distillation waits. When a model is configured it sits behind
``ChatModel``; ``OpenAICompatibleChat`` speaks to OpenAI, Ollama, vLLM
and anything else exposing ``/chat/completions``, and ``FakeChat``
replays scripted replies so the distiller can be tested without a
network.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from typing import TYPE_CHECKING, Optional, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    import httpx

from ..core.errors import SconeError

#: Ceiling for one model call. A hung provider must become a typed
#: error, never a stuck process.
DEFAULT_TIMEOUT = 180.0
logger = logging.getLogger(__name__)


def _log_finished(
    call_id: str, model: str, mode: str, started: float, outcome: str,
    exception_type: Optional[str] = None, first_token_ms: Optional[float] = None,
) -> None:
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    logger.log(
        logging.INFO if outcome == "completed" else logging.WARNING,
        "model_call.finished call_id=%s mode=%s model=%r outcome=%s elapsed_ms=%.3f "
        "first_token_ms=%s exception_type=%s",
        call_id, mode, model, outcome, elapsed_ms, first_token_ms, exception_type,
        extra={"event": "model_call.finished", "call_id": call_id, "model_name": model,
               "mode": mode, "outcome": outcome,
               "elapsed_ms": elapsed_ms, "first_token_ms": first_token_ms,
               "exception_type": exception_type},
    )


class ChatError(SconeError):
    """The model could not be reached or did not return a message."""


@runtime_checkable
class ChatModel(Protocol):
    async def complete(self, system: str, user: str) -> str:
        """One turn: a system prompt and a user message in, the
        assistant's text out."""
        ...


@runtime_checkable
class StructuredChatModel(Protocol):
    """Optional provider capability; ordinary ChatModel adapters need not implement it."""

    async def complete_structured(self, system: str, user: str, schema: dict[str, object]) -> str:
        """Return a complete JSON response constrained by ``schema``."""
        ...


class OpenAICompatibleChat:
    """Any OpenAI-compatible ``/chat/completions`` endpoint over httpx.

    ``temperature`` defaults to zero so the same text distils to the
    same facts twice; servers default to sampling otherwise. ``think``
    is only sent when set, because Ollama honours it for reasoning
    models while real OpenAI endpoints reject unknown fields.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        think: Optional[bool] = None,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
        trust_env: bool = True,
    ) -> None:
        try:
            import httpx
        except ImportError as e:  # pragma: no cover
            raise ImportError("OpenAICompatibleChat needs httpx: pip install 'scone-memory[remote-embed]'") from e
        self._httpx = httpx
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.think = think
        self.timeout = timeout
        self.trust_env = trust_env
        #: An ``httpx.AsyncBaseTransport``; tests hand in a MockTransport.
        self.transport = transport

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    def _body(self, system: str, user: str) -> dict[str, object]:
        body: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
        }
        if self.think is not None:
            body["think"] = self.think
        return body

    async def complete(self, system: str, user: str) -> str:
        return await self._request(self._body(system, user))

    async def complete_structured(
        self, system: str, user: str, schema: dict[str, object], *, max_tokens: int = 2048,
    ) -> str:
        """Constrain extraction output without changing ordinary conversation replies.

        A token ceiling bounds runaway generation. A truncated response is
        an error even if its prefix happens to contain valid JSON.
        """
        body = self._body(system, user)
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "extracted_facts", "strict": True, "schema": schema},
        }
        body["max_tokens"] = max_tokens
        return await self._request(body, require_complete=True)

    async def _request(self, body: dict[str, object], *, require_complete: bool = False) -> str:
        started, call_id = time.perf_counter(), uuid.uuid4().hex[:12]
        mode = "structured" if require_complete else "chat"
        outcome, exception_type = "cancelled", None
        logger.info("model_call.started call_id=%s mode=%s model=%r timeout_s=%s",
                    call_id, mode, self.model, self.timeout,
                    extra={"event": "model_call.started", "call_id": call_id,
                           "model_name": self.model, "mode": mode, "timeout_s": self.timeout})
        try:
            content = await self._send(body, require_complete=require_complete)
            outcome = "completed"
            return content
        except asyncio.CancelledError:
            exception_type = "CancelledError"
            raise
        except Exception as error:
            outcome, exception_type = "failed", type(error.__cause__ or error).__name__
            raise
        finally:
            _log_finished(call_id, self.model, mode, started, outcome, exception_type)

    async def _send(self, body: dict[str, object], *, require_complete: bool = False) -> str:
        httpx = self._httpx
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport,
                                         trust_env=self.trust_env) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=body,
                    headers=self._headers(),
                )
        except httpx.HTTPError as e:
            raise ChatError(f"chat server unreachable: {type(e).__name__}: {e}") from e
        if response.status_code >= 400:
            raise ChatError(f"chat server returned {response.status_code}: {response.text[:200]}")
        content = _content_of(response)
        if require_complete:
            finish_reason = response.json()["choices"][0].get("finish_reason")
            if finish_reason != "stop":
                raise ChatError(f"structured chat response did not complete: {finish_reason!r}")
        return content


class OpenAICompatibleTextModel:
    """The same endpoint as a native ``TextModel`` for conversations: the
    reply streams as public deltas and ends with an explicit completion.

    A server that ignores ``stream`` still answers with one message, which
    is delivered as one delta. A stream that ends without a completion, an
    error status, or a line that is not a chunk is a ChatError, so a turn
    fails visibly rather than completing on half a reply. One instance
    serves one turn; sessions call the factory per turn and aclose after.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        think: Optional[bool] = None,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
        trust_env: bool = True,
        max_output_tokens: int | None = None,
    ) -> None:
        if max_output_tokens is not None and (type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 32768):
            raise ValueError("max_output_tokens must be an integer from 1 to 32768")
        self._max_output_tokens = max_output_tokens
        try:
            import httpx
        except ImportError as e:  # pragma: no cover
            raise ImportError("OpenAICompatibleTextModel needs httpx: pip install 'scone-memory[remote-embed]'") from e
        self._chat = OpenAICompatibleChat(base_url, model, api_key=api_key, temperature=temperature,
                                          think=think, timeout=timeout, transport=transport,
                                          trust_env=trust_env)
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport, trust_env=trust_env)
        self._httpx = httpx

    async def respond(self, messages: list[dict[str, str]]):
        from ..realtime.events import ReplyCompleted, TextDelta

        started, call_id = time.perf_counter(), uuid.uuid4().hex[:12]
        first_token_ms: Optional[float] = None
        outcome, exception_type = "cancelled", None
        logger.info("model_call.started call_id=%s mode=stream model=%r timeout_s=%s",
                    call_id, self._chat.model, self._chat.timeout,
                    extra={"event": "model_call.started", "call_id": call_id,
                           "model_name": self._chat.model, "mode": "stream", "timeout_s": self._chat.timeout})
        events = self._respond(messages)
        try:
            async for event in events:
                if isinstance(event, TextDelta) and first_token_ms is None:
                    first_token_ms = round((time.perf_counter() - started) * 1000, 3)
                    logger.info("model_call.first_token call_id=%s model=%r first_token_ms=%.3f",
                                call_id, self._chat.model, first_token_ms,
                                extra={"event": "model_call.first_token", "call_id": call_id,
                                       "model_name": self._chat.model, "mode": "stream",
                                       "first_token_ms": first_token_ms})
                if isinstance(event, ReplyCompleted):
                    outcome = "completed"
                yield event
        except (asyncio.CancelledError, GeneratorExit) as error:
            exception_type = type(error).__name__
            raise
        except Exception as error:
            outcome, exception_type = "failed", type(error.__cause__ or error).__name__
            raise
        finally:
            try:
                await events.aclose()
            finally:
                _log_finished(call_id, self._chat.model, "stream", started, outcome,
                              exception_type, first_token_ms)

    async def _respond(self, messages: list[dict[str, str]]):
        from ..realtime.events import ReplyCompleted, TextDelta

        body = {"model": self._chat.model, "messages": list(messages), "temperature": self._chat.temperature, "stream": True}
        if self._max_output_tokens is not None:
            body["max_tokens"] = self._max_output_tokens
        if self._chat.think is not None:
            body["think"] = self._chat.think
        try:
            async with self._client.stream("POST", f"{self._chat.base_url}/chat/completions", json=body,
                                           headers=self._chat._headers()) as response:
                if response.status_code >= 400:
                    raise ChatError(f"chat server returned {response.status_code}")
                if not response.headers.get("content-type", "").startswith("text/event-stream"):
                    await response.aread()
                    text = _content_of(response)
                    finish = response.json()["choices"][0].get("finish_reason")
                    if finish not in (None, "stop"):
                        raise ChatError("chat response did not finish normally")
                    if text:
                        yield TextDelta(text)
                    yield ReplyCompleted()
                    return
                completed = False
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue  # comments and blank lines keep the stream alive
                    data = line[5:].strip()
                    if data == "[DONE]":
                        completed = True
                        break
                    try:
                        choice = json.loads(data)["choices"][0]
                        chunk_text = (choice.get("delta") or {}).get("content")
                    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
                        raise ChatError(f"malformed chat stream chunk: {type(e).__name__}") from e
                    if chunk_text:
                        yield TextDelta(chunk_text)
                    finish = choice.get("finish_reason")
                    if finish is not None and finish != "stop":
                        raise ChatError("chat response did not finish normally")
                    if finish == "stop":
                        completed = True
                        break
        except self._httpx.HTTPError as e:
            raise ChatError(f"chat server unreachable: {type(e).__name__}") from e
        if not completed:
            raise ChatError("chat stream ended without completing the reply")
        yield ReplyCompleted()

    async def aclose(self) -> None:
        await self._client.aclose()


def _content_of(response: object) -> str:
    try:
        content = response.json()["choices"][0]["message"]["content"]  # type: ignore[attr-defined]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise ChatError(f"no content in chat response: {e}") from e
    if not isinstance(content, str):
        raise ChatError(f"chat response content is {type(content).__name__}, not text")
    return content


class FakeChat:
    """Scripted replies for tests, consumed in order; every call is kept.

    A reply may be an exception, which is raised when its turn comes.
    Running past the script raises ``ChatError`` so an unexpected extra
    call is loud rather than answered with a stale reply.
    """

    def __init__(self, replies: Sequence[str | Exception] = ()) -> None:
        self.replies: list[str | Exception] = list(replies)
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.replies:
            raise ChatError(f"FakeChat has no reply scripted for call {len(self.calls)}")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
