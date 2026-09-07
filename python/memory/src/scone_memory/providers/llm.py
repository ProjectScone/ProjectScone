"""Chat models behind one protocol, so distillation never names a vendor.

The engine works without a model: episodic recall is untouched, only
fact distillation waits. When a model is configured it sits behind
``ChatModel``; ``OpenAICompatibleChat`` speaks to OpenAI, Ollama, vLLM
and anything else exposing ``/chat/completions``, and ``FakeChat``
replays scripted replies so the distiller can be tested without a
network.
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence, runtime_checkable

from ..core.errors import SconeError

#: Ceiling for one model call. A hung provider must become a typed
#: error, never a stuck process.
DEFAULT_TIMEOUT = 180.0


class ChatError(SconeError):
    """The model could not be reached or did not return a message."""


@runtime_checkable
class ChatModel(Protocol):
    async def complete(self, system: str, user: str) -> str:
        """One turn: a system prompt and a user message in, the
        assistant's text out."""
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
        transport: object | None = None,
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
        httpx = self._httpx
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=self._body(system, user),
                    headers=self._headers(),
                )
        except httpx.HTTPError as e:
            raise ChatError(f"chat server unreachable: {type(e).__name__}: {e}") from e
        if response.status_code >= 400:
            raise ChatError(f"chat server returned {response.status_code}: {response.text[:200]}")
        return _content_of(response)


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
