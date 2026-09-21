"""Native tool turns for explicitly configured self-hosted chat endpoints."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Literal, cast

from ..agents.evidence_loop import ToolCall, ToolStep
from ..agents.usage import ModelTokenUsage
from .llm import _thinking_options
from .self_hosted import validate_self_hosted_endpoint, validate_self_hosted_identifier
from .tool_diagnostics import ToolCallDiagnostics

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate key')
        result[key] = value
    return result


def _constant(value: str) -> object:
    raise ValueError('non-finite JSON')


def _float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError('non-finite JSON')
    return parsed


def _decode(raw: str | bytes) -> object:
    return json.loads(raw, object_pairs_hook=_object, parse_constant=_constant, parse_float=_float)


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError('invalid object')
    return cast(dict[str, object], value)


def _step(raw: bytes, enabled: bool) -> ToolStep:
    choices = _mapping(_decode(raw)).get('choices')
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError('invalid choices')
    choice = _mapping(choices[0])
    message = _mapping(choice.get('message'))
    if message.get('role') != 'assistant' or message.get('refusal') or message.get('function_call') is not None:
        raise ValueError('invalid reply')
    content = message.get('content')
    if content is not None and not isinstance(content, str):
        raise ValueError('invalid text')
    calls = message.get('tool_calls', [])
    if calls is None:
        calls = []
    if not isinstance(calls, list) or len(calls) > 8:
        raise ValueError('invalid calls')
    reason = choice.get('finish_reason')
    if reason != ('tool_calls' if calls else 'stop') or (calls and not enabled):
        raise ValueError('incomplete reply')
    parsed: list[ToolCall] = []
    for raw_call in calls:
        call = _mapping(raw_call)
        function = _mapping(call.get('function'))
        arguments = function.get('arguments')
        if call.get('type') != 'function' or not isinstance(arguments, str) or len(arguments.encode()) > 16000:
            raise ValueError('invalid call')
        parsed.append(ToolCall.model_validate({'id': call.get('id'), 'name': function.get('name'),
                                              'arguments': _mapping(_decode(arguments))}))
    if len({call.id for call in parsed}) != len(parsed):
        raise ValueError('duplicate call id')
    if not calls and (not content or not content.strip()):
        raise ValueError('empty reply')
    return ToolStep(content=content or '', calls=tuple(parsed))


class _StreamedReply:
    """The reply a chat-completions stream is assembling, chunk by chunk.

    Only content is public: it goes to the sink as it arrives. Tool-call
    arguments accumulate by index and are handed to the ordinary parser
    once the stream finishes, so acceptance is decided exactly as for a
    nonstreaming reply. A delta after the finish, more than one choice, a
    line that is not a chunk, or a stream that ends without a finish is a
    ValueError.
    """

    def __init__(self) -> None:
        self.content = ''
        self.refusal = ''
        self.finish: str | None = None
        self.calls: dict[int, dict[str, object]] = {}
        self.usage: ModelTokenUsage | None = None
        self.saw_stream = False
        self.done = False
        self._other = bytearray()

    async def take(self, line: str, sink: Callable[[str], Awaitable[None]]) -> None:
        if not line.startswith('data:'):
            if line and not line.startswith(':'):
                self._other.extend((line + '\n').encode())
            return
        payload = line[5:].strip()
        if self.done:
            raise ValueError('chunk after done')
        if payload == '[DONE]':
            self.done = True
            return
        self.saw_stream = True
        chunk = _mapping(_decode(payload))
        if 'usage' in chunk and chunk['usage'] is not None:
            self.usage = ModelTokenUsage.from_provider(chunk['usage'])
        choices = chunk.get('choices')
        if choices is None or (isinstance(choices, list) and not choices):
            return
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError('invalid choices')
        choice = _mapping(choices[0])
        delta = _mapping(choice.get('delta')) if choice.get('delta') is not None else {}
        content, calls = delta.get('content'), delta.get('tool_calls')
        if content is not None and not isinstance(content, str):
            raise ValueError('invalid text')
        if self.finish is not None and (content or calls):
            raise ValueError('delta after finish')
        refusal = delta.get('refusal')
        if isinstance(refusal, str):
            self.refusal += refusal
        if content:
            self.content += content
            await sink(content)
        if calls is not None:
            if not isinstance(calls, list) or len(calls) > 8:
                raise ValueError('invalid calls')
            for raw_call in calls:
                call = _mapping(raw_call)
                index = call.get('index')
                if type(index) is not int or not 0 <= index < 8:
                    raise ValueError('invalid call index')
                held = self.calls.setdefault(index, {'id': None, 'type': None, 'name': None, 'arguments': ''})
                for name in ('id', 'type'):
                    if call.get(name) is not None:
                        held[name] = call[name]
                function = call.get('function')
                if function is not None:
                    parts = _mapping(function)
                    if parts.get('name') is not None:
                        held['name'] = parts['name']
                    arguments = parts.get('arguments')
                    if arguments is not None:
                        if not isinstance(arguments, str):
                            raise ValueError('invalid call')
                        held['arguments'] = str(held['arguments']) + arguments
        reason = choice.get('finish_reason')
        if reason is not None:
            if not isinstance(reason, str) or self.finish is not None:
                raise ValueError('invalid finish')
            self.finish = reason

    def reply(self) -> bytes:
        """The equivalent nonstreaming body, for the parser that owns
        acceptance; or the body as sent, when the server never streamed."""
        if not self.saw_stream:
            return bytes(self._other)
        # A stream that never finished carries no finish reason, and the
        # parser refuses that exactly as it refuses a nonstreaming reply
        # without one; no second guard is needed here.
        message: dict[str, object] = {'role': 'assistant', 'content': self.content or None}
        if self.refusal:
            message['refusal'] = self.refusal
        if self.calls:
            message['tool_calls'] = [{'id': held['id'], 'type': held['type'] or 'function',
                                      'function': {'name': held['name'], 'arguments': held['arguments']}}
                                     for _, held in sorted(self.calls.items())]
        return json.dumps({'choices': [{'finish_reason': self.finish, 'message': message}]}).encode()


class SelfHostedToolChat:
    """One nonstreaming native tool turn, with bounded response bytes.

    Each request owns its client; proxies and redirects are disabled. No model
    discovery, downloads, fallback provider, or reasoning capture occurs.
    The surrounding EvidenceToolLoop owns aggregate turn budgets/publication.
    """

    _validate_endpoint = staticmethod(validate_self_hosted_endpoint)

    def __init__(self, endpoint: str, model: str, *, api_key: str | None = None,
                 timeout_s: float = 120.0, max_response_bytes: int = 128000,
                 max_tokens: int = 2048, transport: httpx.AsyncBaseTransport | None = None,
                 think: bool | None = None) -> None:
        self._endpoint = self._validate_endpoint(endpoint).rstrip('/') + '/chat/completions'
        self._model = validate_self_hosted_identifier(model)
        if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
                or not math.isfinite(timeout_s) or not 0.01 <= timeout_s <= 600):
            raise ValueError('invalid tool model timeout')
        if type(max_response_bytes) is not int or not 1024 <= max_response_bytes <= 512000:
            raise ValueError('invalid tool response budget')
        if type(max_tokens) is not int or not 1 <= max_tokens <= 8192:
            raise ValueError('invalid tool token budget')
        if think is not None and type(think) is not bool:
            raise ValueError('invalid tool thinking setting')
        if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()
                or len(api_key) > 8192 or any(ord(ch) < 32 or ord(ch) == 127 for ch in api_key)):
            raise ValueError('invalid tool model key')
        self._key, self._timeout, self._max_bytes = api_key, float(timeout_s), max_response_bytes
        self._max_tokens, self._transport = max_tokens, transport
        self._think = think

    async def complete(self, messages: list[dict[str, object]], tools: list[dict[str, object]], *,
                       on_public_text: Callable[[str], Awaitable[None]] | None = None) -> ToolStep:
        """One tool turn. With ``on_public_text``, the turn is requested as a
        stream and every content delta is handed to the sink as it arrives;
        the assembled reply then goes through the same parser as a
        nonstreaming one, so what streamed and what is accepted are one
        reply. Tool-call arguments accumulate silently and reasoning fields
        are never read. A sink that raises fails the turn."""
        body: dict[str, object] = {'model': self._model, 'messages': messages, 'stream': False,
                                  'temperature': 0, 'max_tokens': self._max_tokens,
                                  'tool_choice': 'auto' if tools else 'none'}
        if tools:
            body['tools'] = tools
        return await self._request(body, lambda raw: _step(raw, bool(tools)), on_public_text=on_public_text)

    async def _request(self, body: dict[str, object], parse: Callable[[bytes], ToolStep], *,
                       protocol: Literal['native', 'structured_action', 'structured_answer'] = 'native',
                       on_public_text: Callable[[str], Awaitable[None]] | None = None) -> ToolStep:
        import httpx

        body = {**body, **_thinking_options(self._think)}
        if on_public_text is not None:
            body = {**body, 'stream': True, 'stream_options': {'include_usage': True}}
        encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        if len(encoded) > 1100000:
            raise ValueError('tool model request byte limit')
        headers = {'Content-Type': 'application/json'}
        if self._key:
            headers['Authorization'] = f'Bearer {self._key}'
        started = time.monotonic()
        outcome = 'cancelled'
        diagnostics = ToolCallDiagnostics(protocol, started, len(encoded), self._max_tokens, self._timeout)
        diagnostics.log_started(logger)

        async def request() -> ToolStep:
            async with httpx.AsyncClient(timeout=self._timeout, trust_env=False, follow_redirects=False,
                                        transport=self._transport) as client:
                async with client.stream('POST', self._endpoint, content=encoded, headers=headers) as response:
                    diagnostics.phase = 'response'
                    diagnostics.http_status = response.status_code
                    diagnostics.headers_ms = (time.monotonic() - started) * 1000
                    response.raise_for_status()
                    raw = bytearray()
                    streamed: _StreamedReply | None = None
                    if on_public_text is None:
                        async for part in response.aiter_bytes():
                            diagnostics.response_bytes += len(part)
                            if len(raw) + len(part) > self._max_bytes:
                                diagnostics.failure_kind = 'response_bytes'
                                raise ValueError('tool response byte limit')
                            raw.extend(part)
                    else:
                        streamed = _StreamedReply()
                        async for line in response.aiter_lines():
                            diagnostics.response_bytes += len(line.encode()) + 1
                            if diagnostics.response_bytes > self._max_bytes:
                                diagnostics.failure_kind = 'response_bytes'
                                raise ValueError('tool response byte limit')
                            try:
                                await streamed.take(line, on_public_text)
                            except (ValueError, TypeError, RecursionError):
                                diagnostics.failure_kind = 'invalid_response'
                                raise
                    diagnostics.phase = 'parse'
                    response_body = bytes(raw) if streamed is None else streamed.reply()
                    usage = ModelTokenUsage()
                    try:
                        metadata = _decode(response_body)
                    except (ValueError, TypeError, RecursionError):
                        pass  # The existing parser still owns response acceptance.
                    else:
                        diagnostics.observe(metadata)
                        if isinstance(metadata, dict):
                            usage = ModelTokenUsage.from_provider(metadata.get('usage'))
                    result = parse(response_body)
                    if streamed is not None and streamed.usage is not None:
                        usage = streamed.usage
                    if streamed is not None and on_public_text is not None and not streamed.saw_stream and result.content:
                        # A server that ignored `stream` answered with one
                        # message; the reader still gets it, as one delta.
                        await on_public_text(result.content)
                    result = result.model_copy(update={'usage': usage})
                    diagnostics.phase = 'cleanup'
            return result

        try:
            async with asyncio.timeout(self._timeout):
                result = await asyncio.create_task(request())
                active = asyncio.current_task()
                if active is not None and active.cancelling():
                    raise asyncio.CancelledError()
                if time.monotonic() - started >= self._timeout:
                    raise TimeoutError()
                outcome = 'completed'
                diagnostics.phase = 'completed'
                return result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            outcome = 'failed'
            if diagnostics.failure_kind is None:
                if isinstance(exc, httpx.TimeoutException):
                    diagnostics.failure_kind = 'transport_timeout'
                elif isinstance(exc, TimeoutError):
                    diagnostics.failure_kind = 'request_timeout'
                elif isinstance(exc, httpx.HTTPStatusError):
                    diagnostics.failure_kind = 'http_error'
                elif isinstance(exc, httpx.RequestError):
                    diagnostics.failure_kind = 'transport_error'
                elif diagnostics.phase == 'parse' and isinstance(exc, (ValueError, TypeError, RecursionError)):
                    diagnostics.failure_kind = 'invalid_response'
                else:
                    diagnostics.failure_kind = 'internal_error'
            raise RuntimeError('tool model unavailable') from None
        finally:
            # No prompts, responses, endpoint credentials, or exception text.
            diagnostics.log_finished(logger, outcome)
