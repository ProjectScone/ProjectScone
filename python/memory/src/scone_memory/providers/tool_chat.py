"""Native tool turns for explicitly configured self-hosted chat endpoints."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, cast

from ..agents.evidence_loop import ToolCall, ToolStep
from .self_hosted import validate_self_hosted_endpoint, validate_self_hosted_identifier

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


class SelfHostedToolChat:
    """One nonstreaming native tool turn, with bounded response bytes.

    Each request owns its client; proxies and redirects are disabled. No model
    discovery, downloads, fallback provider, or reasoning capture occurs.
    The surrounding EvidenceToolLoop owns aggregate turn budgets/publication.
    """

    def __init__(self, endpoint: str, model: str, *, api_key: str | None = None,
                 timeout_s: float = 120.0, max_response_bytes: int = 128000,
                 max_tokens: int = 2048, transport: httpx.AsyncBaseTransport | None = None,
                 think: bool | None = None) -> None:
        self._endpoint = validate_self_hosted_endpoint(endpoint).rstrip('/') + '/chat/completions'
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

    async def complete(self, messages: list[dict[str, object]], tools: list[dict[str, object]]) -> ToolStep:
        body: dict[str, object] = {'model': self._model, 'messages': messages, 'stream': False,
                                  'temperature': 0, 'max_tokens': self._max_tokens,
                                  'tool_choice': 'auto' if tools else 'none'}
        if tools:
            body['tools'] = tools
        return await self._request(body, lambda raw: _step(raw, bool(tools)))

    async def _request(self, body: dict[str, object], parse: Callable[[bytes], ToolStep], *,
                       protocol: Literal['native', 'structured_action'] = 'native') -> ToolStep:
        import httpx

        if self._think is not None:
            body = {**body, 'think':self._think}
        encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        if len(encoded) > 1100000:
            raise ValueError('tool model request byte limit')
        headers = {'Content-Type': 'application/json'}
        if self._key:
            headers['Authorization'] = f'Bearer {self._key}'
        started = time.monotonic()
        outcome = 'cancelled'

        async def request() -> ToolStep:
            async with httpx.AsyncClient(timeout=self._timeout, trust_env=False, follow_redirects=False,
                                        transport=self._transport) as client:
                async with client.stream('POST', self._endpoint, content=encoded, headers=headers) as response:
                    response.raise_for_status()
                    raw = bytearray()
                    async for part in response.aiter_bytes():
                        if len(raw) + len(part) > self._max_bytes:
                            raise ValueError('tool response byte limit')
                        raw.extend(part)
                    result = parse(bytes(raw))
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
                return result
        except asyncio.CancelledError:
            raise
        except Exception:
            outcome = 'failed'
            raise RuntimeError('tool model unavailable') from None
        finally:
            # No prompts, responses, endpoint credentials, or exception text.
            logger.info('tool_model.finished protocol=%s outcome=%s elapsed_ms=%.3f', protocol, outcome,
                        (time.monotonic() - started) * 1000)
