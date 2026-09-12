"""Timestamped document transcription through an explicitly selected local service."""
from __future__ import annotations

import asyncio
from ipaddress import ip_address
import json
import math
import struct
from time import monotonic
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from ...core.errors import InvalidInput
from ...ingestion.formats.media import TranscriptionSegment
from ..self_hosted import validate_self_hosted_endpoint, validate_self_hosted_identifier

_MAX_AUDIO_BYTES = 19_200_044


def _remaining(deadline: float) -> float:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise InvalidInput('local document transcription exceeded its time limit')
    return remaining


def _duration(audio: bytes) -> float:
    if not isinstance(audio, bytes) or not 46 <= len(audio) <= _MAX_AUDIO_BYTES:
        raise InvalidInput('document transcription requires a bounded mono 16 kHz PCM WAV')
    header = struct.unpack('<4sI4s4sIHHIIHH4sI', audio[:44])
    expected = (b'RIFF', len(audio) - 8, b'WAVE', b'fmt ', 16, 1, 1, 16000,
                32000, 2, 16, b'data', len(audio) - 44)
    if header != expected or (len(audio) - 44) % 2:
        raise InvalidInput('document transcription requires an exact mono 16 kHz PCM WAV')
    return (len(audio) - 44) / 32000


class LocalDocumentTranscriber:
    """One multipart request for observed segment timestamps from a local model.

    The selected loopback service must implement audio/transcriptions with
    response_format=verbose_json and timestamp_granularities[]=segment. Text-only
    replies are refused. Construction performs no I/O; each call owns and closes
    its HTTP client, so cancellation cannot leave a persistent response behind.
    The host owns the service/model lifecycle. No retries or fallbacks occur.
    allow_empty accepts an explicit empty text and segment list for silent
    windows; text without observed timestamps remains invalid.
    """
    def __init__(self, *, base_url: str, model: str, api_key: str | None = None,
                 timeout: float = 120, max_response_bytes: int = 4_000_000,
                 max_segments: int = 10000, allow_empty: bool = False,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        endpoint = validate_self_hosted_endpoint(base_url)
        host = urlsplit(endpoint).hostname
        try:
            local = ip_address(host or '').is_loopback
        except ValueError:
            local = host == 'localhost'
        if not local:
            raise ValueError('document transcription requires an explicit loopback service')
        self._model = validate_self_hosted_identifier(model)
        try:
            endpoint.encode('utf-8')
            self._model.encode('utf-8')
        except UnicodeError:
            raise ValueError('local transcription endpoint and model must be valid UTF-8') from None
        if api_key is not None and (not isinstance(api_key, str) or not api_key
                or len(api_key) > 4096 or any(not 33 <= ord(c) <= 126 for c in api_key)):
            raise ValueError('api_key must be nonempty printable ASCII without spaces')
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not 0 < timeout <= 600 or not math.isfinite(timeout)):
            raise ValueError('transcription timeout must be finite and in (0, 600]')
        if type(max_response_bytes) is not int or not 1024 <= max_response_bytes <= 12_000_000:
            raise ValueError('transcription response limit must be in 1024..12000000')
        if type(max_segments) is not int or not 1 <= max_segments <= 10000:
            raise ValueError('transcription segment limit must be in 1..10000')
        if type(allow_empty) is not bool:
            raise ValueError('allow_empty must be a boolean')
        self._allow_empty = allow_empty
        self._url, self._key = endpoint + 'audio/transcriptions', api_key
        self._timeout, self._max_response = timeout, max_response_bytes
        self._max_segments, self._transport = max_segments, transport

    async def transcribe(self, audio_wav: bytes) -> tuple[TranscriptionSegment, ...]:
        deadline = monotonic() + self._timeout
        duration = _duration(audio_wav)
        try:
            timeout = _remaining(deadline)
            body = await asyncio.wait_for(self._request(audio_wav), timeout)
        except TimeoutError:
            raise InvalidInput('local document transcription exceeded its time limit') from None
        except httpx.HTTPError:
            raise InvalidInput('local document transcription transport failed') from None
        result = self._segments(body, duration, deadline)
        _remaining(deadline)
        return result

    async def _request(self, audio: bytes) -> bytes:
        headers = {'Accept-Encoding': 'identity', 'Accept': 'application/json'}
        if self._key is not None:
            headers['Authorization'] = 'Bearer ' + self._key
        fields = {'model': self._model, 'response_format': 'verbose_json',
                  'timestamp_granularities[]': 'segment'}
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport,
                                      follow_redirects=False, trust_env=False) as client:
            async with client.stream('POST', self._url, headers=headers, data=fields,
                                     files={'file': ('document.wav', audio, 'audio/wav')}) as response:
                if response.status_code != 200:
                    raise InvalidInput('local document transcription service refused the request')
                if response.headers.get('content-encoding', 'identity').lower() != 'identity':
                    raise InvalidInput('encoded document transcription responses are unsupported')
                if response.headers.get('content-type', '').split(';')[0].strip().lower() != 'application/json':
                    raise InvalidInput('local document transcription requires a JSON response')
                length = response.headers.get('content-length')
                if length is not None and (not length.isascii() or not length.isdecimal()
                        or len(length) > 8 or int(length) > self._max_response):
                    raise InvalidInput('document transcription response exceeds its byte limit')
                body = bytearray()
                async for part in response.aiter_raw():
                    if len(body) + len(part) > self._max_response:
                        raise InvalidInput('document transcription response exceeds its byte limit')
                    body.extend(part)
                return bytes(body)

    def _segments(self, body: bytes, duration: float, deadline: float) -> tuple[TranscriptionSegment, ...]:
        _remaining(deadline)
        try:
            payload = json.loads(body)
        except (ValueError, RecursionError):
            raise InvalidInput('local document transcription returned invalid JSON') from None
        values = payload.get('segments') if isinstance(payload, dict) else None
        if self._allow_empty and values == [] and isinstance(payload, dict):
            text = payload.get('text')
            if isinstance(text, str) and not text.strip():
                return ()
        if not isinstance(values, list) or not 1 <= len(values) <= self._max_segments:
            raise InvalidInput('local document transcription requires bounded observed segments')
        segments: list[TranscriptionSegment] = []
        previous, size = 0.0, 0
        for value in values:
            _remaining(deadline)
            if not isinstance(value, dict):
                raise InvalidInput('local document transcription returned an invalid segment')
            try:
                segment = TranscriptionSegment.model_validate({
                    'text': value.get('text'), 'start_seconds': value.get('start'),
                    'end_seconds': value.get('end')})
                size += len(segment.text.encode('utf-8')) + (2 if segments else 0)
            except (ValidationError, UnicodeError):
                raise InvalidInput('local document transcription returned an invalid segment') from None
            if segment.start_seconds < previous or segment.end_seconds > duration + 1 / 16000:
                raise InvalidInput('local document transcription returned out-of-audio or unordered times')
            if size > 2_000_000:
                raise InvalidInput('document transcription exceeds its text byte limit')
            previous = segment.start_seconds
            segments.append(segment)
        return tuple(segments)
