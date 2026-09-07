"""Shared bounded PCM response handling; provider request shapes stay separate."""

from collections.abc import AsyncIterator
import math
import re

import httpx

from ...realtime.audio import AudioChunk
from ...realtime.persona import VoiceChoice


class SpeechProviderError(RuntimeError):
    """Safe stage error: never includes credentials, source text or remote bodies."""


class PCMSpeech:
    """One active utterance per instance, with a reusable owned HTTP client.

    The caller closes each iterator (including when interrupted), then aclose()
    closes the adapter at session end. Construction performs no network I/O.
    HTTP timeouts bound I/O inactivity; the session owns the total turn deadline.
    There are no retries: an uncertain request may already have been billed.
    """

    provider: str

    def __init__(self, *, api_key: str, model: str, voice: str, sample_rate: int = 24000,
                 timeout: float = 30, chunk_bytes: int = 4096, max_audio_bytes: int = 24_000_000,
                 transport: httpx.AsyncBaseTransport | None = None):
        if not isinstance(api_key, str) or not api_key or len(api_key) > 4096 or any(not 33 <= ord(c) <= 126 for c in api_key):
            raise ValueError('api_key must be nonempty printable ASCII without spaces')
        self._choice = VoiceChoice(provider=self.provider, model=model, voice=voice)
        if not re.fullmatch(r'[A-Za-z0-9_-]+', voice):
            raise ValueError('voice must be an opaque provider ID')
        if type(sample_rate) is not int or sample_rate not in (8000, 16000, 22050, 24000, 44100, 48000):
            raise ValueError('unsupported PCM sample rate')
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        if type(chunk_bytes) is not int or not 512 <= chunk_bytes <= 64000 or chunk_bytes % 2:
            raise ValueError('chunk_bytes must be an even integer in 512..64000')
        if type(max_audio_bytes) is not int or not 2 <= max_audio_bytes <= 64_000_000:
            raise ValueError('max_audio_bytes must be an integer in 2..64000000')
        self._api_key, self._sample_rate = api_key, sample_rate
        self._timeout, self._chunk_bytes, self._max_audio_bytes = timeout, chunk_bytes, max_audio_bytes
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._response: httpx.Response | None = None
        self._closed = self._busy = False

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        if self._closed:
            raise SpeechProviderError('Speech provider is closed')
        if self._busy:
            raise SpeechProviderError('Speech provider already has an active utterance')
        if not isinstance(text, str) or not text.strip() or len(text.encode('utf-8')) > 32000:
            raise ValueError('speech text must be nonblank and at most 32000 UTF-8 bytes')
        self._busy = True
        try:
            if self._client is None:
                self._client = httpx.AsyncClient(transport=self._transport, timeout=self._timeout,
                                                follow_redirects=False, trust_env=False)
            url, headers, body = self._request(text)
            headers = {**headers, 'Accept-Encoding': 'identity'}
            async with self._client.stream('POST', url, headers=headers, json=body) as response:
                self._response = response
                if response.status_code != 200:
                    raise SpeechProviderError(f'{self.provider} speech request failed (HTTP {response.status_code})')
                mime = response.headers.get('content-type', '').split(';', 1)[0].strip().lower()
                if mime not in ('application/octet-stream', 'audio/pcm', 'audio/raw'):
                    raise SpeechProviderError('Speech response is not supported raw PCM')
                if response.headers.get('content-encoding', 'identity').lower() != 'identity':
                    raise SpeechProviderError('Encoded speech response is not supported raw PCM')
                pending = b''
                total = 0
                async for part in response.aiter_raw():
                    if self._closed:
                        raise SpeechProviderError('Speech provider is closed')
                    total += len(part)
                    if total > self._max_audio_bytes:
                        raise SpeechProviderError('Speech response exceeds audio byte limit')
                    data = pending + part
                    complete = len(data) - len(data) % 2
                    for offset in range(0, complete, self._chunk_bytes):
                        if self._closed:
                            raise SpeechProviderError('Speech provider is closed')
                        yield AudioChunk(data[offset:min(offset + self._chunk_bytes, complete)], self._sample_rate)
                    pending = data[complete:]
                if self._closed:
                    raise SpeechProviderError('Speech provider is closed')
                if pending or not total:
                    raise SpeechProviderError('Speech response contains empty or incomplete PCM')
        except httpx.HTTPError:
            raise SpeechProviderError(f'{self.provider} speech transport failed') from None
        finally:
            self._response = None
            self._busy = False

    async def aclose(self) -> None:
        self._closed = True
        try:
            try:
                if self._response is not None:
                    await self._response.aclose()
            finally:
                if self._client is not None:
                    await self._client.aclose()
        except httpx.HTTPError:
            raise SpeechProviderError(f'{self.provider} speech transport cleanup failed') from None
