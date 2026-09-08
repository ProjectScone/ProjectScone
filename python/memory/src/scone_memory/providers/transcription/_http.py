"""Words out of audio, one request per utterance.

A streaming recogniser keeps a socket open and reports partial words as
they arrive. This keeps nothing open. It listens with the gate the rest
of Scone listens with, buffers what was said, and asks once when the
speaker stops.

That trades the first word's latency for a provider needing no websocket
and no partial-result protocol, which is the honest shape for an HTTP
transcription endpoint. Pretending otherwise, by emitting invented
partial transcripts, would give the turn-taking above it something to
act on that is not true.

Nothing is retried. An uncertain request may already have been billed
and may already have been transcribed, and asking twice is the one way
to be sure a caller is charged twice.
"""

from __future__ import annotations

import io
import json
import math
import wave
from collections.abc import AsyncIterator

import httpx

from ...audio.gate import VoiceGate
from ...realtime.audio import AudioChunk, SpeechActivityDetector, SpeechStarted, Transcript, check_audio


class TranscriptionProviderError(RuntimeError):
    """Safe stage error: never the key, the audio, or the remote body."""


class BufferedTranscription:
    """One utterance at a time, over a reusable owned client.

    Construction performs no network work. The HTTP timeout bounds
    inactivity; the session above owns the deadline for a whole turn."""

    provider: str = ""
    requires_api_key = True

    def __init__(self, *, api_key: str | None = None, model: str, rate: int = 16000, timeout: float = 30,
                 max_audio_bytes: int = 8_000_000, max_transcript_bytes: int = 64000,
                 detector: SpeechActivityDetector | None = None,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        if (api_key is None and self.requires_api_key) or (api_key is not None and (
                not isinstance(api_key, str) or not api_key or len(api_key) > 4096
                or any(not 33 <= ord(c) <= 126 for c in api_key))):
            raise ValueError("api_key must be nonempty printable ASCII without spaces")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("a model is required")
        if type(rate) is not int or not 8000 <= rate <= 48000:
            raise ValueError("rate must be an integer in 8000..48000")
        if type(max_audio_bytes) is not int or not 1024 <= max_audio_bytes <= 64_000_000:
            raise ValueError("max_audio_bytes must be in 1024..64000000")
        if type(max_transcript_bytes) is not int or not 1024 <= max_transcript_bytes <= 1_000_000:
            raise ValueError("max_transcript_bytes must be in 1024..1000000")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self._api_key = api_key or ''
        self._model = model
        self._rate = rate
        self._max_audio = max_audio_bytes
        self._max_transcript = max_transcript_bytes
        self._detector = detector if detector is not None else VoiceGate(rate=rate)
        self._timeout, self._transport = timeout, transport
        self._client: httpx.AsyncClient | None = None
        self._response: httpx.Response | None = None
        self._closed = self._busy = False

    # -- the shape a provider fills in ------------------------------------

    def _request(self) -> tuple[str, dict[str, str], dict[str, str]]:
        """Where to ask, what to say who we are, and the fields beside the
        audio. Subclasses give the provider's own contract."""
        raise NotImplementedError

    def _text(self, payload: object) -> str:
        """The transcript out of the provider's answer, or "" when there
        is none to be had."""
        raise NotImplementedError

    # -- listening ---------------------------------------------------------

    def transcribe(self, audio: AsyncIterator[AudioChunk]) -> AsyncIterator[SpeechStarted | Transcript]:
        return self._listen(audio)

    async def _listen(self, audio: AsyncIterator[AudioChunk]) -> AsyncIterator[SpeechStarted | Transcript]:
        if self._closed:
            raise TranscriptionProviderError('Transcription provider is closed')
        if self._busy:
            raise TranscriptionProviderError('Transcription provider already has an active listener')
        self._busy = True
        try:
            async for event in self._listen_audio(audio):
                yield event
        finally:
            self._busy = False

    async def _listen_audio(self, audio: AsyncIterator[AudioChunk]) -> AsyncIterator[SpeechStarted | Transcript]:
        said: list[bytes] = []
        total = 0
        speaking = False
        async for chunk in audio:
            if self._closed:
                raise TranscriptionProviderError('Transcription provider is closed')
            check_audio(chunk, self._max_audio)
            if chunk.sample_rate != self._rate or chunk.channels != 1:
                raise TranscriptionProviderError(
                    f"this recogniser requires mono PCM at {self._rate} Hz")
            heard = await self._detector.detect(chunk)
            if type(heard) is not bool:
                raise TranscriptionProviderError('Speech activity detector must return bool')
            if heard:
                said.append(chunk.pcm)
                total += len(chunk.pcm)
                if total > self._max_audio:
                    # Refused here rather than uploaded and refused there:
                    # an oversized request may be billed before it fails.
                    said.clear()
                    raise TranscriptionProviderError(
                        f"the speaker went on longer than {self._max_audio} bytes of audio allows")
                if not speaking:
                    speaking = True
                    yield SpeechStarted()
            elif speaking:
                speaking = False
                text = await self._ask(b"".join(said))
                said.clear()
                total = 0
                yield Transcript(text=text, final=True)
        if said:
            # The audio ended mid-sentence. Somebody still said something.
            yield Transcript(text=await self._ask(b"".join(said)), final=True)

    async def _ask(self, pcm: bytes) -> str:
        if self._closed:
            raise TranscriptionProviderError('Transcription provider is closed')
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout, transport=self._transport,
                                            follow_redirects=False, trust_env=False)
        url, headers, fields = self._request()
        headers = {**headers, 'Accept-Encoding': 'identity'}
        try:
            async with self._client.stream('POST',
                url, headers=headers, data=fields,
                files={"file": ("speech.wav", self._wav(pcm), "audio/wav")}) as answer:
                self._response = answer
                if answer.status_code != 200:
                    raise TranscriptionProviderError(
                        f"{self.provider} transcription refused with {answer.status_code}")
                if answer.headers.get('content-encoding', 'identity').lower() != 'identity':
                    raise TranscriptionProviderError('Encoded transcription response is not supported')
                body = bytearray()
                async for part in answer.aiter_bytes():
                    if self._closed:
                        raise TranscriptionProviderError('Transcription provider is closed')
                    if len(body) + len(part) > self._max_transcript:
                        raise TranscriptionProviderError('Transcription response exceeds byte limit')
                    body.extend(part)
                if self._closed:
                    raise TranscriptionProviderError('Transcription provider is closed')
        except httpx.HTTPError as exc:
            raise TranscriptionProviderError(
                f"{self.provider} transcription did not answer: {type(exc).__name__}") from None
        finally:
            self._response = None
        try:
            payload = json.loads(body)
        except ValueError:
            raise TranscriptionProviderError(f"{self.provider} sent no transcript") from None
        text = self._text(payload)
        if not isinstance(text, str) or not text.strip():
            # Empty would be indistinguishable from the person saying
            # nothing, and the conversation would carry on around a hole.
            raise TranscriptionProviderError(f"{self.provider} sent no transcript")
        return text.strip()

    def _wav(self, pcm: bytes) -> bytes:
        """The audio with its rate attached. Sending bare samples and
        letting the far end assume a rate is how a voice comes back an
        octave out and a transcript comes back as nonsense."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as sound:
            sound.setnchannels(1)
            sound.setsampwidth(2)
            sound.setframerate(self._rate)
            sound.writeframes(pcm)
        return buffer.getvalue()

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
            raise TranscriptionProviderError('Transcription transport cleanup failed') from None
        finally:
            await self._detector.aclose()
