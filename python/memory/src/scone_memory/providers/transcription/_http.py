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
import wave
from collections.abc import AsyncIterator
from typing import Optional

import httpx

from ...audio.gate import VoiceGate
from ...realtime.audio import AudioChunk, SpeechStarted, Transcript, check_audio


class TranscriptionProviderError(RuntimeError):
    """Safe stage error: never the key, the audio, or the remote body."""


class BufferedTranscription:
    """One utterance at a time, over a reusable owned client.

    Construction performs no network work. The HTTP timeout bounds
    inactivity; the session above owns the deadline for a whole turn."""

    provider: str = ""

    def __init__(self, *, api_key: str, model: str, rate: int = 16000, timeout: float = 30,
                 max_audio_bytes: int = 8_000_000, detector=None,
                 transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("an api key is required")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("a model is required")
        if type(rate) is not int or not 8000 <= rate <= 48000:
            raise ValueError("rate must be an integer in 8000..48000")
        if type(max_audio_bytes) is not int or max_audio_bytes < 1024:
            raise ValueError("max_audio_bytes must be at least 1024")
        self._api_key = api_key
        self._model = model
        self._rate = rate
        self._max_audio = max_audio_bytes
        self._detector = detector if detector is not None else VoiceGate(rate=rate)
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)

    # -- the shape a provider fills in ------------------------------------

    def _request(self) -> tuple[str, dict, dict]:
        """Where to ask, what to say who we are, and the fields beside the
        audio. Subclasses give the provider's own contract."""
        raise NotImplementedError

    def _text(self, payload: object) -> str:
        """The transcript out of the provider's answer, or "" when there
        is none to be had."""
        raise NotImplementedError

    # -- listening ---------------------------------------------------------

    def transcribe(self, audio: AsyncIterator[AudioChunk]) -> AsyncIterator[object]:
        return self._listen(audio)

    async def _listen(self, audio):
        said: list[bytes] = []
        speaking = False
        async for chunk in audio:
            check_audio(chunk, self._max_audio)
            if chunk.sample_rate != self._rate:
                raise TranscriptionProviderError(
                    f"this recogniser listens at {self._rate} Hz, not {chunk.sample_rate}")
            heard = await self._detector.detect(chunk)
            if heard:
                said.append(chunk.pcm)
                if len(b"".join(said)) > self._max_audio:
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
                yield Transcript(text=text, final=True)
        if said:
            # The audio ended mid-sentence. Somebody still said something.
            yield Transcript(text=await self._ask(b"".join(said)), final=True)

    async def _ask(self, pcm: bytes) -> str:
        url, headers, fields = self._request()
        try:
            answer = await self._client.post(
                url, headers=headers, data=fields,
                files={"file": ("speech.wav", self._wav(pcm), "audio/wav")})
        except httpx.HTTPError as exc:
            raise TranscriptionProviderError(
                f"{self.provider} transcription did not answer: {type(exc).__name__}") from None
        if answer.status_code != 200:
            # The status, and nothing the far end wrote: its body is
            # someone else's prose and may quote the request back.
            raise TranscriptionProviderError(
                f"{self.provider} transcription refused with {answer.status_code}")
        try:
            payload = answer.json()
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
        await self._client.aclose()
