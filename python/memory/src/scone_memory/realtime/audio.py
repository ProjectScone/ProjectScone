"""Scone-owned audio events and structural provider interfaces.

No scheduler, device, provider SDK or third-party conversation framework is
imported here. Adapters translate their native data into these public events.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from .events import TextDelta, ReplyCompleted, TextModel

# Shared response protocol for both audio and text sessions.
VoiceModel = TextModel


@dataclass(frozen=True)
class AudioChunk:
    """Signed 16-bit little-endian interleaved PCM; no implicit resampling."""

    pcm: bytes
    sample_rate: int
    channels: int = 1

    def __post_init__(self):
        if not isinstance(self.pcm, bytes) or not self.pcm:
            raise ValueError("PCM must be nonempty bytes")
        if type(self.sample_rate) is not int or not 8000 <= self.sample_rate <= 192000:
            raise ValueError("sample_rate must be an integer in 8000..192000")
        if type(self.channels) is not int or self.channels not in (1, 2):
            raise ValueError("channels must be 1 or 2")
        if len(self.pcm) % (2 * self.channels):
            raise ValueError("PCM must contain complete signed 16-bit sample frames")


@dataclass(frozen=True)
class SpeechStarted:
    """Recognizer/turn detector observed new speech; interrupt current output."""


@dataclass(frozen=True)
class Transcript:
    text: str
    final: bool = True
    speaker: str = "user"


class AudioTransport(Protocol):
    def receive(self) -> AsyncIterator[AudioChunk]: ...
    async def send(self, audio: AudioChunk, turn_id: str) -> None: ...
    async def clear(self, turn_id: str) -> None:
        """Discard queued output for this turn; failure must raise."""
        ...
    async def aclose(self) -> None: ...


class SpeechRecognizer(Protocol):
    def transcribe(self, audio: AsyncIterator[AudioChunk]) -> AsyncIterator[SpeechStarted | Transcript]: ...
    async def aclose(self) -> None: ...


class SpeechSynthesizer(Protocol):
    def synthesize(self, text: str) -> AsyncIterator[AudioChunk]: ...
    async def aclose(self) -> None: ...


class SpeechActivityDetector(Protocol):
    """Stateful per-session detector; True means speech in this PCM chunk.

    The adapter owns sample windows and its model's supported PCM formats.
    It must reject unsupported formats, never silently reinterpret samples.
    """

    async def detect(self, audio: AudioChunk) -> bool: ...
    async def aclose(self) -> None: ...
