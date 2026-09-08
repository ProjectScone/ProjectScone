"""Scone-owned audio events, provider interfaces, and the rules they keep.

No scheduler, device, provider SDK or third-party conversation framework is
imported here. Adapters translate their native data into these public events.

The checks at the end are those rules written once. An adapter is a piece
of somebody else's software, and the ways one can be wrong are the same
whether a turn is run by a voice session or by a pipeline; the two would
otherwise drift into disagreeing about what a well-behaved provider is.
"""

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from .events import TextDelta, ReplyCompleted, TextModel

# Shared response protocol for both audio and text sessions.
VoiceModel = TextModel

#: Longest run of text handed to a synthesizer when no sentence ever ends.
BREATH = 240
#: The end of a sentence: the punctuation, then a space or the end of the text.
SENTENCE = re.compile(r"[.!?](?:\s|$)")


def sentences(text: str) -> tuple[list[str], str]:
    """Split off every finished sentence and return what is left over.

    An answer arrives a word at a time and is spoken a sentence at a
    time, so the first words are heard while the rest is still being
    written. Text with no end in sight is broken at BREATH characters
    rather than held back forever."""
    done: list[str] = []
    while text:
        found = SENTENCE.search(text)
        end = found.end() if found else BREATH if len(text) >= BREATH else 0
        if not end:
            break
        done.append(text[:end])
        text = text[end:]
    return done, text


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


def check_audio(chunk: object, max_bytes: int) -> AudioChunk:
    """The rule every piece of audio has to keep, wherever it came from.

    Returns the chunk, so it reads as a checkpoint in the flow it guards
    rather than as a separate step that could be forgotten."""
    if not isinstance(chunk, AudioChunk) or len(chunk.pcm) > max_bytes:
        raise ValueError("invalid or oversized audio chunk")
    return chunk


def is_question(event: object, *, max_bytes: int = 32000) -> bool:
    """Whether a recognizer has produced a finished question to act on.

    Three outcomes, not two. A malformed event raises, because a
    recognizer that reports nonsense is broken and going on would mean
    trusting the rest of what it says. A partial or empty one is not yet
    a question and is simply waited on. Anything else is one."""
    if not isinstance(event, Transcript):
        raise ValueError("unsupported speech event")
    if type(event.final) is not bool or not isinstance(event.text, str):
        raise ValueError("invalid transcript event")
    if len(event.text.encode("utf-8")) > max_bytes:
        raise ValueError("transcript exceeds byte limit")
    if not event.final or not event.text.strip():
        return False
    if not isinstance(event.speaker, str) or not 1 <= len(event.speaker) <= 128:
        raise ValueError("invalid transcript speaker")
    return True


class Reply:
    """The rules a model's answer has to keep, over the whole stream.

    Public text and a stated ending, nothing after the ending, no empty
    piece, and a bounded size. An object rather than a function because
    half of that is about the stream as a whole: what follows an ending
    is not a longer answer, it is whatever the adapter had lying around,
    and there is no way to see that one event at a time."""

    __slots__ = ("max_bytes", "size", "done")

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.size = 0
        self.done = False

    def accept(self, event: object) -> None:
        if self.done:
            raise RuntimeError("model emitted data after completion")
        if isinstance(event, TextDelta):
            if not isinstance(event.text, str) or not event.text:
                raise ValueError("invalid public text delta")
            self.size += len(event.text.encode("utf-8"))
            if self.size > self.max_bytes:
                raise RuntimeError("voice reply byte limit reached")
        elif isinstance(event, ReplyCompleted):
            self.done = True
        else:
            raise ValueError("unsupported model event; only public text and completion are allowed")
