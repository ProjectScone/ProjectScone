"""A carrier's media stream as frames, and frames back as its messages."""

from __future__ import annotations

import base64
import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union

from ..audio import Resampler, pcm
from ..audio.dtmf import ToneDetector
from ..realtime.audio import AudioChunk
from ..realtime.keypad import keypad_key
from .dialects import Dialect
from .g711 import CODECS


@dataclass(frozen=True)
class CallStarted:
    """The carrier has connected a call to us."""

    stream_id: str
    call_id: Optional[str]
    rate: int
    codec: str


@dataclass(frozen=True)
class CallEnded:
    """The carrier has hung up or stopped streaming."""

    stream_id: str


@dataclass(frozen=True)
class Dtmf:
    """A key pressed on the phone: which, how it arrived (``event`` for the
    carrier's own message), and how far into the caller's audio."""

    digit: str
    source: str = "event"
    offset_ms: Optional[float] = None
    tone_ms: Optional[float] = None


#: Which digits a stream reports: none, those in the carrier's dtmf
#: messages, those heard as tones in the caller's audio, or both.
DIGITS = ("off", "events", "inband", "both")
#: With both, a key reported one way and then heard the other way within
#: this much of the call's audio is the same press, reported once. It runs
#: from where a key's tones were last heard, so a key held down and
#: reported by the carrier when let go is still one press.
DUPLICATE_WINDOW_MS = 1000.0
#: With both, reported digits still waiting to be heard the other way. A
#: flood of carrier digits with no audio between them is not a reason to
#: grow without bound; past it the oldest is forgotten and counted.
MAX_UNPAIRED = 64


@dataclass
class _Hearing:
    """A digit waiting to be paired, and where in the audio it was last heard:
    its report for a carrier event, the end of its tones so far for a tone."""

    digit: Dtmf
    last_ms: float


def _first(source: Mapping[str, Any], keys) -> Optional[str]:
    for key in keys:
        value = source.get(key)
        if isinstance(value, (str, int)) and str(value):
            return str(value)
    return None


class MediaStream:
    """One call. Reads a carrier's messages as frames and writes frames
    back in its envelope, converting rates in both directions so the rest
    of the pipeline never has to know it is on a phone line."""

    def __init__(self, dialect: Dialect, *, rate: Optional[int] = None, digits: str = "events") -> None:
        if digits not in DIGITS:
            raise ValueError(f"digits must be one of {DIGITS}")
        self.dialect = dialect
        self.digits = digits
        #: Dtmf messages whose digit was not a key on a keypad.
        self.unreadable_digits = 0
        #: Keys heard both ways and reported once (``both`` only).
        self.duplicates = 0
        #: Reported digits forgotten before the other way could pair them, past MAX_UNPAIRED.
        self.unpaired_forgotten = 0
        #: Samples of the caller's audio decoded so far, at the line's rate.
        self._heard = 0
        self._tones = ToneDetector(dialect.rate) if digits in ("inband", "both") else None
        self._unpaired: deque[_Hearing] = deque()
        #: The rate the rest of the pipeline runs at. None means the line's.
        self.rate = rate or dialect.rate
        self.stream_id: Optional[str] = None
        self.call_id: Optional[str] = None
        self._decode, self._encode = CODECS[dialect.codec]
        self._inbound = None if self.rate == dialect.rate else Resampler(dialect.rate, self.rate)
        self._outbound = None if self.rate == dialect.rate else Resampler(self.rate, dialect.rate)

    def inbound(self, message: Union[str, bytes, Mapping[str, Any]]) -> list[object]:
        """What this carrier message means. A message we do not act on is
        not an error: carriers add events, and a call should survive one."""
        body = message if isinstance(message, Mapping) else json.loads(message)
        event = body.get("event")
        dialect = self.dialect
        if event == dialect.start_event:
            start = body.get(dialect.start_event) or {}
            self.stream_id = _first(body, [dialect.stream_key]) or _first(start, [dialect.stream_key])
            self.call_id = _first(start, dialect.call_keys) or _first(body, dialect.call_keys)
            return [CallStarted(stream_id=self.stream_id or "", call_id=self.call_id,
                                rate=dialect.rate, codec=dialect.codec)]
        if event == dialect.media_event:
            media = body.get(dialect.media_event) or {}
            payload = _first(media, dialect.payload_keys)
            if not payload:
                return []
            audio = self._decode(base64.b64decode(payload))
            self._heard += len(audio) // 2
            heard = [] if self._tones is None else [
                Dtmf(tone.key, "inband", tone.offset_ms, tone.tone_ms) for tone in self._tones.feed(audio)]
            if self._inbound is not None:
                audio = self._inbound.feed(audio)
            frames: list[object] = [AudioChunk(pcm=audio, sample_rate=self.rate, channels=1)] if audio else []
            frames += [digit for digit in heard if self._first_hearing(digit)]
            self._still_sounding()
            return frames
        if event == dialect.dtmf_event:
            if self.digits not in ("events", "both"):
                return []
            key = keypad_key(_first(body.get(dialect.dtmf_event) or {}, dialect.digit_keys))
            if key is None:
                self.unreadable_digits += 1
                return []
            digit = Dtmf(digit=key, offset_ms=self._heard * 1000 / dialect.rate)
            return [digit] if self._first_hearing(digit) else []
        if event == dialect.stop_event:
            return [CallEnded(stream_id=self.stream_id or _first(body, [dialect.stream_key]) or "")]
        return []

    def _first_hearing(self, digit: Dtmf) -> bool:
        """Whether this digit is a press not already reported the other way.

        Pairs it with the oldest digit reported by the other way, for the
        same key, within DUPLICATE_WINDOW_MS; a paired digit is used up, so
        a key pressed twice and heard both ways is still two digits."""
        if self.digits != "both":
            return True
        at = digit.offset_ms or 0.0
        # Every waiting digit is looked at, not only the oldest: the oldest
        # can be a key still held down while a later one has left the window.
        self._unpaired = deque(waiting for waiting in self._unpaired if at - waiting.last_ms <= DUPLICATE_WINDOW_MS)
        for earlier in self._unpaired:
            if earlier.digit.digit == digit.digit and earlier.digit.source != digit.source:
                self._unpaired.remove(earlier)
                self.duplicates += 1
                return False
        if len(self._unpaired) >= MAX_UNPAIRED:
            self._unpaired.popleft()
            self.unpaired_forgotten += 1
        self._unpaired.append(_Hearing(digit, at))
        return True

    def _still_sounding(self) -> None:
        """The newest waiting digit of a key whose tones are still heard was
        last heard now: a key held down is still being pressed."""
        key = None if self._tones is None else self._tones.sounding
        for waiting in reversed(self._unpaired):
            if waiting.digit.digit == key:
                waiting.last_ms = self._heard * 1000 / self.dialect.rate
                break

    def outbound(self, audio: AudioChunk) -> str:
        """Our audio in this carrier's envelope, at the line's rate."""
        if self.stream_id is None:
            raise ValueError("no stream yet: a carrier reply has to name the call it answers")
        if audio.sample_rate != self.rate:
            raise ValueError(f"this stream speaks {self.rate} Hz, not {audio.sample_rate}")
        data = pcm.to_mono(audio.pcm, audio.channels)
        if self._outbound is not None:
            data = self._outbound.feed(data)
        return json.dumps({
            "event": self.dialect.media_event,
            self.dialect.stream_key: self.stream_id,
            self.dialect.media_event: {"payload": base64.b64encode(self._encode(data)).decode()},
        })

    def clear(self) -> Optional[str]:
        """Tell the carrier to drop what it still holds, for barge-in.
        None where the call has not started or the carrier has no such
        message, in which case speech already sent will still be heard."""
        if self.stream_id is None or self.dialect.clear_event is None:
            return None
        return json.dumps({"event": self.dialect.clear_event, self.dialect.stream_key: self.stream_id})
