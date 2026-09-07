"""A carrier's media stream as frames, and frames back as its messages."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union

from ..audio import Resampler, pcm
from ..realtime.audio import AudioChunk
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
    """A key pressed on the phone."""

    digit: str


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

    def __init__(self, dialect: Dialect, *, rate: Optional[int] = None) -> None:
        self.dialect = dialect
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
            if self._inbound is not None:
                audio = self._inbound.feed(audio)
            return [AudioChunk(pcm=audio, sample_rate=self.rate, channels=1)] if audio else []
        if event == dialect.dtmf_event:
            digit = _first(body.get(dialect.dtmf_event) or {}, dialect.digit_keys)
            return [Dtmf(digit=digit)] if digit else []
        if event == dialect.stop_event:
            return [CallEnded(stream_id=self.stream_id or _first(body, [dialect.stream_key]) or "")]
        return []

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
