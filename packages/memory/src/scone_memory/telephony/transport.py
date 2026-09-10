"""A phone call as an audio transport.

The voice session already knows how to run a conversation over an
``AudioTransport``: audio in, audio out, a way to drop what has been
queued when someone interrupts. A carrier speaks JSON with base64 audio
at 8 kHz over a socket, so the difference is a translation rather than
another kind of session. With this in front of it, the session never
learns it is on a phone.
"""

from __future__ import annotations

from typing import AsyncIterator, Optional

from ..realtime.audio import AudioChunk
from .dialects import Dialect
from .stream import CallEnded, CallStarted, Dtmf, MediaStream

#: Keypresses kept for the session to read. A caller leaning on a key is
#: not a reason to grow without bound.
MAX_DIGITS = 64


class CarrierTransport:
    """One call, as the transport a voice session expects.

    ``socket`` is anything with ``receive_text``, ``send_text`` and
    ``close``: a framework's WebSocket, or a double in a test."""

    def __init__(self, socket, dialect: Dialect, *, rate: int = 16000) -> None:
        self._socket = socket
        self._stream = MediaStream(dialect, rate=rate)
        self.rate = self._stream.rate
        #: Keypresses heard so far, in order.
        self.digits: list[str] = []
        self._ended = False
        self._closed = False

    @property
    def stream_id(self) -> Optional[str]:
        return self._stream.stream_id

    @property
    def call_id(self) -> Optional[str]:
        return self._stream.call_id

    @property
    def ended(self) -> bool:
        """Whether the carrier has stopped the call or the socket is gone."""
        return self._ended

    async def ready(self) -> None:
        """Wait for the call to start, so a reply can name it. A carrier
        sends a start message before any audio; answering before it has
        arrived would be speaking to nobody."""
        while self.stream_id is None and not self._ended:
            await self._read()

    def receive(self) -> AsyncIterator[AudioChunk]:
        """The caller's speech, at the rate this transport was built for."""

        async def audio() -> AsyncIterator[AudioChunk]:
            while not self._ended:
                for frame in await self._read():
                    if isinstance(frame, AudioChunk):
                        yield frame

        return audio()

    async def _read(self) -> list[object]:
        """One carrier message, as frames. A socket that has gone is the
        end of the call, not an error to raise at the session."""
        try:
            frames = self._stream.inbound(await self._socket.receive_text())
        except Exception:  # noqa: BLE001 - the far end hanging up is ordinary
            self._ended = True
            return []
        kept = []
        for frame in frames:
            if isinstance(frame, CallEnded):
                self._ended = True
            elif isinstance(frame, Dtmf):
                if len(self.digits) < MAX_DIGITS:
                    self.digits.append(frame.digit)
            elif isinstance(frame, CallStarted):
                pass  # the stream keeps the identifiers; nothing else to do
            kept.append(frame)
        return kept

    def _live(self) -> None:
        if self._ended:
            raise RuntimeError("the call has ended")
        if self._closed:
            raise RuntimeError("the call has ended: this transport is closed")
        if self.stream_id is None:
            raise RuntimeError("the call has not started: there is nobody to answer yet")

    async def send(self, audio: AudioChunk, turn_id: str) -> None:
        """Speak, at the line's rate and in the carrier's envelope."""
        self._live()
        await self._socket.send_text(self._stream.outbound(audio))

    async def clear(self, turn_id: str) -> None:
        """Drop what the carrier still holds, for a barge-in. A carrier
        with no such message leaves what was sent to be heard, and saying
        so is better than pretending the audio stopped."""
        self._live()
        message = self._stream.clear()
        if message is None:
            raise RuntimeError(f"{self._stream.dialect.name} has no way to drop queued audio")
        await self._socket.send_text(message)

    async def aclose(self) -> None:
        """Hang up once."""
        if self._closed:
            return
        self._closed = True
        self._ended = True
        try:
            await self._socket.close()
        except Exception:  # noqa: BLE001 - the far end may already have gone
            pass
