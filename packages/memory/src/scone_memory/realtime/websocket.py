"""A browser's audio over one WebSocket, as a Scone AudioTransport.

Binary frames carry signed 16-bit little-endian PCM in the format fixed at
construction; text frames carry JSON control. Client to server: ``{"type":
"end"}`` finishes the input. Server to client: audio frames prefixed with a
header naming the turn (see ``frame``), ``{"type": "clear", "turn_id"}`` to
drop a turn's buffered playback, and ``{"type": "error", "reason"}`` before
the socket closes on a frame the format does not admit. A frame the format
does not admit fails the session; nothing is resampled or reinterpreted.
No provider, device or framework is imported here.
"""

from __future__ import annotations

import json
import struct

from .audio import AudioChunk

_FORMAT = struct.Struct("<IB")  # sample_rate, channels; preceded by a length-prefixed turn id
_MAX_TURN_ID = 255


def frame(audio: AudioChunk, turn_id: str) -> bytes:
    """One outgoing audio frame: turn id length, turn id, format, PCM."""
    ident = turn_id.encode("ascii") if isinstance(turn_id, str) else b""
    if not 1 <= len(ident) <= _MAX_TURN_ID:
        raise ValueError("turn_id must be 1..255 ASCII characters")
    return bytes([len(ident)]) + ident + _FORMAT.pack(audio.sample_rate, audio.channels) + audio.pcm


def parse_frame(data: bytes) -> tuple[str, AudioChunk]:
    """The inverse of ``frame``, for clients written in Python and for tests."""
    if not isinstance(data, bytes) or len(data) < 1 + 1 + _FORMAT.size:
        raise ValueError("audio frame is too short")
    length = data[0]
    body = 1 + length
    if length == 0 or len(data) < body + _FORMAT.size:
        raise ValueError("audio frame header is incomplete")
    sample_rate, channels = _FORMAT.unpack_from(data, body)
    return data[1:body].decode("ascii"), AudioChunk(data[body + _FORMAT.size:], sample_rate, channels)


class WebSocketAudioTransport:
    """The AudioTransport for one accepted Starlette WebSocket.

    The socket is accepted, and the format negotiated, by the host before
    this exists; ``receive`` yields every binary frame as a chunk in that
    format and ends on the client's ``end`` control or its departure. Once
    the client has gone, ``send`` and ``clear`` raise, as the protocol
    requires, instead of writing into a closed socket.
    """

    def __init__(self, websocket, *, sample_rate: int, channels: int = 1, max_frame_bytes: int = 64000):
        # The format's own checks apply to a probe chunk so a bad format
        # fails here, at the host, not on the first frame.
        AudioChunk(b"\x00\x00" * channels if channels in (1, 2) else b"\x00\x00", sample_rate, channels)
        if type(max_frame_bytes) is not int or not 2 <= max_frame_bytes <= 1_000_000:
            raise ValueError("max_frame_bytes must be an integer in 2..1000000")
        self._socket = websocket
        self._sample_rate, self._channels, self._max_frame = sample_rate, channels, max_frame_bytes
        self._disconnected = False
        self._closed = False

    async def receive(self):
        while True:
            message = await self._socket.receive()
            if message["type"] == "websocket.disconnect":
                self._disconnected = True
                return
            data = message.get("bytes")
            if data is not None:
                if len(data) > self._max_frame:
                    await self._fail("audio frame exceeds the byte limit")
                    raise ValueError("audio frame exceeds the byte limit")
                try:
                    chunk = AudioChunk(data, self._sample_rate, self._channels)
                except ValueError as error:
                    await self._fail(str(error))
                    raise
                yield chunk
                continue
            control = self._control(message.get("text"))
            if control is None:
                await self._fail("unsupported control message")
                raise ValueError("unsupported control message")
            return

    @staticmethod
    def _control(text):
        """The one accepted control, ``end``; anything else is None."""
        try:
            decoded = json.loads(text) if isinstance(text, str) else None
        except ValueError:
            return None
        return "end" if isinstance(decoded, dict) and decoded.get("type") == "end" and len(decoded) == 1 else None

    async def _fail(self, reason: str):
        """Say why before closing; the reason names the rule, never the data."""
        try:
            await self._socket.send_text(json.dumps({"type": "error", "reason": reason}))
        except Exception:
            pass

    def _live(self):
        if self._disconnected:
            raise RuntimeError("audio client disconnected")
        if self._closed:
            raise RuntimeError("audio transport is closed")

    async def send(self, audio: AudioChunk, turn_id: str) -> None:
        self._live()
        await self._socket.send_bytes(frame(audio, turn_id))

    async def clear(self, turn_id: str) -> None:
        self._live()
        await self._socket.send_text(json.dumps({"type": "clear", "turn_id": turn_id}))

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._disconnected:
            return
        try:
            await self._socket.close()
        except RuntimeError:
            pass  # already closed by the host
