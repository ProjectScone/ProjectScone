"""Ragged arrivals into the exact frames a provider expects."""

from __future__ import annotations

from . import pcm


class Framer:
    """Cuts a stream into frames of a fixed length, holding the remainder
    for the next arrival. Providers bill and behave by frame size, and a
    short frame is usually a protocol error rather than a small one."""

    def __init__(self, *, rate: int = 16000, ms: int = 20, channels: int = 1) -> None:
        self.size = pcm.frame_bytes(rate, ms, channels)
        if self.size <= 0:
            raise ValueError("a frame needs at least one sample")
        self.rate = rate
        self.ms = ms
        self.channels = channels
        self._held = b""

    def push(self, data: bytes) -> list[bytes]:
        """Every whole frame this arrival completes."""
        frames, self._held = pcm.as_frames(self._held + data, self.size)
        return frames

    def flush(self) -> bytes:
        """The last partial frame, padded with quiet. Empty when nothing
        is held, so ending twice is not an extra frame of silence."""
        if not self._held:
            return b""
        tail, self._held = self._held + b"\x00" * (self.size - len(self._held)), b""
        return tail
