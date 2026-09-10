"""Telling speech from a room.

Loudness alone would open on a door closing and close inside a pause, so
the gate needs both ends to persist: an unbroken run of loud frames
opens a turn, and an unbroken run of quiet ones ends it. That hysteresis
is what makes barge-in usable, and it is the same shape a learned
detector plugs into (``SpeechActivityDetector``), so a model can replace
the loudness rule without anything around it changing.
"""

from __future__ import annotations

from ..realtime.audio import AudioChunk
from . import pcm


class VoiceGate:
    """Speech from loudness, with a run required at each end."""

    def __init__(self, *, rate: int = 16000, threshold: float = 0.05,
                 start_ms: int = 120, stop_ms: int = 400) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold is a level from 0 to 1")
        self.rate = rate
        self.threshold = threshold
        self.start_ms = start_ms
        self.stop_ms = stop_ms
        #: Whether a turn is open right now.
        self.speaking = False
        self._loud_ms = 0.0
        self._quiet_ms = 0.0

    async def detect(self, audio: AudioChunk) -> bool:
        """Whether someone is speaking, given this chunk and what came
        before. The rate must be the one the gate was built for: a
        reinterpreted sample rate is a wrong answer, not an approximation."""
        if audio.sample_rate != self.rate:
            raise ValueError(f"this gate listens at {self.rate} Hz, not {audio.sample_rate}")
        data = pcm.to_mono(audio.pcm, audio.channels)
        span = pcm.duration_ms(data, self.rate)
        if pcm.rms(data) >= self.threshold:
            self._loud_ms += span
            self._quiet_ms = 0.0
            if self._loud_ms >= self.start_ms:
                self.speaking = True
        else:
            self._quiet_ms += span
            self._loud_ms = 0.0
            if self._quiet_ms >= self.stop_ms:
                self.speaking = False
        return self.speaking

    async def aclose(self) -> None:
        """Nothing is held open; here so the gate is a detector like any other."""
        return None
