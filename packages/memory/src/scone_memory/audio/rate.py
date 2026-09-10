"""Rate conversion that keeps the voice and drops what cannot exist.

A browser sends 48 kHz; a recognizer wants 16 kHz; a synthesizer answers
at 24 kHz and a phone line wants 8 kHz. Dropping samples to bridge those
is not conversion: a 15 kHz hiss becomes a 1 kHz whistle sitting in the
middle of speech, and it cannot be removed afterwards. So the signal is
low-passed at the lower of the two Nyquist limits first, and only the
output samples that are actually kept are ever computed.
"""

from __future__ import annotations

import math
from math import gcd

from . import pcm

#: Filter length per phase. Sixteen is inaudible for speech and cheap
#: enough for 20 ms frames in plain Python.
QUALITY = 16


def _window_sinc(length: int, cutoff: float, gain: float) -> list[float]:
    """A low-pass with a Hamming window: linear phase, and side lobes far
    enough down that a rejected tone stays rejected."""
    middle = (length - 1) / 2
    taps = []
    for i in range(length):
        x = i - middle
        ideal = 2 * cutoff if x == 0 else math.sin(2 * math.pi * cutoff * x) / (math.pi * x)
        taps.append(ideal * (0.54 - 0.46 * math.cos(2 * math.pi * i / (length - 1))))
    total = sum(taps)
    return [t * gain / total for t in taps]


class Resampler:
    """Converts one rate to another, keeping its filter state across
    chunks so a boundary is not a click. Feed it whatever arrives."""

    def __init__(self, source_rate: int, target_rate: int, *, quality: int = QUALITY) -> None:
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("rates must be positive")
        self.source_rate = source_rate
        self.target_rate = target_rate
        shared = gcd(source_rate, target_rate)
        #: Output samples per input sample, as a ratio in lowest terms.
        self.up = target_rate // shared
        self.down = source_rate // shared
        self.quality = quality
        self._taps = _window_sinc(quality * self.up, 0.5 / max(self.up, self.down), float(self.up))
        self._history = [0] * quality
        self._made = 0

    @property
    def passthrough(self) -> bool:
        """Whether the two rates are the same, in which case nothing is done."""
        return self.up == 1 and self.down == 1

    def feed(self, data: bytes) -> bytes:
        """The part of the stream that can be finished with what has
        arrived. What needs later samples waits for the next call."""
        if self.passthrough:
            return data
        samples = self._history + pcm.to_samples(data)
        #: Index in ``samples`` of the newest input this call can use.
        last = len(samples) - 1
        base = self._made * self.down // self.up
        out = []
        while True:
            position = self._made * self.down
            newest = position // self.up - base + self.quality
            if newest > last:
                break
            phase = position % self.up
            out.append(sum(self._taps[phase + k * self.up] * samples[newest - k] for k in range(self.quality)))
            self._made += 1
        kept = self._made * self.down // self.up - base
        self._history = samples[kept:kept + self.quality] if kept else samples[:self.quality]
        return pcm.to_bytes(out)

    def flush(self) -> bytes:
        """Whatever the filter still holds, with the tail run out."""
        if self.passthrough:
            return b""
        return self.feed(pcm.to_bytes([0] * self.quality))
