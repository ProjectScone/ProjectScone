"""Keys heard in the audio: DTMF tones found with the Goertzel algorithm.

A key on a phone is two tones at once, one from a low group of four
frequencies and one from a high group of four. A carrier usually reports
the key in its own message as well, but not always, and not every path to
a caller has messages. The tones are always in the audio.

Each block of ``WINDOW_MS`` (205 samples at 8 kHz, the classic choice)
is measured at the eight frequencies with Goertzel's recurrence, which
gives the power at one frequency for the cost of a two-tap filter. The
recurrence is run at the exact frequency, not rounded to a bin, so every
tone is measured where the standard puts it. A block is a key when, in
order (the first rule that fails is the ``reason`` a verdict gives):

* ``level``: both tones are at least ``MIN_LEVEL_DBFS``;
* ``share``: the two tones hold at least ``MIN_SHARE`` of the block's
  energy, which is what rejects noise and speech (whose energy is spread
  over many harmonics);
* ``twist``: the high tone is at most ``REVERSE_TWIST_DB`` louder than the
  low, and the low at most ``NORMAL_TWIST_DB`` louder than the high
  (ITU-T Q.24);
* ``group``: each tone is ``GROUP_MARGIN_DB`` above the next strongest in
  its group, so two keys at once are neither;
* ``off_nominal``: each tone is stronger at its nominal frequency than
  ``OFF_NOMINAL`` either side of it. With probes 5 % out, the boundary
  falls at 2.5 %: a tone 1.5 % off (Q.24's must-accept) is a key and one
  3.5 % off (its must-reject) is not.

A key is reported once, when consecutive blocks agreeing on it span
``MIN_TONE_MS``, and not again until ``RELEASE_BLOCKS`` blocks in a row
are not that key, so a key held down is one digit and a dropout of a few
milliseconds does not make two. Blocks start a quarter window apart, so
the span is measured to within 6.4 ms: at a half-window step, 40 ms tones
were heard only 72 % of the time with three blocks, and a 20 ms tone 15 %
of the time with two.

Standard library only; the time a caller's audio takes to measure is in
``benchmarks/dtmf-inband-v1.results.md``.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Optional, Sequence

LOW = (697, 770, 852, 941)
HIGH = (1209, 1336, 1477, 1633)
#: The key at each (low, high) pair, row by low tone.
LAYOUT = ("123A", "456B", "789C", "*0#D")

WINDOW_MS = 25.625
MIN_TONE_MS = 40.0
MIN_LEVEL_DBFS = -35.0
MIN_SHARE = 0.6
REVERSE_TWIST_DB = 4.0
NORMAL_TWIST_DB = 8.0
GROUP_MARGIN_DB = 10.0
OFF_NOMINAL = 0.05
TOLERANCE = 0.015
RELEASE_BLOCKS = 2
FULL_SCALE = 32768.0


def goertzel(block: Sequence[float], frequency: float, rate: int) -> float:
    """The squared magnitude of ``block``'s spectrum at ``frequency``."""
    coefficient = 2 * math.cos(2 * math.pi * frequency / rate)
    s1 = s2 = 0.0
    for sample in block:
        s1, s2 = sample + coefficient * s1 - s2, s1
    return s1 * s1 + s2 * s2 - coefficient * s1 * s2


@dataclass(frozen=True)
class Verdict:
    """What one block is: a key, or None and the first rule it failed."""

    key: Optional[str]
    reason: str


def _db(ratio: float) -> float:
    return 10 * math.log10(ratio) if ratio > 0 else -math.inf


def classify(block: Sequence[float], rate: int, *, min_level_dbfs: float = MIN_LEVEL_DBFS,
             min_share: float = MIN_SHARE) -> Verdict:
    """Whether one block of samples is a key, and if not, why not."""
    size = len(block)
    energy = sum(sample * sample for sample in block)
    floor = FULL_SCALE * 10 ** (min_level_dbfs / 20)
    if size == 0 or energy < size * floor * floor / 2:
        return Verdict(None, "level")
    lows = [goertzel(block, f, rate) for f in LOW]
    highs = [goertzel(block, f, rate) for f in HIGH]
    row = max(range(4), key=lows.__getitem__)
    column = max(range(4), key=highs.__getitem__)
    # A short block measures a tone that is in tolerance but off nominal
    # as weaker than it is (1633 Hz 1.5 % high reads 6.6 dB down at 8 kHz),
    # so each winner's power is the best of nominal and TOLERANCE either side.
    low = max(lows[row], *(goertzel(block, LOW[row] * (1 + d), rate) for d in (-TOLERANCE, TOLERANCE)))
    high = max(highs[column], *(goertzel(block, HIGH[column] * (1 + d), rate) for d in (-TOLERANCE, TOLERANCE)))
    # A sinusoid of amplitude A over N samples measures (A N / 2)^2 and
    # carries N A^2 / 2 of energy, so this share is 1 for a pure key.
    if (low + high) / (size * energy / 2) < min_share:
        return Verdict(None, "share")
    if min(low, high) < (floor * size / 2) ** 2:
        return Verdict(None, "level")
    twist = _db(high / low)
    if twist > REVERSE_TWIST_DB or twist < -NORMAL_TWIST_DB:
        return Verdict(None, "twist")
    for powers, best, power in ((lows, row, low), (highs, column, high)):
        runner_up = max(other for at, other in enumerate(powers) if at != best)
        if _db(power / runner_up) < GROUP_MARGIN_DB:
            return Verdict(None, "group")
    for nominal, power in ((LOW[row], lows[row]), (HIGH[column], highs[column])):
        if max(goertzel(block, nominal * (1 - OFF_NOMINAL), rate),
               goertzel(block, nominal * (1 + OFF_NOMINAL), rate)) >= power:
            return Verdict(None, "off_nominal")
    return Verdict(LAYOUT[row][column], "tone")


@dataclass(frozen=True)
class Tone:
    """A key heard in the audio: where its tones began, in milliseconds of
    the audio fed so far, and how long they had lasted when accepted."""

    key: str
    offset_ms: float
    tone_ms: float


class ToneDetector:
    """Keys in a stream of 16-bit PCM, fed in pieces of any size.

    Holds less than one window of samples between feeds."""

    def __init__(self, rate: int = 8000, *, min_tone_ms: float = MIN_TONE_MS,
                 min_level_dbfs: float = MIN_LEVEL_DBFS, min_share: float = MIN_SHARE) -> None:
        if type(rate) is not int or rate < 4000:
            raise ValueError("rate must be an integer of at least 4000 Hz: the high group reaches 1633 Hz")
        if not math.isfinite(min_tone_ms) or min_tone_ms <= 0:
            raise ValueError("min_tone_ms must be a finite positive number of milliseconds")
        self.rate = rate
        self.window = round(rate * WINDOW_MS / 1000)
        self.hop = self.window // 4
        #: Consecutive agreeing blocks whose span reaches ``min_tone_ms``.
        self.blocks_needed = max(1, math.ceil((min_tone_ms * rate / 1000 - self.window) / self.hop) + 1)
        self._level, self._share = min_level_dbfs, min_share
        self._samples: list[int] = []
        self._first = 0  # the stream position of _samples[0]
        self._key: Optional[str] = None
        self._run = 0
        self._start = 0
        self._reported = False
        self._misses = 0

    def feed(self, data: bytes) -> list[Tone]:
        if len(data) % 2:
            raise ValueError("PCM must be whole 16-bit samples")
        self._samples.extend(struct.unpack(f"<{len(data) // 2}h", data))
        heard: list[Tone] = []
        while len(self._samples) >= self.window:
            verdict = classify(self._samples[:self.window], self.rate,
                               min_level_dbfs=self._level, min_share=self._share)
            tone = self._block(verdict.key, self._first)
            if tone is not None:
                heard.append(tone)
            del self._samples[:self.hop]
            self._first += self.hop
        return heard

    def _block(self, key: Optional[str], start: int) -> Optional[Tone]:
        if key is not None and key == self._key:
            self._run += 1
            self._misses = 0
        elif self._reported and self._misses + 1 < RELEASE_BLOCKS:
            self._misses += 1  # one block inside a press that did not measure as it
            return None
        else:
            self._key, self._run, self._start, self._reported, self._misses = key, int(key is not None), start, False, 0
        if self._key is None or self._reported or self._run < self.blocks_needed:
            return None
        self._reported = True
        span = (self._run - 1) * self.hop + self.window
        return Tone(self._key, round(self._start * 1000 / self.rate, 3), round(span * 1000 / self.rate, 3))
