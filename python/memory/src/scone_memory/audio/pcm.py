"""Signed 16-bit little-endian PCM, the one format we carry.

Byte order is written out rather than inherited from the machine, so a
recording made on one host means the same on another.
"""

from __future__ import annotations

import math
import struct
from typing import Iterable, Sequence

#: Bytes per sample. Everything here is signed 16-bit.
WIDTH = 2
FLOOR, CEILING = -32768, 32767
#: What a full-scale sample is worth, for turning counts into a 0..1 level.
FULL_SCALE = 32768.0


def to_samples(data: bytes) -> list[int]:
    """The samples in ``data``. A partial sample is an error, never a guess."""
    if len(data) % WIDTH:
        raise ValueError(f"PCM must be whole 16-bit samples; got {len(data)} bytes")
    return list(struct.unpack(f"<{len(data) // WIDTH}h", data))


def to_bytes(samples: Iterable[float]) -> bytes:
    """Samples as PCM, each one held at the ends of the range rather than
    allowed to wrap, because a wrapped sample is a loud click."""
    held = [CEILING if s > CEILING else FLOOR if s < FLOOR else int(s) for s in samples]
    return struct.pack(f"<{len(held)}h", *held)


def to_mono(data: bytes, channels: int) -> bytes:
    """One channel from many, by averaging: the shape every recognizer wants."""
    if channels < 1:
        raise ValueError("channels must be at least 1")
    if channels == 1:
        return data
    samples = to_samples(data)
    if len(samples) % channels:
        raise ValueError(f"{len(samples)} samples do not divide into {channels} channels")
    return to_bytes(sum(samples[i:i + channels]) / channels for i in range(0, len(samples), channels))


def gain(data: bytes, factor: float) -> bytes:
    """Louder or quieter, held at the ends of the range."""
    return to_bytes(s * factor for s in to_samples(data))


def rms(data: bytes) -> float:
    """Loudness from 0 to 1, root mean square over full scale."""
    samples = to_samples(data)
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples)) / FULL_SCALE


def duration_ms(data: bytes, rate: int, channels: int = 1) -> float:
    """How long this audio lasts."""
    return len(data) / (WIDTH * channels * rate) * 1000.0


def frame_bytes(rate: int, ms: int, channels: int = 1) -> int:
    """The size of one frame of ``ms`` at this rate."""
    return int(rate * ms / 1000) * WIDTH * channels


def silence(rate: int, ms: int, channels: int = 1) -> bytes:
    """Quiet of a given length, for padding a tail or filling a gap."""
    return b"\x00" * frame_bytes(rate, ms, channels)


def mix(first: bytes, second: bytes) -> bytes:
    """Two streams heard together, summed and held at the ends of the
    range. The shorter one runs out and the longer one carries on."""
    a, b = to_samples(first), to_samples(second)
    if len(a) < len(b):
        a, b = b, a
    return to_bytes(s + (b[i] if i < len(b) else 0) for i, s in enumerate(a))


def peak(data: bytes) -> float:
    """The loudest single sample, 0 to 1: what a limiter watches."""
    samples = to_samples(data)
    return max((abs(s) for s in samples), default=0) / FULL_SCALE


def as_frames(data: bytes, size: int) -> tuple[list[bytes], bytes]:
    """``data`` cut into frames of ``size`` bytes, with the remainder."""
    whole = len(data) // size
    return [data[i * size:(i + 1) * size] for i in range(whole)], data[whole * size:]


def clamp(samples: Sequence[float]) -> list[int]:
    """Samples held inside the range, as integers."""
    return to_samples(to_bytes(samples))
