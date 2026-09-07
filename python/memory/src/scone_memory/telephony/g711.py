"""G.711 companding, the codec every phone line speaks.

Eight bits carry what sixteen did by spending resolution where the ear
is sensitive and saving it where it is not. Both laws are defined by
their decoders, which is what the far end runs, so the decode tables are
written from the standard and the encoder is derived from them: each
sample becomes the code whose decoded value is closest. That makes
encoding the exact inverse of decoding for all 256 codes, and it picks
the nearest representable level where a truncating encoder would round
down.
"""

from __future__ import annotations

import bisect
import struct
from typing import Optional

#: Added before the exponent is taken and removed after, so that small
#: samples still land on a segment boundary.
BIAS = 0x84
_SIGN = 0x80


def _ulaw_table() -> list[int]:
    values = []
    for code in range(256):
        bits = ~code & 0xFF
        magnitude = (((bits & 0x0F) << 3) + BIAS) << ((bits >> 4) & 0x07)
        magnitude -= BIAS
        values.append(-magnitude if bits & _SIGN else magnitude)
    return values


def _alaw_table() -> list[int]:
    values = []
    for code in range(256):
        bits = code ^ 0x55
        exponent = (bits >> 4) & 0x07
        mantissa = bits & 0x0F
        magnitude = (mantissa << 4) + 8 if exponent == 0 else ((mantissa << 4) + 264) << (exponent - 1)
        values.append(-magnitude if bits & _SIGN else magnitude)
    return values


ULAW_TO_PCM = _ulaw_table()
ALAW_TO_PCM = _alaw_table()


class _Nearest:
    """Every 16-bit sample mapped to the code that decodes closest to it,
    built once on first use so importing this module stays cheap."""

    def __init__(self, table: list[int]) -> None:
        self._pairs = sorted((value, code) for code, value in enumerate(table))
        self._values = [value for value, _ in self._pairs]
        self._codes: Optional[bytes] = None

    def codes(self) -> bytes:
        if self._codes is None:
            self._codes = bytes(self._nearest(sample) for sample in range(-32768, 32768))
        return self._codes

    def _nearest(self, sample: int) -> int:
        at = bisect.bisect_left(self._values, sample)
        if at == 0:
            return self._pairs[0][1]
        if at == len(self._pairs):
            return self._pairs[-1][1]
        below, above = self._pairs[at - 1], self._pairs[at]
        return below[1] if sample - below[0] <= above[0] - sample else above[1]


_ULAW = _Nearest(ULAW_TO_PCM)
_ALAW = _Nearest(ALAW_TO_PCM)


def _decode(data: bytes, table: list[int]) -> bytes:
    return struct.pack(f"<{len(data)}h", *(table[byte] for byte in data))


def _encode(data: bytes, nearest: _Nearest) -> bytes:
    codes = nearest.codes()
    return bytes(codes[sample + 32768] for sample in struct.unpack(f"<{len(data) // 2}h", data))


def ulaw_decode(data: bytes) -> bytes:
    """μ-law bytes as 16-bit PCM."""
    return _decode(data, ULAW_TO_PCM)


def ulaw_encode(data: bytes) -> bytes:
    """16-bit PCM as μ-law bytes."""
    return _encode(data, _ULAW)


def alaw_decode(data: bytes) -> bytes:
    """A-law bytes as 16-bit PCM."""
    return _decode(data, ALAW_TO_PCM)


def alaw_encode(data: bytes) -> bytes:
    """16-bit PCM as A-law bytes."""
    return _encode(data, _ALAW)


#: The codecs a carrier may ask for, by the name the dialects use.
CODECS = {
    "ulaw": (ulaw_decode, ulaw_encode),
    "alaw": (alaw_decode, alaw_encode),
    "pcm": (lambda data: data, lambda data: data),
}
