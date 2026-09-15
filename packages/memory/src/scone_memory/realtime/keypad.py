"""Keys pressed on a phone, as events a conversation can take.

A caller on a phone line can answer with the keypad as well as the voice:
an account number, "press 1 for sales", a PIN ended with ``#``. A carrier
reports each key in its own event message, and a key can also be heard in
the audio itself as a pair of tones. Either way the transport hands the
session the same ``Keypress``, saying which of the two it was.

What a key does in the conversation is a ``KeypadPolicy``. ``append``
gives each key to the user's turn as it comes: it joins whatever the
caller has said and not yet finished, and the turn is answered. ``collect``
holds keys until the terminator (``#``), ``timeout`` seconds without a
key, or ``max_digits`` keys, and gives the turn one entry. Every entry
says why it ended (one of ``ENDINGS``), and how and when each key came.
``KeypadCollector`` holds no clock; the time is passed in, so a test
drives it with plain numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

#: The sixteen keys of the DTMF keypad, row by row. Most phones have only
#: the first three columns and ``*`` ``0`` ``#``; the fourth is still a key.
KEYS = "123A456B789C*0#D"
#: How a key arrived: the transport's own event message, or tones heard in the audio.
SOURCES = ("event", "inband")


def keypad_key(value: object) -> Optional[str]:
    """The key ``value`` names, or None when it names none. The fourth
    column is one key whichever case it is written in."""
    text = str(value).strip().upper()
    return text if len(text) == 1 and text in KEYS else None


@dataclass(frozen=True)
class Keypress:
    """One key, how it arrived, and where in the call's audio it was.

    ``offset_ms`` counts the caller's audio from the start of the call, so
    a key from an event and a key heard in the audio sit on one timeline;
    None when the transport cannot place it. ``tone_ms`` is how long the
    tones had lasted when they were accepted, for a key heard in the audio."""

    key: str
    source: str = "event"
    offset_ms: Optional[float] = None
    tone_ms: Optional[float] = None

    def __post_init__(self) -> None:
        if keypad_key(self.key) != self.key:
            raise ValueError(f"a keypress is one of {KEYS!r}")
        if self.source not in SOURCES:
            raise ValueError(f"a keypress arrives by one of {SOURCES}")


APPEND, COLLECT = "append", "collect"
MODES = (APPEND, COLLECT)
#: Why an entry was given to the turn: ``key`` (append: every key at once),
#: ``terminator``, ``timeout``, ``max_digits`` (the bound bit), ``speech``
#: (the caller said something while keys were held), ``input_ended`` and
#: ``session_ended``.
ENDINGS = ("key", "terminator", "timeout", "max_digits", "speech", "input_ended", "session_ended")
#: Seconds an entry waits for its next key.
TIMEOUT_S = 3.0
#: The longest entry, and the longest timeout: together they keep an
#: entry's receipt inside what one record's metadata value can hold.
MAX_DIGITS = 32
MAX_TIMEOUT_S = 60.0
TEMPLATE = "[keypad] {keys}"


@dataclass(frozen=True)
class KeypadPolicy:
    """What a key does in the conversation. ``template`` is the text the
    turn is given, with ``{keys}`` where the keys go."""

    mode: str
    terminator: Optional[str] = "#"
    timeout: float = TIMEOUT_S
    max_digits: int = MAX_DIGITS
    template: str = TEMPLATE

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"keypad mode must be one of {MODES}")
        if self.terminator is not None and keypad_key(self.terminator) != self.terminator:
            raise ValueError(f"a keypad terminator is None or one of {KEYS!r}")
        if (isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float))
                or not math.isfinite(self.timeout) or not 0 < self.timeout <= MAX_TIMEOUT_S):
            raise ValueError(f"keypad timeout must be a number of seconds in (0, {MAX_TIMEOUT_S:g}]")
        if type(self.max_digits) is not int or not 1 <= self.max_digits <= MAX_DIGITS:
            raise ValueError(f"keypad max_digits must be an integer in 1..{MAX_DIGITS}")
        marker = chr(0)
        try:
            rendered = self.template.format(keys=marker)
        except (AttributeError, IndexError, KeyError, ValueError):
            rendered = ""
        if marker not in rendered:
            raise ValueError("keypad template must be text with {keys} in it and no other field")


@dataclass(frozen=True)
class KeypadEntry:
    """Keys given to the turn together: the text the turn is given, each
    key, the clock time the session took each one at, and why it ended."""

    text: str
    presses: tuple[Keypress, ...]
    at: tuple[float, ...]
    ended_by: str

    @property
    def keys(self) -> str:
        return "".join(press.key for press in self.presses)

    def metadata(self, origin: float) -> dict[str, str]:
        """The entry as record metadata. ``keypad_sources`` has ``e`` for a
        key from the transport's event and ``i`` for one heard in the audio;
        ``keypad_started_ms`` is the first key from ``origin`` (the session's
        start) and ``keypad_at_ms`` each key from the first, in milliseconds."""
        first = self.at[0]
        return {"keypad_keys": self.keys, "keypad_ended": self.ended_by,
                "keypad_sources": "".join("i" if press.source == "inband" else "e" for press in self.presses),
                "keypad_started_ms": str(round((first - origin) * 1000)),
                "keypad_at_ms": ",".join(str(round((at - first) * 1000)) for at in self.at)}


class KeypadCollector:
    """The keys of the entry being collected, with the time passed in."""

    def __init__(self, policy: KeypadPolicy) -> None:
        self.policy = policy
        self._presses: list[Keypress] = []
        self._at: list[float] = []
        #: When the entry is given to the turn if no key comes; None when nothing is held.
        self.deadline: Optional[float] = None

    @property
    def pending(self) -> bool:
        return bool(self._presses)

    def press(self, press: Keypress, now: float) -> list[KeypadEntry]:
        released = self.expire(now)  # a key after the timeout starts the next entry
        self._presses.append(press)
        self._at.append(now)
        if self.policy.mode == APPEND:
            return [*released, self._release("key")]
        if press.key == self.policy.terminator:
            return [*released, self._release("terminator")]
        if len(self._presses) >= self.policy.max_digits:
            return [*released, self._release("max_digits")]
        self.deadline = now + self.policy.timeout
        return released

    def expire(self, now: float) -> list[KeypadEntry]:
        if self.deadline is None or now < self.deadline:
            return []
        return [self._release("timeout")]

    def drain(self, now: float, reason: str) -> list[KeypadEntry]:
        return [self._release(reason)] if self.pending else []

    def _release(self, reason: str) -> KeypadEntry:
        keys = "".join(press.key for press in self._presses)
        entry = KeypadEntry(self.policy.template.format(keys=keys), tuple(self._presses), tuple(self._at), reason)
        self._presses, self._at, self.deadline = [], [], None
        return entry
