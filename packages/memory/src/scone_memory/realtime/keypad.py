"""Keys pressed on a phone, as events a conversation can take.

A caller on a phone line can answer with the keypad as well as the voice:
an account number, "press 1 for sales", a PIN ended with ``#``. A carrier
reports each key in its own event message, and a key can also be heard in
the audio itself as a pair of tones. Either way the transport hands the
session the same ``Keypress``, saying which of the two it was.
"""

from __future__ import annotations

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
