"""When the bot may take its turn, once the user's turn has been judged.

A voice session ends a user's turn at the recognizer's pause, or later
when an end-of-turn detector hears a clause left open (``turn_end``), or
when keys from the phone finish it (``keypad``), and answers it at once.
That is the default, ``EndOfTurn``, and it changes nothing. A strategy is
asked about every final transcript, after the detector, and can hold the
turn open instead; a held turn is joined by what the user says next, and
is released by the same bounds and with the same receipts as a clause
the detector held.

``MinSpeech`` holds a turn whose speech lasted less than ``seconds``: a
backchannel ("mm-hm") or a noise taken for words waits for more speech,
for up to the session's ``turn_hold``, and is then answered with the
reason ``strategy_timeout``. Speech is timed from the first sign of it in
the turn (the recognizer's or the activity detector's speech start, or a
partial transcript) to the arrival of its last final transcript, so the
time includes the recognizer's pause and its transcription. With no sign
of speech the duration is unknown, the turn is taken, and the cue says so.

``KeypadSubmit`` holds every spoken turn until the caller presses the
submit key (``#``): keys that do not end with it join the turn and wait
too, and an entry that ends with it finishes the turn, as ``keypad``. A
held turn waits up to the session's ``turn_max_duration`` from its first
transcript or key and is then answered as ``max_duration``. It needs the
session's keypad policy.

A strategy returns the detector's ``Judgement`` unchanged to take the
turn, or an ``incomplete`` one whose cue names the strategy to hold it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional, Protocol

from .keypad import keypad_key
from .turn_end import INCOMPLETE, Judgement

STRATEGIES = ("end_of_turn", "min_speech", "keypad_submit")
#: Seconds a turn's speech must last under ``MinSpeech`` by default, and at most.
MIN_SPEECH_S = 0.8
MAX_MIN_SPEECH_S = 30.0


class TurnStrategy(Protocol):
    """Whether the bot may take its turn now. ``submit`` is the key that
    finishes a turn when keys alone may do so, or None when every keypad
    entry finishes it (today)."""

    @property
    def name(self) -> str: ...

    @property
    def submit(self) -> Optional[str]: ...

    def decide(self, judgement: Judgement, speech_ms: Optional[float]) -> Judgement: ...


@dataclass(frozen=True)
class EndOfTurn:
    """Take every turn when it ends: today's behaviour."""

    name: ClassVar[str] = "end_of_turn"
    submit: ClassVar[Optional[str]] = None

    def decide(self, judgement: Judgement, speech_ms: Optional[float]) -> Judgement:
        return judgement


@dataclass(frozen=True)
class MinSpeech:
    """Hold a turn whose speech is shorter than ``seconds``."""

    seconds: float = MIN_SPEECH_S
    name: ClassVar[str] = "min_speech"
    submit: ClassVar[Optional[str]] = None

    def __post_init__(self) -> None:
        value = self.seconds
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= MAX_MIN_SPEECH_S:
            raise ValueError(f"min_speech must be a number of seconds in (0, {MAX_MIN_SPEECH_S:g}]")

    def decide(self, judgement: Judgement, speech_ms: Optional[float]) -> Judgement:
        if judgement.verdict == INCOMPLETE:
            return judgement  # the detector already holds it
        if speech_ms is None:
            return Judgement(judgement.verdict, f"{judgement.cue}; min_speech: speech duration unknown")
        minimum = self.seconds * 1000
        if speech_ms >= minimum:
            return judgement
        return Judgement(INCOMPLETE, f"min_speech: {speech_ms:.0f} ms < {minimum:.0f} ms")


@dataclass(frozen=True)
class KeypadSubmit:
    """Hold every turn until the caller presses ``key``."""

    key: str = "#"
    name: ClassVar[str] = "keypad_submit"

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or keypad_key(self.key) != self.key:
            raise ValueError("a submit key is one keypad key")

    @property
    def submit(self) -> Optional[str]:
        return self.key

    def decide(self, judgement: Judgement, speech_ms: Optional[float]) -> Judgement:
        return Judgement(INCOMPLETE, f"keypad_submit: waiting for {self.key}")
