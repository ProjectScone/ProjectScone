"""Whether a speaker has finished: from what was said, not only from the pause.

A recognizer ends an utterance when the room goes quiet for long enough
(``audio.gate.VoiceGate`` waits 400 ms). People pause for longer than that
in the middle of a sentence: "I want to book a table for ... four people."
Energy alone hears two turns there, and the assistant answers half a
question. What was said already tells the two apart. A transcript that
stops on "for", "and" or "the", inside an open quote or bracket, or on an
"um", is not finished; one that ends with a full stop or a question mark
is.

So a turn is decided in two places. The recognizer keeps the ordinary
silence threshold and produces its final transcript. A detector then
judges the text heard so far. Complete, or no evidence either way, and
the turn ends there, at the threshold the recognizer already waited. An
open clause holds the turn for up to ``hold`` seconds more; the speaker
starting again (or a noise onset the recognizer takes for speech, or a
streaming recognizer's partial words) keeps it open until
``max_duration`` after its first final transcript, and their next words
join it. Speech that a recognizer reports as an empty final transcript
runs the hold again from its end. (The HTTP recognizers do not report
one: an empty answer is their error, and the session ends.) The
hold is added to the recognizer's pause, never a replacement for it.

Every bound says when it bit. Each released turn carries a receipt whose
reason is one of ``REASONS``: ``silence`` (no evidence, or no detector),
``semantic_complete``, ``semantic_incomplete_timeout`` (the hold ran
out), ``max_duration`` (the whole turn's bound), ``max_bytes`` (a held
turn would outgrow the transcript limit), ``speaker_changed``,
``input_ended`` and ``session_ended`` (the session stopped with a turn
held: its words are recorded and nothing is answered), and ``keypad`` (keys
from the phone joined the turn and finished it; see ``realtime.keypad``).

The detector is a seam (``EndOfTurnDetector``): the lexical rules here
need no model, and ``ChatEndOfTurn`` puts any chat model behind the same
two methods. ``TurnHold`` holds no clock; the time is passed in, so a
test drives it with plain numbers.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..providers.llm import ChatModel

COMPLETE, INCOMPLETE, UNSURE = "complete", "incomplete", "unsure"
VERDICTS = (COMPLETE, INCOMPLETE, UNSURE)
REASONS = ("silence", "semantic_complete", "semantic_incomplete_timeout", "max_duration", "max_bytes",
           "speaker_changed", "input_ended", "session_ended", "keypad")

#: Seconds an open clause may hold a turn past the recognizer's own pause.
HOLD_S = 1.5
#: Seconds from a turn's first final transcript to its release, however it is held.
#: The words in that transcript were spoken before it, so the bound and
#: ``TurnReceipt.turn_ms`` start after the recognizer's pause, not at the first word.
MAX_DURATION_S = 10.0
#: The transcript limit a single final transcript already keeps (audio.is_question).
MAX_BYTES = 32000

#: Words a clause does not end on. Conservative on purpose: "on", "in", "up"
#: and "that" end ordinary sentences ("turn it on", "I like that"), so they
#: are not here, and a transcript that merely lacks punctuation is unsure.
CONJUNCTIONS = frozenset("and but or nor because if unless while although whereas until whether than".split())
PREPOSITIONS = frozenset("to for of at with from about into onto by as between without toward towards upon via".split())
ARTICLES = frozenset("the a an my your our their its every".split())
FILLERS = frozenset("um umm uh uhh uhm er erm hmm".split())
#: Wh-words that can be the object a trailing preposition lost ("who is it
#: for", "where are you from", "how much is it for"). "When", "why" and a
#: bare "how" cannot, so "how do I get from" is still waiting for its
#: object, as "could you send it to" is. A question word first is no
#: evidence on its own that a turn is finished: "what I really need is".
STRANDING = frozenset(["who", "whom", "whose", "what", "which", "where", "how much", "how many"])
#: Only a pronoun just before the preposition lets a wh-phrase excuse it. A
#: noun, verb or adjective there can go on past the preposition ("what's the
#: best way to go", "who should I talk to about it", "what would you
#: recommend for a cold"), and those cuts are the commonest in a spoken
#: question; a pronoun cannot. So "who is this for" is unsure, and a finished
#: "what are you waiting for" is held.
PRONOUNS = frozenset("it this that these those them him her me us you one".split())
#: Marks after which the speaker is plainly still going.
OPEN_ENDINGS = ("...", "…", ",", ";", ":", "-", "–", "—")
#: Closing marks that may follow a sentence's final punctuation.
CLOSERS = ")]\"'”’"
#: Letters and digits in any script: "set a timer for 20" ends on "20", not "for".
WORD = re.compile(r"[^\W_]+(?:'[^\W_]*)*")


@dataclass(frozen=True)
class Judgement:
    """What a detector made of the text so far, and the cue it went on."""

    verdict: str
    cue: str

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS or not isinstance(self.cue, str):
            raise ValueError(f"a judgement is one of {VERDICTS} with a cue")


class EndOfTurnDetector(Protocol):
    """Judges whether transcribed text is a finished turn. Fresh per session."""

    async def judge(self, text: str) -> Judgement: ...
    async def aclose(self) -> None: ...


def judge_text(text: str) -> Judgement:
    """The deterministic rules, in the order they decide.

    An open quote or bracket, a trailing filler, or a trailing mark such as
    a comma is incomplete whatever else is true. Then terminal punctuation
    is complete. A trailing conjunction or article is incomplete; so is a
    trailing preposition, unless a pronoun comes just before it and the text
    opens with a wh-phrase that can be its object ("where did you send it
    to"), which is unsure. Anything
    else is unsure, and is left to the silence that already ended it."""
    stripped = text.strip()
    if stripped.count("(") > stripped.count(")"):
        return Judgement(INCOMPLETE, "open parenthesis")
    if stripped.count('"') % 2 or stripped.count("“") > stripped.count("”"):
        return Judgement(INCOMPLETE, "open quote")
    words = [word.lower() for word in WORD.findall(stripped)]
    if words and words[-1] in FILLERS:
        return Judgement(INCOMPLETE, f"filler '{words[-1]}'")
    for ending in OPEN_ENDINGS:
        if stripped.endswith(ending):
            return Judgement(INCOMPLETE, f"trailing '{ending}'")
    tail = stripped.rstrip(CLOSERS)
    if tail.endswith("?"):
        return Judgement(COMPLETE, "question mark")
    if tail.endswith((".", "!")):
        return Judgement(COMPLETE, "terminal punctuation")
    if not words:
        return Judgement(UNSURE, "no words")
    last, first = words[-1], words[0].split("'")[0]  # "where's" opens like "where"
    if first == "how":
        first = " ".join(words[:2])  # "how much" can be an object; "how do" cannot
    if last in PREPOSITIONS and first in STRANDING and words[-2] in PRONOUNS:
        return Judgement(UNSURE, f"'{last}' may be stranded by '{first}'")
    if last in CONJUNCTIONS or last in ARTICLES or last in PREPOSITIONS:
        return Judgement(INCOMPLETE, f"trailing '{last}'")
    return Judgement(UNSURE, "no cue")


class LexicalEndOfTurn:
    """The rules above behind the detector seam; holds nothing open."""

    async def judge(self, text: str) -> Judgement:
        return judge_text(text)

    async def aclose(self) -> None:
        return None


#: What a chat model is asked; one word back.
CHAT_PROMPT = ("You decide whether a person talking to a voice assistant has finished what they were saying. "
               "You are given the transcript so far. Answer with exactly one word: complete if they have "
               "finished a sentence or question, incomplete if they stopped in the middle of one.")


class ChatEndOfTurn:
    """Any chat model as a detector. An answer that is neither word is unsure;
    a model error propagates, and the session treats it as no evidence."""

    def __init__(self, chat: ChatModel) -> None:
        self._chat = chat

    async def judge(self, text: str) -> Judgement:
        answer = WORD.findall((await self._chat.complete(CHAT_PROMPT, text)).lower())
        if answer[:1] == [COMPLETE]:
            return Judgement(COMPLETE, "model")
        if answer[:1] == [INCOMPLETE]:
            return Judgement(INCOMPLETE, "model")
        return Judgement(UNSURE, "unreadable model answer")

    async def aclose(self) -> None:
        return None


@dataclass(frozen=True)
class TurnReceipt:
    """Why a turn ended. ``held_ms`` is the wait after the last final
    transcript with words in it arrived (a wordless one re-arms the hold
    but is not part of the turn); ``turn_ms`` runs from its first final
    transcript to its release. Both start after the recognizer's pause, so
    neither counts the speech inside a transcript or the pause that ended it."""

    reason: str
    verdict: str
    cue: str
    fragments: int
    held_ms: float
    turn_ms: float

    def as_dict(self) -> dict[str, object]:
        return {"reason": self.reason, "verdict": self.verdict, "cue": self.cue, "fragments": self.fragments,
                "held_ms": self.held_ms, "turn_ms": self.turn_ms}


@dataclass(frozen=True)
class TurnEnd:
    """A released turn: what was said, by whom, and its receipt."""

    text: str
    speaker: str
    receipt: TurnReceipt


def _seconds(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number of seconds")
    return float(value)


class TurnHold:
    """The turn being held, with the time passed in rather than read.

    ``heard`` takes a final transcript and its judgement and returns the
    turns it releases (none, one, or two when another speaker or the byte
    bound releases the held turn first). ``speech_started`` keeps a held
    turn open until the turn's own bound; ``speech_stopped`` (speech that
    ended without words) runs the hold again from then. ``expire`` releases
    a turn whose deadline has come; ``drain`` releases whatever is held when
    input ends, or under ``reason`` when the session stops."""

    def __init__(self, *, hold: float = HOLD_S, max_duration: float = MAX_DURATION_S,
                 max_bytes: int = MAX_BYTES) -> None:
        self.hold = _seconds("hold", hold)
        self.max_duration = _seconds("max_duration", max_duration)
        if self.hold > self.max_duration:
            raise ValueError("hold cannot be longer than max_duration")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        self.max_bytes = max_bytes
        self._fragments: list[str] = []
        self._speaker = ""
        self._judgement = Judgement(UNSURE, "")
        self._first = self._last = 0.0
        #: When the held turn is released if nothing else happens; None when nothing is held.
        self.deadline: float | None = None
        self._capped = False

    @property
    def pending(self) -> bool:
        return bool(self._fragments)

    def _joins(self, text: str, speaker: str) -> bool:
        return (speaker == self._speaker
                and len(" ".join([*self._fragments, text]).encode("utf-8")) <= self.max_bytes)

    def text_with(self, text: str, speaker: str) -> str:
        """The text a detector should judge if this transcript arrives now."""
        return " ".join([*self._fragments, text]) if self._joins(text, speaker) else text

    def heard(self, text: str, speaker: str, judgement: Judgement, now: float) -> list[TurnEnd]:
        released: list[TurnEnd] = []
        if self.pending and not self._joins(text, speaker):
            released.append(self._release("speaker_changed" if speaker != self._speaker else "max_bytes", now))
        if not self.pending:
            self._first, self._speaker = now, speaker
        self._fragments.append(text)
        self._last, self._judgement = now, judgement
        if judgement.verdict == COMPLETE:
            released.append(self._release("semantic_complete", now))
        elif judgement.verdict == UNSURE:
            released.append(self._release("silence", now))
        elif now - self._first >= self.max_duration:
            released.append(self._release("max_duration", now))
        else:
            self._arm(now)
        return released

    def keyed(self, text: str, now: float, reason: str = "keypad") -> list[TurnEnd]:
        """Keys from the phone: they join the held turn, whoever it was
        spoken by, and finish it. With nothing held they are a turn of their
        own, by ``user``. Keys that would outgrow the held turn release it
        first, as a transcript would."""
        released: list[TurnEnd] = []
        speaker = self._speaker if self.pending else "user"
        if self.pending and not self._joins(text, speaker):
            released.append(self._release("max_bytes", now))
        if not self.pending:
            self._first, self._speaker = now, "user"
        self._fragments.append(text)
        self._last, self._judgement = now, Judgement(COMPLETE, "keypad")
        released.append(self._release(reason, now))
        return released

    def _arm(self, now: float) -> None:
        bound = self._first + self.max_duration
        self._capped = now + self.hold >= bound
        self.deadline = bound if self._capped else now + self.hold

    def speech_started(self, now: float) -> None:
        if self.pending:
            self.deadline, self._capped = self._first + self.max_duration, True

    def speech_stopped(self, now: float) -> None:
        # The clause is as open as it was; the words it waits for did not come.
        if self.pending:
            self._arm(now)

    def expire(self, now: float) -> list[TurnEnd]:
        if self.deadline is None or now < self.deadline:
            return []
        return [self._release("max_duration" if self._capped else "semantic_incomplete_timeout", now)]

    def drain(self, now: float, reason: str = "input_ended") -> list[TurnEnd]:
        return [self._release(reason, now)] if self.pending else []

    def _release(self, reason: str, now: float) -> TurnEnd:
        receipt = TurnReceipt(reason, self._judgement.verdict, self._judgement.cue, len(self._fragments),
                              round((now - self._last) * 1000, 3), round((now - self._first) * 1000, 3))
        end = TurnEnd(" ".join(self._fragments), self._speaker, receipt)
        self._fragments, self.deadline = [], None
        return end
