"""A user who has gone quiet: when to ask whether they are still there, and when to stop.

A caller can put the phone down on the table, walk away from the browser,
or wait for the assistant to say something first. Without a rule the
session listens to nothing until its own deadline, half an hour by
default. An ``IdlePolicy`` says how long the conversation waits for the
user before it counts an idle (``timeout``), what it says then
(``prompt``, or nothing), and after how many idles in a row it ends
(``end_after``, or never).

The wait is for the user, so only time the conversation spends waiting
on them counts. It starts with the conversation. The user speaking stops
it (they are not idle, however long they talk), and the end of their
speech, their words or a key start it again from then, with the count of
idles cleared. The bot speaking pauses it, and so does anything else the
session runs for the user, such as a tool; pauses are counted, and the
wait starts again from when the last one ends. An idle's prompt is the
bot speaking, so the next idle is a whole timeout after the prompt was
spoken. The idle that reaches ``end_after`` ends the conversation and
nothing follows it.

Every idle has a receipt (``IdleReceipt``): which idle in a row it was,
what was done (one of ``ACTIONS``), and how long the user had been silent.
``IdleWatch`` holds no clock; the time is passed in, so a test drives it
with plain numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

#: What an idle does: says the prompt, is only noted (the policy has no
#: prompt), or ends the conversation (it is the ``end_after``-th in a row).
ACTIONS = ("prompt", "noted", "end")
PROMPT = "Are you still there?"
END_AFTER = 3


@dataclass(frozen=True)
class IdlePolicy:
    """How long to wait for a quiet user, what to say, and when to give up."""

    timeout: float
    prompt: Optional[str] = PROMPT
    end_after: Optional[int] = END_AFTER

    def __post_init__(self) -> None:
        value = self.timeout
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("idle timeout must be a finite positive number of seconds")
        if self.prompt is not None and (not isinstance(self.prompt, str) or not self.prompt.strip()):
            raise ValueError("an idle prompt is None or text to say")
        if self.end_after is not None and (type(self.end_after) is not int or self.end_after < 1):
            raise ValueError("idle end_after is None or a positive integer")
        if self.prompt is None and self.end_after is None:
            raise ValueError("an idle policy with no prompt and no end does nothing")


@dataclass(frozen=True)
class IdleReceipt:
    """One idle: its place in a run of idles with nothing said between
    them, what was done, and the milliseconds the user had been silent
    while the conversation waited on them."""

    count: int
    action: str
    silent_ms: float

    def metadata(self) -> dict[str, str]:
        return {"idle_count": str(self.count), "idle_action": self.action,
                "idle_silent_ms": str(round(self.silent_ms))}


class IdleWatch:
    """The wait for the user, with the time passed in rather than read."""

    def __init__(self, policy: IdlePolicy, now: float) -> None:
        self.policy = policy
        #: Idles in a row with nothing from the user between them.
        self.count = 0
        #: Whether an idle ended the conversation.
        self.ended = False
        self._pauses = 0
        self._speaking = False
        self._since = 0.0
        #: When the next idle is, or None while the wait is paused or over.
        self.deadline: Optional[float] = None
        self._arm(now)

    def _arm(self, now: float) -> None:
        waiting = not (self.ended or self._pauses or self._speaking)
        if waiting:
            self._since = now
        self.deadline = now + self.policy.timeout if waiting else None

    def speaking(self, now: float) -> None:
        """The user began speaking, or is still speaking."""
        self.count, self._speaking, self.deadline = 0, True, None

    def spoke(self, now: float) -> None:
        """The user's speech ended, their words arrived, or they pressed a key."""
        self.count, self._speaking = 0, False
        self._arm(now)

    def restart(self, now: float) -> None:
        """The user is in the middle of a turn (a clause held open, keys
        being entered): not idle, and the wait starts again from now."""
        self._arm(now)

    def pause(self) -> None:
        """The bot began speaking, or a tool began running."""
        self._pauses, self.deadline = self._pauses + 1, None

    def resume(self, now: float) -> None:
        """One pause ended; the wait starts again when none is left."""
        if not self._pauses:
            raise RuntimeError("an idle watch resumed without a pause")
        self._pauses -= 1
        self._arm(now)

    def expire(self, now: float) -> Optional[IdleReceipt]:
        if self.deadline is None or now < self.deadline:
            return None
        self.count += 1
        end = self.policy.end_after is not None and self.count >= self.policy.end_after
        action = "end" if end else "prompt" if self.policy.prompt is not None else "noted"
        receipt = IdleReceipt(self.count, action, round((now - self._since) * 1000, 3))
        self.ended = end
        self._arm(now)
        return receipt
