"""Where the reply an agent is writing goes while it is being written.

A sink receives the model's content as deltas, in order, as the turn
produces them; it never receives tool-call arguments, tool results, or
anything a provider marks as reasoning. Text streamed in a turn that then
called tools was not the answer, and the sink is told to withdraw it. A
model that cannot stream, a structured answer, or a step replayed from
the journal delivers the answer as one delta once it is known, so a reader
is never told less than the loop accepted. The loop does not close the
sink: whoever owns the reader's window does, once the receipt is written.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class PublicText(Protocol):
    async def append(self, text: str) -> None:
        """One content delta of the reply being written."""
        ...

    def withdraw(self) -> None:
        """What was delivered so far in this turn was not the answer."""
        ...
